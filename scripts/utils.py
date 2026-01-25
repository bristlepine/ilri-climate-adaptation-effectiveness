"""
utils.py

Shared utilities used across multiple pipeline steps:
- auth + headers
- robust HTTP requests w/ retries + rate-limit backoff
- JSON/YAML IO helpers
- DOI parsing/normalization + small text helpers
- NEW: dataframe helpers for stable keys + first-column carry-through
"""

import os
import re
import json
import time
import hashlib
import html
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import random
import requests
import yaml
import pandas as pd


# -----------------------------
# DOI helpers
# -----------------------------
DOI_REGEX = re.compile(r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", re.IGNORECASE)

def normalize_doi(x: str) -> str:
    """
    Normalize DOI to a core form (no https://doi.org/ prefix), lowercase.
    Also strips trailing periods that often break APIs.
    """
    if x is None:
        return ""
    try:
        # handle pandas NaN
        if isinstance(x, float) and pd.isna(x):
            return ""
    except Exception:
        pass

    d = str(x).strip().lower()
    for p in ("https://doi.org/", "http://doi.org/", "doi:"):
        if d.startswith(p):
            d = d[len(p):]
    # common junk at end in CSVs / refs
    d = d.strip().rstrip(".")
    return d

def doi_from_text(s: str) -> Optional[str]:
    if not isinstance(s, str):
        return None
    m = DOI_REGEX.search(s)
    return normalize_doi(m.group(1)) if m else None

def doi_to_url(doi: str) -> str:
    """Turn a DOI core into a https://doi.org/... URL (or '' if missing)."""
    core = normalize_doi(doi)
    return f"https://doi.org/{core}" if core else ""


# -----------------------------
# Text helpers
# -----------------------------
def strip_tags(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s or "")
    s = html.unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def find_first_str(o, wanted_keys: Tuple[str, ...]) -> Optional[str]:
    """Recursively find the first non-empty string value for any of wanted_keys."""
    if isinstance(o, dict):
        for k, v in o.items():
            if k in wanted_keys and isinstance(v, str) and v.strip():
                return v.strip()
        for v in o.values():
            got = find_first_str(v, wanted_keys)
            if got:
                return got
    elif isinstance(o, list):
        for it in o:
            got = find_first_str(it, wanted_keys)
            if got:
                return got
    return None

def clean_text(s: str) -> str:
    """
    Stable alphanumeric key for matching titles.
    Example: "A Study: 2020!" -> "astudy2020"
    """
    if not isinstance(s, str):
        return ""
    return re.sub(r"[^a-z0-9]", "", s.lower())


# -----------------------------
# DataFrame helpers (NEW)
# -----------------------------
def ensure_first_col(df: pd.DataFrame, col: str, default: str = "") -> pd.DataFrame:
    """
    Ensure `col` exists and is the first column. Returns a re-ordered copy view.
    """
    if col not in df.columns:
        df[col] = default
    cols = [col] + [c for c in df.columns if c != col]
    return df[cols]

def record_key(doi: str, title: str) -> str:
    """
    Stable join key used across steps:
      - DOI core if available
      - else cleaned title key
    """
    d = normalize_doi(doi)
    if d:
        return d
    return clean_text(title or "")


# -----------------------------
# Auth + headers
# -----------------------------
@dataclass
class ScopusAuth:
    api_key: str
    inst_token: Optional[str] = None

def scopus_headers(auth: ScopusAuth) -> Dict[str, str]:
    h = {"X-ELS-APIKey": auth.api_key, "Accept": "application/json"}
    if auth.inst_token:
        h["X-ELS-Insttoken"] = auth.inst_token
    return h


# -----------------------------
# Simple IO helpers
# -----------------------------
def load_json(path: str, default):
    try:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default

def save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

def read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# -----------------------------
# Hash helpers (caching)
# -----------------------------
def query_hash(q: str) -> str:
    return hashlib.sha256(q.strip().encode("utf-8")).hexdigest()

def queries_signature(queries: List[Tuple[str, str]]) -> str:
    payload = [{"name": n, "query": q} for n, q in queries]
    s = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(s).hexdigest()


# -----------------------------
# HTTP helpers
# -----------------------------
def _rate_headers(r: requests.Response) -> dict:
    keys = ["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset", "Retry-After"]
    return {k: r.headers.get(k) for k in keys if r.headers.get(k) is not None}

def request_with_retries(
    session: requests.Session,
    method: str,
    url: str,
    headers: dict,
    params: Optional[dict] = None,
    data: Optional[dict] = None,
    tries: int = 12,
    timeout: Tuple[float, float] = (30.0, 180.0),  # (connect_timeout, read_timeout)
) -> Tuple[dict, dict]:
    """
    Returns: (json_data, last_rate_headers)

    Retries on:
      - 429 (rate limited) with Retry-After if present
      - 500/502/503/504 (transient server errors)
      - network exceptions: ReadTimeout, Timeout, ConnectionError, ChunkedEncodingError

    Best-practice defaults for Scopus bulk retrieval:
      - timeout=(30, 180): avoids hanging forever, but tolerates slow responses
      - tries=12: gives enough runway through flaky periods
      - exponential backoff w/ jitter, capped at 60s
    """
    backoff = 1.0
    last_rate: dict = {}

    def _sleep_with_jitter(base: float) -> None:
        # add small jitter so many retries don't sync up
        time.sleep(base + random.uniform(0.0, min(0.5, base * 0.1)))

    method_u = (method or "GET").upper()

    for attempt in range(tries):
        try:
            if method_u == "POST":
                r = session.post(
                    url,
                    headers=headers,
                    data=data if data is not None else params,
                    timeout=timeout,
                )
            else:
                r = session.get(
                    url,
                    headers=headers,
                    params=params,
                    timeout=timeout,
                )

            last_rate = _rate_headers(r)

            if r.status_code == 200:
                # Occasionally APIs return HTML on errors with 200; safeguard json parse
                try:
                    return r.json(), last_rate
                except Exception as e:
                    raise RuntimeError(f"HTTP 200 but JSON parse failed: {str(e)[:200]} :: {r.text[:400]}")

            # Rate limit
            if r.status_code == 429:
                ra = r.headers.get("Retry-After")
                if ra:
                    try:
                        wait_s = max(1.0, float(ra))
                    except Exception:
                        wait_s = backoff
                else:
                    wait_s = backoff

                _sleep_with_jitter(wait_s)
                backoff = min(backoff * 2, 60.0)
                continue

            # Transient server errors
            if r.status_code in (500, 502, 503, 504):
                _sleep_with_jitter(backoff)
                backoff = min(backoff * 2, 60.0)
                continue

            # Other non-retryable HTTP errors
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:800]}")

        except (
            requests.exceptions.ReadTimeout,
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
        ):
            # transient network failure
            if attempt == tries - 1:
                raise
            _sleep_with_jitter(backoff)
            backoff = min(backoff * 2, 60.0)
            continue

    raise RuntimeError(f"Failed after retries. Last rate headers: {last_rate}")
