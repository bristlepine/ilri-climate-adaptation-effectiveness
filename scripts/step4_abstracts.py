#!/usr/bin/env python3
"""
step4_abstracts.py

Step 4 (Abstracts):
  - Reads outputs/step3/step3_benchmark_list.csv
  - Fetches abstracts for ALL DOIs.
  - Logic: Elsevier -> Semantic Scholar -> OpenAlex -> Crossref -> Unpaywall -> Landing Page (JSON-LD/Meta).
  - Deep scraping fallback (landing page + JSON-LD/meta).
  - Caching: reuses successful fetches AND caches failures to avoid re-downloading on reruns (TTL-controlled).

Outputs:
  - outputs/step4/step4_abstracts.csv
  - outputs/step4/step4_abstracts_cache.json
  - outputs/step4/step4_abstracts_summary.json
  - outputs/step4/step4_abstracts_missing.csv
"""

from __future__ import annotations

import json
import os
import re
import time
import csv
from datetime import datetime, timezone, timedelta
from html.parser import HTMLParser
from typing import Optional, Tuple
from urllib.parse import quote

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

import config as cfg
from utils import (
    load_json,
    save_json,
    normalize_doi,
    strip_tags,
    find_first_str,
    request_with_retries,
    utc_now_iso,
    ScopusAuth,
    scopus_headers,
)

DEFAULT_ELSEVIER_ARTICLE_URL = "https://api.elsevier.com/content/article/doi/"
DEFAULT_SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/"
DEFAULT_OPENALEX_WORKS_URL = "https://api.openalex.org/works/"
DEFAULT_CROSSREF_WORKS_URL = "https://api.crossref.org/works/"
DEFAULT_UNPAYWALL_URL = "https://api.unpaywall.org/v2/"

DEFAULT_STEP4_SLEEP_S = 0.05
DEFAULT_CACHE_TTL_DAYS = 30
DEFAULT_FORCE_REFRESH = False

# --- IMPORTANT: fixed output schema to prevent malformed CSV ---
OUT_COLUMNS = [
    # join keys / benchmark-provenance
    "doi",
    "doi_clean",
    "record_key",
    "citation_raw",
    "title_benchmark",
    "type",
    "identified_via",
    # fetched content
    "title_fetched",
    "abstract",
    "landing_url",
    # fetch bookkeeping
    "fetch_status",
    "source",
    "status_code",
    "error",
    "fetched_utc",
    "last_attempt_utc",
    "fail_count",
]

def _now_dt_utc() -> datetime:
    return datetime.now(timezone.utc)

def _parse_iso_utc(s: str) -> Optional[datetime]:
    if not isinstance(s, str) or not s.strip():
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except Exception:
        return None

def _doi_key(x: str) -> str:
    d = normalize_doi(x)
    d = (d or "").strip().rstrip(" .),;]}>")
    return d

def _clean_cell(s: Optional[str]) -> str:
    """
    Make text CSV-safe and consistent:
    - stringify
    - collapse newlines/odd whitespace
    """
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _http_get_json(
    session: requests.Session,
    url: str,
    *,
    headers: Optional[dict] = None,
    params: Optional[dict] = None,
    tries: int = 6,
) -> Tuple[Optional[dict], Optional[str], Optional[int]]:
    try:
        data, _rate = request_with_retries(
            session=session,
            method="GET",
            url=url,
            headers=headers or {},
            params=params or {},
            data=None,
            tries=tries,
        )
        return data, None, 200
    except Exception as ex:
        msg = str(ex)[:300]
        m = re.search(r"HTTP\s+(\d{3})", msg)
        code = int(m.group(1)) if m else None
        return None, msg, code

def _build_openalex_abstract(inv: dict) -> Optional[str]:
    if not isinstance(inv, dict) or not inv:
        return None

    max_pos = -1
    for _, pos_list in inv.items():
        if isinstance(pos_list, list) and pos_list:
            try:
                max_pos = max(max_pos, max(int(p) for p in pos_list))
            except Exception:
                continue

    if max_pos < 0 or max_pos > 200000:
        return None

    words = [""] * (max_pos + 1)
    for word, pos_list in inv.items():
        if not isinstance(word, str) or not isinstance(pos_list, list):
            continue
        for p in pos_list:
            try:
                pi = int(p)
            except Exception:
                continue
            if 0 <= pi <= max_pos and not words[pi]:
                words[pi] = word

    out = " ".join(w for w in words if w)
    out = re.sub(r"\s+", " ", out).strip()
    return out or None

def elsevier_article_fetch(session: requests.Session, auth: ScopusAuth, base_url: str, doi: str) -> dict:
    url = base_url.rstrip("/") + "/" + quote(doi, safe="")
    h = scopus_headers(auth)
    params = {"view": "META_ABS", "httpAccept": "application/json"}
    data, err, code = _http_get_json(session, url, headers=h, params=params)

    if err or not data:
        return {"error": err or "no_data", "status_code": code, "source": "elsevier_api"}

    title = find_first_str(data, ("dc:title", "title"))
    abst = find_first_str(data, ("dc:description", "ce:abstract", "abstract", "description"))
    abst = strip_tags(abst) if abst else None
    return {
        "title_fetched": _clean_cell(title),
        "abstract": _clean_cell(abst) if abst else "",
        "source": "elsevier_api",
        "status_code": 200,
    }

def semantic_scholar_fetch(session: requests.Session, doi: str, email: str | None, base_url: str) -> dict:
    url = base_url.rstrip("/") + "/DOI:" + quote(doi, safe=":/")
    h = {"Accept": "application/json", "User-Agent": f"step4 ({email or 'no-email'})"}
    data, err, code = _http_get_json(session, url, headers=h, params={"fields": "title,abstract"})

    if err or not data:
        return {"error": err or "no_data", "status_code": code, "source": "semantic_scholar"}

    return {
        "title_fetched": _clean_cell(data.get("title")),
        "abstract": _clean_cell((data.get("abstract") or "").strip()) if data.get("abstract") else "",
        "source": "semantic_scholar",
        "status_code": 200,
    }

def openalex_fetch(session: requests.Session, doi: str, email: str | None, base_url: str) -> dict:
    url = base_url.rstrip("/") + "/https://doi.org/" + quote(doi, safe=":/")
    h = {"Accept": "application/json", "User-Agent": f"step4 ({email or 'no-email'})"}
    params = {"mailto": email} if email else {}
    data, err, code = _http_get_json(session, url, headers=h, params=params)

    if err or not data:
        return {"error": err or "no_data", "status_code": code, "source": "openalex"}

    inv = (data or {}).get("abstract_inverted_index")
    abst = _build_openalex_abstract(inv) if inv else None

    loc = (data or {}).get("primary_location") or {}
    landing = loc.get("landing_page_url")

    return {
        "title_fetched": _clean_cell(data.get("title")),
        "abstract": _clean_cell(abst) if abst else "",
        "landing_url": _clean_cell(landing) if landing else "",
        "source": "openalex",
        "status_code": 200,
    }

def crossref_fetch(session: requests.Session, doi: str, email: str | None, base_url: str) -> dict:
    url = base_url.rstrip("/") + "/" + quote(doi)
    h = {"Accept": "application/json", "User-Agent": f"step4 ({email or 'no-email'})"}
    params = {"mailto": email} if email else {}
    data, err, code = _http_get_json(session, url, headers=h, params=params)

    if err or not data:
        return {"error": err or "no_data", "status_code": code, "source": "crossref"}

    msg = (data or {}).get("message") or {}
    title = (msg.get("title") or [None])[0]
    abst = msg.get("abstract")
    abst = strip_tags(abst) if abst else None
    return {
        "title_fetched": _clean_cell(title),
        "abstract": _clean_cell(abst) if abst else "",
        "source": "crossref",
        "status_code": 200,
    }

def unpaywall_fetch(session: requests.Session, doi: str, email: str | None, base_url: str) -> dict:
    if not email:
        return {"error": "no_email", "status_code": None, "source": "unpaywall"}

    url = base_url.rstrip("/") + "/" + quote(doi)
    h = {"Accept": "application/json", "User-Agent": f"step4 ({email})"}
    data, err, code = _http_get_json(session, url, headers=h, params={"email": email})

    if err or not data:
        return {"error": err or "no_data", "status_code": code, "source": "unpaywall"}

    best = (data or {}).get("best_oa_location") or {}
    return {"landing_url": _clean_cell(best.get("url")), "source": "unpaywall", "status_code": 200}

class MetaParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.metas = {}

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "meta":
            return
        d = {k.lower(): v for k, v in attrs}
        name = d.get("name") or d.get("property")
        content = d.get("content")
        if name and content:
            self.metas[name.lower()] = content

def fetch_landing_abstract(session: requests.Session, url: str) -> dict:
    try:
        h = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        backoff = 1.0
        last_status = None
        for _ in range(3):
            r = session.get(url, headers=h, timeout=20, allow_redirects=True)
            last_status = r.status_code
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(backoff)
                backoff = min(backoff * 1.7, 10.0)
                continue
            break

        if last_status != 200:
            return {"error": f"HTTP {last_status}", "status_code": last_status, "source": "landing_page"}

        raw = r.text or ""
        raw_head = raw[:250000]

        for match in re.finditer(
            r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>',
            raw_head, re.DOTALL | re.IGNORECASE
        ):
            blob = match.group(1).strip()
            if not blob:
                continue
            try:
                j = json.loads(blob)
            except Exception:
                continue

            items = j if isinstance(j, list) else [j]
            for item in items:
                if not isinstance(item, dict):
                    continue
                for k in ("description", "abstract"):
                    v = item.get(k)
                    if isinstance(v, str) and len(v) > 50:
                        return {
                            "abstract": _clean_cell(strip_tags(v)),
                            "source": "landing_json_ld",
                            "status_code": 200
                        }

        p = MetaParser()
        p.feed(raw_head)
        for k in ("citation_abstract", "dc.description", "og:description", "description"):
            v = p.metas.get(k)
            if isinstance(v, str) and len(v) > 50:
                return {
                    "abstract": _clean_cell(strip_tags(v)),
                    "source": "landing_meta",
                    "status_code": 200
                }

        return {"error": "no_abstract_found", "status_code": 200, "source": "landing_page"}
    except Exception as e:
        return {"error": str(e)[:300], "status_code": None, "source": "landing_page"}

def _should_retry_cached_failure(cached: dict, *, force_refresh: bool, ttl_days: Optional[int]) -> bool:
    if force_refresh:
        return True
    if ttl_days is None:
        return False
    last = _parse_iso_utc(cached.get("last_attempt_utc") or "")
    if not last:
        return True
    return (_now_dt_utc() - last) >= timedelta(days=int(ttl_days))

def _write_chunk_csv(out_csv: str, rows: list[dict], wrote_header: bool) -> bool:
    if not rows:
        return wrote_header

    df_out = pd.DataFrame(rows)

    # Force stable schema (prevents ParserError later)
    for c in OUT_COLUMNS:
        if c not in df_out.columns:
            df_out[c] = ""
    df_out = df_out[OUT_COLUMNS].copy()

    # Clean cells (esp. newlines) consistently
    for c in df_out.columns:
        df_out[c] = df_out[c].map(_clean_cell)

    df_out.to_csv(
        out_csv,
        mode="a",
        header=not wrote_header,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        quoting=csv.QUOTE_MINIMAL,
    )
    return True

def step4_fetch_abstracts(config: dict) -> dict:
    load_dotenv()

    out_dir = config.get("out_dir") or getattr(cfg, "out_dir", "outputs")

    api_key = config.get("scopus_api_key") or os.getenv("SCOPUS_API_KEY")
    inst_token = config.get("scopus_inst_token") or os.getenv("SCOPUS_INST_TOKEN")
    email = config.get("contact_email") or os.getenv("CONTACT_EMAIL")

    step4_sleep_s = float(config.get("step4_sleep_s", getattr(cfg, "STEP4_SLEEP_S", DEFAULT_STEP4_SLEEP_S)))
    cache_ttl_days = config.get("step4_cache_ttl_days", getattr(cfg, "STEP4_CACHE_TTL_DAYS", DEFAULT_CACHE_TTL_DAYS))
    cache_ttl_days = None if cache_ttl_days in (None, "None", "none", "") else int(cache_ttl_days)
    force_refresh = bool(config.get("step4_force_refresh", getattr(cfg, "STEP4_FORCE_REFRESH", DEFAULT_FORCE_REFRESH)))

    in_csv = os.path.join(out_dir, "step3", "step3_benchmark_list.csv")
    step4_dir = os.path.join(out_dir, "step4")
    out_csv = os.path.join(step4_dir, "step4_abstracts.csv")
    cache_path = os.path.join(step4_dir, "step4_abstracts_cache.json")
    out_summary = os.path.join(step4_dir, "step4_abstracts_summary.json")
    missing_csv = os.path.join(step4_dir, "step4_abstracts_missing.csv")

    os.makedirs(step4_dir, exist_ok=True)

    if not os.path.exists(in_csv):
        raise SystemExit(f"[step4] Missing input: {in_csv}")

    # Read step3 robustly and as strings
    df = pd.read_csv(in_csv, engine="python", dtype=str, keep_default_na=False, na_filter=False)
    if "doi" not in df.columns:
        raise SystemExit("[step4] Input missing required column: doi")

    # Carry forward cleaned citation fields (these help downstream)
    if "citation_raw" not in df.columns:
        df["citation_raw"] = ""
    if "record_key" not in df.columns:
        df["record_key"] = ""
    if "title" not in df.columns:
        df["title"] = ""

    df["doi_clean"] = df["doi"].apply(_doi_key)
    df = df[df["doi_clean"].astype(str).str.len() > 0].drop_duplicates(subset=["doi_clean"]).copy()

    total = int(len(df))
    print(f"[step4] Processing {total} unique DOIs")

    cache: dict = load_json(cache_path, default={})

    # rewrite output every run (cache prevents refetch)
    if os.path.exists(out_csv):
        os.remove(out_csv)

    wrote_header = False
    chunk: list[dict] = []
    failed_records: list[dict] = []

    stats = {
        "total_rows": total,
        "cached_ok": 0,
        "cached_failed_skipped": 0,
        "fresh_ok": 0,
        "fresh_failed": 0,
        "ok_by_source": {},
    }

    auth = ScopusAuth(api_key=api_key or "", inst_token=inst_token or None)

    pbar = tqdm(total=total, desc="[step4] Fetching abstracts", unit="paper")

    with requests.Session() as session:
        for _, row in df.iterrows():
            doi = row["doi_clean"]

            base_rec = {
                "doi": _clean_cell(row.get("doi") or doi),
                "doi_clean": doi,
                "record_key": _clean_cell(row.get("record_key")),
                "citation_raw": _clean_cell(row.get("citation_raw")),
                "title_benchmark": _clean_cell(row.get("title")),
                "type": _clean_cell(row.get("type")),
                "identified_via": _clean_cell(row.get("identified_via")),

                "title_fetched": "",
                "abstract": "",
                "landing_url": "",

                "fetch_status": "no_abstract",
                "source": "",
                "status_code": "",
                "error": "",
                "fetched_utc": "",
                "last_attempt_utc": "",
                "fail_count": "",
            }

            cached = cache.get(doi)

            # Use cached OK unless force_refresh
            if isinstance(cached, dict) and cached.get("fetch_status") == "ok" and not force_refresh:
                out_rec = dict(base_rec)
                out_rec.update(cached)  # but we still force OUT_COLUMNS on write
                stats["cached_ok"] += 1
                src = out_rec.get("source") or "unknown"
                stats["ok_by_source"][src] = stats["ok_by_source"].get(src, 0) + 1

                chunk.append(out_rec)
                pbar.update(1)
                if len(chunk) >= 50:
                    wrote_header = _write_chunk_csv(out_csv, chunk, wrote_header)
                    chunk = []
                continue

            # Skip cached failures until TTL says retry
            if isinstance(cached, dict) and cached.get("fetch_status") != "ok" and not _should_retry_cached_failure(
                cached, force_refresh=force_refresh, ttl_days=cache_ttl_days
            ):
                out_rec = dict(base_rec)
                out_rec.update(cached)
                stats["cached_failed_skipped"] += 1

                failed_records.append({
                    "doi_clean": doi,
                    "title_benchmark": out_rec.get("title_benchmark"),
                    "last_source_attempted": out_rec.get("source"),
                    "fetch_status": out_rec.get("fetch_status"),
                    "error": out_rec.get("error"),
                })

                chunk.append(out_rec)
                pbar.update(1)
                if len(chunk) >= 50:
                    wrote_header = _write_chunk_csv(out_csv, chunk, wrote_header)
                    chunk = []
                continue

            rec = dict(base_rec)
            rec["last_attempt_utc"] = utc_now_iso()

            # Chain of responsibility
            if api_key:
                pbar.set_postfix_str("Elsevier")
                res = elsevier_article_fetch(session, auth, DEFAULT_ELSEVIER_ARTICLE_URL, doi)
                if res.get("abstract"):
                    rec.update(res)
                    rec["fetch_status"] = "ok"
                else:
                    rec["source"] = res.get("source", "elsevier_api")
                    rec["status_code"] = res.get("status_code") or ""
                    rec["error"] = res.get("error") or ""

            if rec["fetch_status"] != "ok":
                pbar.set_postfix_str("SemanticScholar")
                res = semantic_scholar_fetch(session, doi, email, DEFAULT_SEMANTIC_SCHOLAR_URL)
                if res.get("abstract"):
                    rec.update(res)
                    rec["fetch_status"] = "ok"
                else:
                    rec["source"] = rec["source"] or res.get("source") or ""
                    rec["status_code"] = rec["status_code"] or (res.get("status_code") or "")
                    rec["error"] = rec["error"] or (res.get("error") or "")

            if rec["fetch_status"] != "ok":
                pbar.set_postfix_str("OpenAlex")
                res = openalex_fetch(session, doi, email, DEFAULT_OPENALEX_WORKS_URL)
                if res.get("landing_url"):
                    rec["landing_url"] = res.get("landing_url") or rec["landing_url"]
                if res.get("abstract"):
                    rec.update(res)
                    rec["fetch_status"] = "ok"
                else:
                    rec["source"] = rec["source"] or res.get("source") or ""
                    rec["status_code"] = rec["status_code"] or (res.get("status_code") or "")
                    rec["error"] = rec["error"] or (res.get("error") or "")

            if rec["fetch_status"] != "ok":
                pbar.set_postfix_str("Crossref")
                res = crossref_fetch(session, doi, email, DEFAULT_CROSSREF_WORKS_URL)
                if res.get("abstract"):
                    rec.update(res)
                    rec["fetch_status"] = "ok"
                else:
                    rec["source"] = rec["source"] or res.get("source") or ""
                    rec["status_code"] = rec["status_code"] or (res.get("status_code") or "")
                    rec["error"] = rec["error"] or (res.get("error") or "")

            if rec["fetch_status"] != "ok":
                pbar.set_postfix_str("Landing")
                landing = rec.get("landing_url") or ""
                if not landing and email:
                    upw = unpaywall_fetch(session, doi, email, DEFAULT_UNPAYWALL_URL)
                    landing = upw.get("landing_url") or landing
                if not landing:
                    landing = f"https://doi.org/{doi}"

                res = fetch_landing_abstract(session, landing)
                if res.get("abstract"):
                    rec.update(res)
                    rec["landing_url"] = landing
                    rec["fetch_status"] = "ok"
                else:
                    rec["landing_url"] = landing
                    rec["source"] = rec["source"] or res.get("source") or ""
                    rec["status_code"] = rec["status_code"] or (res.get("status_code") or "")
                    rec["error"] = rec["error"] or (res.get("error") or "")

            # finalize + cache
            rec["last_attempt_utc"] = utc_now_iso()
            if rec["fetch_status"] == "ok":
                rec["fetched_utc"] = rec["last_attempt_utc"]
                stats["fresh_ok"] += 1
                src = rec.get("source") or "unknown"
                stats["ok_by_source"][src] = stats["ok_by_source"].get(src, 0) + 1
                rec["fail_count"] = str((cached or {}).get("fail_count") or 0)
            else:
                stats["fresh_failed"] += 1
                fail_count = int((cached or {}).get("fail_count") or 0) + 1
                rec["fail_count"] = str(fail_count)

                failed_records.append({
                    "doi_clean": doi,
                    "title_benchmark": rec.get("title_benchmark"),
                    "last_source_attempted": rec.get("source"),
                    "fetch_status": rec.get("fetch_status"),
                    "error": rec.get("error"),
                })

            # always keep cells clean
            rec["abstract"] = _clean_cell(rec.get("abstract"))
            rec["title_fetched"] = _clean_cell(rec.get("title_fetched"))

            cache[doi] = rec

            chunk.append(rec)
            pbar.update(1)

            if len(chunk) >= 50:
                wrote_header = _write_chunk_csv(out_csv, chunk, wrote_header)
                chunk = []
                save_json(cache_path, cache)

            if step4_sleep_s > 0:
                time.sleep(step4_sleep_s)

        pbar.close()

    if chunk:
        wrote_header = _write_chunk_csv(out_csv, chunk, wrote_header)

    save_json(cache_path, cache)

    if failed_records:
        pd.DataFrame(failed_records).to_csv(missing_csv, index=False)
    else:
        if os.path.exists(missing_csv):
            os.remove(missing_csv)

    summary = {
        "total_rows": stats["total_rows"],
        "cached_ok": stats["cached_ok"],
        "cached_failed_skipped": stats["cached_failed_skipped"],
        "fresh_ok": stats["fresh_ok"],
        "fresh_failed": stats["fresh_failed"],
        "ok_total": int(stats["cached_ok"] + stats["fresh_ok"]),
        "failed_total": int(stats["cached_failed_skipped"] + stats["fresh_failed"]),
        "ok_by_source": stats["ok_by_source"],
        "cache_ttl_days": cache_ttl_days,
        "force_refresh": bool(force_refresh),
        "timestamp_utc": utc_now_iso(),
        "out_columns": OUT_COLUMNS,
    }
    save_json(out_summary, summary)

    print(
        f"[step4] Done. ok_total={summary['ok_total']} failed_total={summary['failed_total']} "
        f"(cached_ok={summary['cached_ok']} fresh_ok={summary['fresh_ok']})"
    )

    return {"status": "ok", "path": out_csv}

def run(config: dict) -> dict:
    return step4_fetch_abstracts(config)

def run_step4(config: dict) -> dict:
    return step4_fetch_abstracts(config)

def main(config: dict | None = None) -> dict:
    return step4_fetch_abstracts(config or {})

if __name__ == "__main__":
    step4_fetch_abstracts({})
