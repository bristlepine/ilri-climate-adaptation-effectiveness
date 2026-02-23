#!/usr/bin/env python3
"""
step9_enrich_abstracts.py

Step 9: Enrich Step 8 Scopus-cleaned records with abstracts,
using the same chain as step4, but NOT limited to DOIs.

Strategy (deterministic, cache-first):
  1) If DOI present:
       Elsevier (doi) -> Semantic Scholar -> OpenAlex -> Crossref -> Unpaywall -> Landing scrape
  2) Else if Scopus ID present:
       Elsevier (scopus_id) -> Landing scrape (prism_url)
  3) Else if prism_url present:
       Landing scrape (prism_url)
  4) Otherwise:
       excluded (no identifiers)

Inputs:
  - outputs/step8/step8_scopus_cleaned.csv   (under scripts/outputs by default)

Outputs (ALL under outputs/step9/):
  - step9_scopus_enriched.csv
  - step9_scopus_enriched.ris
  - step9_abstract_cache.json
  - step9_summary.json
  - step9_missing.csv

Env vars (same as step4):
  - SCOPUS_API_KEY
  - SCOPUS_INST_TOKEN (optional)
  - CONTACT_EMAIL (recommended; required for Unpaywall)

Determinism:
  - No CLI args.
  - Cache-first, stable ordering (input order).
  - Failures are cached; retried only after TTL unless FORCE_REFRESH=True.

Run:
  python step9_enrich_abstracts.py
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from html.parser import HTMLParser
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

import pandas as pd
import requests

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# ----------------------------
# Deterministic settings (edit here)
# ----------------------------

# outputs folder lives next to this script: scripts/outputs
OUT_DIR = str((Path(__file__).resolve().parent / "outputs").resolve())

SLEEP_S = 0.05
CACHE_TTL_DAYS: Optional[int] = 30   # set to None to never retry cached failures
FORCE_REFRESH = False                # set True ONLY if you want to refetch despite cache
MAX_RECORDS: Optional[int] = None    # set to an int to cap work (debugging), else None


# ----------------------------
# Constants / endpoints
# ----------------------------

DEFAULT_ELSEVIER_ARTICLE_URL = "https://api.elsevier.com/content/article/doi/"
DEFAULT_ELSEVIER_ABSTRACT_SCOPUSID_URL = "https://api.elsevier.com/content/abstract/scopus_id/"
DEFAULT_SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/"
DEFAULT_OPENALEX_WORKS_URL = "https://api.openalex.org/works/"
DEFAULT_CROSSREF_WORKS_URL = "https://api.crossref.org/works/"
DEFAULT_UNPAYWALL_URL = "https://api.unpaywall.org/v2/"


# ----------------------------
# Small utilities
# ----------------------------

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

def utc_now_iso() -> str:
    return _now_dt_utc().isoformat().replace("+00:00", "Z")

def safe_str(x) -> str:
    if x is None:
        return ""
    s = str(x)
    s = s.replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_doi(doi: str) -> str:
    s = safe_str(doi)
    if not s:
        return ""
    s = s.strip()
    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s, flags=re.IGNORECASE).strip()
    s = s.rstrip(" .),;]}>")
    return s.lower()

def load_json(path: str, default: Any) -> Any:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default

def save_json(path: str, data: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def _should_retry_cached_failure(cached: dict, *, force_refresh: bool, ttl_days: Optional[int]) -> bool:
    if force_refresh:
        return True
    if ttl_days is None:
        return False
    last = _parse_iso_utc(cached.get("last_attempt_utc") or "")
    if not last:
        return True
    return (_now_dt_utc() - last) >= timedelta(days=int(ttl_days))

def strip_tags(s: str) -> str:
    s = safe_str(s)
    if not s:
        return ""
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def extract_scopus_id_from_prism_url(url: str) -> str:
    u = safe_str(url)
    if not u:
        return ""
    m = re.search(r"/scopus_id/(\d+)", u)
    return m.group(1) if m else ""


# ----------------------------
# Elsevier auth/header helpers
# ----------------------------

@dataclass(frozen=True)
class ScopusAuth:
    api_key: str
    inst_token: Optional[str] = None

def scopus_headers(auth: ScopusAuth) -> dict:
    h = {"Accept": "application/json"}
    if auth.api_key:
        h["X-ELS-APIKey"] = auth.api_key
    if auth.inst_token:
        h["X-ELS-Insttoken"] = auth.inst_token
    return h


# ----------------------------
# HTTP helpers
# ----------------------------

def _http_get_json(
    session: requests.Session,
    url: str,
    *,
    headers: Optional[dict] = None,
    params: Optional[dict] = None,
    tries: int = 6,
    timeout: float = 20.0,
) -> Tuple[Optional[dict], Optional[str], Optional[int]]:
    last_err = None
    last_code = None
    backoff = 1.0
    for _ in range(max(1, int(tries))):
        try:
            r = session.get(url, headers=headers or {}, params=params or {}, timeout=timeout)
            last_code = r.status_code
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(backoff)
                backoff = min(backoff * 1.7, 12.0)
                continue
            if r.status_code != 200:
                return None, f"HTTP {r.status_code}", r.status_code
            return r.json(), None, 200
        except Exception as ex:
            last_err = str(ex)[:300]
            time.sleep(backoff)
            backoff = min(backoff * 1.7, 12.0)
    return None, last_err or "request_failed", last_code


# ----------------------------
# Source fetchers
# ----------------------------

def _find_first(obj: Any, keys: Tuple[str, ...]) -> Optional[str]:
    if not isinstance(obj, (dict, list)):
        return None
    if isinstance(obj, dict):
        for k in keys:
            if k in obj and isinstance(obj[k], (str, int, float)):
                return str(obj[k])
        for v in obj.values():
            r = _find_first(v, keys)
            if r:
                return r
    else:
        for it in obj:
            r = _find_first(it, keys)
            if r:
                return r
    return None

def elsevier_article_fetch(session: requests.Session, auth: ScopusAuth, doi: str) -> dict:
    if not auth.api_key:
        return {"error": "no_api_key", "status_code": None, "source": "elsevier_doi"}

    url = DEFAULT_ELSEVIER_ARTICLE_URL.rstrip("/") + "/" + requests.utils.quote(doi, safe="")
    h = scopus_headers(auth)
    params = {"view": "META_ABS", "httpAccept": "application/json"}
    data, err, code = _http_get_json(session, url, headers=h, params=params)

    if err or not data:
        return {"error": err or "no_data", "status_code": code, "source": "elsevier_doi"}

    title = _find_first(data, ("dc:title", "title"))
    abstract = _find_first(data, ("dc:description", "ce:abstract", "abstract", "description"))

    return {
        "title_fetched": safe_str(title),
        "abstract": safe_str(strip_tags(abstract)) if abstract else "",
        "source": "elsevier_doi",
        "status_code": 200,
    }

def elsevier_scopusid_fetch(session: requests.Session, auth: ScopusAuth, scopus_id: str) -> dict:
    if not auth.api_key:
        return {"error": "no_api_key", "status_code": None, "source": "elsevier_scopus_id"}

    url = DEFAULT_ELSEVIER_ABSTRACT_SCOPUSID_URL.rstrip("/") + "/" + requests.utils.quote(scopus_id, safe="")
    h = scopus_headers(auth)
    # FULL tends to include abstracts when present
    params = {"view": "FULL", "httpAccept": "application/json"}
    data, err, code = _http_get_json(session, url, headers=h, params=params)

    if err or not data:
        return {"error": err or "no_data", "status_code": code, "source": "elsevier_scopus_id"}

    title = _find_first(data, ("dc:title", "title"))
    abstract = _find_first(data, ("dc:description", "ce:abstract", "abstract", "description"))

    return {
        "title_fetched": safe_str(title),
        "abstract": safe_str(strip_tags(abstract)) if abstract else "",
        "source": "elsevier_scopus_id",
        "status_code": 200,
    }

def semantic_scholar_fetch(session: requests.Session, doi: str, email: str | None) -> dict:
    url = DEFAULT_SEMANTIC_SCHOLAR_URL.rstrip("/") + "/DOI:" + requests.utils.quote(doi, safe=":/")
    h = {"Accept": "application/json", "User-Agent": f"step9 ({email or 'no-email'})"}
    data, err, code = _http_get_json(session, url, headers=h, params={"fields": "title,abstract"}, tries=4)

    if err or not data:
        return {"error": err or "no_data", "status_code": code, "source": "semantic_scholar"}

    return {
        "title_fetched": safe_str(data.get("title")),
        "abstract": safe_str(data.get("abstract") or ""),
        "source": "semantic_scholar",
        "status_code": 200,
    }

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

def openalex_fetch(session: requests.Session, doi: str, email: str | None) -> dict:
    url = DEFAULT_OPENALEX_WORKS_URL.rstrip("/") + "/https://doi.org/" + requests.utils.quote(doi, safe=":/")
    h = {"Accept": "application/json", "User-Agent": f"step9 ({email or 'no-email'})"}
    params = {"mailto": email} if email else {}
    data, err, code = _http_get_json(session, url, headers=h, params=params, tries=4)

    if err or not data:
        return {"error": err or "no_data", "status_code": code, "source": "openalex"}

    inv = (data or {}).get("abstract_inverted_index")
    abst = _build_openalex_abstract(inv) if inv else None

    loc = (data or {}).get("primary_location") or {}
    landing = loc.get("landing_page_url")

    return {
        "title_fetched": safe_str(data.get("title")),
        "abstract": safe_str(abst) if abst else "",
        "landing_url": safe_str(landing) if landing else "",
        "source": "openalex",
        "status_code": 200,
    }

def crossref_fetch(session: requests.Session, doi: str, email: str | None) -> dict:
    url = DEFAULT_CROSSREF_WORKS_URL.rstrip("/") + "/" + requests.utils.quote(doi, safe="")
    h = {"Accept": "application/json", "User-Agent": f"step9 ({email or 'no-email'})"}
    params = {"mailto": email} if email else {}
    data, err, code = _http_get_json(session, url, headers=h, params=params, tries=4)

    if err or not data:
        return {"error": err or "no_data", "status_code": code, "source": "crossref"}

    msg = (data or {}).get("message") or {}
    title = (msg.get("title") or [None])[0]
    abst = msg.get("abstract")
    return {
        "title_fetched": safe_str(title),
        "abstract": safe_str(strip_tags(abst)) if abst else "",
        "source": "crossref",
        "status_code": 200,
    }

def unpaywall_fetch(session: requests.Session, doi: str, email: str | None) -> dict:
    if not email:
        return {"error": "no_email", "status_code": None, "source": "unpaywall"}
    url = DEFAULT_UNPAYWALL_URL.rstrip("/") + "/" + requests.utils.quote(doi, safe="")
    h = {"Accept": "application/json", "User-Agent": f"step9 ({email})"}
    data, err, code = _http_get_json(session, url, headers=h, params={"email": email}, tries=3)

    if err or not data:
        return {"error": err or "no_data", "status_code": code, "source": "unpaywall"}

    best = (data or {}).get("best_oa_location") or {}
    return {"landing_url": safe_str(best.get("url")), "source": "unpaywall", "status_code": 200}

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
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        backoff = 1.0
        last_status = None
        for _ in range(3):
            r = session.get(url, headers=h, timeout=20, allow_redirects=True)
            last_status = r.status_code
            if r.status_code in (429,) or r.status_code >= 500:
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
            raw_head,
            re.DOTALL | re.IGNORECASE,
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
                        return {"abstract": safe_str(strip_tags(v)), "source": "landing_json_ld", "status_code": 200}

        p = MetaParser()
        p.feed(raw_head)
        for k in ("citation_abstract", "dc.description", "og:description", "description"):
            v = p.metas.get(k)
            if isinstance(v, str) and len(v) > 50:
                return {"abstract": safe_str(strip_tags(v)), "source": "landing_meta", "status_code": 200}

        return {"error": "no_abstract_found", "status_code": 200, "source": "landing_page"}
    except Exception as e:
        return {"error": str(e)[:300], "status_code": None, "source": "landing_page"}


# ----------------------------
# RIS writing (same as step8 writer)
# ----------------------------

def split_pages(page_range: str) -> Tuple[str, str]:
    pr = safe_str(page_range)
    if not pr:
        return "", ""
    if "-" in pr:
        a, b = pr.split("-", 1)
        return a.strip(), b.strip()
    return pr.strip(), ""

def ris_write_record(f, rec: dict) -> None:
    ty = rec.get("TY") or "JOUR"
    f.write(f"TY  - {ty}\r\n")
    for au in rec.get("AU", []):
        au = safe_str(au)
        if au:
            f.write(f"AU  - {au}\r\n")
    if rec.get("TI"):
        f.write(f"TI  - {safe_str(rec['TI'])}\r\n")
    if rec.get("T2"):
        f.write(f"T2  - {safe_str(rec['T2'])}\r\n")
    if rec.get("PY"):
        f.write(f"PY  - {safe_str(rec['PY'])}\r\n")
    if rec.get("VL"):
        f.write(f"VL  - {safe_str(rec['VL'])}\r\n")
    if rec.get("IS"):
        f.write(f"IS  - {safe_str(rec['IS'])}\r\n")
    if rec.get("SP"):
        f.write(f"SP  - {safe_str(rec['SP'])}\r\n")
    if rec.get("EP"):
        f.write(f"EP  - {safe_str(rec['EP'])}\r\n")
    if rec.get("SN"):
        f.write(f"SN  - {safe_str(rec['SN'])}\r\n")
    if rec.get("DO"):
        f.write(f"DO  - {safe_str(rec['DO'])}\r\n")
    if rec.get("UR"):
        f.write(f"UR  - {safe_str(rec['UR'])}\r\n")
    if rec.get("AB"):
        f.write(f"AB  - {safe_str(rec['AB'])}\r\n")
    f.write("ER  - \r\n\r\n")

def df_to_ris(df: pd.DataFrame, out_path: str) -> int:
    n = 0
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        for _, r in df.iterrows():
            title = safe_str(r.get("title"))
            journal = safe_str(r.get("publicationName")) or safe_str(r.get("xref_journal"))
            year = safe_str(r.get("year"))
            doi_core = normalize_doi(r.get("doi"))
            url = f"https://doi.org/{doi_core}" if doi_core else safe_str(r.get("prism_url"))

            authors = []
            aj = safe_str(r.get("authors_joined"))
            if aj:
                authors = [a.strip() for a in aj.split(";") if a.strip()]
            else:
                an = safe_str(r.get("author_names"))
                if an:
                    authors = [a.strip() for a in an.split(";")] if ";" in an else [an]
                else:
                    cr = safe_str(r.get("creator"))
                    if cr:
                        authors = [cr]

            vol = safe_str(r.get("volume")) or safe_str(r.get("xref_volume"))
            iss = safe_str(r.get("issue")) or safe_str(r.get("xref_issue"))
            pages = safe_str(r.get("page_range")) or safe_str(r.get("xref_page"))
            sp, ep = split_pages(pages)
            issn = safe_str(r.get("issn")) or safe_str(r.get("xref_issn"))

            abstract = safe_str(r.get("abstract")) or safe_str(r.get("xref_abstract"))

            ty = "JOUR" if journal else "GEN"
            rec = {
                "TY": ty,
                "AU": authors,
                "TI": title,
                "T2": journal,
                "PY": year,
                "VL": vol,
                "IS": iss,
                "SP": sp,
                "EP": ep,
                "SN": issn,
                "DO": doi_core,
                "UR": url,
                "AB": abstract,
            }
            ris_write_record(f, rec)
            n += 1
    return n


# ----------------------------
# Step 9 core
# ----------------------------

def step9_enrich_abstracts(
    *,
    out_dir: str,
    sleep_s: float,
    cache_ttl_days: Optional[int],
    force_refresh: bool,
    max_records: Optional[int],
) -> dict:
    # inputs from step8
    step8_dir = os.path.join(out_dir, "step8")
    in_csv = os.path.join(step8_dir, "step8_scopus_cleaned.csv")

    # outputs to step9
    step9_dir = os.path.join(out_dir, "step9")
    os.makedirs(step9_dir, exist_ok=True)

    out_csv = os.path.join(step9_dir, "step9_scopus_enriched.csv")
    out_ris = os.path.join(step9_dir, "step9_scopus_enriched.ris")
    cache_path = os.path.join(step9_dir, "step9_abstract_cache.json")
    summary_path = os.path.join(step9_dir, "step9_summary.json")
    missing_csv = os.path.join(step9_dir, "step9_missing.csv")

    if not os.path.exists(in_csv):
        raise SystemExit(f"[step9] Missing input: {in_csv}")

    if load_dotenv is not None:
        try:
            load_dotenv()
        except Exception:
            pass

    api_key = os.getenv("SCOPUS_API_KEY", "").strip()
    inst_token = os.getenv("SCOPUS_INST_TOKEN", "").strip() or None
    email = os.getenv("CONTACT_EMAIL", "").strip() or None
    auth = ScopusAuth(api_key=api_key, inst_token=inst_token)

    df = pd.read_csv(in_csv, dtype=str, keep_default_na=False, na_filter=False)

    # Ensure expected columns exist
    for c in ("doi", "title", "publicationName", "year", "prism_url", "dedupe_key", "scopus_id", "xref_abstract"):
        if c not in df.columns:
            df[c] = ""

    # step9 columns
    for c in (
        "abstract",
        "abstract_source",
        "abstract_status",
        "abstract_error",
        "abstract_fetched_utc",
        "abstract_last_attempt_utc",
        "abstract_fail_count",
    ):
        if c not in df.columns:
            df[c] = ""

    # derived identifiers
    df["doi_core"] = df["doi"].apply(normalize_doi)
    df["scopus_id_core"] = df["scopus_id"].apply(lambda x: safe_str(x).strip())
    # fill from prism_url if missing
    missing_sid = df["scopus_id_core"].str.strip() == ""
    df.loc[missing_sid, "scopus_id_core"] = df.loc[missing_sid, "prism_url"].apply(extract_scopus_id_from_prism_url)

    def row_has_any_abstract(row) -> bool:
        a = safe_str(row.get("abstract"))
        xa = safe_str(row.get("xref_abstract"))
        return bool(a.strip()) or bool(xa.strip())

    # Preflight breakdown
    total_rows = int(len(df))
    has_abs = df.apply(row_has_any_abstract, axis=1)
    has_doi = df["doi_core"].astype(str).str.strip() != ""
    has_sid = df["scopus_id_core"].astype(str).str.strip() != ""
    has_prism = df["prism_url"].astype(str).str.strip() != ""

    eligible_doi = (~has_abs) & has_doi
    eligible_sid = (~has_abs) & (~has_doi) & has_sid
    eligible_landing_only = (~has_abs) & (~has_doi) & (~has_sid) & has_prism
    excluded = (~has_abs) & (~has_doi) & (~has_sid) & (~has_prism)

    idxs = df[(eligible_doi | eligible_sid | eligible_landing_only)].index.tolist()

    # Stable, deterministic ordering: as appears in input
    if max_records is not None:
        idxs = idxs[: max(0, int(max_records))]

    preflight = {
        "total_rows": total_rows,
        "already_has_abstract": int(has_abs.sum()),
        "eligible_via_doi": int(eligible_doi.sum()),
        "eligible_via_scopus_id": int(eligible_sid.sum()),
        "eligible_via_landing_only": int(eligible_landing_only.sum()),
        "excluded_no_identifiers": int(excluded.sum()),
        "will_process_count": int(len(idxs)),
        "force_refresh": bool(force_refresh),
        "cache_ttl_days": cache_ttl_days,
        "has_scopus_api_key": bool(bool(api_key)),
        "has_contact_email": bool(bool(email)),
        "timestamp_utc": utc_now_iso(),
        "input_csv": in_csv,
        "output_dir": step9_dir,
    }

    print("------------------------------------------------------------")
    print("▶️ Step 9: Abstract enrichment [step9_enrich_abstracts]")
    print("------------------------------------------------------------")
    print(
        "[step9] Preflight: "
        f"total={preflight['total_rows']} | "
        f"already_has={preflight['already_has_abstract']} | "
        f"eligible_doi={preflight['eligible_via_doi']} | "
        f"eligible_scopus_id={preflight['eligible_via_scopus_id']} | "
        f"eligible_landing_only={preflight['eligible_via_landing_only']} | "
        f"excluded_no_id={preflight['excluded_no_identifiers']} | "
        f"will_process={preflight['will_process_count']}"
    )
    if not api_key:
        print("[step9] Note: SCOPUS_API_KEY missing -> Elsevier calls will be skipped (lower hit-rate).")
    if not email:
        print("[step9] Note: CONTACT_EMAIL missing -> Unpaywall disabled.")

    cache: Dict[str, dict] = load_json(cache_path, default={})

    stats = {
        **preflight,
        "cached_ok": 0,
        "cached_failed_skipped": 0,
        "fresh_ok": 0,
        "fresh_failed": 0,
        "ok_by_source": {},
        "processed": 0,
    }

    missing_rows = []

    it = idxs
    if tqdm is not None:
        it = tqdm(idxs, desc="[step9] Enriching abstracts", unit="rec")

    with requests.Session() as session:
        for idx in it:
            doi = safe_str(df.at[idx, "doi_core"])
            sid = safe_str(df.at[idx, "scopus_id_core"])
            prism_url = safe_str(df.at[idx, "prism_url"])

            # Choose a deterministic cache key:
            # prefer DOI, else Scopus ID, else prism URL
            if doi:
                cache_key = f"doi:{doi}"
            elif sid:
                cache_key = f"scopus:{sid}"
            else:
                cache_key = f"url:{prism_url}"

            cached = cache.get(cache_key)

            # cache hit ok
            if isinstance(cached, dict) and cached.get("fetch_status") == "ok" and not force_refresh:
                df.at[idx, "abstract"] = safe_str(cached.get("abstract"))
                df.at[idx, "abstract_source"] = safe_str(cached.get("source"))
                df.at[idx, "abstract_status"] = "ok"
                df.at[idx, "abstract_error"] = ""
                df.at[idx, "abstract_fetched_utc"] = safe_str(cached.get("fetched_utc"))
                df.at[idx, "abstract_last_attempt_utc"] = safe_str(cached.get("last_attempt_utc"))
                df.at[idx, "abstract_fail_count"] = safe_str(cached.get("fail_count"))
                stats["cached_ok"] += 1
                src = safe_str(cached.get("source") or "unknown")
                stats["ok_by_source"][src] = stats["ok_by_source"].get(src, 0) + 1
                stats["processed"] += 1
                continue

            # cached failure skip until TTL
            if isinstance(cached, dict) and cached.get("fetch_status") != "ok" and not _should_retry_cached_failure(
                cached, force_refresh=force_refresh, ttl_days=cache_ttl_days
            ):
                df.at[idx, "abstract_status"] = safe_str(cached.get("fetch_status") or "failed")
                df.at[idx, "abstract_source"] = safe_str(cached.get("source"))
                df.at[idx, "abstract_error"] = safe_str(cached.get("error"))
                df.at[idx, "abstract_last_attempt_utc"] = safe_str(cached.get("last_attempt_utc"))
                df.at[idx, "abstract_fail_count"] = safe_str(cached.get("fail_count"))
                stats["cached_failed_skipped"] += 1
                missing_rows.append(
                    {
                        "cache_key": cache_key,
                        "doi_core": doi,
                        "scopus_id": sid,
                        "title": safe_str(df.at[idx, "title"]),
                        "last_source_attempted": safe_str(cached.get("source")),
                        "fetch_status": safe_str(cached.get("fetch_status")),
                        "error": safe_str(cached.get("error")),
                    }
                )
                stats["processed"] += 1
                continue

            # fresh attempt record
            rec = {
                "cache_key": cache_key,
                "doi_core": doi,
                "scopus_id": sid,
                "title_fetched": "",
                "abstract": "",
                "landing_url": "",
                "fetch_status": "no_abstract",
                "source": "",
                "status_code": "",
                "error": "",
                "fetched_utc": "",
                "last_attempt_utc": utc_now_iso(),
                "fail_count": str(int((cached or {}).get("fail_count") or 0)),
            }

            # ----------------
            # Fetch chain
            # ----------------

            # If DOI exists, run DOI chain
            if doi:
                # Elsevier DOI
                res = elsevier_article_fetch(session, auth, doi)
                if res.get("abstract"):
                    rec.update(res)
                    rec["fetch_status"] = "ok"
                else:
                    rec["source"] = rec["source"] or res.get("source", "")
                    rec["status_code"] = rec["status_code"] or (res.get("status_code") or "")
                    rec["error"] = rec["error"] or (res.get("error") or "")

                # Semantic Scholar
                if rec["fetch_status"] != "ok":
                    res = semantic_scholar_fetch(session, doi, email)
                    if res.get("abstract"):
                        rec.update(res)
                        rec["fetch_status"] = "ok"
                    else:
                        rec["source"] = rec["source"] or res.get("source", "")
                        rec["status_code"] = rec["status_code"] or (res.get("status_code") or "")
                        rec["error"] = rec["error"] or (res.get("error") or "")

                # OpenAlex
                if rec["fetch_status"] != "ok":
                    res = openalex_fetch(session, doi, email)
                    if res.get("landing_url"):
                        rec["landing_url"] = res.get("landing_url") or rec["landing_url"]
                    if res.get("abstract"):
                        rec.update(res)
                        rec["fetch_status"] = "ok"
                    else:
                        rec["source"] = rec["source"] or res.get("source", "")
                        rec["status_code"] = rec["status_code"] or (res.get("status_code") or "")
                        rec["error"] = rec["error"] or (res.get("error") or "")

                # Crossref
                if rec["fetch_status"] != "ok":
                    res = crossref_fetch(session, doi, email)
                    if res.get("abstract"):
                        rec.update(res)
                        rec["fetch_status"] = "ok"
                    else:
                        rec["source"] = rec["source"] or res.get("source", "")
                        rec["status_code"] = rec["status_code"] or (res.get("status_code") or "")
                        rec["error"] = rec["error"] or (res.get("error") or "")

                # Landing page (with Unpaywall helper)
                if rec["fetch_status"] != "ok":
                    landing = rec.get("landing_url") or ""
                    if not landing:
                        upw = unpaywall_fetch(session, doi, email)
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
                        rec["source"] = rec["source"] or res.get("source", "")
                        rec["status_code"] = rec["status_code"] or (res.get("status_code") or "")
                        rec["error"] = rec["error"] or (res.get("error") or "")

            # Else: try scopus_id chain
            elif sid:
                # Elsevier Scopus ID
                res = elsevier_scopusid_fetch(session, auth, sid)
                if res.get("abstract"):
                    rec.update(res)
                    rec["fetch_status"] = "ok"
                else:
                    rec["source"] = rec["source"] or res.get("source", "")
                    rec["status_code"] = rec["status_code"] or (res.get("status_code") or "")
                    rec["error"] = rec["error"] or (res.get("error") or "")

                # Landing scrape on prism_url (best we have)
                if rec["fetch_status"] != "ok":
                    landing = prism_url or ""
                    if not landing:
                        landing = f"https://api.elsevier.com/content/abstract/scopus_id/{sid}"
                    res = fetch_landing_abstract(session, landing)
                    if res.get("abstract"):
                        rec.update(res)
                        rec["landing_url"] = landing
                        rec["fetch_status"] = "ok"
                    else:
                        rec["landing_url"] = landing
                        rec["source"] = rec["source"] or res.get("source", "")
                        rec["status_code"] = rec["status_code"] or (res.get("status_code") or "")
                        rec["error"] = rec["error"] or (res.get("error") or "")

            # Else: landing-only
            else:
                landing = prism_url
                res = fetch_landing_abstract(session, landing)
                if res.get("abstract"):
                    rec.update(res)
                    rec["landing_url"] = landing
                    rec["fetch_status"] = "ok"
                else:
                    rec["landing_url"] = landing
                    rec["source"] = rec["source"] or res.get("source", "")
                    rec["status_code"] = rec["status_code"] or (res.get("status_code") or "")
                    rec["error"] = rec["error"] or (res.get("error") or "")

            rec["last_attempt_utc"] = utc_now_iso()

            if rec["fetch_status"] == "ok":
                rec["fetched_utc"] = rec["last_attempt_utc"]
                stats["fresh_ok"] += 1
                src = safe_str(rec.get("source") or "unknown")
                stats["ok_by_source"][src] = stats["ok_by_source"].get(src, 0) + 1
            else:
                stats["fresh_failed"] += 1
                rec["fail_count"] = str(int(rec["fail_count"] or "0") + 1)
                missing_rows.append(
                    {
                        "cache_key": cache_key,
                        "doi_core": doi,
                        "scopus_id": sid,
                        "title": safe_str(df.at[idx, "title"]),
                        "last_source_attempted": safe_str(rec.get("source")),
                        "fetch_status": safe_str(rec.get("fetch_status")),
                        "error": safe_str(rec.get("error")),
                    }
                )

            # Write back to df
            if rec.get("abstract"):
                df.at[idx, "abstract"] = safe_str(rec.get("abstract"))
            df.at[idx, "abstract_source"] = safe_str(rec.get("source"))
            df.at[idx, "abstract_status"] = safe_str(rec.get("fetch_status"))
            df.at[idx, "abstract_error"] = safe_str(rec.get("error"))
            df.at[idx, "abstract_fetched_utc"] = safe_str(rec.get("fetched_utc"))
            df.at[idx, "abstract_last_attempt_utc"] = safe_str(rec.get("last_attempt_utc"))
            df.at[idx, "abstract_fail_count"] = safe_str(rec.get("fail_count"))

            cache[cache_key] = rec
            stats["processed"] += 1

            # Periodic cache flush
            if (stats["fresh_ok"] + stats["fresh_failed"]) % 50 == 0:
                save_json(cache_path, cache)

            if sleep_s and sleep_s > 0:
                time.sleep(float(sleep_s))

    # Final cache save
    save_json(cache_path, cache)

    # final abstract present count (either abstract or xref_abstract)
    abs_any = (df["abstract"].astype(str).str.strip() != "") | (df["xref_abstract"].astype(str).str.strip() != "")

    # Drop internal derived columns from output
    out_df = df.drop(columns=["doi_core", "scopus_id_core"], errors="ignore")

    # Write CSV
    out_df.to_csv(out_csv, index=False)

    # Write RIS
    ris_n = df_to_ris(out_df, out_ris)

    # Missing CSV
    if missing_rows:
        pd.DataFrame(missing_rows).to_csv(missing_csv, index=False)
    else:
        if os.path.exists(missing_csv):
            os.remove(missing_csv)

    summary = dict(stats)
    summary.update(
        {
            "ok_total": int(stats["cached_ok"] + stats["fresh_ok"]),
            "failed_total": int(stats["cached_failed_skipped"] + stats["fresh_failed"]),
            "final_abstract_present": int(abs_any.sum()),
            "final_abstract_missing": int(len(df) - int(abs_any.sum())),
            "ris_records_written": int(ris_n),
            "output_csv": out_csv,
            "output_ris": out_ris,
            "cache_path": cache_path,
            "summary_path": summary_path,
            "missing_csv": missing_csv,
            "completed_timestamp_utc": utc_now_iso(),
        }
    )
    save_json(summary_path, summary)

    print(
        f"[step9] Done. will_process={preflight['will_process_count']} "
        f"ok_total={summary['ok_total']} failed_total={summary['failed_total']} "
        f"final_present={summary['final_abstract_present']} final_missing={summary['final_abstract_missing']}"
    )
    print(f"[step9] Wrote: {out_csv}")
    print(f"[step9] Wrote: {out_ris} ({ris_n} records)")
    print(f"[step9] Cache: {cache_path}")
    print(f"[step9] Summary: {summary_path}")
    if missing_rows:
        print(f"[step9] Missing: {missing_csv} ({len(missing_rows)} rows)")

    return summary


def run(config: dict) -> dict:
    """
    Runner entrypoint expected by scripts/run.py.
    Deterministic: uses constants above unless you deliberately edit them in the file.
    """
    return step9_enrich_abstracts(
        out_dir=OUT_DIR,
        sleep_s=SLEEP_S,
        cache_ttl_days=CACHE_TTL_DAYS,
        force_refresh=FORCE_REFRESH,
        max_records=MAX_RECORDS,
    )


def run_step9(config: dict) -> dict:
    return run(config)


def main() -> int:
    _ = step9_enrich_abstracts(
        out_dir=OUT_DIR,
        sleep_s=SLEEP_S,
        cache_ttl_days=CACHE_TTL_DAYS,
        force_refresh=FORCE_REFRESH,
        max_records=MAX_RECORDS,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
