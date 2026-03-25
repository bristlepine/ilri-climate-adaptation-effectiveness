#!/usr/bin/env python3
"""
step9a_enrich_from_ris.py

Step 9a: Supplement step9's enriched records by:
  1) Parsing one or more user-supplied RIS files (exported from EPPI Reviewer or Scopus)
     and injecting any abstracts found there into records still missing one.
  2) Re-attempting API enrichment for records that remain missing after the RIS merge
     (bypassing the cached failures from step9).

Supports multiple input files via a glob pattern (e.g. step9a1_ExportedRis_*.txt).
Tracks an iteration label and timestamp in the summary for provenance.
Generates a diff vs the step9 baseline showing what changed.

Inputs:
  - outputs/step9/step9_scopus_enriched.csv
  - RIS files matched by ris_glob (set in config.py or STEP9A_RIS_GLOB env var)

Outputs (ALL under outputs/step9a/):
  - step9a_scopus_enriched.csv
  - step9a_scopus_enriched.ris
  - step9a_abstract_cache.json
  - step9a_summary.json
  - step9a_missing.csv
  - step9a_diff.csv           (records where abstract changed vs step9)

Env vars:
  - SCOPUS_API_KEY
  - SCOPUS_INST_TOKEN (optional)
  - CONTACT_EMAIL
  - STEP9A_RIS_GLOB   (optional; glob override)

Run:
  python step9a_enrich_from_ris.py
"""

from __future__ import annotations

import glob as glob_module
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from html.parser import HTMLParser
from typing import Optional, Tuple, Dict, Any, List
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
# Deterministic settings
# ----------------------------

OUT_DIR = str((Path(__file__).resolve().parent / "outputs").resolve())
DATA_DIR = str((Path(__file__).resolve().parent / "data").resolve())

SLEEP_S = 0.05
CACHE_TTL_DAYS: Optional[int] = 30
FORCE_REFRESH_API = False
MAX_RECORDS: Optional[int] = None


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
# Utilities
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
    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s, flags=re.IGNORECASE).strip()
    s = s.rstrip(" .),;]}>")
    return s.lower()

def normalize_title(t: str) -> str:
    s = safe_str(t).lower()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

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

def extract_scopus_id_from_url(url: str) -> str:
    u = safe_str(url)
    if not u:
        return ""
    m = re.search(r"/scopus_id/(\d+)", u)
    return m.group(1) if m else ""

def extract_doi_from_url(url: str) -> str:
    u = safe_str(url)
    if not u:
        return ""
    # doi.org URLs
    m = re.match(r"^https?://(dx\.)?doi\.org/(.+)", u, flags=re.IGNORECASE)
    if m:
        return normalize_doi(m.group(2))
    return ""


# ----------------------------
# RIS parser
# ----------------------------

def parse_ris_file(ris_path: str) -> Tuple[List[dict], dict]:
    """
    Parse a RIS file. Returns (records_with_abstract, file_stats).

    file_stats keys: total_records, with_abstract, without_abstract, file_path
    Only records with a non-empty abstract are included in the returned list.

    Handles EPPI Reviewer export format:
      - T1 / TI  → title
      - AB / N2  → abstract (multi-line accumulation)
      - DO        → doi
      - UR        → if scopus URL → scopus_id; if doi URL → doi
      - U1        → eppi_id
      - JF / T2  → journal
    """
    if not os.path.exists(ris_path):
        raise FileNotFoundError(f"RIS file not found: {ris_path}")

    TAG_MAP = {
        "TI": "title",
        "T1": "title",
        "AB": "abstract",
        "N2": "abstract",
        "DO": "doi",
        "JF": "journal",
        "T2": "journal",
        "PY": "year",
        "Y1": "year",
        "AU": "authors",
        "A1": "authors",
        "U1": "eppi_id",
    }
    # UR is handled specially below

    records_with_abstract: List[dict] = []
    total_records = 0
    current: Dict[str, Any] = {}

    def flush(rec: dict) -> None:
        nonlocal total_records
        total_records += 1

        # Normalise UR: extract scopus_id or doi
        ur = safe_str(rec.pop("_ur", ""))
        if ur:
            sid_from_ur = extract_scopus_id_from_url(ur)
            doi_from_ur = extract_doi_from_url(ur) if not sid_from_ur else ""
            if sid_from_ur and not rec.get("scopus_id_core"):
                rec["scopus_id_core"] = sid_from_ur
            if doi_from_ur and not rec.get("doi_core"):
                rec["doi_core"] = doi_from_ur

        # Normalise doi from DO field
        if rec.get("doi") and not rec.get("doi_core"):
            rec["doi_core"] = normalize_doi(rec["doi"])

        # Clean scopus_id — strip "2-s2.0-" prefix
        sid = safe_str(rec.get("scopus_id_core", ""))
        sid = re.sub(r"^2-s2\.0-", "", sid).strip()
        rec["scopus_id_core"] = sid

        rec["title_norm"] = normalize_title(rec.get("title", ""))

        if isinstance(rec.get("authors"), list):
            rec["authors_joined"] = "; ".join(a for a in rec["authors"] if a)

        ab = safe_str(rec.get("abstract", ""))
        if ab:
            records_with_abstract.append(rec)

    try:
        with open(ris_path, "r", encoding="utf-8-sig", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        raise IOError(f"Could not read RIS file: {e}")

    for raw_line in lines:
        line = raw_line.rstrip("\r\n")

        if re.match(r"^ER\s*-", line):
            if current:
                flush(current)
            current = {}
            continue

        m = re.match(r"^([A-Z][A-Z0-9])\s{2}-\s?(.*)", line)
        if m:
            tag, value = m.group(1), m.group(2).strip()
            if tag == "UR":
                current["_ur"] = safe_str(value)
            elif tag in TAG_MAP:
                field = TAG_MAP[tag]
                if field == "authors":
                    current.setdefault("authors", [])
                    if value:
                        current["authors"].append(safe_str(value))
                elif field == "abstract":
                    existing = safe_str(current.get("abstract", ""))
                    current["abstract"] = (existing + " " + safe_str(value)).strip() if existing else safe_str(value)
                else:
                    if not current.get(field):
                        current[field] = safe_str(value)
        else:
            # Continuation line for abstract
            if "abstract" in current and line.strip() and not re.match(r"^[A-Z][A-Z0-9]\s{2}-", line):
                current["abstract"] = (current["abstract"] + " " + line.strip()).strip()

    if current:
        flush(current)

    without_abstract = total_records - len(records_with_abstract)
    file_stats = {
        "file_path": ris_path,
        "total_records": total_records,
        "with_abstract": len(records_with_abstract),
        "without_abstract": without_abstract,
    }
    return records_with_abstract, file_stats


def load_all_ris_files(ris_paths: List[str]) -> Tuple[List[dict], List[dict]]:
    """
    Parse all RIS files. Returns (all_records_with_abstract, per_file_stats).
    """
    all_records: List[dict] = []
    all_stats: List[dict] = []
    for path in ris_paths:
        records, stats = parse_ris_file(path)
        all_records.extend(records)
        all_stats.append(stats)
        print(f"[step9a]   {os.path.basename(path)}: "
              f"total={stats['total_records']} | with_abstract={stats['with_abstract']} | "
              f"without_abstract={stats['without_abstract']}")
    return all_records, all_stats


def build_ris_lookup(records: List[dict]) -> Tuple[dict, dict, dict]:
    doi_lookup: Dict[str, str] = {}
    scopus_lookup: Dict[str, str] = {}
    title_lookup: Dict[str, str] = {}
    for rec in records:
        ab = safe_str(rec.get("abstract", ""))
        if not ab:
            continue
        doi = rec.get("doi_core", "")
        sid = rec.get("scopus_id_core", "")
        tn = rec.get("title_norm", "")
        if doi:
            doi_lookup.setdefault(doi, ab)
        if sid:
            scopus_lookup.setdefault(sid, ab)
        if tn:
            title_lookup.setdefault(tn, ab)
    return doi_lookup, scopus_lookup, title_lookup


def resolve_ris_paths(
    ris_glob: Optional[str],
    ris_path_override: Optional[str],
    data_dir: str,
) -> List[str]:
    """
    Resolve the list of RIS files to process, in priority order:
      1. ris_path_override (single explicit path)
      2. ris_glob (glob pattern)
      3. STEP9A_RIS_GLOB env var
      4. STEP9A_RIS env var (single path, legacy)
      5. Default: data_dir/step9a_update.ris
    """
    if ris_path_override and os.path.exists(ris_path_override):
        return [ris_path_override]

    pattern = (
        ris_glob
        or os.environ.get("STEP9A_RIS_GLOB", "").strip()
        or os.environ.get("STEP9A_RIS", "").strip()
    )
    if pattern:
        # Check if it's a glob pattern or a direct path
        if any(c in pattern for c in ("*", "?", "[")):
            paths = sorted(glob_module.glob(pattern))
        else:
            paths = [pattern] if os.path.exists(pattern) else []
        if paths:
            return paths

    # Default fallback
    default = os.path.join(data_dir, "step9a_update.ris")
    return [default] if os.path.exists(default) else []


# ----------------------------
# API fetchers (same as step9)
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

def elsevier_article_fetch(session, auth, doi):
    if not auth.api_key:
        return {"error": "no_api_key", "status_code": None, "source": "elsevier_doi"}
    url = DEFAULT_ELSEVIER_ARTICLE_URL.rstrip("/") + "/" + requests.utils.quote(doi, safe="")
    data, err, code = _http_get_json(session, url, headers=scopus_headers(auth),
                                     params={"view": "META_ABS", "httpAccept": "application/json"})
    if err or not data:
        return {"error": err or "no_data", "status_code": code, "source": "elsevier_doi"}
    abstract = _find_first(data, ("dc:description", "ce:abstract", "abstract", "description"))
    return {"title_fetched": safe_str(_find_first(data, ("dc:title", "title"))),
            "abstract": safe_str(strip_tags(abstract)) if abstract else "",
            "source": "elsevier_doi", "status_code": 200}

def elsevier_scopusid_fetch(session, auth, scopus_id):
    if not auth.api_key:
        return {"error": "no_api_key", "status_code": None, "source": "elsevier_scopus_id"}
    url = DEFAULT_ELSEVIER_ABSTRACT_SCOPUSID_URL.rstrip("/") + "/" + requests.utils.quote(scopus_id, safe="")
    data, err, code = _http_get_json(session, url, headers=scopus_headers(auth),
                                     params={"view": "FULL", "httpAccept": "application/json"})
    if err or not data:
        return {"error": err or "no_data", "status_code": code, "source": "elsevier_scopus_id"}
    abstract = _find_first(data, ("dc:description", "ce:abstract", "abstract", "description"))
    return {"title_fetched": safe_str(_find_first(data, ("dc:title", "title"))),
            "abstract": safe_str(strip_tags(abstract)) if abstract else "",
            "source": "elsevier_scopus_id", "status_code": 200}

def semantic_scholar_fetch(session, doi, email):
    url = DEFAULT_SEMANTIC_SCHOLAR_URL.rstrip("/") + "/DOI:" + requests.utils.quote(doi, safe=":/")
    h = {"Accept": "application/json", "User-Agent": f"step9a ({email or 'no-email'})"}
    data, err, code = _http_get_json(session, url, headers=h, params={"fields": "title,abstract"}, tries=4)
    if err or not data:
        return {"error": err or "no_data", "status_code": code, "source": "semantic_scholar"}
    return {"title_fetched": safe_str(data.get("title")),
            "abstract": safe_str(data.get("abstract") or ""),
            "source": "semantic_scholar", "status_code": 200}

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
    return re.sub(r"\s+", " ", out).strip() or None

def openalex_fetch(session, doi, email):
    url = DEFAULT_OPENALEX_WORKS_URL.rstrip("/") + "/https://doi.org/" + requests.utils.quote(doi, safe=":/")
    h = {"Accept": "application/json", "User-Agent": f"step9a ({email or 'no-email'})"}
    params = {"mailto": email} if email else {}
    data, err, code = _http_get_json(session, url, headers=h, params=params, tries=4)
    if err or not data:
        return {"error": err or "no_data", "status_code": code, "source": "openalex"}
    inv = (data or {}).get("abstract_inverted_index")
    abst = _build_openalex_abstract(inv) if inv else None
    loc = (data or {}).get("primary_location") or {}
    return {"title_fetched": safe_str(data.get("title")),
            "abstract": safe_str(abst) if abst else "",
            "landing_url": safe_str(loc.get("landing_page_url") or ""),
            "source": "openalex", "status_code": 200}

def crossref_fetch(session, doi, email):
    url = DEFAULT_CROSSREF_WORKS_URL.rstrip("/") + "/" + requests.utils.quote(doi, safe="")
    h = {"Accept": "application/json", "User-Agent": f"step9a ({email or 'no-email'})"}
    params = {"mailto": email} if email else {}
    data, err, code = _http_get_json(session, url, headers=h, params=params, tries=4)
    if err or not data:
        return {"error": err or "no_data", "status_code": code, "source": "crossref"}
    msg = (data or {}).get("message") or {}
    abst = msg.get("abstract")
    return {"title_fetched": safe_str((msg.get("title") or [None])[0]),
            "abstract": safe_str(strip_tags(abst)) if abst else "",
            "source": "crossref", "status_code": 200}

def unpaywall_fetch(session, doi, email):
    if not email:
        return {"error": "no_email", "status_code": None, "source": "unpaywall"}
    url = DEFAULT_UNPAYWALL_URL.rstrip("/") + "/" + requests.utils.quote(doi, safe="")
    h = {"Accept": "application/json", "User-Agent": f"step9a ({email})"}
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

def fetch_landing_abstract(session, url):
    try:
        h = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
             "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
        backoff = 1.0
        last_status = None
        for _ in range(3):
            r = session.get(url, headers=h, timeout=20, allow_redirects=True)
            last_status = r.status_code
            if r.status_code in (429,) or r.status_code >= 500:
                time.sleep(backoff); backoff = min(backoff * 1.7, 10.0); continue
            break
        if last_status != 200:
            return {"error": f"HTTP {last_status}", "status_code": last_status, "source": "landing_page"}
        raw_head = (r.text or "")[:250000]
        for match in re.finditer(r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>',
                                 raw_head, re.DOTALL | re.IGNORECASE):
            blob = match.group(1).strip()
            if not blob:
                continue
            try:
                j = json.loads(blob)
            except Exception:
                continue
            for item in (j if isinstance(j, list) else [j]):
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
# RIS writer
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
    for tag, key in [("TI", "TI"), ("T2", "T2"), ("PY", "PY"), ("VL", "VL"),
                     ("IS", "IS"), ("SP", "SP"), ("EP", "EP"), ("SN", "SN"),
                     ("DO", "DO"), ("UR", "UR"), ("AB", "AB")]:
        if rec.get(key):
            f.write(f"{tag}  - {safe_str(rec[key])}\r\n")
    f.write("ER  - \r\n\r\n")

def df_to_ris(df: pd.DataFrame, out_path: str) -> int:
    n = 0
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        for _, r in df.iterrows():
            doi_core = normalize_doi(r.get("doi"))
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
            pages = safe_str(r.get("page_range")) or safe_str(r.get("xref_page"))
            sp, ep = split_pages(pages)
            journal = safe_str(r.get("publicationName")) or safe_str(r.get("xref_journal"))
            rec = {
                "TY": "JOUR" if journal else "GEN",
                "AU": authors,
                "TI": safe_str(r.get("title")),
                "T2": journal,
                "PY": safe_str(r.get("year")),
                "VL": safe_str(r.get("volume")) or safe_str(r.get("xref_volume")),
                "IS": safe_str(r.get("issue")) or safe_str(r.get("xref_issue")),
                "SP": sp, "EP": ep,
                "SN": safe_str(r.get("issn")) or safe_str(r.get("xref_issn")),
                "DO": doi_core,
                "UR": f"https://doi.org/{doi_core}" if doi_core else safe_str(r.get("prism_url")),
                "AB": safe_str(r.get("abstract")) or safe_str(r.get("xref_abstract")),
            }
            ris_write_record(f, rec)
            n += 1
    return n


# ----------------------------
# Step 9a core
# ----------------------------

def step9a_enrich_from_ris(
    *,
    out_dir: str,
    data_dir: str,
    sleep_s: float,
    cache_ttl_days: Optional[int],
    force_refresh_api: bool,
    max_records: Optional[int],
    ris_glob: Optional[str] = None,
    ris_path_override: Optional[str] = None,
    iteration: str = "step9a",
) -> dict:

    step9_dir = os.path.join(out_dir, "step9")
    step9a_dir = os.path.join(out_dir, "step9a")
    os.makedirs(step9a_dir, exist_ok=True)

    in_csv = os.path.join(step9_dir, "step9_scopus_enriched.csv")
    if not os.path.exists(in_csv):
        raise SystemExit(f"[step9a] Missing step9 output: {in_csv}")

    out_csv      = os.path.join(step9a_dir, "step9a_scopus_enriched.csv")
    out_ris      = os.path.join(step9a_dir, "step9a_scopus_enriched.ris")
    cache_path   = os.path.join(step9a_dir, "step9a_abstract_cache.json")
    summary_path = os.path.join(step9a_dir, "step9a_summary.json")
    missing_csv  = os.path.join(step9a_dir, "step9a_missing.csv")
    diff_csv     = os.path.join(step9a_dir, "step9a_diff.csv")

    if load_dotenv is not None:
        try:
            load_dotenv()
        except Exception:
            pass

    api_key    = os.getenv("SCOPUS_API_KEY", "").strip()
    inst_token = os.getenv("SCOPUS_INST_TOKEN", "").strip() or None
    email      = os.getenv("CONTACT_EMAIL", "").strip() or None
    auth       = ScopusAuth(api_key=api_key, inst_token=inst_token)

    run_timestamp = utc_now_iso()

    print("------------------------------------------------------------")
    print(f"▶️  Step 9a: Abstract enrichment from RIS + API retry")
    print(f"    Iteration : {iteration}")
    print(f"    Timestamp : {run_timestamp}")
    print("------------------------------------------------------------")

    # --- Load step9 base ---
    df = pd.read_csv(in_csv, dtype=str, keep_default_na=False, na_filter=False)
    for c in ("doi", "title", "publicationName", "year", "prism_url",
              "dedupe_key", "scopus_id", "xref_abstract",
              "abstract", "abstract_source", "abstract_status",
              "abstract_error", "abstract_fetched_utc",
              "abstract_last_attempt_utc", "abstract_fail_count"):
        if c not in df.columns:
            df[c] = ""

    df["doi_core"]      = df["doi"].apply(normalize_doi)
    df["scopus_id_core"] = df["scopus_id"].apply(lambda x: safe_str(x).strip())
    mask_no_sid = df["scopus_id_core"].str.strip() == ""
    df.loc[mask_no_sid, "scopus_id_core"] = df.loc[mask_no_sid, "prism_url"].apply(
        extract_scopus_id_from_url
    )
    df["title_norm"] = df["title"].apply(normalize_title)

    # Snapshot of step9 abstracts for diff
    step9_abstracts = df["abstract"].copy()

    def row_has_abstract(row) -> bool:
        return bool(safe_str(row.get("abstract")).strip()) or bool(safe_str(row.get("xref_abstract")).strip())

    has_abs_before = df.apply(row_has_abstract, axis=1)
    total_rows     = len(df)
    missing_before = int((~has_abs_before).sum())

    print(f"[step9a] Loaded {total_rows} rows from step9. "
          f"Have abstract: {int(has_abs_before.sum())} | Missing: {missing_before}")

    # --- Resolve RIS file paths ---
    ris_paths = resolve_ris_paths(ris_glob, ris_path_override, data_dir)

    # --- Phase 1: RIS merge ---
    ris_merged   = 0
    per_file_stats: List[dict] = []
    ris_total_records     = 0
    ris_total_with_ab     = 0
    doi_lookup = scopus_lookup = title_lookup = {}

    if ris_paths:
        print(f"[step9a] Parsing {len(ris_paths)} RIS file(s):")
        all_ris_records, per_file_stats = load_all_ris_files(ris_paths)
        ris_total_records = sum(s["total_records"] for s in per_file_stats)
        ris_total_with_ab = sum(s["with_abstract"] for s in per_file_stats)
        doi_lookup, scopus_lookup, title_lookup = build_ris_lookup(all_ris_records)
        print(f"[step9a] RIS totals: records={ris_total_records} | with_abstract={ris_total_with_ab} | "
              f"DOI keys={len(doi_lookup)} | Scopus ID keys={len(scopus_lookup)} | Title keys={len(title_lookup)}")

        for idx in df[~has_abs_before].index:
            doi = safe_str(df.at[idx, "doi_core"])
            sid = safe_str(df.at[idx, "scopus_id_core"])
            tn  = safe_str(df.at[idx, "title_norm"])

            ab = ""
            source = ""
            if doi and doi in doi_lookup:
                ab, source = doi_lookup[doi], "ris_doi_match"
            elif sid and sid in scopus_lookup:
                ab, source = scopus_lookup[sid], "ris_scopus_id_match"
            elif tn and tn in title_lookup:
                ab, source = title_lookup[tn], "ris_title_match"

            if ab:
                df.at[idx, "abstract"]                    = ab
                df.at[idx, "abstract_source"]             = source
                df.at[idx, "abstract_status"]             = "ok"
                df.at[idx, "abstract_error"]              = ""
                df.at[idx, "abstract_fetched_utc"]        = utc_now_iso()
                df.at[idx, "abstract_last_attempt_utc"]   = utc_now_iso()
                ris_merged += 1
    else:
        print(f"[step9a] WARNING: No RIS files found. Skipping RIS merge.")
        print(f"[step9a]   Set step9a_ris_glob in config.py, or place a file at:")
        print(f"[step9a]   {os.path.join(data_dir, 'step9a_update.ris')}")

    has_abs_after_ris = df.apply(row_has_abstract, axis=1)
    missing_after_ris = int((~has_abs_after_ris).sum())
    print(f"[step9a] After RIS merge: +{ris_merged} abstracts | still missing={missing_after_ris}")

    # --- Phase 2: API retry for still-missing ---
    still_missing        = ~has_abs_after_ris
    has_doi              = df["doi_core"].str.strip() != ""
    has_sid              = df["scopus_id_core"].str.strip() != ""
    has_prism            = df["prism_url"].str.strip() != ""

    eligible_doi          = still_missing & has_doi
    eligible_sid          = still_missing & (~has_doi) & has_sid
    eligible_landing_only = still_missing & (~has_doi) & (~has_sid) & has_prism
    excluded              = still_missing & (~has_doi) & (~has_sid) & (~has_prism)

    retry_idxs = df[eligible_doi | eligible_sid | eligible_landing_only].index.tolist()
    if max_records is not None:
        retry_idxs = retry_idxs[: max(0, int(max_records))]

    print(f"[step9a] API retry: doi={int(eligible_doi.sum())} | scopus_id_only={int(eligible_sid.sum())} | "
          f"landing_only={int(eligible_landing_only.sum())} | no_id={int(excluded.sum())} | "
          f"will_process={len(retry_idxs)}")

    # Inherit step9 cache + any existing step9a cache
    step9_cache_path = os.path.join(step9_dir, "step9_abstract_cache.json")
    cache: Dict[str, dict] = load_json(step9_cache_path, default={})
    cache.update(load_json(cache_path, default={}))

    api_fresh_ok = api_fresh_failed = api_cached_ok = 0
    ok_by_source: Dict[str, int] = {}
    missing_rows: List[dict] = []

    it = retry_idxs
    if tqdm is not None:
        it = tqdm(retry_idxs, desc="[step9a] API retry", unit="rec")

    with requests.Session() as session:
        for idx in it:
            doi       = safe_str(df.at[idx, "doi_core"])
            sid       = safe_str(df.at[idx, "scopus_id_core"])
            prism_url = safe_str(df.at[idx, "prism_url"])

            cache_key = f"doi:{doi}" if doi else (f"scopus:{sid}" if sid else f"url:{prism_url}")
            cached    = cache.get(cache_key)

            # Cache hit ok → use it
            if isinstance(cached, dict) and cached.get("fetch_status") == "ok" and not force_refresh_api:
                df.at[idx, "abstract"]                  = safe_str(cached.get("abstract"))
                df.at[idx, "abstract_source"]           = safe_str(cached.get("source"))
                df.at[idx, "abstract_status"]           = "ok"
                df.at[idx, "abstract_error"]            = ""
                df.at[idx, "abstract_fetched_utc"]      = safe_str(cached.get("fetched_utc"))
                df.at[idx, "abstract_last_attempt_utc"] = safe_str(cached.get("last_attempt_utc"))
                df.at[idx, "abstract_fail_count"]       = safe_str(cached.get("fail_count"))
                api_cached_ok += 1
                src = safe_str(cached.get("source") or "unknown")
                ok_by_source[src] = ok_by_source.get(src, 0) + 1
                continue

            # step9a always retries cached failures (that's the whole point)
            rec = {
                "cache_key": cache_key,
                "doi_core": doi, "scopus_id": sid,
                "title_fetched": "", "abstract": "", "landing_url": "",
                "fetch_status": "no_abstract", "source": "", "status_code": "",
                "error": "", "fetched_utc": "",
                "last_attempt_utc": utc_now_iso(),
                "fail_count": str(int((cached or {}).get("fail_count") or 0)),
            }

            def _try(res):
                if res.get("abstract"):
                    rec.update(res); rec["fetch_status"] = "ok"; return True
                rec["source"]      = rec["source"]      or res.get("source", "")
                rec["status_code"] = rec["status_code"] or str(res.get("status_code") or "")
                rec["error"]       = rec["error"]       or (res.get("error") or "")
                return False

            if doi:
                _try(elsevier_article_fetch(session, auth, doi)) or \
                _try(semantic_scholar_fetch(session, doi, email)) or \
                _try(openalex_fetch(session, doi, email)) or \
                _try(crossref_fetch(session, doi, email))

                if rec["fetch_status"] != "ok":
                    landing = rec.get("landing_url") or ""
                    if not landing:
                        upw = unpaywall_fetch(session, doi, email)
                        landing = upw.get("landing_url") or ""
                    if not landing:
                        landing = f"https://doi.org/{doi}"
                    res = fetch_landing_abstract(session, landing)
                    rec["landing_url"] = landing
                    _try(res)

            elif sid:
                _try(elsevier_scopusid_fetch(session, auth, sid))
                if rec["fetch_status"] != "ok":
                    landing = prism_url or f"https://api.elsevier.com/content/abstract/scopus_id/{sid}"
                    rec["landing_url"] = landing
                    _try(fetch_landing_abstract(session, landing))

            else:
                rec["landing_url"] = prism_url
                _try(fetch_landing_abstract(session, prism_url))

            rec["last_attempt_utc"] = utc_now_iso()

            if rec["fetch_status"] == "ok":
                rec["fetched_utc"] = rec["last_attempt_utc"]
                api_fresh_ok += 1
                src = safe_str(rec.get("source") or "unknown")
                ok_by_source[src] = ok_by_source.get(src, 0) + 1
            else:
                api_fresh_failed += 1
                rec["fail_count"] = str(int(rec["fail_count"] or "0") + 1)
                missing_rows.append({
                    "cache_key": cache_key,
                    "doi_core": doi, "scopus_id": sid,
                    "title": safe_str(df.at[idx, "title"]),
                    "last_source_attempted": safe_str(rec.get("source")),
                    "fetch_status": safe_str(rec.get("fetch_status")),
                    "error": safe_str(rec.get("error")),
                })

            if rec.get("abstract"):
                df.at[idx, "abstract"] = safe_str(rec["abstract"])
            df.at[idx, "abstract_source"]           = safe_str(rec.get("source"))
            df.at[idx, "abstract_status"]           = safe_str(rec.get("fetch_status"))
            df.at[idx, "abstract_error"]            = safe_str(rec.get("error"))
            df.at[idx, "abstract_fetched_utc"]      = safe_str(rec.get("fetched_utc"))
            df.at[idx, "abstract_last_attempt_utc"] = safe_str(rec.get("last_attempt_utc"))
            df.at[idx, "abstract_fail_count"]       = safe_str(rec.get("fail_count"))

            cache[cache_key] = rec

            if (api_fresh_ok + api_fresh_failed) % 50 == 0:
                save_json(cache_path, cache)

            if sleep_s and sleep_s > 0:
                time.sleep(float(sleep_s))

    save_json(cache_path, cache)

    # Records with no identifiers — add to missing list
    for idx in df[excluded].index:
        missing_rows.append({
            "cache_key": "", "doi_core": safe_str(df.at[idx, "doi_core"]),
            "scopus_id": safe_str(df.at[idx, "scopus_id_core"]),
            "title": safe_str(df.at[idx, "title"]),
            "last_source_attempted": "none",
            "fetch_status": "no_identifiers",
            "error": "no doi, scopus_id, or prism_url",
        })

    # --- Diff vs step9 baseline ---
    diff_rows: List[dict] = []
    for idx in df.index:
        old_ab = safe_str(step9_abstracts.get(idx, ""))
        new_ab = safe_str(df.at[idx, "abstract"])
        if not old_ab and new_ab:
            diff_rows.append({
                "dedupe_key":       safe_str(df.at[idx, "dedupe_key"]) if "dedupe_key" in df.columns else "",
                "scopus_id":        safe_str(df.at[idx, "scopus_id_core"]),
                "doi_core":         safe_str(df.at[idx, "doi_core"]),
                "title":            safe_str(df.at[idx, "title"]),
                "abstract_source":  safe_str(df.at[idx, "abstract_source"]),
                "abstract_preview": new_ab[:200],
            })

    # --- Write outputs ---
    abs_any = (df["abstract"].str.strip() != "") | (df["xref_abstract"].str.strip() != "")
    out_df  = df.drop(columns=["doi_core", "scopus_id_core", "title_norm"], errors="ignore")
    out_df.to_csv(out_csv, index=False)
    ris_n = df_to_ris(out_df, out_ris)

    if missing_rows:
        pd.DataFrame(missing_rows).to_csv(missing_csv, index=False)
    elif os.path.exists(missing_csv):
        os.remove(missing_csv)

    if diff_rows:
        pd.DataFrame(diff_rows).to_csv(diff_csv, index=False)
    elif os.path.exists(diff_csv):
        os.remove(diff_csv)

    summary = {
        "iteration":              iteration,
        "run_timestamp_utc":      run_timestamp,
        "completed_timestamp_utc": utc_now_iso(),
        # Inputs
        "input_csv":              in_csv,
        "ris_files":              [s["file_path"] for s in per_file_stats],
        "ris_per_file_stats":     per_file_stats,
        "ris_total_records":      ris_total_records,
        "ris_total_with_abstract": ris_total_with_ab,
        "ris_total_without_abstract": ris_total_records - ris_total_with_ab,
        # Before/after
        "total_rows":             total_rows,
        "abstract_present_step9": int(has_abs_before.sum()),
        "abstract_missing_step9": missing_before,
        # Phase 1: RIS
        "ris_merged":             ris_merged,
        "abstract_missing_after_ris": missing_after_ris,
        # Phase 2: API
        "api_retry_eligible":     len(retry_idxs),
        "api_cached_ok":          api_cached_ok,
        "api_fresh_ok":           api_fresh_ok,
        "api_fresh_failed":       api_fresh_failed,
        "ok_by_source":           ok_by_source,
        # Final
        "final_abstract_present": int(abs_any.sum()),
        "final_abstract_missing": int(len(df) - int(abs_any.sum())),
        "gained_vs_step9":        len(diff_rows),
        # Outputs
        "output_csv":             out_csv,
        "output_ris":             out_ris,
        "cache_path":             cache_path,
        "summary_path":           summary_path,
        "missing_csv":            missing_csv,
        "diff_csv":               diff_csv,
        "ris_records_written":    int(ris_n),
    }
    save_json(summary_path, summary)

    print(
        f"[step9a] Done. "
        f"ris_merged={ris_merged} | "
        f"api_fresh_ok={api_fresh_ok} | "
        f"api_fresh_failed={api_fresh_failed} | "
        f"final_present={summary['final_abstract_present']} | "
        f"final_missing={summary['final_abstract_missing']} | "
        f"gained_vs_step9={len(diff_rows)}"
    )
    print(f"[step9a] Outputs → {step9a_dir}")
    if diff_rows:
        print(f"[step9a] Diff (new abstracts vs step9): {diff_csv} ({len(diff_rows)} records)")

    return summary


def run(config: dict) -> dict:
    return step9a_enrich_from_ris(
        out_dir=OUT_DIR,
        data_dir=DATA_DIR,
        sleep_s=SLEEP_S,
        cache_ttl_days=CACHE_TTL_DAYS,
        force_refresh_api=FORCE_REFRESH_API,
        max_records=MAX_RECORDS,
        ris_glob=config.get("step9a_ris_glob") or None,
        ris_path_override=config.get("step9a_update_ris") or None,
        iteration=config.get("step9a_iteration") or "step9a",
    )


def run_step9a(config: dict) -> dict:
    return run(config)


def main() -> int:
    _ = step9a_enrich_from_ris(
        out_dir=OUT_DIR,
        data_dir=DATA_DIR,
        sleep_s=SLEEP_S,
        cache_ttl_days=CACHE_TTL_DAYS,
        force_refresh_api=FORCE_REFRESH_API,
        max_records=MAX_RECORDS,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
