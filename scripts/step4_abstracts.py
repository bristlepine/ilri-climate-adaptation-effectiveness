#!/usr/bin/env python3
"""
step4_abstracts.py

Step 4 (Abstracts):
  - Reads outputs/step3/step3_benchmark_list.csv
  - Fetches abstracts for ALL DOIs.
  - Logic: Elsevier -> Semantic Scholar -> OpenAlex -> Crossref -> Unpaywall -> Landing Page (JSON-LD/Meta).
  - IMPROVEMENT: "Deep Scraping" (reads full landing page + JSON-LD) to fix failures.
  - IMPROVEMENT: Correct Summary Math (Total = Cached + Fresh Success + Failures).
"""

from __future__ import annotations

import os
import re
import json
import time
import html as _html
from typing import Dict, Optional, Tuple
from urllib.parse import quote
from html.parser import HTMLParser

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

import config as cfg

# --- CONFIG ---
DEFAULT_ELSEVIER_ARTICLE_URL = "https://api.elsevier.com/content/article/doi/"
DEFAULT_SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/"
DEFAULT_OPENALEX_WORKS_URL = "https://api.openalex.org/works/"
DEFAULT_CROSSREF_WORKS_URL = "https://api.crossref.org/works/"
DEFAULT_UNPAYWALL_URL = "https://api.unpaywall.org/v2/"

def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def _normalize_doi(x: str) -> str:
    d = str(x).strip().lower()
    for p in ("https://doi.org/", "http://doi.org/", "doi:"):
        if d.startswith(p):
            d = d[len(p):]
    return d

def _load_json(path: str, default):
    try:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default

def _save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

def _strip_tags(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s or "")
    s = _html.unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _find_first_str(o, wanted_keys: Tuple[str, ...]) -> Optional[str]:
    if isinstance(o, dict):
        for k, v in o.items():
            if k in wanted_keys and isinstance(v, str) and v.strip():
                return v.strip()
        for v in o.values():
            got = _find_first_str(v, wanted_keys)
            if got: return got
    elif isinstance(o, list):
        for it in o:
            got = _find_first_str(it, wanted_keys)
            if got: return got
    return None

def _http_get_json(url: str, headers: dict, params: Optional[dict] = None, tries: int = 3) -> Tuple[Optional[dict], Optional[str], int]:
    backoff = 1.0
    with requests.Session() as session:
        for _ in range(tries):
            try:
                r = session.get(url, headers=headers, params=params, timeout=30)
                if r.status_code == 200:
                    return r.json(), None, 200
                if r.status_code == 429:
                    time.sleep(backoff + 1)
                    backoff *= 1.5
                    continue
                if r.status_code >= 500:
                    time.sleep(backoff)
                    backoff *= 1.5
                    continue
                return None, f"HTTP {r.status_code}", r.status_code
            except Exception as ex:
                time.sleep(backoff)
    return None, "timeout_or_error", 0

# --- FETCHERS ---
def elsevier_article_fetch(api_key: str, inst_token: str, base_url: str, doi: str) -> dict:
    url = base_url.rstrip("/") + "/" + quote(doi, safe="")
    h = {"X-ELS-APIKey": api_key, "Accept": "application/json"}
    if inst_token: h["X-ELS-Insttoken"] = inst_token
    params = {"view": "META_ABS", "httpAccept": "application/json"}
    data, err, _ = _http_get_json(url, headers=h, params=params)
    if err: return {"error": err}
    
    title = _find_first_str(data, ("dc:title", "title"))
    abst = _find_first_str(data, ("dc:description", "ce:abstract", "abstract", "description"))
    return {"title": title, "abstract": _strip_tags(abst) if abst else None, "source": "elsevier_api", "status": 200}

def semantic_scholar_fetch(doi: str, email: str, base_url: str) -> dict:
    url = base_url.rstrip("/") + "/DOI:" + quote(doi, safe=":/")
    h = {"User-Agent": f"step4 ({email or 'test'})"}
    data, err, _ = _http_get_json(url, headers=h, params={"fields": "title,abstract"})
    if err: return {"error": err}
    return {"title": data.get("title"), "abstract": (data.get("abstract") or "").strip() or None, "source": "semantic_scholar", "status": 200}

def openalex_fetch(doi: str, email: str, base_url: str) -> dict:
    url = base_url.rstrip("/") + "/https://doi.org/" + quote(doi, safe=":/")
    h = {"User-Agent": f"step4 ({email or 'test'})"}
    data, err, _ = _http_get_json(url, headers=h, params={"mailto": email} if email else {})
    if err: return {"error": err}
    
    inv = (data or {}).get("abstract_inverted_index")
    abst = None
    if inv:
        pos_map = {p: word for word, pos_list in inv.items() for p in pos_list}
        if pos_map: abst = " ".join(pos_map[k] for k in sorted(pos_map.keys()))
    
    loc = (data or {}).get("primary_location") or {}
    return {"title": data.get("title"), "abstract": abst, "landing_url": loc.get("landing_page_url"), "source": "openalex", "status": 200}

def crossref_fetch(doi: str, email: str, base_url: str) -> dict:
    url = base_url.rstrip("/") + "/" + quote(doi)
    h = {"User-Agent": f"step4 ({email or 'test'})"}
    data, err, _ = _http_get_json(url, headers=h, params={"mailto": email} if email else {})
    if err: return {"error": err}
    msg = (data or {}).get("message") or {}
    return {"title": (msg.get("title") or [None])[0], "abstract": _strip_tags(msg.get("abstract")), "source": "crossref", "status": 200}

def unpaywall_fetch(doi: str, email: str, base_url: str) -> dict:
    if not email: return {"error": "no_email"}
    url = base_url.rstrip("/") + "/" + quote(doi)
    h = {"User-Agent": f"step4 ({email})"}
    data, err, _ = _http_get_json(url, headers=h, params={"email": email})
    if err: return {"error": err}
    best = (data or {}).get("best_oa_location") or {}
    return {"landing_url": best.get("url"), "source": "unpaywall", "status": 200}

# --- LANDING PAGE SCRAPER (Improved) ---
class MetaParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.metas = {}
    def handle_starttag(self, tag, attrs):
        if tag.lower() == "meta":
            d = {k.lower(): v for k, v in attrs}
            name = d.get("name") or d.get("property")
            content = d.get("content")
            if name and content: self.metas[name.lower()] = content

def fetch_landing_abstract(url: str) -> dict:
    try:
        h = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Chrome/91.0.4472.114 Safari/537.36"}
        r = requests.get(url, headers=h, timeout=20)
        if r.status_code != 200: return {"error": f"HTTP {r.status_code}"}
        
        raw = r.text
        # 1. JSON-LD
        for match in re.finditer(r'<script type="application/ld\+json">(.*?)</script>', raw, re.DOTALL):
            try:
                j = json.loads(match.group(1))
                items = j if isinstance(j, list) else [j]
                for item in items:
                    for k in ["description", "abstract"]:
                        if k in item and len(item[k]) > 50:
                            return {"abstract": _strip_tags(item[k]), "source": "landing_json_ld", "status": 200}
            except: continue

        # 2. Meta Tags
        p = MetaParser()
        p.feed(raw[:200000])
        for k in ["citation_abstract", "dc.description", "og:description", "description"]:
            if k in p.metas and len(p.metas[k]) > 50:
                return {"abstract": _strip_tags(p.metas[k]), "source": "landing_meta", "status": 200}
        return {"error": "no_abstract_found"}
    except Exception as e:
        return {"error": str(e)}

# --- MAIN ---
def step4_fetch_abstracts(config: dict) -> dict:
    load_dotenv()
    out_dir = config.get("out_dir") or getattr(cfg, "out_dir", "outputs")
    
    api_key = os.getenv("SCOPUS_API_KEY")
    inst_token = os.getenv("SCOPUS_INST_TOKEN")
    email = os.getenv("CONTACT_EMAIL")
    
    in_csv = os.path.join(out_dir, "step3", "step3_benchmark_list.csv")
    step4_dir = os.path.join(out_dir, "step4")
    out_csv = os.path.join(step4_dir, "step4_abstracts.csv")
    cache_path = os.path.join(step4_dir, "step4_abstracts_cache.json")
    out_summary = os.path.join(step4_dir, "step4_abstracts_summary.json")
    os.makedirs(step4_dir, exist_ok=True)

    if not os.path.exists(in_csv): return {"status": "error"}
    
    df = pd.read_csv(in_csv)
    if "doi" not in df.columns: return {"status": "error"}
    
    df["_doi_norm"] = df["doi"].apply(lambda x: _normalize_doi(x) if pd.notna(x) else None)
    df = df[df["_doi_norm"].notna()].drop_duplicates(subset=["_doi_norm"]).copy()
    
    total = len(df)
    print(f"\n[Step 4] Processing {total} unique DOIs...")

    cache = _load_json(cache_path, default={})
    chunk = []
    
    if os.path.exists(out_csv): os.remove(out_csv)
    wrote_header = False

    pbar = tqdm(total=total, desc="Fetching Abstracts", unit="paper")
    
    # Track stats for EVERYTHING (Cached + Fresh)
    final_stats = {"ok": 0, "failed": 0, "sources": {}}

    failed_records = []

    for _, row in df.iterrows():
        doi = row["_doi_norm"]
        
        # Check Cache (Successes only)
        if doi in cache and cache[doi].get("fetch_status") == "ok":
            rec = cache[doi]
            # Ensure stats count cached items
            final_stats["ok"] += 1
            src = rec.get("source", "unknown")
            final_stats["sources"][src] = final_stats["sources"].get(src, 0) + 1
            
            chunk.append(rec)
            pbar.update(1)
            if len(chunk) >= 50:
                pd.DataFrame(chunk).to_csv(out_csv, mode="a", header=not wrote_header, index=False)
                wrote_header = True
                chunk = []
            continue

        # Fresh Fetch
        rec = {
            "doi": doi,
            "title": row.get("title"),
            "type": row.get("type"),
            "identified_via": row.get("identified_via"),
            "abstract": None,
            "source": None,
            "fetch_status": "no_abstract"
        }

        # Chain of Responsibility
        if api_key:
            pbar.set_postfix_str("Elsevier...")
            res = elsevier_article_fetch(api_key, inst_token, DEFAULT_ELSEVIER_ARTICLE_URL, doi)
            if res.get("abstract"): rec.update(res); rec["fetch_status"] = "ok"

        if rec["fetch_status"] != "ok":
            pbar.set_postfix_str("SemanticScholar...")
            res = semantic_scholar_fetch(doi, email, DEFAULT_SEMANTIC_SCHOLAR_URL)
            if res.get("abstract"): rec.update(res); rec["fetch_status"] = "ok"

        if rec["fetch_status"] != "ok":
            pbar.set_postfix_str("OpenAlex...")
            res = openalex_fetch(doi, email, DEFAULT_OPENALEX_WORKS_URL)
            if res.get("abstract"): rec.update(res); rec["fetch_status"] = "ok"
            rec["landing_url"] = res.get("landing_url") # Keep for fallback

        if rec["fetch_status"] != "ok":
            pbar.set_postfix_str("Crossref...")
            res = crossref_fetch(doi, email, DEFAULT_CROSSREF_WORKS_URL)
            if res.get("abstract"): rec.update(res); rec["fetch_status"] = "ok"

        if rec["fetch_status"] != "ok" and email:
            pbar.set_postfix_str("Landing Page...")
            # Try OpenAlex landing URL first, then Unpaywall
            landing = rec.get("landing_url")
            if not landing:
                upw = unpaywall_fetch(doi, email, DEFAULT_UNPAYWALL_URL)
                landing = upw.get("landing_url")
            
            if not landing: landing = f"https://doi.org/{doi}"

            res = fetch_landing_abstract(landing)
            if res.get("abstract"): rec.update(res); rec["fetch_status"] = "ok"

        # Update Stats & Cache
        if rec["fetch_status"] == "ok":
            cache[doi] = rec
            final_stats["ok"] += 1
            src = rec.get("source", "unknown")
            final_stats["sources"][src] = final_stats["sources"].get(src, 0) + 1
        else:
            final_stats["failed"] += 1
            failed_records.append({
                "doi": rec["doi"],
                "title": rec.get("title"),
                "type": rec.get("type"),
                "identified_via": rec.get("identified_via"),
                "last_source_attempted": rec.get("source"),
                "fetch_status": rec.get("fetch_status")
            })

        chunk.append(rec)
        pbar.update(1)
        
        if len(chunk) >= 50:
            pd.DataFrame(chunk).to_csv(out_csv, mode="a", header=not wrote_header, index=False)
            wrote_header = True
            chunk = []
            _save_json(cache_path, cache)

    pbar.close()
    if chunk:
        pd.DataFrame(chunk).to_csv(out_csv, mode="a", header=not wrote_header, index=False)
    _save_json(cache_path, cache)

    # Save Correct Summary
    summary = {
        "total_rows": total,
        "ok": final_stats["ok"],
        "failed": final_stats["failed"],
        "ok_by_source": final_stats["sources"],
        "timestamp_utc": _now_utc()
    }
    _save_json(out_summary, summary)

    print(f"\n[Step 4] Done. OK: {final_stats['ok']} | Failed: {final_stats['failed']}")
    print(f"[Step 4] Sources: {json.dumps(final_stats['sources'])}")

    missing_csv = os.path.join(step4_dir, "step4_abstracts_missing.csv")

    if failed_records:
        pd.DataFrame(failed_records).to_csv(missing_csv, index=False)
    
    return {"status": "ok", "path": out_csv}

# Aliases
def run(config): return step4_fetch_abstracts(config)
def run_step4(config): return step4_fetch_abstracts(config)
def main(config): return step4_fetch_abstracts(config)

if __name__ == "__main__":
    step4_fetch_abstracts({})