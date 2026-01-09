#!/usr/bin/env python3
"""
step3_benchmark.py

Step 3 (benchmark prep + DOI enrichment):
  - Load scripts/Benchmark List - List.csv
  - INTELLIGENTLY EXTRACT TITLES from "Study" column (strip Authors/Year)
  - Parse DOIs from text (best-effort)
  - If DOI missing, enrich via Crossref/OpenAlex/SemanticScholar
  - STRICT MODE: Any DOI flagged as "Needs Review" is discarded automatically.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from difflib import SequenceMatcher
from typing import Optional

# Ensure tqdm is installed: pip install tqdm
from tqdm import tqdm
import pandas as pd
import requests

import config as cfg

# --- CONFIGURATION & THRESHOLDS ---
DOI_REGEX = re.compile(r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", re.IGNORECASE)
YEAR_SPLIT_REGEX = re.compile(r"[\s\.,]\(\d{4}[a-z]?\)[\.,]?\s*")

CROSSREF_WORKS_URL = "https://api.crossref.org/works"
OPENALEX_WORKS_URL = "https://api.openalex.org/works"
SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

# THRESHOLDS
DOI_AUTO_ACCEPT_SCORE = 0.90     # Auto-accept if > 90%
DOI_MIN_CANDIDATE_SCORE = 0.60   # Candidate threshold

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

def _extract_clean_title(full_text: str) -> str:
    if not isinstance(full_text, str): return ""
    parts = YEAR_SPLIT_REGEX.split(full_text, maxsplit=1)
    cleaned = parts[1].strip() if len(parts) > 1 and len(parts[1].strip()) > 10 else full_text.strip()
    return re.sub(r"^[\W_]+", "", cleaned)

def _normalize_doi(x: str) -> str:
    d = str(x).strip().lower()
    for p in ("https://doi.org/", "http://doi.org/", "doi:"):
        if d.startswith(p): d = d[len(p):]
    return f"https://doi.org/{d}"

def _doi_from_text(s: str) -> Optional[str]:
    if not isinstance(s, str): return None
    m = DOI_REGEX.search(s)
    return _normalize_doi(m.group(1)) if m else None

def _title_norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _title_score(a: str, b: str) -> float:
    a2, b2 = _title_norm(a), _title_norm(b)
    if not a2 or not b2: return 0.0
    base = SequenceMatcher(None, a2, b2).ratio()
    if len(a2) > 15 and len(b2) > 15:
        if a2 in b2 or b2 in a2: return 1.0
    return base

def _http_get_json_simple(url: str, headers=None, params=None, timeout=60) -> Optional[dict]:
    try:
        r = requests.get(url, headers=headers or {}, params=params or {}, timeout=timeout)
        if r.status_code != 200: return None
        return r.json()
    except Exception:
        return None

# --- API HELPERS ---
def _crossref_candidates(title: str, mailto: str | None = None, rows: int = 5) -> list[dict]:
    params = {"query.bibliographic": title, "rows": rows}
    if mailto: params["mailto"] = mailto
    data = _http_get_json_simple(CROSSREF_WORKS_URL, params=params)
    items = ((data or {}).get("message") or {}).get("items") or []
    out = []
    for it in items:
        t = it.get("title")[0] if it.get("title") else None
        year = it.get("issued", {}).get("date-parts", [[None]])[0][0]
        doi = it.get("DOI")
        if doi: out.append({"source": "crossref", "doi": _normalize_doi(doi), "title": t, "year": year})
    return out

def _openalex_candidates(title: str, mailto: str | None = None, per_page: int = 5) -> list[dict]:
    params = {"search": title, "per-page": per_page}
    if mailto: params["mailto"] = mailto
    data = _http_get_json_simple(OPENALEX_WORKS_URL, params=params)
    results = (data or {}).get("results") or []
    out = []
    for it in results:
        doi = it.get("doi")
        if doi: out.append({"source": "openalex", "doi": _normalize_doi(doi), "title": it.get("title"), "year": it.get("publication_year")})
    return out

def _semantic_candidates(title: str, contact_email: str | None = None, limit: int = 5) -> list[dict]:
    params = {"query": title, "limit": limit, "fields": "title,year,externalIds"}
    headers = {"Accept": "application/json", "User-Agent": f"step3-enrichment ({contact_email or 'no-email'})"}
    data = _http_get_json_simple(SEMANTIC_SCHOLAR_SEARCH_URL, headers=headers, params=params)
    results = (data or {}).get("data") or []
    out = []
    for it in results:
        doi = it.get("externalIds", {}).get("DOI")
        if doi: out.append({"source": "semantic_scholar", "doi": _normalize_doi(doi), "title": it.get("title"), "year": it.get("year")})
    return out

def _best_doi_for_title(cleaned_title: str, year: int | None, contact_email: str | None, mailto: str | None) -> dict:
    cands = []
    cands += _crossref_candidates(cleaned_title, mailto=mailto)
    cands += _openalex_candidates(cleaned_title, mailto=mailto)
    cands += _semantic_candidates(cleaned_title, contact_email=contact_email)

    doi_counts = {}
    for c in cands:
        d = c.get("doi")
        if d: doi_counts[d] = doi_counts.get(d, 0) + 1

    scored = []
    for c in cands:
        if not c.get("doi") or not c.get("title"): continue
        base_score = _title_score(cleaned_title, str(c.get("title") or ""))
        if doi_counts.get(c["doi"], 0) > 1: base_score += 0.05
        if year and c.get("year"):
            try:
                if abs(int(year) - int(c["year"])) > 1: base_score -= 0.20
            except: pass
        scored.append({**c, "score": min(1.0, float(base_score))})

    scored.sort(key=lambda x: x["score"], reverse=True)
    best = scored[0] if scored and scored[0]["score"] >= DOI_MIN_CANDIDATE_SCORE else None
    
    if not best:
        return {"doi": None, "doi_source": None, "doi_score": 0.0, "matched_title": None, "needs_review": False}

    return {
        "doi": best["doi"],
        "doi_source": best["source"],
        "doi_score": best["score"],
        "matched_title": best["title"],
        "needs_review": bool(best["score"] < DOI_AUTO_ACCEPT_SCORE)
    }

def _cache_key_for_title(title: str) -> str:
    return hashlib.sha256(_title_norm(title).encode("utf-8")).hexdigest()

# --- MAIN ---
def step3_build_benchmark_list(config: dict) -> dict:
    out_dir = config.get("out_dir") or getattr(cfg, "out_dir", "outputs")
    benchmark_csv = config.get("benchmark_csv") or getattr(cfg, "benchmark_csv", None)
    if not benchmark_csv: raise SystemExit("Missing benchmark_csv configuration.")

    step3_dir = os.path.join(out_dir, "step3")
    out_csv = os.path.join(step3_dir, "step3_benchmark_list.csv")
    out_enriched_only_csv = os.path.join(step3_dir, "step3_benchmark_list.enriched_only.csv")
    out_summary = os.path.join(step3_dir, "step3_benchmark_list.summary.json")
    cache_path = os.path.join(step3_dir, "step3_benchmark_doi_enrichment_cache.json")
    os.makedirs(step3_dir, exist_ok=True)

    print(f"\n[Step 3] Reading Benchmark: {benchmark_csv}")
    bench = pd.read_csv(benchmark_csv)

    colmap = {c.strip(): c for c in bench.columns}
    study_col = colmap.get("Study", "Study")
    
    bench["_doi_parsed"] = bench[study_col].apply(_doi_from_text)
    raw_studies = bench[study_col].fillna("").astype(str)
    cleaned_titles = raw_studies.apply(_extract_clean_title)

    # --- INITIAL DOI STATS (pre-enrichment) ---
    initial_doi_mask = bench["_doi_parsed"].notna()
    initial_doi_count = int(initial_doi_mask.sum())
    # ----------------------------------------

    out = pd.DataFrame({
        "original_study": raw_studies,
        "clean_title": cleaned_titles,
        "type": bench[colmap.get("Type", "Type")],
        "doi": bench["_doi_parsed"],
        "identified_via": bench[colmap.get("Identified Via", "Identified Via")]
    })

    contact_email = os.getenv("CONTACT_EMAIL")
    cache = _load_json(cache_path, default={})
    
    print(f"[Step 3] Processing {len(out)} rows (Threshold: {DOI_AUTO_ACCEPT_SCORE*100:.0f}%)...")
    pbar = tqdm(total=len(out), desc="Enriching DOIs", unit="row")

    results_meta = []
    for i, row in out.iterrows():
        existing = row.get("doi")
        if isinstance(existing, str) and existing.strip():
            results_meta.append({"source": "given", "score": 1.0, "review": False, "match": None})
            out.at[i, "doi"] = _normalize_doi(existing)
            pbar.update(1)
            continue

        clean_t = row["clean_title"]
        if not clean_t or len(clean_t) < 5:
            results_meta.append({"source": None, "score": 0.0, "review": False, "match": None})
            pbar.update(1)
            continue

        ck = _cache_key_for_title(clean_t)
        if ck not in cache:
            pbar.set_postfix_str(f"Search: {clean_t[:20]}...")
            got = _best_doi_for_title(clean_t, year=None, contact_email=contact_email, mailto=contact_email)
            cache[ck] = got
            if i % 25 == 0: _save_json(cache_path, cache)

        got = cache[ck]
        if got.get("doi"):
            out.at[i, "doi"] = _normalize_doi(got["doi"])
            
        results_meta.append({
            "source": got.get("doi_source"),
            "score": float(got.get("doi_score") or 0.0),
            "review": bool(got.get("needs_review")),
            "match": got.get("matched_title")
        })
        pbar.update(1)

    pbar.close()
    _save_json(cache_path, cache)

    out["doi_source"] = [x["source"] for x in results_meta]
    out["doi_score"] = [x["score"] for x in results_meta]
    out["doi_needs_review"] = [x["review"] for x in results_meta]
    out["doi_matched_title"] = [x["match"] for x in results_meta]

    # --- AUTOMATIC REJECTION LOGIC ---
    print(f"[Step 3] Strict Mode: Rejecting {out['doi_needs_review'].sum()} entries flagged for review.")
    
    # Set DOI to None where review is needed
    out.loc[out["doi_needs_review"] == True, "doi"] = None
    out.loc[out["doi_needs_review"] == True, "doi_source"] = "rejected_low_score"
    # ---------------------------------
    # --- DOI FLOW FLAGS (POST–STRICT MODE) ---
    out["_has_doi_final"] = out["doi"].notna()

    # initial DOI presence comes from pre-enrichment parsing
    out["_has_doi_initial"] = bench["_doi_parsed"].notna().values

    out["_doi_given"] = out["doi_source"] == "given"
    out["_doi_enriched"] = out["_has_doi_final"] & (out["doi_source"] != "given")
    out["_doi_rejected"] = out["doi_source"] == "rejected_low_score"
    # ----------------------------------------


    out = out.rename(columns={"original_study": "title"})
    final_cols = [
        "title",
        "type",
        "doi",
        "identified_via",
        "doi_source",
        "doi_score",
        "doi_needs_review",
        "doi_matched_title"
    ]    

    out[final_cols].to_csv(out_csv, index=False)
    enriched_only = out[(out["doi_source"].fillna("") != "given") & (out["doi"].notna())].copy()
    enriched_only[final_cols].to_csv(out_enriched_only_csv, index=False)
    type_counts = out["type"].value_counts(dropna=False).to_dict()
    identified_via_counts = out["identified_via"].value_counts(dropna=False).to_dict()
    doi_by_type_deep = (
    out
    .groupby("type", dropna=False)
    .agg(
            total=("title", "count"),
            initial_with_doi=("_has_doi_initial", "sum"),
            doi_given=("_doi_given", "sum"),
            doi_enriched=("_doi_enriched", "sum"),
            doi_rejected=("_doi_rejected", "sum"),
            final_with_doi=("_has_doi_final", "sum"),
            final_missing=("doi", lambda x: x.isna().sum()),
        )
        .reset_index()
        .to_dict(orient="records")
    )

    doi_by_identified_via_deep = (
    out
    .groupby("identified_via", dropna=False)
    .agg(
            total=("title", "count"),
            initial_with_doi=("_has_doi_initial", "sum"),
            doi_given=("_doi_given", "sum"),
            doi_enriched=("_doi_enriched", "sum"),
            doi_rejected=("_doi_rejected", "sum"),
            final_with_doi=("_has_doi_final", "sum"),
            final_missing=("doi", lambda x: x.isna().sum()),
        )
        .reset_index()
        .to_dict(orient="records")
    )

    identified_via_doi_coverage = (
        out
        .assign(has_final_doi=out["doi"].notna())
        .groupby("identified_via", dropna=False)["has_final_doi"]
        .agg(total="count", with_doi="sum")
        .reset_index()
        .assign(
            pct_with_doi=lambda d: (d["with_doi"] / d["total"]).round(3)
        )
        .to_dict(orient="records")
    )

    summary = {
        "total": len(out),
        "initial_with_doi": initial_doi_count,
        "doi_enriched": len(enriched_only),
        "doi_needs_review_but_rejected": int(out["doi_needs_review"].sum()),
        "missing": int(out["doi"].isna().sum()),
        "by_type": type_counts,
        "by_identified_via": identified_via_counts,
        "identified_via_doi_coverage": identified_via_doi_coverage,
        "doi_flow_by_type": doi_by_type_deep,
        "doi_flow_by_identified_via": doi_by_identified_via_deep
    }
    _save_json(out_summary, summary)

    print(f"\n[Step 3] Done. Enriched (Clean): {summary['doi_enriched']}")
    print(f"[Step 3] Missing/Rejected: {summary['missing']}")
    return {"status": "ok", "path": out_csv}

# Aliases
def run(config): return step3_build_benchmark_list(config)
def run_step3(config): return step3_build_benchmark_list(config)
def main(config): return step3_build_benchmark_list(config)

if __name__ == "__main__":
    step3_build_benchmark_list({})