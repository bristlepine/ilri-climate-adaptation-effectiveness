#!/usr/bin/env python3
"""
step4b_gsch_abstracts.py

Step 4b: Retrieve abstracts for Google Scholar records (step2b output).

Google Scholar exports from step2b lack abstracts. This script:
  1. Reads step2b_google_scholar_pending.csv
  2. Attempts to fetch abstracts via DOI (if available)
  3. Falls back to title search via CrossRef, Semantic Scholar, OpenAlex
  4. Outputs step4b_google_scholar_with_abstracts.csv

The output can be merged back with other step2b net-new records before step12.

Usage:
  conda run -n ilri01 python scripts/step4b_gsch_abstracts.py
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import requests
from tqdm import tqdm

from utils import normalize_doi, request_with_retries, utc_now_iso


# =============================================================================
# API endpoints for abstract retrieval
# =============================================================================

CROSSREF_WORKS_URL = "https://api.crossref.org/works/"
SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/"
OPENALEX_WORKS_URL = "https://api.openalex.org/works/"

SLEEP_S = 0.2  # rate limiting


def fetch_via_doi(doi: str) -> Optional[Dict[str, Any]]:
    """Fetch title + abstract via DOI (CrossRef + Semantic Scholar)."""
    if not doi:
        return None

    # Try CrossRef
    try:
        url = f"{CROSSREF_WORKS_URL}{doi}"
        r = request_with_retries(url, timeout=5)
        if r and r.status_code == 200:
            data = r.json().get("message", {})
            return {
                "abstract": data.get("abstract", ""),
                "title": data.get("title", ""),
                "source": "crossref",
            }
    except Exception:
        pass

    # Try Semantic Scholar with DOI
    try:
        url = f"{SEMANTIC_SCHOLAR_URL}DOI:{doi}"
        r = request_with_retries(url, timeout=5)
        if r and r.status_code == 200:
            data = r.json()
            return {
                "abstract": data.get("abstract", ""),
                "title": data.get("title", ""),
                "source": "semantic_scholar",
            }
    except Exception:
        pass

    return None


def fetch_via_title(title: str) -> Optional[Dict[str, Any]]:
    """Fetch abstract via title search (CrossRef, Semantic Scholar, OpenAlex)."""
    if not title or len(title) < 10:
        return None

    # Try CrossRef title search
    try:
        url = CROSSREF_WORKS_URL
        params = {"query": title, "rows": 1}
        r = request_with_retries(url, params=params, timeout=5)
        if r and r.status_code == 200:
            items = r.json().get("message", {}).get("items", [])
            if items:
                item = items[0]
                # Simple title match: if similarity is high enough, return abstract
                fetched_title = item.get("title", [title])[0] if isinstance(item.get("title"), list) else item.get("title", "")
                if fetched_title.lower()[:30] == title.lower()[:30]:  # rough match on first 30 chars
                    return {
                        "abstract": item.get("abstract", ""),
                        "title": fetched_title,
                        "source": "crossref_title_search",
                    }
    except Exception:
        pass

    time.sleep(SLEEP_S)

    # Try Semantic Scholar title search
    try:
        url = f"{SEMANTIC_SCHOLAR_URL}search"
        params = {"query": title, "limit": 1}
        r = request_with_retries(url, params=params, timeout=5)
        if r and r.status_code == 200:
            data = r.json().get("data", [])
            if data:
                paper = data[0]
                fetched_title = paper.get("title", title)
                if fetched_title.lower()[:30] == title.lower()[:30]:
                    return {
                        "abstract": paper.get("abstract", ""),
                        "title": fetched_title,
                        "source": "semantic_scholar_title_search",
                    }
    except Exception:
        pass

    time.sleep(SLEEP_S)

    return None


def retrieve_abstract(row: Dict) -> Dict:
    """
    Attempt to retrieve abstract for a Google Scholar record.
    Returns row with added 'abstract', 'abstract_source', 'abstract_status'.
    """
    row = row.copy()
    title = str(row.get("title", "")).strip()
    doi = normalize_doi(row.get("doi", ""))

    abstract_source = ""
    abstract_status = "not_found"

    # 1. If already has abstract, mark it
    if row.get("abstract", "").strip():
        abstract_status = "already_present"
        abstract_source = "google_scholar_export"
    # 2. Try DOI-based retrieval
    elif doi:
        result = fetch_via_doi(doi)
        if result and result.get("abstract"):
            row["abstract"] = result["abstract"]
            abstract_source = result["source"]
            abstract_status = "retrieved_via_doi"
            time.sleep(SLEEP_S)
        # 3. Fall back to title search
        elif title:
            result = fetch_via_title(title)
            if result and result.get("abstract"):
                row["abstract"] = result["abstract"]
                abstract_source = result["source"]
                abstract_status = "retrieved_via_title"
                time.sleep(SLEEP_S)
    # 4. Title-only search
    elif title:
        result = fetch_via_title(title)
        if result and result.get("abstract"):
            row["abstract"] = result["abstract"]
            abstract_source = result["source"]
            abstract_status = "retrieved_via_title"
            time.sleep(SLEEP_S)

    row["abstract_source"] = abstract_source
    row["abstract_status"] = abstract_status
    return row


# =============================================================================
# Main
# =============================================================================

def run(config: dict) -> dict:
    c = config or {}
    root = Path(__file__).parent
    out_root = Path(str(c.get("out_dir", "")) or root / "outputs")
    step2b_dir = out_root / "step2b"
    step4b_dir = out_root / "step4b"
    step4b_dir.mkdir(parents=True, exist_ok=True)

    # Input: Google Scholar records from step2b
    gsch_input = step2b_dir / "step2b_google_scholar_pending.csv"
    if not gsch_input.exists():
        print(f"[step4b] No Google Scholar records to process: {gsch_input}")
        return {"error": "input file not found"}

    print(f"[step4b] Loading Google Scholar records: {gsch_input}")
    df = pd.read_csv(gsch_input, dtype=str).fillna("")
    print(f"[step4b] Loaded {len(df):,} records")

    # Retrieve abstracts
    print(f"[step4b] Retrieving abstracts...")
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        rec = retrieve_abstract(row.to_dict())
        records.append(rec)

    df_out = pd.DataFrame(records)

    # Summary
    status_counts = df_out["abstract_status"].value_counts().to_dict()
    source_counts = df_out["abstract_source"].value_counts().to_dict()

    # Write outputs
    out_csv = step4b_dir / "step4b_google_scholar_with_abstracts.csv"
    summary_json = step4b_dir / "step4b_summary.json"

    df_out.to_csv(out_csv, index=False)

    summary = {
        "total_records": int(len(df_out)),
        "abstract_status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "abstract_source_counts": {str(k): int(v) for k, v in source_counts.items()},
        "output_csv": str(out_csv),
        "timestamp_utc": utc_now_iso(),
    }

    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[step4b] ── Results ─────────────────────────────────")
    for status, count in status_counts.items():
        print(f"[step4b]   {status}: {count:,}")
    print(f"[step4b]   Total: {len(df_out):,}")
    print(f"[step4b]   Written to: {out_csv}")
    print(f"[step4b] ─────────────────────────────────────────────")
    return summary


if __name__ == "__main__":
    import config as cfg
    run(cfg.config if hasattr(cfg, "config") else {})
