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

from utils import normalize_doi, utc_now_iso


# =============================================================================
# API endpoints for abstract retrieval
# =============================================================================

CROSSREF_WORKS_URL = "https://api.crossref.org/works/"
OPENALEX_WORKS_URL = "https://api.openalex.org/works"

HEADERS = {"User-Agent": "research-abstract-retrieval/1.0 (mailto:zarrar85@gmail.com)"}
SLEEP_S = 0.15


def _reconstruct_openalex_abstract(inverted_index: dict) -> str:
    """Reconstruct abstract from OpenAlex inverted index format."""
    if not inverted_index:
        return ""
    positions: dict = {}
    for word, pos_list in inverted_index.items():
        for pos in pos_list:
            positions[pos] = word
    return " ".join(positions[k] for k in sorted(positions.keys()))


def fetch_via_doi(doi: str) -> Optional[Dict[str, Any]]:
    """Fetch abstract via DOI: OpenAlex first, CrossRef fallback."""
    if not doi:
        return None

    # OpenAlex DOI lookup
    try:
        r = requests.get(f"{OPENALEX_WORKS_URL}/doi:{doi}", headers=HEADERS, timeout=15)
        if r.status_code == 200:
            data = r.json()
            inv = data.get("abstract_inverted_index")
            if inv:
                return {"abstract": _reconstruct_openalex_abstract(inv), "source": "openalex_doi"}
    except Exception:
        pass

    time.sleep(SLEEP_S)

    # CrossRef fallback
    try:
        r = requests.get(f"{CROSSREF_WORKS_URL}{doi}", headers=HEADERS, timeout=15)
        if r.status_code == 200:
            ab = r.json().get("message", {}).get("abstract", "")
            if ab:
                return {"abstract": re.sub(r"<[^>]+>", "", ab).strip(), "source": "crossref_doi"}
    except Exception:
        pass

    return None


def fetch_via_title(title: str) -> Optional[Dict[str, Any]]:
    """Fetch abstract via title search: OpenAlex first, CrossRef fallback."""
    if not title or len(title) < 10:
        return None

    # OpenAlex title search
    try:
        r = requests.get(OPENALEX_WORKS_URL,
            params={"search": title, "per-page": 1}, headers=HEADERS, timeout=15)
        if r.status_code == 200:
            results = r.json().get("results", [])
            if results:
                fetched = results[0].get("title", "")
                if fetched and title.lower()[:40] in fetched.lower():
                    inv = results[0].get("abstract_inverted_index")
                    if inv:
                        return {"abstract": _reconstruct_openalex_abstract(inv), "source": "openalex_title"}
    except Exception:
        pass

    time.sleep(SLEEP_S)

    # CrossRef title search fallback
    try:
        r = requests.get(CROSSREF_WORKS_URL,
            params={"query": title, "rows": 1}, headers=HEADERS, timeout=15)
        if r.status_code == 200:
            items = r.json().get("message", {}).get("items", [])
            if items:
                fetched = items[0].get("title", [""])[0] if isinstance(items[0].get("title"), list) else ""
                if fetched and title.lower()[:40] in fetched.lower():
                    ab = items[0].get("abstract", "")
                    if ab:
                        return {"abstract": re.sub(r"<[^>]+>", "", ab).strip(), "source": "crossref_title"}
    except Exception:
        pass

    return None


def retrieve_abstract(row: Dict) -> Dict:
    """Attempt abstract retrieval for a record. Returns row with status fields added."""
    row = row.copy()
    title = str(row.get("title", "")).strip()
    doi = normalize_doi(row.get("doi", ""))

    abstract_source = ""
    abstract_status = "not_found"

    if row.get("abstract", "").strip():
        abstract_status = "already_present"
        abstract_source = "existing"
    elif doi:
        result = fetch_via_doi(doi)
        if result and result.get("abstract"):
            row["abstract"] = result["abstract"]
            abstract_source = result["source"]
            abstract_status = "retrieved_via_doi"
        elif title:
            time.sleep(SLEEP_S)
            result = fetch_via_title(title)
            if result and result.get("abstract"):
                row["abstract"] = result["abstract"]
                abstract_source = result["source"]
                abstract_status = "retrieved_via_title"
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

    # Input: all net-new records from step2b
    net_new_csv = step2b_dir / "step2b_net_new.csv"
    if not net_new_csv.exists():
        print(f"[step4b] Net-new records not found: {net_new_csv} — run step2b first.")
        return {"error": "input file not found"}

    df = pd.read_csv(net_new_csv, dtype=str).fillna("")
    missing_mask = df["abstract"].str.strip() == ""
    missing = df[missing_mask].copy()

    print(f"[step4b] Net-new records: {len(df):,} total, {missing_mask.sum():,} missing abstracts")

    if missing.empty:
        print("[step4b] No missing abstracts — nothing to do.")
        return {"total_records": len(df), "patched": 0}

    # Skip IDB factsheets — confirmed no abstract section in source PDFs
    to_try = missing[missing["source_db"] != "IDB"].copy()
    skipped_idb = missing_mask.sum() - len(to_try)
    print(f"[step4b] Skipping {skipped_idb} IDB factsheets (no abstract section in source)")
    print(f"[step4b] Attempting retrieval for {len(to_try):,} records via CrossRef + Semantic Scholar...")

    records = []
    for _, row in tqdm(to_try.iterrows(), total=len(to_try)):
        rec = retrieve_abstract(row.to_dict())
        records.append(rec)

    df_patched = pd.DataFrame(records)
    status_counts = df_patched["abstract_status"].value_counts().to_dict()

    # Patch abstracts back into the full net-new dataframe
    patched_count = 0
    for _, row in df_patched.iterrows():
        if row.get("abstract", "").strip() and row.get("abstract_status") != "already_present":
            mask = (df["record_key"] == row["record_key"])
            if mask.any():
                df.loc[mask, "abstract"] = row["abstract"]
                patched_count += 1

    # Save updated net-new CSV in place
    df.to_csv(net_new_csv, index=False)

    # Save detailed results for audit
    summary_json = step4b_dir / "step4b_summary.json"
    df_patched.to_csv(step4b_dir / "step4b_retrieval_log.csv", index=False)

    still_missing = (df["abstract"].str.strip() == "").sum()
    summary = {
        "total_net_new": int(len(df)),
        "missing_before": int(missing_mask.sum()),
        "skipped_idb": int(skipped_idb),
        "attempted": int(len(to_try)),
        "patched": int(patched_count),
        "still_missing": int(still_missing),
        "abstract_status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "timestamp_utc": utc_now_iso(),
    }

    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[step4b] ── Results ─────────────────────────────────")
    for status, count in status_counts.items():
        print(f"[step4b]   {status}: {count:,}")
    print(f"[step4b]   Patched into net-new CSV: {patched_count:,}")
    print(f"[step4b]   Still missing after retrieval: {still_missing:,}")
    print(f"[step4b] ─────────────────────────────────────────────")
    return summary


if __name__ == "__main__":
    import config as cfg
    run(cfg.config if hasattr(cfg, "config") else {})
