#!/usr/bin/env python3
"""
Generate missing_papers.csv for get_missing.py (and Jenn).

Sources:
  - outputs/step14/step14_no_fulltext.csv   authoritative list of papers needing retrieval
  - outputs/step13/step13_manual.csv        adds scopus_id + pub metadata
  - missing_papers/missing_papers.csv       preserves manually_retrieved marks from prior run

Outputs:
  - missing_papers/missing_papers.csv              master (read by get_missing.py)
  - missing_papers/versions/missing_papers_vNN_YYYY-MM-DD.csv   versioned snapshot

Run from project root:
  python missing_papers/generate_missing_papers.py
"""

from __future__ import annotations
import re
from datetime import date
from pathlib import Path

import pandas as pd

HERE      = Path(__file__).resolve().parent
ROOT      = HERE.parent
VERSIONS  = HERE / "versions"
MASTER    = HERE / "missing_papers.csv"
S14_NOFT  = ROOT / "scripts/outputs/step14/step14_no_fulltext.csv"
S13_MAN   = ROOT / "scripts/outputs/step13/step13_manual.csv"

OUT_COLS  = ["dedupe_key", "doi", "scopus_id", "title", "year", "pub", "manually_retrieved"]


def next_version() -> str:
    VERSIONS.mkdir(exist_ok=True)
    existing = sorted(VERSIONS.glob("missing_papers_v*.csv"))
    if not existing:
        return "v01"
    last = existing[-1].stem          # e.g. missing_papers_v03_2026-04-10
    m = re.search(r"_v(\d+)_", last)
    n = int(m.group(1)) + 1 if m else 1
    return f"v{n:02d}"


def main():
    # Load the authoritative no-fulltext list
    s14 = pd.read_csv(S14_NOFT, dtype=str).fillna("")
    print(f"step14_no_fulltext: {len(s14):,} records")

    # Load step13_manual for scopus_id + pub metadata
    s13 = pd.read_csv(S13_MAN, dtype=str).fillna("")
    s13 = s13[["dedupe_key", "scopus_id", "pub"]].drop_duplicates("dedupe_key")

    # Merge
    df = s14.merge(s13, on="dedupe_key", how="left")
    df["scopus_id"] = df.get("scopus_id", "").fillna("")
    df["pub"]       = df.get("pub", "").fillna("")

    # Preserve manually_retrieved marks from existing master
    df["manually_retrieved"] = ""
    if MASTER.exists():
        old = pd.read_csv(MASTER, dtype=str).fillna("")
        if "manually_retrieved" in old.columns and "dedupe_key" in old.columns:
            marked = old[old["manually_retrieved"].str.strip() != ""][["dedupe_key", "manually_retrieved"]]
            if not marked.empty:
                df = df.merge(marked, on="dedupe_key", how="left", suffixes=("", "_old"))
                df["manually_retrieved"] = df["manually_retrieved_old"].fillna("").combine_first(df["manually_retrieved"])
                df = df.drop(columns=["manually_retrieved_old"], errors="ignore")
                print(f"  Preserved {len(marked):,} manually_retrieved marks from previous master")

    # Final column order
    for col in OUT_COLS:
        if col not in df.columns:
            df[col] = ""
    df = df[OUT_COLS].sort_values(["year", "title"], ascending=[False, True])

    # Write master
    df.to_csv(MASTER, index=False)
    print(f"Master -> {MASTER}  ({len(df):,} records)")

    # Write versioned snapshot
    ver = next_version()
    today = date.today().isoformat()
    versioned = VERSIONS / f"missing_papers_{ver}_{today}.csv"
    df.to_csv(versioned, index=False)
    print(f"Version -> {versioned}")

    # Summary
    with_doi    = (df["doi"].str.strip() != "").sum()
    without_doi = (df["doi"].str.strip() == "").sum()
    marked      = (df["manually_retrieved"].str.strip() != "").sum()
    print(f"\n  Total   : {len(df):,}")
    print(f"  With DOI: {with_doi:,}")
    print(f"  No DOI  : {without_doi:,}")
    if marked:
        print(f"  Marked manually_retrieved: {marked:,}")


if __name__ == "__main__":
    main()
