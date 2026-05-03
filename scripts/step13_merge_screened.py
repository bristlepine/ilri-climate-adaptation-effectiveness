#!/usr/bin/env python3
"""
step13_merge_screened.py

Merge abstract screening results from Scopus (step12) and multidatabase (step12b)
into a single pool of all "Include" decisions for full-text retrieval (step13+).

Reads:
  - scripts/outputs/step12/step12_results.csv (Scopus, ~6,218 includes)
  - scripts/outputs/step12b/step12b_results.csv (multidatabase, ~2,337 includes)

Outputs:
  - scripts/outputs/step13/step13_all_included.csv (combined pool for FT retrieval)
  - scripts/outputs/step13/step13_merge_summary.json

Usage:
  python scripts/step13_merge_screened.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "scripts" / "outputs" / "step13"

STEP12_CSV = ROOT / "scripts" / "outputs" / "step12" / "step12_results.csv"
STEP12B_CSV = ROOT / "scripts" / "outputs" / "step12b" / "step12b_results.csv"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Step 13: Merge Screened Records (Scopus + Multidatabase)")
    print(f"{'='*70}\n")

    # Load step12 (Scopus)
    print(f"Loading Scopus screening results: {STEP12_CSV.name}")
    df12 = pd.read_csv(STEP12_CSV, dtype=str).fillna("")
    df12_include = df12[df12["screen_decision"] == "Include"].copy()
    print(f"  Scopus includes: {len(df12_include):,} / {len(df12):,}")

    # Load step12b (multidatabase)
    print(f"Loading multidatabase screening results: {STEP12B_CSV.name}")
    df12b = pd.read_csv(STEP12B_CSV, dtype=str).fillna("")
    df12b_include = df12b[df12b["screen_decision"] == "Include"].copy()
    print(f"  Multidatabase includes: {len(df12b_include):,} / {len(df12b):,}")

    # Add source tag
    df12_include["source_batch"] = "scopus"
    df12b_include["source_batch"] = "multidatabase"

    # Combine
    combined = pd.concat([df12_include, df12b_include], ignore_index=True)
    print(f"\n  Total included records: {len(combined):,}")

    # Remove duplicates by DOI (prefer Scopus if same DOI)
    if "doi" in combined.columns:
        # Filter out empty DOIs
        combined_with_doi = combined[combined["doi"].str.strip() != ""].copy()
        combined_no_doi = combined[combined["doi"].str.strip() == ""].copy()

        # Deduplicate on DOI (keep first = Scopus)
        combined_dedup = combined_with_doi.drop_duplicates("doi", keep="first")
        print(f"  After DOI dedup: {len(combined_dedup):,} (removed {len(combined_with_doi) - len(combined_dedup):,} duplicates)")

        # Recombine with non-DOI records
        combined = pd.concat([combined_dedup, combined_no_doi], ignore_index=True)
        print(f"  Total after including non-DOI records: {len(combined):,}")

    # Write output
    out_csv = OUT_DIR / "step13_all_included.csv"
    combined.to_csv(out_csv, index=False)
    print(f"\n  Written: {out_csv.name}")

    # Summary
    summary = {
        "scopus_included": int(len(df12_include)),
        "multidatabase_included": int(len(df12b_include)),
        "combined_before_dedup": int(len(df12_include) + len(df12b_include)),
        "combined_after_doi_dedup": int(len(combined)),
        "unique_dois": int(combined["doi"].nunique()) if "doi" in combined.columns else None,
        "by_source": combined["source_batch"].value_counts().to_dict() if "source_batch" in combined.columns else {},
        "output_csv": str(out_csv),
    }

    summary_json = OUT_DIR / "step13_merge_summary.json"
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  Results")
    print(f"{'='*70}")
    print(f"  Scopus included:        {summary['scopus_included']:,}")
    print(f"  Multidatabase included: {summary['multidatabase_included']:,}")
    print(f"  Combined (no dedup):    {summary['combined_before_dedup']:,}")
    print(f"  After DOI dedup:        {summary['combined_after_doi_dedup']:,}")
    if summary["by_source"]:
        for src, count in summary["by_source"].items():
            print(f"    - {src}: {count:,}")
    print(f"{'='*70}\n")

    return summary


if __name__ == "__main__":
    main()
