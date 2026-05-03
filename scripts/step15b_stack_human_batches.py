#!/usr/bin/env python3
"""
step15b_stack_human_batches.py

Stack and filter human-coded batches into a unified dataset.

Reads all completed coder CSVs from documentation/coding/systematic-map/rounds/FT-R*/
and:
1. Filters to only rows where confirmed_include = yes
2. Stacks all batches into a single DataFrame
3. Validates field schema against expected columns
4. Outputs step15_human.csv with metadata columns (batch, coder_id, notes)

Usage:
  conda run -n ilri01 python scripts/step15b_stack_human_batches.py

Requires:
  - Completed coder CSVs in documentation/coding/systematic-map/rounds/FT-R*/
    (named like coding_ft_r2a_INITIALS.csv, coding_ft_r1a_INITIALS.csv, etc.)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).parent.parent
ROUNDS_ROOT = ROOT / "documentation" / "coding" / "systematic-map" / "rounds"
OUT_ROOT = ROOT / "scripts" / "outputs" / "step15"

# Field schema expected in human-coded CSVs
EXPECTED_FIELDS = [
    "doi", "filename", "publication_year", "publication_type", "country_region",
    "geographic_scale", "producer_type", "marginalized_subpopulations",
    "adaptation_focus", "process_outcome_domains", "indicators_measured",
    "methodological_approach", "purpose_of_assessment", "data_sources",
    "temporal_coverage", "cost_data_reported", "strengths_and_limitations",
    "lessons_learned", "coder_id", "notes",
]

OPTIONAL_FIELDS = [
    "validity_notes",  # may appear in reconciliation CSVs
    "reconciliation_notes",  # reconciliation metadata
    "confirmed_include",  # inclusion gate (FT-R2a onwards)
]

# All valid multi-value separators in human coding
MULTIVALUE_FIELDS = {
    "marginalized_subpopulations", "methodological_approach", "process_outcome_domains",
    "data_sources", "producer_type",
}


def normalize_multivalue(val: Optional[str]) -> str:
    """Normalize multi-value fields: split, strip, lowercase, deduplicate, rejoin."""
    if not val or (isinstance(val, float) and pd.isna(val)):
        return ""
    val = str(val).strip()
    if not val:
        return ""
    # Split on ; or ,
    parts = re.split(r"[;,]", val)
    # Strip, lowercase, deduplicate, filter blanks
    parts = sorted(set(p.strip().lower() for p in parts if p.strip()))
    return ";".join(parts)


def load_coder_csv(path: Path) -> Optional[pd.DataFrame]:
    """Load a coder CSV, validate schema, normalize multi-value fields."""
    try:
        df = pd.read_csv(path, dtype=str).fillna("")

        # Check that required fields are present
        missing = set(EXPECTED_FIELDS) - set(df.columns)
        if missing:
            print(f"  ⚠️  {path.name}: Missing fields {missing}")
            return None

        # Normalize multi-value fields
        for field in MULTIVALUE_FIELDS:
            if field in df.columns:
                df[field] = df[field].apply(normalize_multivalue)

        # Extract batch name from filename or use coder_id
        batch = path.parent.name  # e.g., "FT-R2a"
        df["batch"] = batch

        return df
    except Exception as e:
        print(f"  ❌ {path.name}: {e}")
        return None


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Step 15b: Stack Human-Coded Batches")
    print(f"{'='*70}\n")

    # Find all coder CSVs
    coder_csvs = sorted(ROUNDS_ROOT.glob("FT-R*/coding_*.csv"))

    if not coder_csvs:
        print(f"No coder CSVs found in {ROUNDS_ROOT}")
        print(f"Expected files like: documentation/coding/systematic-map/rounds/FT-R*/coding_*.csv")
        return {"error": "no coder CSVs found"}

    print(f"Found {len(coder_csvs)} coder CSV(s):")
    for csv in coder_csvs:
        print(f"  - {csv.relative_to(ROOT)}")
    print()

    # Load all CSVs
    dfs = []
    total_rows = 0
    included_rows = 0
    excluded_rows = 0

    for csv_path in coder_csvs:
        batch = csv_path.parent.name
        df = load_coder_csv(csv_path)
        if df is None:
            continue

        total_rows += len(df)

        # Filter to confirmed_include = yes (if column exists)
        if "confirmed_include" in df.columns:
            before = len(df)
            df = df[df["confirmed_include"].str.lower() == "yes"]
            included_rows += len(df)
            excluded_rows += (before - len(df))
            print(f"  {batch}: {len(df)} included (of {before} total)")
        else:
            # Older codebooks (R1a) don't have confirmed_include; include all
            included_rows += len(df)
            print(f"  {batch}: {len(df)} rows (no confirmed_include filter)")

        dfs.append(df)

    if not dfs:
        print(f"No valid coder CSVs loaded")
        return {"error": "no valid coder CSVs"}

    # Stack all batches
    stacked = pd.concat(dfs, ignore_index=True)
    print(f"\nStacked: {len(stacked)} rows from {len(dfs)} batch(es)")

    # Ensure required columns exist (backfill if missing)
    for col in EXPECTED_FIELDS:
        if col not in stacked.columns:
            stacked[col] = ""

    # Reorder columns: metadata first, then standard fields
    col_order = ["batch", "doi", "filename", "coder_id", "notes"] + [
        c for c in EXPECTED_FIELDS if c not in ["batch", "doi", "filename", "coder_id", "notes"]
    ]
    col_order = [c for c in col_order if c in stacked.columns]

    stacked = stacked[col_order]

    # Write output
    out_csv = OUT_ROOT / "step15_human.csv"
    stacked.to_csv(out_csv, index=False)
    print(f"\nWritten: {out_csv.name}")

    # Summary
    summary = {
        "total_coder_rows": int(total_rows),
        "included_after_filter": int(included_rows),
        "excluded_before_filter": int(excluded_rows),
        "final_human_records": int(len(stacked)),
        "batches": len(dfs),
        "unique_dois": int(stacked["doi"].nunique()),
        "output_csv": str(out_csv),
    }

    summary_json = OUT_ROOT / "step15b_summary.json"
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  Results")
    print(f"{'='*70}")
    print(f"  Coder rows loaded:      {summary['total_coder_rows']:,}")
    print(f"  Included (confirmed):   {summary['included_after_filter']:,}")
    print(f"  Excluded (false pos):   {summary['excluded_before_filter']:,}")
    print(f"  Final dataset:          {summary['final_human_records']:,} rows")
    print(f"  Unique papers:          {summary['unique_dois']:,} DOIs")
    print(f"  Batches merged:         {summary['batches']}")
    print(f"{'='*70}\n")

    return summary


if __name__ == "__main__":
    import sys
    result = main()
    if isinstance(result, dict) and "error" in result:
        sys.exit(1)
