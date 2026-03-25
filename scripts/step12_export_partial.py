#!/usr/bin/env python3
"""
step12_export_partial.py

Export a partial step12_results.csv from the JSONL cache while step12 is
still running. Safe to run at any time — read-only access to the cache.

Usage:
    python step12_export_partial.py

Writes:
    outputs/step12/step12_results_partial.csv
    outputs/step12/step12_results_partial.meta.json
"""

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

OUT_ROOT = Path(__file__).resolve().parent / "outputs"
JSONL    = OUT_ROOT / "step12" / "step12_results_details.jsonl"
OUT_CSV  = OUT_ROOT / "step12" / "step12_results_partial.csv"
OUT_META = OUT_ROOT / "step12" / "step12_results_partial.meta.json"


def main() -> None:
    if not JSONL.exists():
        sys.exit(f"Cache not found: {JSONL}")

    records = {}
    with open(JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
                k = j.get("key", "")
                if k:
                    records[k] = j          # last-record-wins (same as step12)
            except Exception:
                continue

    if not records:
        sys.exit("Cache is empty — nothing to export yet.")

    df = pd.DataFrame(list(records.values()))

    # Put decision cols first
    decision_cols = ["screen_decision", "screen_reasons"]
    other_cols    = [c for c in df.columns if c not in decision_cols and not c.startswith("_")]
    col_order     = [c for c in decision_cols + other_cols if c in df.columns]
    df[col_order].to_csv(OUT_CSV, index=False)

    # Summary
    decisions = df["screen_decision"].fillna("").str.strip()
    counts    = decisions.value_counts(dropna=False).to_dict()

    excl_by_crit: Counter = Counter()
    import re
    for d, r in zip(decisions, df.get("screen_reasons", pd.Series()).fillna("").astype(str)):
        if d == "Exclude":
            for m in re.finditer(r"\b([1-5]_[a-zA-Z0-9]+)\s*:", r):
                excl_by_crit[m.group(1)] += 1

    meta = {
        "rows_in_cache":        len(df),
        "decision_counts":      {k: int(v) for k, v in counts.items()},
        "excluded_by_criterion": dict(sorted(excl_by_crit.items(), key=lambda kv: (-kv[1], kv[0]))),
        "note": "Partial export — step12 is still running. Re-run to refresh.",
    }

    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Exported {len(df):,} records → {OUT_CSV}")
    print(f"Decision counts: {counts}")
    print(f"Meta            → {OUT_META}")


if __name__ == "__main__":
    main()
