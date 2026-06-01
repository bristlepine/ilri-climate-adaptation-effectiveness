#!/usr/bin/env python3
"""
step14b_clean_codings.py

Normalize and clean raw human-coded CSVs returned by coders for FT-R2a through
FT-R3 batches. Writes a standardized cleaned file per batch (e.g.
coding_ft-r2a_AMB.csv) ready for step14c_compare_coding.py and
step15b_stack_human_batches.py.

Per-batch fixes:
  FT-R2a  "coding_ft-r2a_template - coding_ft-r2a_.csv"
          → coding_ft-r2a_AMB.csv
          - Title-case Yes/No → lowercase yes/no

  FT-R2b  "coding_ft_r2b_SZC.csv - coding_ft_r2b_SZC.csv.csv"
          → coding_ft-r2b_SZC.csv
          - Drop extra columns (SZC, Unnamed:*)
          - Title-case → lowercase

  FT-R2c  "coding_ft-r2c_template - coding_ft-r2c_template.csv"
          → coding_ft-r2c_LJ.csv
          - Drop coder_initials column
          - Strip trailing whitespace; Title-case → lowercase

  FT-R2d  coding_ft-r2d_AZ.csv.csv
          → coding_ft-r2d_AMB.csv
          - "confirmed_include" literal string → "yes"
          - Rename Publication_tpe → publication_type
          - Drop rows with empty doi

  FT-R3   ft-r3_AZ.csv
          → coding_ft-r3_AMB.csv
          - Straightforward copy with normalization

Usage:
  conda run -n ilri01 python scripts/step14b_clean_codings.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
STEP14B = ROOT / "scripts" / "outputs" / "step14b"

# Columns kept in cleaned output (in order).
# Matches the template schema minus procurement_status.
STANDARD_COLS = [
    "doi", "title", "year", "confirmed_include",
    "publication_year", "publication_type", "country_region",
    "geographic_scale", "producer_type", "marginalized_subpopulations",
    "adaptation_focus", "process_outcome_domains", "indicators_measured",
    "methodological_approach", "purpose_of_assessment", "data_sources",
    "temporal_coverage", "cost_data_reported", "strengths_and_limitations",
    "lessons_learned", "coder_id", "notes",
]


def norm_include(v) -> str:
    """Normalize confirmed_include to lowercase yes / no / ''."""
    if pd.isna(v):
        return ""
    s = str(v).strip().lower()
    if s in ("yes", "y", "confirmed_include"):
        return "yes"
    if s in ("no", "n"):
        return "no"
    return s if s else ""


def keep_standard(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in STANDARD_COLS if c in df.columns]
    return df[cols].copy()


# ── per-batch cleaners ─────────────────────────────────────────────────────────

def clean_r2a(src: Path, dst: Path) -> pd.DataFrame:
    df = pd.read_csv(src, dtype=str).fillna("")
    df["confirmed_include"] = df["confirmed_include"].apply(norm_include)
    df = keep_standard(df)
    df.to_csv(dst, index=False)
    return df


def clean_r2b(src: Path, dst: Path) -> pd.DataFrame:
    df = pd.read_csv(src, dtype=str).fillna("")
    # SZC column = coder initials stored as a column instead of in coder_id
    if "SZC" in df.columns and "coder_id" not in df.columns:
        df = df.rename(columns={"SZC": "coder_id"})
    else:
        df = df.drop(columns=["SZC"], errors="ignore")
    extra = [c for c in df.columns if c.startswith("Unnamed")]
    df = df.drop(columns=extra, errors="ignore")
    df["confirmed_include"] = df["confirmed_include"].apply(norm_include)
    df = keep_standard(df)
    df.to_csv(dst, index=False)
    return df


def clean_r2c(src: Path, dst: Path) -> pd.DataFrame:
    df = pd.read_csv(src, dtype=str).fillna("")
    df = df.drop(columns=["coder_initials"], errors="ignore")
    df["confirmed_include"] = df["confirmed_include"].apply(norm_include)
    df = keep_standard(df)
    df.to_csv(dst, index=False)
    return df


def clean_r2d(src: Path, dst: Path) -> pd.DataFrame:
    df = pd.read_csv(src, dtype=str).fillna("")
    df = df.rename(columns={"Publication_tpe": "publication_type"})
    df["confirmed_include"] = df["confirmed_include"].apply(norm_include)
    df = df[df["doi"].str.strip() != ""]
    df = keep_standard(df)
    df.to_csv(dst, index=False)
    return df


def clean_r3(src: Path, dst: Path) -> pd.DataFrame:
    df = pd.read_csv(src, dtype=str).fillna("")
    df["confirmed_include"] = df["confirmed_include"].apply(norm_include)
    df = keep_standard(df)
    df.to_csv(dst, index=False)
    return df


BATCHES: list[tuple[str, str, str, object]] = [
    ("FT-R2a", "coding_ft-r2a_template - coding_ft-r2a_.csv",              "coding_ft-r2a_AMB.csv", clean_r2a),
    ("FT-R2b", "coding_ft_r2b_SZC.csv - coding_ft_r2b_SZC.csv.csv",        "coding_ft-r2b_SZC.csv", clean_r2b),
    ("FT-R2c", "coding_ft-r2c_template - coding_ft-r2c_template.csv",       "coding_ft-r2c_LJ.csv",  clean_r2c),
    ("FT-R2d", "coding_ft-r2d_AZ.csv.csv",                                  "coding_ft-r2d_AMB.csv", clean_r2d),
    ("FT-R3",  "ft-r3_AZ.csv",                                              "coding_ft-r3_AMB.csv",  clean_r3),
]


def main() -> None:
    print(f"\n{'='*70}")
    print("  Step 14b: Clean Human Coding Files")
    print(f"{'='*70}\n")

    for round_name, src_name, dst_name, clean_fn in BATCHES:
        src = STEP14B / round_name / src_name
        dst = STEP14B / round_name / dst_name
        if not src.exists():
            print(f"  ⚠️  {round_name}: source not found — {src_name}")
            continue
        df = clean_fn(src, dst)
        yes = (df["confirmed_include"] == "yes").sum() if "confirmed_include" in df.columns else "?"
        no  = (df["confirmed_include"] == "no").sum()  if "confirmed_include" in df.columns else "?"
        print(f"  ✓  {round_name} → {dst_name}  [{yes} included, {no} excluded, {len(df)} total]")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
