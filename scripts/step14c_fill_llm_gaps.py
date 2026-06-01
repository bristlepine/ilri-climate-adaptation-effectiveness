#!/usr/bin/env python3
"""
step14c_fill_llm_gaps.py

Identify human-coded papers that the LLM has not yet screened, and register
their manually-procured PDFs so that step14_screen_fulltext.py can process them.

Background
----------
step14 (LLM full-text screening) only ran on auto-retrieved papers that were
already in step13_manifest.csv.  Human coding batches (FT-R2a…FT-R3) include
papers that were manually procured as PDFs and therefore never entered
step13_manifest.  This script bridges that gap.

What it does
------------
1. Loads every completed human-coded CSV (via the same _CODER_RE pattern as
   step15b) and collects all unique DOIs.
2. Loads step14_results.csv to find DOIs that have already been LLM-screened.
3. Computes the difference: human-coded DOIs absent from step14 results.
4. For each missing DOI, searches all batch PDF folders for a matching file.
5. Ensures the DOI is present in step12_results.csv with screen_decision=Include.
6. Adds a new row to step13_manifest.csv with status="ok" pointing at the PDF.
7. Reports gaps where no PDF was found (manual action required).

After running this script, re-run step14_screen_fulltext.py — it is resume-safe
and will skip already-processed papers, only running on the newly registered ones.

Usage
-----
  conda run -n ilri01 python scripts/step14c_fill_llm_gaps.py [--dry-run]

Flags
-----
  --dry-run   Report gaps and matches without writing any files.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT         = Path(__file__).resolve().parent.parent
STEP12_CSV   = ROOT / "scripts" / "outputs" / "step12" / "step12_results.csv"
STEP13_CSV   = ROOT / "scripts" / "outputs" / "step13" / "step13_manifest.csv"
STEP14_CSV   = ROOT / "scripts" / "outputs" / "step14" / "step14_results.csv"
STEP14B_ROOT = ROOT / "scripts" / "outputs" / "step14b"
OUT_ROOT     = ROOT / "scripts" / "outputs" / "step14c"

_CODER_RE = re.compile(r"coding_.+_([A-Za-z]{1,5})\.csv$", re.IGNORECASE)
_SKIP     = {"template", "llm", "reconciled", "fixed", "papers"}

# PDF folder names that may exist inside each batch directory
_PDF_FOLDER_PATTERNS = ["PDFs", "pdfs", "pdf"]


# ── helpers ──────────────────────────────────────────────────────────────────

def _norm_doi(doi: str) -> str:
    """Lowercase, strip, remove leading 'doi:' prefix."""
    return re.sub(r"^doi:", "", str(doi).strip().lower())


def _doi_to_filename_variants(doi: str) -> list[str]:
    """
    Return plausible PDF filename stems for a given DOI.
    Batch PDFs use several naming conventions:
      - slashes dropped:          10.1007978-3-642-22266-5_10
      - slashes → underscores:   10.1007_978-3-642-22266-5_10
      - 'doi_' prefix + underscores: doi_10.1007_978-3-642-22266-5_10
    """
    d = _norm_doi(doi)
    variants = [
        d,
        d.replace("/", "_"),
        d.replace("/", "-"),
        d.replace("/", ""),
        "doi_" + d.replace("/", "_"),
        "doi_" + d.replace("/", "-"),
        "doi_" + d.replace("/", ""),
    ]
    return list(dict.fromkeys(variants))  # deduplicate, preserve order


def _norm_stem(stem: str) -> str:
    """Normalise a PDF filename stem for fuzzy matching."""
    s = stem.lower().strip()
    s = unicodedata.normalize("NFKD", s)
    # remove 'doi_' or 'doi-' prefix
    s = re.sub(r"^doi[_-]", "", s)
    # collapse separators
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


def _find_pdf_in_batch_dirs(doi: str) -> Optional[Path]:
    """
    Search all batch PDF folders for a file matching this DOI.
    Returns the absolute Path if found, else None.
    """
    variants = _doi_to_filename_variants(doi)
    norm_variants = {_norm_stem(v) for v in variants}

    for batch_dir in sorted(STEP14B_ROOT.iterdir()):
        if not batch_dir.is_dir():
            continue
        # Find PDF subfolder(s) in this batch
        pdf_dirs: list[Path] = []
        for child in batch_dir.iterdir():
            if child.is_dir() and any(p in child.name for p in _PDF_FOLDER_PATTERNS):
                pdf_dirs.append(child)

        for pdf_dir in pdf_dirs:
            for pdf_file in pdf_dir.glob("*.pdf"):
                stem_norm = _norm_stem(pdf_file.stem)
                if stem_norm in norm_variants:
                    return pdf_file

    return None


def _discover_coder_csvs() -> list[Path]:
    found: list[Path] = []
    if STEP14B_ROOT.exists():
        for p in sorted(STEP14B_ROOT.glob("FT-R*/coding_*.csv")):
            if any(s in p.name.lower() for s in _SKIP):
                continue
            if _CODER_RE.search(p.name):
                found.append(p)
    return found


# ── main ─────────────────────────────────────────────────────────────────────

def main(dry_run: bool = False) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Step 14c: Fill LLM Gaps (identify + register missing PDFs)")
    print(f"  Mode: {'DRY RUN — no files written' if dry_run else 'LIVE'}")
    print(f"{'='*70}\n")

    # ── 1. Human-coded DOIs ───────────────────────────────────────────────────
    coder_csvs = _discover_coder_csvs()
    if not coder_csvs:
        print("No completed coder CSVs found. Nothing to do.")
        sys.exit(0)

    human_rows: list[dict] = []
    for csv_path in coder_csvs:
        df = pd.read_csv(csv_path, dtype=str).fillna("")
        for _, row in df.iterrows():
            doi = _norm_doi(row.get("doi", ""))
            if doi:
                human_rows.append({
                    "doi": doi,
                    "title": row.get("title", ""),
                    "batch": csv_path.parent.name,
                    "confirmed_include": row.get("confirmed_include", ""),
                })

    human_df = pd.DataFrame(human_rows).drop_duplicates(subset=["doi"])
    print(f"Human-coded unique DOIs: {len(human_df):,} (from {len(coder_csvs)} batch files)")

    # ── 2. Already LLM-screened DOIs ─────────────────────────────────────────
    if not STEP14_CSV.exists():
        print(f"  ⚠️  step14_results.csv not found at {STEP14_CSV}")
        llm_dois: set[str] = set()
    else:
        step14 = pd.read_csv(STEP14_CSV, dtype=str).fillna("")
        llm_dois = {_norm_doi(d) for d in step14["doi"].dropna() if d.strip()}
        print(f"LLM-screened DOIs:       {len(llm_dois):,}")

    # ── 3. Gap ───────────────────────────────────────────────────────────────
    missing = human_df[~human_df["doi"].isin(llm_dois)].copy()
    print(f"Missing from LLM:        {len(missing):,}\n")

    if missing.empty:
        print("No gaps found — all human-coded DOIs have been LLM-screened.")
        return

    print(f"{'─'*70}")
    print(f"  Gap details")
    print(f"{'─'*70}")
    for batch, grp in missing.groupby("batch"):
        print(f"  {batch}: {len(grp)} missing DOIs")
        for _, r in grp.iterrows():
            print(f"    - {r['doi']}")
    print()

    # ── 4. Find PDFs ──────────────────────────────────────────────────────────
    results: list[dict] = []
    for _, row in missing.iterrows():
        pdf_path = _find_pdf_in_batch_dirs(row["doi"])
        results.append({
            "doi": row["doi"],
            "title": row["title"],
            "batch": row["batch"],
            "confirmed_include": row["confirmed_include"],
            "pdf_found": pdf_path is not None,
            "pdf_path": str(pdf_path) if pdf_path else "",
        })

    found   = [r for r in results if r["pdf_found"]]
    no_pdf  = [r for r in results if not r["pdf_found"]]

    print(f"PDFs matched:  {len(found):,}")
    print(f"PDFs missing:  {len(no_pdf):,}\n")

    if no_pdf:
        print("  ⚠️  No PDF found for:")
        for r in no_pdf:
            print(f"    [{r['batch']}] {r['doi']}")
        print()

    if not found:
        print("No PDFs available to register. Re-retrieve manually and rerun.")
        return

    if dry_run:
        print("[DRY RUN] Would register the following in step13_manifest:")
        for r in found:
            print(f"  {r['doi']}  →  {r['pdf_path']}")
        return

    # ── 5. Ensure DOI is in step12_results with Include ───────────────────────
    s12 = pd.read_csv(STEP12_CSV, dtype=str).fillna("")
    s12_dois = {_norm_doi(d) for d in s12["doi"].dropna()}

    s12_new_rows: list[dict] = []
    for r in found:
        if r["doi"] not in s12_dois:
            print(f"  Adding to step12: {r['doi']}")
            s12_new_rows.append({
                "screen_decision": "Include",
                "screen_reasons": "manually procured for human coding batch",
                "dedupe_key": f"doi:{r['doi']}",
                "title": r["title"],
                "doi": r["doi"],
            })

    if s12_new_rows:
        s12_append = pd.DataFrame(s12_new_rows)
        s12_combined = pd.concat([s12, s12_append], ignore_index=True)
        s12_combined.to_csv(STEP12_CSV, index=False)
        print(f"  step12_results.csv: added {len(s12_new_rows)} rows\n")
    else:
        print("  All gap DOIs already in step12_results.csv\n")

    # ── 6. Register PDFs in step13_manifest ───────────────────────────────────
    s13 = pd.read_csv(STEP13_CSV, dtype=str).fillna("")
    s13_ok_dois = {
        _norm_doi(d)
        for d, st in zip(s13["doi"].fillna(""), s13["status"].fillna(""))
        if st == "ok"
    }

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    s13_new_rows: list[dict] = []

    for r in found:
        if r["doi"] in s13_ok_dois:
            print(f"  Already in step13 (ok): {r['doi']}")
            continue
        pdf = Path(r["pdf_path"])
        size_kb = round(pdf.stat().st_size / 1024, 1) if pdf.exists() else 0
        print(f"  Registering in step13: {r['doi']}  ({size_kb} KB)")
        s13_new_rows.append({
            "dedupe_key": f"doi:{r['doi']}",
            "doi": r["doi"],
            "scopus_id": "",
            "title": r["title"],
            "year": "",
            "pub": "",
            "source_db": "manual",
            "missing_abstract": "",
            "status": "ok",
            "source": "manual_pdf",
            "file_path": str(pdf),
            "file_size_kb": str(size_kb),
            "url": "",
            "note": f"manually procured for batch {r['batch']}",
            "timestamp_utc": ts,
        })

    if s13_new_rows:
        s13_append = pd.DataFrame(s13_new_rows)
        s13_combined = pd.concat([s13, s13_append], ignore_index=True)
        s13_combined.to_csv(STEP13_CSV, index=False)
        print(f"\n  step13_manifest.csv: added {len(s13_new_rows)} rows")
    else:
        print("\n  All matched PDFs already registered in step13_manifest.csv")

    # ── 7. Write gap report ───────────────────────────────────────────────────
    import json
    report = {
        "human_unique_dois": len(human_df),
        "llm_screened_dois": len(llm_dois),
        "missing_dois": len(missing),
        "pdfs_found_and_registered": len(s13_new_rows),
        "pdfs_not_found": len(no_pdf),
        "dois_with_no_pdf": [r["doi"] for r in no_pdf],
        "registered": [{"doi": r["doi"], "pdf": r["pdf_path"]} for r in found],
    }
    report_path = OUT_ROOT / "step14c_gap_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Gap report: {report_path.name}")

    print(f"\n{'='*70}")
    print(f"  Summary")
    print(f"{'='*70}")
    print(f"  Gap DOIs found:        {len(missing):,}")
    print(f"  PDFs matched:          {len(found):,}")
    print(f"  Registered in step13:  {len(s13_new_rows):,}")
    print(f"  PDFs not found:        {len(no_pdf):,}")
    print(f"\n  Next step: run step14_screen_fulltext.py to process new papers")
    print(f"  (resume-safe — will skip already-screened DOIs)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Report gaps without writing any files")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
