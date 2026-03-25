#!/usr/bin/env python3
"""
step14_screen_fulltext.py

Step 14: Full-text screening of included records from step 12.

For each record that passed abstract screening (step 12 Include), reads the
downloaded full text (from step 13) and applies the PCCM eligibility criteria
for a second, more stringent screen.  Records without a retrieved full text are
flagged for manual screening.

Decision logic (same as step 12 — binary only):
  - Any criterion 'no'  → Exclude
  - All 'yes' or any 'unclear' (no 'no') → Include
  - No full text available → flag as 'needs_manual_screening'

Inputs:
  - outputs/step12/step12_results.csv     (abstract screening decisions)
  - outputs/step13/step13_manifest.csv    (retrieved full-text file paths)
  - outputs/step13/fulltext/              (downloaded PDF/HTML files)
  - scripts/criteria.yml

Outputs (under outputs/step14/):
  - step14_results.csv          final Include/Exclude per record
  - step14_results.meta.json    counts, timing, criterion breakdown
  - step14_no_fulltext.csv      records needing manual screening (no PDF)
  - step14_summary.png          summary figure

Run:
  python step14_screen_fulltext.py
  (or via run.py with run_step14 = 1)

TODO: Implement full-text extraction (PDF → text), LLM screening call,
      and output writing.  Structure mirrors step12_screen_full.py.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

# ---------------------------------------------------------------------------
# SETTINGS (fill in when implementing)
# ---------------------------------------------------------------------------

DEFAULT_MODEL   = "qwen2.5:14b"
OLLAMA_BASE_URL = "http://localhost:11434"
SLEEP_S         = 0.1
PRINT_EVERY     = 100


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _out_dir(out_root: Path) -> Path:
    d = out_root / "step14"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _step12_csv(out_root: Path) -> Path:
    return out_root / "step12" / "step12_results.csv"

def _step13_manifest(out_root: Path) -> Path:
    return out_root / "step13" / "step13_manifest.csv"

def _fulltext_dir(out_root: Path) -> Path:
    return out_root / "step13" / "fulltext"


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def run(config: dict) -> dict:
    """Pipeline entrypoint called by run.py."""
    out_root = Path(config.get("out_dir", "outputs"))
    out_dir  = _out_dir(out_root)

    print("[step14] *** NOT YET IMPLEMENTED ***")
    print(f"[step14] Will read: {_step12_csv(out_root)}")
    print(f"[step14] Will read: {_step13_manifest(out_root)}")
    print(f"[step14] Full texts: {_fulltext_dir(out_root)}")
    print(f"[step14] Output dir: {out_dir}")
    print("[step14] Steps to implement:")
    print("  1. Load step12 Include records")
    print("  2. Join with step13 manifest to get local file paths")
    print("  3. Extract text from PDF/HTML (e.g. pdfminer, pypdf, trafilatura)")
    print("  4. Screen each text with LLM using criteria.yml (same as step12)")
    print("  5. Write step14_results.csv, meta.json, no_fulltext.csv, summary.png")

    return {"status": "not_implemented"}


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    run({"out_dir": str(here / "outputs")})
