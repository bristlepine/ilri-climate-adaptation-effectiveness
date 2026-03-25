#!/usr/bin/env python3
"""
step15_extract_data.py

Step 15: Structured data extraction (coding) for all records that pass
full-text screening in step 14.

For each included paper, extracts the attributes needed to populate the
systematic map.  Coding categories follow the PCCM framework defined in
the systematic map protocol.

Coding categories (to be finalised against the protocol):
  - study_design        (RCT, quasi-experimental, observational, etc.)
  - intervention_type   (e.g. CSA practice, index insurance, early warning)
  - outcome_type        (e.g. yield, income, food security, resilience index)
  - outcome_measure     (quantitative / qualitative / mixed)
  - population_group    (smallholder farmers, pastoralists, fisher folk, etc.)
  - gender_focus        (women-specific / gender-disaggregated / not reported)
  - country             (ISO3)
  - region              (SSA / South Asia / LAC / etc.)
  - agroecological_zone
  - climate_hazard      (drought, flood, heat stress, etc.)
  - adaptation_phase    (anticipatory / reactive / transformative)
  - time_horizon        (short / medium / long term)
  - sample_size
  - year_of_study

Inputs:
  - outputs/step14/step14_results.csv   (full-text screened Include records)
  - outputs/step13/fulltext/            (downloaded full texts)
  - scripts/criteria.yml               (for context / definitions)

Outputs (under outputs/step15/):
  - step15_coded.csv          one row per included study, all coding columns
  - step15_coded.meta.json    counts, completion rates, timing
  - step15_incomplete.csv     records where extraction was partial / uncertain

Run:
  python step15_extract_data.py
  (or via run.py with run_step15 = 1)

TODO: Define final coding schema, implement LLM extraction prompt per
      category, add human-review flag for low-confidence extractions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------

DEFAULT_MODEL   = "qwen2.5:14b"
OLLAMA_BASE_URL = "http://localhost:11434"
SLEEP_S         = 0.1
PRINT_EVERY     = 50


# ---------------------------------------------------------------------------
# Coding schema (placeholder — expand with final protocol categories)
# ---------------------------------------------------------------------------

CODING_FIELDS: Dict[str, str] = {
    "study_design":        "RCT | quasi-experimental | observational | review | other",
    "intervention_type":   "free text — map to controlled vocabulary",
    "outcome_type":        "free text — map to controlled vocabulary",
    "outcome_measure":     "quantitative | qualitative | mixed",
    "population_group":    "free text",
    "gender_focus":        "women-specific | gender-disaggregated | not reported",
    "country":             "ISO3 code(s)",
    "region":              "SSA | South Asia | LAC | MENA | EAP | other",
    "agroecological_zone": "free text",
    "climate_hazard":      "drought | flood | heat stress | erratic rainfall | multiple | other",
    "adaptation_phase":    "anticipatory | reactive | transformative | unclear",
    "time_horizon":        "short (<2yr) | medium (2-5yr) | long (>5yr) | unclear",
    "sample_size":         "integer or range",
    "year_of_study":       "integer or range",
}


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _out_dir(out_root: Path) -> Path:
    d = out_root / "step15"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _step14_csv(out_root: Path) -> Path:
    return out_root / "step14" / "step14_results.csv"

def _fulltext_dir(out_root: Path) -> Path:
    return out_root / "step13" / "fulltext"


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def run(config: dict) -> dict:
    """Pipeline entrypoint called by run.py."""
    out_root = Path(config.get("out_dir", "outputs"))
    out_dir  = _out_dir(out_root)

    print("[step15] *** NOT YET IMPLEMENTED ***")
    print(f"[step15] Will read: {_step14_csv(out_root)}")
    print(f"[step15] Full texts: {_fulltext_dir(out_root)}")
    print(f"[step15] Output dir: {out_dir}")
    print("[step15] Coding fields to extract:")
    for field, desc in CODING_FIELDS.items():
        print(f"  {field:25s}: {desc}")
    print("[step15] Steps to implement:")
    print("  1. Load step14 Include records + full texts")
    print("  2. For each record, run LLM extraction prompt per coding field")
    print("  3. Flag low-confidence extractions for human review")
    print("  4. Write step15_coded.csv, meta.json, incomplete.csv")

    return {"status": "not_implemented"}


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    run({"out_dir": str(here / "outputs")})
