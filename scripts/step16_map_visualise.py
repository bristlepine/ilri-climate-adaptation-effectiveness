#!/usr/bin/env python3
"""
step16_map_visualise.py

Step 16: Systematic map visualisations and figures.

Reads the coded dataset from step 15 and produces the evidence map figures
required for the systematic map report and journal submission.

Planned outputs:
  1. ROSES flow diagram      — record counts at each pipeline stage
  2. Geographic heat map     — study counts by country / region
  3. Intervention × Outcome  — bubble chart / heat map (core evidence map)
  4. Study design breakdown  — bar chart by design type
  5. Temporal trends         — publications per year
  6. Gender focus breakdown  — stacked bar
  7. Climate hazard matrix   — intervention × hazard heat map
  8. Adaptation phase chart  — by region or intervention type

Inputs:
  - outputs/step15/step15_coded.csv     (coded study attributes)
  - outputs/step12/step12_results.meta.json  (screening counts for ROSES)
  - outputs/step14/step14_results.meta.json  (full-text screening counts)

Outputs (under outputs/step16/):
  - roses_flow.png
  - geo_heatmap.png / geo_heatmap.html   (interactive choropleth)
  - intervention_outcome_bubble.png
  - study_design_bar.png
  - temporal_trends.png
  - gender_focus_bar.png
  - climate_hazard_matrix.png
  - adaptation_phase_bar.png
  - step16_figures.meta.json

Run:
  python step16_map_visualise.py
  (or via run.py with run_step16 = 1)

TODO: Implement each figure using matplotlib / plotly / geopandas.
      Controlled vocabulary for intervention/outcome categories must be
      finalised in step 15 before this step can be built out.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _out_dir(out_root: Path) -> Path:
    d = out_root / "step16"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _step15_csv(out_root: Path) -> Path:
    return out_root / "step15" / "step15_coded.csv"

def _step12_meta(out_root: Path) -> Path:
    return out_root / "step12" / "step12_results.meta.json"

def _step14_meta(out_root: Path) -> Path:
    return out_root / "step14" / "step14_results.meta.json"


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def run(config: dict) -> dict:
    """Pipeline entrypoint called by run.py."""
    out_root = Path(config.get("out_dir", "outputs"))
    out_dir  = _out_dir(out_root)

    print("[step16] *** NOT YET IMPLEMENTED ***")
    print(f"[step16] Will read: {_step15_csv(out_root)}")
    print(f"[step16] Will read: {_step12_meta(out_root)}")
    print(f"[step16] Will read: {_step14_meta(out_root)}")
    print(f"[step16] Output dir: {out_dir}")
    print("[step16] Figures to produce:")
    figures = [
        "roses_flow.png             — ROSES flow diagram (record counts at each stage)",
        "geo_heatmap.png/html       — study counts by country/region (choropleth)",
        "intervention_outcome.png   — intervention × outcome bubble/heat map",
        "study_design_bar.png       — breakdown by study design",
        "temporal_trends.png        — publications per year",
        "gender_focus_bar.png       — gender focus breakdown",
        "climate_hazard_matrix.png  — intervention × climate hazard heat map",
        "adaptation_phase_bar.png   — adaptation phase by region/intervention",
    ]
    for f in figures:
        print(f"  {f}")
    print("[step16] Dependencies: step15 coded data + finalised coding vocabulary")

    return {"status": "not_implemented"}


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    run({"out_dir": str(here / "outputs")})
