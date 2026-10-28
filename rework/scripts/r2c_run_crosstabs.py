#!/usr/bin/env python3
"""
r2c_run_crosstabs.py

Re-run scripts/step16c_crosstabs.py (domain x time period, producer type x
region — added for the v04 amendment as Figures 11-12) against the rework
data, the same monkeypatch-wrapper pattern as r2_run_figures.py /
r2b_export_pngs.py: load the module, repoint its OUT constant at
rework/outputs, then call its figure functions directly with the rework
step15_human.csv — zero edits to the shared script.

Run (after r1_build_database.py):
  conda run -n ilri01 python rework/scripts/r2c_run_crosstabs.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

REWORK_ROOT = Path(__file__).parent.parent
REPO_ROOT = REWORK_ROOT.parent
STEP16C_PATH = REPO_ROOT / "scripts" / "step16c_crosstabs.py"
HUMAN_CSV = REWORK_ROOT / "outputs" / "step15" / "step15_human.csv"
OUT_DIR = REWORK_ROOT / "outputs" / "step16" / "interactive" / "human"


def main() -> None:
    spec = importlib.util.spec_from_file_location("step16c_crosstabs", STEP16C_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    module.OUT = OUT_DIR
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(HUMAN_CSV)
    n = len(df)
    print(f"[r2c] Loaded {n} studies from {HUMAN_CSV}")

    f1 = module.fig1_domain_time(df, n=n)
    module.save(f1, "crosstab_domain_time")

    f2 = module.fig2_producer_region(df, n=n)
    module.save(f2, "crosstab_producer_region")

    print("[r2c] Done.")


if __name__ == "__main__":
    main()
