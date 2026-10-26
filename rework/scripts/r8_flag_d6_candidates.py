#!/usr/bin/env python3
"""
r8_flag_d6_candidates.py

Flags the D6 candidate subset: confirmed-include papers (rework/outputs/
step15/step15_human.csv) whose outcome focus is yields/income, assessed with
a quantitative method. Forrest's 2026-10-26 ask (see work/handover/ilri.md)
was "flag which of the confirmed-include papers are process/outcome-domain
papers using yield/income methods" — a vague prose criterion, so this script
makes the operational definition explicit and reproducible rather than a
one-off judgment call buried in chat history.

Criterion (confirmed with Zarrar 2026-10-26 — deliberately grounded only in
controlled-vocabulary codebook fields, not free-text parsing of
indicators_measured: that field is free text, and a first attempt at a
stricter regex over it produced false negatives — e.g. "wheat and barley
yield 1988-2017" with no adjacent unit, "catchability (kg/manday)",
"Household income per adult equivalent (KES)" — because no fixed unit
vocabulary can cover free text reliably):

    process_outcome_domains contains yields_productivity OR income_assets
    AND
    methodological_approach contains quantitative OR experimental
    OR modeling_with_empirical_validation

Earlier, rejected alternatives (for the record, not reused):
  - domain tag alone (no method filter): 53/108 — too broad, includes
    qualitative-only papers where yield/income is discussed, not measured.
  - domain tag + method filter + free-text unit regex on indicators_measured:
    30/108 — undercounts due to incomplete unit vocabulary (see above).

Run (after r1_build_database.py):
  conda run -n ilri01 python rework/scripts/r8_flag_d6_candidates.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REWORK_ROOT = Path(__file__).parent.parent
HUMAN_CSV = REWORK_ROOT / "outputs" / "step15" / "step15_human.csv"
OUT_CSV = REWORK_ROOT / "outputs" / "d6_candidates.csv"

OUTCOME_DOMAINS = ["yields_productivity", "income_assets"]
QUANT_METHODS = ["quantitative", "experimental", "modeling_with_empirical_validation"]


def main() -> None:
    df = pd.read_csv(HUMAN_CSV, dtype=str).fillna("")

    domain_mask = pd.Series(False, index=df.index)
    for d in OUTCOME_DOMAINS:
        domain_mask |= df["process_outcome_domains"].str.contains(d, case=False)

    method_lower = df["methodological_approach"].str.lower()
    method_mask = pd.Series(False, index=df.index)
    for m in QUANT_METHODS:
        method_mask |= method_lower.str.contains(m)

    candidate_mask = domain_mask & method_mask
    candidates = df[candidate_mask][
        ["doi", "title", "process_outcome_domains", "methodological_approach",
         "indicators_measured"]
    ].copy()

    print(f"[r8] {len(df)} confirmed-include records")
    print(f"[r8] {domain_mask.sum()} tagged yields_productivity/income_assets")
    print(f"[r8] {candidate_mask.sum()} also use a quantitative/experimental/"
          f"modeling method -> D6 candidates")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(OUT_CSV, index=False)
    print(f"[r8] Wrote {len(candidates)} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()
