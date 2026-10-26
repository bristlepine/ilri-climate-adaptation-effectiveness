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

Output is a single two-tab workbook (outputs/d6_candidates.xlsx) rather than
a bare CSV, so the criteria travel with the list instead of living only in
this docstring or chat history: "D6 candidates" (the flagged papers) and
"Methodology" (this docstring's reasoning, in sheet form).

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
from openpyxl.styles import Alignment

REWORK_ROOT = Path(__file__).parent.parent
HUMAN_CSV = REWORK_ROOT / "outputs" / "step15" / "step15_human.csv"
OUT_XLSX = REWORK_ROOT / "outputs" / "d6_candidates.xlsx"

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

    methodology_rows = [
        ("What this is",
         f"{candidate_mask.sum()} of {len(df)} D5 confirmed-include papers, flagged as "
         f"the subset to hand off to D6."),
        ("Why",
         "Forrest's 2026-10-26 ask (logged in work/handover/ilri.md): \"flag which of the "
         "confirmed-include papers are process/outcome-domain papers using yield/income "
         "methods.\" That phrase wasn't an operational definition anywhere in the data — "
         "this sheet is Claude's attempt to make one explicit and reproducible, confirmed "
         "with Zarrar 2026-10-26. It is not Forrest's own wording."),
        ("", ""),
        ("CRITERION", ""),
        ("A paper is a candidate if BOTH:", ""),
        ("1. Domain",
         f"process_outcome_domains contains yields_productivity or income_assets "
         f"({domain_mask.sum()}/{len(df)} papers meet this alone)"),
        ("2. Method",
         "methodological_approach contains quantitative, experimental, or "
         "modeling_with_empirical_validation — i.e. the outcome was actually measured "
         "with a quantitative method, not just discussed qualitatively"),
        ("Why these two fields",
         "Both are controlled-vocabulary codebook fields (fixed value sets), not free "
         "text — deliberate, see \"Rejected alternatives\" below."),
        ("", ""),
        ("REJECTED ALTERNATIVES", ""),
        ("Domain tag alone, no method filter",
         f"{domain_mask.sum()} papers — too broad: includes papers where yield/income is "
         f"mentioned but only discussed qualitatively (narrative), not measured."),
        ("Domain tag + method filter + a regex over indicators_measured "
         "requiring an explicit unit (kg/ha, $, etc.)",
         "30 papers — tried first, since indicators_measured names the actual indicator. "
         "Rejected: it's free text, so the regex's fixed unit vocabulary produced false "
         "negatives — e.g. \"wheat and barley yield 1988-2017\" (no adjacent unit), "
         "\"catchability (kg/manday)\" (unanticipated unit format), \"Household income per "
         "adult equivalent (KES)\" (currency code not in the list). No fixed vocabulary "
         "reliably covers free text, so this was dropped in favour of the two "
         "controlled-vocabulary fields above, which don't have that failure mode."),
        ("", ""),
        ("REPRODUCE", ""),
        ("Command", "conda run -n ilri01 python rework/scripts/r8_flag_d6_candidates.py"),
        ("Source",
         f"rework/outputs/step15/step15_human.csv ({len(df)} confirmed-include records, "
         f"per rework/scripts/r1_build_database.py's confirmed_include filter)."),
    ]
    methodology = pd.DataFrame(methodology_rows, columns=["", ""])

    OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        candidates.to_excel(writer, sheet_name="D6 candidates", index=False)
        methodology.to_excel(writer, sheet_name="Methodology", index=False, header=False)

        for sheet_name, frame, widths in [
            ("D6 candidates", candidates, [16, 45, 32, 28, 45]),
            ("Methodology", methodology, [45, 100]),
        ]:
            ws = writer.sheets[sheet_name]
            for i, w in enumerate(widths, start=1):
                ws.column_dimensions[chr(64 + i)].width = w
            for row in ws.iter_rows():
                for cell in row:
                    cell.alignment = Alignment(wrap_text=True, vertical="top")

    print(f"[r8] Wrote {len(candidates)} candidate rows + methodology sheet -> {OUT_XLSX}")


if __name__ == "__main__":
    main()
