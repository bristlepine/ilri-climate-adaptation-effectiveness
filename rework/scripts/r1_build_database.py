#!/usr/bin/env python3
"""
r1_build_database.py

Convert Forrest's rework output (rework/data/D5_rework_database.csv, 152 rows,
one per ILRI01-FT-XXX record) into a step15_human.csv-equivalent dataset the
original pipeline's step16_map_visualise.py can consume unmodified, plus a
step15b_summary.json-equivalent for the PRISMA box counts.

Filter: final_include == "yes" (115 / 151 coded records) — the full rework
recommendation, per the 2026-10-21 decision with Zarrar. Rows are joined back
to the original pipeline implicitly via DOI (title as fallback), matching how
step16_map_visualise.py and deliverables/_gen_prisma.py already key on doi.

Run:
  conda run -n ilri01 python rework/scripts/r1_build_database.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REWORK_ROOT = Path(__file__).parent.parent
REPO_ROOT = REWORK_ROOT.parent
DATA_IN = REWORK_ROOT / "data" / "D5_rework_database.csv"
OUT_DIR = REWORK_ROOT / "outputs" / "step15"

# Original (pre-rework) counts, from scripts/outputs/step15/step15b_summary.json —
# the full-text coder-assignment pool is unchanged by the rework; only the
# eligibility call on the already-coded 151 records changes.
ORIG_SUMMARY = REPO_ROOT / "scripts" / "outputs" / "step15" / "step15b_summary.json"

# step15_human.csv column order (must match exactly for step16_map_visualise.py
# and _gen_prisma.py's downstream consumers to read it without changes).
HUMAN_COLUMNS = [
    "batch", "doi", "title", "coder_id", "notes",
    "publication_year", "publication_type", "country_region", "geographic_scale",
    "producer_type", "marginalized_subpopulations", "adaptation_focus",
    "process_outcome_domains", "indicators_measured", "methodological_approach",
    "purpose_of_assessment", "data_sources", "temporal_coverage",
    "cost_data_reported", "strengths_and_limitations", "lessons_learned",
]

CODEBOOK_FIELDS = [
    "publication_year", "publication_type", "country_region", "geographic_scale",
    "producer_type", "marginalized_subpopulations", "adaptation_focus",
    "process_outcome_domains", "indicators_measured", "methodological_approach",
    "purpose_of_assessment", "data_sources", "temporal_coverage",
    "cost_data_reported", "strengths_and_limitations", "lessons_learned",
]


def main() -> None:
    db = pd.read_csv(DATA_IN, dtype=str)
    print(f"[r1] Loaded {len(db)} rows from {DATA_IN.name}")

    included = db[db["final_include"].astype(str).str.lower() == "yes"].copy()
    excluded = db[db["final_include"].astype(str).str.lower() == "no"].copy()
    uncoded = db[db["final_include"].isna()].copy()
    print(f"[r1] final_include: yes={len(included)}  no={len(excluded)}  "
          f"uncoded(no full text)={len(uncoded)}")

    out = pd.DataFrame()
    out["batch"] = included["source_round"]
    out["doi"] = included["doi"]
    out["title"] = included["title"]
    out["coder_id"] = included["reviewer"]
    out["notes"] = included["reviewer_notes"].fillna("")
    # Flag borderline inclusions in notes so they're traceable downstream without
    # a schema change (step15_human.csv has no borderline column).
    borderline_mask = included["borderline_flag"].astype(str).str.lower() == "true"
    out.loc[borderline_mask, "notes"] = (
        out.loc[borderline_mask, "notes"].fillna("") + " [borderline_inclusion]"
    ).str.strip(" |")

    for field in CODEBOOK_FIELDS:
        out[field] = included[field]

    out = out[HUMAN_COLUMNS]

    dupe_dois = out["doi"].dropna()
    dupe_dois = dupe_dois[dupe_dois.duplicated()]
    if len(dupe_dois):
        print(f"[r1] WARNING: {len(dupe_dois)} duplicate DOI(s) in output: "
              f"{dupe_dois.tolist()}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "step15_human.csv"
    out.to_csv(out_csv, index=False)
    print(f"[r1] Wrote {len(out)} rows -> {out_csv}")

    # step15b_summary.json equivalent for _gen_prisma.py / step16's PRISMA numbers.
    orig = json.loads(ORIG_SUMMARY.read_text()) if ORIG_SUMMARY.exists() else {}
    total_coder_rows = orig.get("total_coder_rows", 180)
    orig_excluded = orig.get("excluded_before_filter", 29)
    new_excluded_at_ft = len(excluded)  # rework exclusions, on top of the original gate
    summary = {
        "total_coder_rows": total_coder_rows,
        "included_after_filter": len(out),
        "excluded_before_filter": orig_excluded + new_excluded_at_ft,
        "final_human_records": len(out),
        "batches": orig.get("batches"),
        "unique_dois": int(out["doi"].nunique()),
        "output_csv": str(out_csv),
        "rework_note": (
            f"Prior pipeline excluded {orig_excluded} at the RA confirmed_include "
            f"gate (unchanged). Forrest's rework additionally excludes "
            f"{new_excluded_at_ft} of the {orig.get('final_human_records', 151)} "
            f"records the RAs had included (23 clear + 13 borderline — full "
            f"rework recommendation adopted per 2026-10-21 decision), and keeps "
            f"4 out-of-window (2026) records and FT-021 (edited book, unsplit) "
            f"provisionally, pending co-author sign-off — see "
            f"rework/data/D5_Coding_Rework_Executive_Summary.md."
        ),
    }
    summary_path = OUT_DIR / "step15b_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[r1] Wrote summary -> {summary_path}")
    print(f"[r1] PRISMA: {total_coder_rows} coder rows -> "
          f"{summary['excluded_before_filter']} excluded -> {len(out)} included "
          f"(was {orig.get('final_human_records', 151)} included / "
          f"{orig_excluded} excluded)")


if __name__ == "__main__":
    main()
