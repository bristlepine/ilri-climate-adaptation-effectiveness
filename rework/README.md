# D5 Coding Rework — impact pipeline

Tracks the impact of the October 2026 full-text re-coding (Forrest's agentic rework —
lead reviewer Claude Opus 5.5) on the D5 deliverable: the Annex 6 ROSES database, PRISMA
flow, and all human-track figures (EGM, geographic, methodology, equity, temporal).

**This folder is purely additive.** Nothing under `scripts/`, `scripts/outputs/`, or
`deliverables/` is modified by anything in here. The original pipeline stays runnable
and its outputs stay the historical record of the *v02* deliverable.

## Provenance

`data/` holds frozen copies of Forrest's rework outputs, pulled from the Drive folder
below on **2026-10-21** (Drive files can change — this is the snapshot everything here
is built from; re-pull and note the new date if Forrest issues a revision):

- Source folder: `ILRI D5 Rework` — <https://drive.google.com/drive/folders/1GAa-HxjyvsimyO2LszfAqTaubW94Qyb6>
- `D5_Rework_Outputs/` subfolder (file IDs, for re-pulling):
  - `D5_rework_database.csv` — `1mZALRk5lGt_TL_-V1wIUOWsgvY_SE0Yf`
  - `D5_rework_eligibility.csv` — `1kMYxpHTTtLfOh1WoTfGqqdCHJu-uci87`
  - `D5_rework_record_summary.csv` — `1oaVWKLnYlycxWiV8a9k3ggcZ1rEwVBbA`
  - `D5_rework_deterministic_checks.csv` — `1gpyXXw7jFz8LfUuVpIUEwKGgF8eSj-s-`
  - `D5_rework_vs_prior_comparison.csv` — `1VkABC20UldDCpO0tW7Q5lCOkfQ9axYzs`
  - `D5_rework_field_rationale_evidence.csv` — `1SGXP6dZzyCoRLOJVi3NW5mo9l3QmbmP2`
  - `D5_rework_blinded_audit.csv` — `1N1hJemkcl3KjXRBPOScwF3d6nuPs4OYI`
  - `D5_rework_coding.xlsx` (workbook bundling the above sheets — not pulled; CSVs are
    the working copies) — `1OnygDBW-QhtD17Q27Nxu4QE4gY8-kP43`
- Executive summary / full report: `D5_Coding_Rework_Executive_Summary.md`,
  `D5_Coding_Rework_Report.md` (IDs `1cVvmTRDwE_w7OpVVsJI5mOSQnZRxn50T` /
  `1cIb_UPY52ZlYsXQdt3WZ4zyy_ObtCj0j`)

`forrest_scripts/` is a placeholder — the actual agentic rework workspace
(`rework_workspace/`: reviewer brief, codebook copy, extracted texts, per-record JSON
coding, second-review JSON, lead decisions, scripts) has **not yet been handed over**
by Forrest. Drop it in here once received; nothing downstream depends on it, it's kept
for audit-trail completeness only.

## Decisions baked into this prelim pass (confirmed with Zarrar 2026-10-21)

- **Inclusion set:** full rework recommendation — all 36 exclusions accepted
  (23 clear + 13 borderline), giving **115 / 151 included**. Borderline cases get a
  human review pass before this becomes the final (non-prelim) D5.
- **Open edge cases left as-is, footnoted rather than resolved:**
  - 4 records outside the 2005–2025 protocol window (FT-093, FT-096, FT-111, FT-137)
    — provisionally retained.
  - FT-021 (edited book) — provisionally kept as a single record, not split into chapters.
  - FT-005 (no full text) — unchanged treatment.

See `data/D5_Coding_Rework_Executive_Summary.md` §"Decisions needed from co-authors" for
the full list this prelim pass does *not* resolve.

## Pipeline

```
rework/
  data/                     Forrest's rework outputs (frozen, see Provenance above)
  forrest_scripts/          placeholder for the rework_workspace/ handover
  scripts/
    r1_build_database.py    D5_rework_database.csv -> step15_human.csv-equivalent
                             schema, filtered to final_include == yes (115 rows) +
                             step15b_summary.json-equivalent, into outputs/step15/
    r2_run_figures.py       calls scripts/step16_map_visualise.py's run() with
                             out_dir=rework/outputs (zero edits to that script needed —
                             it already takes out_dir via config) to regenerate every
                             human-track figure against the new data
    r3_gen_prisma_rework.py parameterized copy of deliverables/_gen_prisma.py pointed
                             at rework/outputs, with a PRELIMINARY banner
    r4_diff_report.py       before/after PRISMA numbers + per-field category shift
                             tables (reusing Forrest's vs_prior_comparison /
                             record_summary) -> outputs/diffs/impact_memo.md
    r5_diff_figures.py      visual diffs: side-by-side before/after montage per
                             figure, + a self-verified field-value-shift bar chart
                             (computed directly from both step15_human.csv files,
                             not cited — see its docstring) -> outputs/diffs/
    r6_build_pdf.py         assembles outputs/diffs/impact_memo.pdf: the memo's
                             text/tables + every diff visual, one PDF, no external
                             deps (matplotlib PdfPages — pandoc had no PDF engine
                             and weasyprint's system libs weren't installed here)
    r7_collect_d5_figures.py   copies the exact 9 figures publish_d5_from_d4.py
                             embeds (Figures 1-9) into one flat folder, named and
                             ready to drop straight back into the live paths —
                             see outputs/d5_figures/ below
  outputs/
    step15/                 new step15_human.csv (115 rows) + step15b_summary.json
    step16/                 new figures, same filenames as scripts/outputs/step16/ —
                             diff is just a folder-vs-folder compare
    diffs/
      impact_memo.md        the human-readable before/after writeup
      impact_memo.pdf       same content + all diff visuals, one file to share
      prisma_side_by_side.png, field_shifts.png, side_by_side/*.png
    d5_figures/             all 9 figures that go into the D5 docx, flat, named
                             exactly as deliverables/publish_d5_from_d4.py expects
                             (prisma_flow_d5.png, evidence_gap_map.png,
                             geographic_map.png, geographic_bar.png,
                             methodology.png, temporal_trends.png, equity.png,
                             saturation.png, llm_vs_human.png) — 8 are the rework
                             versions, saturation.png is carried over unchanged
                             (it reflects RA coding-batch saturation, not
                             eligibility, so the rework doesn't touch it). Once
                             the numbers are approved, copy each file over its
                             original path (r7's printed manifest has the exact
                             paths) and re-run publish_d5_from_d4.py.
```

Run order: `r1_build_database.py` → `r2_run_figures.py` → `r2b_export_pngs.py` →
`r3_gen_prisma_rework.py` → `r4_diff_report.py` → `r5_diff_figures.py` →
`r6_build_pdf.py` → `r7_collect_d5_figures.py`.

Join key between old and new data is **DOI** (title as fallback for the handful of
records without one) — Forrest's `D5_rework_database.csv` carries both `doi` and
`title` alongside its internal `ILRI01-FT-XXX` ID, which the original pipeline's
`step15_human.csv` does not use.
