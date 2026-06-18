# CLAUDE.md — ILRI Climate Adaptation Effectiveness

> **Git rule:** Never `git push` without explicit confirmation from the user.

Bristlepine / ILRI systematic map on methods for tracking climate adaptation for smallholder producers in LMICs. CEE/Campbell standards.

**Git commits:** Never include `Co-Authored-By: Claude` or any AI attribution. Clean messages only.

**Conda env:** `ilri01`  
**Credentials:** `deliverables/.credentials/token.json` (Google Drive/Sheets OAuth)  
**Run all scripts from repo root** with `conda run -n ilri01 python scripts/<script>.py`

---

## Key paths

| What | Where |
|------|-------|
| All scripts | `scripts/` |
| Step outputs | `scripts/outputs/step{N}/` |
| Human coding rounds | `scripts/outputs/step14b/` |
| Round assignments | `scripts/outputs/step14b/assignments.csv` |
| Coding templates | `scripts/outputs/step14b/FT-RX/coding_ft-rX_template.csv` |
| Missing papers list | `scripts/outputs/step14b/FT-RX/papers_ft-rX_missing.csv` |
| Codebook (PDF) | `documentation/coding/systematic-map/rounds/CODEBOOK_FT.pdf` |
| Guidance notes | `documentation/coding/systematic-map/guidance.md` |
| Google Drive root | [Coding rounds folder](https://drive.google.com/drive/folders/13p22XfvB6sNtTtnMS-dkI1t-joMn-6Bo) |

---

## Human coding round workflow

Rounds are named `FT-R3`, `FT-R4`, ... (R1=calibration, R2x=legacy). Each round = 20 papers assigned to one coder.

### Creating new rounds
```bash
conda run -n ilri01 python scripts/step14b_batch_draw.py --rounds N
# dry-run first:
conda run -n ilri01 python scripts/step14b_batch_draw.py --rounds N --dry-run
# then push dry-run batches to Drive:
conda run -n ilri01 python scripts/step14b_batch_draw.py --push
```

### When Jenn has updated missing papers (the usual cycle)

Jenn manually procures PDFs for papers that couldn't be auto-retrieved. She uploads them to the Drive round folder and fills in the `status` column on the `papers_ft-rX_missing` Google Sheet with:
- `done` — PDF procured and uploaded to Drive
- `exclude` — paper should be excluded (exact match)
- `exclude - <reason>` — excluded with reason (e.g. "exclude - conference not a paper")
- `NA` — could not retrieve

**Step 1 — Pull each round Jenn updated:**
```bash
conda run -n ilri01 python scripts/step14b_batch_draw.py --pull FT-R5
conda run -n ilri01 python scripts/step14b_batch_draw.py --pull FT-R6
conda run -n ilri01 python scripts/step14b_batch_draw.py --pull FT-R7
```
This: downloads procured PDFs from Drive, syncs Jenn's status updates into the local missing CSV, and updates the coding template with a `procurement_status` column. Papers Jenn excluded get `confirmed_include=no` pre-filled.

**Step 2 — Review the updated templates:**
Check `procurement_status` summary printed by the script. Verify excludes look right. Flag any papers still `missing` that may need follow-up with Jenn.

**Step 3 — Assign coders:**
Edit `scripts/outputs/step14b/assignments.csv` — fill in `coder_name`, `coder_email` for each round. Coders seen so far:

| Name | Email |
|------|-------|
| ambar | ambzar37@gmail.com |
| samar | samarzahrachaudhary@gmail.com |

**Step 4 — Sync assignments to Drive:**
```bash
conda run -n ilri01 python scripts/step14b_batch_draw.py --sync-assignments
```

### Other useful flags
```bash
--fix-instructions   # Re-upload instruction PDFs for all pushed rounds (e.g. after codebook update)
--dedupe             # Trash duplicate PDFs in Drive pdfs/ folders
```

---

## Procurement status values (in coding template)

| Value | Meaning |
|-------|---------|
| *(empty)* | Had PDF from the start — no procurement needed |
| `done` | Jenn procured and uploaded the PDF |
| `missing` | Still no PDF — needs follow-up |
| `exclude` | Excluded during procurement (exact match in Jenn's sheet) |
| `Skip - exclude - <reason>` | Excluded with reason |
| `Skip - NA` | Could not retrieve |
| `Skip - book` | Full book, no chapter PDF available |

Papers with `Skip-*` or `exclude` get `confirmed_include=no` and a note pre-filled by the script.

---

## Known issues / watch-outs

- Papers with **no DOI** in `papers_ft-rX_missing.csv` won't have their `procurement_status` updated even if Jenn marked them — the script matches by DOI. Check these manually if Jenn marks them as excluded.
- `assignments.csv` has a `status` and `completed_date` column (beyond the script's fieldnames) — preserve these manually when editing.

---

> **Git rule:** Never `git push` without explicit confirmation from the user.

---

## Figure pipeline (step16 → deliverables)

### How figures are generated

All figures derive from the coded dataset (`scripts/outputs/step15/step15_coded.csv` + `step15_human.csv`). Run in order:

```bash
# 1. Regenerate all figures (LLM + human + compare JSONs + LLM PNGs)
conda run -n ilri01 python scripts/step16_map_visualise.py

# 2. Export PNGs for human and compare tracks (from their Plotly JSONs)
conda run -n ilri01 python scripts/step16b_export_pngs.py
```

### Figure folder structure

| Folder | Contents | Use |
|--------|----------|-----|
| `scripts/outputs/step16/*.png` | LLM static PNGs (matplotlib) | LLM reference |
| `scripts/outputs/step16/interactive/human/*.png` | Human Plotly PNGs (n=86) | **PRIMARY — use in deliverables** |
| `scripts/outputs/step16/interactive/compare/*.png` | Human vs LLM Plotly PNGs | Comparison / appendix |
| `scripts/outputs/step16/interactive/human/*.json` | Human Plotly JSONs | Frontend interactive |
| `scripts/outputs/step16/interactive/compare/*.json` | Compare Plotly JSONs | Frontend interactive |

**Always use `interactive/human/` PNGs in deliverables.** These match the frontend interactive figures exactly (same Plotly JSON source). The LLM matplotlib PNGs in `step16/` look different (matplotlib style, not Plotly).

### Canonical label mappings (in step16_map_visualise.py)

Human figures apply canonical labels via:
- `PRODUCER_LABELS` — maps `crop` → `Crop farmers`, `livestock` → `Livestock`, etc.
- `DOMAIN_LABELS` — maps `decision_making_planning` → `Decision-making & Planning`, etc.
- `_METHOD_LABELS` — maps `quantitative` → `Quantitative`, `interviews` → `Qualitative`, etc.

If a new coded value isn't appearing correctly in figures, add it to the relevant dict in `step16_map_visualise.py` (around lines 118–580).

### PRISMA flow diagram

The PRISMA box diagram (dual-track: human primary + LLM exploratory) is at:
- `deliverables/prisma_flow_d5.png` — generated by `deliverables/_gen_prisma.py`
- Regenerate: `conda run -n ilri01 python deliverables/_gen_prisma.py`
- Key numbers: 40,653 identified → 26,182 after dedup → 8,748 FTs sought → 86 human included (PRIMARY) + 2,368 LLM included (exploratory)

---

## Deliverables

### Deliverable numbering

| Our ID | ILRI folder name | Status |
|--------|-----------------|--------|
| D1 | Deliverable 1 — Inception Report | ✓ Submitted |
| D2 | Deliverable 2 — Draft Protocol | ✓ Submitted |
| D3 | Deliverable 3 — Final Protocol | ✓ Submitted (Zenodo: 10.5281/zenodo.19811629) |
| D4 | Deliverable 4 — First Draft Systematic Map | ✓ Submitted (Zenodo: 10.5281/zenodo.19811622) |
| D5 | **Deliverable 3 — EGM Report** (ILRI's numbering) | 🔄 In progress |
| D6 | Deliverable 4 — SR Protocol Draft | ⏳ After D5 |

Note: ILRI's Teams folder numbering differs from our internal numbering. Their "Deliverable 3" = our D5.

### D5 — Final Systematic Map (current)

**Approach:** Download D4 GDoc as Word (preserves watermark/styling) → modify programmatically → embed human figures.

**Script:** `deliverables/publish_d5_from_d4.py`

```bash
python3 deliverables/publish_d5_from_d4.py
# Output: deliverables/Deliverable_5_Bristlepine_Final_Systematic_Map_v01.docx
```

**What it does:**
1. Downloads D4 v02 from Google Drive as .docx (ID: `14Hejjfr63HdxH21bqBxLZe3AsXxmLBrXYqJJk7ZbBxU`)
2. Text replacements: "First Draft" → "Final", removes "Preliminary", updates dates/numbers
3. Replaces pipeline table (clean new table, no doubled headers)
4. Inserts human figures inline: PRISMA (Fig 1), EGM (Fig 2), geo (Figs 3–4), methodology (Fig 5), temporal (Fig 6), equity (Fig 7), saturation (Fig 8), LLM vs human (Fig 9)

**Figure sources for D5:**

| Figure | File |
|--------|------|
| PRISMA flow | `deliverables/prisma_flow_d5.png` |
| Evidence Gap Map | `step16/interactive/human/evidence_gap_map.png` |
| Geographic map | `step16/interactive/human/geographic_map.png` |
| Geographic bar | `step16/interactive/human/geographic_bar.png` |
| Methodology | `step16/interactive/human/methodology.png` |
| Temporal trends | `step16/interactive/human/temporal_trends.png` |
| Equity | `step16/interactive/human/equity.png` |
| Saturation | `step16/saturation.png` |
| LLM vs Human | `step16/llm_vs_human.png` |

**Google Drive deliverables folder:** `1alv6Ic7vtKZc_xo6AdlKP1exshMh-fR6`  
**D4 v02 GDoc ID:** `14Hejjfr63HdxH21bqBxLZe3AsXxmLBrXYqJJk7ZbBxU`  
**D5 GDoc ID (broken rebuild, ignore):** `14UaK_9lXCuE_AsOYXNvfImYlEndLUL4AM0lVz4thEAY`

**To submit:** Upload `.docx` + `deliverables/evidence_gap_map_d5.html` to Teams → "Deliverable 3 — EGM Report (draft folder)", tag @Aditi and @Neal.

### Full rebuild sequence (after new coding rounds)

```bash
# 1. Stack human batches
conda run -n ilri01 python scripts/step15b_stack_human_batches.py

# 2. Update saturation curve
conda run -n ilri01 python scripts/step15c_saturation.py

# 3. Regenerate all figures
conda run -n ilri01 python scripts/step16_map_visualise.py
conda run -n ilri01 python scripts/step16b_export_pngs.py

# 4. Rebuild D5 Word doc
python3 deliverables/publish_d5_from_d4.py
```
