# CLAUDE.md — ILRI Climate Adaptation Effectiveness

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
