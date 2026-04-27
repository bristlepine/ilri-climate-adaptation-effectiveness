# Deliverables Tracker

Cross-reference with [TODO.md](../TODO.md) for full task breakdown.
Last updated: 2026-04-27

---

## Status Key

| Symbol | Meaning |
|---|---|
| ✓ | Complete and submitted/published |
| 🔄 | In progress |
| ⚠ | Overdue |
| ⏳ | Pending upstream task |
| — | Not started |

---

## Deliverables

| ID | Title | Due | Status | Google Doc | Zenodo | GitHub |
|---|---|---|---|---|---|---|
| **D1** | Inception Report | Nov 2025 | ✓ | — | — | — |
| **D2** | Draft Systematic Map Protocol | Dec 2025 | ✓ | [v3 (GDoc)](https://docs.google.com/document/d/1uMRf7ZN2yzOusklRAPAwnou16hPH4A1SHzkH1qC3keA/edit) | — | — |
| **D3** | Final Systematic Map Protocol + Protocol Amendment (D5.7) | Jan 2026 | ✓ | [v1 (GDoc)](https://docs.google.com/document/d/1XN0YdGPnOBEMVLxvekGQ-ztYngOQkx84kn7r2v0Y2qU/edit) · [v02 (GDoc)](https://docs.google.com/document/d/1yLnB2b--XOtrMSQ1ekgBu6XKGcDWQ1Ggvn7mKQqjaqg/edit) · [v03/D5.7 (GDoc)](https://docs.google.com/document/d/1Jqhy5LYm2FniFaTuQPkA8HXKZ1C4sSPMyGed0efGas4/edit) | [10.5281/zenodo.19811629](https://zenodo.org/records/19811629) | — |
| **D4** | First Draft Systematic Map (preliminary) | 14 Apr 2026 | ✓ | [v01 (GDoc)](https://docs.google.com/document/d/1i2dQUoXuNPwjvV_fT5CMiy0fvADy8oWCGK3sdHGApCs/edit) · [v02 (GDoc)](https://docs.google.com/document/d/14Hejjfr63HdxH21bqBxLZe3AsXxmLBrXYqJJk7ZbBxU/edit) | [10.5281/zenodo.19811622](https://zenodo.org/records/19811622) | — |
| **D5** | Final Systematic Map | 1 May 2026 | ⏳ | — | — | — |
| **D6** | Draft SR/Meta-analysis Protocol | 15 May 2026 | ⏳ | — | — | — |
| **D7** | Final SR/Meta-analysis Protocol | 29 May 2026 | ⏳ | — | — | — |
| **D8** | Draft SR/Meta-analysis | 26 Jun 2026 | ⏳ | — | — | — |
| **D9** | Final SR/Meta-analysis | 31 Jul 2026 | ⏳ | — | — | — |
| **D10** | PowerPoint — key findings | 31 Jul 2026 | ⏳ | — | — | — |

---

## D4 — First Draft Systematic Map (Preliminary)

**Status:** Published on Zenodo. [DOI: 10.5281/zenodo.19811622](https://zenodo.org/records/19811622)

| Step | Task | Status |
|---|---|---|
| 1 | Create Google Doc (from D3 template) | ✓ [v01](https://docs.google.com/document/d/1i2dQUoXuNPwjvV_fT5CMiy0fvADy8oWCGK3sdHGApCs/edit) |
| 2 | Review and edit doc | ✓ |
| 3 | Export PDF → `deliverables/` | ✓ `Deliverable 4_Bristlepine_First Draft Systematic Map_v02.pdf` |
| 4 | Publish to Zenodo | ✓ [10.5281/zenodo.19811622](https://zenodo.org/records/19811622) |
| 5 | Create GitHub release (`gh release create`) | — |
| 6 | Submit link to ILRI | — |

**Pipeline stats (as of 2026-04-27, all databases):**
- Total records across 5 databases: 39,113
- After deduplication: 25,208 unique
- Abstract screening included: 8,555
- Full texts retrieved: 3,476
- Coded records (LLM-assisted): 2,750

---

## D5 — Final Systematic Map

**Status:** Pipeline in progress. All tasks blocking D5 are tracked in TODO.md.

| Step | Task | Owner | Due | Status |
|---|---|---|---|---|
| D5.2 | Multi-DB search (WoS, CAB, AGRIS, ASP) | Zarrar | 22 Apr | ✓ |
| D5.3 | Grey literature search (~20 repos) | Colleagues | 25 Apr | — |
| D5.4 | Abstract screening net-new | Pipeline | 28 Apr | ✓ (step12b done) |
| D5.5 | Full-text retrieval + screening | Pipeline | 28 Apr | 🔄 step13/14 done; missing papers with Jenn |
| D5.6 | Data extraction calibration | Jennifer + Caroline | 28 Apr | ⏳ FT-R1a in progress |
| D5.7 | Protocol amendment write-up | Zarrar | 28 Apr | 🔄 D3 v02 GDoc ready for review |
| D5.8 | Final map + visualisations | Pipeline | 1 May | ⏳ |

---

## Scripts

| Script | Purpose | Location |
|---|---|---|
| `gdocs_auth.py` | One-time Google OAuth | `deliverables/` |
| `gdocs_create_d3v02.py` | Create D3 v02 (protocol amendment) Google Doc | `deliverables/` |
| `gdocs_create_d4.py` | Create D4 Google Doc from D3 template | `deliverables/` |
| `gdocs_review_comments.py` | Read, reply to, and resolve comments | `deliverables/` |
| `zenodo_publish.py` | Publish PDF to Zenodo, mint DOI | `deliverables/` |

---

## Publishing Commands

```bash
# Export PDF first (Google Doc → File → Download → PDF)
# Then:

# Create D3 v02 Google Doc (protocol amendment)
conda run -n ilri01 python deliverables/gdocs_create_d3v02.py

# Publish D3 v02 to Zenodo as new version of existing D3 record (ID: 18370029)
conda run -n ilri01 python deliverables/zenodo_publish.py \
  --file deliverables/D3_v02.pdf \
  --title "Deliverable 3: Final Systematic Map Protocol (v02, amended) — Measuring what matters: tracking climate adaptation processes and outcomes for smallholder producers in the agriculture sector" \
  --version v02 \
  --new-version 18370029 \
  --confirm

# Publish D4 to Zenodo
conda run -n ilri01 python deliverables/zenodo_publish.py \
  --file deliverables/D4_v01.pdf \
  --title "Deliverable 4: First Draft Systematic Map — Measuring what matters: tracking climate adaptation processes and outcomes for smallholder producers in the agriculture sector (Preliminary)" \
  --version v01 \
  --confirm

# Create GitHub release
gh release create d4-v01 "deliverables/D4_v01.pdf" \
  --title "D4: First Draft Systematic Map (Preliminary)" \
  --notes "First draft systematic map, Scopus-based, clearly labelled preliminary. DOI: https://doi.org/10.XXXX/zenodo.XXXXXXX"

# Check comments on D4
conda run -n ilri01 python deliverables/gdocs_review_comments.py \
  --doc-id 1i2dQUoXuNPwjvV_fT5CMiy0fvADy8oWCGK3sdHGApCs --md
```
