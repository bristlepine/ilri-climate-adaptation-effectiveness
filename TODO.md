# Deliverables Tracker — Climate Adaptation Effectiveness Systematic Map

**Contract:** ILRI / Bristlepine consulting
**Last updated:** 2026-04-28

---

## Legend

| Symbol | Meaning |
|---|---|
| ✓ Submitted | Delivered and accepted |
| ✓ Done | Internal task complete |
| 🔄 In progress | Actively under way |
| ⏳ Pending | Waiting on upstream task |
| — Not started | Not yet begun |
| ⚠ Overdue | Past revised target |

---

## Critical — next binding deliverable: D5 Final Systematic Map (1 May 2026)

**3 days remaining.** D4 ✓ submitted 27 Apr. D3 amendment ✓ submitted 27 Apr.

| Owner | Task | Unlocks | Due | Status |
|---|---|---|---|---|
| **Jennifer** | Retrieve missing full texts via Cornell campus access | D5.5 | ASAP | ⏳ Pending |
| **Zarrar** | Submit D4 — GitHub release + Zenodo DOI | — | 14 Apr | ✓ Done 27 Apr |
| **Zarrar** | Run WoS, CAB, AGRIS, ASP queries | D5.4 | 22 Apr | ✓ Done |
| **Colleagues** | Grey literature manual search ~20 repos | D5.4 | 25 Apr | ⏳ Pending |
| **Zarrar** | Protocol amendment v2 — Zenodo update | D5.8 | 28 Apr | ✓ Done 27 Apr |
| **Zarrar** | Build step15b, step15c, step16 toggle | D5.8 | 30 Apr | 🔄 In progress |
| **Zarrar** | Assemble and submit D5 deliverable document | D5 shipped | 1 May | — Not started |

---

## Active Deliverables

| ID | Description | Type | Output | Revised due | Owner | Status |
|---|---|---|---|---|---|---|
| **D4** | **First draft systematic map — searchable database + evidence gap map (Scopus-based, preliminary)** | Report + Database | ILRI folder + GitHub + Zenodo DOI | ~~3 Apr~~ **27 Apr 2026** | Zarrar | ✓ Submitted 27 Apr 2026 |
| D4.2 | ROSES flow diagram — Scopus-based, labelled preliminary pending multi-database integration | Internal | `scripts/outputs/step16/roses_flow.png` | 12 Apr 2026 | Zarrar | ✓ Done — auto-regenerates on each step16 re-run as pipeline progresses |
| D4.3 | Preliminary searchable database — step15_coded.csv served via frontend at /systematic-map; CSV download live; interactive table pending D5.6 | Internal | `frontend/public/map/step15_coded.csv` | 12 Apr 2026 | Zarrar | ✓ Done — preliminary CSV live on site |
| D4.4 | Submit D4 draft to ILRI — GitHub release + Zenodo DOI, clearly labelled preliminary | Internal | ILRI folder + Zenodo | **27 Apr 2026** | Zarrar | ✓ Done 27 Apr 2026 |
| **D5** | **Final systematic map — searchable database + evidence gap map, multi-database, published** | Report + Database | ILRI folder + GitHub + Zenodo DOI → CGSpace | **1 May 2026** | All | 🔄 In progress — 3 days remaining |
| D5.1 | ~~Full-text calibration~~ — **Dropped 11 Apr 2026** due to timeline constraints; Caroline and Jennifer notified; D3 protocol does not explicitly require FT-stage calibration (only abstract stage); to be noted in protocol amendment D5.7 | Internal | — | — | — | ✗ Abandoned |
| D5.2 | Multi-database search — WoS Core Collection, CAB Abstracts, AGRIS, Academic Search Premier; adapted strings ready in `scripts/data/multidatabase/`; manual export via Cornell Library login (no VPN needed); RIS files → step2b_multidatabase_dedupe.py | Internal | `scripts/outputs/step2b/` | 22 Apr 2026 | Zarrar | 🔄 In progress — strings ready, exports pending |
| D5.3 | Grey literature manual search — ~20 repositories per D3 §3.3 (CGIAR, World Bank, 3ie, GCF, FAO, IFAD, regional development banks) | Internal | `scripts/data/grey_literature/` | 25 Apr 2026 | Colleagues | — Not started |
| D5.4 | Abstract screening — net-new records from additional databases; validated R2b/R3a criteria applied; deduplication against Scopus corpus | Internal | `scripts/outputs/step12/` | 28 Apr 2026 | Pipeline | ⏳ Pending D5.2–D5.3 |
| D5.5 | Full-text screening — pipeline run on all retrieved full texts; supplemented with manual campus library collection | Internal | `scripts/outputs/step14/` | 28 Apr 2026 | Pipeline | ⏳ Pending — retrieval at 64.4% (4,002/6,218); step13_missing_papers_01.csv ready for next campus pass; step14 can start |
| D5.6 | Data extraction — stratified random sample of retrieved papers given to human coders for combined FT screening + data extraction (5–6 batches of 100, one coder per batch); coders confirm inclusion and extract data; false positives from LLM FT screening documented and removed; figures built from human-confirmed data (primary); LLM full-corpus extraction shown as toggle overlay; saturation curve demonstrates information saturation; randomization stratified by geography and documented in frontend | Internal | `scripts/outputs/step15/step15_human.csv` | 28 Apr 2026 | Coders + Zarrar | 🔄 In progress — FT-R2a (batch 1, 100 papers) assigned |
| D5.7 | Protocol amendment v2 — Zenodo versioned update documenting all D3 deviations; all co-authors notified | Internal | Zenodo (existing DOI, v2) | 28 Apr 2026 | Zarrar | ✓ Done 27 Apr 2026 |
| D5.8 | Final systematic map — updated ROSES flow diagram, searchable extraction database (human-coded primary, LLM toggle), evidence gap map, saturation curve | Internal | `scripts/outputs/step16/` + frontend | **1 May 2026** | All | ⏳ Pending D5.2–D5.6 |
| **D6** | **First draft SR/meta-analysis protocol** | Report | ILRI folder + GitHub + Zenodo DOI | **15 May 2026** | Zarrar | — Not started |
| D6.1 | Draft protocol informed by systematic map findings — scope, RQs, inclusion criteria, analysis plan | Internal | `documentation/` | 15 May 2026 | Zarrar | ⏳ Pending D5 |
| **D7** | **Final SR/meta-analysis protocol — published on Zenodo** | Report | ILRI folder + GitHub + Zenodo DOI → CGSpace | **29 May 2026** | Zarrar + team | — Not started |
| D7.1 | Final SR/meta-analysis protocol + Zenodo DOI; protocol amendment v2 submitted concurrently | Internal | Zenodo | 29 May 2026 | Zarrar + team | ⏳ Pending D6 |
| **D8** | **First draft SR/meta-analysis ready for journal submission** | Journal paper | ILRI folder + GitHub + Zenodo DOI | **26 Jun 2026** | All | — Not started |
| D8.1 | Effect size extraction from all included studies; data quality checks | Internal | `scripts/outputs/step15/` | 12 Jun 2026 | Zarrar | ⏳ Pending D7 |
| D8.2 | Meta-analysis and evidence synthesis | Internal | `scripts/outputs/` | 19 Jun 2026 | Zarrar | ⏳ Pending D8.1 |
| D8.3 | Draft SR/meta-analysis manuscript — introduction, methods, results, discussion | Internal | `documentation/` | 26 Jun 2026 | All | ⏳ Pending D8.2 |
| **D9** | **Final SR/meta-analysis — journal-ready** | Journal paper | ILRI folder + GitHub + Zenodo DOI | **31 Jul 2026** | All | — Not started |
| D9.1 | Final revision incorporating reviewer feedback; journal submission | Internal | Journal submission | 31 Jul 2026 | All | ⏳ Pending D8 |
| **D10** | **PowerPoint — all outputs and key findings for lay audience** | Presentation | ILRI folder + GitHub + Zenodo DOI | **31 Jul 2026** | Zarrar | — Not started |
| D10.1 | PowerPoint summarising protocols, systematic map, SR/meta-analysis, key findings — ILRI format | Internal | ILRI folder | 31 Jul 2026 | Zarrar | ⏳ Pending D9 |

---

## Completed

| ID | Description | Delivered | Status |
|---|---|---|---|
| **D1** | **Inception Report** | 26 Nov 2025 | ✓ Submitted |
| D1.1 | Final research questions and PCCM framework | 26 Nov 2025 | ✓ Done |
| D1.2 | Search string design (Scopus) + volume estimate | 26 Nov 2025 | ✓ Done |
| D1.3 | Gantt chart and project timeline | 26 Nov 2025 | ✓ Done |
| **D2** | **First draft scoping protocol** | 31 Dec 2025 | ✓ Submitted |
| D2.1 | Draft PCCM eligibility criteria | 31 Dec 2025 | ✓ Done |
| D2.2 | Draft methodology appendix | 31 Dec 2025 | ✓ Done |
| **D3** | **Final scoping protocol — published on Zenodo** | 30 Jan 2026 | ✓ Submitted |
| D3.1 | Final eligibility criteria (all 5 PCCM criteria) | 30 Jan 2026 | ✓ Done |
| D3.2 | Final methodology appendix (v1) | 30 Jan 2026 | ✓ Done |
| D3.3 | Zenodo v1 DOI release | 30 Jan 2026 | ✓ Done |
| D3 amendment | Protocol amendment v2 — D3 deviations documented, co-authors notified, Zenodo updated | 27 Apr 2026 | ✓ Done |
| D4.1 | Full-corpus abstract screening — Scopus (~17,021 records), sensitivity 0.966/0.970 | Apr 2026 | ✓ Done |
| **D4** | **First draft systematic map — searchable database + evidence gap map, preliminary** | 27 Apr 2026 | ✓ Submitted |

---

## Immediate priorities (week of 28 April 2026)

1. **D5.6 scripts** — Build `step15b_stack_human_batches.py`, `step15c_saturation.py`; update `setup_round.py` to log manifest; update `step16` with `--source` flag + frontend toggle
2. **FT-R2a** — Coder to complete 100-paper batch and upload CSV; Zarrar to run step15b + step15c once received
3. **D5.7** — Protocol amendment v2: document D5.6 design change (single-coder batches, saturation focus, no reconciliation), D5.1 abandonment, field schema change (equity_inclusion removed) → due 28 Apr
4. **D5** — Assemble and submit final deliverable document — due 1 May 2026

---

## Implementation notes

### D5.1 — Full-text calibration pipeline (ABANDONED 11 Apr 2026)

Dropped due to timeline constraints. D3 protocol does not mandate FT-stage calibration (§4.2 covers abstract stage only). Caroline and Jennifer notified. To be documented in protocol amendment D5.7 as a deviation with justification.

### Protocol amendment note — field changes (17 Apr 2026)

Field 16 (`equity_inclusion`) removed from the extraction schema. Its content is now captured under field 6 (`marginalized_subpopulations`), which has been extended with an `other` value (with free-text note in the `notes` column) to cover equity dimensions such as disability, caste, and religion that are not already named. This change should be documented in the protocol amendment (D5.7) as a post-protocol schema simplification, with justification that the two fields were substantially overlapping and the consolidated field retains full information.

Calibration round size reduced from 100 to 5 papers per round for the FT coding phase. To be documented in D5.7 as a scope adjustment.

### D5.6 — Data extraction design (updated 28 Apr 2026)

Human coders receive a **stratified random sample of retrieved papers** and perform **combined FT screening + data extraction** in a single pass. The LLM FT screening (step14) is an internal efficiency tool — it is not presented as the methodological gate. Human coders ARE the FT screen for their batch.

**Inclusion criteria integrity:**
- Abstract screening criteria (steps 10–12) remain unchanged and are not re-litigated here
- Coders apply the same 5 PCCM inclusion criteria to confirm each paper is a valid include
- A `confirmed_include` column (`yes` / `no`) is added to the template; a one-paragraph criteria reminder appears on the codebook cover page
- Papers marked `no` are removed from `step15_human.csv` and counted as LLM FT screening false positives; false positive rate reported in D5

**Stratified random sampling:**
- Pool = step15_coded.csv papers with a retrieved PDF, excluding calibration papers
- Strata = geography (SSA / South Asia / SE Asia / Latin America / Other) based on LLM's `country_region` field; allocation proportional to stratum size in the pool
- Each batch uses a unique seed (42, 43, 44, …); previously drawn DOIs excluded
- Every draw logged to `round_manifest.csv`: round, seed, strata sizes, allocation, population_n, draw_date
- Manifest shown in D5 and in frontend randomization panel — proves representativeness

**Batch workflow:**
1. `setup_round.py --round FT-R2a --sample 100 --seed 42` — stratified draw, uploads to Drive, logs manifest
2. Coder downloads template CSV, reads each paper, fills `confirmed_include` + all 16 fields
3. Zarrar downloads completed CSV → places in round folder locally
4. Run `step15b_stack_human_batches.py` → filters out `confirmed_include=no`, rebuilds `step15_human.csv`
5. Run `step15c_saturation.py` → updates saturation curve PNG
6. Run `step16_map_visualise.py --source both` → updates all figures + frontend toggle

**Information saturation:**
- `step15c_saturation.py` computes distribution of key fields (outcome domains, adaptation categories, geographic spread) at each 100-paper increment
- Measures change between successive batches (% new categories + distribution shift)
- Flattening = saturation; reported in D5 even if not fully reached (properly randomized sample is defensible regardless — per Neal)

**Scripts to build/update:**

| Script | Status | Change needed |
|--------|--------|---------------|
| `documentation/coding/systematic-map/rounds/setup_round.py` | ✓ Exists | Stratified sampling + manifest logging |
| Codebook (FT-R2a onwards) | ✓ Exists | Add `confirmed_include` field + criteria reminder |
| `scripts/step15b_stack_human_batches.py` | — Not built | Stack batches, filter confirmed_include=no |
| `scripts/step15c_saturation.py` | — Not built | Saturation curve per 100-paper increment |
| `scripts/step16_map_visualise.py` | ✓ Exists | `--source` flag + frontend toggle |

### Frontend — AWS/Next.js site (deployed, live)

Structure (updated 11 Apr 2026):
- `/` — Home (video hero, unchanged)
- `/about` — Project background, RQ, PCCM framework, two-stage approach, team
- `/deliverables` — All D1–D10 as cards with status badges and Zenodo/GitHub/CGSpace links
- `/systematic-map` — ROSES diagram, preliminary CSV download, evidence map figures (lightbox), interactive table placeholder
- `/systematic-review` — Skeleton, D6–D9 cards, coming May–Jul 2026

**To update the site after any pipeline run:** `python scripts/stepX_sync_frontend.py` copies step16 PNGs + step15_coded.csv → `frontend/public/map/`, then deploy (push to main triggers AWS).

Deliverable DOIs to fill in `frontend/src/lib/deliverables.ts` as they are minted:
- D3: Zenodo DOI pending
- D4: Zenodo DOI — mint on GitHub release (D4.4)
- D5–D10: pending

### Pipeline — pending re-runs

- Re-run step9/step9a once Elsevier token is stable to recover remaining ~1,314 missing abstracts
- Re-run step12 (full-corpus screening) once multi-database records are integrated
