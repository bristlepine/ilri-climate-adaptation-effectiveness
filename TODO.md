# Deliverables Tracker — Climate Adaptation Effectiveness Systematic Map

**Contract:** ILRI / Bristlepine consulting
**Last updated:** 2026-04-09

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

Everything below must complete for D5 to ship on time. 21 days remaining.

| Owner | Task | Unlocks | Due |
|---|---|---|---|
| **Zarrar** | Generate FT calibration record set (step14a) and send to Caroline + Jennifer | Caroline + Jennifer FT screening → D5.5 | ASAP |
| **Jennifer** | Retrieve missing full texts via Cornell campus access — run step13 from campus network (or VPN); share zip of outputs with Zarrar | D5.5 full-text screening | ASAP |
| **Zarrar** | Submit D4 draft — GitHub release + Zenodo DOI, labelled preliminary | — | 14 Apr |
| **Zarrar** | Run WoS, CAB Abstracts, AGRIS, Academic Search Premier queries (D5.2) | D5.4 abstract screening of net-new records | 22 Apr |
| **Caroline** | Screen ~100 FT calibration records independently on receipt; return filled CSV with decisions and notes | D5.1 IRR analysis → D5.5 | 22 Apr |
| **Jennifer** | Screen ~100 FT calibration records independently on receipt; return filled CSV with decisions and notes | D5.1 IRR analysis → D5.5 | 22 Apr |
| **Colleagues** | Complete grey literature manual search — ~20 repositories (CGIAR, World Bank, 3ie, GCF, FAO, IFAD, regional development banks) per D3 §3.3 | D5.4 | 25 Apr |
| **Caroline** | Spot-check ~10% of data extraction outputs; resolve discrepancies with Jennifer | D5.6 → D5.8 | 28 Apr |
| **Jennifer** | Spot-check ~10% of data extraction outputs; resolve discrepancies with Caroline | D5.6 → D5.8 | 28 Apr |
| **Zarrar** | Draft protocol amendment v2 — document all D3 deviations, notify co-authors, publish to Zenodo | D5.7 → D5.8 | 28 Apr |
| **Zarrar** | Assemble and submit D5 deliverable document | D5 shipped | 1 May |

---

## Active Deliverables

| ID | Description | Type | Output | Revised due | Owner | Status |
|---|---|---|---|---|---|---|
| **D4** | **First draft systematic map — searchable database + evidence gap map (Scopus-based, preliminary)** | Report + Database | ILRI folder + GitHub + Zenodo DOI | ~~3 Apr~~ **14 Apr 2026** | Zarrar | ⚠ Overdue |
| D4.2 | ROSES flow diagram — Scopus-based, labelled preliminary pending multi-database integration | Internal | `scripts/outputs/step16/` | 12 Apr 2026 | Zarrar | 🔄 In progress |
| D4.3 | Preliminary searchable database — all Scopus-included records with title, abstract, DOI, year, country, screening decision | Internal | `scripts/outputs/step16/` | 12 Apr 2026 | Zarrar | 🔄 In progress |
| D4.4 | Submit D4 draft to ILRI — GitHub release + Zenodo DOI, clearly labelled preliminary | Internal | ILRI folder + Zenodo | **14 Apr 2026** | Zarrar | — Not started |
| **D5** | **Final systematic map — searchable database + evidence gap map, multi-database, published** | Report + Database | ILRI folder + GitHub + Zenodo DOI → CGSpace | **1 May 2026** | All | — Not started |
| D5.1 | Full-text calibration — draw ~100 records from step12 INCLUDEs with retrieved full texts; dual human screen (Caroline, Jennifer); LLM calibrated against reconciled gold standard; IRR ≥ 0.95 sensitivity | Internal | `scripts/outputs/step14b/` | 22 Apr 2026 | Caroline, Jennifer, Zarrar | — Not started |
| D5.2 | Multi-database search — WoS Core Collection, CAB Abstracts, AGRIS, Academic Search Premier; search strings adapted per syntax; all hit counts and dates documented | Internal | `scripts/outputs/step2b/` | 22 Apr 2026 | Zarrar | 🔄 In progress |
| D5.3 | Grey literature manual search — ~20 repositories per D3 §3.3 (CGIAR, World Bank, 3ie, GCF, FAO, IFAD, regional development banks) | Internal | `scripts/data/grey_literature/` | 25 Apr 2026 | Colleagues | — Not started |
| D5.4 | Abstract screening — net-new records from additional databases; validated R2b/R3a criteria applied; deduplication against Scopus corpus | Internal | `scripts/outputs/step12/` | 28 Apr 2026 | Pipeline | ⏳ Pending D5.2–D5.3 |
| D5.5 | Full-text screening — pipeline run post full-text calibration confirmation (D5.1); supplemented with manual campus library collection | Internal | `scripts/outputs/step14/` | 28 Apr 2026 | Pipeline | ⏳ Pending D5.1 |
| D5.6 | Data extraction — automated coding of all included records; ~10% random spot-check by Caroline and Jennifer; κ ≥ 0.60; all discrepancies resolved | Internal | `scripts/outputs/step15/` | 28 Apr 2026 | Caroline, Jennifer, Zarrar | ⏳ Pending D5.5 |
| D5.7 | Protocol amendment v2 — Zenodo versioned update documenting all D3 deviations; all co-authors notified | Internal | Zenodo (existing DOI, v2) | 28 Apr 2026 | Zarrar | — Not started |
| D5.8 | Final systematic map — updated ROSES flow diagram, full searchable extraction database, evidence gap map | Internal | `scripts/outputs/step16/` | **1 May 2026** | All | ⏳ Pending D5.1–D5.6 |
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
| D4.1 | Full-corpus abstract screening — Scopus (~17,021 records), sensitivity 0.966/0.970 | Apr 2026 | ✓ Done |

---

## Immediate priorities (week of 9 April 2026)

1. **D4.2 / D4.3** — Generate ROSES flow diagram and preliminary searchable database from Scopus screening results → target 12 Apr
2. **D4.4** — Submit D4 draft to ILRI → target 14 Apr
3. **D5.1** — Begin full-text calibration design; send record set to Caroline and Jennifer → target start 14 Apr
4. **D5.2** — WoS, CAB, AGRIS, Academic Search Premier queries → running in parallel

---

## Implementation notes

### D5.1 — Full-text calibration pipeline

Scripts to build (insert between step13 and current step14):

- `step14a_generate_ft_calibration.py` — draw ~100 records from step12 INCLUDEs with full texts retrieved; output blank `scripts/results/EPPI Review - FT-R1a.csv` (cols: Id, Item, DOI, Abstract, Caroline Staub, Jennifer Cisse, Reconciled, Reconciliation Notes) + rendered criteria guidance .md to send to reviewers
- `step14b_ft_irr_analysis.py` — after reviewers return filled CSV: run LLM on full texts → fill LLM column → compute IRR (reuse step11 functions: `cohen_kappa`, `confusion_vs_reconciled`, `detect_columns`); output figures + JSON to `outputs/step14b/`
- `step14c_test_ft_criteria.py` — after revising criteria: re-run LLM on calibration set → compare to Reconciled → print false negatives with quotes to guide next revision
- `scripts/criteria_ft_r1.yml` — same 5 criteria as `criteria.yml`, same existing abstract guidance fields, plus two new fields per criterion: `ft_include_further_guidelines` / `ft_exclude_further_guidelines` (start minimal; FT stage can be stricter than abstract)

Renaming required once step14a/b/c inserted: existing step14→15, step15→16, step16→17 across script files, `config.py`, `run.py`, `METHODOLOGY.md`, `methodology.py`, `README.md`, output dirs.

Extend `step11_criteria_evolution.py` with a `ROUNDS_FT` list to track FT criteria changes separately.

### Pipeline — pending re-runs

- Re-run step9/step9a once Elsevier token is stable to recover remaining ~1,314 missing abstracts
- Re-run step12 (full-corpus screening) once multi-database records are integrated
