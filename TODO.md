# Deliverables Tracker — Climate Adaptation Effectiveness Systematic Map

**Contract:** ILRI / Bristlepine consulting
**Last updated:** 2026-04-10

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
| **Jennifer** | Retrieve missing full texts via Cornell campus access — run step13 from campus network (or VPN); share zip of outputs with Zarrar | D5.5 full-text screening | ASAP |
| **Zarrar** | Submit D4 draft — GitHub release + Zenodo DOI, labelled preliminary | — | 14 Apr |
| **Zarrar** | Run WoS, CAB Abstracts, AGRIS, Academic Search Premier queries (D5.2) | D5.4 abstract screening of net-new records | 22 Apr |
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
| D4.2 | ROSES flow diagram — Scopus-based, labelled preliminary pending multi-database integration | Internal | `scripts/outputs/step16/roses_flow.png` | 12 Apr 2026 | Zarrar | ✓ Done — auto-regenerates on each step16 re-run as pipeline progresses |
| D4.3 | Preliminary searchable database — step15_coded.csv served via frontend at /systematic-map; CSV download live; interactive table pending D5.6 | Internal | `frontend/public/map/step15_coded.csv` | 12 Apr 2026 | Zarrar | ✓ Done — preliminary CSV live on site |
| D4.4 | Submit D4 draft to ILRI — GitHub release + Zenodo DOI, clearly labelled preliminary | Internal | ILRI folder + Zenodo | **14 Apr 2026** | Zarrar | — Not started |
| **D5** | **Final systematic map — searchable database + evidence gap map, multi-database, published** | Report + Database | ILRI folder + GitHub + Zenodo DOI → CGSpace | **1 May 2026** | All | — Not started |
| D5.1 | ~~Full-text calibration~~ — **Dropped 11 Apr 2026** due to timeline constraints; Caroline and Jennifer notified; D3 protocol does not explicitly require FT-stage calibration (only abstract stage); to be noted in protocol amendment D5.7 | Internal | — | — | — | ✗ Abandoned |
| D5.2 | Multi-database search — WoS Core Collection, CAB Abstracts, AGRIS, Academic Search Premier; adapted strings ready in `scripts/data/multidatabase/`; manual export via Cornell Library login (no VPN needed); RIS files → step2b_multidatabase_dedupe.py | Internal | `scripts/outputs/step2b/` | 22 Apr 2026 | Zarrar | 🔄 In progress — strings ready, exports pending |
| D5.3 | Grey literature manual search — ~20 repositories per D3 §3.3 (CGIAR, World Bank, 3ie, GCF, FAO, IFAD, regional development banks) | Internal | `scripts/data/grey_literature/` | 25 Apr 2026 | Colleagues | — Not started |
| D5.4 | Abstract screening — net-new records from additional databases; validated R2b/R3a criteria applied; deduplication against Scopus corpus | Internal | `scripts/outputs/step12/` | 28 Apr 2026 | Pipeline | ⏳ Pending D5.2–D5.3 |
| D5.5 | Full-text screening — pipeline run on all retrieved full texts; supplemented with manual campus library collection | Internal | `scripts/outputs/step14/` | 28 Apr 2026 | Pipeline | ⏳ Pending Jennifer FT retrieval |
| D5.6 | Data extraction — iterative human-LLM co-coding in 100-paper buckets; H1 + H2 + LLM code simultaneously; reconcile → update codebook + LLM guidance; kappa + sensitivity per closed field; category saturation for emergent fields (Adaptation focus, Indicators); converge in ~3–4 rounds; LLM runs full corpus with uncertainty flagging; flagged records reviewed by humans | Internal | `scripts/outputs/step15/` | 28 Apr 2026 | Coders + Zarrar | ⏳ Pending D5.5 |
| D5.7 | Protocol amendment v2 — Zenodo versioned update documenting all D3 deviations; all co-authors notified | Internal | Zenodo (existing DOI, v2) | 28 Apr 2026 | Zarrar | — Not started |
| D5.8 | Final systematic map — updated ROSES flow diagram, full searchable extraction database, evidence gap map | Internal | `scripts/outputs/step16/` + frontend | **1 May 2026** | All | ⏳ Pending D5.2–D5.6 |
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

### D5.1 — Full-text calibration pipeline (ABANDONED 11 Apr 2026)

Dropped due to timeline constraints. D3 protocol does not mandate FT-stage calibration (§4.2 covers abstract stage only). Caroline and Jennifer notified. To be documented in protocol amendment D5.7 as a deviation with justification.

### D5.6 — Data extraction calibration design

Iterative human-LLM co-development of codebook across 100-paper buckets. Covers both closed fields (pre-specified from PCCM/Table 3) and emergent fields (Adaptation focus, Indicators measured — per D3 §6). Two convergences tracked simultaneously: LLM accuracy (kappa) and category saturation (no new codes appearing).

**Per bucket (RNa → RNb):**
1. H1, H2, and LLM all code the same 100 papers **independently and simultaneously**
2. H1 + H2 reconcile disagreements → gold standard RNa
3. Compare LLM against gold standard → kappa + sensitivity **per field**
4. Review LLM-invented or missed categories — these are diagnostic of underspecified guidance, not just LLM errors
5. Update codebook AND LLM prompt/guidance to reflect reconciled categories + new emergent codes
6. LLM recodes same bucket (RNb) → verify improvement on updated guidance

**Stopping rules:**
- **Closed fields** (country, producer type, method, etc.): lock field once κ ≥ 0.60 and sensitivity ≥ 0.80 in same bucket; do not retest locked fields in subsequent buckets
- **Emergent fields** (adaptation focus, indicators measured): continue until no new categories appear across two consecutive buckets (Neal's saturation criterion)
- Expected convergence after 3–4 buckets (300–400 papers); stop earlier if both criteria met

**Full corpus run:**
- LLM codes all included records using locked codebook + final guidance
- LLM outputs a confidence score and "other/unclear" flag per emergent field
- All "other/unclear" outputs routed to human review — safety net for novel patterns in corpus tail
- Outputs are the canonical extraction database for D5.8

**Infrastructure:** reuse step11 `cohen_kappa`, `confusion_vs_reconciled`; extend to one metric per field; track emergent category lists per bucket separately

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
