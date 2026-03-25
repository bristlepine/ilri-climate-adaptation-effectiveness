[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17809739.svg)](https://doi.org/10.5281/zenodo.17809739)

# ILRI – Measuring what matters: tracking climate adaptation processes and outcomes for smallholder producers in the agriculture sector.
### Evidence synthesis and systematic reviews on climate change and agri-food systems

This repository hosts the methods, workflows, protocols, documentation, and outputs for the ILRI evidence-synthesis project assessing the effectiveness of climate adaptation interventions for smallholder producers. The project includes a scoping review, systematic evidence map, and systematic review/meta-analysis, following CEE, Campbell Collaboration, and Cochrane standards.

---

## Contributors

### Principal Investigators & Authors  
- Jennifer Denno Cissé — Bristlepine Resilience Consultants — ORCID: 0000-0001-5637-1941 — jenn@bristlep.com  
- Caroline G. Staub — Bristlepine Resilience Consultants — caroline@bristlep.com  
- Zarrar Khan — Bristlepine Resilience Consultants — ORCID: 0000-0002-8147-8553 — zarrar@bristlep.com  

### Systematic Review Methodology Support  
- Neal Haddaway — Evidence Synthesis Specialist

### Project Coordination  
- Aditi Mukherji — Principal Scientist, Climate Action — ILRI  

---

## How to Cite This Repository

Cissé, J. D., Staub, C. G., & Khan, Z. (2025). ILRI – Measuring what matters: tracking climate adaptation processes and outcomes for smallholder producers in the agriculture sector. Version 1.0. Zenodo. https://doi.org/10.5281/zenodo.17809739

Each deliverable will receive its own versioned DOI, listed below.

---

## Repository & Output Locations

| Platform | Purpose | URL / Identifier |
|----------|---------|------------------|
| **GitHub** | Code, methods, workflows, documentation | https://github.com/bristlepine/ilri-climate-adaptation-effectiveness |
| **Zenodo** | DOI-minted snapshots of releases | https://doi.org/10.5281/zenodo.17809739 |
| **CGSpace** | Permanent ILRI archive for final outputs | Handle: To be added |
| **Journal Publications** | Scoping review, systematic map, systematic review | To be added |

---

## Deliverables Summary (Aligned to Contract)

| No. | Deliverable | Type | Due Date | Status | DOI |
|-----|-------------|------|----------|--------|------|
| 1 | [Inception Report](https://github.com/bristlepine/ilri-climate-adaptation-effectiveness/blob/main/deliverables/Deliverable%201_Inception%20Report_IL01_v1.pdf) | Final RQs, search plan, Gantt chart | Interim | Complete | https://doi.org/10.5281/zenodo.17861055 |
| 2 | [Draft Systematic Map Protocol](https://github.com/bristlepine/ilri-climate-adaptation-effectiveness/blob/main/deliverables/Deliverable%202_Draft%20Systematic%20Map%20Protocol_v3.pdf) | Interim Report | Jan 2, 2026 | Complete | https://doi.org/10.5281/zenodo.17809739 |
| 3 | [Final Systematic Map Protocol](https://github.com/bristlepine/ilri-climate-adaptation-effectiveness/blob/main/deliverables/Deliverable%203_Bristlepine_Final%20Systematic%20Map%20Protocol_v1.pdf)| Final | Jan 30, 2026 | Complete | https://doi.org/10.5281/zenodo.17809739 |
| 4 | Draft Scoping Review + Evidence Database | Interim | Feb 27, 2026 | In Progress | TBD |
| 5 | Final Scoping Review + Systematic Map + Database | Final | Mar 27, 2026 | In Progress | TBD |
| 6 | Draft Systematic Review / Meta-analysis Protocol | Interim | May 1, 2026 | Not Started | TBD |
| 7 | Final SR/MA Protocol (CGSpace) | Final | May 29, 2026 | Not Started | TBD |
| 8 | Draft Systematic Review / Meta-analysis Manuscript | Interim | Jun 26, 2026 | Not Started | TBD |
| 9 | Final Systematic Review / Meta-analysis Manuscript | Final | Jul 31, 2026 | Not Started | TBD |
| 10 | Final Stakeholder Presentation | Final | Jul 31, 2026 | Not Started | TBD |

---

## Repository Structure

<!-- AUTO-STRUCTURE:START -->
```bash
repo-root/
├── CITATION.cff
├── LICENSE
├── deliverables/
│   ├── 01_inception_report/
│   ├── 02_draft_systematic_map_protocol/
│   ├── 03_final_systematic_map_protocol/
├── environment.yml
├── frontend/                          # Next.js dashboard (visualisations)
├── scripts/
│   ├── config.py                      # Master on/off switches for all steps
│   ├── run.py                         # Pipeline entrypoint — runs enabled steps
│   ├── criteria.yml                   # PCCM eligibility criteria (all rounds)
│   ├── criteria_r2a.yml               # Round 2a guidance overrides
│   ├── utils.py                       # Shared helpers (auth, retries, DOI utils)
│   ├── scopus.py                      # Scopus API wrapper
│   ├── search_strings.yml             # Boolean search strings by database
│   │
│   ├── step1_counts.py                # Query Scopus for record counts
│   ├── step2_retrieve.py              # Download full Scopus record set
│   ├── step3_benchmark.py             # Benchmark against known-relevant papers
│   ├── step4_abstracts.py             # Retrieve missing abstracts (multi-source)
│   ├── step5_eligibility.py           # Rule-based pre-filter
│   ├── step6_visualize.py             # Descriptive plots of raw corpus
│   ├── step7_scopus_check.py          # Validate Scopus metadata completeness
│   ├── step8_dedupe.py                # Deduplication across sources
│   ├── step9_merge.py                 # Merge and consolidate records
│   ├── step9a_enrich_from_ris.py      # Enrich corpus from EPPI-exported RIS files
│   ├── step10_check.py                # LLM calibration screening (sample)
│   ├── step11_irr.py                  # Inter-rater reliability analysis
│   ├── step12_screen_full.py          # LLM full-corpus abstract screening (~17k)
│   ├── step12_export_partial.py       # Export partial results from cache mid-run
│   ├── step13_retrieve_fulltext.py    # Full-text retrieval (Unpaywall/Elsevier/S2/OA)
│   │
│   ├── data/                          # Input data files (RIS exports, calibration sets)
│   └── results/                       # EPPI Reviewer exports and intermediate outputs
```
<!-- AUTO-STRUCTURE:END -->

---

## Pipeline Overview

The pipeline is controlled via `scripts/config.py` — set any step flag to `1` to enable it, then run:

```bash
cd scripts
python run.py
```

All steps are resume-safe via JSONL caching. Long-running steps (10, 12, 13) can be interrupted and restarted without reprocessing completed records.

| Step | Script | Description | Status |
|------|--------|-------------|--------|
| 1 | `step1_counts.py` | Scopus query counts | Complete |
| 2 | `step2_retrieve.py` | Download Scopus corpus | Complete |
| 3 | `step3_benchmark.py` | Benchmark against known papers | Complete |
| 4 | `step4_abstracts.py` | Retrieve missing abstracts | Complete |
| 5 | `step5_eligibility.py` | Rule-based pre-filter | Complete |
| 6 | `step6_visualize.py` | Descriptive corpus plots | Complete |
| 7 | `step7_scopus_check.py` | Metadata validation | Complete |
| 8 | `step8_dedupe.py` | Deduplication | Complete |
| 9 | `step9_merge.py` | Record consolidation | Complete |
| 9a | `step9a_enrich_from_ris.py` | Enrich from EPPI RIS exports | Complete |
| 10 | `step10_check.py` | LLM calibration screening | Complete |
| 11 | `step11_irr.py` | Inter-rater reliability | Complete |
| 12 | `step12_screen_full.py` | LLM full-corpus screening (17,021 records → 4,892 Include, 1,314 missing abstract, 10,815 Exclude) | Complete |
| 13 | `step13_retrieve_fulltext.py` | Full-text retrieval for included records | Pending |
| 14 | _(planned)_ | Full-text screening | Planned |
| 15 | _(planned)_ | Data extraction and coding | Planned |
| 16 | _(planned)_ | Systematic map figures and visualisations | Planned |

---
