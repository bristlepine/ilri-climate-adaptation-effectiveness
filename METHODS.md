# Pipeline Methods

Technical walkthrough of the systematic map pipeline. Each step is a standalone Python script in `scripts/`; all steps are resume-safe (results cached to disk).

For the academic methodology appendix (search strategy, eligibility criteria, IRR analysis, ROSES reporting) see [`documentation/methodology/methodology.md`](documentation/methodology/methodology.md).

---

## Pipeline overview

| Stage | Steps | What happens |
|-------|-------|-------------|
| **1 — Build corpus** | 1, 2, 2b, 3, 4, 4b, 5, 6, 7, 8, 9, 9a | Download records from Scopus and 4 additional databases, deduplicate, fetch missing abstracts, apply rule-based pre-filter, consolidate into a single enriched corpus |
| **2 — Abstract screening** | 10, 11, 12 | Calibrate and validate LLM eligibility screening against a human gold standard, then screen ~17k abstracts; conservative default is include |
| **3 — Full-text retrieval** | 13, 13a, 13b, 13c, 13d | Auto-retrieve PDFs/HTML via Unpaywall, Elsevier, Semantic Scholar, OpenAlex; rescan for fakes; retry failures; export campus-retrieval list for paywalled papers |
| **4 — Screening & coding** | 14, 15, 15b, 15c | Full-text LLM screen on downloaded papers; extract 19 structured fields per study; merge human coding rounds; plot saturation curve |
| **5 — Outputs** | 16, 16b, 16c | Generate all systematic map figures (EGM, geographic, methodology, equity, temporal, saturation, PRISMA); sync to frontend |

**Record flow:**

```
40,653 identified (Scopus + 4 databases)
  → 26,182 after deduplication
  → ~17,000 with abstracts entering LLM screening
  → 8,748 full texts sought
  → 86 human-coded (primary track) + 2,368 LLM-coded (exploratory track)
```

---

## Table of contents

- [Prerequisites](#prerequisites)
- [Stage 1 — Build and clean the corpus](#stage-1--build-and-clean-the-corpus)
  - [Step 1 — Scopus query counts](#step-1--scopus-query-counts)
  - [Step 2 — Retrieve Scopus records](#step-2--retrieve-scopus-records)
  - [Step 2b — Multi-database deduplication](#step-2b--multi-database-deduplication)
  - [Step 3 — Benchmark matching](#step-3--benchmark-matching)
  - [Step 4 — Abstract retrieval](#step-4--abstract-retrieval)
  - [Step 4b — Google Scholar supplement](#step-4b--google-scholar-supplement)
  - [Step 5 — Rule-based pre-filter](#step-5--rule-based-pre-filter)
  - [Step 6 — Corpus visualisation](#step-6--corpus-visualisation)
  - [Step 7 — Scopus metadata check](#step-7--scopus-metadata-check)
  - [Step 8 — Deduplication and cleaning](#step-8--deduplication-and-cleaning)
  - [Step 9 — Abstract enrichment](#step-9--abstract-enrichment)
  - [Step 9a — RIS-based enrichment](#step-9a--ris-based-enrichment)
- [Stage 2 — Abstract screening](#stage-2--abstract-screening)
  - [Step 10 — LLM calibration](#step-10--llm-calibration)
  - [Step 11 — Inter-rater reliability analysis](#step-11--inter-rater-reliability-analysis)
  - [Step 12 — Full-corpus abstract screening](#step-12--full-corpus-abstract-screening)
- [Stage 3 — Full-text retrieval](#stage-3--full-text-retrieval)
  - [Step 13 — Automated retrieval](#step-13--automated-retrieval)
  - [Step 13a — HTML rescan](#step-13a--html-rescan)
  - [Step 13b — Retry failures](#step-13b--retry-failures)
  - [Step 13c — Title-search retrieval](#step-13c--title-search-retrieval)
  - [Step 13d — Manifest reconciliation](#step-13d--manifest-reconciliation)
- [Stage 4 — Full-text screening and data extraction](#stage-4--full-text-screening-and-data-extraction)
  - [Step 14 — Full-text screening](#step-14--full-text-screening)
  - [Step 15 — Data extraction and coding](#step-15--data-extraction-and-coding)
  - [Step 15b — Stack human coding batches](#step-15b--stack-human-coding-batches)
  - [Step 15c — Saturation curve](#step-15c--saturation-curve)
- [Stage 5 — Outputs](#stage-5--outputs)
  - [Step 16 — Visualisations](#step-16--visualisations)
  - [PRISMA flow diagram](#prisma-flow-diagram)
- [Human coding rounds](#human-coding-rounds)
- [Key files](#key-files)

---

## Prerequisites

- Conda environment `ilri01` (see README)
- `.env` in repo root with API keys: `SCOPUS_API_KEY`, `SCOPUS_INST_TOKEN`, `SEMANTIC_SCHOLAR_API_KEY`, `UNPAYWALL_EMAIL`
- `scripts/outputs/` folder (not in repo — shared via Google Drive)
- Ollama running locally with `qwen2.5:14b` pulled (steps 12, 14, 15)

Run all scripts from the repo root:

```bash
conda run -n ilri01 python scripts/<script>.py
```

---

## Stage 1 — Build and clean the corpus

### Step 1 — Scopus query counts (`step1_scopus_query_counts.py`)

Reads search strings from `scripts/search_strings.yml` and queries the Scopus search API for record counts only (no full retrieval). Useful for iterating on query design before committing to a full download.

**Output:** `outputs/step1/` — per-query count CSV and a summary chart.

---

### Step 2 — Retrieve Scopus records (`step2_scopus_retrieve_records.py`)

Downloads the full Scopus corpus matching the combined query. Uses the Elsevier API with institutional token. Results are paged and cached — safe to interrupt and resume.

**Output:** `outputs/step2/step2_scopus_records.csv` — one row per record with title, abstract, DOI, year, source.

---

### Step 2b — Multi-database deduplication (`step2b_multidatabase_dedupe.py`)

Imports RIS/CSV exports from additional databases (Web of Science, CAB Abstracts, AGRIS, Academic Search Premier) and deduplicates them against the Scopus corpus using three methods in order: DOI match → exact title+year → fuzzy title match (Jaccard ≥ 0.85). Net-new records are output in the same schema as step 2 so they can be concatenated for screening.

Place exported files in `scripts/data/multidatabase/` under subfolders `wos/`, `cab/`, `agris/`, `asp/`.

**Output:** `outputs/step2b/step2b_net_new.csv`, `step2b_duplicates.csv`, `step2b_summary.json`.

---

### Step 3 — Benchmark matching (`step3_benchmark_match.py`)

Matches the corpus against a list of known-relevant papers (`scripts/Benchmark List - List.csv`) by DOI and fuzzy title. Used to verify that the search strategy recovers known papers, and to flag any benchmark papers missing from the corpus.

**Output:** `outputs/step3/` — match results and a coverage summary.

---

### Step 4 — Abstract retrieval (`step4_fetch_abstracts.py`)

For records missing abstracts, attempts retrieval from Semantic Scholar → OpenAlex → Elsevier API → CrossRef. Results cached per DOI.

**Output:** `outputs/step4/step4_enriched.csv` — corpus with abstracts filled where available.

---

### Step 4b — Google Scholar supplement (`step4b_gsch_abstracts.py`)

Supplemental abstract fetch from Google Scholar for records that step 4 could not resolve. Requires browser automation (Playwright). Run manually when needed.

---

### Step 5 — Rule-based pre-filter (`step5_eligibility.py`)

Applies simple rule-based filters (e.g. publication type, language) to exclude records that are clearly out of scope before the LLM screening stage. Conservative — errs toward inclusion.

**Output:** `outputs/step5/step5_eligible.csv`.

---

### Step 6 — Corpus visualisation (`step6_visualize.py`)

Descriptive plots of the corpus before screening: records by year, by source database, by abstract availability.

**Output:** `outputs/step6/` — HTML heatmap and PNG plots.

---

### Step 7 — Scopus metadata check (`step7_scopus_check.py`)

Checks metadata completeness (abstract coverage, DOI coverage, year distribution) and verifies benchmark paper recovery. Produces a stacked bar chart comparing benchmark match rates across databases.

**Output:** `outputs/step7/` — summary CSV and chart.

---

### Step 8 — Deduplication and cleaning (`step8_clean_scopus.py`)

Final deduplication across the combined corpus (Scopus + multi-database net-new). Normalises field names, strips leading/trailing whitespace, and resolves any remaining duplicate DOIs.

**Output:** `outputs/step8/step8_cleaned.csv` — the clean combined corpus entering abstract screening.

---

### Step 9 — Abstract enrichment (`step9_enrich_abstracts.py`)

Consolidates abstract coverage across all retrieval attempts (steps 2, 4, 4b). Produces the final enriched CSV that feeds into LLM screening.

**Output:** `outputs/step9/step9_enriched.csv`.

---

### Step 9a — RIS-based enrichment (`step9a_enrich_from_ris.py`)

Supplements step 9 by parsing EPPI Reviewer RIS exports for records that APIs could not resolve. Each iteration is versioned (e.g. `step9a1`). Configure in `config.py`:

```python
step9a_iteration  = "step9a1"
step9a_ris_glob   = "scripts/data/step9a1_ExportedRis_*.txt"
```

**Output:** `outputs/step9a/step9a_scopus_enriched.csv` — this is the final corpus used by all downstream steps.

---

## Stage 2 — Abstract screening

### Step 10 — LLM calibration (`step10_llm_calibrate.py`)

Screens the calibration sample (a set of papers pre-screened by two human reviewers and reconciled into a gold standard) using the LLM with the current eligibility criteria. Reports sensitivity, specificity, kappa, and a confusion matrix. Iterate on `documentation/coding/abstract-screening/criteria/criteria.yml` until metrics are acceptable before proceeding to full-corpus screening.

**Model:** Ollama `qwen2.5:14b`, temperature 0. Configure via `config.py`:

```python
step10_calibration_ris = "scripts/data/calibration_r3_107.ris.txt"
step10_criteria_yml    = "documentation/.../criteria.yml"
step10_run_label       = "r3a"
```

**Output:** `outputs/step10/<run_label>/` — per-record decisions, confusion matrix, IRR plots.

---

### Step 11 — Inter-rater reliability analysis (`step11_irr_analysis.py`)

Analyses agreement between LLM decisions and human gold standard across calibration rounds. Computes Cohen's kappa and tracks criterion-level disagreements. Used to diagnose which criteria need revision.

**Output:** `outputs/step11/` — kappa convergence chart, per-criterion breakdown.

---

### Step 12 — Full-corpus abstract screening (`step12_screen_abstracts.py`)

Applies the validated eligibility criteria to all ~17k records with abstracts. Decision logic: any criterion returning "no" → exclude; all "yes" or "unclear" → include; missing abstract → include (conservative default).

**Model:** Ollama `qwen2.5:14b`. All decisions cached as JSONL — safe to interrupt and resume.

**Output:** `outputs/step12/step12_results.csv`, `step12_results_details.jsonl`, `step12_missing_abstracts.csv`.

---

## Stage 3 — Full-text retrieval

### Step 13 — Automated retrieval (`step13_retrieve_fulltext.py`)

For each record included in step 12, attempts to download the full text via: Unpaywall → Elsevier API → Semantic Scholar → OpenAlex. Files saved to `outputs/step13/fulltext/` named `doi_<DOI>.pdf` or `.html`.

**Output:** `outputs/step13/step13_manifest.csv` — one row per included record with file path and retrieval status.

---

### Step 13a — HTML rescan (`step13a_rescan_fulltexts.py`)

Rescans all HTML files in `fulltext/` for paywalls and fake pages (login walls, error pages). Flags them for re-retrieval. Run after copying in new files from a collaborator.

---

### Step 13b — Retry failures (`step13b_retry.py`)

Retries all records with status `failed` or `403` using browser-style headers and an OpenAlex fallback. Run after step 13a.

---

### Step 13c — Title-search retrieval (`step13c_title_retrieval.py`)

For records that couldn't be retrieved by DOI, tries title search via CrossRef → OpenAlex, then CORE API, then MDPI direct PDF. Targets records with no DOI or non-resolving DOIs.

---

### Step 13d — Manifest reconciliation (`step13d_update_manifest.py`)

Run last in the retrieval sequence. Reconciles the manifest against files actually present in `fulltext/`, regenerates the manual retrieval CSV, exports a versioned campus-retrieval list (`step13_missing_papers_01.csv`, `_02.csv`, …), and updates `step13_summary.json`.

The campus-retrieval list contains publisher-blocked papers (403) that a collaborator can download via an institutional library proxy and upload to Google Drive.

**Typical sequence after a collaborator retrieval run:**

```bash
unzip retrieved.zip -d scripts/outputs/step13/fulltext/
conda run -n ilri01 python scripts/step13a_rescan_fulltexts.py
conda run -n ilri01 python scripts/step13b_retry.py
conda run -n ilri01 python scripts/step13c_title_retrieval.py
conda run -n ilri01 python scripts/step13d_update_manifest.py
```

---

## Stage 4 — Full-text screening and data extraction

### Step 14 — Full-text screening (`step14_screen_fulltext.py`)

For each record that passed abstract screening, reads the downloaded full text and applies a second, more stringent PCCM screen. Same decision logic as step 12 (no MAYBE). Records with no full text are saved to `step14_no_fulltext.csv` and can be reprocessed after manual retrieval.

Text extraction: PDF via pypdf; HTML via trafilatura / BeautifulSoup. Truncated to first ~12,000 characters (intro + methods most informative for screening).

**Model:** Ollama `qwen2.5:14b`, decisions cached as JSONL.

**Output:** `outputs/step14/step14_results.csv`, `step14_no_fulltext.csv`, `step14_extraction_failed.csv`.

---

### Step 15 — Data extraction and coding (`step15_extract_data.py`)

Codes all included records against the systematic map coding schema (defined in the protocol, Table 3, D3). Extracts 19 fields per study: producer type, geographic coverage, climate hazard, adaptation process/outcome domains, methodology, equity dimensions, and more.

Coding source hierarchy (tracked per record):
- `full_text` — coded from downloaded PDF/HTML
- `abstract_only` — no full text retrieved
- `missing_abstract` — titles only
- `needs_manual` — text extraction failed

When a full text becomes available for a previously `abstract_only` record, the cache key changes and the record is automatically re-extracted on the next run.

**Model:** Ollama `qwen2.5:14b`, up to 12,000 characters of full text per record.

**Output:** `outputs/step15/step15_coded.csv` — one row per coded study, `step15_needs_review.csv`.

---

### Step 15b — Stack human coding batches (`step15b_stack_human_batches.py`)

Merges completed human coding rounds (from `documentation/coding/systematic-map/rounds/`) into `outputs/step15/step15_human.csv`. Run after each new round of human coding is finalised.

---

### Step 15c — Saturation curve (`step15c_saturation.py`)

Plots cumulative new coded values (evidence saturation) across the human coding rounds. Used to assess whether additional coding rounds would yield new information.

**Output:** `outputs/step16/saturation.png`.

---

## Stage 5 — Outputs

### Step 16 — Visualisations (`step16_map_visualise.py`)

Generates all systematic map figures from `step15_coded.csv`. Produces both LLM-track (matplotlib) and human-track (Plotly) versions. The human-track Plotly PNGs are the primary figures — they match the interactive frontend exactly and should be used in all deliverables.

**Figures produced:**

| Figure | File (human track) |
|--------|--------------------|
| Evidence Gap Map (EGM) | `outputs/step16/interactive/human/evidence_gap_map.png` |
| Geographic choropleth | `outputs/step16/interactive/human/geographic_map.png` |
| Geographic bar chart | `outputs/step16/interactive/human/geographic_bar.png` |
| Methodology breakdown | `outputs/step16/interactive/human/methodology.png` |
| Temporal trends | `outputs/step16/interactive/human/temporal_trends.png` |
| Equity dimensions | `outputs/step16/interactive/human/equity.png` |
| Saturation curve | `outputs/step16/saturation.png` |
| LLM vs human comparison | `outputs/step16/llm_vs_human.png` |

**Step 16b** (`step16b_export_pngs.py`) — exports PNGs from the Plotly JSONs (human and compare tracks). Run after step 16.

**Step 16c** (`step16c_crosstabs.py`) — generates cross-tabulations used in the report text.

**Full rebuild sequence after new coding rounds:**

```bash
conda run -n ilri01 python scripts/step15b_stack_human_batches.py
conda run -n ilri01 python scripts/step15c_saturation.py
conda run -n ilri01 python scripts/step16_map_visualise.py
conda run -n ilri01 python scripts/step16b_export_pngs.py
```

---

### PRISMA flow diagram

Generated separately from the main pipeline:

```bash
conda run -n ilri01 python deliverables/_gen_prisma.py
```

**Output:** `deliverables/prisma_flow_d5.png` — dual-track PRISMA (human primary + LLM exploratory). Key numbers: 40,653 identified → 26,182 after dedup → 8,748 full texts sought → 86 human included (primary) + 2,368 LLM included (exploratory).

---

## Human coding rounds

Full-text coding is done by human RAs in batches of ~20 papers per round (FT-R3, FT-R4, …). Rounds are managed via `step14b_batch_draw.py`.

**Create new rounds:**

```bash
conda run -n ilri01 python scripts/step14b_batch_draw.py --rounds N --dry-run
conda run -n ilri01 python scripts/step14b_batch_draw.py --push
```

**Pull completed rounds (after Jenn updates missing papers):**

```bash
conda run -n ilri01 python scripts/step14b_batch_draw.py --pull FT-R5
```

**Assign coders:** edit `scripts/outputs/step14b/assignments.csv`, then sync:

```bash
conda run -n ilri01 python scripts/step14b_batch_draw.py --sync-assignments
```

Codebook: `documentation/coding/systematic-map/rounds/CODEBOOK_FT.pdf`
Round folders and coding templates: `documentation/coding/systematic-map/rounds/`
Google Drive (coding rounds): https://drive.google.com/drive/folders/13p22XfvB6sNtTtnMS-dkI1t-joMn-6Bo

---

## Key files

| File | Purpose |
|------|---------|
| `scripts/config.py` | Step flags (0/1), paths, model settings |
| `scripts/run.py` | Orchestrator — runs steps in order per config flags |
| `scripts/search_strings.yml` | Scopus query structure |
| `scripts/Benchmark List - List.csv` | Known-relevant papers for recall checking |
| `documentation/coding/abstract-screening/criteria/criteria.yml` | Active eligibility criteria (abstract screening) |
| `documentation/coding/systematic-map/llm-criteria/criteria_sysmap_v2.yml` | Active coding schema (data extraction) |
| `documentation/coding/systematic-map/rounds/CODEBOOK_FT.pdf` | Human coder codebook |
| `scripts/outputs/step9a/step9a_scopus_enriched.csv` | Final combined corpus entering screening |
| `scripts/outputs/step15/step15_coded.csv` | Final coded dataset (LLM track) |
| `scripts/outputs/step15/step15_human.csv` | Final coded dataset (human track — primary) |
