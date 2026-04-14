[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17809739.svg)](https://doi.org/10.5281/zenodo.17809739)

# Measuring What Matters: Tracking Climate Adaptation for Smallholder Producers

ILRI / Bristlepine Resilience Consultants — 2025–2026

Systematic map and evidence synthesis on methods used to track climate adaptation processes and outcomes for smallholder producers in LMICs. Follows CEE/Campbell standards.

Full methodology: [`documentation/methodology/METHODOLOGY.md`](documentation/methodology/METHODOLOGY.md)
Deliverables tracker: [`TODO.md`](TODO.md)

---

## Setup (one-time)

```bash
git clone https://github.com/bristlepine/ilri-climate-adaptation-effectiveness.git
cd ilri-climate-adaptation-effectiveness
conda env create -f environment.yml
conda activate ilri01
```

Place the `.env` file (API keys) in the repo root. Place the `outputs/` folder at `scripts/outputs/`.

---

## Pipeline

Run individual steps directly, or set flags in `scripts/config.py` and run `python scripts/run.py`.

All steps are resume-safe — interrupt and restart without reprocessing completed records.

---

### Stage 1 — Build & clean the corpus

| Step | Script | Description |
|------|--------|-------------|
| 1 | `step1_scopus_query_counts.py` | Scopus query hit counts |
| 2 | `step2_scopus_retrieve_records.py` | Download full Scopus corpus |
| 2b | `step2b_multidatabase_dedupe.py` | Import WoS / CAB / AGRIS / ASP exports, deduplicate against Scopus → net-new records |
| 3 | `step3_benchmark_match.py` | Match corpus against known-relevant benchmark papers |
| 4 | `step4_fetch_abstracts.py` | Retrieve missing abstracts (multi-source) |
| 5 | `step5_eligibility.py` | Rule-based pre-filter |
| 6 | `step6_visualize.py` | Descriptive corpus plots |
| 7 | `step7_scopus_check.py` | Metadata completeness check |
| 8 | `step8_clean_scopus.py` | Deduplication and cleaning |
| 9 | `step9_enrich_abstracts.py` | Record consolidation and abstract enrichment |
| 9a | `step9a_enrich_from_ris.py` | Enrich from EPPI RIS exports |

Multi-database export instructions: [`scripts/data/multidatabase/DATABASE_EXPORT_INSTRUCTIONS.md`](scripts/data/multidatabase/DATABASE_EXPORT_INSTRUCTIONS.md)

---

### Stage 2 — Abstract screening

| Step | Script | Description |
|------|--------|-------------|
| 10 | `step10_llm_calibrate.py` | LLM calibration on sample set (human IRR rounds) |
| 11 | `step11_irr_analysis.py` | Inter-rater reliability — kappa, sensitivity, confusion matrix |
| 12 | `step12_screen_abstracts.py` | LLM full-corpus abstract screening (~17k records) |

---

### Stage 3 — Full-text retrieval

| Step | Script | Description |
|------|--------|-------------|
| 13 | `step13_retrieve_fulltext.py` | Automated retrieval — Unpaywall, Elsevier API, Semantic Scholar, OpenAlex |
| 13a | `step13a_rescan_fulltexts.py` | Rescan HTML files for paywalls / fake pages — run after copying in new files |
| 13b | `step13b_update_manifest.py` | Reconcile manifest with fulltext/ folder, regenerate missing papers list and summary |
| 13c | `step13c_retry.py` | Targeted retry for failed records — handles 403s, DNS errors, 404s with browser headers and OpenAlex fallback |

Typical sequence after a collaborator retrieval run:

```bash
# 1. Copy new files into fulltext/
unzip retrieved.zip -d scripts/outputs/step13/fulltext/

# 2. Rescan for fake HTMLs
python scripts/step13a_rescan_fulltexts.py

# 3. Reconcile manifest + refresh missing list
python scripts/step13b_update_manifest.py

# 4. Retry remaining failures
python scripts/step13c_retry.py

# 5. Refresh summary again
python scripts/step13b_update_manifest.py
```

---

### Stage 4 — Full-text screening & data extraction

| Step | Script | Description |
|------|--------|-------------|
| 14 | `step14_screen_fulltext.py` | LLM full-text screening of retrieved papers |
| 15 | `step15_extract_data.py` | Data extraction and coding (iterative human-LLM co-coding) |

---

### Stage 5 — Outputs

| Step | Script | Description |
|------|--------|-------------|
| 16 | `step16_map_visualise.py` | ROSES flow diagram, evidence map figures, systematic map outputs |
| — | `stepX_sync_frontend.py` | Copy step16 PNGs + step15 CSV → `frontend/public/map/` for deployment |

```bash
python scripts/stepX_sync_frontend.py
```

---

## Collaborator: running step13 from campus

See setup instructions above. You only need steps 1–4 (git clone, conda, .env, outputs folder).

Connect to your institutional VPN or run on campus, then:

```bash
conda activate ilri01
python scripts/step13_retrieve_fulltext.py
```

Progress is saved continuously — safe to interrupt and resume. When done, zip the outputs and upload to Google Drive:

```bash
zip -r retrieved.zip scripts/outputs/step13/retrieved/
```

Google Drive: https://drive.google.com/drive/folders/1f5y8kjVAcHXBm74AM2wOXsdxrCTnh-ll

---

## Citation

Cissé, J. D., Staub, C. G., & Khan, Z. (2025). *Measuring what matters: tracking climate adaptation processes and outcomes for smallholder producers in the agriculture sector.* Zenodo. https://doi.org/10.5281/zenodo.17809739

---

## Links

| | |
|---|---|
| GitHub | https://github.com/bristlepine/ilri-climate-adaptation-effectiveness |
| Zenodo | https://doi.org/10.5281/zenodo.17809739 |
| Live site | https://bristlepine.com |
| CGSpace | Handle: TBD |
