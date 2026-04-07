[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17809739.svg)](https://doi.org/10.5281/zenodo.17809739)

# ILRI – Measuring what matters: tracking climate adaptation processes and outcomes for smallholder producers in the agriculture sector.
### Evidence synthesis and systematic reviews on climate change and agri-food systems

This repository hosts the methods, workflows, protocols, documentation, and outputs for the ILRI evidence-synthesis project assessing the effectiveness of climate adaptation interventions for smallholder producers. The project includes a scoping review, systematic evidence map, and systematic review/meta-analysis, following CEE, Campbell Collaboration, and Cochrane standards.

---

## Table of Contents

- [Running the Pipeline: Step-by-Step Guide for Collaborators](#running-the-pipeline-step-by-step-guide-for-collaborators)
- [Contributors](#contributors)
- [How to Cite This Repository](#how-to-cite-this-repository)
- [Repository & Output Locations](#repository--output-locations)
- [Deliverables Summary](#deliverables-summary-aligned-to-contract)
- [Repository Structure](#repository-structure)
- [Pipeline Overview](#pipeline-overview)

---

## Running the Pipeline: Step-by-Step Guide for Collaborators

This guide is for team members who need to run a pipeline step on their own computer — no prior programming experience required.

> **Why this matters:** Running the full-text retrieval step (`step13`) from an institutional network connection (on campus or via institutional VPN) unlocks access to Wiley, Springer, Taylor & Francis, SAGE, and other publishers via your institution's IP agreements — no additional login required. This can recover full texts for 60–80% of the ~6,000 included papers.

---

### Part 1: Install the required software (one-time setup)

You will need three things installed on your computer. If you already have them, skip ahead.

**A. Git** — used to download the project code

- Mac: Open the **Terminal** app (search "Terminal" in Spotlight). Type `git --version` and press Enter. If you see a version number, you already have it. If not, it will prompt you to install it — click Install.
- Windows: Download from [git-scm.com/download/win](https://git-scm.com/download/win) and run the installer with default settings.

**B. Miniconda** — used to set up the Python environment

- Download the installer for your system from [docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
- Run the installer with default settings. When asked "Add Miniconda to PATH", say **Yes**.
- After installing, close and reopen Terminal (or Command Prompt on Windows).

**C. Terminal / Command Prompt**

- **Mac:** Press `Cmd + Space`, type `Terminal`, press Enter.
- **Windows:** Press the Windows key, type `Anaconda Prompt`, open it. Use this instead of Command Prompt for all steps below.

---

### Part 2: Download the project code (one-time setup)

In Terminal, type the following commands one at a time, pressing Enter after each:

```bash
git clone https://github.com/bristlepine/ilri-climate-adaptation-effectiveness.git
```

This downloads the project into a new folder called `ilri-climate-adaptation-effectiveness`. Then navigate into it:

```bash
cd ilri-climate-adaptation-effectiveness
```

You should now be "inside" the project folder. You can verify this by typing `ls` (Mac) or `dir` (Windows) — you should see files like `README.md` and `environment.yml`.

---

### Part 3: Set up the Python environment (one-time setup)

This installs all the Python packages the project needs:

```bash
conda env create -f environment.yml
```

This will take a few minutes. When it finishes, activate the environment:

```bash
conda activate ilri01
```

You should see `(ilri01)` appear at the start of your command line. **You need to run this activation command every time you open a new Terminal window before running any scripts.**

---

### Part 4: Add the credentials file (one-time setup)

The pipeline needs API keys to access Elsevier and OpenAI. The project coordinator will send you a small text file called `.env`. Place this file in the main project folder (the same folder that contains `README.md`).

> **Mac tip:** Files starting with a dot (`.`) are hidden by default. To see them in Finder: press `Cmd + Shift + .` to toggle hidden files visible.

The `.env` file should look like this (Zarrar will fill in the actual values):

```
SCOPUS_API_KEY=your_key_here
SCOPUS_INST_TOKEN=your_token_here
OPENAI_API_KEY=your_key_here
```

Do not share this file or commit it to GitHub — it contains private credentials.

---

### Part 5: Download the data outputs (one-time setup)

The pipeline's intermediate data files are too large for GitHub. The project coordinator will share a folder called `outputs` via Google Drive (or similar). Download it and place it here:

```
ilri-climate-adaptation-effectiveness/
  scripts/
    outputs/        ← place the shared folder here
```

So the full path would be: `ilri-climate-adaptation-effectiveness/scripts/outputs/`

---

### Part 6: Connect to your institutional VPN (before running step13)

For full-text retrieval to work at its best, your computer needs to appear to publishers as being on your institution's network. This happens automatically if you are:

- **On campus** — no extra steps needed
- **Off campus** — connect to your institution's VPN first (contact your IT department for instructions)

Once connected, publishers like Wiley, Springer, and Taylor & Francis will automatically grant full-text access based on your institution's IP agreements — no login required on their sites.

---

### Part 7: Run the full-text retrieval

Make sure:
- Your Terminal shows `(ilri01)` at the start of the line
- You are inside the project folder (`cd ilri-climate-adaptation-effectiveness`)
- You are connected to Cornell VPN (or on campus)

Then run:

```bash
python scripts/step13_retrieve_fulltext.py
```

You will see progress printed to the screen as it processes each paper. It is normal for this to take several hours — it is downloading PDFs for thousands of papers one by one.

**If it stops or you need to close your laptop:** Don't worry. The script saves its progress as it goes. Just run the same command again and it will pick up exactly where it left off — it will not re-download anything it already retrieved.

---

### Part 8: Share the results with your team

When the script finishes (or even partway through if you need to stop), zip the output folder and send it to Zarrar:

**Mac:**
```bash
zip -r step13_outputs.zip scripts/outputs/step13/
```

**Windows:**
Right-click the `step13` folder inside `scripts/outputs/` → Send to → Compressed (zipped) folder.

Then share the zip file with the project coordinator. Even a partial run is very useful — every paper retrieved helps.

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
│   ├── step1_scopus_query_counts.py         # Query Scopus for record counts
│   ├── step2_scopus_retrieve_records.py     # Download full Scopus record set
│   ├── step3_benchmark_match.py             # Benchmark against known-relevant papers
│   ├── step4_fetch_abstracts.py             # Retrieve missing abstracts (multi-source)
│   ├── step5_eligibility.py                 # Rule-based pre-filter
│   ├── step6_visualize.py                   # Descriptive plots of raw corpus
│   ├── step7_scopus_check.py                # Validate Scopus metadata completeness
│   ├── step8_clean_scopus.py                # Deduplication and cleaning
│   ├── step9_enrich_abstracts.py            # Merge and enrich abstracts
│   ├── step9a_enrich_from_ris.py            # Enrich corpus from EPPI-exported RIS files
│   ├── step10_llm_calibrate.py              # LLM calibration screening (sample set)
│   ├── step11_irr_analysis.py               # Inter-rater reliability analysis
│   ├── step12_screen_abstracts.py           # LLM full-corpus abstract screening (~17k)
│   ├── step12_screen_abstracts_export_partial.py  # Export partial results from cache mid-run
│   ├── step13_retrieve_fulltext.py          # Full-text retrieval (Unpaywall/Elsevier/S2/OA)
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
| 1 | `step1_scopus_query_counts.py` | Scopus query counts | Complete |
| 2 | `step2_scopus_retrieve_records.py` | Download Scopus corpus | Complete |
| 3 | `step3_benchmark_match.py` | Benchmark against known papers | Complete |
| 4 | `step4_fetch_abstracts.py` | Retrieve missing abstracts | Complete |
| 5 | `step5_eligibility.py` | Rule-based pre-filter | Complete |
| 6 | `step6_visualize.py` | Descriptive corpus plots | Complete |
| 7 | `step7_scopus_check.py` | Metadata validation | Complete |
| 8 | `step8_clean_scopus.py` | Deduplication and cleaning | Complete |
| 9 | `step9_enrich_abstracts.py` | Record consolidation and abstract enrichment | Complete |
| 9a | `step9a_enrich_from_ris.py` | Enrich from EPPI RIS exports | Complete |
| 10 | `step10_llm_calibrate.py` | LLM calibration screening | Complete |
| 11 | `step11_irr_analysis.py` | Inter-rater reliability | Complete |
| 12 | `step12_screen_abstracts.py` | LLM full-corpus abstract screening (17,021 records → 4,892 Include, 1,314 missing abstract, 10,815 Exclude) | Complete |
| 13 | `step13_retrieve_fulltext.py` | Full-text retrieval for included records | Pending |
| 14 | _(planned)_ | Full-text screening | Planned |
| 15 | _(planned)_ | Data extraction and coding | Planned |
| 16 | _(planned)_ | Systematic map figures and visualisations | Planned |

---
