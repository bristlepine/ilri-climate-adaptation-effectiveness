"""
config.py

Shared configuration for the Scopus pipeline:
- which steps to run
- key paths (inputs + output root)
- API endpoints
"""

import os

# -----------------------------
# Which steps to run (0/1)
# -----------------------------
run_step1 = 0 # Count records (sanity check)
run_step2 = 0 # Retrieve IDs from Scopus
run_step3 = 0  # Match/Enrich Benchmark DOIs
run_step4 = 0  # Fetch Abstracts (APIs + Scraping)
run_step5 = 0 # Eligibility Check (Screening)
run_step6 = 0 # Visualization (HTML Heatmap)
run_step7 = 0 # Benchmark Check (Stacked Bar Analysis) <--- Enabled
run_step8 = 0 # Clean Scopus
run_step9 = 0 # Abstract Enrichment
run_step9a = 0 # Abstract Enrichment from RIS + API retry (supplements step9)
run_step10 = 0 # Title/abstract check (Step 10)
run_step11 = 0 # Inter-rater reliability analysis (Step 11)
run_step12 = 0 # Full-corpus screening (Step 12)
run_step13 = 0 # Full-text retrieval (Step 13)
run_step14 = 0 # Full-text screening (Step 14)
run_step15 = 0 # Data extraction / coding (Step 15)
run_step16 = 1 # Systematic map visualisations (Step 16)

# optional convenience list
runsteps = [
    i for i, flag in enumerate(
        [run_step1, run_step2, run_step3, run_step4, run_step5, run_step6, run_step7, run_step8,
         run_step9, run_step9a, run_step10, run_step11, run_step12, run_step13,
         run_step14, run_step15, run_step16],
        start=1
    )
    if flag
]

# -----------------------------
# Paths (rooted at scripts/)
# -----------------------------
here = os.path.dirname(os.path.abspath(__file__))

search_strings_yml = os.path.join(here, "search_strings.yml")
benchmark_csv = os.path.join(here, "Benchmark List - List.csv")
out_dir = os.path.join(here, "outputs")

# -----------------------------
# Step 9a inputs
# -----------------------------
step9a_iteration  = "step9a1"
step9a_ris_glob   = os.path.join(here, "data", "step9a1_ExportedRis_*.txt")
step9a_update_ris = None  # single-file override; set to a path string to use instead of step9a_ris_glob

# -----------------------------
# Step 10 inputs / run label
# -----------------------------
_criteria_dir = os.path.join(here, "..", "documentation", "coding", "abstract-screening", "criteria")

step10_calibration_ris = os.path.join(here, "data", "calibration_r3_107.ris.txt")
step10_criteria_yml    = os.path.join(_criteria_dir, "criteria.yml")
step10_run_label       = "r3a"

# -----------------------------
# Step 12 inputs
# -----------------------------
step12_criteria_yml = os.path.join(_criteria_dir, "criteria.yml")
step12_model        = ""          # leave blank to use DEFAULT_MODEL in step12
step12_run_limit    = None        # None = all rows; set int for partial run

# -----------------------------
# Step 13 inputs
# -----------------------------
step13_run_limit    = None        # None = all included rows; set int for partial run

# -----------------------------
# Step 14 inputs
# -----------------------------
step14_criteria_yml = os.path.join(_criteria_dir, "criteria.yml")
step14_model        = ""          # leave blank to use DEFAULT_MODEL in step14
step14_run_limit    = None        # None = all rows

# -----------------------------
# Step 15 inputs
# -----------------------------
_extraction_criteria_dir = os.path.join(here, "..", "documentation", "coding", "systematic-map", "llm-criteria")
_rounds_dir              = os.path.join(here, "..", "documentation", "coding", "systematic-map", "rounds")

step15_criteria_yml = os.path.join(_extraction_criteria_dir, "criteria_sysmap_v1.yml")  # update to v2, v3... after each round
step15_model        = ""          # leave blank to use DEFAULT_MODEL in step15
step15_run_limit    = None        # None = all rows

# Calibration-round mode — set these to run LLM on a specific round batch
# instead of the full corpus.  Leave step15_round_template blank ("") to
# run full-corpus mode.
#
# To run FT-R1a:  set step15_round_template to the template CSV path below
#                 and flip run_step15 = 1
step15_round_template = os.path.join(_rounds_dir, "FT-R1a", "coding_ft_r1a_XX.csv")
step15_pdfs_dir       = os.path.join(_rounds_dir, "FT-R1a", "FT-R1a pdfs")
step15_round_out_csv  = os.path.join(_rounds_dir, "FT-R1a", "coding_ft_r1a_LLM.csv")

# -----------------------------
# Step 16 inputs
# -----------------------------
# (no additional settings required yet)

# -----------------------------
# Endpoints
# -----------------------------
scopus_search_url = "https://api.elsevier.com/content/search/scopus"
elsevier_article_url = "https://api.elsevier.com/content/article/doi/"

semantic_scholar_url = "https://api.semanticscholar.org/graph/v1/paper/"
openalex_works_url = "https://api.openalex.org/works/"
crossref_works_url = "https://api.crossref.org/works/"
unpaywall_url = "https://api.unpaywall.org/v2/"

# optional: grouped endpoints dict (your run.py can pass this through)
endpoints = {
    "scopus_search_url": scopus_search_url,
    "elsevier_article_url": elsevier_article_url,
    "semantic_scholar_url": semantic_scholar_url,
    "openalex_works_url": openalex_works_url,
    "crossref_works_url": crossref_works_url,
    "unpaywall_url": unpaywall_url,
}