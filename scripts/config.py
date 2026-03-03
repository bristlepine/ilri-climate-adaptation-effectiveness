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
run_step10 = 0 # Title/abstract check (Step 10)
run_step11 = 1 # Inter-rater reliability analysis (Step 11)

# optional convenience list
runsteps = [
    i for i, flag in enumerate(
        [run_step1, run_step2, run_step3, run_step4, run_step5, run_step6, run_step7, run_step8,
         run_step9, run_step10, run_step11],
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
# Step 10 inputs / run label
# -----------------------------
step10_calibration_ris = os.path.join(here, "data", "calibration_r2_103.ris.txt")
step10_criteria_yml    = os.path.join(here, "criteria.yml")
step10_run_label       = "r2"  # appended to all step10 output filenames (e.g. step10_check_r1b.csv)
                                 # set to "" to use original names (will overwrite previous run)

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