"""
update_d5_gdoc.py

Rewrites the D5 Google Doc to follow D4's document structure:
  - Title page + authors
  - 1. Executive Summary
  - 2. Background
  - 3. Methods (brief — 3.1 Search, 3.2 Dedup, 3.3 Screening, 3.4 Data extraction)
  - 4. Final Results (4.1 Search summary + Table 1, 4.2 Evidence gap map + figures)
  - 5. Searchable Database
  - 6. Limitations
  - 7. Conclusions
  - 8. Acknowledgements
  - 9. References
  - Appendix: Technical Methods (calibration table, saturation, software)

Figures are inserted as placeholder paragraphs — drag-and-drop the PNG files
from scripts/outputs/step16/ into the Google Doc at the marked locations.

Run: python3 deliverables/update_d5_gdoc.py
"""

from __future__ import annotations

import time
from pathlib import Path

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

CREDS_DIR  = Path(__file__).resolve().parent / ".credentials"
TOKEN_FILE = CREDS_DIR / "token.json"
SCOPES     = [
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/drive",
]

D5_DOC_ID = "14UaK_9lXCuE_AsOYXNvfImYlEndLUL4AM0lVz4thEAY"

# ── content blocks ─────────────────────────────────────────────────────────────
# Each block: (style, text)
# Styles: TITLE, SUBTITLE, HEADING_1, HEADING_2, BODY, FIGURE, NOTE, BULLET

DOCUMENT = [

    # ── Cover page ──────────────────────────────────────────────────────────────
    ("TITLE",    "Measuring what matters: tracking climate adaptation processes and outcomes "
                 "for smallholder producers in the agriculture sector"),
    ("SUBTITLE", "Deliverable 5: Final Systematic Map"),
    ("SUBTITLE", "June 2026, v01"),
    ("BODY",     "Jennifer Denno Cissé*, jenn@bristlep.com, USA, Bristlepine Resilience Consultants\n"
                 "Zarrar Khan, zarrar@bristlep.com, USA, Bristlepine Resilience Consultants\n"
                 "Caroline G. Staub, caroline@bristlep.com, USA, Bristlepine Resilience Consultants"),
    ("BODY",     "Keywords: Climate Adaptation, Adaptation Outcomes, Adaptation Process, "
                 "Smallholder Farmers, Monitoring and Evaluation (M&E), Agriculture, "
                 "Systematic Map, Evidence Synthesis, LMIC"),

    # ── 1. Executive Summary ────────────────────────────────────────────────────
    ("HEADING_1", "1.  Executive Summary"),
    ("BODY",
     "This document presents the final systematic map of methods used to track and measure "
     "climate adaptation processes and outcomes for smallholder producers across low- and "
     "middle-income countries (LMICs). It fulfils Deliverable 5 of the contract with the "
     "International Livestock Research Institute (ILRI)."),
    ("BODY",
     "A total of 39,113 records were identified across five bibliographic databases and 24 "
     "grey literature sources. After deduplication, 25,208 unique records entered title and "
     "abstract screening. The validated LLM screening tool (sensitivity 0.966–0.970; κ = "
     "0.720–0.721) identified 8,558 records for full-text assessment. Of 3,476 full texts "
     "retrieved, 2,368 were included after full-text screening."),
    ("BODY",
     "A random sample of 86 studies was selected for human-coded data extraction across five "
     "batches. Information saturation was reached by batch FT-R2c (49 papers): all three tracked "
     "dimensions — process/outcome domains, methodological approaches, and producer types — showed "
     "no new canonical categories across the final two batches. This confirms the human-coded "
     "findings are representative of the broader evidence base."),
    ("BODY", "Key findings from the human-coded results (n = 86):"),
    ("BULLET", "A recent but thin evidence base: 96% of studies published since 2015 (median: 2022). "
               "The evidence base is growing quickly but long-term impact evidence remains sparse."),
    ("BULLET", "Marginalized groups are largely invisible: 86% of studies provide no equity disaggregation "
               "by gender, age, or ethnicity. This is a critical gap for inclusive adaptation policy."),
    ("BULLET", "Non-crop producers are severely underrepresented: fisheries/aquaculture (<6%) and "
               "agroforestry (<5%) appear in very few studies. Evidence skews heavily toward crop farming."),
    ("BULLET", "Cost data are almost entirely absent: approximately 80% of studies report no cost or "
               "efficiency data, making value-for-money assessment very difficult."),
    ("BULLET", "Three countries dominate: Ethiopia, Ghana, and Kenya account for 28% of all studies. "
               "Most of Africa, Asia, and Latin America remain very poorly covered."),
    ("BULLET", "Process outcomes are studied; impact outcomes are not: knowledge, adoption, and "
               "decision-making are well covered. Income, wellbeing, and resilience remain sparse."),
    ("BODY",
     "An interactive evidence gap map and searchable database are available at the project website "
     "(https://main.d3a9jahzuu7xai.amplifyapp.com/systematic-map). The LLM-extracted corpus "
     "(n = 2,368) is available as an exploratory reference alongside the human-coded results."),
    ("BODY",
     "The systematic map protocol was amended to document all deviations from the pre-registered "
     "protocol (Deliverable 3 v03, Zenodo: 10.5281/zenodo.19811629)."),

    # ── 2. Background ───────────────────────────────────────────────────────────
    ("HEADING_1", "2.  Background"),
    ("BODY",
     "The background, theory of change, stakeholder engagement process, and full PCCM eligibility "
     "criteria underpinning this systematic map are documented in the protocol published as "
     "Deliverable 3 (Bristlepine Resilience Consultants, January 2026; Zenodo: 10.5281/zenodo.19811629)."),
    ("BODY",
     "A growing number of climate adaptation interventions are being deployed across the agriculture "
     "sector to support smallholder producers in LMICs. Despite large investments, the evidence base "
     "on whether and how adaptation processes and outcomes are tracked — and what methods are used to "
     "do so — remains fragmented. This systematic map provides a structured overview of that evidence "
     "base to support better M&E practice and research prioritisation."),
    ("BODY",
     "This systematic map covers adaptation processes (knowledge, decision-making, adoption, "
     "behavioural change, participation, institutional governance, access to services) and adaptation "
     "outcomes (yields, income, livelihoods, wellbeing, risk reduction, resilience) as specified in "
     "the PCCM framework (Population · Concept · Context · Method)."),

    # ── 3. Methods ──────────────────────────────────────────────────────────────
    ("HEADING_1", "3.  Methods"),
    ("BODY",
     "This systematic map follows the protocol published as Deliverable 3 (Bristlepine Resilience "
     "Consultants, January 2026). A brief summary of methods is provided here. Technical details — "
     "including calibration data, software stack, and saturation analysis — are in the Appendix."),

    ("HEADING_2", "3.1  Search Strategy"),
    ("BODY",
     "Searches were conducted in Scopus (Elsevier) in January 2026, using a structured Boolean "
     "search string developed around the PCCM framework. The Scopus search returned 17,021 records. "
     "Additional databases searched in parallel included Web of Science Core Collection (15,179 "
     "records), CAB Abstracts (5,723), AGRIS (3), and Academic Search Premier (1,187). "
     "Twenty-four grey literature sources were searched manually. Total records across all sources: "
     "39,113."),

    ("HEADING_2", "3.2  Deduplication"),
    ("BODY",
     "Records from all databases were deduplicated using a three-pass algorithm: exact DOI match, "
     "then exact title and year match, then fuzzy title match within the same year (Jaccard "
     "similarity ≥ 0.85). This reduced 39,113 records to 25,208 unique records (14,905 duplicates "
     "removed)."),

    ("HEADING_2", "3.3  Screening"),
    ("BODY",
     "Title and abstract screening was conducted using a validated LLM-assisted screening tool "
     "(Ollama/qwen2.5:14b), calibrated against the PCCM eligibility criteria through six "
     "calibration rounds with two independent human reviewers (Caroline Staub, Jennifer Cisse). "
     "The final criteria achieved sensitivity = 0.966–0.970 and κ = 0.720–0.721, exceeding the "
     "pre-specified thresholds (sensitivity ≥ 0.95; κ ≥ 0.60). A total of 8,558 records were "
     "included at title/abstract stage. Full-text screening of 3,476 retrieved full texts "
     "identified 2,368 included records."),

    ("HEADING_2", "3.4  Random Selection and Data Extraction"),
    ("BODY",
     "Given the large volume of included studies, data extraction used a random sampling approach "
     "rather than coding the full corpus. Papers were drawn in batches of 20 using a pure random "
     "draw (fixed integer seeds: 42, 43, …) with each batch explicitly excluding DOIs already "
     "assigned in prior batches, making overlap impossible by construction."),
    ("BODY",
     "Data are extracted using a structured codebook comprising 16 fields, including publication "
     "year, country/region, producer type, climate context, adaptation focus (process or outcome "
     "domain), methodological approach, equity and inclusion dimensions, and study strengths and "
     "limitations. Coders also apply the five PCCM inclusion criteria to confirm each paper is a "
     "valid include; false positives from LLM full-text screening are removed and counted."),
    ("BODY",
     "Information saturation was tracked at each batch across three dimensions: process/outcome "
     "domains (13 canonical values), methodological approaches (5), and producer types (5). "
     "Saturation was reached by batch FT-R2c (49 papers coded): zero new canonical categories "
     "were added across the final two batches (FT-R2d and FT-R3, covering an additional 37 papers). "
     "A total of five batches and 86 papers were coded."),

    # ── 4. Final Results ────────────────────────────────────────────────────────
    ("HEADING_1", "4.  Final Results"),

    ("HEADING_2", "4.1  Search and Screening Summary"),
    ("BODY",
     "Table 1 summarises the search and screening results by database. A total of 39,113 records "
     "were retrieved across five databases. After deduplication, 25,208 unique records remained. "
     "Title and abstract screening identified 8,558 records for full-text assessment "
     "(Scopus: 6,218; multi-database net-new: 2,340). Of 3,476 full texts retrieved (40%), "
     "3,464 were screened and 2,368 were included after full-text screening."),
    ("BODY",
     "Table 1. Search and screening results by database. FT = full text; combined FT screening "
     "was conducted across all databases together; individual-database FT counts show retrieval "
     "only. Records requiring manual full-text access (n = 5,243) are not reflected in these totals."),
    ("NOTE",  "[ TABLE 1 IS ALREADY IN THIS DOCUMENT — pipeline table above ]"),
    ("BODY",
     "Figure 1 presents the ROSES (Reporting Standards for Systematic Evidence Syntheses) flow "
     "diagram showing the full record flow across all 29 sources and all screening stages."),
    ("FIGURE", "[ FIGURE 1 — ROSES flow diagram: scripts/outputs/step16/roses_flow.png ]"),

    ("HEADING_2", "4.2  Evidence Gap Map"),
    ("BODY",
     "The evidence gap map (EGM) is the primary output of this deliverable. It plots the distribution "
     "of included human-coded studies across process/outcome domains (y-axis) and producer types "
     "(x-axis). Bubble size is proportional to the number of studies in each cell. Grey markers "
     "indicate evidence gaps — combinations with no studies."),
    ("BODY",
     "Figure 2 presents the EGM based on the 86 human-coded studies, which constitute the "
     "authoritative output for this deliverable. An interactive version is available online, with "
     "a toggle to compare human-coded and LLM-extracted results."),
    ("FIGURE", "[ FIGURE 2 — Evidence Gap Map (human, n=86): "
               "scripts/outputs/step16/interactive/human/evidence_gap_map.png ]"),
    ("BODY",  "Key observations from the evidence gap map:"),
    ("BULLET", "Process domains (knowledge/awareness, decision-making, uptake/adoption) are more "
               "densely covered than outcome domains (income, wellbeing, resilience). No studies "
               "in the human sample directly measured resilience or adaptive capacity outcomes "
               "for livestock, fisheries, agroforestry, or mixed-system producers."),
    ("BULLET", "Crop farming dominates across all domains. Livestock, fisheries/aquaculture, "
               "agroforestry, and mixed-system producers are consistently underrepresented, "
               "with most cells showing evidence gaps."),
    ("BULLET", "Institutional governance and access to services are nearly absent across all "
               "producer types — a critical gap for systemic adaptation measurement."),

    ("HEADING_2", "4.3  Geographic Distribution"),
    ("BODY",
     "Studies are heavily concentrated in Sub-Saharan Africa and South Asia. Ethiopia, Ghana, "
     "and Kenya account for 28% of all human-coded studies. Coverage is very sparse across "
     "Central America, Southeast Asia, the Sahel, and the Middle East and North Africa (MENA)."),
    ("FIGURE", "[ FIGURE 3 — Geographic distribution (choropleth): "
               "scripts/outputs/step16/geographic_map.png ]"),
    ("FIGURE", "[ FIGURE 4 — Top countries by study count: "
               "scripts/outputs/step16/geographic_bar.png ]"),

    ("HEADING_2", "4.4  Temporal Trends"),
    ("BODY",
     "Publication volume has grown sharply since 2015, with 96% of included studies published "
     "in the past decade and a median year of 2022. This reflects both growth in the field and "
     "improved publication infrastructure — but it also means that long-term impact evidence "
     "is very thin, as most studies were published too recently to measure multi-year outcomes."),
    ("FIGURE", "[ FIGURE 5 — Publications per year: "
               "scripts/outputs/step16/temporal_trends.png ]"),

    ("HEADING_2", "4.5  Methodological Approaches"),
    ("BODY",
     "Survey-based and qualitative approaches dominate the evidence base. Experimental designs "
     "(randomised controlled trials, quasi-experiments) represent fewer than 10% of studies, "
     "severely limiting causal inference about adaptation effectiveness. Participatory and "
     "mixed-method approaches are common, particularly for process-level outcomes. This pattern "
     "is broadly consistent across both the human-coded sample and the larger LLM corpus."),
    ("FIGURE", "[ FIGURE 6 — Methodological approaches: "
               "scripts/outputs/step16/methodology_bar.png ]"),

    ("HEADING_2", "4.6  Equity and Inclusion"),
    ("BODY",
     "86% of included studies provide no disaggregation by any equity dimension. Where equity "
     "is addressed, gender is the most common focus — but still appears in fewer than 15% of "
     "studies. Youth, indigenous peoples, people with disabilities, and pastoralist/migratory "
     "groups are nearly absent from the evidence base. This is a critical gap: adaptation "
     "outcomes are rarely equivalent across social groups, yet the evidence base does not "
     "systematically track who benefits."),
    ("FIGURE", "[ FIGURE 7 — Equity and inclusion dimensions: "
               "scripts/outputs/step16/equity_bar.png ]"),

    ("HEADING_2", "4.7  Information Saturation"),
    ("BODY",
     "Figure 8 shows the saturation curve tracking how quickly new evidence categories were "
     "discovered as human coding progressed. All three tracked dimensions plateaued by 49 "
     "papers (batch FT-R2c). The dashed line marks the 95% saturation threshold. Zero new "
     "canonical categories were added in the final two batches (37 additional papers). This "
     "provides evidence that the human-coded sample, while modest in size, is sufficiently "
     "representative to characterise the evidence space."),
    ("FIGURE", "[ FIGURE 8 — Information saturation curve: "
               "scripts/outputs/step16/saturation.png ]"),

    ("HEADING_2", "4.8  LLM Corpus — Exploratory Reference"),
    ("BODY",
     "In parallel, the LLM-assisted pipeline extracted data from all 2,368 LLM-screened "
     "included records. This larger corpus is available on the project website as an exploratory "
     "reference, toggled separately from the human-coded results. The distribution of key "
     "variables is broadly consistent between the human and LLM tracks, supporting the "
     "robustness of the human-coded findings. However, LLM-extracted data have not been "
     "individually validated and should not be treated as authoritative."),
    ("FIGURE", "[ FIGURE 9 — LLM vs Human comparison: "
               "scripts/outputs/step16/llm_vs_human.png ]"),

    # ── 5. Searchable Database ──────────────────────────────────────────────────
    ("HEADING_1", "5.  Searchable Database"),
    ("BODY",
     "The searchable database is available at: "
     "https://main.d3a9jahzuu7xai.amplifyapp.com/systematic-map"),
    ("BODY",
     "The database includes all 2,368 LLM-screened records with key metadata fields (year, "
     "title, country, producer type, domain type, methodological approach). Users can filter "
     "by any combination of these fields and download the full CSV. A toggle on the website "
     "switches between the human-coded results (n = 86, authoritative) and the LLM corpus "
     "(n = 2,368, exploratory reference)."),

    # ── 6. Limitations ──────────────────────────────────────────────────────────
    ("HEADING_1", "6.  Limitations"),
    ("BODY",
     "Several limitations should be considered when interpreting the findings of this systematic map."),
    ("BODY",
     "Full-text retrieval rate. Only 40% of abstract-screened records (3,476 of 8,748 full texts "
     "sought) were retrieved automatically. Records requiring manual library access are not "
     "included in the full-text screening totals. Records without a retrieved full text are not "
     "excluded from the database — absence of a full text is not grounds for exclusion — but "
     "they have not been coded."),
    ("BODY",
     "Human sample size. The 86 human-coded papers are sufficient for saturation across the "
     "three tracked dimensions, but individual cells in the evidence gap map may have very "
     "low counts. Findings about specific domain-producer combinations should be treated as "
     "indicative rather than definitive."),
    ("BODY",
     "LLM corpus (exploratory only). The 2,368 LLM-extracted records have not been individually "
     "validated by human reviewers. LLM extraction errors in individual fields are likely. "
     "The LLM corpus is provided as an exploratory reference and is clearly labelled as such "
     "on the project website."),
    ("BODY",
     "Grey literature. Manual searching of grey literature sources was conducted but is likely "
     "incomplete. Organisational reports and policy documents that are not indexed in "
     "bibliographic databases may be underrepresented."),

    # ── 7. Conclusions ──────────────────────────────────────────────────────────
    ("HEADING_1", "7.  Conclusions"),
    ("BODY",
     "This final systematic map provides a structured characterisation of the evidence base on "
     "methods used to track and measure climate adaptation processes and outcomes for smallholder "
     "producers in LMICs. The evidence base is recent, growing, and geographically concentrated — "
     "but it has significant gaps."),
    ("BODY",
     "The dominant finding is an evidence mismatch: the literature measures early-stage process "
     "outcomes (knowledge, adoption, decision-making) far better than it measures downstream "
     "impact outcomes (income, wellbeing, resilience). For non-crop producers — fishers, "
     "pastoralists, agroforestry practitioners — even process-level evidence is sparse. "
     "Equity disaggregation is almost entirely absent."),
    ("BODY",
     "These gaps have direct implications for M&E practice: current monitoring systems are not "
     "designed to detect adaptation impact, and they systematically overlook marginalised groups "
     "and non-crop livelihood systems. Addressing these gaps will require deliberate research "
     "investments in long-term cohort studies, equity-disaggregated data collection, and "
     "expansion beyond crop-focused producer types."),
    ("BODY",
     "The next phase of this work (Deliverable 6) will develop a systematic review and "
     "meta-analysis protocol targeting the most tractable domains identified in this map — "
     "focusing on studies with sufficient outcome measurement to enable quantitative synthesis."),

    # ── 8. Acknowledgements ──────────────────────────────────────────────────────
    ("HEADING_1", "8.  Acknowledgements"),
    ("BODY",
     "This work was funded by the International Livestock Research Institute (ILRI) under the "
     "CGIAR Initiative on Climate Resilience (ClimBeR). We thank Aditi Krishnapriyan and Neal "
     "Hockley for their guidance and review throughout this project."),

    # ── 9. References ─────────────────────────────────────────────────────────
    ("HEADING_1", "9.  References"),
    ("BODY",
     "Bristlepine Resilience Consultants (2026). Deliverable 3: Final Systematic Map Protocol — "
     "Measuring what matters: tracking climate adaptation processes and outcomes for smallholder "
     "producers in the agriculture sector. Zenodo. https://doi.org/10.5281/zenodo.19811629"),
    ("BODY",
     "Bristlepine Resilience Consultants (2026). Deliverable 4: First Draft Systematic Map "
     "(Preliminary) — Measuring what matters. Zenodo. https://doi.org/10.5281/zenodo.19811622"),
    ("BODY",
     "Collaboration for Environmental Evidence (2018). Guidelines and Standards for Evidence "
     "Synthesis in Environmental Management. Version 5.0. Pullin, A.S., Frampton, G.K., Livoreil, "
     "B. & Petrokofsky, G. (eds). www.environmentalevidence.org/information-for-authors"),
    ("BODY",
     "Haddaway, N.R., Macura, B., Whaley, P. and Pullin, A.S. (2018). ROSES Reporting standards "
     "for Systematic Evidence Syntheses: pro forma, flow diagram and descriptive summary of the "
     "plan and conduct of environmental systematic reviews and systematic maps. Environmental "
     "Evidence, 7, 7."),
    ("BODY",
     "O'Mara-Eves, A. et al. (2015). Using text mining for study identification in systematic "
     "reviews: a systematic review of current approaches. Systematic Reviews, 4, 5."),

    # ── Appendix: Technical Methods ─────────────────────────────────────────────
    ("HEADING_1", "Appendix: Technical Methods"),
    ("BODY",
     "This appendix provides technical details on the computational pipeline, calibration "
     "rounds, and software stack. It supplements Section 3 and is intended for reviewers "
     "who want to assess methodological rigour."),

    ("HEADING_2", "A1.  Calibration Rounds"),
    ("BODY",
     "Two independent human reviewers (Caroline Staub, Jennifer Cisse) screened each "
     "calibration batch in EPPI Reviewer, reconciling disagreements into a gold standard. "
     "The LLM was assessed against the gold standard after each round. Criteria were revised "
     "between rounds. Screening was withheld from the full corpus until sensitivity ≥ 0.95 "
     "and κ ≥ 0.60 were both achieved."),
    ("BODY",
     "Calibration results by round:\n"
     "  Round              | n    | Sensitivity | Specificity | LLM κ  | Human κ | Pass\n"
     "  R1 — initial       | 205  | 0.776       | 0.703       | 0.436  | 0.500   | No\n"
     "  R1a — 1st revision | 205  | 0.761       | 0.797       | 0.534  | 0.500   | No\n"
     "  R1b — 2nd revision | 205  | 0.866       | 0.819       | 0.645  | 0.500   | No\n"
     "  R2a — 3rd revision | 103  | 0.897       | 0.905       | 0.770  | 0.765   | No\n"
     "  R2b — 4th revision | 103  | 0.966       | 0.838       | 0.720  | 0.765   | YES ✓\n"
     "  R3a — stability    | 107  | 0.970       | 0.824       | 0.721  | 0.703   | YES ✓\n"
     "\n"
     "R2b 95% CI (Wilson): 0.828–0.994. R3a: 0.847–0.995. Pooled (60/62 true positives): 0.890–0.991."),

    ("HEADING_2", "A2.  Full-text Screening Details"),
    ("BODY",
     "Full-text screening used the same LLM (qwen2.5:14b, temperature 0.0) applied to "
     "retrieved PDF/HTML content. Per-criterion decisions (yes/no/unclear) with quoted "
     "supporting passages were required. Unverifiable quotes downgraded to unclear. Uncertain "
     "defaults to include. Of 3,476 full texts retrieved, 3,464 were screened; 12 could not "
     "be processed due to format or access issues. Results: 2,368 included, 1,096 excluded."),

    ("HEADING_2", "A3.  Human Coding Batches"),
    ("BODY",
     "Human coding batches (FT-R2a through FT-R3) each comprised 20 papers drawn with a "
     "fixed integer seed. Each batch explicitly excluded DOIs already assigned in prior batches. "
     "Coders applied the 5 PCCM inclusion criteria to confirm inclusion, then completed all "
     "16 extraction fields. Papers marked as false positives by coders were removed from "
     "step15_human.csv and counted as LLM FT-screening errors."),
    ("BODY",
     "Batch summary: FT-R2a (seed 42): 20 papers → 17 confirmed; FT-R2b (seed 43): 20 papers "
     "→ 18 confirmed; FT-R2c (seed 44): 9 papers (saturation reached mid-batch) → 9 confirmed; "
     "FT-R2d (seed 45): 20 papers → 18 confirmed; FT-R3 (seed 46): 20 papers → 18 confirmed. "
     "Total: 89 assigned → 86 confirmed includes (3 false positives removed, 3.4% FP rate)."),

    ("HEADING_2", "A4.  Software Stack"),
    ("BODY",
     "LLM inference: Ollama (qwen2.5:14b) — local, deterministic, temperature 0.0.\n"
     "Record retrieval: Elsevier Scopus REST API, CrossRef, OpenAlex, Semantic Scholar.\n"
     "Full-text access: Unpaywall API, Elsevier full-text API.\n"
     "PDF parsing: pypdf. HTML parsing: trafilatura, BeautifulSoup4.\n"
     "Data handling: pandas. Visualisation: matplotlib, Plotly + kaleido.\n"
     "IRR statistics: custom Python (Cohen's κ, Wilson confidence intervals).\n"
     "Human screening: EPPI Reviewer (cloud-based systematic review platform).\n"
     "Website: Next.js (React) hosted on AWS Amplify.\n"
     "All code and calibration data: github.com/bristlepine/ilri-climate-adaptation-effectiveness"),

]


# ── Google Docs API helpers ────────────────────────────────────────────────────

def get_creds() -> Credentials:
    creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return creds


# ── Build the document ─────────────────────────────────────────────────────────

def _make_insert(index: int, text: str) -> dict:
    return {"insertText": {"location": {"index": index}, "text": text}}


def _make_style(start: int, end: int, style: str) -> dict:
    return {
        "updateParagraphStyle": {
            "range": {"startIndex": start, "endIndex": end},
            "paragraphStyle": {"namedStyleType": style},
            "fields": "namedStyleType",
        }
    }


def run():
    creds = get_creds()
    docs  = build("docs", "v1", credentials=creds)

    # ── Step 1: Delete all existing body content ──────────────────────────────
    print("[d5] Reading current document...")
    body = docs.documents().get(documentId=D5_DOC_ID).execute()["body"]
    content = body["content"]

    # Find the full range to delete (everything except the trailing newline)
    start_idx = content[0].get("startIndex", 1)
    end_idx   = content[-1].get("endIndex", 2) - 1   # -1 to leave the final \n

    if end_idx > start_idx:
        print(f"[d5] Deleting {end_idx - start_idx} chars (index {start_idx}→{end_idx})...")
        docs.documents().batchUpdate(
            documentId=D5_DOC_ID,
            body={"requests": [
                {"deleteContentRange": {"range": {"startIndex": start_idx, "endIndex": end_idx}}}
            ]}
        ).execute()
        time.sleep(2)

    # ── Step 2: Insert all content blocks ────────────────────────────────────
    # Build the full text first (insert from the end to keep indices stable,
    # or insert at index 1 and build forward)
    # Simpler: insert all as one big batch at index 1, then apply styles.

    print("[d5] Building text content...")

    # Collect lines with style metadata
    lines = []   # list of (text, named_style)
    for style, text in DOCUMENT:
        if style == "TITLE":
            lines.append((text + "\n", "TITLE"))
        elif style == "SUBTITLE":
            lines.append((text + "\n", "SUBTITLE"))
        elif style == "HEADING_1":
            lines.append((text + "\n", "HEADING_1"))
        elif style == "HEADING_2":
            lines.append((text + "\n", "HEADING_2"))
        elif style == "BULLET":
            lines.append(("• " + text + "\n", "NORMAL_TEXT"))
        elif style == "FIGURE":
            lines.append((text + "\n", "NORMAL_TEXT"))
        elif style == "NOTE":
            lines.append((text + "\n", "NORMAL_TEXT"))
        else:  # BODY
            lines.append((text + "\n", "NORMAL_TEXT"))

    # Insert in batches of 40 to stay under API limits
    # Insert from the END so indices don't shift (or insert all at once from index 1)
    # Strategy: insert everything at index 1, building the document forward.
    # We need to track cumulative character positions.

    INSERT_AT = 1
    batch_size = 30
    flat_text = "".join(t for t, _ in lines)

    # Insert in chunks
    pos = 0
    total = len(flat_text)
    chunks = []
    while pos < total:
        end = min(pos + 10_000, total)   # 10k chars per request
        chunks.append(flat_text[pos:end])
        pos = end

    print(f"[d5] Inserting {total:,} chars in {len(chunks)} chunks...")
    for i, chunk in enumerate(chunks):
        # Find current end of doc to insert at
        body = docs.documents().get(documentId=D5_DOC_ID).execute()["body"]["content"]
        cur_end = body[-1].get("endIndex", 2) - 1
        docs.documents().batchUpdate(
            documentId=D5_DOC_ID,
            body={"requests": [_make_insert(cur_end, chunk)]}
        ).execute()
        print(f"[d5]   Chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
        time.sleep(0.5)

    # ── Step 3: Apply paragraph styles ───────────────────────────────────────
    # Re-read doc and apply styles by scanning for heading text
    print("[d5] Applying paragraph styles...")
    time.sleep(2)

    body = docs.documents().get(documentId=D5_DOC_ID).execute()["body"]["content"]
    style_requests = []

    # Build a map: paragraph_text → named_style (for headings)
    heading_styles = {}
    for style, text in DOCUMENT:
        if style in ("TITLE", "SUBTITLE", "HEADING_1", "HEADING_2"):
            named = {
                "TITLE": "TITLE",
                "SUBTITLE": "SUBTITLE",
                "HEADING_1": "HEADING_1",
                "HEADING_2": "HEADING_2",
            }[style]
            heading_styles[text.strip()] = named

    for elem in body:
        para = elem.get("paragraph", {})
        if not para:
            continue
        text = "".join(
            r.get("textRun", {}).get("content", "")
            for r in para.get("elements", [])
        ).strip()
        if text in heading_styles:
            si = elem.get("startIndex", 0)
            ei = elem.get("endIndex", si + len(text) + 1)
            style_requests.append(_make_style(si, ei, heading_styles[text]))

    # Apply in batches of 50
    for i in range(0, len(style_requests), 50):
        batch = style_requests[i:i+50]
        docs.documents().batchUpdate(
            documentId=D5_DOC_ID,
            body={"requests": batch}
        ).execute()
        time.sleep(0.5)

    print(f"[d5] Applied {len(style_requests)} heading styles.")

    print(f"\n[d5] Done!")
    print(f"  URL: https://docs.google.com/document/d/{D5_DOC_ID}/edit")
    print()
    print("NEXT STEPS — insert figures manually in the Google Doc:")
    print("  Figure 1  — roses_flow.png              (after §4.1)")
    print("  Figure 2  — human/evidence_gap_map.png  (after §4.2 intro)")
    print("  Figure 3  — geographic_map.png           (after §4.3)")
    print("  Figure 4  — geographic_bar.png           (after Figure 3)")
    print("  Figure 5  — temporal_trends.png          (after §4.4)")
    print("  Figure 6  — methodology_bar.png          (after §4.5)")
    print("  Figure 7  — equity_bar.png               (after §4.6)")
    print("  Figure 8  — saturation.png               (after §4.7)")
    print("  Figure 9  — llm_vs_human.png             (after §4.8)")
    print()
    print("All PNGs are in: scripts/outputs/step16/")
    print("Human EGM PNG:   scripts/outputs/step16/interactive/human/evidence_gap_map.png")
    print("Saturation PNG:  scripts/outputs/step16/saturation.png")


if __name__ == "__main__":
    run()
