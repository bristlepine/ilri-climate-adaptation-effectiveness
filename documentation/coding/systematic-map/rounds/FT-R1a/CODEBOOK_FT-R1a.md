# Systematic Map Codebook — FT-R1a
*ILRI / Bristlepine Resilience Consultants — 2025–2026*

---

## Where to find everything

**[→ Open the FT-R1a folder on Google Drive](https://drive.google.com/drive/folders/1DLifotAW4YcYZZ-5iYd0T0X0Q-N3MOmQ?usp=sharing)**

| What | Where |
|------|-------|
| Papers to code | `FT-R1a pdfs/` subfolder inside the Drive folder above |
| Blank coding sheet | `coding_ft_r1a_XX.csv` inside the Drive folder |
| Your completed sheet | Save as `coding_ft_r1a_INITIALS.csv` (e.g. `coding_ft_r1a_AZ.csv`) and upload back to the same folder |

---

## How to code

1. Download `coding_ft_r1a_XX.csv` and save a copy as `coding_ft_r1a_INITIALS.csv`
2. Open each paper from the `FT-R1a pdfs/` subfolder
3. Fill in all 16 fields for each paper — one row per paper
4. Enter your initials in `coder_id` — same initials on every row
5. Upload your completed file back to the Drive folder when done

**Rules:**

- Code independently — do not discuss with the other coder until instructed
- If a field is not reported in the paper: leave blank or enter `not_reported`
- If you are unsure: make your best judgement and add a note in the `notes` column

---

<div class="page-break"></div>

## Inclusion criteria

These papers are a **calibration sample** — they were hand-picked because they are known to meet all five inclusion criteria. Use this round to practise applying the criteria and calibrate your judgement against other coders. Set `confirmed_include = yes` for all five papers unless you have a strong reason to disagree (note it in `notes`).

---

### 1. Population

**Include if:** Individuals or households engaged in small-scale agricultural production (crop, livestock, fisheries/aquaculture, agroforestry) in an LMIC. Also include marginalized producer groups (women, youth, landless farmers, indigenous peoples, ethnic minorities, migrant/seasonal workers).

**Exclude if:** Large-scale commercial agriculture, agribusiness without smallholder focus, or non-agricultural livelihoods only.

**When in doubt:**

| Scenario | Decision |
|----------|----------|
| Mixed samples (smallholders + larger farms) | INCLUDE — if smallholders are mentioned and analyzed as a distinct group |
| Generic "Farmers" label in LMIC contexts | INCLUDE — unless study clearly focuses on commercial-scale operations |
| Industrial or corporate agriculture | EXCLUDE |

---

### 2. Concept (Adaptation)

**Include if:** Study assesses or documents adaptation processes (learning, decision-making, adoption, participation, behavior change) OR adaptation outcomes (productivity, income, livelihoods, wellbeing, resilience, risk reduction).

**Exclude if:**
- Climate impacts/vulnerability assessment ONLY (no adaptation action measured)
- Mitigation only (reducing emissions)
- General development or agronomy without explicit climate risk framing
- Purely theoretical or conceptual (no intervention or measurement described)

**When in doubt:**

| Scenario | Decision |
|----------|----------|
| Adoption/uptake of climate-smart agriculture or water management | INCLUDE |
| Livelihood diversification in response to climate stress | INCLUDE |
| Vulnerability index or risk score without coping/adaptation responses | EXCLUDE |
| Farmer awareness/perception of climate change alone | EXCLUDE |

---

### 3. Context (Climate + Agriculture)

**Include if:** Agricultural setting (crop, livestock, fisheries, agroforestry) with explicit climate hazard link — drought, flooding, temperature extremes, rainfall variability, or other climate stressors affecting production.

**Exclude if:**
- Non-agricultural sector
- Agricultural study with no climate dimension
- Climate mentioned only as background (not driving the research question)
- Agricultural impacts/responses unrelated to climate

**When in doubt:**

| Scenario | Decision |
|----------|----------|
| "Water scarcity" or "aridity" in semi-arid regions (implies climate) | INCLUDE |
| "Food insecurity" in LMIC rural contexts with adaptation responses | INCLUDE |
| Generic "livelihood change" in non-climate context | EXCLUDE |
| Agriculture + environmental change (without climate specificity) | EXCLUDE |

---

### 4. Methodology

**Include if:** Study describes, evaluates, or applies a measurable method or framework to assess adaptation (indicators, metrics, surveys, qualitative interviews, participatory assessments, models with empirical data). Published between 2005–2025.

**Exclude if:**
- Purely theoretical/conceptual (no method or data)
- Systematic reviews, meta-analyses, or literature reviews
- Pre-2005 publication
- Opinion pieces or policy briefs without empirical method

**When in doubt:**

| Scenario | Decision |
|----------|----------|
| Narrative case study of farmer adaptation | INCLUDE — qualitative empirical method |
| Agent-based model with survey calibration | INCLUDE — model + empirical data |
| Pilot application of an adaptation tool | INCLUDE |
| Crop variety trial without farmer adoption analysis | EXCLUDE |

---

### 5. Geography

**Include if:** Study conducted in one or more Low- or Middle-Income Countries (LMICs) or Global South countries. Single or multi-country studies OK if at least one LMIC is involved.

**Exclude if:** OECD high-income countries only (USA, UK, EU, Australia, Canada, Japan, South Korea, etc.) with no LMIC component.

**When in doubt:**

| Scenario | Decision |
|----------|----------|
| China (all provinces) | INCLUDE — upper-middle-income |
| South Africa | INCLUDE — upper-middle-income |
| Multi-country study (e.g. Peru + Germany) | INCLUDE — Peru qualifies |
| Brazil, Vietnam, Thailand, Iran | INCLUDE — all are LMIC |

---

<div class="page-break"></div>

## Quick reference

| Field | Valid values |
|-------|-------------|
| `publication_type` | `journal_article` `report` `working_paper` `thesis` `other` |
| `geographic_scale` | `local` `sub-national` `national` `multi-country` `regional` |
| `producer_type` | `crop` `livestock` `fisheries_aquaculture` `agroforestry` `mixed` `undefined` |
| `marginalized_subpopulations` | `women` `youth` `people_with_disabilities` `landless` `indigenous_peoples` `ethnic_minorities` `migrant_seasonal_workers` `other` `none_reported` |
| `adaptation_focus` | **free text** |
| `process_outcome_domains` | `knowledge_awareness_learning` `decision_making_planning` `uptake_adoption` `behavioral_change` `participation_coproduction` `institutional_governance` `access_information_services` `yields_productivity` `income_assets` `livelihoods` `wellbeing` `risk_reduction` `resilience_adaptive_capacity` |
| `indicators_measured` | **free text** |
| `methodological_approach` | `qualitative` `quantitative` `participatory` `modeling_with_empirical_validation` `experimental` |
| `purpose_of_assessment` | `research` `project_learning` `program_evaluation` `donor_reporting` `national_reporting` |
| `data_sources` | `surveys` `administrative_data` `remote_sensing` `participatory_methods` `secondary_data` `other` |
| `temporal_coverage` | `cross_sectional` `seasonal` `longitudinal` `repeated_cross_sectional` |
| `cost_data_reported` | `yes` `no` |
| `strengths_and_limitations` | **free text** |
| `lessons_learned` | **free text** |

<div class="page-break"></div>

## The 16 coding fields

### 1. `publication_year`

Year the study was published. If multiple versions exist, use the most recent.

**Valid values:** integer — e.g. `2021`

**Where to look:** The publication year is typically listed at the very top or bottom of the pages, alongside the article's title, volume, and page numbers.

---

### 2. `publication_type`

| Value | Meaning |
|-------|---------|
| `journal_article` | Peer-reviewed journal |
| `report` | Project/programme report, evaluation report, policy brief |
| `working_paper` | Pre-publication working paper or discussion paper |
| `thesis` | PhD or master's dissertation |
| `other` | Book chapter, conference paper, dataset paper |

**Where to look:**

- **Headers, Footers, and Margins:** Check the very top or bottom of pages for journal name, volume, issue, and page numbers—these immediately identify peer-reviewed journal articles.

- **Cover Pages (Grey Literature):** Non-peer-reviewed reports often feature a standalone cover page with the logos and names of publishing organizations (NGOs, World Bank, FAO, research institutes, government ministries).

- **Disclaimers and Acknowledgements:** Look at footnotes on the first page or inside covers. Working papers and discussion papers often include explicit disclaimers stating the research is preliminary, circulating for discussion, or lacks formal peer review.

- **Degree Requirements (Theses/Dissertations):** The title page will typically include text stating "A thesis/dissertation submitted in partial fulfillment of the requirements for the degree of [Master's/PhD]..." followed by the university name.

---

### 3. `country_region`

Country or countries where the study was conducted. Use a region label only if no countries are named.

**Format:** free text; separate multiple entries with semicolons

**Examples:** `Tanzania` | `Ghana; Mali; Burkina Faso` | `East Africa` | `Global`

**Where to look:** Scan the **article's title, abstract, introduction, and Methods / Study Area description**—authors typically state the study location early and clearly.

---

### 4. `geographic_scale`

| Value | Meaning |
|-------|---------|
| `local` | Village, community, single site |
| `sub-national` | District, province, region within a country |
| `national` | One country |
| `multi-country` | Two or more specific named countries |
| `regional` | Broad region without country-level focus |

**Where to look:** Check the **Methods / Study Area / Site Description section**—this is the most definitive place. Authors must clearly state whether their study focused on a single village, a district within a country, the entire country, multiple named countries, or a broad geographic region.

---

### 5. `producer_type`

Select all that apply; separate with semicolons.

| Value | Meaning |
|-------|---------|
| `crop` | Crop farming systems |
| `livestock` | Livestock farming systems |
| `fisheries_aquaculture` | Fisheries or aquaculture systems |
| `agroforestry` | Agroforestry systems |
| `mixed` | Explicitly mixed crop-livestock |
| `undefined` | Generic "smallholder farmers" with no production system specified |

**Where to look:** Look in the Abstract and Introduction: Authors usually state their target population and main agricultural focus right away. If not, check the Methods / Study Area / Site Description sections: This is the most reliable place to look, as authors must describe the economic and agricultural baselines of the communities they are surveying.

---

### 6. `marginalized_subpopulations`

Code only what is **explicitly stated** in the paper — do not infer. Select all that apply; separate with semicolons.

| Value | Meaning |
|-------|---------|
| `women` | Women explicitly targeted or gender-disaggregated analysis |
| `youth` | Young farmers or age-disaggregated outcomes |
| `people_with_disabilities` | Persons with disabilities explicitly named |
| `landless` | Landless households or tenure-insecure groups |
| `indigenous_peoples` | Indigenous communities named |
| `ethnic_minorities` | Ethnic minority groups named |
| `migrant_seasonal_workers` | Migrant or seasonal agricultural workers |
| `other` | Any other marginalised group explicitly named — add a note in `notes` |
| `none_reported` | No marginalised groups mentioned |

**Where to look:**

- **Abstract and Introduction:** Authors often state upfront if their study specifically targets or evaluates a vulnerable subgroup.

- **Methods / Sampling / Study Area sections:** This section details exactly who was surveyed and how they were chosen. Look for descriptions of "purposive sampling" criteria—e.g., "deliberately selecting participants based on gender (if known) and/or youth."

- **Results and Descriptive Statistics (Tables):** Check baseline household characteristics tables and demographic summaries to identify which groups were explicitly studied.

- **Discussion and Conclusion:** Authors often highlight how interventions impact marginalized groups differently—e.g., "inherent resource inequities between men and women constrain adaptation."

---

### 7. `adaptation_focus` *(free text)*

The specific climate adaptation action, intervention, or practice the study tracks. Write 5–15 words based on what the paper describes.

**Examples:**
- `crop varieties and genetics`
- `water management and irrigation`
- `climate information and advisory services`
- `livelihood diversification (non-farm)`
- `agroforestry`
- `livestock management`
- `financial strategies`
- `social and community networking`

**Where to look:**

- **Abstract and Introduction:** Authors usually highlight the primary interventions or adaptation strategies they are investigating right away.

- **Methods / Interventions / Study Design sections:** This is where authors detail the specific strategies tested, promoted, or surveyed in the study.

- **Results and Descriptive Statistics (Tables):** Look for tables detailing adoption rates, summary statistics, or regression variables—these list the exact practices tracked.

- **Discussion and Conclusion:** If the study is a modeling or vulnerability assessment that doesn't test a specific intervention, authors will often propose an adaptation focus in these sections.

- **Appendix / Survey Instrument:** If the researchers include their questionnaire, it will show the exact adaptation options they asked farmers about.

**Example:** In the Malawi study, the "Sampling methods and agroecological experiments" section provides a categorized list of the exact methods farmers selected: integration of trees, soil fertility methods, crop diversification, and livelihood diversification (e.g., small livestock).

---

### 8. `process_outcome_domains`

What the study measures. Select all that apply; separate with semicolons.

**Rule:** Code only domains that are **explicitly measured** — not domains that could be inferred. If the paper measures yield but not income, code `yields_productivity` only.

| Value | Meaning |
|-------|---------|
| `knowledge_awareness_learning` | Awareness of climate change, knowledge of options, learning outcomes |
| `decision_making_planning` | Farmer decision-making, use of forecasts in planning |
| `uptake_adoption` | Rates of adoption of a practice or technology |
| `behavioral_change` | Changes in farming or land management behaviour |
| `participation_coproduction` | Participation in programmes, co-design, farmer group engagement |
| `institutional_governance` | Policy uptake, institutional change, governance |
| `access_information_services` | Access to extension, credit, markets, climate information |
| `yields_productivity` | Crop yield, livestock productivity, fish catch |
| `income_assets` | Household income, asset accumulation |
| `livelihoods` | Food security, livelihood diversification |
| `wellbeing` | Health, nutrition, subjective wellbeing |
| `risk_reduction` | Reduced exposure or sensitivity to climate shocks |
| `resilience_adaptive_capacity` | Adaptive capacity indices, resilience scores, vulnerability indices |

**Where to look:**

- **Methods / Variable Descriptions / Summary Statistics (Tables):** Authors will list the specific process indicators they tracked in these sections.

- **Results / Findings sections:** Qualitative and quantitative changes in knowledge, learning, and participation are reported here.

- **Discussion and Conclusion:** Authors often interpret how an intervention affected social processes, learning, or institutional dynamics in these sections.

**Example:** In the Central Vietnam study, the results section explicitly reports on farmers' "capacity to learn," "capacity to decide," and "capacity to act" in a dedicated results subsection.

---

### 9. `indicators_measured` *(free text)*

The specific indicators or metrics used. Extract from the methods or results sections. Include units where stated.

**Examples:**
- `yield (t/ha), income (USD/season), food security score (HFIAS)`
- `adoption rate (%), area under improved variety (ha)`
- `climate resilience index score`

**Where to look:**

- **Tables (Variable Descriptions and Summary Statistics):** Authors almost always consolidate their indicators into comprehensive tables for clarity.

- **Methodology / Data Collection / Variables sections:** The narrative text in these sections describes the specific tools, scales, or census data used to capture the indicators.

- **Appendix / Survey Instrument:** If you need to see the exact questions asked to construct an indicator, the appendix is the most reliable place. For example, the Northern Ghana study includes "Appendix B: Survey questionnaire with indicators scoring criterion," which shows the precise multiple-choice questions used and how points were assigned to measure economic resources, social capital, and technology.

---

### 10. `methodological_approach`

Select all that apply; separate with semicolons.

| Value | Meaning |
|-------|---------|
| `qualitative` | Qualitative research design (interviews, focus groups, ethnography) |
| `quantitative` | Quantitative research design (surveys, statistical analysis) |
| `participatory` | Only when participation is the **primary** design (PRA, photovoice) — not when farmers are merely surveyed |
| `modeling_with_empirical_validation` | Crop/climate/agent-based models validated with field data |
| `experimental` | RCTs, natural experiments, quasi-experimental approaches |

**Where to look:**

- **Methods / Methodology / Materials and Methods section:** Authors typically dedicate this entire section to explaining their research design, often breaking it down into specific subsections.

- **Data Collection / Sampling subsections:** Here, authors detail whether they used surveys, expert interviews, focus groups, or secondary data.

- **Data Analysis / Empirical Approach subsection:** This subsection explains the statistical or qualitative frameworks used to evaluate the data.

- **Abstract:** Authors almost always provide a concise, one-to-two-sentence summary of their methods right at the beginning of the paper.

- **Introduction (final paragraphs):** Right after stating their research objectives at the end of the introduction, authors frequently include a brief preview of their methodology.

**Example:** The Malawi study introduces its methodology by stating: "A participatory research approach combined with pre-post longitudinal study design was used to examine changes..."

---

### 11. `purpose_of_assessment`

| Value | Meaning |
|-------|---------|
| `research` | Academic study with no programme affiliation |
| `project_learning` | M&E embedded in an ongoing project |
| `program_evaluation` | Formal evaluation of a completed programme |
| `donor_reporting` | Report produced for a donor or funder |
| `national_reporting` | Contributing to NAPs, NDCs, or national monitoring frameworks |

**Where to look:**

- **Abstract:** Authors typically provide a high-level summary of why the study was conducted right at the beginning of the document.

- **Introduction (end of section):** Standard academic writing conventions dictate that authors explicitly outline their goals, objectives, or research questions at the very end of the introduction section, right before diving into the methodology. **Example:** "The specific research objective of this study is to test whether agroecological farming methods... can improve food security..."

- **Conclusion / Summary:** Authors often briefly restate the core purpose of their assessment in the concluding section when summarising what their study ultimately set out to achieve and what it found.

---

### 12. `data_sources`

Select all that apply; separate with semicolons.

| Value | Meaning |
|-------|---------|
| `surveys` | Primary household or farm surveys |
| `administrative_data` | Government or programme records |
| `remote_sensing` | Satellite or aerial imagery |
| `participatory_methods` | PRA, focus groups, photovoice as a data source |
| `secondary_data` | Existing datasets, literature, or other secondary sources |
| `other` | Any data source not listed above |

**Where to look:**

- **Methods / Methodology / Materials and Methods section:** This is the primary and most detailed location. Authors almost always include specific subsections dedicated to explaining where their data came from. **Example:** In the Northern Ghana study, the "Data collection" subsection under "Research methods" explicitly outlines the use of primary data (expert interviews and structured questionnaires) and secondary data (a systematic literature review).

- **Data / Data and Variables sections:** Sometimes, authors dedicate a top-level section specifically to the data used, especially in economic or quantitative studies relying heavily on existing secondary datasets or household panels. **Example:** The East Africa study features a section titled "Data, description of variables, and summary statistics," which explains that their data came from a baseline household survey collected by the CGIAR Research Program on Climate Change, Agriculture, and Food Security (CCAFS).

---

### 13. `temporal_coverage`

| Value | Meaning |
|-------|---------|
| `cross_sectional` | Single point in time or single season |
| `seasonal` | Intra-annual, across one or a few seasons |
| `longitudinal` | Multi-year, panel data, repeated measures with the same individuals |
| `repeated_cross_sectional` | Multi-year, repeated measures with different individuals each time |

**Where to look:**

- **Abstract:** Authors frequently summarize the study's timeframe or experimental design right at the beginning. **Example:** The Malawi study abstract explicitly mentions a "four-year study," baseline and follow-up surveys conducted in 2011 and 2013, and the use of "Longitudinal mixed effects models."

- **Methodology / Data Collection / Study Design sections:** This is the most definitive place to find temporal coverage—authors must detail exactly when and how often they collected their data.

- **Data and Variables / Climate Modeling sections:** If the study relies on secondary data, historical records, or predictive modeling (rather than field surveys), these sections will define the exact time periods analyzed.

---

### 14. `cost_data_reported`

| Value | Meaning |
|-------|---------|
| `yes` | Paper explicitly reports cost figures or cost-effectiveness ratios |
| `no` | No cost data — a qualitative mention of "limited resources" counts as `no` |

**Where to look:**

- **Results / Economic Evaluation sections:** If the core purpose of the study involves an economic or financial assessment, the exact cost figures, cost-benefit analyses, or cost-effectiveness ratios will be reported in the results, usually accompanied by data tables.

- **Discussion and Conclusion:** Authors may summarize the exact financial requirements for implementing an adaptation strategy at a regional scale, or they may compare its cost-effectiveness against other measures.

**Quick-search tip:** Use your document viewer's search function to scan for currency symbols or acronyms (e.g., $, USD, VND, Euro) and economic keywords like "cost-effectiveness," "CBA" (cost-benefit analysis), "expenditure," or "budget."

---

### 15. `strengths_and_limitations` *(free text)*

Author-reported strengths and limitations. Look in the Discussion, Conclusion, and Limitations sections. Extract verbatim where possible.

**Format:** `Strength: <text>. Limitation: <text>.`

**Where to look:**

- **Discussion Section:** This is the most common place to find both strengths and limitations. As authors interpret their findings, they typically reflect on the robustness of their study design, highlight how their research makes strong contributions to existing literature, and acknowledge any methodological, sample size, or data constraints.

- **Conclusion:** Authors often briefly summarize the main strengths (e.g., "This study provides novel insights into...") and note any boundaries to their findings, frequently tying these limitations directly to recommendations for future research.

- **Dedicated "Limitations" subsection:** Some journals require authors to include a specific subsection—usually at the very end of the Discussion—explicitly detailing the study's constraints.

---

### 16. `lessons_learned` *(free text)*

Key lessons or recommendations the authors report. 1–3 sentences. Focus on what they say should be done differently — not the findings themselves. Enter `not_reported` if absent.

**Where to look:** Look at the Discussion section: This is the primary area where authors interpret their results, explain why certain interventions worked or failed, and extract practical insights. The Conclusion (or specific "Implications" and "Recommendations" subsections): Authors consistently use the final sections of the paper to synthesise their overarching lessons for policymakers, donors, or future researchers. The Abstract (Final Sentences): Authors frequently distill their most critical lesson learned into the final lines of the abstract.

---

*Questions: zarrar@bristlep.com*
