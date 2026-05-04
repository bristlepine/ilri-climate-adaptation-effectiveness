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

<div style="page-break-before: always"></div>

## Inclusion criteria

These papers are a calibration sample for the systematic map. Read each paper and confirm it meets **all five criteria** below. The paper must satisfy ALL five — if it fails any one, note the reason in `notes`.

---

### 1. Population

**Include if:** Individuals or households engaged in small-scale agricultural production (crop, livestock, fisheries/aquaculture, agroforestry) in an LMIC. Also include marginalized producer groups (women, youth, landless farmers, indigenous peoples, ethnic minorities, migrant/seasonal workers).

**Exclude if:** Large-scale commercial agriculture, agribusiness without smallholder focus, or non-agricultural livelihoods only.

**When in doubt:**
- Mixed samples (smallholders + larger farms): INCLUDE if smallholders are mentioned and analyzed
- "Farmers" in LMIC contexts without explicit "smallholder" label: INCLUDE unless clearly commercial scale
- Industrial or corporate agriculture: EXCLUDE

---

### 2. Concept (Adaptation)

**Include if:** Study assesses or documents adaptation processes (learning, decision-making, adoption, participation, behavior change) OR adaptation outcomes (productivity, income, livelihoods, wellbeing, resilience, risk reduction).

**Exclude if:**
- Climate impacts/vulnerability assessment ONLY (no adaptation action measured)
- Mitigation only (reducing emissions)
- General development or agronomy without explicit climate risk framing
- Purely theoretical or conceptual (no intervention or measurement described)

**When in doubt:**
- Adoption/uptake of climate-smart agriculture or water management: INCLUDE
- Livelihood diversification in response to climate stress: INCLUDE
- Vulnerability index or risk score without coping/adaptation responses: EXCLUDE
- Farmer awareness/perception of climate change alone: EXCLUDE

---

### 3. Context (Climate + Agriculture)

**Include if:** Agricultural setting (crop, livestock, fisheries, agroforestry) with explicit climate hazard link — drought, flooding, temperature extremes, rainfall variability, or other climate stressors affecting production.

**Exclude if:**
- Non-agricultural sector
- Agricultural study with no climate dimension
- Climate mentioned only as background (not driving the research question)
- Agricultural impacts/responses unrelated to climate

**When in doubt:**
- "Water scarcity" or "aridity" in semi-arid regions (implies climate): INCLUDE
- "Food insecurity" in LMIC rural contexts with adaptation responses: INCLUDE
- Generic "livelihood change" in non-climate context: EXCLUDE
- Agriculture + environmental change (without climate specificity): EXCLUDE

---

### 4. Methodology

**Include if:** Study describes, evaluates, or applies a measurable method or framework to assess adaptation (indicators, metrics, surveys, qualitative interviews, participatory assessments, models with empirical data). Published between 2005–2025.

**Exclude if:**
- Purely theoretical/conceptual (no method or data)
- Systematic reviews, meta-analyses, or literature reviews
- Pre-2005 publication
- Opinion pieces or policy briefs without empirical method

**When in doubt:**
- Narrative case study of farmer adaptation: INCLUDE (qualitative empirical method)
- Agent-based model with survey calibration: INCLUDE (model + data)
- Pilot application of an adaptation tool: INCLUDE
- Crop variety trial without farmer adoption analysis: EXCLUDE

---

### 5. Geography

**Include if:** Study conducted in one or more Low- or Middle-Income Countries (LMICs) or Global South countries. Single or multi-country studies OK if at least one LMIC is involved.

**Exclude if:** OECD high-income countries only (USA, UK, EU, Australia, Canada, Japan, South Korea, etc.) with no LMIC component.

**When in doubt:**
- China (all provinces): INCLUDE (upper-middle-income)
- South Africa: INCLUDE (upper-middle-income)
- Multi-country study (e.g. Peru + Germany): INCLUDE (Peru qualifies)
- Brazil, Vietnam, Thailand, Iran: INCLUDE (all are LMIC)

---

<div style="page-break-before: always"></div>

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

---

<div style="page-break-before: always"></div>

## The 16 coding fields

### 1. `publication_year`

Year the study was published. If multiple versions exist, use the most recent.

**Valid values:** integer — e.g. `2021`

---

### 2. `publication_type`

| Value | Meaning |
|-------|---------|
| `journal_article` | Peer-reviewed journal |
| `report` | Project/programme report, evaluation report, policy brief |
| `working_paper` | Pre-publication working paper or discussion paper |
| `thesis` | PhD or master's dissertation |
| `other` | Book chapter, conference paper, dataset paper |

---

### 3. `country_region`

Country or countries where the study was conducted. Use a region label only if no countries are named.

**Format:** free text; separate multiple entries with semicolons

**Examples:** `Tanzania` | `Ghana; Mali; Burkina Faso` | `East Africa` | `Global`

---

### 4. `geographic_scale`

| Value | Meaning |
|-------|---------|
| `local` | Village, community, single site |
| `sub-national` | District, province, region within a country |
| `national` | One country |
| `multi-country` | Two or more specific named countries |
| `regional` | Broad region without country-level focus |

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

**Where to look:** Abstract and Introduction (authors state target population and agricultural focus early); Methods / Study Area / Site Description (most reliable — authors describe the agricultural baseline of communities surveyed).

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

**Where to look:** Abstract and Introduction (often stated immediately if the study targets a vulnerable subgroup); Methods / Sampling (look for purposive sampling criteria, e.g. "deliberately selected based on gender or youth status").

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

**Where to look:** Abstract and Introduction (primary interventions highlighted early); Methods / Study Design (specific strategies tested or surveyed); Results tables (adoption rates and regression variables list exact practices tracked); Discussion and Conclusion (if the study is a modeling or vulnerability assessment, the adaptation focus may be proposed here rather than tested); Appendix / Survey Instrument (if included, shows exact adaptation options asked about).

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

**Where to look:** Methods / Variable Descriptions / Summary Statistics tables (specific indicators listed); Results (qualitative and quantitative changes reported here); Discussion and Conclusion (how the intervention affected social processes, learning, or institutional dynamics).

---

### 9. `indicators_measured` *(free text)*

The specific indicators or metrics used. Extract from the methods or results sections. Include units where stated.

**Examples:**
- `yield (t/ha), income (USD/season), food security score (HFIAS)`
- `adoption rate (%), area under improved variety (ha)`
- `climate resilience index score`

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

---

### 11. `purpose_of_assessment`

| Value | Meaning |
|-------|---------|
| `research` | Academic study with no programme affiliation |
| `project_learning` | M&E embedded in an ongoing project |
| `program_evaluation` | Formal evaluation of a completed programme |
| `donor_reporting` | Report produced for a donor or funder |
| `national_reporting` | Contributing to NAPs, NDCs, or national monitoring frameworks |

**Where to look:** Abstract (high-level summary of why the study was conducted); Introduction (research questions or objectives are typically stated explicitly at the end of the introduction); Conclusion (authors restate core purpose when summarising findings).

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

---

### 13. `temporal_coverage`

| Value | Meaning |
|-------|---------|
| `cross_sectional` | Single point in time or single season |
| `seasonal` | Intra-annual, across one or a few seasons |
| `longitudinal` | Multi-year, panel data, repeated measures with the same individuals |
| `repeated_cross_sectional` | Multi-year, repeated measures with different individuals each time |

**Where to look:** Methods / Data Collection / Study Design (most definitive — authors detail when and how often data were collected); Data and Variables / Climate Modeling sections (for studies using secondary data or historical records, these define the time period analysed).

---

### 14. `cost_data_reported`

| Value | Meaning |
|-------|---------|
| `yes` | Paper explicitly reports cost figures or cost-effectiveness ratios |
| `no` | No cost data — a qualitative mention of "limited resources" counts as `no` |

---

### 15. `strengths_and_limitations` *(free text)*

Author-reported strengths and limitations. Look in the Discussion, Conclusion, and Limitations sections. Extract verbatim where possible.

**Format:** `Strength: <text>. Limitation: <text>.`

---

### 16. `lessons_learned` *(free text)*

Key lessons or recommendations the authors report. 1–3 sentences. Focus on what they say should be done differently — not the findings themselves. Enter `not_reported` if absent.

**Where to look:** Discussion (primary area where authors interpret results and extract practical insights); Conclusion / Implications / Recommendations subsections; Abstract (final sentences often distill the most critical lesson).

---

*Questions: zarrar@bristlep.com*
