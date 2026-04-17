# LLM Extraction Criteria — sysmap v1

**Version:** sysmap_v1
**Used in rounds:** FT-R1a
**Created:** 2026-04-14
**Previous version:** n/a (initial)
**Changes from previous:** n/a — initial criteria

---

> **Human coders:** see [`../CODEBOOK.md`](../CODEBOOK.md) — this file is the LLM's version of the same instructions.

## Task description

You are extracting structured data from a scientific paper. The paper has been identified as relevant to a systematic map examining methods used to track climate adaptation processes and outcomes for smallholder agricultural producers in low- and middle-income countries (LMICs).

Your task is to fill in 19 fields for this paper, following the field definitions and decision rules below. Return your output as a CSV row with values in the order listed. Use semicolons to separate multiple values within a single field. Use `not_reported` if a field cannot be determined from the text.

---

## Field-by-field extraction instructions

### 1. `publication_year`
Extract the year of publication as a 4-digit integer. If multiple versions exist (e.g., preprint then journal), use the most recent.

### 2. `publication_type`
Choose one:
- `journal_article` — peer-reviewed journal paper (including open-access journals)
- `report` — project, programme, or evaluation report; policy brief
- `working_paper` — pre-publication working paper or discussion paper
- `thesis` — PhD or master's dissertation
- `other` — book chapter, conference paper, dataset paper

### 3. `country_region`
List the country or countries where the study was conducted, separated by semicolons. Use the most specific label available:
- Named countries: `Kenya; Ethiopia`
- Region only (no countries named): `Sub-Saharan Africa`
- Global multi-country review with no specific country: `Global`

### 4. `geographic_scale`
Choose one:
- `local` — village, community, watershed, single site
- `sub-national` — district, province, state, region within a country
- `national` — one country, national-level analysis
- `multi-country` — two or more specific countries
- `regional` — broad region (West Africa, South Asia) without country-level focus

### 5. `producer_type`
Select all that apply, separated by semicolons:
- `crop` — crop farmers (including rice, maize, vegetable, fruit)
- `livestock` — pastoralists, agropastoralists, dairy/beef farmers
- `fisheries_aquaculture` — artisanal fishing, small-scale aquaculture
- `agroforestry` — tree crop farmers, silvopastoral systems
- `mixed` — mixed crop-livestock systems, or "smallholder farmers" with no further specification

### 6. `marginalized_subpopulations`
Select all that apply, separated by semicolons. Code only what is explicitly stated:
- `women` — women targeted or gender-disaggregated analysis
- `youth` — young farmers or age-disaggregated outcomes
- `landless` — landless or near-landless households
- `indigenous_peoples` — Indigenous communities named
- `ethnic_minorities` — ethnic minority groups named
- `migrant_seasonal_workers` — migrant or seasonal agricultural workers
- `none_reported` — no marginalized groups mentioned

### 7. `adaptation_focus` [EMERGENT — free text]
Describe the specific climate adaptation action, intervention, or practice the paper tracks or evaluates. Be specific and concise (5–15 words). Do not use generic terms like "climate change adaptation" alone.

Examples:
- `drought-tolerant maize varieties`
- `seasonal climate forecast dissemination for smallholder decision-making`
- `index-based livestock insurance`
- `community-based watershed management`
- `farmer field schools for climate-resilient agriculture`

### 8. `domain_type`
Choose one or more, separated by semicolons:
- `adaptation_process` — the study measures *how* farmers adopt, learn, participate, or engage (the steps toward adaptation)
- `adaptation_outcome` — the study measures *what happened* as a result (yields, income, wellbeing, resilience)
- `both` — study tracks both a process (e.g. adoption rate) and an outcome (e.g. yield change)

Decision rule: a study measuring adoption rates only = `adaptation_process`. A study measuring yield impacts only = `adaptation_outcome`. A study measuring both = `both`.

### 9. `process_outcome_domains`
Select all that apply, separated by semicolons.

**Process domains:**
- `knowledge_awareness_learning` — knowledge of climate change or adaptation options; learning outcomes; awareness surveys
- `decision_making_planning` — farmer decision-making processes; use of information in planning
- `uptake_adoption` — rates of practice or technology adoption
- `behavioral_change` — changes in farming or land management behaviour
- `participation_coproduction` — participation in programmes, co-design, farmer group engagement
- `institutional_governance` — policy uptake, institutional change, governance
- `access_information_services` — access to extension, credit, markets, climate information

**Outcome domains:**
- `yields_productivity` — crop yield, livestock productivity, fish catch
- `income_assets` — household income, asset accumulation, savings
- `livelihoods` — broader livelihood outcomes, food security, diversification
- `wellbeing` — health, nutrition, subjective wellbeing
- `risk_reduction` — reduced exposure or sensitivity to climate shocks
- `resilience_adaptive_capacity` — adaptive capacity, resilience indices, vulnerability indices

### 10. `indicators_measured` [EMERGENT — free text]
List the specific indicators or metrics used in the study, comma-separated. Extract actual indicators mentioned in the methods or results section.

Examples:
- `yield (t/ha), income (USD/season), food security score (HFIAS)`
- `adoption rate (%), area under improved variety (ha)`
- `self-reported adaptive capacity index (5-point Likert scale)`

### 11. `methodological_approach`
Choose one:
- `qualitative` — interviews, focus groups, ethnography, document analysis only
- `quantitative` — surveys, trials, observational data, statistical analysis
- `mixed_methods` — explicit combination of qualitative and quantitative
- `participatory` — participatory rural appraisal, participatory mapping, photovoice, farmer-led trials — use when participation is the *primary* design
- `modeling_with_empirical_validation` — crop, climate, or agent-based models validated with field data

### 12. `purpose_of_assessment`
Choose one:
- `research` — academic study with no specific programme affiliation
- `program_evaluation` — commissioned evaluation of a specific programme
- `project_learning` — monitoring and learning embedded in an ongoing project
- `donor_reporting` — explicitly for funder accountability
- `national_reporting` — contributing to NAPs, NDCs, national monitoring frameworks

### 13. `data_sources`
Select all that apply, separated by semicolons:
- `surveys` — household surveys, farm questionnaires
- `administrative_data` — government records, market data, national statistics
- `remote_sensing` — satellite imagery, drone data
- `participatory_methods` — PRA, FGDs, key informant interviews
- `secondary_data` — existing datasets, meta-analysis inputs, systematic review data

### 14. `temporal_coverage`
Choose one:
- `cross_sectional` — single point in time or single season
- `seasonal` — tracks across one or a few seasons (intra-annual)
- `longitudinal` — multi-year data, panel data, repeated measures

### 15. `cost_data_reported`
- `yes` — the paper explicitly reports cost figures, cost-effectiveness ratios, or resource requirements for the adaptation
- `no` — cost not reported, or only mentioned qualitatively (e.g. "limited resources")

### 16. `equity_inclusion`
Select all that apply, separated by semicolons. Use `none_reported` if none:
- `gender` — gender-disaggregated analysis or gender-targeted intervention
- `youth` — explicit focus on young farmers or age-disaggregated outcomes
- `land_tenure` — land access or tenure security as an explicit variable
- `disability` — explicit inclusion of persons with disabilities
- `other` — any other equity dimension (e.g. caste, religion)
- `none_reported` — equity not mentioned

### 17. `strengths_and_limitations`
Extract author-reported strengths and limitations as free text. Label each:
- `Strength: <text>` for strengths
- `Limitation: <text>` for limitations

Extract verbatim where possible. Look in the Discussion, Conclusion, and Limitations sections. If the paper has no explicit strengths section but does state limitations, record only the limitations. If neither are stated, enter `not_reported`.

### 18. `lessons_learned`
Summarise key lessons or recommendations reported by the authors in 1–3 sentences. Focus on what the authors say should be done differently or recommend for policy/practice. This is distinct from findings — it is the authors' evaluative take.

If no lessons or recommendations are stated, enter `not_reported`.

### 19. `validity_notes`
Note any issues that affect how much weight this study can carry in the synthesis. Flag:
- Small sample (n < 30)
- Single-site study (limits generalisability)
- Self-reported outcomes (social desirability bias risk)
- Short follow-up (< 1 season)
- No control or comparison group
- Attrition not reported

If no validity concerns are apparent, enter `none_flagged`.

---

## Output format

Return a single CSV row with values in this column order:

```
publication_year, publication_type, country_region, geographic_scale,
producer_type, marginalized_subpopulations, adaptation_focus, domain_type,
process_outcome_domains, indicators_measured, methodological_approach,
purpose_of_assessment, data_sources, temporal_coverage, cost_data_reported,
equity_inclusion, strengths_and_limitations, lessons_learned, validity_notes
```

Use semicolons within fields for multi-select values. Do not use commas within free-text fields (use semicolons or periods instead to avoid CSV parsing errors). Wrap free-text fields in double quotes.
