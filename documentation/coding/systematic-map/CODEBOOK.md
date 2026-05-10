# Systematic Map Codebook — D5.6
*ILRI / Bristlepine Resilience Consultants — 2025–2026*

> **Scope:** Systematic map data extraction only (D4/D5). Not the systematic review (D8/D9).

---

## The 16 coding fields

Use this as your reference while coding. Valid values must be used exactly as written (except free-text fields, marked ⚠).

---

### 1. `publication_year`
Year the study was published. Use the most recent version date if both preprint and journal exist.
**Valid values:** integer, e.g. `2021`

---

### 2. `publication_type`

| Value | Meaning |
|-------|---------|
| `journal_article` | Peer-reviewed journal (including open-access) |
| `report` | Project/programme report, evaluation report, policy brief |
| `working_paper` | Pre-publication working paper or discussion paper |
| `thesis` | PhD or master's dissertation |
| `other` | Book chapter, conference paper, dataset paper |

---

### 3. `country_region`
Country or countries where the study was conducted. Use region label only if no countries are named.
**Valid values:** free text, semicolons between multiple entries

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
Select all that apply, semicolons between multiple values.

| Value | Meaning |
|-------|---------|
| `crop` | Crop farming systems |
| `livestock` | Livestock farming systems |
| `fisheries_aquaculture` | Fisheries or aquaculture systems |
| `agroforestry` | Agroforestry systems |
| `mixed` | Explicitly mixed crop-livestock |
| `undefined` | Generic "smallholder farmers" with no further system specified |

---

### 6. `marginalized_subpopulations`
Code only what is explicitly stated — do not infer. Select all that apply.

| Value | Meaning |
|-------|---------|
| `women` | Women explicitly targeted or gender-disaggregated analysis |
| `youth` | Young farmers or age-disaggregated outcomes |
| `people_with_disabilities` | Persons with disabilities explicitly named |
| `landless` | Landless households or tenure-insecure groups |
| `indigenous_peoples` | Indigenous communities named |
| `ethnic_minorities` | Ethnic minority groups named |
| `migrant_seasonal_workers` | Migrant or seasonal agricultural workers |
| `other` | Any other marginalised group explicitly named (e.g. caste-based groups, religious minorities) — add a brief note in the `notes` field |
| `none_reported` | No marginalised groups mentioned |

---

### 7. `adaptation_focus` ⚠ *free text — emergent*
The specific climate adaptation action, intervention, or practice the study tracks. 5–15 words.

**Examples:** `crop management practices` | `crop varieties and genetics` | `water management and irrigation` | `soil and land management` | `agroforestry` | `livestock management` | `livelihood diversification (non-farm)` | `financial strategies` | `climate information and advisory services` | `migration and mobility` | `social and community networking` | `post-harvest and consumption adjustments`

---

### 8. `process_outcome_domains`
Select all that apply, semicolons between multiple values.

| Code | Meaning |
|------|---------|
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

---

### 9. `indicators_measured` ⚠ *free text*
The specific indicators or metrics used. Extract actual indicators from methods/results. Include units.

**Examples:** `yield (t/ha), income (USD/season), food security score (HFIAS)` | `adoption rate (%), area under improved variety (ha)`

---

### 10. `methodological_approach`
Select all that apply, semicolons between multiple values. For explicitly mixed-methods studies: select both `qualitative` and `quantitative`.

| Value | Meaning |
|-------|---------|
| `qualitative` | Qualitative research design (interviews, focus groups, ethnography) |
| `quantitative` | Quantitative research design (surveys, statistical analysis) |
| `participatory` | Only when participation IS the primary design (PRA, photovoice) — not when farmers are merely surveyed |
| `modeling_with_empirical_validation` | Crop/climate/agent-based models validated with field data |
| `experimental` | RCTs, natural experiments, quasi-experimental approaches — purely lab/field experiments without social application excluded |

---

### 11. `purpose_of_assessment`

| Value | Meaning |
|-------|---------|
| `research` | Academic study with no programme affiliation |
| `project_learning` | M&E embedded in an ongoing project |
| `program_evaluation` | Formal evaluation of a completed programme |
| `donor_reporting` | Report produced for a donor/funder |
| `national_reporting` | Contributing to NAPs, NDCs, national monitoring frameworks |

---

### 12. `data_sources`
Select all that apply.

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
| `repeated_cross_sectional` | Multi-year, repeated measures with different individuals/units each time |

---

### 14. `cost_data_reported`

| Value | Meaning |
|-------|---------|
| `yes` | Paper explicitly reports cost figures or cost-effectiveness ratios |
| `no` | No cost data reported — qualitative mention of "limited resources" counts as `no` |

---

### 15. `strengths_and_limitations` ⚠ *free text*
Author-reported strengths and limitations combined (protocol Table 3). Label each component.

**Format:** `Strength: <text>. Limitation: <text>.`
Look in Discussion, Conclusion, and Limitations sections. Extract verbatim where possible.

---

### 16. `lessons_learned` ⚠ *free text*
Key lessons or recommendations the authors report. 1–3 sentences. Focus on what they say should be done differently — not findings. Enter `not_reported` if absent.

---

## Reconciliation (Step 2)

**Initial round (FT-R1a, 5 papers) — full reconciliation:**

1. Open both sheets side by side
2. For each paper and each field:
   - Agree → enter that value in `reconciled_ft_r1a.csv`
   - Disagree → discuss until you reach agreement, then enter it
3. Record the reason for any substantive disagreement in `reconciliation_notes`
4. Cannot agree → escalate to lead researcher

Do not average numeric fields — always discuss to a single agreed value.

**Subsequent rounds — random subset convergence:**

Each coder independently codes their assigned papers. After each round, a random subset is compared across coders to check agreement. The goal is convergence — when agreement is stable across two consecutive subsets, coding is considered consistent. Subset size agreed with the lead researcher before each round.

---

## Inter-rater reliability (IRR)

After reconciliation, run:

```bash
python scripts/step11_irr_analysis.py \
  --h1 documentation/coding/systematic-map/rounds/FT-R1a/coding_H1.csv \
  --h2 documentation/coding/systematic-map/rounds/FT-R1a/coding_H2.csv \
  --gold documentation/coding/systematic-map/rounds/FT-R1a/reconciled.csv \
  --llm documentation/coding/systematic-map/rounds/FT-R1a/coding_LLM.csv \
  --out documentation/coding/systematic-map/rounds/FT-R1a/irr_R1a.json
```

### Kappa interpretation

| κ | Interpretation | Action |
|---|---|---|
| < 0.20 | Slight | Major revision — field may need redesign |
| 0.21–0.40 | Fair | Significant revision — add examples, tighten definitions |
| 0.41–0.60 | Moderate | Revision needed — review disagreement patterns |
| 0.61–0.80 | Substantial | Approaching threshold — fine-tuning only |
| > 0.80 | Almost perfect | Ready to lock |

**Threshold to lock a closed field: κ ≥ 0.60 AND sensitivity ≥ 0.80 in the same batch.**

### Emergent field (`adaptation_focus`)
Tracked by category saturation, not kappa. Saturated when no new codes appear across two consecutive batches.

---

## Stopping rules

- **Closed fields** — locked once κ ≥ 0.60 AND sensitivity ≥ 0.80 in the same batch. Locked fields are not retested.
- **Emergent field** — `adaptation_focus` saturated when no new codes appear across two consecutive batches.
- **Full corpus run** — proceed to step15 once all closed fields are locked AND `adaptation_focus` is saturated.

---

## How rounds work

### Round naming

| Label | Who codes | Papers |
|-------|-----------|--------|
| `FT-R1a` | H1 + H2 + LLM independently | First 5 papers |
| `FT-R1b` | **LLM only** (re-codes same R1a papers after criteria update) | Same 5 papers as R1a |
| `FT-R2a` | H1 + H2 + LLM independently | Next 5 papers (6–10) |
| `FT-R2b` | **LLM only** (re-codes R2a papers after criteria update) | Same 5 as R2a |

**`a` rounds** = full human + LLM pass on new papers. **`b` rounds** = LLM verification only, same papers, after criteria update. Human coders only ever code in `a` rounds.

### Each round — six steps

```
Step 1 — Code independently
  H1, H2, and LLM all code the same 5 papers. No discussion.
  ↓
Step 2 — Reconcile (H1 + H2 only)
  Compare field by field. Agree on gold standard → reconciled.csv
  ↓
Step 3 — Compute IRR
  Run step11. Kappa + sensitivity per field and overall.
  ↓
Step 4 — Review disagreements
  Which fields underperformed? Was guidance unclear? Categories missing?
  ↓
Step 5 — Update codebook and LLM criteria
  Update CODEBOOK.md with clarified definitions and new examples.
  Copy llm-criteria/criteria_vN.yml → criteria_v(N+1).yml with field-level
  improvements. Log the change in llm-criteria/CHANGELOG.md.
  ↓
Step 6 — Check stopping rules
  All closed fields locked AND emergent fields saturated → run full corpus.
  Otherwise → LLM re-codes same batch (FT-R1b) to verify improvement, then
  proceed to next round (FT-R2a) with updated criteria.
```

---


## Folder structure

```
documentation/coding/systematic-map/
├── CODEBOOK.md                 ← this document
├── llm-criteria/
│   ├── criteria_sysmap_v1.yml         ← machine-readable LLM prompt (loaded by step15)
│   ├── criteria_sysmap_v1.md          ← human-readable companion
│   └── CHANGELOG.md
└── rounds/
    ├── FT-R1a/
    │   ├── FT-R1a pdfs/              ← full-text PDFs/HTMLs (on Google Drive, not in git)
    │   ├── coding_ft_r1a_XX.csv      ← blank template (coders save-as coding_ft_r1a_CS.csv etc.)
    │   ├── reconciled_ft_r1a.csv     ← gold standard after reconciliation
    │   └── irr_ft_r1a.json           ← kappa, sensitivity, field-level stats (generated)
    │   [on Drive only: coding_ft_r1a_CS.csv, coding_ft_r1a_JD.csv, coding_ft_r1a_LLM.csv]
    ├── FT-R1b/                       ← LLM re-code only (same papers as R1a)
    ├── FT-R2a/                 ← next batch
    └── ...

documentation/coding/abstract-screening/
└── criteria/                   ← eligibility criteria used in abstract screening (locked)
```

---

## Round log

| Round | Papers | Date | κ (subset) | Status |
|-------|--------|------|------------|--------|
| FT-R1a | 5 | — | — | ⏳ In progress |
| FT-R2a | 5 | — | — | — Not started |

*Update after each round's IRR run.*

---

## Quick reference — valid values

| Field | Values |
|-------|--------|
| `publication_type` | journal_article, report, working_paper, thesis, other |
| `geographic_scale` | local, sub-national, national, multi-country, regional |
| `producer_type` | crop, livestock, fisheries_aquaculture, agroforestry, mixed, undefined |
| `marginalized_subpopulations` | women, youth, people_with_disabilities, landless, indigenous_peoples, ethnic_minorities, migrant_seasonal_workers, other, none_reported |
| `adaptation_focus` | **free text** |
| `process_outcome_domains` | knowledge_awareness_learning, decision_making_planning, uptake_adoption, behavioral_change, participation_coproduction, institutional_governance, access_information_services, yields_productivity, income_assets, livelihoods, wellbeing, risk_reduction, resilience_adaptive_capacity |
| `indicators_measured` | **free text** |
| `methodological_approach` | qualitative, quantitative, participatory, modeling_with_empirical_validation, experimental |
| `purpose_of_assessment` | project_learning, program_evaluation, donor_reporting, national_reporting, research |
| `data_sources` | surveys, administrative_data, remote_sensing, participatory_methods, secondary_data, other |
| `temporal_coverage` | cross_sectional, seasonal, longitudinal, repeated_cross_sectional |
| `cost_data_reported` | yes, no |
| `strengths_and_limitations` | **free text** (Strength: ... / Limitation: ...) |
| `lessons_learned` | **free text** |

---

## Note for lead researcher — LLM parallel workflow

In addition to the human coding rounds described above, a large-language model (LLM) independently codes the full corpus using the same schema (`llm-criteria/criteria_sysmap_v1.yml`). This runs in the background and is invisible to human coders.

The LLM output serves as a secondary quality signal: after each human round, agreement rates between human codes and LLM codes are computed alongside inter-rater statistics. This does **not** replace human reconciliation — it supplements it. Human agreement on the random subset remains the primary stopping criterion.

LLM criteria are versioned (`criteria_sysmap_v1.yml`, `criteria_v2.yml`, …) and updated after each round's review. Changes are logged in `llm-criteria/CHANGELOG.md`.
