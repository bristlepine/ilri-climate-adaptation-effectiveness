# Systematic Map Codebook — D5.6
*ILRI / Bristlepine Resilience Consultants — 2025–2026*

> **Scope:** Systematic map data extraction only (D4/D5). Not the systematic review (D8/D9).

---

## Quick start for coders

Everything is on Google Drive — no GitHub or local setup needed.

**[→ Open the FT-R1a folder on Google Drive](https://drive.google.com/drive/folders/1DLifotAW4YcYZZ-5iYd0T0X0Q-N3MOmQ?usp=sharing)**

**Three steps:**

1. **Get your papers** — open the `FT-R1a pdfs` subfolder in the Drive link above
2. **Open your coding sheet** — `coding_H1.csv` (Coder 1) or `coding_H2.csv` (Coder 2)
3. **Code each paper** using the schema below — one row per paper, all 19 fields — save back to Drive when done

**Rules while coding:**
- Fill in `coder_id` with your initials (e.g. `CS`, `JD`) — same initials every row
- Code independently — no discussion with the other coder until reconciliation
- If a field is genuinely not reported: leave blank or enter `not_reported`
- If unsure: make your best judgement and add a note in the `notes` column
- Do not look at the other coder's sheet

**Your sheet columns:**
```
doi, filename,
publication_year, publication_type, country_region, geographic_scale,
producer_type, marginalized_subpopulations, adaptation_focus, domain_type,
process_outcome_domains, indicators_measured, methodological_approach,
purpose_of_assessment, data_sources, temporal_coverage, cost_data_reported,
equity_inclusion, strengths_and_limitations, lessons_learned, validity_notes,
coder_id, notes
```
The first two columns (`doi`, `filename`) are pre-filled. Fill the rest.

---

## The 19 coding fields

Use this as your reference while coding. Valid values must be used exactly as written (except free-text fields, marked ⚠).

---

### 1. `publication_year`
Year the study was published. Use the most recent version date if both preprint and journal exist.
**Valid values:** integer, e.g. `2021`

---

### 2. `publication_type`
**Valid values:** `journal_article` | `report` | `working_paper` | `thesis` | `other`
- `journal_article` — peer-reviewed journal (including open-access)
- `report` — project/programme report, evaluation report, policy brief
- `working_paper` — pre-publication working paper or discussion paper
- `thesis` — PhD or master's dissertation
- `other` — book chapter, conference paper, dataset paper

---

### 3. `country_region`
Country or countries where the study was conducted. Use region label only if no countries are named.
**Valid values:** free text, semicolons between multiple entries

**Examples:** `Tanzania` | `Ghana; Mali; Burkina Faso` | `East Africa` | `Global`

---

### 4. `geographic_scale`
**Valid values:** `local` | `sub-national` | `national` | `multi-country` | `regional`
- `local` — village, community, single site
- `sub-national` — district, province, region within a country
- `national` — one country
- `multi-country` — two or more specific named countries
- `regional` — broad region without country-level focus

---

### 5. `producer_type`
Select all that apply, semicolons between multiple values.
**Valid values:** `crop` | `livestock` | `fisheries_aquaculture` | `agroforestry` | `mixed`
- `mixed` — explicitly mixed crop-livestock, or generic "smallholder farmers" with no system specified

---

### 6. `marginalized_subpopulations`
Code only what is explicitly stated — do not infer. Select all that apply.
**Valid values:** `women` | `youth` | `landless` | `indigenous_peoples` | `ethnic_minorities` | `migrant_seasonal_workers` | `none_reported`

---

### 7. `adaptation_focus` ⚠ *free text — emergent*
The specific climate adaptation action, intervention, or practice the study tracks. 5–15 words.

**Examples:** `drought-tolerant maize varieties` | `seasonal climate forecasts for farmer decision-making` | `index-based livestock insurance` | `farmer field schools for climate-resilient agriculture`

---

### 8. `domain_type`
**Valid values:** `adaptation_process` | `adaptation_outcome` | `both`
- **Process** — measures *how* farmers adopt, learn, participate (steps toward adaptation)
- **Outcome** — measures *what happened* (yields, income, wellbeing, resilience)
- **Both** — tracks both a process and its outcome in the same study
- Decision rule: adoption rate only → `adaptation_process`; yield impact only → `adaptation_outcome`

---

### 9. `process_outcome_domains`
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

### 10. `indicators_measured` ⚠ *free text — emergent*
The specific indicators or metrics used. Extract actual indicators from methods/results. Include units.

**Examples:** `yield (t/ha), income (USD/season), food security score (HFIAS)` | `adoption rate (%), area under improved variety (ha)`

---

### 11. `methodological_approach`
**Valid values:** `qualitative` | `quantitative` | `mixed_methods` | `participatory` | `modeling_with_empirical_validation`
- `participatory` — only when participation IS the primary design (PRA, photovoice), not when farmers are merely surveyed
- `modeling_with_empirical_validation` — crop/climate/agent-based models validated with field data

---

### 12. `purpose_of_assessment`
**Valid values:** `project_learning` | `program_evaluation` | `donor_reporting` | `national_reporting` | `research`
- `research` — academic study with no programme affiliation
- `project_learning` — M&E embedded in an ongoing project
- `national_reporting` — contributing to NAPs, NDCs, national monitoring frameworks

---

### 13. `data_sources`
Select all that apply.
**Valid values:** `surveys` | `administrative_data` | `remote_sensing` | `participatory_methods` | `secondary_data`

---

### 14. `temporal_coverage`
**Valid values:** `cross_sectional` | `seasonal` | `longitudinal`
- `cross_sectional` — single point in time or single season
- `seasonal` — intra-annual, across one or a few seasons
- `longitudinal` — multi-year, panel data, repeated measures

---

### 15. `cost_data_reported`
**Valid values:** `yes` | `no`
Code `yes` only if the paper explicitly reports cost figures or cost-effectiveness ratios. Qualitative mention of "limited resources" = `no`.

---

### 16. `equity_inclusion`
Select all that apply. Use `none_reported` if equity is not mentioned.
**Valid values:** `gender` | `youth` | `land_tenure` | `disability` | `other` | `none_reported`

---

### 17. `strengths_and_limitations` ⚠ *free text*
Author-reported strengths and limitations combined (protocol Table 3). Label each component.

**Format:** `Strength: <text>. Limitation: <text>.`
Look in Discussion, Conclusion, and Limitations sections. Extract verbatim where possible.

---

### 18. `lessons_learned` ⚠ *free text*
Key lessons or recommendations the authors report. 1–3 sentences. Focus on what they say should be done differently — not findings. Enter `not_reported` if absent.

---

### 19. `validity_notes` ⚠ *free text*
Notes on methodological robustness that affect how much weight this study carries. Flag: small sample (n < 30), single site, self-reported outcomes, short follow-up, no control group, attrition not reported. Enter `none_flagged` if no concerns.

---

## Reconciliation (Step 2)

After both coders finish independently:

1. Open both sheets side by side
2. For each paper and each field:
   - Agree → enter that value in `reconciled.csv`
   - Disagree → discuss until you reach agreement, then enter it
3. Record the reason for any substantive disagreement in `reconciliation_notes`
4. Cannot agree → escalate to lead researcher

Do not average numeric fields — always discuss to a single agreed value.

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

### Emergent fields (`adaptation_focus`, `indicators_measured`)
Tracked by category saturation, not kappa. Saturated when no new codes appear across two consecutive batches.

---

## Stopping rules

- **Closed fields** — locked once κ ≥ 0.60 AND sensitivity ≥ 0.80 in the same batch. Locked fields are not retested.
- **Emergent fields** — saturated when no new codes appear across two consecutive batches.
- **Full corpus run** — proceed to step15 once all closed fields are locked AND both emergent fields are saturated. Expected: 3–4 rounds (300–400 papers).

---

## How rounds work

### Round naming

| Label | Who codes | Papers |
|-------|-----------|--------|
| `FT-R1a` | H1 + H2 + LLM independently | First 100 papers |
| `FT-R1b` | **LLM only** (re-codes same R1a papers after criteria update) | Same 100 papers as R1a |
| `FT-R2a` | H1 + H2 + LLM independently | Next 100 papers (101–200) |
| `FT-R2b` | **LLM only** (re-codes R2a papers after criteria update) | Same 100 as R2a |

**`a` rounds** = full human + LLM pass on new papers. **`b` rounds** = LLM verification only, same papers, after criteria update. Human coders only ever code in `a` rounds.

### Each round — six steps

```
Step 1 — Code independently
  H1, H2, and LLM all code the same 100 papers. No discussion.
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

## LLM criteria tracking

After each round's reconciliation, the LLM extraction prompt is updated and versioned — analogous to how the abstract screening criteria evolved across calibration rounds.

```
documentation/coding/systematic-map/llm-criteria/
├── criteria_v1.yml     ← loaded by step15; used in FT-R1a
├── criteria_v1.md      ← human-readable companion
├── criteria_v2.yml     ← updated after FT-R1a reconciliation; used in FT-R1b and FT-R2a
└── CHANGELOG.md        ← one line per version: round, date, what changed and why
```

**To update after a round:**
1. Review fields where LLM underperformed (κ < 0.60 or sensitivity < 0.80)
2. Copy `criteria_vN.yml` → `criteria_v(N+1).yml`
3. Add `r1_further_guidance:` (or `r2_`, etc.) to any field that needed clarification
4. Add one line to `CHANGELOG.md`
5. Update `step15_criteria_yml` in `config.py` to point to the new file

---

## Folder structure

```
documentation/coding/systematic-map/
├── CODEBOOK.md                 ← this document
├── llm-criteria/
│   ├── criteria_v1.yml         ← machine-readable LLM prompt (loaded by step15)
│   ├── criteria_v1.md          ← human-readable companion
│   └── CHANGELOG.md
└── rounds/
    ├── FT-R1a/
    │   ├── FT-R1a pdfs/               ← the 100 full-text PDFs/HTMLs for this round (not in git)
    │   ├── coding_H1.csv       ← Coder 1's codes
    │   ├── coding_H2.csv       ← Coder 2's codes
    │   ├── coding_LLM.csv      ← LLM codes (generated by step15)
    │   ├── reconciled.csv      ← gold standard after H1+H2 reconciliation
    │   └── irr_R1a.json        ← kappa, sensitivity, field-level stats
    ├── FT-R1b/                 ← LLM re-code only (same papers as R1a)
    ├── FT-R2a/                 ← next 100 papers
    └── ...

documentation/coding/abstract-screening/
└── criteria/                   ← eligibility criteria used in abstract screening (locked)
```

---

## Round log

| Round | Papers | Date | H-H κ | LLM sensitivity | Status |
|-------|--------|------|--------|-----------------|--------|
| FT-R1a | 100 | — | — | — | ⏳ In progress |
| FT-R1b | same 100 | — | — | — | — Not started |
| FT-R2a | 100 | — | — | — | — Not started |

*Update after each round's IRR run.*

---

## Quick reference — valid values

| Field | Values |
|-------|--------|
| `publication_type` | journal_article, report, working_paper, thesis, other |
| `geographic_scale` | local, sub-national, national, multi-country, regional |
| `producer_type` | crop, livestock, fisheries_aquaculture, agroforestry, mixed |
| `marginalized_subpopulations` | women, youth, landless, indigenous_peoples, ethnic_minorities, migrant_seasonal_workers, none_reported |
| `adaptation_focus` | **free text** |
| `domain_type` | adaptation_process, adaptation_outcome, both |
| `process_outcome_domains` | knowledge_awareness_learning, decision_making_planning, uptake_adoption, behavioral_change, participation_coproduction, institutional_governance, access_information_services, yields_productivity, income_assets, livelihoods, wellbeing, risk_reduction, resilience_adaptive_capacity |
| `indicators_measured` | **free text** |
| `methodological_approach` | qualitative, quantitative, mixed_methods, participatory, modeling_with_empirical_validation |
| `purpose_of_assessment` | project_learning, program_evaluation, donor_reporting, national_reporting, research |
| `data_sources` | surveys, administrative_data, remote_sensing, participatory_methods, secondary_data |
| `temporal_coverage` | cross_sectional, seasonal, longitudinal |
| `cost_data_reported` | yes, no |
| `equity_inclusion` | gender, youth, land_tenure, disability, other, none_reported |
| `strengths_and_limitations` | **free text** (Strength: ... / Limitation: ...) |
| `lessons_learned` | **free text** |
| `validity_notes` | **free text** |
