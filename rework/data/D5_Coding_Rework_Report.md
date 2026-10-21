# D5 Systematic Map — Expert Re-coding of the Full-Text Corpus and Comparison with the Prior Coding

*Climate adaptation effectiveness for smallholder producers in LMICs (Bristlepine Resilience Consultants for ILRI). Prepared 16 October 2026 for co-authors. Re-coding by an AI reviewer team: Claude Opus project lead with Claude Sonnet Ph.D.-level reviewer agents.*

---

## Executive summary

**What was done.** Every full text in the D5 corpus was re-screened and re-coded against the Systematic Map Protocol (D3) and `CODEBOOK_FT`. The corpus has 152 records, of which 151 have a full text (FT-005 has none; FT-021 is a 20-file edited book). For each record, a reviewer agent read the full paper and blind-coded eligibility plus all 16 codebook fields, with a rationale, verbatim evidence (with page) and a confidence rating. Only after saving the blind coding did the reviewer open the prior (non-expert) coding and adjudicate every field. Any record the first reviewer did not clearly include went to an independent, blinded second reviewer, and the project lead made the final call. Three quality checks were run on the rework itself: an automated check of codebook validity, a check that each evidence quote appears in the source text, and a blinded A/B audit of a random sample of disagreements.

**Headline findings**

1. **About a quarter of the database does not meet the protocol's eligibility criteria.** The prior database treated all 152 records as included. After dual review and lead adjudication, **36 of 151 coded records (24%) should be excluded**: 23 clear exclusions and 13 borderline. **115 are confirmed as included**, 14 of them borderline. The most common reason is that climate is only background, not the driver of the research question (16 records). The others are no adaptation process or outcome measured (9), on-station agronomic trials with no farmer or adoption analysis (5), non-empirical reviews or workshop papers (3), and non-smallholder populations (3).
2. **Field-level coding disagrees substantively on about half the fields.** Across 2,416 field comparisons, 38% agreed exactly and 16% differed only in formatting. 30% partly overlapped and 17% disagreed outright. The average record has **7.5 of 16 fields** with a substantive difference (median 8). Only 3 of 151 prior records were rated "good"; 108 were "acceptable" and 40 "poor".
3. **Much of the prior coding does not use the codebook's controlled vocabulary.** For example, only 19% of prior `process_outcome_domains` cells and about half of `methodological_approach` and `data_sources` cells use valid codes. Free text, commas, misspellings and invented codes are common. The rework has **0 invalid codes and 0 format violations**.
4. **The errors are systematic, not random, and they would bias the evidence map:**
   - **`youth` is over-coded** (51 → 8 records) and **`none_reported` under-used** (57 → 90). Groups were coded when they appeared in sample descriptions, not because they were targeted or disaggregated.
   - **Scale** was often `local` when studies span several villages or districts (`local` 40 → 15; `sub-national` about 90 → 118). `regional` was used for multi-country studies (4 → 0; `multi-country` 6 → 12).
   - **`mixed` producer type is over-used** (51 → 20) where crop and livestock systems are studied separately.
   - **Study design:** `participatory` is over-coded (38 → 13), since farmer surveys are not participatory designs. `experimental` is under-coded (11 → 24), with quasi-experimental and matching designs missed.
   - **Outcome domains** were frequently inferred, not measured. `decision_making_planning` was over-coded (75 → 33) and `risk_reduction` likewise (32 → 13). `access_information_services` was under-coded (36 → 75).
   - **`cost_data_reported` = yes** was over-coded (32 → 23).
   - **Free-text fields:** 35% of `strengths_and_limitations` and 36% of `lessons_learned` entries disagree outright. They often hold findings instead of recommendations, or coder opinion instead of the authors' own appraisal.
5. **There are bibliographic errors.** 13 records have DOI, title or year problems. Notable ones:
   - FT-092: the prior title and year belong to a different (cited) paper.
   - FT-137: the database DOI belongs to a cited reference.
   - FT-052: the curator's "Correct DOI" annotation is itself wrong. The original DOI resolves to the article; the annotated one comes from an erroneous "Cite as" line in the PDF.
   - At least 7 prior codings were evidently made from the abstract only or describe a different paper.
6. **The rework holds up under independent checks.** Of 3,398 evidence quotes, 97.8% were found verbatim in the source text; the rest are mostly PDF-extraction artefacts. In a **blinded A/B audit** of 86 randomly sampled disagreements (15 records), independent auditors preferred the rework coding in **89.5%** of cases. They preferred the prior coding in 3.5% and found both acceptable in 7%.

**Decisions needed from co-authors** (detail in §5):
- Confirm the 36 recommended exclusions, especially the 13 flagged as borderline.
- Confirm the 14 borderline inclusions.
- Decide on 4 included records published in 2026, outside the protocol's 2005–2025 window.
- Decide whether to split the edited book FT-021 into chapter-level records.
- Adopt the corrected DOIs and titles.
- Decide what to do with FT-005, which has no full text.

---

## 1. Purpose

The D5 database (`EGM_database_v02.xlsx`, sheet "Systematic map database") was pre-filled by non-expert coders in nine rounds (FT-R2a to FT-R7). This exercise re-applied the protocol and codebook to the whole corpus, with explicit rationale and citations for each decision. It aimed to:
- produce a corrected database;
- identify misclassifications and disagreements with the prior coding;
- characterise their nature and consequences for the systematic map.

## 2. Methods

### 2.1 Team design

| Role | Agent | Responsibility |
|---|---|---|
| Project lead | Claude Opus 5.5 | Read the protocol (D3) and codebook. Designed the coding framework, reviewer brief and output schema. Piloted and refined the brief. Assigned batches. Adjudicated contested eligibility. Ran the QA and statistics, and wrote this report. |
| Reviewer agents (51 batches, about 3 records each) | Claude Sonnet, instructed to act as Ph.D.-level researchers | Full-text reading, blind coding of eligibility and 16 fields with rationale, evidence and confidence, then field-by-field adjudication against the prior coding. |
| Second eligibility reviewers (11 batches) | Claude Sonnet | Independent, blinded re-screening of all contested records. |
| Auditors (3 batches) | Claude Sonnet | Blinded A/B adjudication of a random sample of disagreements. |

### 2.2 Materials and rules

- The binding rules were `CODEBOOK_FT` (pinned GitHub version 22924ef) and the D3 eligibility table (PCCM plus period, language, publication type and geography).
- The reviewer brief (`rework_workspace/CODER_BRIEF.md`) restates these rules. It adds clarifications, refined after a two-record pilot, for recurrent ambiguities:
  - tie-breaks between `seasonal` and `longitudinal`;
  - FGDs used only for questionnaire design;
  - what counts as an author-stated strength;
  - treatment of excluded records.
- Full texts were extracted to page-marked text. Two scanned PDFs (FT-090, FT-130) were read visually from rendered page images. The edited book FT-021 was read chapter by chapter.

### 2.3 Procedure

1. **Blind coding.** The reviewer read the full text and coded six eligibility criteria and the 16 fields. Each field got a value, a 1–3 sentence rationale citing the decisive codebook rule, 1–3 verbatim quotes with page, and a confidence rating. This was saved to disk **before** the prior coding was opened.
2. **Comparison and adjudication.** The reviewer opened the prior row and classified each field:
   - Agreement: `exact`, `equivalent_format_only`, `partial`, `disagree` or `prior_missing`.
   - Verdict: rework correct, prior correct, both acceptable, both partly correct, or ambiguous.
   - Error types: one or more of nine categories.
   
   Reviewers could revise their own coding if the prior caught something; this happened in 15 fields.
3. **Dual eligibility review.** All 53 records the first reviewer did not clearly include (24 `no`, 29 `uncertain`) went to a second reviewer. The second reviewer was blinded to both the first reviewer and the prior coding, and was told to stay inclusive at the margin, as systematic-map practice requires. The lead adjudicated each record and set a `borderline_flag` where judgement was finely balanced.
4. **QA of the rework.** An automated validation checked codes, single- and multi-select rules, year format, the 5–15 word `adaptation_focus` limit and the strengths/limitations prefix format. A separate script checked every evidence quote against the source text.
5. **Blinded audit.** For 15 randomly sampled included records, every substantive disagreement on 11 key fields (86 items) was shown to an independent auditor as "Coding A" and "Coding B". Formatting was normalised and the order randomised. The auditor judged against the paper and codebook alone.

### 2.4 Deviations and process notes

- **Interrupted batches.** Usage limits interrupted 21 batches. They were resumed under a written resume rule: completed records were skipped, and blind-only records had their adjudication completed after a fresh re-read.
- **Blinding not strictly met.** In 5 records the reviewer saw the prior coding before saving the blind file: FT-031, FT-058, FT-125, FT-134 and FT-137. Each reviewer reports forming their judgement from the full text first. These records are flagged in the database (`blinding_note`).
- **Verdict labels harmonised.** Some reviewers used the "prior correct" label for fields that simply agreed. These 97 labels were recoded to "both acceptable"; no verdict on a substantive disagreement was changed.

## 3. Results

### 3.1 Eligibility

| Stage | Include | Uncertain | Exclude |
|---|---|---|---|
| Prior database | 152 (all) | – | – |
| First expert reviewer (151 coded) | 98 | 29 | 24 |
| Second reviewer, on the 53 contested | 13 | 10 | 30 |
| **Final, after lead adjudication** | **115** (14 borderline) | – | **36** (13 borderline) |

The first and second reviewers agreed closely on clear exclusions: 23 of the 24 first-reviewer `no` decisions were confirmed. They agreed less on `uncertain` records. Of these 29, the second reviewer included 12, excluded 7 and stayed uncertain on 10.

**Reasons for the 36 exclusions**

| Reason | n | Records |
|---|---|---|
| Climate only background, or no climate-hazard link (context) | 16 | FT-039, 043, 057, 067, 069, 076, 079, 085, 094, 122, 126, 129, 139, 141, 147, 148 |
| No adaptation process or outcome measured (vulnerability index, perception, information or engineering only) | 9 | FT-011, 022, 058, 072, 078, 088, 130, 132, 140 |
| Researcher-managed agronomic trial with no farmer or adoption analysis | 5 | FT-027, 040, 044, 090, 100 |
| Not empirical (literature review, opinion or workshop paper, narrative chapter) | 3 | FT-016, 041, 115 |
| Not a smallholder population (commercial or national scale) | 3 | FT-018, 123, 127 |

Exclusions are spread across all rounds, from 0 in FT-R6 to 6 in FT-R2d and FT-R7. They are not confined to one coder. The lead's rationale for each record is in Appendix A and in the `lead_rationale` column of the database.

### 3.2 Field-level agreement

| Field | Agree (exact or format-only) | of which format-only | Partial overlap | Disagree | Reviewer verdict: rework correct |
|---|---|---|---|---|---|
| publication_year | 91% | 0% | 1% | 8% | 6% |
| publication_type | 96% | 40% | 0% | 4% | 2% |
| country_region | 56% | 30% | 41% | 2% | 44% |
| geographic_scale | 70% | 10% | 3% | 26% | 30% |
| producer_type | 56% | 4% | 15% | 30% | 32% |
| marginalized_subpopulations | 42% | 5% | 21% | 37% | 52% |
| adaptation_focus | 42% | 42% | 52% | 5% | 50% |
| process_outcome_domains | 3% | 2% | 82% | 15% | 83% |
| indicators_measured | 49% | 49% | 46% | 4% | 38% |
| methodological_approach | 39% | 13% | 54% | 6% | 44% |
| purpose_of_assessment | 93% | 0% | 1% | 5% | 2% |
| data_sources | 34% | 11% | 58% | 9% | 54% |
| temporal_coverage | 75% | 1% | 0% | 24% | 21% |
| cost_data_reported | 83% | 19% | 0% | 17% | 17% |
| strengths_and_limitations | 12% | 12% | 50% | 35% | 72% |
| lessons_learned | 12% | 11% | 50% | 36% | 68% |

*Rows may not sum to 100% because of rounding and a small share (≤3%) of blank prior cells. The remaining verdicts are "both acceptable", "both partly correct" or "ambiguous". Ambiguity was highest for `lessons_learned` (16%), `process_outcome_domains` (13%), `strengths_and_limitations` (13%) and `methodological_approach` (12%).*

**Machine-computed check (independent of the reviewers' judgements).** Prior cells were first normalised for formatting only: case, separators and known misspellings. Agreement with the rework was then:

| Field | Prior cells using valid codebook format | Match after normalisation | Mean set overlap (Jaccard) |
|---|---|---|---|
| process_outcome_domains | 19% | 3% | 0.31 |
| data_sources | 51% | 32% | 0.59 |
| methodological_approach | 49% | 36% | 0.63 |
| marginalized_subpopulations | 59% | 42% | 0.51 |
| producer_type | 97% | 54% | 0.59 |
| publication_type | 59% | 95% | 0.95 |
| geographic_scale | 83% | 70% | 0.71 |
| cost_data_reported | 79% | 74% | 0.74 |
| temporal_coverage | 91% | 75% | 0.75 |
| purpose_of_assessment | 98% | 93% | 0.94 |
| publication_year | 99% | 91% | 0.91 |

### 3.3 Nature of the prior-coding errors

Field-level error tags assigned during adjudication (a field may carry several):

| Error type | Occurrences | Typical example |
|---|---|---|
| Invalid code or format in a categorical field | 547 | "qualitative, quantitative, participatory"; "sub_national \\ local"; "un_clear"; "none_reported" used for cost |
| Free-text format non-compliance | 330 | Strengths and limitations not in the required format; `adaptation_focus` over 15 words or a list |
| Omission (applicable code missed) | 271 | `secondary_data` missed for climate records; `access_information_services` missed when extension or credit is analysed |
| Over-inclusion (code not supported) | 269 | Domains "could be inferred" but not measured; `youth` from age statistics |
| Misclassification | 200 | `local` for multi-village studies; `mixed` for separate crop and livestock systems |
| Inferred rather than explicit | 139 | Marginalised groups coded without targeting or disaggregation |
| Findings recorded as lessons | 121 | Results restated in `lessons_learned` instead of the authors' recommendations |
| Incomplete or truncated | 100 | Strengths and limitations cut off; indicators missing |
| Factual error | 36 | Wrong year, country, DOI or title; text from another paper |

**Prior coding apparently based on the abstract, or on a different document:**
- FT-011, FT-087, FT-131 and FT-151 were coded from the abstract only. In FT-151 the coder's own note says so.
- FT-037's `adaptation_focus` and other fields describe content not in the chapter.
- FT-099's and FT-144's `strengths_and_limitations` describe a different study.
- FT-090 appears to have been coded without access to the PDF.

### 3.4 Consequences for the evidence map

Code frequencies across the 151 coded records. Prior values are lower-cased raw codes with obvious variants merged; invalid codes are not counted.

| Field | Code | Prior | Rework | Direction |
|---|---|---|---|---|
| geographic_scale | local | 40 | 15 | Prior over-used `local` for multi-village or multi-district studies |
| | sub-national | ~90 | 118 | |
| | multi-country / regional | 6 / 4 | 12 / 0 | `regional` used where countries are named |
| producer_type | mixed | 51 | 20 | `mixed` over-used |
| | crop / livestock | 72 / 28 | 100 / 40 | |
| | fisheries_aquaculture / agroforestry | 10 / 6 | 17 / 11 | Under-recognised systems |
| marginalized_subpopulations | youth | 51 | 8 | Strong over-coding |
| | women | 61 | 47 | |
| | none_reported | 57 | 90 | |
| methodological_approach | participatory | 38 | 13 | Surveys mistaken for participatory design |
| | experimental | 11 | 24 | Quasi-experimental designs (PSM, IV, DiD, field experiments) missed |
| | modeling_with_empirical_validation | 12 | 4 | Regression mislabelled as modelling |
| process_outcome_domains | decision_making_planning | 75 | 33 | Inferred, not measured |
| | access_information_services | 36 | 75 | Under-coded |
| | risk_reduction / behavioral_change | 32 / 44 | 13 / 24 | Over-coded |
| | uptake_adoption / income_assets | 71 / 35 | 91 / 49 | Under-coded |
| data_sources | secondary_data / other | 52 / 20 | 77 / 35 | Climate records and field measurements missed |
| temporal_coverage | longitudinal | 40 | 24 | Recall questions treated as longitudinal |
| cost_data_reported | yes | 32 | 23 | Costs as covariates or barriers treated as cost data |

These shifts change the substantive picture the map would present, most visibly for equity coverage (youth, women), study design and the balance between process and outcome domains. If the prior coding were used as is, the map would overstate:
- attention to youth;
- participatory and modelling designs;
- decision-making and risk-reduction evidence.

It would understate:
- quasi-experimental designs;
- access-to-services outcomes;
- adoption and income outcomes.

### 3.5 Bibliographic metadata

Reviewers flagged DOI, title or year problems in 13 records. Two more records (FT-120, FT-121) have a blank database DOI but a correct curator annotation. All are listed in the `Record_summary` and `Rework_database` sheets.

| Record | Issue | Recommended action |
|---|---|---|
| FT-052 | The database DOI (10.37425/eajsti.v4i2.525) is **correct** and resolves to the EAJSTI article. The curator's "Correct DOI" (10.3329/bjar.v42i3.34505) copies an erroneous "Cite as" line printed in the PDF; it resolves to a Bangladesh Journal of Agricultural Research article. | Keep the database DOI and discard the annotation |
| FT-137 | The database DOI and year belong to a cited reference (Yanda et al. 2019). | Use 10.58721/eajhss.v5i1.1530, year 2026 |
| FT-092 | The prior title and year (2010) belong to a different, cited paper. | Use Mohan et al. (2021), "Tank Rehabilitation Strategies: Prioritizing Investment Options" |
| FT-027 | The prior title belongs to a different paper; `publication_year` 2025 is the HAL deposit date. | Use the article title; year 2022 |
| FT-016 | The title field holds an institutional header. | Use "Producción Agrícola Sostenible y Seguridad Alimentaria en Nicaragua" |
| FT-004, 022, 069, 084, 101, 110, 143, 150 | Publication year wrong, usually online date versus issue year. | Use the issue year per the codebook |
| FT-120, FT-121 | DOI blank in the database; the curator's "Correct DOI" matches the document. | Adopt the annotation |

### 3.6 Patterns by prior coder and round

| Prior coder | Records | Mean fields with substantive disagreement | Rated poor | Finally excluded |
|---|---|---|---|---|
| (blank coder_id) | 52 | 7.5 | 17% | 13 |
| AZ | 33 | 7.1 | 18% | 11 |
| SZC | 37 | 8.3 | **65%** | 8 |
| LJ | 14 | 6.6 | 0% | 2 |
| SV | 9 | 6.6 | 11% | 2 |
| JK | 6 | 7.3 | 0% | 0 |

Round FT-R2b had 90% of its records rated poor. FT-R6 had none rated poor and no exclusions. Disagreement levels are broadly similar across coders (6.6–8.3 fields per record), so the problems reflect the codebook being hard to apply without training more than any single coder. Coder SZC and round FT-R2b stand out on overall quality, mainly because of invalid codes and free-text fields.

### 3.7 Validity of the rework

| Check | Result |
|---|---|
| Codebook validity of rework categorical fields (valid codes, single-select rules, `none_reported` and `undefined` used alone) | 0 violations in 151 records |
| Free-text rules (`adaptation_focus` 5–15 words; "Strength: … Limitation: …" format) | 0 violations |
| Evidence quotes found verbatim in the source text (after normalising ligatures, hyphenation and whitespace) | 3,324 of 3,398 (97.8%). Spot checks of the 74 not found showed line or column breaks, table interleaving and footnote markers; 11 are a known checker failure on one file's character encoding (FT-027); a few are light paraphrases marked as verbatim. No fabricated content was found. |
| Reviewer confidence (2,416 field codings) | 65% high, 33% medium, 2% low |
| **Blinded A/B audit** (86 disagreements, 15 random included records) | **Rework preferred 77 (89.5%)**; prior preferred 3 (3.5%); both acceptable 6 (7.0%); neither 0 |

The three items where the auditor preferred the prior coding (`data_sources` in FT-006, `producer_type` in FT-047, `strengths_and_limitations` in FT-049) are minor judgement calls. They show the rework is not infallible, and co-authors should treat medium- and low-confidence codings accordingly.

## 4. Records for specific attention

- **FT-021 (edited book, *Beyond Agricultural Impacts*).**
  - Coded as one included record at book level (multi-country, union of chapter codes).
  - The reviewer recommends **splitting it into chapter-level records**. Ch. 3–5 and 7–10 qualify on their own. Ch. 6 and 12 are borderline. Ch. 1, 2, 11 and 13 are not empirical.
  - A chapter table is in the record's `reviewer_notes`.
- **FT-005.** There is no full text in the corpus (prior `Full text = No`). It is not re-coded and should be retrieved or removed.
- **Duplicate check.** The reviewer notes that FT-037 (book chapter) may duplicate a GeoJournal article by the same authors.
- **FT-107.** This is a Research Square preprint of a 2022 *Climatic Change* article. Following the "most recent version" rule it is coded as a 2022 journal article; consider replacing the file with the published version.

## 5. Decisions requested from co-authors

1. **Exclusions (36).** Confirm the 23 clear exclusions. Review the 13 borderline exclusions: FT-011, 018, 039, 043, 072, 078, 088, 100, 122, 126, 127, 132 and 140. FT-127 needs confirmation of farm scale.
2. **Borderline inclusions (14).** FT-014, 050, 064, 066, 070, 087, 092, 097, 103, 110, 116, 137, 142 and 149 were included at the margin, consistent with inclusive mapping practice.
3. **Publication window.** The codebook says "2005 or later" but the protocol says 2005–2025. Four included records were published in 2026: FT-093, FT-096, FT-111 and FT-137. FT-086 is 2026 by issue year but was online in November 2025. Decide whether to retain them and record the decision as a protocol deviation.
4. **FT-021.** Decide whether to keep the book as one record or split it into chapters.
5. **Metadata.** Adopt the DOI, title and year corrections in §3.5.
6. **Codebook clarifications for future rounds.** Recommended additions:
   - a disaggregation rule for gender and age, where a regressor alone does not qualify;
   - a decision rule for "climate as one of several drivers";
   - treatment of researcher-managed trials;
   - whether quasi-experimental econometrics (PSM, IV, ESR) counts as `experimental`;
   - online versus issue year;
   - whether insurance premiums count as cost data.

## 6. Recommendations

- **Adopt the rework database** as the working D5 database, subject to the decisions in §5. Every value has a rationale and page-referenced evidence for audit.
- **Report the eligibility correction transparently** in the ROSES flow diagram: 36 additional full-text exclusions, with reasons.
- **Have a human expert spot-check the borderline set** (27 records) and a sample of the low- and medium-confidence field codings, especially `process_outcome_domains`, `methodological_approach` and the two free-text appraisal fields.
- **Before further rounds,** add validation to the coding instrument: drop-down controlled vocabularies, single-select enforcement and length checks. This would remove about 40% of the error tags seen here, which are format errors alone.

## 7. Limitations

- **AI reviewers.** The reviewers are AI agents, not human experts. Their judgements were structured, evidence-based and checked, but co-authors should review them, especially borderline eligibility calls.
- **Single-reviewer coding.** Records the first reviewer clearly included (98) and all field coding were done by one reviewer. Dual review was limited to contested eligibility. The blinded audit gives sample-based, not exhaustive, validation.
- **Self-adjudication.** The first reviewer adjudicated its own coding against the prior. After normalising labels, reviewers never sided with the prior on a remaining substantive disagreement, though they revised their own coding in 15 fields. This is why the blinded third-party audit is important: it found the prior preferable in 3.5% of sampled disagreements.
- **Text extraction.** Two scanned documents were read from images. Table-heavy PDFs may extract imperfectly; reviewers were instructed to check the PDF where needed.
- **Process deviations.** Five records did not strictly meet the blind-first procedure (§2.4).
- **Disagreement counts.** These include free-text fields, where reasonable coders vary in wording. The categorical-field statistics (§3.2, machine-computed check) are the more objective measure.

## 8. Deliverables

| File | Content |
|---|---|
| `D5_Rework_Outputs/D5_rework_coding.xlsx` | Workbook with sheets: README, Rework_database, Rationale_evidence, Eligibility, Comparison, Record_summary, Deterministic_checks, Blinded_audit |
| `D5_Rework_Outputs/D5_rework_database.csv` | One row per record (152): final eligibility, borderline flag, lead rationale, first- and second-reviewer decisions, all 16 fields, reviewer notes, metadata checks |
| `D5_Rework_Outputs/D5_rework_field_rationale_evidence.csv` | 2,416 rows: value, rationale, verbatim evidence with page, confidence |
| `D5_Rework_Outputs/D5_rework_eligibility.csv` | Per-criterion (PCCM + period/language) decisions and rationale, with second-reviewer results |
| `D5_Rework_Outputs/D5_rework_vs_prior_comparison.csv` | 2,416 field comparisons: prior vs rework value, agreement, verdict, error types, explanation |
| `D5_Rework_Outputs/D5_rework_record_summary.csv` | Per-record disagreement counts, prior-quality rating, summary |
| `D5_Rework_Outputs/D5_rework_deterministic_checks.csv` | Machine-computed format validity and set overlap |
| `D5_Rework_Outputs/D5_rework_blinded_audit.csv` | Unblinded audit verdicts |
| `rework_workspace/` | Audit trail: reviewer brief, codebook copy, extracted texts, per-record JSON coding, second-review JSON, lead decisions, scripts |

---

## Appendix A. Final exclusions with lead rationale

| Record | Round | Borderline | Rationale |
|---|---|---|---|
| FT-011 | R2b | yes | Fire-mitigation/conservation policy game; drought is a treatment, not the adaptation assessed |
| FT-016 | R2b | | Documentary literature review; no empirical data |
| FT-018 | R3 | yes | Intensive export-oriented aquaculture; no smallholder analysis; adaptation modelled |
| FT-022 | R2a | | Vulnerability/adaptive-capacity index; no adaptation action measured |
| FT-027 | R2a | | On-station agronomic trial; no farmer adoption analysis |
| FT-039 | R2d | yes | Stress is soil sodicity, not a climate hazard |
| FT-040 | R2b | | On-station crop trial; no adoption analysis |
| FT-041 | R2a | | Workshop/opinion paper; no measurement |
| FT-043 | R3 | yes | Climate one background stressor in a sustainability-transition game |
| FT-044 | R2b | | Soil moisture/temperature trial; no producers |
| FT-057 | R7 | | Fertilizer/yield response; no climate framing |
| FT-058 | R2d | | Regional delta risk mapping; no producers; no adaptation measured |
| FT-067 | R2c | | Regenerative farming framed around mitigation |
| FT-069 | R2c | | Perceptions of organic farming; climate only in abstract |
| FT-072 | R2d | yes | Baseline constraints/perception survey; no adaptation measured |
| FT-076 | R5 | | Fire-risk land-use economics; mitigation framing |
| FT-078 | R2c | yes | Fog-harvesting technical potential; no producers or outcomes |
| FT-079 | R7 | | Adaptation to market/policy change, not climate |
| FT-085 | R7 | | Compliance with traditional rangeland rules; climate background |
| FT-088 | R2a | yes | Information in groundwater governance; no adaptation measured |
| FT-090 | R4 | | Replicated agronomic trial; no farmer analysis |
| FT-094 | R4 | | Improved varieties/fertilizer adoption; climate only as motivation |
| FT-100 | R7 | yes | Rainout-shelter trial; no farmers or adoption analysis |
| FT-115 | R4 | | Narrative overview chapter; no own method |
| FT-122 | R2a | yes | Land-restoration practices not framed as adaptation; climate background |
| FT-123 | R5 | | National-scale seafood modelling; no small-scale producers |
| FT-126 | R4 | yes | Fisher socio-economic profile; climate one background risk |
| FT-127 | R7 | yes | Large mechanized farms (≥420 ha); confirm scale |
| FT-129 | R2d | | ICT/leadership survey; climate background |
| FT-130 | R7 | | Greywater treatment engineering; no adaptation measured |
| FT-132 | R3 | yes | Indigenous forecast skill; no adaptation process or outcome |
| FT-139 | R2c | | Soil-conservation adoption; no climate hazard |
| FT-140 | R2d | yes | Groundwater-saving intention; no climate-adaptation framing |
| FT-141 | R3 | | Peri-urban dairy technology adoption; climate background; 2026 |
| FT-147 | R2d | | Residue burning; mitigation/pollution framing |
| FT-148 | R3 | | Landscape-change mapping; adaptation incidental |

## Appendix B. Borderline inclusions

| Record | Round | Basis for inclusion at the margin |
|---|---|---|
| FT-014 | R2d | About 2 ha households despite "large-scale" wording; drought-driven drip-irrigation adoption |
| FT-050 | R2c | Designs and evaluates an index-insurance tool for pastoralists (tool-based methodology) |
| FT-064 | R2c | Oasis farmers' responses to arid-zone water scarcity; resilience assessment |
| FT-066 | R5 | Household survey including weather-information use and adaptive capacity |
| FT-070 | R2b | Participatory scenario planning for adaptive capacity of small-scale fishers |
| FT-087 | R2b | Agroforestry adoption explicitly framed as climate adaptation |
| FT-092 | R5 | Economic appraisal of tank rehabilitation as a water-scarcity response |
| FT-097 | R2b | Agroecological adoption decisions in semi-arid Tunisia |
| FT-103 | R7 | Livelihood responses to storms, floods and salinity within a human-security framing |
| FT-110 | R2a | Water-conservation behaviour in a drought-stressed basin |
| FT-116 | R3 | Quinoa adoption as a drought- and salinity-tolerant diversification option |
| FT-137 | R7 | Adaptation strategies and perceived effectiveness recorded; published 2026 |
| FT-142 | R3 | Measured coping responses to drought-linked food and water insecurity |
| FT-149 | R2a | Maladaptive outcomes of a GLOF adaptation project (protocol lists maladaptation as an outcome) |
