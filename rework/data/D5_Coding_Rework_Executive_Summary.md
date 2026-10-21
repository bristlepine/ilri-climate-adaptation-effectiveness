# D5 Coding Rework — Executive Summary

*Climate adaptation effectiveness for smallholder producers in LMICs (Bristlepine / ILRI). 16 October 2026. The full report, D5_Coding_Rework_Report (.md/.pdf), gives the methods, detailed results, record lists and limitations. The data are in D5_Rework_Outputs/.*


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

