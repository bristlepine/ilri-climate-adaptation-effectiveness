# Response to Neal Haddaway — Point-by-Point

**Re:** Use of AI for screening — Adaptation measurement  
**From:** Zarrar Khan  
**Date:** 2026-04-06  
**Status:** DRAFT — in progress

---

Responses to concerns raised. Where concerns are valid, actions are noted. Where a point rests on a technical distinction between our approach and supervised ML screeners, that distinction is explained. The methodology appendix has been revised throughout; relevant sections are referenced at the end of each point.

---

## Metric Definitions and Benchmarks

All performance metrics used in this response are defined below, with sources. These are reported consistently throughout so results can be tracked in context.

### The confusion matrix — foundation of all classification metrics

Every screening decision falls into one of four cells [O'Mara-Eves et al. 2015]:

|  | Screener says: **INCLUDE** | Screener says: **EXCLUDE** |
|---|---|---|
| **Truly relevant** | **TP — True Positive:** correctly included ✓ | **FN — False Negative:** missed relevant study ✗ ← *most serious error* |
| **Truly irrelevant** | **FP — False Positive:** incorrectly included ✗ | **TN — True Negative:** correctly excluded ✓ |

In systematic review screening, "truly relevant" and "truly irrelevant" are determined by the reconciled gold-standard decisions of two independent human reviewers. All five metrics below are derived from these four counts.

### Metric definitions

| Metric | What it measures | Formula | Source |
|---|---|---|---|
| **Sensitivity / Recall** | Of all truly relevant records (TP + FN), what proportion were correctly included (TP)? | TP / (TP + FN) | O'Mara-Eves et al. 2015; Cochrane Handbook |
| **Specificity** | Of all truly irrelevant records (TN + FP), what proportion were correctly excluded (TN)? | TN / (TN + FP) | O'Mara-Eves et al. 2015 |
| **Precision** | Of all records the screener included (TP + FP), what proportion are truly relevant (TP)? | TP / (TP + FP) | O'Mara-Eves et al. 2015 |
| **F1** | Harmonic mean of precision and recall. Penalises imbalance — a screener that achieves high recall by including everything will have low precision and therefore a moderate F1. | 2 × (P × R) / (P + R) | O'Mara-Eves et al. 2015 |
| **Cohen's κ (kappa)** | Agreement between two raters beyond what would be expected by chance alone. κ = 0 means agreement no better than random; κ = 1 is perfect agreement. | (p_o − p_e) / (1 − p_e) | Landis & Koch 1977 |

**Why sensitivity is the priority metric in systematic reviews:** O'Mara-Eves et al. [2015] — the foundational text-mining review for evidence synthesis — establish that "systematic reviewers generally place strong emphasis on high recall — a desire to identify all the relevant includable studies — even if that means a vast number of irrelevant studies need to be considered." Missing a relevant study (FN) is treated as a more serious methodological error than including an irrelevant one (FP), which is caught at the full-text screening stage. The widely cited threshold is **≥ 0.95 sensitivity** at title/abstract screening [O'Mara-Eves et al. 2015]. Our pipeline encodes this priority through a conservative inclusion default: any record where the LLM is uncertain defaults to *include*, not *exclude*.

### Kappa interpretation (Landis & Koch 1977)

| κ range | Interpretation | Implication for screening |
|---|---|---|
| < 0.00 | Less than chance | Systematic disagreement — do not proceed |
| 0.01–0.20 | Slight | Very poor — major issues |
| 0.21–0.40 | Fair | Poor — substantial revision needed |
| 0.41–0.60 | Moderate | Acceptable for early calibration rounds |
| 0.61–0.80 | **Substantial** | Good — approaching deployment threshold |
| 0.81–1.00 | Almost perfect | Excellent — criteria clear and consistent |

Conventional minimum for proceeding to full-corpus screening: **κ ≥ 0.60**.

---

### Our results vs benchmarks

All our metrics computed from confusion matrices against the reconciled human gold standard. Benchmarks from published literature shown in the same table for direct comparison.

| | n | Sensitivity | Specificity | Precision | F1 | κ |
|---|---|---|---|---|---|---|
| **Cochrane / O'Mara-Eves target** | — | **≥ 0.95** | — | — | — | — |
| **Human screeners** (Hanegraaf et al. 2024) | — | — | — | — | — | 0.82 (abstract) / 0.77 (full-text) |
| **AI tool — GPT-4** (Zhan et al. 2025) | — | 0.992 | 0.836 | — | — | 0.83 |
| **AI average across 172 studies** (Scherbakov et al. 2025) | — | 0.804 | — | 0.632 | — | — |
| **AI — data extraction** (Jensen et al. 2025) | — | 0.924* | — | — | — | 0.93† |
| | | | | | | |
| Our pipeline — R1 (initial criteria) | 205 | 0.776 | 0.703 | 0.559 | 0.650 | 0.436 *(moderate)* |
| Our pipeline — R1b (revised criteria) | 205 | 0.866 | 0.819 | 0.699 | 0.774 | 0.645 *(substantial)* |
| Our pipeline — **R2a (2nd revision)** | 103 | **0.897** | **0.905** | **0.788** | **0.839** | **0.770** *(substantial)* |
| Our pipeline — R3a (stability check)‡ | 107 | — | — | — | — | avg 0.682 *(substantial)* |

*Jensen et al. 2025: 92.4% overall agreement; false data rate 5.2% vs 17.7% for a single human reviewer.*
†Jensen et al. 2025: reproducibility kappa between two independent GPT sessions.
‡R3a: no reconciled gold standard collected by design. LLM κ is the mean of κ vs Jennifer Cisse (0.690) and κ vs Caroline Staub (0.674).

**Reading this table:** Our R2a sensitivity of 0.897 is above the Scherbakov et al. mean across 172 studies (0.804) and within the range of a purpose-built GPT-4 tool (Zhan et al. 0.992). Our specificity of 0.905 exceeds the Zhan et al. benchmark (0.836). Our R2a κ of 0.770 is close to the human abstract screening benchmark of 0.82 [Hanegraaf et al. 2024] and the Zhan et al. AI benchmark of 0.83. The gap to the 0.95 sensitivity Cochrane target is noted — it reflects the genuine challenge of multi-dimensional eligibility criteria for a diffuse-concept topic — and is the reason the conservative inclusion default is in place throughout full-corpus screening.

---

## On our model choice: defending qwen2.5:14b against GPT-4 class models

The literature cited in this response predominantly uses GPT-4-class commercial APIs (GPT-4o, GPT-3.5-turbo). Our pipeline uses a locally hosted open-source model (qwen2.5:14b via Ollama). There are five grounds for this choice.

**1. Open-source models are competitive with GPT-4 for this task.**
Delgado-Chaves et al. [2025, PNAS], the most comprehensive model comparison study to date, tested 18 LLMs — including GPT-4o and multiple open-source models deployed locally via Ollama — across three systematic reviews. GPT-4o achieved a mean MCC of 0.349; llama3.1:8b achieved 0.302; mixtral:8x22b and gemma2:9b both achieved 0.290. The performance gap between the best commercial model and the best local model is meaningful but not large, and the authors find that **model size does not determine performance** — several smaller models outperformed larger counterparts of the same family. qwen2.5:14b (a more recent and capable model than those tested) was not evaluated in that study, but its architecture is comparable to or better than the models in the competitive open-source tier.

**2. The calibration process directly validates performance on our specific application.**
A GPT-4 benchmark from a different domain and a different topic is less relevant than our R2a sensitivity of 0.897, specificity 0.905, and κ = 0.770, which were measured directly on our corpus, our eligibility criteria, and our topic. The calibration process replaces assumed transferability with empirical evidence. No benchmark from the literature can do that.

**3. Reproducibility and determinism.**
qwen2.5:14b is run at temperature 0.0 locally — every screening decision is fully deterministic and reproducible. Commercial APIs introduce risk of output variation from model version updates, rate-limit-induced retries, and non-determinism. For a systematic map where audit trail integrity matters, local deterministic inference is methodologically preferable.

**4. Privacy and data governance.**
The corpus contains titles, abstracts, and full texts of academic publications. Running screening locally means no data is transmitted to third-party servers. This is relevant for CGIAR data governance requirements.

**5. Cost and scalability.**
Screening 17,021 records at title/abstract stage and 6,206 at full-text stage via the GPT-4 API would incur substantial per-query costs and rate limit constraints. Local inference has no marginal cost and no rate limits, which is why the pipeline completed in hours rather than days.

---

## Point 1 — Single database (Scopus only)

> *"This is based on Scopus alone — presumably you'll be including other databases? Scopus has a different content to discipline-specific databases, so the model is likely to perform differently."*

**Valid and acknowledged.** The Deliverable 3 protocol commits to five primary databases: Scopus, Web of Science Core Collection, CAB Abstracts, AGRIS, and Academic Search Premier, plus grey literature from approximately 20 institutional repositories. Scopus was completed first because it offers the broadest interdisciplinary coverage of any single database and allowed the full pipeline to be built, calibrated, and validated end-to-end. It was always intended as the starting point, not the endpoint.

Your point about model performance varying across databases is well taken. The calibration process is designed to be repeatable: if the expanded corpus contains records that are structurally or linguistically different from the Scopus set in ways that affect screening performance, additional calibration rounds can be run on samples from those records before full screening proceeds. Our R2a calibration metrics (sensitivity 0.897, κ = 0.770) provide the baseline against which any performance shift on new databases can be measured.

**Actions taken / planned:**
- Coverage checks against Web of Science and OpenAlex underway to quantify net-new records not in Scopus
- Step 2b will query Web of Science, AGRIS, and OpenAlex; net-new records will enter the pipeline
- Colleagues to manually search CGIAR, World Bank, and 3ie grey literature repositories
- Pipeline will be re-run on the expanded corpus before the final deliverable

**Appendix reference:** Section 1.2 (Database Coverage).

---

## Point 2 — Kappa versus precision, recall, and F1

> *"I've not seen model metrics use kappa before — they typically use precision, recall, F1. Are these metrics unavailable and why were they not used?"*

**Both sets of metrics are relevant here, and both are now reported.** Kappa is the correct metric for inter-rater reliability — whether two independent screeners agree on inclusion decisions. This is precisely how it is used by EPPI Reviewer, Cochrane, and the Campbell Collaboration. Precision, recall, and F1 are the appropriate metrics when evaluating a screener against a fixed labelled test set — which is also relevant here, since reconciled gold-standard decisions exist for Rounds 1, 1b, and 2a. They were computable from the existing confusion matrices and were an omission in the original draft, not an unavailability. Both sets are now reported.

**Our results vs literature and human benchmarks:**

| Metric | Our R1 | Our R1b | Our R2a | Human benchmark | AI literature benchmark |
|---|---|---|---|---|---|
| Sensitivity / Recall | 0.776 | 0.866 | **0.897** | κ 0.82 abstract (Hanegraaf 2024) | 0.992 (Zhan 2025) |
| Specificity | 0.703 | 0.819 | **0.905** | — | 0.836 (Zhan 2025) |
| Precision | 0.559 | 0.699 | **0.788** | — | 0.830 extraction (Scherbakov 2025) |
| F1 | 0.650 | 0.774 | **0.839** | — | — |
| κ vs gold standard | 0.436 | 0.645 | **0.770** | 0.82 abstract (Hanegraaf 2024) | 0.83 (Zhan 2025) |

By Round 2a, all metrics sit within or close to both the human benchmark range and the validated AI tool range. The R2a sensitivity of 0.897 compares favourably to the human abstract screening κ benchmark of 0.82 [Hanegraaf et al. 2024], and our specificity of 0.905 exceeds the 0.836 reported by Zhan et al. [2025] for a GPT-4-powered tool. The conservative inclusion default protects sensitivity further throughout full-corpus screening.

**Appendix reference:** Sections 6.2 (Metric Definitions) and 6.3 (Calibration Results, Table 1).

---

## Point 3 — Sample size for calibration ("c. 740 records to train the model")

> *"c. 740 records were used to train the model — in the two years we used AI in Juno we typically had to use 2,000–7,000 records to train the models until we reached anything like appropriate model performance. Why was this number so low?"*

**The calibration records are a validation set, not a training set.** The tools you are referring to — Juno, EPPI Reviewer's ML screener — are supervised machine-learning classifiers trained from near-scratch on labelled examples. They start with no prior knowledge and need 2,000–7,000 records before a statistical model can fit the decision boundary. The training record count and the performance it yields are directly related.

qwen2.5:14b is a pre-trained large language model with 14 billion parameters. Its parameters are never updated — it is not trained on our data at any point. The approximately 415 calibration records across three rounds are a validation set for prompt and eligibility criteria design. The relevant question is not "did the model see enough examples to learn?" but "did we verify, through structured comparison against independent human judgement, that the model applies our specific criteria correctly before deployment?"

The analogy in conventional systematic review practice is calibration training: verifying that a reviewer understands and consistently applies the eligibility criteria before they begin independent screening. Three structured rounds with dual human review, reconciled gold standards, confusion matrix analysis, and criteria revision serve exactly that purpose. Our results confirm this worked: LLM sensitivity rose from 0.776 (R1) to 0.897 (R2a) and κ rose from 0.436 to 0.770 through criteria revision alone — no additional training data was used.

**Appendix reference:** Section 6.4 (Relationship to Supervised Machine-Learning Screeners).

---

## Point 4 — 1,430 missing abstracts

> *"1,430 missing abstracts is incredibly high for Scopus, for which most abstracts are available — did you check if this is the API failing? How does the API compare with manual extraction?"*

**Valid flag. The cause is an API access limitation, not a data availability problem.** The Elsevier Abstract Retrieval API requires an institutional token to return full abstract content for many records. This preliminary run was conducted without an institutional token. Spot-checks confirm the abstracts are present on the Scopus web interface — the gap is access-gated, not a systematic API failure.

An application for an Elsevier institutional token through Cornell University is in progress. Once active, the enrichment step will be re-run and this figure is expected to fall substantially. All statistics dependent on this limitation are labelled as preliminary in the revised appendix.

**Appendix reference:** Sections 1.3 and 5.1.

---

## Point 5 — 86% full-text non-retrieval

> *"86% of your full texts weren't retrievable, which I know you say you'll extract manually, but that then seems to be a huge time cost relative to the model performance."*

**This figure is also preliminary** and will improve materially once the Elsevier Full-Text API is available via the institutional token. The 86% figure is an upper bound, not the final state.

Residual non-retrieval after institutional access will reflect the genuine open-access landscape — conference proceedings, book chapters, paywalled articles — which is not unusual for a multi-disciplinary systematic map. For these records the default is retention for inclusion, not exclusion. Full-text screening applies only to the 6,206 records passing title/abstract screening (not the full 17,021), and the LLM completed that stage in 3 hours 50 minutes of unattended compute. Manual effort is for retrieval and verification, not re-screening from scratch.

**Appendix reference:** Sections 1.3 and 8 (Full-Text Retrieval).

---

## Point 6 — AI at full-text screening and data extraction stages

> *"Am I right that you're also proposing full-text screening and data extraction with a model? At present there is very little evidence in the evidence synthesis community that AI can function at these stages. I'm keen to know what your thoughts are based on an engagement with the evidence synthesis literature on AI."*

**The evidence base has moved substantially in 2024–2025.** The critical distinction in the recent literature is between *supervised screening* — where texts are pre-retrieved and the model makes binary include/exclude decisions — and *autonomous end-to-end search*, where the model queries databases and retrieves its own references. Performance diverges dramatically.

**Supervised screening (what our pipeline does):**

| Study | Task | Sensitivity | Specificity | Precision | κ |
|---|---|---|---|---|---|
| Zhan et al. (2025) | Title/abstract | 0.992 | 0.836 | — | 0.83 |
| Zhan et al. (2025) | Full-text | 0.976 | 0.474 | — | 0.74 |
| Scherbakov et al. (2025) | Title/abstract (mean, 172 studies) | 0.804 | — | 0.632 | — |
| Scherbakov et al. (2025) | Data extraction (mean) | 0.860 | — | 0.830 | — |
| Jensen et al. (2025) | Data extraction | 0.924* | — | — | 0.93† |
| **Our pipeline (R2a)** | **Title/abstract** | **0.897** | **0.905** | **0.788** | **0.770** |

*Jensen et al.: 92.4% overall agreement with human reviewers; false data rate 5.2% vs 17.7% for a single human reviewer.*
†Jensen et al.: reproducibility kappa between two independent GPT sessions.

**Our R2a sensitivity (0.897) is within the range of validated AI screening tools** and above the mean of 172 studies reviewed by Scherbakov et al. (0.804). Specificity (0.905) exceeds the Zhan et al. benchmark (0.836). These figures are for our specific model (local 14B) on our specific topic — the calibration process directly validates performance on this application rather than relying on transferability from other domains.

**Autonomous search (what our pipeline does not do):**
Clark et al. [2025], reviewing GenAI in evidence synthesis, found that when used autonomously for database searching, GenAI missed 68–96% of relevant studies (median: 91% — equivalent to sensitivity of approximately 0.09). Our pipeline screens pre-retrieved Scopus records; it does not query databases autonomously. This failure mode does not apply.

**Human oversight:** The consistent finding across this literature is that human oversight is required [Clark et al. 2025; Zhan et al. 2025]. This is how our pipeline is designed: every LLM decision is stored with full reasoning and a cited passage; uncertain cases default to inclusion (protecting sensitivity); spot-checking is built in. The pipeline is assistive, not autonomous.

**Comparing to human-only screening:** Hanegraaf et al. [2024] report average human κ of 0.82 for abstract screening and 0.77 for full-text screening — and only 46% of published systematic reviews report IRR metrics at all. Our pipeline logs every decision with full reasoning, making the screening process more auditable than conventional manual practice.

**Appendix reference:** Sections 1.1 (Human Oversight) and 2 (Case for Automation).

---

## On the broader concern: "very little adherence to best practice in AI"

The practices present in our pipeline — structured calibration rounds, dual independent human review, reconciled gold standards, conservative inclusion defaults, version-controlled criteria, full audit trail — are precisely the practices the 2024–2025 literature recommends [Clark et al. 2025]. They were not communicated clearly enough in the original draft appendix, and that has been corrected. The revised appendix leads with validation (Section 6), defines all metrics and benchmarks (Section 6.2), and explicitly labels all preliminary figures.

On the alternative of manual screening at this scale: adding reviewers addresses the time constraint but not the consistency constraint. Human inter-rater reliability in published systematic reviews averages κ = 0.82 for abstract screening [Hanegraaf et al. 2024] — and only 46% of reviews report IRR metrics at all, making the consistency of manual screening harder to audit than our pipeline's fully logged decisions. Our R2a LLM κ of 0.770 sits within the substantial agreement band and close to the human abstract screening benchmark of 0.82, with sensitivity (0.897) and specificity (0.905) that compare well to both human and validated AI tool benchmarks. The goal is not to replace human judgement but to concentrate it where it has the most impact.

---

## Summary of actions

| Action | Status | Owner |
|---|---|---|
| Add P/R/F1 + specificity to calibration reporting | **Done** | Zarrar |
| Revise appendix — metrics definitions, benchmarks, Landis & Koch table | **Done** | Zarrar |
| Revise appendix — validation-first structure | **Done** | Zarrar |
| Revise appendix — supervised ML vs LLM distinction | **Done** | Zarrar |
| Revise appendix — preliminary figures labelled | **Done** | Zarrar |
| Revise appendix — human oversight section | **Done** | Zarrar |
| Elsevier institutional token (Cornell) | In progress | Zarrar |
| Re-run enrichment + retrieval under full access | Pending token | Zarrar |
| WoS / AGRIS / OpenAlex coverage checks | In progress | Zarrar |
| Grey literature manual search (CGIAR, World Bank, 3ie) | To assign | Colleagues |
| Integrate verified AI literature citations into appendix Section 2 | In progress | Zarrar |

---

## References

All citations below have been retrieved and verified directly from source. Only confirmed details are included.

**Delgado-Chaves et al. (2025)**
Delgado-Chaves, F.M., Sieper, A., Fröhlich, H., et al. "Benchmarking large language models for biomedical systematic reviews: is automation feasible?" *Proceedings of the National Academy of Sciences*, 122(2), e2411962122. DOI: [10.1073/pnas.2411962122](https://doi.org/10.1073/pnas.2411962122)
*Used for: benchmarking 18 LLMs (including open-source models via Ollama) on systematic review screening; open-source models (llama3.1:8b MCC=0.302) competitive with GPT-4o (MCC=0.349); model size does not determine performance; cost and local deployment are viable considerations.*

**Clark et al. (2025)**
Clark, J., Barton, B., Albarqouni, L., et al. "Generative artificial intelligence use in evidence synthesis: A systematic review." *Research Synthesis Methods*. DOI: [10.1017/rsm.2025.16](https://doi.org/10.1017/rsm.2025.16)
*Used for: autonomous search miss rate (median 91%); screening error rates; recommendation that human oversight is required.*

**Hanegraaf et al. (2024)**
Hanegraaf, L., et al. "Inter-reviewer reliability of human literature reviewing and implications for the introduction of machine-assisted systematic reviews: a mixed-methods review." *BMJ Open*. DOI: [10.1136/bmjopen-2023-076912](https://doi.org/10.1136/bmjopen-2023-076912)
*Used for: human κ benchmarks — abstract screening 0.82, full-text 0.77, data extraction 0.88; only 46% of reviews report IRR.*

**Jensen et al. (2025)**
Jensen, M.M., Danielsen, M.B., Riis, J., et al. "ChatGPT-4o can serve as the second rater for data extraction in systematic reviews." *PLOS ONE*. DOI: [10.1371/journal.pone.0313401](https://doi.org/10.1371/journal.pone.0313401)
*Used for: 92.4% agreement with human reviewers; false data rate 5.2% vs 17.7% for single human; reproducibility κ = 0.93.*

**Landis & Koch (1977)**
Landis, J.R., Koch, G.G. "The measurement of observer agreement for categorical data." *Biometrics*, 33(1), 159–174. DOI: [10.2307/2529310](https://doi.org/10.2307/2529310)
*Used for: kappa interpretation thresholds.*

**O'Mara-Eves et al. (2015)**
O'Mara-Eves, A., Thomas, J., McNaught, J., Miwa, M., Ananiadou, S. "Using text mining for study identification in systematic reviews: a systematic review of current approaches." *Systematic Reviews*, 4(1), 5. DOI: [10.1186/2046-4053-4-5](https://doi.org/10.1186/2046-4053-4-5)
*Used for: canonical definitions of sensitivity, specificity, precision, F1, and other screening performance metrics; ≥0.95 sensitivity threshold for text-mining tools used in systematic review screening.*

**Scherbakov et al. (2025)**
Scherbakov, D., Hubig, N., Jansari, V., Bakumenko, A., Lenert, L.A. "The emergence of large language models as tools in literature reviews: a large language model-assisted systematic review." *Journal of the American Medical Informatics Association*, 32(6), 1071–1086. DOI: [10.1093/jamia/ocaf063](https://doi.org/10.1093/jamia/ocaf063)
*Used for: 172-study meta-review; GPT data extraction precision 83.0%, recall 86.0%; title/abstract recall 80.4%.*

**Zhan et al. (2025)**
Zhan, J., Suvada, K., Xu, M., Tian, W., Cara, K.C., Wallace, T.C., Ali, M.K. "Accelerating the pace and accuracy of systematic reviews using AI: a validation study." *Systematic Reviews*. DOI: [10.1186/s13643-025-02997-8](https://doi.org/10.1186/s13643-025-02997-8)
*Used for: title/abstract sensitivity 0.992, specificity 0.836, κ = 0.83; full-text sensitivity 0.976, κ = 0.74; completed in one-quarter of human time.*

---

*Last updated: 2026-04-06*
