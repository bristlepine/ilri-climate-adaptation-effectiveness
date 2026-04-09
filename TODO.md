# TODO

Track active work items. Completed items are removed. Add new items as needed.

---

## Reviewer response (Neal Haddaway)
- [ ] Send response email with link to RESPONSE_TO_NEAL.md and step11_evolution.png
- [ ] Full-text screening calibration — run a dedicated calibration round for qwen2.5:14b against a full-text gold standard on our corpus (acknowledged gap; condition for accepting full-text screening results)
  - **Implementation plan:** Insert as step14a/b/c between retrieval (step13) and corpus screening (step14→renamed step15)
  - `step14a_generate_ft_calibration.py` — draw ~100 records from step12 INCLUDEs with full texts retrieved; output blank `scripts/results/EPPI Review - FT-R1a.csv` (cols: Id, Item, DOI, Abstract, Caroline Staub, Jennifer Cisse, Reconciled, Reconciliation Notes) + rendered criteria guidance .md to send to reviewers
  - `step14b_ft_irr_analysis.py` — after reviewers return filled CSV: run LLM on full texts → fill LLM column → compute IRR (reuse step11 functions: `cohen_kappa`, `confusion_vs_reconciled`, `detect_columns`); output figures + JSON to `outputs/step14b/`
  - `step14c_test_ft_criteria.py` — after revising criteria: re-run LLM on calibration set → compare to Reconciled → print false negatives with quotes to guide next revision
  - `scripts/criteria_ft_r1.yml` — same 5 criteria as `criteria.yml`, same existing abstract guidance fields, plus two new fields per criterion: `ft_include_further_guidelines` / `ft_exclude_further_guidelines` (start minimal; FT stage can be stricter than abstract)
  - Rename existing step14→15, step15→16, step16→17 across: script files, `config.py`, `run.py`, `METHODOLOGY.md`, `methodology.py`, `README.md`, output dirs
  - Extend `step11_criteria_evolution.py` with a `ROUNDS_FT` list to track FT criteria changes

## Data coverage
- [ ] Run WoS, AGRIS, OpenAlex, Academic Search Premier queries (step2b); quantify net-new records vs Scopus
- [ ] Assign grey literature manual search: CGIAR, World Bank, 3ie (~20 repositories per D3 protocol)
- [ ] Supplement automated full-text retrieval with manual campus library collection

## Pipeline — next runs
- [ ] Re-run step9/step9a once Elsevier token is stable to recover remaining ~1,314 missing abstracts
- [ ] Re-run step12 (full-corpus screening) once multi-database records are integrated

---

*Last updated: 2026-04-08*
