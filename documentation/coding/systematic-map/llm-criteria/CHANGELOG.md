# LLM Extraction Criteria — Changelog

One line per version. Each entry links to the round that triggered the change and summarises what was updated and why.

| Version | Used in rounds | Date | Summary of changes |
|---------|---------------|------|--------------------|
| v1 | FT-R1a, FT-R1b | 2026-04-14 | Initial criteria — 16 fields per updated codebook (domain_type and validity_notes removed). Tightened field definitions for geographic_scale (study population scope, not data collection site), producer_type (undefined decision rule), marginalized_subpopulations (none_reported mutually exclusive), process_outcome_domains (dedicated indicators only), methodological_approach (participatory as primary design only), data_sources (controlled vocab; interviews → participatory_methods) |

---

## How to add an entry

After reconciling a round and updating the criteria file:

1. Copy `criteria_sysmap_vN.md` → `criteria_v(N+1).md`
2. Apply changes to the new file
3. Add a row to the table above: version, rounds, date, one-sentence summary
4. The summary should name the specific field(s) changed and the reason (e.g. "Revised `domain_type` decision rule — LLM was coding process+outcome papers as `both` when only one domain was assessed; added threshold example")
