#!/usr/bin/env python3
"""
step14a_generate_ft_calibration.py

Draw a stratified random sample of ~100 INCLUDEs from the abstract screening
(step12) that have retrieved full texts (step13), for use in full-text
calibration (D5.1).

Reads:
  outputs/step12/step12_results.csv      abstract screening decisions
  outputs/step13/step13_manifest.csv     full-text retrieval status + file paths

Writes:
  outputs/step14a/ft_calibration_pool.csv    full candidate pool (all INCLUDEs with FT)
  outputs/step14a/ft_calibration_sample.csv  sampled 100 records + metadata
  results/EPPI Review - FT-R1a.csv           blank reviewer form (send to Caroline + Jennifer)
  outputs/step14a/FT_Criteria_Guidance.md    criteria reference sheet for reviewers

Sampling strategy:
  Stratified by year band (5 bands, ~20 per band) to ensure temporal coverage.
  Fixed seed for reproducibility.
"""

from __future__ import annotations

import csv
import json
import os
import random
import textwrap
from collections import defaultdict
from pathlib import Path

import yaml

# ── Paths ──────────────────────────────────────────────────────────────────────
HERE        = Path(__file__).resolve().parent
OUT_DIR     = HERE / "outputs" / "step14a"
RESULTS_DIR = HERE / "results"

STEP12_CSV    = HERE / "outputs" / "step12" / "step12_results.csv"
STEP13_CSV    = HERE / "outputs" / "step13" / "step13_manifest.csv"
CRITERIA_YML  = HERE / "criteria_ft_r1.yml"          # FT criteria (may not exist yet)
CRITERIA_FALL = HERE / "criteria_r3a.yml"             # fallback: latest abstract criteria

POOL_CSV      = OUT_DIR / "ft_calibration_pool.csv"
SAMPLE_CSV    = OUT_DIR / "ft_calibration_sample.csv"
REVIEWER_CSV  = RESULTS_DIR / "EPPI Review - FT-R1a.csv"
GUIDANCE_MD   = OUT_DIR / "FT_Criteria_Guidance.md"

# ── Settings ───────────────────────────────────────────────────────────────────
SAMPLE_N      = 100
RANDOM_SEED   = 42

YEAR_BANDS = [
    ("2005–2012", 2005, 2012),
    ("2013–2016", 2013, 2016),
    ("2017–2019", 2017, 2019),
    ("2020–2022", 2020, 2022),
    ("2023–2026", 2023, 2026),
]
PER_BAND = SAMPLE_N // len(YEAR_BANDS)   # 20 each


# ── Helpers ────────────────────────────────────────────────────────────────────

def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def band_for_year(year_str: str) -> str:
    try:
        y = int(str(year_str).strip()[:4])
    except (ValueError, TypeError):
        return "unknown"
    for label, lo, hi in YEAR_BANDS:
        if lo <= y <= hi:
            return label
    return "unknown"


def load_criteria(path: Path) -> dict:
    if path.exists():
        with path.open(encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load step12 INCLUDEs ──────────────────────────────────────────────────
    step12_rows = read_csv(STEP12_CSV)
    includes = {r["dedupe_key"]: r for r in step12_rows if r.get("screen_decision") == "Include"}
    print(f"Step12 INCLUDEs: {len(includes)}")

    # ── Load step13 manifest — keep only 'retrieved' records ─────────────────
    step13_rows = read_csv(STEP13_CSV)
    retrieved   = {r["dedupe_key"]: r for r in step13_rows if r.get("status") == "retrieved"}
    print(f"Step13 retrieved: {len(retrieved)}")

    # ── Candidate pool: INCLUDEs that have a retrieved full text ──────────────
    pool: list[dict] = []
    for key, s12 in includes.items():
        s13 = retrieved.get(key)
        if not s13:
            continue
        pool.append({
            "dedupe_key":     key,
            "scopus_id":      s12.get("scopus_id", ""),
            "doi":            s12.get("doi", ""),
            "title":          s12.get("title", ""),
            "year":           s12.get("year", ""),
            "pub":            s12.get("publicationName", ""),
            "abstract":       s12.get("_abstract_best", s12.get("abstract", "")),
            "abstract_src":   s12.get("_abstract_best_src", ""),
            "ft_file":        s13.get("file_path", ""),
            "ft_source":      s13.get("source", ""),
            "ft_url":         s13.get("url", ""),
            "year_band":      band_for_year(s12.get("year", "")),
        })

    print(f"Candidate pool (INCLUDEs + retrieved FT): {len(pool)}")

    # Write full pool
    pool_fields = ["dedupe_key", "scopus_id", "doi", "title", "year", "pub",
                   "abstract", "abstract_src", "ft_file", "ft_source", "ft_url", "year_band"]
    write_csv(POOL_CSV, pool, pool_fields)
    print(f"Pool written → {POOL_CSV}")

    # ── Stratified sample ─────────────────────────────────────────────────────
    rng = random.Random(RANDOM_SEED)
    by_band: dict[str, list[dict]] = defaultdict(list)
    for rec in pool:
        by_band[rec["year_band"]].append(rec)

    sample: list[dict] = []
    band_counts: dict[str, int] = {}
    for label, lo, hi in YEAR_BANDS:
        candidates = by_band.get(label, [])
        n = min(PER_BAND, len(candidates))
        chosen = rng.sample(candidates, n)
        sample.extend(chosen)
        band_counts[label] = n
        print(f"  Band {label}: {len(candidates)} candidates → sampled {n}")

    # Fill any shortfall from remaining pool records not yet sampled
    if len(sample) < SAMPLE_N:
        sampled_keys = {r["dedupe_key"] for r in sample}
        remainder = [r for r in pool if r["dedupe_key"] not in sampled_keys]
        rng.shuffle(remainder)
        extra = remainder[: SAMPLE_N - len(sample)]
        sample.extend(extra)
        print(f"  Added {len(extra)} extras to reach {len(sample)} total")

    # Sort by year for readability
    sample.sort(key=lambda r: (r.get("year", ""), r.get("title", "")))

    sample_fields = pool_fields + []
    write_csv(SAMPLE_CSV, sample, pool_fields)
    print(f"Sample written ({len(sample)} records) → {SAMPLE_CSV}")

    # ── Reviewer CSV (EPPI-style blank form) ──────────────────────────────────
    reviewer_rows = []
    for rec in sample:
        reviewer_rows.append({
            "Id":                  rec["scopus_id"],
            "Item":                rec["title"],
            "DOI":                 rec["doi"],
            "Year":                rec["year"],
            "Publication":         rec["pub"],
            "Abstract":            rec["abstract"],
            "Full_Text_File":      rec["ft_file"],
            "Caroline Staub":      "",
            "Jennifer Cisse":      "",
            "Reconciled":          "",
            "Reconciliation Notes": "",
        })
    reviewer_fields = ["Id", "Item", "DOI", "Year", "Publication", "Abstract",
                       "Full_Text_File", "Caroline Staub", "Jennifer Cisse",
                       "Reconciled", "Reconciliation Notes"]
    write_csv(REVIEWER_CSV, reviewer_rows, reviewer_fields)
    print(f"Reviewer form written → {REVIEWER_CSV}")

    # ── Summary JSON ──────────────────────────────────────────────────────────
    summary = {
        "total_includes":        len(includes),
        "total_retrieved":       len(retrieved),
        "candidate_pool":        len(pool),
        "sample_n":              len(sample),
        "random_seed":           RANDOM_SEED,
        "band_counts":           band_counts,
    }
    with (OUT_DIR / "step14a_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ── Criteria guidance document ────────────────────────────────────────────
    yml_path = CRITERIA_YML if CRITERIA_YML.exists() else CRITERIA_FALL
    criteria = load_criteria(yml_path)
    criteria_src = "criteria_ft_r1.yml" if CRITERIA_YML.exists() else "criteria_r3a.yml (abstract stage — FT criteria pending)"
    write_guidance(GUIDANCE_MD, criteria, criteria_src, len(sample))

    print(f"Criteria guidance written → {GUIDANCE_MD}")
    print(f"\nDone. Send to Caroline + Jennifer:")
    print(f"  1) {REVIEWER_CSV}")
    print(f"  2) {GUIDANCE_MD}")
    print(f"  3) The full-text files listed in Full_Text_File column")
    print(f"\nBand breakdown:")
    for label, n in band_counts.items():
        print(f"  {label}: {n}")


# ── Criteria guidance renderer ─────────────────────────────────────────────────

def write_guidance(path: Path, criteria: dict, src: str, n: int) -> None:
    crit_section = ""
    crits = (criteria.get("criteria") or {})
    for key, c in crits.items():
        name = c.get("name", key)
        include = c.get("include", "")
        exclude = c.get("exclude", "")
        ft_inc  = c.get("ft_include_further_guidelines", "")
        ft_exc  = c.get("ft_exclude_further_guidelines", "")
        # Fall back to latest abstract-stage guidance if no FT-specific guidance yet
        abs_inc = c.get("r2_include_further_guidelines", "")
        abs_exc = c.get("r2_exclude_further_guidelines", "")

        crit_section += f"### {name}\n\n"
        crit_section += f"**Include if:** {include}\n\n"
        crit_section += f"**Exclude if:** {exclude}\n\n"

        if ft_inc:
            crit_section += f"**Full-text — additional INCLUDE guidance:**\n{textwrap.fill(ft_inc.strip(), 90)}\n\n"
        elif abs_inc:
            crit_section += f"**Additional INCLUDE guidance** *(abstract-stage; FT-specific guidance pending)*:\n{textwrap.fill(abs_inc.strip(), 90)}\n\n"

        if ft_exc:
            crit_section += f"**Full-text — additional EXCLUDE guidance:**\n{textwrap.fill(ft_exc.strip(), 90)}\n\n"
        elif abs_exc:
            crit_section += f"**Additional EXCLUDE guidance** *(abstract-stage; FT-specific guidance pending)*:\n{textwrap.fill(abs_exc.strip(), 90)}\n\n"

        crit_section += "---\n\n"

    content = f"""\
# Full-Text Calibration — Criteria Guidance

**Round:** FT-R1a (first full-text calibration)
**Records to screen:** {n}
**Criteria source:** `{src}`

---

## Instructions

You will receive a CSV (`EPPI Review - FT-R1a.csv`) with {n} records.
Each row includes the paper title, DOI, abstract, and a path to the full-text file.

For each record, open the full-text file and enter your decision in your column:
- `INCLUDE` — paper meets all five criteria based on full-text review
- `EXCLUDE` — paper fails one or more criteria; note which in the Reconciliation Notes column if desired

**Important:**
- Base your decision on the **full text**, not the abstract alone
- The full-text stage applies the same five criteria as the abstract stage but you can now verify claims the abstract left ambiguous
- When genuinely uncertain, lean toward EXCLUDE at this stage (the abstract screen was inclusive; full-text is more precise)
- Record your decision in your named column; leave the Reconciled column blank

Return your completed CSV to Zarrar by **22 April 2026**.

---

## The five criteria

{crit_section}
## Decision guide (quick reference)

| Criterion | INCLUDE | EXCLUDE |
|---|---|---|
| 1 Population | Smallholder / low-income producers (crop, livestock, fisheries, forestry) in LMIC | Large commercial farms, agribusiness without smallholder focus |
| 2 Concept | Assesses adaptation processes or outcomes, adoption, adaptive capacity | Mitigation only; impact-only; general agronomy with no adaptation framing |
| 3 Context | Agricultural setting with explicit climate hazard linkage | Non-agricultural; climate mentioned only incidentally |
| 4 Methodology | Empirical evidence (qualitative or quantitative); scenario-based testing | Purely theoretical; conceptual only; literature reviews / meta-analyses |
| 5 Geography | Low- or Middle-Income Country (LMIC) setting | Exclusively high-income / OECD countries |

All five criteria must be met for INCLUDE. A single EXCLUDE criterion is sufficient to exclude.
"""
    path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
