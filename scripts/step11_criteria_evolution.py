"""
step11_criteria_evolution.py

Diffs consecutive criteria YAML files (criteria_r1.yml, criteria_r1a.yml, ...)
and produces a human-readable changelog showing exactly what changed between
calibration rounds. Outputs sit alongside the step11 performance evolution files.

Usage:
    conda run -n ilri01 python scripts/step11_criteria_evolution.py

Outputs:
    scripts/outputs/step11/criteria_evolution.md   — human-readable changelog
    scripts/outputs/step11/criteria_evolution.csv  — one row per change
"""

import csv
import difflib
import re
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPTS_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = SCRIPTS_DIR / "outputs" / "step11"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Ordered list of (round_label, filename).
# Each file is the criteria as used for that round — do not edit existing entries.
# Add a new entry here when a new criteria version is created for a new round.
# R3a used criteria_r2b.yml with no changes — so no new entry for R3a.
ROUNDS = [
    ("R1",  "criteria_r1.yml"),
    ("R1a", "criteria_r1a.yml"),
    ("R1b", "criteria_r1b.yml"),
    ("R2",  "criteria_r2.yml"),
    ("R2a", "criteria_r2a.yml"),   # frozen: as-run for R2a calibration (commit 5059f8c)
    ("R2b", "criteria_r2b.yml"),   # 3 amendments post-R2a; calibrated on R2 sample
    ("R3a", "criteria_r3a.yml"),   # frozen: as-run for R3a calibration (criteria.yml)
]

CRITERION_ORDER = [
    "1_population",
    "2_concept",
    "3_context",
    "4_methodology",
    "5_geography",
]

# Fields to diff per criterion (in display order).
# Any key ending in _include_further_guidelines or _exclude_further_guidelines is also
# picked up dynamically — add new round prefixes here as needed.
FIELDS = [
    "include",
    "exclude",
    "r1_include_further_guidelines",
    "r1_exclude_further_guidelines",
    "r2_include_further_guidelines",
    "r2_exclude_further_guidelines",
]
FIELD_LABELS = {
    "include":                       "Include rule",
    "exclude":                       "Exclude rule",
    "r1_include_further_guidelines": "R1 include guidance",
    "r1_exclude_further_guidelines": "R1 exclude guidance",
    "r2_include_further_guidelines": "R2 include guidance",
    "r2_exclude_further_guidelines": "R2 exclude guidance",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def normalise(text) -> str:
    """Strip excess whitespace from multi-line YAML strings."""
    if text is None:
        return ""
    return " ".join(str(text).split())


def unified_diff_lines(old: str, new: str) -> str:
    """Return a compact inline diff, or empty string if identical."""
    if old == new:
        return ""
    old_words = old.split()
    new_words = new.split()
    sm = difflib.SequenceMatcher(None, old_words, new_words)
    parts = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            parts.append(" ".join(old_words[i1:i2]))
        elif tag == "replace":
            parts.append(f"[-{' '.join(old_words[i1:i2])}-]")
            parts.append(f"[+{' '.join(new_words[j1:j2])}+]")
        elif tag == "delete":
            parts.append(f"[-{' '.join(old_words[i1:i2])}-]")
        elif tag == "insert":
            parts.append(f"[+{' '.join(new_words[j1:j2])}+]")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_changelog():
    rows = []  # for CSV

    md_lines = [
        "# Eligibility Criteria Evolution Across Calibration Rounds",
        "",
        "Each section shows what changed between consecutive rounds.",
        "`[-old-]` = removed, `[+new+]` = added.",
        "",
    ]

    # Load all available rounds
    loaded = []
    for label, fname in ROUNDS:
        path = SCRIPTS_DIR / fname
        if not path.exists():
            print(f"  Skipping {fname} (not found)")
            continue
        loaded.append((label, load(path)))

    if len(loaded) < 2:
        print("Need at least two criteria files to diff.")
        return

    for i in range(1, len(loaded)):
        prev_label, prev_data = loaded[i - 1]
        curr_label, curr_data = loaded[i]

        md_lines.append(f"## {prev_label} → {curr_label}")
        md_lines.append("")

        prev_criteria = prev_data.get("criteria", {})
        curr_criteria = curr_data.get("criteria", {})

        any_change = False

        for ckey in CRITERION_ORDER:
            prev_c = prev_criteria.get(ckey, {})
            curr_c = curr_criteria.get(ckey, {})
            cname = curr_c.get("name") or prev_c.get("name") or ckey

            changes_in_criterion = []

            for field in FIELDS:
                old_val = normalise(prev_c.get(field))
                new_val = normalise(curr_c.get(field))
                if old_val == new_val:
                    continue

                label = FIELD_LABELS.get(field, field)
                diff_str = unified_diff_lines(old_val, new_val)

                if not old_val:
                    summary = f"**Added** {label}: {new_val}"
                    change_type = "added"
                elif not new_val:
                    summary = f"**Removed** {label}: {old_val}"
                    change_type = "removed"
                else:
                    summary = f"**Revised** {label}: {diff_str}"
                    change_type = "revised"

                changes_in_criterion.append(f"  - {summary}")
                rows.append({
                    "transition":   f"{prev_label}→{curr_label}",
                    "criterion":    cname,
                    "field":        label,
                    "change_type":  change_type,
                    "prev_value":   old_val,
                    "new_value":    new_val,
                })
                any_change = True

            if changes_in_criterion:
                md_lines.append(f"### {cname}")
                md_lines.extend(changes_in_criterion)
                md_lines.append("")

        if not any_change:
            md_lines.append("*No changes to eligibility criteria between these rounds.*")
            md_lines.append("")

        # Hard filter changes
        prev_hf = prev_data.get("hard_filters", {})
        curr_hf = curr_data.get("hard_filters", {})
        if prev_hf != curr_hf:
            md_lines.append("### Hard filters")
            for k in set(list(prev_hf.keys()) + list(curr_hf.keys())):
                if prev_hf.get(k) != curr_hf.get(k):
                    md_lines.append(f"  - **{k}:** `{prev_hf.get(k)}` → `{curr_hf.get(k)}`")
            md_lines.append("")

        md_lines.append("---")
        md_lines.append("")

    # Write markdown
    md_path = OUTPUTS_DIR / "criteria_evolution.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Written: {md_path}")

    # Write CSV
    csv_path = OUTPUTS_DIR / "criteria_evolution.csv"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["transition", "criterion", "field", "change_type", "prev_value", "new_value"])
            w.writeheader()
            w.writerows(rows)
        print(f"Written: {csv_path}")
    else:
        print("No changes found between any rounds.")


if __name__ == "__main__":
    build_changelog()
