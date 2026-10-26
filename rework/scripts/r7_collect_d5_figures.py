#!/usr/bin/env python3
"""
r7_collect_d5_figures.py

Collects every figure that deliverables/publish_d5_from_d4.py actually embeds
into the D5 docx (its FIGURES + NEW_FIGURES lists, Figures 1-9) into one flat
folder, rework/outputs/d5_figures/, named exactly as that script expects them.

This makes the swap mechanical: once the rework numbers are approved, drop
this folder's contents over the corresponding original paths (see the table
this script prints) and re-run publish_d5_from_d4.py — no path hunting.

Sources:
  - 8 of 9 figures come from the rework run (r2_run_figures.py /
    r3_gen_prisma_rework.py) — these change because they're built from the
    confirmed_include rework database (currently 108/151 records, per the
    2026-10-26 borderline adjudication — llm_vs_human.png included: it
    compares the unchanged LLM corpus against whichever human.csv is on disk).
  - 1 is carried over UNCHANGED from the original pipeline: saturation.png,
    the information-saturation curve across RA coding batches — a property
    of the coding *process*, not the eligibility recoding, so the rework
    doesn't touch it. See the per-file notes in the printed manifest.

Run (after r1-r3; r5/r6 not required):
  conda run -n ilri01 python rework/scripts/r7_collect_d5_figures.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

REWORK_ROOT = Path(__file__).parent.parent
REPO_ROOT = REWORK_ROOT.parent

ORIG_STEP16 = REPO_ROOT / "scripts" / "outputs" / "step16"
ORIG_HUMAN = ORIG_STEP16 / "interactive" / "human"
ORIG_DELIV = REPO_ROOT / "deliverables"
ORIG_SUMMARY = REPO_ROOT / "scripts" / "outputs" / "step15" / "step15b_summary.json"

NEW_STEP16 = REWORK_ROOT / "outputs" / "step16"
NEW_HUMAN = NEW_STEP16 / "interactive" / "human"
NEW_DIFFS = REWORK_ROOT / "outputs" / "diffs"
NEW_SUMMARY = REWORK_ROOT / "outputs" / "step15" / "step15b_summary.json"

OUT_DIR = REWORK_ROOT / "outputs" / "d5_figures"

_orig_n = json.loads(ORIG_SUMMARY.read_text())["final_human_records"] if ORIG_SUMMARY.exists() else "?"
_new_n = json.loads(NEW_SUMMARY.read_text())["final_human_records"] if NEW_SUMMARY.exists() else "?"
_CHANGED_N = f"CHANGED (n={_orig_n} -> {_new_n})"

# (output filename, source path, where it replaces in the live repo, changed?)
MANIFEST = [
    ("prisma_flow_d5.png", NEW_DIFFS / "prisma_flow_d5_rework.png",
     "deliverables/prisma_flow_d5.png", _CHANGED_N),
    ("evidence_gap_map.png", NEW_HUMAN / "evidence_gap_map.png",
     "scripts/outputs/step16/interactive/human/evidence_gap_map.png", "CHANGED"),
    ("geographic_map.png", NEW_HUMAN / "geographic_map.png",
     "scripts/outputs/step16/interactive/human/geographic_map.png", "CHANGED"),
    ("geographic_bar.png", NEW_HUMAN / "geographic_bar.png",
     "scripts/outputs/step16/interactive/human/geographic_bar.png", "CHANGED"),
    ("methodology.png", NEW_HUMAN / "methodology.png",
     "scripts/outputs/step16/interactive/human/methodology.png", "CHANGED"),
    ("temporal_trends.png", NEW_HUMAN / "temporal_trends.png",
     "scripts/outputs/step16/interactive/human/temporal_trends.png", "CHANGED"),
    ("equity.png", NEW_HUMAN / "equity.png",
     "scripts/outputs/step16/interactive/human/equity.png", "CHANGED"),
    ("saturation.png", ORIG_STEP16 / "saturation.png",
     "scripts/outputs/step16/saturation.png",
     "UNCHANGED (coding-process saturation, not affected by eligibility rework)"),
    ("llm_vs_human.png", NEW_STEP16 / "llm_vs_human.png",
     "scripts/outputs/step16/llm_vs_human.png", f"CHANGED (human side now n={_new_n})"),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{'file':<24} {'status':<55} replaces")
    print("-" * 110)
    missing = []
    for out_name, src, replaces, status in MANIFEST:
        if not src.exists():
            missing.append((out_name, src))
            print(f"{out_name:<24} {'MISSING — ' + status:<55} {replaces}")
            continue
        shutil.copy2(src, OUT_DIR / out_name)
        print(f"{out_name:<24} {status:<55} {replaces}")

    if missing:
        print(f"\n[r7] WARNING: {len(missing)} file(s) missing — run the earlier "
              f"rework scripts (r1-r3) first.")
    print(f"\n[r7] Collected {len(MANIFEST) - len(missing)}/{len(MANIFEST)} figures "
          f"-> {OUT_DIR}")
    print("[r7] To swap into the live deliverable: copy each file over the path "
          "in the 'replaces' column above, then re-run "
          "deliverables/publish_d5_from_d4.py.")


if __name__ == "__main__":
    main()
