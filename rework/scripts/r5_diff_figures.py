#!/usr/bin/env python3
"""
r5_diff_figures.py

Visual diffs for the D5 rework: side-by-side before/after montages of every
human-track figure, plus grouped bar charts of the field-value shifts that
actually feed those figures (self-verified: recomputed directly from the
two step15_human.csv files, not cited from the rework report prose — see
r4_diff_report.py's docstring for why that distinction matters).

Colour: Before = blue #2a78d6, After = orange #eb6834 (categorical slots 1-2
from the dataviz skill's default palette; validated: all checks pass,
CVD normal-vision floor 33.6 — see rework/README.md).

Outputs (under rework/outputs/diffs/):
  side_by_side/<figure>.png   — old (left) vs new (right), labeled
  field_shifts.png            — grouped bar chart, 6 key fields, old vs new
  prisma_side_by_side.png     — original vs rework PRISMA

Run (after r1-r4):
  conda run -n ilri01 python rework/scripts/r5_diff_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

REWORK_ROOT = Path(__file__).parent.parent
REPO_ROOT = REWORK_ROOT.parent

OLD_HUMAN_DIR = REPO_ROOT / "scripts" / "outputs" / "step16" / "interactive" / "human"
NEW_HUMAN_DIR = REWORK_ROOT / "outputs" / "step16" / "interactive" / "human"
OLD_PRISMA = REPO_ROOT / "deliverables" / "prisma_flow_d5.png"
NEW_PRISMA = REWORK_ROOT / "outputs" / "diffs" / "prisma_flow_d5_rework.png"

OLD_HUMAN_CSV = REPO_ROOT / "scripts" / "outputs" / "step15" / "step15_human.csv"
NEW_HUMAN_CSV = REWORK_ROOT / "outputs" / "step15" / "step15_human.csv"

OUT_DIR = REWORK_ROOT / "outputs" / "diffs"
SBS_DIR = OUT_DIR / "side_by_side"

BEFORE_COLOR = "#2a78d6"
AFTER_COLOR = "#eb6834"
LABEL_BG = "#f3f4f6"
TEXT_COLOR = "#374151"

FIGURES = [
    ("evidence_gap_map", "Evidence Gap Map"),
    ("geographic_bar", "Geographic Bar"),
    ("geographic_map", "Geographic Map"),
    ("methodology", "Methodological Approach"),
    ("methodology_detail", "Methodology (detail)"),
    ("equity", "Equity & Inclusion"),
    ("temporal_trends", "Temporal Trends"),
    ("domain_heatmap", "Domain Heatmap"),
    ("domain_type", "Domain Type"),
    ("producer_type", "Producer Type"),
    ("domain_venn", "Process/Outcome Venn"),
]

# field -> (split sep, label)
DIFF_FIELDS = [
    ("marginalized_subpopulations", ";", "Marginalized subpopulations"),
    ("geographic_scale", ";", "Geographic scale"),
    ("producer_type", ";", "Producer type"),
    ("methodological_approach", ";", "Methodological approach"),
    ("process_outcome_domains", ";", "Process/outcome domains"),
    ("cost_data_reported", ";", "Cost data reported"),
]

_NORM = {
    "sub_national": "sub-national", "sub national": "sub-national",
    "sub-national \\ local": "sub-national", "subnational": "sub-national",
    "multi_country": "multi-country", "multi country": "multi-country",
    "local": "local", "small_holder_farmer": "smallholder_farmer",
    "small_holder_farmers": "smallholder_farmer",
    "small_holder_farmer;women": "smallholder_farmer;women",
}


def _norm_value(v: str) -> str:
    v = v.strip().lower()
    return _NORM.get(v, v)


def _value_counts(df: pd.DataFrame, field: str, sep: str) -> pd.Series:
    counts: dict[str, int] = {}
    for raw in df[field].dropna().astype(str):
        for part in raw.replace(",", sep).split(sep):
            part = _norm_value(part)
            if not part or part in {"nan", "none"}:
                continue
            counts[part] = counts.get(part, 0) + 1
    return pd.Series(counts).sort_values(ascending=False)


def _label_strip(img: Image.Image, text: str, color: str) -> Image.Image:
    """Add a coloured label strip above an image."""
    from PIL import ImageDraw, ImageFont

    strip_h = 70
    w, h = img.size
    out = Image.new("RGB", (w, h + strip_h), "white")
    draw = ImageDraw.Draw(out)
    draw.rectangle([0, 0, w, strip_h], fill=color)
    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 28
        )
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((w - tw) / 2, (strip_h - th) / 2 - bbox[1]), text, fill="white", font=font)
    out.paste(img, (0, strip_h))
    return out


def make_side_by_side(old_path: Path, new_path: Path, out_path: Path,
                       old_label: str, new_label: str) -> None:
    old_img = Image.open(old_path).convert("RGB")
    new_img = Image.open(new_path).convert("RGB")

    target_h = max(old_img.height, new_img.height)
    old_img = old_img.resize((int(old_img.width * target_h / old_img.height), target_h))
    new_img = new_img.resize((int(new_img.width * target_h / new_img.height), target_h))

    old_img = _label_strip(old_img, old_label, BEFORE_COLOR)
    new_img = _label_strip(new_img, new_label, AFTER_COLOR)

    gap = 24
    w = old_img.width + new_img.width + gap
    h = max(old_img.height, new_img.height)
    canvas = Image.new("RGB", (w, h), "white")
    canvas.paste(old_img, (0, 0))
    canvas.paste(new_img, (old_img.width + gap, 0))
    canvas.save(out_path)
    print(f"[r5] -> {out_path.relative_to(REWORK_ROOT)}")


def make_field_shifts_chart(old_df: pd.DataFrame, new_df: pd.DataFrame) -> None:
    n_fields = len(DIFF_FIELDS)
    fig, axes = plt.subplots(n_fields, 1, figsize=(9, 2.3 * n_fields))
    fig.suptitle(
        "Field-value shifts: prior (RA) coding vs rework coding\n"
        "(counts among each dataset's own included records — n=151 before, n=115 after)",
        fontsize=12, fontweight="bold", y=1.0, color=TEXT_COLOR,
    )

    for ax, (field, sep, label) in zip(axes, DIFF_FIELDS):
        old_vc = _value_counts(old_df, field, sep)
        new_vc = _value_counts(new_df, field, sep)
        values = sorted(set(old_vc.index) | set(new_vc.index),
                         key=lambda v: -(old_vc.get(v, 0) + new_vc.get(v, 0)))[:8]

        y = range(len(values))
        h = 0.35
        old_vals = [old_vc.get(v, 0) for v in values]
        new_vals = [new_vc.get(v, 0) for v in values]

        ax.barh([i + h / 2 for i in y], old_vals, height=h, color=BEFORE_COLOR, label="Before (n=151)")
        ax.barh([i - h / 2 for i in y], new_vals, height=h, color=AFTER_COLOR, label="After (n=115)")

        for i, v in enumerate(old_vals):
            if v:
                ax.text(v + 0.5, i + h / 2, str(v), va="center", fontsize=8, color=TEXT_COLOR)
        for i, v in enumerate(new_vals):
            if v:
                ax.text(v + 0.5, i - h / 2, str(v), va="center", fontsize=8, color=TEXT_COLOR)

        ax.set_yticks(list(y))
        ax.set_yticklabels(values, fontsize=8.5, color=TEXT_COLOR)
        ax.invert_yaxis()
        ax.set_title(label, fontsize=10.5, fontweight="bold", loc="left", color=TEXT_COLOR)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(left=False)
        ax.set_xlabel("")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=BEFORE_COLOR),
        plt.Rectangle((0, 0), 1, 1, color=AFTER_COLOR),
    ]
    fig.legend(handles, ["Before (prior RA coding, n=151)", "After (rework, n=115)"],
               loc="lower center", ncol=2, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    out_path = OUT_DIR / "field_shifts.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[r5] -> {out_path.relative_to(REWORK_ROOT)}")


def main() -> None:
    SBS_DIR.mkdir(parents=True, exist_ok=True)

    for name, label in FIGURES:
        old_png = OLD_HUMAN_DIR / f"{name}.png"
        new_png = NEW_HUMAN_DIR / f"{name}.png"
        if not old_png.exists() or not new_png.exists():
            print(f"[r5] SKIP {name}: missing {'old' if not old_png.exists() else 'new'} PNG")
            continue
        make_side_by_side(
            old_png, new_png, SBS_DIR / f"{name}.png",
            f"BEFORE — {label} (n=151)", f"AFTER — {label} (n=115, rework)",
        )

    if OLD_PRISMA.exists() and NEW_PRISMA.exists():
        make_side_by_side(
            OLD_PRISMA, NEW_PRISMA, OUT_DIR / "prisma_side_by_side.png",
            "BEFORE — PRISMA (n=151 included)", "AFTER — PRISMA (n=115, rework, prelim)",
        )

    old_df = pd.read_csv(OLD_HUMAN_CSV, dtype=str)
    new_df = pd.read_csv(NEW_HUMAN_CSV, dtype=str)
    make_field_shifts_chart(old_df, new_df)

    print("[r5] Done.")


if __name__ == "__main__":
    main()
