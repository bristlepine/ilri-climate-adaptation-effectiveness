#!/usr/bin/env python3
"""
step16_map_visualise.py

Step 16: Systematic map visualisations.

Produces all figures required for the systematic map report and journal
submission, including the ROSES flow diagram and evidence map figures.

Figures produced:
  1. roses_flow.png              — ROSES flow diagram (record counts at each stage)
  2. temporal_trends.png         — publications per year
  3. producer_type_bar.png       — breakdown by producer type
  4. methodology_bar.png         — breakdown by methodological approach
  5. domain_heatmap.png          — process/outcome domains × producer type heat map
  6. geographic_bar.png          — top countries by study count
  7. geographic_map.png          — choropleth (requires geopandas; skipped if unavailable)
  8. domain_type_bar.png         — adaptation process vs outcome vs both
  9. equity_bar.png              — equity & inclusion dimensions

All figures are data-driven from step15_coded.csv and step meta.json files.
Re-running after more full texts are retrieved automatically updates all figures.

Inputs:
  - outputs/step15/step15_coded.csv
  - outputs/step15/step15_coded.meta.json
  - outputs/step12/step12_results.meta.json
  - outputs/step13/step13_summary.json
  - outputs/step14/step14_results.meta.json

Outputs (under outputs/step16/):
  - roses_flow.png
  - temporal_trends.png
  - producer_type_bar.png
  - methodology_bar.png
  - domain_heatmap.png
  - geographic_bar.png
  - geographic_map.png  (optional — requires geopandas)
  - domain_type_bar.png
  - equity_bar.png
  - step16_figures.meta.json

Run:
  python step16_map_visualise.py
  (or via run.py with run_step16 = 1)
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# =============================================================================
# Colour palette
# =============================================================================

BLUE    = "#2196F3"
TEAL    = "#009688"
GREEN   = "#4CAF50"
ORANGE  = "#FF9800"
RED     = "#F44336"
PURPLE  = "#9C27B0"
GREY    = "#9E9E9E"
DKGREY  = "#424242"

PALETTE = [BLUE, TEAL, GREEN, ORANGE, RED, PURPLE, GREY,
           "#00BCD4", "#8BC34A", "#FF5722", "#3F51B5", "#795548"]


# =============================================================================
# Paths
# =============================================================================

def _out_dir(out_root: Path) -> Path:
    d = out_root / "step16"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _step15_csv(out_root: Path) -> Path:
    return out_root / "step15" / "step15_coded.csv"

def _step15_meta(out_root: Path) -> Path:
    return out_root / "step15" / "step15_coded.meta.json"

def _step12_meta(out_root: Path) -> Path:
    return out_root / "step12" / "step12_results.meta.json"

def _step13_meta(out_root: Path) -> Path:
    return out_root / "step13" / "step13_summary.json"

def _step14_meta(out_root: Path) -> Path:
    return out_root / "step14" / "step14_results.meta.json"


# =============================================================================
# Helpers
# =============================================================================

def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_coded(out_root: Path) -> pd.DataFrame:
    p = _step15_csv(out_root)
    if not p.exists():
        print(f"[step16] step15 coded CSV not found: {p}")
        return pd.DataFrame()
    df = pd.read_csv(p, engine="python", on_bad_lines="skip")
    # Only figures on full-text coded records
    full = df[df.get("coding_source", pd.Series(dtype=str)).astype(str) == "full_text"].copy()
    print(f"[step16] Full-text coded records available for figures: {len(full):,}")
    return full


_SKIP = {"nan", "", "not_found", "n/a", "none", "unknown", "unclear"}

def _split_multi(series: pd.Series) -> List[str]:
    """Expand semicolon-separated multi-value fields into a flat list."""
    out: List[str] = []
    for val in series.dropna().astype(str):
        for part in val.split(";"):
            part = part.strip()
            if part and part.lower() not in _SKIP:
                out.append(part)
    return out


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _save(fig, path: Path) -> None:
    import matplotlib.pyplot as plt
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[step16] Saved -> {path.name}")


# =============================================================================
# 1. ROSES FLOW DIAGRAM
# =============================================================================

def _roses_flow(out_root: Path, out_dir: Path) -> None:
    """Draw the ROSES flow diagram using numbers from meta.json files."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[step16] matplotlib not available — skipping ROSES diagram")
        return

    s12 = _load_json(_step12_meta(out_root))
    s13 = _load_json(_step13_meta(out_root))
    s14 = _load_json(_step14_meta(out_root))
    s15 = _load_json(_step15_meta(out_root))

    # Pull numbers
    n_identified    = s12.get("rows_total", 0)
    dec12           = s12.get("decision_counts", {})
    n_abs_include   = dec12.get("Include", 0)
    n_abs_exclude   = dec12.get("Exclude", 0)
    n_missing_abs   = s12.get("missing_abstract_count", 0)
    excl12          = s12.get("excluded_by_criterion", {})

    status13        = s13.get("status_counts", {})
    n_ft_retrieved  = status13.get("retrieved", 0)
    n_ft_not_found  = status13.get("needs_manual", 0)

    dec14           = s14.get("decision_counts", {})
    n_ft_include    = dec14.get("Include", 0)
    n_ft_exclude    = dec14.get("Exclude", 0)
    n_ft_manual     = dec14.get("Needs_Manual", 0)
    excl14          = s14.get("excluded_by_criterion", {})

    src15           = s15.get("coding_source_counts", {})
    n_coded_ft      = src15.get("full_text", 0)
    n_pending       = sum(v for k, v in src15.items() if k != "full_text")

    # wider canvas, side boxes pushed further right to avoid overlap
    fig, ax = plt.subplots(figsize=(18, 22))
    ax.set_xlim(0, 14)
    ax.set_ylim(5.5, 24)
    ax.axis("off")
    fig.patch.set_facecolor("#FAFAFA")

    CX  = 5.0    # main flow centre
    EX  = 11.0   # side box centre (well separated from main)
    BW  = 4.2    # main box width
    BH  = 0.95   # main box height
    EW  = 4.0    # side box width
    GAP = 1.8    # vertical gap between boxes

    def box(x, y, w, h, text, color=BLUE, fontsize=9, text_color="white"):
        rect = mpatches.FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.12",
            facecolor=color, edgecolor="white", linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center",
                fontsize=fontsize, color=text_color,
                fontweight="bold", multialignment="center")

    def side_box(x, y, w, h, text, color="#FFF3E0", text_color=DKGREY):
        rect = mpatches.FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.12",
            facecolor=color, edgecolor=ORANGE, linewidth=1.2,
        )
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center",
                fontsize=8, color=text_color, multialignment="center")

    def arrow(x, y1, y2):
        ax.annotate("", xy=(x, y2 + 0.05), xytext=(x, y1 - 0.05),
                    arrowprops=dict(arrowstyle="->", color=DKGREY, lw=1.5))

    def horiz_arrow(x1, x2, y):
        ax.annotate("", xy=(x2 - 0.05, y), xytext=(x1 + 0.05, y),
                    arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))

    MX = 7.0   # true horizontal centre of the 0-14 canvas

    # ── Title ──────────────────────────────────────────────────────────────
    ax.text(MX, 23.4, "ROSES Flow Diagram", ha="center", va="center",
            fontsize=15, fontweight="bold", color=DKGREY)
    ax.text(MX, 22.95,
            "Measuring What Matters: Tracking Climate Adaptation Processes and Outcomes for Smallholder Producers",
            ha="center", va="center", fontsize=8, color=GREY, style="italic")

    # ── Legend (centred, below subtitle) ──────────────────────────────────
    leg_items = [
        (TEAL,      "Identification"),
        (BLUE,      "Screening / Retrieval"),
        (GREEN,     "Included / Coded"),
        ("#FFF3E0", "Exclusions / Notes", ORANGE),
    ]
    total_w = sum(len(lbl) * 0.13 + 0.7 for *_, lbl in
                  [(c, l) for c, l, *_ in leg_items])
    lx = MX - total_w / 2
    leg_y = 22.52
    for item in leg_items:
        fc     = item[0]
        lbl    = item[1]
        ec     = item[2] if len(item) > 2 else "white"
        ax.add_patch(mpatches.FancyBboxPatch(
            (lx, leg_y - 0.13), 0.26, 0.26,
            boxstyle="round,pad=0.03", facecolor=fc, edgecolor=ec, linewidth=1.0))
        ax.text(lx + 0.32, leg_y, lbl, va="center", fontsize=7.5, color=DKGREY)
        lx += len(lbl) * 0.13 + 0.7

    # ── Layout: boxes spaced evenly top-to-bottom ──────────────────────────
    # Main box Y positions
    Y1 = 21.5   # Identification
    Y2 = 19.8   # Abstract screening
    Y3 = 17.8   # Abstract included
    Y4 = 15.8   # Full-text retrieval attempted
    Y5 = 13.8   # Full texts retrieved
    Y6 = 11.8   # Full-text screening
    Y7 =  9.8   # Included
    Y8 =  7.8   # Data extraction

    box(CX, Y1, BW, BH, f"Records identified from Scopus\nn = {n_identified:,}", color=TEAL)
    arrow(CX, Y1 - BH/2, Y2 + BH/2)

    box(CX, Y2, BW, BH,
        f"Abstract screening\nn = {n_identified:,} screened", color=BLUE)
    crit_lines = "\n".join(
        f"  {k}: {v:,}" for k, v in sorted(excl12.items(), key=lambda x: -x[1])
    )
    side_box(EX, Y2, EW, 1.8,
             f"Excluded at abstract screening\nn = {n_abs_exclude:,}\n\nBy criterion:\n{crit_lines}")
    horiz_arrow(CX + BW/2, EX - EW/2, Y2)
    arrow(CX, Y2 - BH/2, Y3 + BH/2)

    box(CX, Y3, BW, BH,
        f"Records included after abstract screening\nn = {n_abs_include:,}  (incl. {n_missing_abs:,} missing abstracts)",
        color=BLUE)
    side_box(EX, Y3, EW, 0.9,
             f"Missing abstracts included conservatively\nn = {n_missing_abs:,}")
    horiz_arrow(CX + BW/2, EX - EW/2, Y3)
    arrow(CX, Y3 - BH/2, Y4 + BH/2)

    box(CX, Y4, BW, BH,
        f"Full-text retrieval attempted\nn = {n_abs_include:,}", color=BLUE)
    src13 = s13.get("source_breakdown", {})
    src_lines = "\n".join(
        f"  {k}: {v:,}" for k, v in sorted(src13.items(), key=lambda x: -x[1])
    )
    side_box(EX, Y4, EW, 1.6,
             f"Not retrieved (needs manual)\nn = {n_ft_not_found:,}\n\nRetrieved by source:\n{src_lines}")
    horiz_arrow(CX + BW/2, EX - EW/2, Y4)
    arrow(CX, Y4 - BH/2, Y5 + BH/2)

    box(CX, Y5, BW, BH,
        f"Full texts retrieved\nn = {n_ft_retrieved:,}", color=BLUE)
    arrow(CX, Y5 - BH/2, Y6 + BH/2)

    box(CX, Y6, BW, BH,
        f"Full-text screening\nn = {n_ft_retrieved:,} screened", color=BLUE)
    crit14_lines = "\n".join(
        f"  {k}: {v:,}" for k, v in sorted(excl14.items(), key=lambda x: -x[1])
    ) if excl14 else "  (none)"
    side_box(EX, Y6, EW, 1.5,
             f"Excluded at full-text screening\nn = {n_ft_exclude:,}\n\nBy criterion:\n{crit14_lines}")
    horiz_arrow(CX + BW/2, EX - EW/2, Y6)
    arrow(CX, Y6 - BH/2, Y7 + BH/2)

    box(CX, Y7, BW, BH,
        f"Studies included in systematic map\nn = {n_ft_include:,}", color=GREEN)
    arrow(CX, Y7 - BH/2, Y8 + BH/2)

    box(CX, Y8, BW, BH,
        f"Data extraction (coded from full text)\nn = {n_coded_ft:,}  |  Pending full text: {n_pending:,}",
        color=GREEN)
    side_box(EX, Y8, EW, 1.1,
             f"Pending full text retrieval\nn = {n_pending:,}\n(abstract-only / missing / extraction failed)")
    horiz_arrow(CX + BW/2, EX - EW/2, Y8)

    ax.text(MX, Y8 - BH/2 - 0.35, f"Generated: {_now_utc()}",
            ha="center", fontsize=7.5, color=GREY, style="italic")

    _save(fig, out_dir / "roses_flow.png")


# =============================================================================
# 2. TEMPORAL TRENDS
# =============================================================================

def _temporal_trends(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    col = "publication_year_value"
    if col not in df.columns:
        return

    years = pd.to_numeric(df[col], errors="coerce").dropna().astype(int)
    years = years[(years >= 2005) & (years <= 2027)]
    if years.empty:
        return

    counts = years.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(counts.index, counts.values, color=BLUE, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                str(val), ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Publication Year", fontsize=11)
    ax.set_ylabel("Number of Studies", fontsize=11)
    ax.set_ylim(0, counts.max() * 1.12)
    ax.set_title(f"Studies by Publication Year  (n={len(years):,})", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, out_dir / "temporal_trends.png")


# =============================================================================
# 3. PRODUCER TYPE
# =============================================================================

def _producer_type_bar(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    col = "producer_type_value"
    if col not in df.columns:
        return

    items = _split_multi(df[col])
    if not items:
        return

    counts = pd.Series(items).value_counts()
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(counts.index[::-1], counts.values[::-1],
                   color=TEAL, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, counts.values[::-1]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                str(val), va="center", fontsize=9)
    ax.set_xlabel("Number of Studies", fontsize=11)
    ax.set_title(f"Producer Types Studied  (n={len(df):,}; multi-select)", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, out_dir / "producer_type_bar.png")


# =============================================================================
# 4. METHODOLOGY
# =============================================================================

def _methodology_bar(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    col = "methodological_approach_value"
    if col not in df.columns:
        return

    vals = df[col].dropna().astype(str).str.strip()
    vals = vals[vals.str.lower().isin(["", "nan", "not_found"]) == False]
    if vals.empty:
        return

    counts = vals.value_counts()
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = PALETTE[:len(counts)]
    bars = ax.barh(counts.index[::-1], counts.values[::-1],
                   color=colors[::-1], edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, counts.values[::-1]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                str(val), va="center", fontsize=9)
    ax.set_xlabel("Number of Studies", fontsize=11)
    ax.set_title(f"Methodological Approach  (n={len(vals):,})", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, out_dir / "methodology_bar.png")


# =============================================================================
# 5. DOMAIN HEATMAP (process/outcome domains × producer type)
# =============================================================================

def _domain_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    dom_col  = "process_outcome_domains_value"
    prod_col = "producer_type_value"
    if dom_col not in df.columns or prod_col not in df.columns:
        return

    rows = []
    for _, row in df.iterrows():
        domains   = [d.strip() for d in str(row.get(dom_col, "")).split(";")
                     if d.strip() and d.strip().lower() not in _SKIP]
        prodtypes = [p.strip() for p in str(row.get(prod_col, "")).split(";")
                     if p.strip() and p.strip().lower() not in _SKIP]
        for d in domains:
            for p in prodtypes:
                rows.append({"domain": d, "producer": p})

    if not rows:
        return

    ct = pd.DataFrame(rows).pivot_table(
        index="domain", columns="producer", aggfunc="size", fill_value=0
    )
    if ct.empty:
        return

    fig, ax = plt.subplots(figsize=(max(8, ct.shape[1] * 1.5), max(6, ct.shape[0] * 0.6)))
    im = ax.imshow(ct.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(ct.columns)))
    ax.set_yticks(range(len(ct.index)))
    ax.set_xticklabels(ct.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(ct.index, fontsize=9)
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            val = ct.values[i, j]
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center", fontsize=9,
                        color="white" if val > ct.values.max() * 0.6 else DKGREY)
    from matplotlib.colorbar import make_axes
    plt.colorbar(im, ax=ax, shrink=0.7, label="Number of Studies")
    ax.set_title("Process/Outcome Domains × Producer Type", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, out_dir / "domain_heatmap.png")


# =============================================================================
# 6. GEOGRAPHIC BAR
# =============================================================================

def _geographic_bar(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    col = "country_region_value"
    if col not in df.columns:
        return

    items = _split_multi(df[col])
    if not items:
        return

    counts = pd.Series(items).value_counts()
    fig_h = max(6, len(counts) * 0.35)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    colors = [BLUE if i < 10 else GREY for i in range(len(counts))]
    bars = ax.barh(counts.index[::-1], counts.values[::-1],
                   color=colors[::-1], edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, counts.values[::-1]):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                str(val), va="center", fontsize=8)
    ax.set_xlabel("Number of Studies", fontsize=11)
    ax.set_title(f"Countries / Regions by Study Count  (n={len(df):,})",
                 fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, out_dir / "geographic_bar.png")


# =============================================================================
# 7. GEOGRAPHIC CHOROPLETH (optional — requires geopandas)
# =============================================================================

def _geographic_map(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return
    try:
        import geopandas as gpd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[step16] geopandas not available — skipping choropleth map")
        return

    col = "country_region_value"
    if col not in df.columns:
        return

    items = _split_multi(df[col])
    if not items:
        return

    counts = pd.Series(items).value_counts().reset_index()
    counts.columns = ["country", "count"]

    try:
        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    except Exception:
        print("[step16] Could not load naturalearth dataset — skipping choropleth")
        return

    merged = world.merge(counts, left_on="name", right_on="country", how="left")
    merged["count"] = merged["count"].fillna(0)

    fig, ax = plt.subplots(figsize=(15, 8))
    world.plot(ax=ax, color="#EEEEEE", edgecolor="#CCCCCC", linewidth=0.3)
    merged[merged["count"] > 0].plot(
        column="count", ax=ax, cmap="YlOrRd",
        legend=True,
        legend_kwds={"label": "Number of Studies", "shrink": 0.5},
    )
    ax.set_title("Geographic Distribution of Studies", fontsize=13, fontweight="bold")
    ax.axis("off")
    _save(fig, out_dir / "geographic_map.png")


# =============================================================================
# 8. DOMAIN TYPE BAR
# =============================================================================

def _domain_type_bar(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    col = "domain_type_value"
    if col not in df.columns:
        return

    vals = df[col].dropna().astype(str).str.strip()
    vals = vals[vals.str.lower().isin(["", "nan", "not_found"]) == False]
    if vals.empty:
        return

    counts = vals.value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [BLUE, GREEN, ORANGE][:len(counts)]
    bars = ax.bar(counts.index, counts.values,
                  color=colors, edgecolor="white", linewidth=0.5, width=0.5)
    total = counts.sum()
    for bar, val in zip(bars, counts.values):
        pct = val / total * 100 if total else 0
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + total * 0.01,
                f"{val:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Number of Studies", fontsize=11)
    ax.set_title(f"Adaptation Domain Type  (n={len(vals):,})", fontsize=12, fontweight="bold")
    ax.set_ylim(0, counts.max() * 1.2)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, out_dir / "domain_type_bar.png")


# =============================================================================
# 9. EQUITY & INCLUSION BAR
# =============================================================================

def _equity_bar(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    col = "equity_inclusion_value"
    if col not in df.columns:
        return

    items = [i for i in _split_multi(df[col])
             if i.lower() not in ("none_reported", "nan", "")]
    if not items:
        return

    counts = pd.Series(items).value_counts()
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(counts.index, counts.values,
                  color=PURPLE, edgecolor="white", linewidth=0.5, width=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                str(val), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Number of Studies", fontsize=11)
    ax.set_title("Equity & Inclusion Dimensions Reported  (multi-select)",
                 fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, out_dir / "equity_bar.png")


# =============================================================================
# COMBINED DASHBOARD
# =============================================================================

def _dashboard(df: pd.DataFrame, out_dir: Path) -> None:
    """Single faceted figure combining all key panels for an at-a-glance snapshot."""
    if df.empty:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        return

    fig = plt.figure(figsize=(20, 13))
    fig.patch.set_facecolor("#FAFAFA")
    fig.suptitle(
        "Systematic Map — Evidence Snapshot\n"
        "Measuring What Matters: Climate Adaptation for Smallholder Producers",
        fontsize=13, fontweight="bold", y=0.98, color=DKGREY,
    )

    gs = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.65, wspace=0.38,
                           top=0.93, bottom=0.06, left=0.07, right=0.97)

    def _spine(ax):
        ax.spines[["top", "right"]].set_visible(False)

    n = len(df)
    note = f"n = {n:,} full-text coded studies"

    # ── Panel 1: Temporal trends ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    col = "publication_year_value"
    if col in df.columns:
        years = pd.to_numeric(df[col], errors="coerce").dropna().astype(int)
        years = years[(years >= 2005) & (years <= 2027)]
        if not years.empty:
            counts = years.value_counts().sort_index()
            bars = ax1.bar(counts.index, counts.values, color=BLUE, edgecolor="white", linewidth=0.4)
            for bar, val in zip(bars, counts.values):
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                         str(val), ha="center", va="bottom", fontsize=6.5)
            ax1.set_ylim(0, counts.max() * 1.18)
            ax1.set_xlabel("Year", fontsize=9)
            ax1.set_ylabel("Studies", fontsize=9)
    ax1.set_title(f"Publications per Year  ({note})", fontsize=10, fontweight="bold")
    _spine(ax1)

    # ── Panel 2: Domain type ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    col = "domain_type_value"
    if col in df.columns:
        vals = df[col].dropna().astype(str).str.strip()
        vals = vals[~vals.str.lower().isin(_SKIP)]
        if not vals.empty:
            counts = vals.value_counts()
            colors = [BLUE, GREEN, ORANGE][:len(counts)]
            bars = ax2.bar(range(len(counts)), counts.values,
                           color=colors, edgecolor="white", linewidth=0.4, width=0.5)
            ax2.set_xticks(range(len(counts)))
            ax2.set_xticklabels(
                [l.replace("_", "\n") for l in counts.index],
                fontsize=7.5
            )
            for bar, val in zip(bars, counts.values):
                ax2.text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + 0.2,
                         str(val), ha="center", va="bottom", fontsize=8)
    ax2.set_title("Domain Type", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Studies", fontsize=9)
    _spine(ax2)

    # ── Panel 3: Producer type ────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    col = "producer_type_value"
    if col in df.columns:
        items = _split_multi(df[col])
        if items:
            counts = pd.Series(items).value_counts().head(8)
            ax3.barh(counts.index[::-1], counts.values[::-1],
                     color=TEAL, edgecolor="white", linewidth=0.4)
            for i, val in enumerate(counts.values[::-1]):
                ax3.text(val + 0.1, i, str(val), va="center", fontsize=7.5)
    ax3.set_title("Producer Types", fontsize=10, fontweight="bold")
    ax3.set_xlabel("Studies", fontsize=9)
    _spine(ax3)

    # ── Panel 4: Methodology ──────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    col = "methodological_approach_value"
    if col in df.columns:
        vals = df[col].dropna().astype(str).str.strip()
        vals = vals[~vals.str.lower().isin(_SKIP)]
        if not vals.empty:
            counts = vals.value_counts().head(8)
            colors = PALETTE[:len(counts)]
            ax4.barh(counts.index[::-1], counts.values[::-1],
                     color=colors[::-1], edgecolor="white", linewidth=0.4)
            for i, val in enumerate(counts.values[::-1]):
                ax4.text(val + 0.1, i, str(val), va="center", fontsize=7.5)
    ax4.set_title("Methodology", fontsize=10, fontweight="bold")
    ax4.set_xlabel("Studies", fontsize=9)
    _spine(ax4)

    # ── Panel 5: Geography (top 25) ───────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    col = "country_region_value"
    if col in df.columns:
        items = _split_multi(df[col])
        if items:
            counts = pd.Series(items).value_counts().head(25)
            ax5.barh(counts.index[::-1], counts.values[::-1],
                     color=ORANGE, edgecolor="white", linewidth=0.4)
            for i, val in enumerate(counts.values[::-1]):
                ax5.text(val + 0.1, i, str(val), va="center", fontsize=7)
            ax5.tick_params(axis="y", labelsize=7)
    ax5.set_title("Top 25 Countries / Regions", fontsize=10, fontweight="bold")
    ax5.set_xlabel("Studies", fontsize=9)
    _spine(ax5)

    # ── Panel 6: Process/outcome domains (full width) ─────────────────────
    ax6 = fig.add_subplot(gs[2, :2])
    dom_col  = "process_outcome_domains_value"
    if dom_col in df.columns:
        items = _split_multi(df[dom_col])
        if items:
            counts = pd.Series(items).value_counts().head(14)
            colors = [GREEN if any(k in v.lower() for k in
                      ["yield", "income", "livelihood", "wellbeing", "risk", "resilience"])
                      else BLUE for v in counts.index]
            # Truncate long labels to keep left margin tight
            labels = [l if len(l) <= 42 else l[:40] + "…" for l in counts.index]
            ax6.barh(labels[::-1], counts.values[::-1],
                     color=colors[::-1], edgecolor="white", linewidth=0.4)
            for i, val in enumerate(counts.values[::-1]):
                ax6.text(val + 0.1, i, str(val), va="center", fontsize=7.5)
            ax6.tick_params(axis="y", labelsize=8)
    ax6.set_title("Process & Outcome Domains  (blue=process, green=outcome)",
                  fontsize=10, fontweight="bold")
    ax6.set_xlabel("Studies", fontsize=9)
    _spine(ax6)

    # ── Panel 7: Equity ───────────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 2])
    col = "equity_inclusion_value"
    if col in df.columns:
        items = [i for i in _split_multi(df[col])
                 if i.lower() not in ("none_reported", "nan", "")]
        if items:
            counts = pd.Series(items).value_counts()
            ax7.bar(range(len(counts)), counts.values,
                    color=PURPLE, edgecolor="white", linewidth=0.4, width=0.6)
            short = [l.replace("land_tenure", "land ten.").replace("_", " ")
                     .replace("indigenous peoples", "indigenous").replace("ethnic minorities", "ethnic min.")
                     for l in counts.index]
            ax7.set_xticks(range(len(counts)))
            ax7.set_xticklabels(short, fontsize=7, rotation=45, ha="right")
            for i, val in enumerate(counts.values):
                ax7.text(i, val + 0.2, str(val), ha="center", va="bottom", fontsize=8)
    ax7.set_title("Equity & Inclusion", fontsize=10, fontweight="bold")
    ax7.set_ylabel("Studies", fontsize=9)
    _spine(ax7)

    fig.text(0.5, 0.005, f"Generated: {_now_utc()}",
             ha="center", fontsize=7, color=GREY, style="italic")

    _save(fig, out_dir / "dashboard.png")


# =============================================================================
# Entrypoint
# =============================================================================

def run(config: dict) -> dict:
    out_root = Path(config.get("out_dir", "outputs"))
    out_dir  = _out_dir(out_root)

    print(f"[step16] Output dir: {out_dir}")
    t0 = time.time()

    figures_saved: List[str] = []

    # ROSES — always produced (reads from meta.json files, not coded data)
    print("[step16] Producing ROSES flow diagram...")
    _roses_flow(out_root, out_dir)
    figures_saved.append("roses_flow.png")

    # Load coded data for all other figures
    df = _load_coded(out_root)

    if df.empty:
        print("[step16] No full-text coded records yet — only ROSES diagram produced.")
        print("[step16] Re-run after step15 completes to produce all figures.")
    else:
        print("[step16] Producing evidence map figures...")
        for fn, label in [
            (_dashboard,         "dashboard.png"),
            (_temporal_trends,   "temporal_trends.png"),
            (_producer_type_bar, "producer_type_bar.png"),
            (_methodology_bar,   "methodology_bar.png"),
            (_domain_heatmap,    "domain_heatmap.png"),
            (_geographic_bar,    "geographic_bar.png"),
            (_geographic_map,    "geographic_map.png"),
            (_domain_type_bar,   "domain_type_bar.png"),
            (_equity_bar,        "equity_bar.png"),
        ]:
            try:
                fn(df, out_dir)
                figures_saved.append(label)
            except Exception as e:
                print(f"[step16] WARNING: {label} failed — {type(e).__name__}: {e}")

    elapsed = time.time() - t0
    meta = {
        "figures_saved":    figures_saved,
        "coded_records":    len(df),
        "elapsed_seconds":  round(elapsed, 1),
        "timestamp_utc":    _now_utc(),
    }
    meta_path = out_dir / "step16_figures.meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[step16] Meta -> {meta_path}")
    print(f"[step16] Done — {len(figures_saved)} figures in {elapsed:.1f}s")
    return meta


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    run({"out_dir": str(here / "outputs")})
