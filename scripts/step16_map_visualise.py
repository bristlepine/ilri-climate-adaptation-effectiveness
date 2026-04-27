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
# Evidence Gap Map constants
# Must match the actual coded field values in step15_coded.csv
# =============================================================================

# Process domains (coded values from process_outcome_domains_value)
PROCESS_DOMAINS: List[str] = [
    "knowledge_awareness_learning",
    "decision_making_planning",
    "uptake_adoption",
    "behavioral_change",
    "participation_coproduction",
    "institutional_governance",
    "access_information_services",
]

# Outcome domains
OUTCOME_DOMAINS: List[str] = [
    "yields_productivity",
    "income_assets",
    "livelihoods",
    "wellbeing",
    "risk_reduction",
    "resilience_adaptive_capacity",
]

# Process first, then outcome (matches display order: process on top)
ALL_DOMAINS: List[str] = PROCESS_DOMAINS + OUTCOME_DOMAINS
N_PROCESS = len(PROCESS_DOMAINS)   # 7

# Producer types (coded values from producer_type_value)
PRODUCER_TYPES: List[str] = [
    "crop",
    "livestock",
    "fisheries_aquaculture",
    "agroforestry",
    "mixed",
]

# Human-readable labels for display
DOMAIN_LABELS: Dict[str, str] = {
    "knowledge_awareness_learning":  "Knowledge & Awareness",
    "decision_making_planning":      "Decision-making & Planning",
    "uptake_adoption":               "Uptake & Adoption",
    "behavioral_change":             "Behavioral Change",
    "participation_coproduction":    "Participation & Co-production",
    "institutional_governance":      "Institutional Governance",
    "access_information_services":   "Access to Services",
    "yields_productivity":           "Yields & Productivity",
    "income_assets":                 "Income & Assets",
    "livelihoods":                   "Livelihoods",
    "wellbeing":                     "Wellbeing",
    "risk_reduction":                "Risk Reduction",
    "resilience_adaptive_capacity":  "Resilience & Adaptive Capacity",
}

PRODUCER_LABELS: Dict[str, str] = {
    "crop":                  "Crop farmers",
    "livestock":             "Livestock",
    "fisheries_aquaculture": "Fisheries/Aquaculture",
    "agroforestry":          "Agroforestry",
    "mixed":                 "Mixed/General",
}

# Frontend data directory
_HERE         = Path(__file__).resolve().parent
_FRONTEND_MAP = _HERE.parent / "frontend" / "public" / "map"


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
    # Use parse_ok=True as the authoritative indicator that the LLM processed the record.
    # Records not yet reached by step15 have parse_ok=NaN.
    if "parse_ok" in df.columns:
        coded = df[df["parse_ok"].astype(str).str.lower() == "true"].copy()
    else:
        # Fallback if parse_ok column absent: non-empty publication_year_value
        yr_col = "publication_year_value"
        if yr_col in df.columns:
            mask = df[yr_col].astype(str).str.strip().isin(["", "nan", "not_found"])
            coded = df[~mask].copy()
        else:
            coded = df.copy()
    print(f"[step16] LLM-coded records for figures: {len(coded):,} / {len(df):,} total in CSV")
    return coded


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


def _save_plotly(fig: Any, name: str, out_dir: Path) -> None:
    """Save a Plotly figure as JSON for the frontend interactive viewer."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return
    try:
        int_dir = out_dir / "interactive"
        int_dir.mkdir(exist_ok=True)
        fig.write_json(str(int_dir / f"{name}.json"))
        print(f"[step16] Interactive JSON -> interactive/{name}.json")
    except Exception as e:
        print(f"[step16] WARNING: Could not save interactive {name} — {e}")


def _save_csv(data: "pd.DataFrame", name: str, out_dir: Path) -> None:
    """Save a small CSV alongside the interactive JSON for download."""
    try:
        int_dir = out_dir / "interactive"
        int_dir.mkdir(exist_ok=True)
        data.to_csv(int_dir / f"{name}.csv", index=False)
    except Exception as e:
        print(f"[step16] WARNING: Could not save CSV {name} — {e}")


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
    # Use rows_llm_coded (actual LLM output present) if available; fall back to
    # coding_source_counts.full_text only as a last resort so ROSES stays honest.
    n_coded_ft      = s15.get("rows_llm_coded") or src15.get("full_text", 0)
    n_ft_designated = src15.get("full_text", 0)          # total assigned to full-text
    n_pending       = n_ft_designated - n_coded_ft        # not yet coded

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

_METHOD_LABELS: Dict[str, str] = {
    "quantitative":                       "Quantitative",
    "qualitative":                        "Qualitative",
    "participatory":                      "Participatory",
    "participatory_methods":              "Participatory",
    "modeling_with_empirical_validation": "Modelling",
    "experimental":                       "Experimental / RCT",
    "secondary_data":                     "Secondary data",
    "remote_sensing":                     "Remote sensing / GIS",
    "surveys":                            "Survey / Questionnaire",
    "longitudinal":                       "Longitudinal / Panel",
    "mixed-methods":                      "Mixed methods",
    "other":                              "Other",
}


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

    # Expand semicolon-separated multi-values, then apply label map
    items = _split_multi(df[col])
    if not items:
        return

    raw_counts = pd.Series(items).value_counts()
    # Merge synonyms via label map
    merged: Dict[str, int] = {}
    for raw, n in raw_counts.items():
        label = _METHOD_LABELS.get(raw.lower().strip(), raw.replace("_", " ").title())
        merged[label] = merged.get(label, 0) + n
    counts = pd.Series(merged).sort_values(ascending=False).head(12)

    fig, ax = plt.subplots(figsize=(10, max(4, len(counts) * 0.52)))
    colors = PALETTE[:len(counts)]
    bars = ax.barh(counts.index[::-1], counts.values[::-1],
                   color=colors[::-1], edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, counts.values[::-1]):
        ax.text(bar.get_width() + counts.max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=9, color=DKGREY)
    ax.set_xlabel("Number of Studies", fontsize=11)
    ax.set_title(f"Methodological Approach  (n={len(items):,} method tags, multi-select)",
                 fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=10)
    ax.margins(x=0.14)
    fig.tight_layout()
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

    TOP_N = 25
    MIN_COUNT = 2
    counts = pd.Series(items).value_counts()
    # Keep top N with at least MIN_COUNT studies
    counts = counts[counts >= MIN_COUNT].head(TOP_N)
    if counts.empty:
        return

    fig_h = max(5, len(counts) * 0.38)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    colors = [BLUE if i < 10 else TEAL for i in range(len(counts))]
    bars = ax.barh(counts.index[::-1], counts.values[::-1],
                   color=colors[::-1], edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, counts.values[::-1]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=9, color=DKGREY)
    ax.set_xlabel("Number of Studies", fontsize=11)
    ax.set_title(
        f"Top Countries by Study Count  (n={len(df):,} coded records, top {len(counts)} shown)",
        fontsize=12, fontweight="bold",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=10)
    ax.margins(x=0.12)
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
        _NE_ZIP = _HERE / "data" / "naturalearth" / "ne_110m_admin_0_countries.zip"
        world = gpd.read_file(str(_NE_ZIP))
    except Exception:
        print("[step16] Could not load naturalearth dataset — skipping choropleth")
        return

    name_col = "NAME" if "NAME" in world.columns else "name"
    merged = world.merge(counts, left_on=name_col, right_on="country", how="left")
    merged["count"] = merged["count"].fillna(0)

    # Zoom to cover Africa, Asia, and all of the Americas
    XLIM = (-125, 155)
    YLIM = (-57, 60)

    fig, ax = plt.subplots(figsize=(14, 7))
    # Base layer: all countries in light grey
    world.plot(ax=ax, color="#EEEEEE", edgecolor="none")
    # Fill layer: countries with studies (no edge — borders drawn on top)
    merged[merged["count"] > 0].plot(
        column="count", ax=ax, cmap="YlOrRd",
        legend=True,
        legend_kwds={"label": "Number of Studies", "shrink": 0.45, "pad": 0.01},
    )
    # Border layer: draw all country borders on top of fill
    world.plot(ax=ax, color="none", edgecolor="#AAAAAA", linewidth=0.4)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_title(
        f"Geographic Distribution of Studies  (n={len(df):,} coded records)",
        fontsize=13, fontweight="bold",
    )
    ax.axis("off")
    fig.tight_layout()
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

    col = "process_outcome_domains_value"
    if col not in df.columns:
        return

    items = _split_multi(df[col])
    if not items:
        return

    counts = pd.Series(items).value_counts()
    # Shorten raw coded values to readable labels
    DOMAIN_TYPE_LABELS = {
        "process":          "Process domain",
        "outcome":          "Outcome domain",
        "both":             "Both",
        "process_outcome":  "Both",
    }
    labels = [DOMAIN_TYPE_LABELS.get(k.lower().strip(), k.replace("_", " ").title())
              for k in counts.index]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [BLUE, GREEN, ORANGE][:len(counts)]
    bars = ax.bar(labels, counts.values,
                  color=colors, edgecolor="white", linewidth=0.5, width=0.5)
    total = counts.sum()
    for bar, val in zip(bars, counts.values):
        pct = val / total * 100 if total else 0
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + counts.max() * 0.02,
                f"{val:,}  ({pct:.1f}%)", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Number of Studies", fontsize=11)
    ax.set_title(f"Process vs. Outcome Domains  (n={len(df):,} coded records)",
                 fontsize=12, fontweight="bold")
    # Extra headroom so annotations don't clip
    ax.set_ylim(0, counts.max() * 1.28)
    ax.tick_params(axis="x", labelsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
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

    # Column was renamed in the codebook; try both names
    col = next((c for c in ("marginalized_subpopulations_value", "equity_inclusion_value")
                if c in df.columns), None)
    if col is None:
        return

    items = [i for i in _split_multi(df[col])
             if i.lower() not in ("none_reported", "nan", "", "other")]
    if not items:
        return

    import textwrap as _tw
    counts = pd.Series(items).value_counts().head(12)
    wrapped = ["\n".join(_tw.wrap(k.replace("_", " ").title(), width=16))
               for k in counts.index]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(counts)), counts.values,
                  color=PURPLE, edgecolor="white", linewidth=0.5, width=0.6)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + counts.max() * 0.02,
                str(val), ha="center", va="bottom", fontsize=10)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(wrapped, fontsize=9, ha="center")
    ax.set_ylabel("Number of Studies", fontsize=11)
    ax.set_title("Marginalised Subpopulations Reported  (multi-select, top 12)",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, counts.max() * 1.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
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
# INTERACTIVE PLOTLY FIGURES
# =============================================================================

def _egm_interactive(df: pd.DataFrame, out_root: Path, out_dir: Path) -> None:
    """Interactive Evidence Gap Map (bubble chart) using Plotly."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return

    dom_col  = "process_outcome_domains_value"
    prod_col = "producer_type_value"
    if df.empty or dom_col not in df.columns or prod_col not in df.columns:
        return

    # Count studies per (domain, producer_type) cell
    cell_counts: Dict[Tuple[str, str], int] = {}
    n_contributing = 0
    for _, row in df.iterrows():
        domains   = [d.strip() for d in str(row.get(dom_col, "")).split(";")
                     if d.strip() and d.strip().lower() not in _SKIP]
        prodtypes = [p.strip() for p in str(row.get(prod_col, "")).split(";")
                     if p.strip() and p.strip().lower() not in _SKIP]
        contributed = False
        for d in domains:
            for p in prodtypes:
                cell_counts[(d, p)] = cell_counts.get((d, p), 0) + 1
                contributed = True
        if contributed:
            n_contributing += 1

    n_out   = len(ALL_DOMAINS) - N_PROCESS   # number of outcome domains
    total   = len(ALL_DOMAINS)

    traces = []

    # Use labels for display
    d_labels = [DOMAIN_LABELS.get(d, d) for d in ALL_DOMAINS]
    p_labels  = [PRODUCER_LABELS.get(p, p) for p in PRODUCER_TYPES]
    max_n     = max((cell_counts.get((d, p), 0) for d in ALL_DOMAINS for p in PRODUCER_TYPES), default=1)
    max_n     = max(max_n, 1)

    # Process domain bubbles (blue)
    proc_x, proc_y, proc_size, proc_text = [], [], [], []
    for d in PROCESS_DOMAINS:
        dl = DOMAIN_LABELS.get(d, d)
        for p in PRODUCER_TYPES:
            pl = PRODUCER_LABELS.get(p, p)
            n = cell_counts.get((d, p), 0)
            if n > 0:
                proc_x.append(pl)
                proc_y.append(dl)
                proc_size.append(n)
                proc_text.append(f"<b>{dl}</b><br>Producer: {pl}<br>Studies: {n}")
    if proc_x:
        traces.append(go.Scatter(
            x=proc_x, y=proc_y, mode="markers+text",
            name="Process domain",
            marker=dict(
                size=proc_size, sizemode="area",
                sizeref=max_n / (38 ** 2), sizemin=10,
                color=BLUE, opacity=0.85,
                line=dict(color="white", width=1.5),
            ),
            text=[str(s) for s in proc_size],
            textposition="middle center",
            textfont=dict(size=10, color="white"),
            hovertext=proc_text, hoverinfo="text",
            showlegend=False,
        ))

    # Outcome domain bubbles (green)
    out_x, out_y, out_size, out_text = [], [], [], []
    for d in OUTCOME_DOMAINS:
        dl = DOMAIN_LABELS.get(d, d)
        for p in PRODUCER_TYPES:
            pl = PRODUCER_LABELS.get(p, p)
            n = cell_counts.get((d, p), 0)
            if n > 0:
                out_x.append(pl)
                out_y.append(dl)
                out_size.append(n)
                out_text.append(f"<b>{dl}</b><br>Producer: {pl}<br>Studies: {n}")
    if out_x:
        traces.append(go.Scatter(
            x=out_x, y=out_y, mode="markers+text",
            name="Outcome domain",
            marker=dict(
                size=out_size, sizemode="area",
                sizeref=max_n / (38 ** 2), sizemin=10,
                color=GREEN, opacity=0.85,
                line=dict(color="white", width=1.5),
            ),
            text=[str(s) for s in out_size],
            textposition="middle center",
            textfont=dict(size=10, color="white"),
            hovertext=out_text, hoverinfo="text",
            showlegend=False,
        ))

    # Gap cells (light grey)
    gap_x, gap_y, gap_text = [], [], []
    for d in ALL_DOMAINS:
        dl = DOMAIN_LABELS.get(d, d)
        for p in PRODUCER_TYPES:
            pl = PRODUCER_LABELS.get(p, p)
            if cell_counts.get((d, p), 0) == 0:
                gap_x.append(pl)
                gap_y.append(dl)
                gap_text.append(f"<b>{dl}</b><br>Producer: {pl}<br><i>No studies — evidence gap</i>")
    if gap_x:
        traces.append(go.Scatter(
            x=gap_x, y=gap_y, mode="markers",
            name="Evidence gap",
            marker=dict(size=14, color="#E0E0E0", line=dict(color="#BDBDBD", width=1)),
            hovertext=gap_text, hoverinfo="text",
            showlegend=False,
        ))

    # Dedicated legend-only traces (fixed visible size)
    for cat, color in [("Process domain", BLUE), ("Outcome domain", GREEN)]:
        traces.append(go.Scatter(
            x=[None], y=[None], name=cat, mode="markers",
            marker=dict(size=14, color=color, opacity=0.85, line=dict(color="white", width=1.5)),
            showlegend=True,
        ))
    traces.append(go.Scatter(
        x=[None], y=[None], name="Evidence gap", mode="markers",
        marker=dict(size=10, color="#E0E0E0", line=dict(color="#BDBDBD", width=1)),
        showlegend=True,
    ))

    # Background shading: process=light blue (top), outcome=light green (bottom)
    # With autorange=reversed, index 0 (first process domain) is at top
    proc_labels = [DOMAIN_LABELS[d] for d in PROCESS_DOMAINS]
    out_labels  = [DOMAIN_LABELS[d] for d in OUTCOME_DOMAINS]

    fig = go.Figure(data=traces)
    fig.add_shape(type="rect",
        xref="paper", yref="y",
        x0=0, x1=1,
        y0=out_labels[0], y1=proc_labels[-1],   # outcome band (bottom)
        fillcolor="#E8F5E9", opacity=0.3, line_width=0, layer="below",
    )
    fig.add_shape(type="rect",
        xref="paper", yref="y",
        x0=0, x1=1,
        y0=proc_labels[0], y1=out_labels[-1],   # process band (top)
        fillcolor="#E3F2FD", opacity=0.3, line_width=0, layer="below",
    )
    # Separator line between process and outcome sections.
    # categoryarray = list(reversed(d_labels)) puts outcome domains at
    # the low indices (0 … N_OUTCOME-1) and process domains at the high
    # indices (N_OUTCOME … N_TOTAL-1).  The gap between the first outcome
    # domain (index N_OUTCOME-1 = 5) and the last process domain
    # (index N_OUTCOME = 6) is at exactly 5.5.
    _n_out = len(OUTCOME_DOMAINS)
    fig.add_shape(type="line",
        xref="paper", yref="y",
        x0=0, x1=1,
        y0=_n_out - 0.5, y1=_n_out - 0.5,
        line=dict(color="#9E9E9E", width=1.5, dash="dot"),
    )

    fig.update_layout(
        title=dict(
            text=f"<b>Evidence Gap Map</b><br><sup>Bubble size = number of studies · process domains (top) vs outcome domains (bottom) (n={n_contributing:,})</sup>",
            x=0.5, xanchor="center", font=dict(size=14),
        ),
        height=700,
        xaxis=dict(
            title=dict(text="Producer Type", standoff=20),
            side="top",
            categoryorder="array", categoryarray=p_labels,
            showgrid=True, gridcolor="#F0F0F0",
            tickfont=dict(size=12, color=DKGREY),
        ),
        yaxis=dict(
            categoryorder="array",
            # Process domains on top, outcome on bottom — reverse the full list
            categoryarray=list(reversed(d_labels)),
            showgrid=True, gridcolor="#F0F0F0",
            tickfont=dict(size=11, color=DKGREY),
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Lato, Arial, sans-serif", color=DKGREY),
        legend=dict(orientation="h", yanchor="bottom", y=-0.14, xanchor="center", x=0.5),
        hovermode="closest",
        margin=dict(l=210, r=80, t=130, b=80),
    )

    # Side annotations for domain groups
    fig.add_annotation(
        x=1.02, xref="paper",
        y=proc_labels[N_PROCESS // 2], yref="y",
        text="<b>PROCESS<br>DOMAINS</b>",
        font=dict(size=10, color=BLUE),
        showarrow=False, xanchor="left", yanchor="middle",
        textangle=90,
    )
    fig.add_annotation(
        x=1.02, xref="paper",
        y=out_labels[len(OUTCOME_DOMAINS) // 2], yref="y",
        text="<b>OUTCOME<br>DOMAINS</b>",
        font=dict(size=10, color=GREEN),
        showarrow=False, xanchor="left", yanchor="middle",
        textangle=90,
    )

    _save_plotly(fig, "evidence_gap_map", out_dir)

    # Export as high-res PNG (requires kaleido)
    try:
        png_path = out_dir / "evidence_gap_map.png"
        fig.write_image(str(png_path), scale=2, width=1100, height=780)
        print(f"[step16] Saved -> evidence_gap_map.png (Plotly/kaleido)")
    except Exception as _e:
        print(f"[step16] kaleido PNG export failed for EGM ({_e})")

    # CSV export
    egm_rows = []
    for d in ALL_DOMAINS:
        for p in PRODUCER_TYPES:
            n = cell_counts.get((d, p), 0)
            egm_rows.append({
                "domain_code": d,
                "domain_label": DOMAIN_LABELS.get(d, d),
                "producer_code": p,
                "producer_label": PRODUCER_LABELS.get(p, p),
                "n_studies": n,
            })
    egm_df = pd.DataFrame(egm_rows)
    _save_csv(egm_df, "evidence_gap_map", out_dir)


def _geographic_interactive(df: pd.DataFrame, out_dir: Path) -> None:
    """Geographic choropleth (primary) + bar chart (secondary), both as interactive Plotly JSON."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return

    col = "country_region_value"
    if df.empty or col not in df.columns:
        return

    # Skip regional/global labels that can't be mapped
    SKIP_GEO = {
        "global", "sub-saharan africa", "east africa", "west africa",
        "central africa", "southern africa", "north africa",
        "south asia", "south-east asia", "southeast asia", "latin america",
        "caribbean", "lmics", "low-income countries", "not_reported", "",
    }
    country_counts: Dict[str, int] = {}
    for val in df[col].dropna().astype(str):
        for c in val.split(";"):
            c = c.strip()
            if c and c.lower() not in SKIP_GEO:
                country_counts[c] = country_counts.get(c, 0) + 1

    if not country_counts:
        return

    # Shared CSV (used by both views)
    geo_df = (pd.DataFrame(list(country_counts.items()), columns=["country", "n_studies"])
              .sort_values("n_studies", ascending=False))
    _save_csv(geo_df, "geographic_map", out_dir)

    # ── Choropleth (default view) ────────────────────────────────────────────
    fig_map = go.Figure(go.Choropleth(
        locations=list(country_counts.keys()),
        z=list(country_counts.values()),
        locationmode="country names",
        colorscale="YlOrRd",
        colorbar_title="Studies",
        hovertemplate="<b>%{location}</b><br>Studies: %{z}<extra></extra>",
    ))
    fig_map.update_layout(
        title=dict(
            text=f"<b>Geographic Distribution of Studies</b>"
                 f"<br><sup>Hover for country-level detail · multi-country studies counted in each (n={len(df):,})</sup>",
            x=0.5, xanchor="center", font=dict(size=14),
        ),
        geo=dict(
            showframe=False, showcoastlines=True,
            projection_type="natural earth",
            showland=True, landcolor="#F5F5F5",
            showocean=True, oceancolor="#EBF4FB",
            showcountries=True, countrycolor="#D0D0D0",
        ),
        height=480,
        paper_bgcolor="white",
        font=dict(family="Lato, Arial, sans-serif", size=11, color=DKGREY),
        margin=dict(l=0, r=0, t=90, b=0),
    )
    _save_plotly(fig_map, "geographic_map", out_dir)

    # ── Bar chart (secondary view) ───────────────────────────────────────────
    sorted_counts = sorted(country_counts.items(), key=lambda x: -x[1])
    countries = [c for c, _ in sorted_counts]
    values    = [v for _, v in sorted_counts]

    fig_bar = go.Figure(go.Bar(
        x=values, y=countries, orientation="h",
        marker_color=BLUE,
        hovertemplate="<b>%{y}</b><br>Studies: %{x}<extra></extra>",
        text=values, textposition="outside",
    ))
    fig_bar.update_layout(
        title=dict(
            text=f"<b>Studies by Country / Region</b><br><sup>Multi-country studies counted once per country (n={len(df):,})</sup>",
            x=0.5, xanchor="center", font=dict(size=14),
        ),
        height=max(400, len(countries) * 22 + 120),
        xaxis_title="Number of Studies",
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Lato, Arial, sans-serif", size=11, color=DKGREY),
        margin=dict(l=160, r=60, t=80, b=40),
        showlegend=False,
    )
    _save_plotly(fig_bar, "geographic_bar", out_dir)


def _temporal_interactive(df: pd.DataFrame, out_dir: Path) -> None:
    """Interactive temporal trends chart using Plotly."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return

    col = "publication_year_value"
    if df.empty or col not in df.columns:
        return

    years = pd.to_numeric(df[col], errors="coerce").dropna().astype(int)
    years = years[(years >= 2005) & (years <= 2027)]
    if years.empty:
        return

    counts = years.value_counts().sort_index()

    fig = go.Figure(go.Bar(
        x=counts.index.tolist(),
        y=counts.values.tolist(),
        marker_color=BLUE,
        hovertemplate="<b>%{x}</b><br>Studies: %{y}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text=f"<b>Publications Over Time</b><br><sup>Included studies by publication year (n={len(years):,})</sup>",
            x=0.5, xanchor="center", font=dict(size=14),
        ),
        height=380,
        xaxis_title="Publication Year",
        yaxis_title="Number of Studies",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Lato, Arial, sans-serif", size=11, color=DKGREY),
        margin=dict(l=60, r=40, t=100, b=60),
    )
    temp_df = pd.DataFrame({"year": counts.index.tolist(), "n_studies": counts.values.tolist()})
    _save_csv(temp_df, "temporal_trends", out_dir)
    _save_plotly(fig, "temporal_trends", out_dir)


def _producer_interactive(df: pd.DataFrame, out_dir: Path) -> None:
    """Interactive producer type bar chart using Plotly."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return

    col = "producer_type_value"
    if df.empty or col not in df.columns:
        return

    items = _split_multi(df[col])
    if not items:
        return

    counts = pd.Series(items).value_counts()

    fig = go.Figure(go.Bar(
        x=counts.values.tolist(),
        y=counts.index.tolist(),
        orientation="h",
        marker_color=TEAL,
        hovertemplate="<b>%{y}</b><br>Studies: %{x}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text=f"<b>Producer Types Studied</b><br><sup>Multi-select: studies may represent multiple producer types (n={len(df):,})</sup>",
            x=0.5, xanchor="center", font=dict(size=14),
        ),
        height=380,
        xaxis_title="Number of Studies",
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Lato, Arial, sans-serif", size=11, color=DKGREY),
        margin=dict(l=180, r=40, t=100, b=60),
    )
    prod_df = pd.DataFrame({"producer_type": counts.index.tolist(), "n_studies": counts.values.tolist()})
    _save_csv(prod_df, "producer_type", out_dir)
    _save_plotly(fig, "producer_type", out_dir)


def _methodology_interactive(df: pd.DataFrame, out_dir: Path) -> None:
    """Interactive methodology bar chart using Plotly."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return

    col = "methodological_approach_value"
    if df.empty or col not in df.columns:
        return

    vals = df[col].dropna().astype(str).str.strip()
    vals = vals[vals.str.lower().isin(["", "nan", "not_found"]) == False]
    if vals.empty:
        return

    counts = vals.value_counts()

    fig = go.Figure(go.Bar(
        x=counts.values.tolist(),
        y=counts.index.tolist(),
        orientation="h",
        marker_color=PALETTE[:len(counts)],
        hovertemplate="<b>%{y}</b><br>Studies: %{x}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text=f"<b>Methodological Approach</b><br><sup>Primary study design, multi-select (n={len(vals):,})</sup>",
            x=0.5, xanchor="center", font=dict(size=14),
        ),
        height=380,
        xaxis_title="Number of Studies",
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Lato, Arial, sans-serif", size=11, color=DKGREY),
        margin=dict(l=200, r=40, t=100, b=60),
    )
    meth_df = pd.DataFrame({"methodology": counts.index.tolist(), "n_studies": counts.values.tolist()})
    _save_csv(meth_df, "methodology", out_dir)
    _save_plotly(fig, "methodology", out_dir)


def _domain_type_interactive(df: pd.DataFrame, out_dir: Path) -> None:
    """Interactive process/outcome domain bar chart using Plotly."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return

    col = "process_outcome_domains_value"
    if df.empty or col not in df.columns:
        return

    items = _split_multi(df[col])
    if not items:
        return

    counts = pd.Series(items).value_counts()
    bar_colors = [BLUE if i % 2 == 0 else GREEN for i in range(len(counts))]

    fig = go.Figure(go.Bar(
        x=counts.index.tolist(),
        y=counts.values.tolist(),
        marker_color=bar_colors,
        hovertemplate="<b>%{x}</b><br>Studies: %{y}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text=f"<b>Process/Outcome Domains</b><br><sup>Multi-select: studies may span multiple domains (n={len(df):,})</sup>",
            x=0.5, xanchor="center", font=dict(size=14),
        ),
        height=420,
        yaxis_title="Number of Studies",
        xaxis=dict(tickangle=-35),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Lato, Arial, sans-serif", size=11, color=DKGREY),
        margin=dict(l=60, r=40, t=110, b=120),
    )
    dtype_df = pd.DataFrame({"domain": counts.index.tolist(), "n_studies": counts.values.tolist()})
    _save_csv(dtype_df, "domain_type", out_dir)
    _save_plotly(fig, "domain_type", out_dir)


def _equity_interactive(df: pd.DataFrame, out_dir: Path) -> None:
    """Interactive equity bar chart using Plotly."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return

    col = "marginalized_subpopulations_value"
    if df.empty or col not in df.columns:
        return

    items = [i for i in _split_multi(df[col])
             if i.lower() not in ("none_reported", "nan", "")]
    if not items:
        return

    counts = pd.Series(items).value_counts()

    fig = go.Figure(go.Bar(
        x=counts.index.tolist(),
        y=counts.values.tolist(),
        marker_color=PURPLE,
        hovertemplate="<b>%{x}</b><br>Studies: %{y}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text=f"<b>Equity & Inclusion Dimensions Reported</b><br><sup>Multi-select: studies may address multiple dimensions (n={len(df):,})</sup>",
            x=0.5, xanchor="center", font=dict(size=14),
        ),
        height=380,
        yaxis_title="Number of Studies",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Lato, Arial, sans-serif", size=11, color=DKGREY),
        margin=dict(l=60, r=40, t=100, b=80),
    )
    eq_df = pd.DataFrame({"equity_dimension": counts.index.tolist(), "n_studies": counts.values.tolist()})
    _save_csv(eq_df, "equity", out_dir)
    _save_plotly(fig, "equity", out_dir)


def _domain_heatmap_interactive(df: pd.DataFrame, out_dir: Path) -> None:
    """Interactive domain heatmap using Plotly."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return

    dom_col  = "process_outcome_domains_value"
    prod_col = "producer_type_value"
    if df.empty or dom_col not in df.columns or prod_col not in df.columns:
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

    fig = go.Figure(go.Heatmap(
        z=ct.values.tolist(),
        x=ct.columns.tolist(),
        y=ct.index.tolist(),
        colorscale="YlOrRd",
        hovertemplate="<b>%{y}</b><br>%{x}<br>Studies: %{z}<extra></extra>",
        text=ct.values.tolist(),
        texttemplate="%{text}",
    ))
    fig.update_layout(
        title=dict(
            text=f"<b>Process/Outcome Domains × Producer Type</b><br><sup>Number of studies per cell (n={len(df):,})</sup>",
            x=0.5, xanchor="center", font=dict(size=14),
        ),
        height=max(400, ct.shape[0] * 28 + 150),
        xaxis=dict(tickangle=-30, side="top"),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Lato, Arial, sans-serif", size=10, color=DKGREY),
        margin=dict(l=240, r=40, t=130, b=60),
    )
    _save_plotly(fig, "domain_heatmap", out_dir)

    # Tidy CSV
    heat_rows = []
    for d in ct.index:
        for p in ct.columns:
            n = int(ct.loc[d, p])
            if n > 0:
                heat_rows.append({"domain": d, "producer_type": p, "n_studies": n})
    heat_df = pd.DataFrame(heat_rows)
    _save_csv(heat_df, "domain_heatmap", out_dir)


def _roses_interactive(out_root: Path, out_dir: Path) -> None:
    """Interactive Plotly ROSES flow diagram using Sankey."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return

    s12  = _load_json(_step12_meta(out_root))
    s12b = _load_json(out_root / "step12b" / "step12b_results.meta.json")
    s2b  = _load_json(out_root / "step2b" / "step2b_summary.json")
    s13  = _load_json(_step13_meta(out_root))
    s14  = _load_json(_step14_meta(out_root))
    s15  = _load_json(_step15_meta(out_root))

    # Scopus numbers
    n_scopus       = s12.get("rows_total", 0)
    dec12          = s12.get("decision_counts", {})
    n_sc_include   = dec12.get("Include", 0)
    n_sc_exclude   = dec12.get("Exclude", 0)
    excl12         = s12.get("excluded_by_criterion", {})

    # Other databases — all non-Scopus sources combined (WoS, CAB, ASP, AGRIS, …)
    by_db_imported  = s2b.get("by_database_imported", {})
    by_db_net_new   = s2b.get("by_database_net_new", {})
    n_total_dup     = s2b.get("total_duplicates", 0)
    n_other_net     = s2b.get("total_net_new", 0)

    dec12b          = s12b.get("decision_counts", {})
    n_other_include = dec12b.get("Include", 0)
    n_other_exclude = dec12b.get("Exclude", 0)
    n_other_manual  = dec12b.get("Needs_Manual", 0)

    # Full-text retrieval — use per-database breakdown so Scopus-specific numbers
    # are correct. Scopus records are keyed as "unknown" in step13 by_database.
    by_db13        = s13.get("by_database", {})
    scopus13       = by_db13.get("unknown", s13.get("status_counts", {}))
    n_ft_retrieved = scopus13.get("retrieved", 0)   # Scopus-only FT retrieved
    n_ft_not_found = scopus13.get("needs_manual", 0) # Scopus-only not found

    dec14          = s14.get("decision_counts", {})
    n_ft_include   = dec14.get("Include", 0)
    n_ft_exclude   = dec14.get("Exclude", 0)
    excl14         = s14.get("excluded_by_criterion", {})

    src15          = s15.get("coding_source_counts", {})
    n_coded_ft     = s15.get("rows_llm_coded") or src15.get("full_text", 0)

    # n_ft_retrieved is Scopus-only; use all-database total as denominator
    # for subsequent nodes (step14/15 processed all databases jointly).
    status13_all      = s13.get("status_counts", {})
    n_ft_retrieved_all = status13_all.get("retrieved", n_ft_retrieved)

    n_ft_screened       = n_ft_include + n_ft_exclude
    n_ft_pending_screen = max(0, n_ft_retrieved_all - n_ft_screened)

    excl12_text  = "; ".join(f"{k}: {v}" for k, v in sorted(excl12.items(),  key=lambda x: -x[1])[:3])
    excl14_text  = "; ".join(f"{k}: {v}" for k, v in sorted(excl14.items(),  key=lambda x: -x[1])[:3])

    # Helper for percentage labels
    _pct = lambda n, d: f"{n/d*100:.0f}%" if d > 0 else "—"

    # Per-database imported/net-new counts (ordered: WoS, CAB, ASP, AGRIS, others)
    _DB_ORDER = ["Web of Science", "CAB Abstracts", "Academic Search Premier", "AGRIS"]
    _db_imp = {db: by_db_imported.get(db, 0) for db in _DB_ORDER}
    _db_net = {db: by_db_net_new.get(db, 0)  for db in _DB_ORDER}
    # Include any extra databases not in the hard-coded order
    for db in by_db_imported:
        if db not in _db_imp:
            _db_imp[db] = by_db_imported[db]
            _db_net[db] = by_db_net_new.get(db, 0)
    _dbs = [db for db in _db_imp if _db_imp[db] > 0]

    # Short display names
    _DB_SHORT = {
        "Web of Science": "WoS",
        "CAB Abstracts":  "CAB",
        "Academic Search Premier": "ASP",
        "AGRIS": "AGRIS",
    }

    # ── Derived all-database totals ──────────────────────────────────────────
    n_abs_inc        = n_sc_include + n_other_include          # 8,555
    n_abs_exc        = n_sc_exclude + n_other_exclude + n_other_manual  # 16,653
    n_ft_not_ret_all = max(0, n_abs_inc - n_ft_retrieved_all)  # 5,079
    n_total_id       = n_scopus + sum(_db_imp[db] for db in _dbs)  # 39,113
    n_screened       = n_abs_inc + n_abs_exc                   # 25,208

    # ── Nodes ────────────────────────────────────────────────────────────────
    # All databases feed into a shared dedup pool, then a single pipeline:
    #   0          Scopus identified
    #   1…N        One node per other database (N = len(_dbs))
    #   N+1        Duplicates removed [TERMINAL]
    #   N+2        Abstract screening (unique records pool)
    #   N+3        Abstract included
    #   N+4        Abstract excluded [TERMINAL]
    #   N+5        Full texts retrieved
    #   N+6        Not retrievable [TERMINAL]
    #   N+7        Included for coding
    #   N+8        Full texts excluded [TERMINAL]
    #   N+9        Pending: FT screening [TERMINAL]  (only if > 0)

    N       = len(_dbs)
    I_DUP   = N + 1
    I_SCR   = N + 2   # abstract screening / unique records
    I_INC   = N + 3
    I_EXC   = N + 4
    I_FTRET = N + 5
    I_FTNR  = N + 6
    I_CODED = N + 7
    I_FTEXC = N + 8
    I_PEND  = N + 9

    labels = (
        [f"Scopus<br>identified<br>n={n_scopus:,} ({_pct(n_scopus, n_total_id)})"]
        + [f"{_DB_SHORT.get(db, db)}<br>identified<br>n={_db_imp[db]:,} ({_pct(_db_imp[db], n_total_id)})" for db in _dbs]
        + [
            f"Duplicates removed<br>n={n_total_dup:,} ({_pct(n_total_dup, n_total_id)})",
            f"Abstract screening<br>n={n_screened:,} ({_pct(n_screened, n_total_id)})",
            f"Abstract included<br>n={n_abs_inc:,} ({_pct(n_abs_inc, n_screened)})",
            f"Abstract excluded<br>n={n_abs_exc:,} ({_pct(n_abs_exc, n_screened)})",
            f"Full texts retrieved<br>n={n_ft_retrieved_all:,} ({_pct(n_ft_retrieved_all, n_abs_inc)})",
            f"Not retrievable<br>n={n_ft_not_ret_all:,} ({_pct(n_ft_not_ret_all, n_abs_inc)})",
            f"Included for coding<br>n={n_ft_include:,} ({_pct(n_ft_include, n_ft_retrieved_all)})",
            f"Full texts excluded<br>n={n_ft_exclude:,} ({_pct(n_ft_exclude, n_ft_retrieved_all)})",
            f"Pending: FT screening<br>n={n_ft_pending_screen:,} ({_pct(n_ft_pending_screen, n_ft_retrieved_all)})",
        ]
    )

    C_ENTRY   = "#4e79a7"   # steel blue  — identified / screening nodes
    C_EXC     = "#c0392b"   # crimson     — duplicates / abstract excluded
    C_INC     = "#2d8f4e"   # forest green — abstract/FT included
    C_INC_LT  = "#5ab56e"   # medium green — FT retrieved
    C_EXC_LT  = "#9b59b6"   # purple      — FT excluded
    C_INC_EXT = "#3cb371"   # sea green   — coded
    C_PEND    = "#6baed6"   # sky blue    — pending

    node_colors = (
        [C_ENTRY]         # 0   Scopus
        + [C_ENTRY] * N   # 1…N other databases
        + [
            C_EXC,        # N+1 Duplicates removed
            C_ENTRY,      # N+2 Abstract screening
            C_INC,        # N+3 Abstract included
            C_EXC,        # N+4 Abstract excluded
            C_INC_LT,     # N+5 FT retrieved
            C_PEND,       # N+6 Not retrievable
            C_INC_EXT,    # N+7 Included for coding
            C_EXC_LT,     # N+8 FT excluded
            C_PEND,       # N+9 Pending FT screening
        ]
    )

    sources, targets, values, link_colors, link_labels = [], [], [], [], []

    def add(s, t, v, c, lbl=""):
        if v > 0:
            sources.append(s); targets.append(t); values.append(v)
            link_colors.append(c); link_labels.append(lbl)

    # Scopus: all records are unique (it is the reference set — no cross-DB dups)
    add(0, I_SCR, n_scopus, "rgba(78,121,167,0.25)",
        f"Scopus: {n_scopus:,} identified (reference set, no cross-DB duplicates)")

    # Other databases: each splits into duplicates vs net-new
    for j, db in enumerate(_dbs):
        idx   = j + 1
        n_dup = _db_imp[db] - _db_net[db]
        n_net = _db_net[db]
        short = _DB_SHORT.get(db, db)
        add(idx, I_DUP, n_dup, "rgba(192,57,43,0.18)",
            f"{short}: {n_dup:,} duplicates removed")
        add(idx, I_SCR, n_net, "rgba(78,121,167,0.20)",
            f"{short}: {n_net:,} net-new unique records")

    # Shared screening pipeline
    add(I_SCR,   I_INC,   n_abs_inc,        "rgba(45,143,78,0.30)",
        f"Abstract included: {n_abs_inc:,}" + (f"<br>Top criteria: {excl12_text}" if excl12_text else ""))
    add(I_SCR,   I_EXC,   n_abs_exc,        "rgba(192,57,43,0.18)",
        f"Abstract excluded: {n_abs_exc:,}")
    add(I_INC,   I_FTRET, n_ft_retrieved_all, "rgba(90,181,110,0.30)",
        f"Full texts retrieved: {n_ft_retrieved_all:,}")
    add(I_INC,   I_FTNR,  n_ft_not_ret_all, "rgba(107,174,214,0.22)",
        f"Not retrievable: {n_ft_not_ret_all:,}")
    add(I_FTRET, I_CODED, n_ft_include,     "rgba(45,143,78,0.32)",
        f"Included for coding: {n_ft_include:,}" + (f"<br>Top criteria: {excl14_text}" if excl14_text else ""))
    add(I_FTRET, I_FTEXC, n_ft_exclude,     "rgba(155,89,182,0.22)",
        f"Full texts excluded: {n_ft_exclude:,}")
    add(I_FTRET, I_PEND,  n_ft_pending_screen, "rgba(107,174,214,0.20)",
        f"Pending FT screening: {n_ft_pending_screen:,}")

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=55, thickness=22,
            line=dict(color="white", width=1),
            label=labels,
            color=node_colors,
            hovertemplate="%{label}<extra></extra>",
        ),
        link=dict(
            source=sources, target=targets, value=values,
            color=link_colors, label=link_labels,
            hovertemplate="%{label}<extra></extra>",
        ),
    ))

    fig.update_layout(
        title=dict(
            text=(
                "<b>ROSES Flow Diagram — Scopus + WoS + CAB + ASP + AGRIS</b><br>"
                "<sup>Record flow across all screening stages · hover nodes and links for detail</sup>"
            ),
            x=0.5, xanchor="center", font=dict(size=14),
        ),
        height=540,
        font=dict(family="Lato, Arial, sans-serif", size=12, color=DKGREY),
        paper_bgcolor="white",
        margin=dict(l=30, r=30, t=90, b=30),
    )
    _save_plotly(fig, "roses_flow", out_dir)

    try:
        png_path = out_dir / "roses_flow.png"
        fig.write_image(str(png_path), scale=2, width=960, height=540)
        print(f"[step16] Saved -> roses_flow.png (Plotly/kaleido)")
    except Exception as _e:
        print(f"[step16] kaleido PNG export failed ({_e}) — static matplotlib PNG kept")

    _db_rows = [(f"{_DB_SHORT.get(db, db)} — identified", _db_imp[db]) for db in _dbs]
    roses_df = pd.DataFrame({
        "stage": (
            ["Scopus — identified"]
            + [r[0] for r in _db_rows]
            + ["Duplicates removed", "Abstract screening (unique records)",
               "Abstract included", "Abstract excluded",
               "Full texts retrieved", "Not retrievable",
               "Included for coding", "Full texts excluded",
               "Pending: FT screening"]
        ),
        "n": (
            [n_scopus]
            + [r[1] for r in _db_rows]
            + [n_total_dup, n_abs_inc + n_abs_exc,
               n_abs_inc, n_abs_exc,
               n_ft_retrieved_all, n_ft_not_ret_all,
               n_ft_include, n_ft_exclude,
               n_ft_pending_screen]
        ),
    })
    _save_csv(roses_df, "roses_flow", out_dir)


_BLANK = {"nan", "not_reported", "not_found", "none_reported", ""}

def _export_studies_json(df: pd.DataFrame, out_dir: Path) -> None:
    """Export a simplified studies JSON for the frontend searchable database.
    Only includes records that have been LLM-coded (publication_year_value non-empty).
    """
    KEEP = [
        ("doi",                              "doi"),
        ("title",                            "title"),
        ("publication_year_value",           "year"),
        ("publication_type_value",           "pub_type"),
        ("country_region_value",             "country"),
        ("geographic_scale_value",           "geo_scale"),
        ("producer_type_value",              "producer_type"),
        ("adaptation_focus_value",           "adaptation_focus"),
        ("process_outcome_domains_value",    "domain_type"),
        ("methodological_approach_value",    "methodology"),
        ("marginalized_subpopulations_value","equity"),
    ]

    rows = []
    for _, row in df.iterrows():
        rec = {}
        for src_col, dst_col in KEEP:
            val = str(row.get(src_col, "") or "").strip()
            if val.lower() in _BLANK:
                val = ""
            # Year fallback: use Scopus metadata 'year' when LLM extraction is missing
            if dst_col == "year" and not val:
                fallback = str(row.get("year", "") or "").strip()
                if fallback and fallback.lower() not in _BLANK:
                    val = fallback
            rec[dst_col] = val
        rows.append(rec)

    out_path = out_dir / "interactive" / "studies.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, separators=(",", ":"))
    print(f"[step16] Studies JSON -> interactive/studies.json ({len(rows)} records)")


def _export_studies_csv(df: pd.DataFrame, out_dir: Path) -> None:
    """Export simplified studies CSV for download."""
    KEEP = [
        ("doi",                              "doi"),
        ("title",                            "title"),
        ("publication_year_value",           "year"),
        ("country_region_value",             "country"),
        ("producer_type_value",              "producer_type"),
        ("adaptation_focus_value",           "adaptation_focus"),
        ("process_outcome_domains_value",    "domain_type"),
        ("methodological_approach_value",    "methodology"),
        ("marginalized_subpopulations_value","equity"),
    ]
    out_df = pd.DataFrame()
    for src_col, dst_col in KEEP:
        if src_col in df.columns:
            col_vals = df[src_col].fillna("").astype(str).str.strip()
            # Year fallback: use Scopus metadata 'year' where LLM extraction is missing
            if dst_col == "year" and "year" in df.columns:
                meta_year = df["year"].fillna("").astype(str).str.strip()
                col_vals = col_vals.where(
                    ~col_vals.str.lower().isin(_BLANK), meta_year
                )
            out_df[dst_col] = col_vals

    _save_csv(out_df, "studies", out_dir)
    print(f"[step16] Studies CSV -> interactive/studies.csv ({len(out_df)} records)")


def _generate_db_summary(out_root: Path, data_dir: Path) -> None:
    """Generate frontend/public/map/data/db_summary.json from pipeline metadata."""
    import json as _json

    s2b  = _load_json(out_root / "step2b"  / "step2b_summary.json")
    s12  = _load_json(_step12_meta(out_root))
    s12b = _load_json(out_root / "step12b" / "step12b_results.meta.json")
    s13  = _load_json(_step13_meta(out_root))
    s14  = _load_json(_step14_meta(out_root))
    s15  = _load_json(_step15_meta(out_root))

    by_imp = s2b.get("by_database_imported", {})
    by_net = s2b.get("by_database_net_new",  {})
    n_dup  = s2b.get("total_duplicates", 0)
    n_uniq = s2b.get("total_net_new", 0) + s12.get("rows_total", 0)

    dec12  = s12.get("decision_counts",  {})
    dec12b = s12b.get("decision_counts", {})
    n_sc_inc   = dec12.get("Include",  0)
    n_sc_exc   = dec12.get("Exclude",  0)
    n_oth_inc  = dec12b.get("Include", 0)
    n_oth_exc  = dec12b.get("Exclude", 0) + dec12b.get("Needs_Manual", 0)
    n_abs_inc  = n_sc_inc + n_oth_inc
    n_abs_exc  = n_sc_exc + n_oth_exc

    by_db13     = s13.get("by_database", {})
    n_ft_ret    = s13.get("status_counts", {}).get("retrieved", 0)
    n_ft_not    = n_abs_inc - n_ft_ret

    dec14       = s14.get("decision_counts", {})
    n_ft_inc    = dec14.get("Include", 0)
    n_ft_exc    = n_ft_ret - n_ft_inc

    _DB_ORDER = ["Web of Science", "CAB Abstracts", "Academic Search Premier", "AGRIS"]
    node_labels = ["Scopus<br>n={:,}".format(s12.get("rows_total", 0))]
    node_colors = ["#1f77b4"]
    db_colors   = ["#2ca02c", "#e377c2", "#ff7f0e", "#9467bd"]
    for db, col in zip(_DB_ORDER, db_colors):
        n = by_imp.get(db, 0)
        if n:
            node_labels.append(f"{db}<br>n={n:,}")
            node_colors.append(col)

    n_db_nodes = len(node_labels)
    IDX_DUP    = n_db_nodes
    IDX_UNIQ   = n_db_nodes + 1
    IDX_ABSINC = n_db_nodes + 2
    IDX_ABSEXC = n_db_nodes + 3
    IDX_FTRET  = n_db_nodes + 4
    IDX_FTNOT  = n_db_nodes + 5
    IDX_CODED  = n_db_nodes + 6
    IDX_FTEXC  = n_db_nodes + 7

    node_labels += [
        f"Duplicates removed<br>n={n_dup:,}",
        f"Unique records<br>n={n_uniq:,}",
        f"Abstract included<br>n={n_abs_inc:,}",
        f"Abstract excluded<br>n={n_abs_exc:,}",
        f"Full texts retrieved<br>n={n_ft_ret:,}",
        f"Not retrievable<br>n={n_ft_not:,}",
        f"Included for coding<br>n={n_ft_inc:,}",
        f"Full texts excluded<br>n={n_ft_exc:,}",
    ]
    node_colors += ["#d62728", "#17becf", "#2ca02c", "#aec7e8",
                    "#ff7f0e", "#c7c7c7", "#1f77b4", "#ffbb78"]

    sources, targets, values, link_colors = [], [], [], []

    def add(s, t, v, c):
        if v > 0:
            sources.append(s); targets.append(t); values.append(v); link_colors.append(c)

    # Scopus → unique (no dups)
    add(0, IDX_UNIQ, s12.get("rows_total", 0), "rgba(31,119,180,0.30)")
    # Other DBs → duplicates / net-new
    for i, (db, col) in enumerate(zip(_DB_ORDER, db_colors)):
        idx = i + 1
        if idx >= n_db_nodes:
            break
        rgba = col.replace("#", "")
        r, g, b = int(rgba[0:2],16), int(rgba[2:4],16), int(rgba[4:6],16)
        add(idx, IDX_DUP,  by_imp.get(db,0) - by_net.get(db,0), f"rgba({r},{g},{b},0.20)")
        add(idx, IDX_UNIQ, by_net.get(db, 0),                    f"rgba({r},{g},{b},0.30)")

    add(IDX_UNIQ, IDX_ABSINC, n_abs_inc, "rgba(44,160,44,0.30)")
    add(IDX_UNIQ, IDX_ABSEXC, n_abs_exc, "rgba(174,199,232,0.20)")
    add(IDX_ABSINC, IDX_FTRET, n_ft_ret, "rgba(255,127,14,0.30)")
    add(IDX_ABSINC, IDX_FTNOT, n_ft_not, "rgba(199,199,199,0.20)")
    add(IDX_FTRET,  IDX_CODED, n_ft_inc, "rgba(31,119,180,0.35)")
    add(IDX_FTRET,  IDX_FTEXC, n_ft_exc, "rgba(255,187,120,0.25)")

    summary = (
        f"{n_uniq + n_dup:,} records identified · {n_dup:,} duplicates removed · "
        f"{n_uniq:,} unique · {n_abs_inc:,} abstract included · "
        f"{n_ft_ret:,} full texts retrieved · {n_ft_inc:,} coded"
    )

    doc = {"data": [{"type": "sankey", "arrangement": "snap",
        "node": {"label": node_labels, "color": node_colors,
                 "pad": 20, "thickness": 20,
                 "line": {"color": "white", "width": 1},
                 "hovertemplate": "%{label}<extra></extra>"},
        "link": {"source": sources, "target": targets, "value": values,
                 "color": link_colors,
                 "hovertemplate": "%{value:,} records<extra></extra>"}}],
    "layout": {"title": {"text": "<b>Multi-Database Search & Deduplication</b><br>"
                                  "<sup>Five databases · hover nodes and links for counts</sup>",
                          "x": 0.5, "xanchor": "center", "font": {"size": 14}},
               "font": {"family": "Lato, Arial, sans-serif", "size": 11, "color": "#424242"},
               "paper_bgcolor": "white", "height": 520,
               "margin": {"l": 20, "r": 20, "t": 90, "b": 30},
               "annotations": [{"x": 0.5, "y": -0.04, "xref": "paper", "yref": "paper",
                                 "text": summary, "showarrow": False,
                                 "font": {"size": 9, "color": "#888888"}, "align": "center"}]}}

    out_path = data_dir / "db_summary.json"
    out_path.write_text(_json.dumps(doc))
    print(f"[step16] db_summary.json → {out_path}")


def _sync_frontend(out_dir: Path, out_root: Path) -> None:
    """Sync step16 outputs to the Next.js frontend public directory."""
    import shutil

    frontend_map = _FRONTEND_MAP
    frontend_map.mkdir(parents=True, exist_ok=True)

    data_dir = frontend_map / "data"
    data_dir.mkdir(exist_ok=True)

    # Static PNGs
    n_png = 0
    for png in out_dir.glob("*.png"):
        shutil.copy2(png, frontend_map / png.name)
        n_png += 1

    # Interactive JSONs + CSVs
    n_json = 0
    int_dir = out_dir / "interactive"
    if int_dir.exists():
        for f in list(int_dir.glob("*.json")) + list(int_dir.glob("*.csv")):
            shutil.copy2(f, data_dir / f.name)
            if f.suffix == ".json":
                n_json += 1
    n_csv = len(list(data_dir.glob("*.csv"))) if data_dir.exists() else 0

    # db_summary.json — generated from pipeline metadata
    _generate_db_summary(out_root, data_dir)

    print(f"[step16] Synced to frontend: {n_png} PNGs + {n_json} JSONs + {n_csv} CSVs -> {_FRONTEND_MAP}")


# =============================================================================
# Entrypoint
# =============================================================================

def run(config: dict) -> dict:
    out_root = Path(config.get("out_dir", "outputs"))
    out_dir  = _out_dir(out_root)

    print(f"[step16] Output dir: {out_dir}")
    t0 = time.time()

    figures_saved: List[str] = []

    # ROSES — interactive Sankey first (exports both JSON and PNG via kaleido).
    # The old matplotlib _roses_flow is called as a fallback only if kaleido fails.
    print("[step16] Producing ROSES flow diagram...")
    _roses_interactive(out_root, out_dir)   # writes roses_flow.json + roses_flow.png
    figures_saved.append("roses_flow.json")
    figures_saved.append("roses_flow.png")

    # Load coded data for figures (LLM-coded only) and full CSV for studies table
    df = _load_coded(out_root)

    if df.empty:
        print("[step16] No full-text coded records yet — only ROSES diagram produced.")
        print("[step16] Re-run after step15 completes to produce all figures.")
    else:
        print("[step16] Producing evidence map figures (static)...")
        for fn, label in [
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

        print("[step16] Producing evidence map figures (interactive)...")
        for fn, label in [
            (_egm_interactive,           "evidence_gap_map.json"),
            (_geographic_interactive,    "geographic_map.json + geographic_bar.json"),
            (_temporal_interactive,      "temporal_trends.json"),
            (_producer_interactive,      "producer_type.json"),
            (_methodology_interactive,   "methodology.json"),
            (_domain_type_interactive,   "domain_type.json"),
            (_equity_interactive,        "equity.json"),
            (_domain_heatmap_interactive,"domain_heatmap.json"),
        ]:
            try:
                if fn == _egm_interactive:
                    fn(df, out_root, out_dir)
                else:
                    fn(df, out_dir)
                figures_saved.append(label)
            except Exception as e:
                print(f"[step16] WARNING: {label} failed — {type(e).__name__}: {e}")

        try:
            # studies.json — coded records only (columns populated)
            _export_studies_json(df, out_dir)
            figures_saved.append("studies.json")
        except Exception as e:
            print(f"[step16] WARNING: studies.json failed — {e}")

        try:
            # studies.csv — coded records only (same filter as studies.json)
            _export_studies_csv(df, out_dir)
            figures_saved.append("studies.csv")
        except Exception as e:
            print(f"[step16] WARNING: studies.csv failed — {e}")

    # Sync to frontend
    try:
        _sync_frontend(out_dir, out_root)
    except Exception as e:
        print(f"[step16] WARNING: frontend sync failed — {e}")

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
