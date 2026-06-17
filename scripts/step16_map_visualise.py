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

HUMAN_COLOR  = "#E07B39"   # amber/orange — human coding track


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


def _save_json_to(fig: Any, path: Path) -> None:
    """Write a Plotly figure JSON to an exact path (used by human/compare figure functions)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_json(str(path))
    print(f"[step16] -> {path.name}")


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
    # Quantitative
    "quantitative":                            "Quantitative",
    "surveys":                                 "Quantitative",
    "survey":                                  "Quantitative",
    "econometric":                             "Quantitative",
    "longitudinal":                            "Quantitative",
    "longitudinal_panel":                      "Quantitative",
    # Qualitative
    "qualitative":                             "Qualitative",
    "interviews":                              "Qualitative",
    "interview":                               "Qualitative",
    "semi_structured_interviews":              "Qualitative",
    "semi-structured interviews":              "Qualitative",
    "focus_groups":                            "Qualitative",
    "ethnographic":                            "Qualitative",
    "constant_comparative_methods":            "Qualitative",
    "case_study":                              "Qualitative",
    # Participatory (subset of qualitative)
    "participatory":                           "Participatory (qualitative)",
    "participatory_methods":                   "Participatory (qualitative)",
    "participatory_action_research":           "Participatory (qualitative)",
    # Experimental / RCT (subset of quantitative)
    "experimental":                            "Experimental / RCT (quantitative)",
    "rct":                                     "Experimental / RCT (quantitative)",
    "quasi_experimental":                      "Experimental / RCT (quantitative)",
    "quasi-experimental":                      "Experimental / RCT (quantitative)",
    "experimentation":                         "Experimental / RCT (quantitative)",
    "sureys":                                  "Quantitative",
    # Mixed methods
    "mixed_methods":                           "Mixed methods",
    "mixed-methods":                           "Mixed methods",
    "mixed_method":                            "Mixed methods",
    "mixed methods":                           "Mixed methods",
    "mixed method":                            "Mixed methods",
    # Modelling
    "modeling_with_empirical_validation":      "Modelling",
    "modelling_with_empirical_validation":     "Modelling",
    "emperical_validation_modeling":           "Modelling",
    "modelling_with_emperical_validation":     "Modelling",
    "modeling_with_emperical_validation":      "Modelling",
    "biophysical_model":                       "Modelling",
    "agent_based_model":                       "Modelling",
    "simulation":                              "Modelling",
    # Secondary data
    "secondary_data":                          "Secondary data",
    "literature_review":                       "Secondary data",
    "meta_analysis":                           "Secondary data",
    # Remote sensing / GIS
    "remote_sensing":                          "Remote sensing / GIS",
    "gis":                                     "Remote sensing / GIS",
    # Additional typos / free-text variants from human coders
    "sureys":                                  "Quantitative",
    "surey":                                   "Quantitative",
    "quanitative":                             "Quantitative",
    "experimentation":                         "Experimental / RCT (quantitative)",
    "experimental_design":                     "Experimental / RCT (quantitative)",
    "rct_experimental":                        "Experimental / RCT (quantitative)",
    "review":                                  "Secondary data",
    "desk_review":                             "Secondary data",
    "document_review":                         "Secondary data",
    "action_research":                         "Participatory",
    "focus_group_discussions":                 "Qualitative",
    "key_informant_interviews":                "Qualitative",
    "field_observation":                       "Qualitative",
    "observation":                             "Qualitative",
    # Other
    "other":                                   "Other",
}

# =============================================================================
# 5. EQUITY / MARGINALIZED SUBPOPULATIONS
# =============================================================================

_EQUITY_LABELS: Dict[str, str] = {
    # Women / gender
    "women":                              "Women / Gender",
    "women_farmers":                      "Women / Gender",
    "female":                             "Women / Gender",
    "female_farmers":                     "Women / Gender",
    "gender":                             "Women / Gender",
    # Men (separate dimension — track explicitly)
    "men":                                "Men / Gender",
    "male":                               "Men / Gender",
    "male_decision_making":               "Men / Gender",
    # Youth
    "youth":                              "Youth",
    "young_farmers":                      "Youth",
    "children":                           "Youth",
    # Indigenous peoples
    "indigenous_people":                  "Indigenous peoples",
    "indigenous_peoples":                 "Indigenous peoples",
    "indigenous":                         "Indigenous peoples",
    "maasai_pastoralists":                "Indigenous peoples",
    "maasai pastoralists":                "Indigenous peoples",
    "pastoralists":                       "Indigenous peoples",
    # People with disabilities
    "people_with_disabilities":           "People with disabilities",
    "disability":                         "People with disabilities",
    "disabled":                           "People with disabilities",
    # Elderly
    "elders":                             "Elderly",
    "elderly":                            "Elderly",
    "older_adults":                       "Elderly",
    # Ethnic minorities / marginalised communities
    "ethnic_minorities":                  "Ethnic minorities",
    "caste":                              "Ethnic minorities",
    "minority_groups":                    "Ethnic minorities",
    # Migrant / seasonal workers
    "migrant_seasonal_workers":           "Migrant / seasonal workers",
    "migrants":                           "Migrant / seasonal workers",
    "seasonal_workers":                   "Migrant / seasonal workers",
    # Landless / land-poor
    "landless":                           "Landless",
    "land_poor":                          "Landless",
    "poor_rural_farmers":                 "Landless",
    # Other
    "others":                             "Other",
    "other":                              "Other",
}

# Values that are NOT equity dimensions (skip silently)
_EQUITY_SKIP = {
    "nan", "", "not_found", "n/a", "unknown", "unclear",
    "in_doubt", "in doubt", "adults", "medium", "poor", "rich", "rural",
    "small_holder_farmers", "small_holder_farmer", "smallholder_rural_farmers",
    "smallholder rural farmers", "smallholder_farmers", "farmers",
    "poor rural farmers", "poor_rural_farmers",
}

_EQUITY_NONE_VALUES = {
    "none_reported", "none reported", "none", "not_reported",
    "no marginalized groups", "no_marginalized_groups",
    "none identified", "none_identified",
}


def _equity_label(raw: str) -> str | None:
    """Canonical label for a raw equity value.
    Returns 'No marginalized groups' for none_reported.
    Returns None for values that should be silently skipped.
    """
    key = raw.lower().strip()
    if key in _EQUITY_NONE_VALUES:
        return "No marginalized groups"
    if key in _EQUITY_SKIP:
        return None
    label = _EQUITY_LABELS.get(key) or _EQUITY_LABELS.get(key.replace(" ", "_"))
    if label:
        return label
    return raw.replace("_", " ").title()


def _method_label(raw: str) -> str:
    """Canonical label for a raw methodology value."""
    key = raw.lower().strip()
    label = _METHOD_LABELS.get(key) or _METHOD_LABELS.get(key.replace(" ", "_"))
    return label or raw.replace("_", " ").title()


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
        label = _method_label(raw)
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

    raw_items = [i for i in _split_multi(df[col])
                 if i.lower() not in _EQUITY_SKIP]
    if not raw_items:
        return
    merged_eq: Dict[str, int] = {}
    for raw in raw_items:
        label = _equity_label(raw)
        if label is None:
            label = raw.replace("_", " ").title()
        merged_eq[label] = merged_eq.get(label, 0) + 1
    eq_series = pd.Series(merged_eq).sort_values(ascending=False).head(12)

    import textwrap as _tw
    counts = eq_series
    wrapped = ["\n".join(_tw.wrap(k, width=16)) for k in counts.index]

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

    raw_items = _split_multi(df[col])
    if not raw_items:
        return
    merged_mi: Dict[str, int] = {}
    for raw in raw_items:
        label = _method_label(raw)
        merged_mi[label] = merged_mi.get(label, 0) + 1
    counts = pd.Series(merged_mi).sort_values(ascending=False).head(12)

    fig = go.Figure(go.Bar(
        x=counts.values.tolist(),
        y=counts.index.tolist(),
        orientation="h",
        marker_color=PALETTE[:len(counts)],
        hovertemplate="<b>%{y}</b><br>Studies: %{x}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text=f"<b>Methodological Approach</b><br><sup>Primary study design, multi-select (n={len(df):,})</sup>",
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

    raw_items = [i for i in _split_multi(df[col]) if i.lower() not in _EQUITY_SKIP]
    if not raw_items:
        return
    merged_eq: Dict[str, int] = {}
    for raw in raw_items:
        label = _equity_label(raw) or raw.replace("_", " ").title()
        merged_eq[label] = merged_eq.get(label, 0) + 1
    counts = pd.Series(merged_eq).sort_values(ascending=False)

    eq_colors_llm = [RED if c == "No marginalized groups" else PURPLE
                     for c in counts.index]
    fig = go.Figure(go.Bar(
        x=counts.index.tolist(),
        y=counts.values.tolist(),
        marker_color=eq_colors_llm,
        hovertemplate="<b>%{x}</b><br>Studies: %{y}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text=f"<b>Equity & Inclusion Dimensions Reported</b><br>"
                 f"<sup>Red = no marginalized group focus. Multi-select (n={len(df):,})</sup>",
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

    # ── Human coding numbers (authoritative: step15_human.csv + step15b_summary.json) ──
    # Prefer step15b_summary.json for totals (covers all rounds incl. legacy FT-R1x).
    # Fall back to scanning step14b cleaned CSVs if the summary is absent.
    _s15b_summary = out_root / "step15" / "step15b_summary.json"
    _s15_human    = out_root / "step15" / "step15_human.csv"
    n_h_screened = 0; n_h_included = 0; n_h_excluded = 0; n_h_batches = 0

    if _s15b_summary.exists():
        import json as _json
        with open(_s15b_summary) as _f:
            _s15b = _json.load(_f)
        n_h_screened  = _s15b.get("total_coder_rows", 0)
        n_h_included  = _s15b.get("final_human_records", 0)
        n_h_excluded  = n_h_screened - n_h_included
        n_h_batches   = _s15b.get("batches", 1)
    elif _s15_human.exists():
        _hdf = pd.read_csv(_s15_human, dtype=str).fillna("")
        n_h_included  = len(_hdf)
        n_h_screened  = n_h_included   # best estimate without raw files
        n_h_batches   = _hdf["batch"].nunique() if "batch" in _hdf.columns else 1
    else:
        _step14b_dir = out_root / "step14b"
        _coder_re    = re.compile(r"coding_ft-.+_([A-Za-z]{1,5})\.csv$", re.IGNORECASE)
        _skip_names  = {"template", "llm", "reconciled", "fixed", "papers"}
        if _step14b_dir.exists():
            for _batch_csv in sorted(_step14b_dir.glob("FT-R*/coding_ft-*.csv")):
                if any(s in _batch_csv.name.lower() for s in _skip_names):
                    continue
                if not _coder_re.search(_batch_csv.name):
                    continue
                try:
                    _bdf = pd.read_csv(_batch_csv, dtype=str).fillna("")
                    if _bdf["confirmed_include"].str.strip().eq("").all():
                        continue
                    n_h_screened += len(_bdf)
                    n_h_included += (_bdf["confirmed_include"].str.lower() == "yes").sum()
                    n_h_excluded += (_bdf["confirmed_include"].str.lower() == "no").sum()
                except Exception:
                    pass
        n_h_batches = max(1, n_h_screened // 20)

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
    #   N+7        LLM full-text screening: included
    #   N+8        Full texts excluded by LLM [TERMINAL]
    #   N+9        Pending: LLM FT screening [TERMINAL]
    #   N+10       Human FT coding batches
    #   N+11       Human confirmed included [TERMINAL — PRIMARY OUTPUT]
    #   N+12       Human excluded [TERMINAL]

    N       = len(_dbs)
    I_DUP     = N + 1
    I_SCR     = N + 2   # abstract screening / unique records
    I_INC     = N + 3
    I_EXC     = N + 4
    I_FTRET   = N + 5
    I_FTNR    = N + 6
    I_CODED   = N + 7   # LLM included
    I_FTEXC   = N + 8
    I_PEND    = N + 9
    I_H_BATCH = N + 10  # human coding batches
    I_H_INC   = N + 11  # human confirmed included (PRIMARY)
    I_H_EXC   = N + 12  # human excluded

    labels = (
        [f"Scopus<br>identified<br>n={n_scopus:,} ({_pct(n_scopus, n_total_id)})"]
        + [f"{_DB_SHORT.get(db, db)}<br>identified<br>n={_db_imp[db]:,} ({_pct(_db_imp[db], n_total_id)})" for db in _dbs]
        + [
            f"Duplicates removed<br>n={n_total_dup:,} ({_pct(n_total_dup, n_total_id)})",
            f"Abstract screening<br>n={n_screened:,} ({_pct(n_screened, n_total_id)})",
            f"Abstract included<br>n={n_abs_inc:,} ({_pct(n_abs_inc, n_screened)})",
            f"Abstract excluded<br>n={n_abs_exc:,} ({_pct(n_abs_exc, n_screened)})",
            f"Full texts retrieved<br>n={n_ft_retrieved_all:,} auto<br>+ manual procurement",
            f"Not retrievable<br>n={n_ft_not_ret_all:,} ({_pct(n_ft_not_ret_all, n_abs_inc)})",
            f"LLM screening<br>(auto-retrieved only)<br>n={n_ft_include:,} included ({_pct(n_ft_include, n_ft_retrieved_all)})",
            f"LLM excluded<br>n={n_ft_exclude:,} ({_pct(n_ft_exclude, n_ft_retrieved_all)})",
            f"Pending LLM<br>n={n_ft_pending_screen:,}",
            f"Human batches<br>{n_h_batches} rounds · n={n_h_screened:,}",
            f"<b>Human included ★</b><br>n={n_h_included:,} ({_pct(n_h_included, n_h_screened)})<br>primary output",
            f"Human excluded<br>n={n_h_excluded:,} ({_pct(n_h_excluded, n_h_screened)})",
        ]
    )

    C_ENTRY   = "#4e79a7"   # steel blue  — identified / screening nodes
    C_EXC     = "#c0392b"   # crimson     — duplicates / abstract excluded
    C_INC     = "#2d8f4e"   # forest green — abstract/FT included
    C_INC_LT  = "#5ab56e"   # medium green — FT retrieved
    C_EXC_LT  = "#9b59b6"   # purple      — FT excluded
    C_INC_EXT = "#3cb371"   # sea green   — LLM coded
    C_PEND    = "#6baed6"   # sky blue    — pending
    C_HUMAN   = "#d4772a"   # amber/orange — human coding (primary)
    C_H_INC   = "#b85c00"   # dark amber  — human confirmed included
    C_H_EXC   = "#c0392b"   # crimson     — human excluded

    node_colors = (
        [C_ENTRY]         # 0   Scopus
        + [C_ENTRY] * N   # 1…N other databases
        + [
            C_EXC,        # N+1  Duplicates removed
            C_ENTRY,      # N+2  Abstract screening
            C_INC,        # N+3  Abstract included
            C_EXC,        # N+4  Abstract excluded
            C_INC_LT,     # N+5  FT retrieved
            C_PEND,       # N+6  Not retrievable
            C_INC_EXT,    # N+7  LLM included
            C_EXC_LT,     # N+8  LLM FT excluded
            C_PEND,       # N+9  Pending LLM
            C_HUMAN,      # N+10 Human batches
            C_H_INC,      # N+11 Human confirmed included (PRIMARY)
            C_H_EXC,      # N+12 Human excluded
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
    # LLM track (automated, auto-retrieved papers only)
    add(I_FTRET, I_CODED, n_ft_include,       "rgba(45,143,78,0.32)",
        f"LLM included: {n_ft_include:,}" + (f"<br>Top criteria: {excl14_text}" if excl14_text else ""))
    add(I_FTRET, I_FTEXC, n_ft_exclude,       "rgba(155,89,182,0.22)",
        f"LLM full texts excluded: {n_ft_exclude:,}")
    add(I_FTRET, I_PEND,  n_ft_pending_screen,"rgba(107,174,214,0.20)",
        f"Pending LLM screening: {n_ft_pending_screen:,}")

    # Human track (primary — all retrieved incl. manually procured, sampled batches)
    add(I_FTRET,   I_H_BATCH, n_h_screened, "rgba(212,119,42,0.35)",
        f"Human FT coding: {n_h_screened:,} papers assigned ({n_h_batches} batches)")
    add(I_H_BATCH, I_H_INC,   n_h_included, "rgba(184,92,0,0.45)",
        f"Human confirmed included: {n_h_included:,} (PRIMARY)")
    add(I_H_BATCH, I_H_EXC,   n_h_excluded, "rgba(192,57,43,0.25)",
        f"Human excluded: {n_h_excluded:,}")

    # ── Explicit node positions (freeform layout) ────────────────────────────
    # DB source column; shared pipeline; LLM track (lower); human track (upper-right).
    # Human nodes occupy the top-right of the final column to visually indicate
    # they are the primary output. LLM nodes sit below as the supporting track.
    n_db = N + 1  # Scopus + N other-DB nodes
    _db_ys = [round(0.02 + i * 0.96 / max(n_db - 1, 1), 4) for i in range(n_db)]

    _X = dict(db=0.01, dup=0.18, scr=0.28, abs=0.50, ft=0.68, llm=0.81, human=0.87, final=0.97)
    node_x = (
        [_X["db"]] * n_db
        + [_X["dup"],                            # I_DUP
           _X["scr"],                            # I_SCR
           _X["abs"],  _X["abs"],                # I_INC, I_EXC
           _X["ft"],   _X["ft"],                 # I_FTRET, I_FTNR
           _X["llm"],  _X["llm"],  _X["llm"],    # I_CODED, I_FTEXC, I_PEND  (LLM track)
           _X["human"],                          # I_H_BATCH
           _X["final"], _X["final"]]             # I_H_INC, I_H_EXC
    )
    node_y = (
        _db_ys
        + [0.50,   # I_DUP: center
           0.50,   # I_SCR: center
           0.35,   # I_INC: upper
           0.82,   # I_EXC: lower
           0.28,   # I_FTRET: upper-center (feeds both LLM and human tracks)
           0.78,   # I_FTNR: lower
           0.58,   # I_CODED: LLM included — mid
           0.76,   # I_FTEXC: LLM excluded — lower
           0.90,   # I_PEND: LLM pending — bottom
           0.10,   # I_H_BATCH: human batches — near top
           0.03,   # I_H_INC: human included — TOP (primary output)
           0.22]   # I_H_EXC: human excluded — upper
    )

    fig = go.Figure(go.Sankey(
        arrangement="freeform",
        node=dict(
            pad=20, thickness=22,
            x=node_x, y=node_y,
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
                "<b>ROSES Flow Diagram — Scopus + 27 other sources</b><br>"
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
                fallback = row.get("year", "")
                try:
                    fallback = str(int(float(fallback)))
                except (ValueError, TypeError):
                    fallback = str(fallback or "").strip()
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
                meta_year = pd.to_numeric(df["year"], errors="coerce").apply(
                    lambda x: str(int(x)) if pd.notna(x) else ""
                )
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

    # Human and compare subdirs
    import shutil as _shutil
    for subdir in ["human", "compare"]:
        src_sub = int_dir / subdir
        dst_sub = data_dir / subdir
        if src_sub.exists():
            dst_sub.mkdir(exist_ok=True)
            n_sub = 0
            for f in src_sub.glob("*.json"):
                _shutil.copy2(f, dst_sub / f.name)
                n_sub += 1
            print(f"[step16] Synced {n_sub} {subdir}/ JSONs -> {dst_sub}")

    print(f"[step16] Synced to frontend: {n_png} PNGs + {n_json} JSONs + {n_csv} CSVs -> {_FRONTEND_MAP}")


# =============================================================================
# LLM vs Human comparison figure
# =============================================================================

# Maps human CSV column → LLM _value column, display label, field type
_LVH_FIELDS = [
    ("producer_type",          "producer_type_value",          "Producer Type",          "multi"),
    ("methodological_approach","methodological_approach_value", "Methodology",            "multi"),
    ("geographic_scale",       "geographic_scale_value",        "Geographic Scale",       "cat"),
    ("publication_type",       "publication_type_value",        "Publication Type",       "cat"),
    ("temporal_coverage",      "temporal_coverage_value",       "Temporal Coverage",      "cat"),
    ("process_outcome_domains","process_outcome_domains_value", "Process/Outcome Domains","multi"),
]


def _lvh_counts(series: pd.Series, multi: bool) -> pd.Series:
    """Return value counts normalised to % from a series of (possibly semi-colon-joined) values."""
    if multi:
        items = _split_multi(series)
    else:
        items = [v.strip().lower() for v in series.dropna().astype(str)
                 if v.strip().lower() not in _SKIP]
    if not items:
        return pd.Series(dtype=float)
    counts = pd.Series(items).value_counts()
    return (counts / counts.sum() * 100).round(1)


def _llm_vs_human_comparison(llm_df: pd.DataFrame, out_root: Path, out_dir: Path) -> None:
    """Grouped bar chart: LLM distribution vs human-coded distribution for key fields."""
    human_path = out_root / "step15" / "step15_human.csv"
    if not human_path.exists():
        print("[step16] step15_human.csv not found — skipping LLM vs human comparison")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        return

    human_df = pd.read_csv(human_path, dtype=str).fillna("")

    n_panels = len(_LVH_FIELDS)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 7))
    if n_panels == 1:
        axes = [axes]

    COLOR_LLM   = TEAL
    COLOR_HUMAN = "#E07B39"

    for ax, (human_col, llm_col, label, ftype) in zip(axes, _LVH_FIELDS):
        multi = ftype == "multi"

        llm_pct   = _lvh_counts(llm_df[llm_col]   if llm_col   in llm_df.columns   else pd.Series(dtype=str), multi)
        human_pct = _lvh_counts(human_df[human_col] if human_col in human_df.columns else pd.Series(dtype=str), multi)

        if llm_pct.empty and human_pct.empty:
            ax.set_visible(False)
            continue

        all_cats = sorted(set(llm_pct.index) | set(human_pct.index),
                          key=lambda c: -(llm_pct.get(c, 0) + human_pct.get(c, 0)))[:12]

        y      = np.arange(len(all_cats))
        height = 0.35
        llm_v   = [llm_pct.get(c, 0)   for c in all_cats]
        human_v = [human_pct.get(c, 0) for c in all_cats]

        ax.barh(y + height / 2, llm_v,   height, color=COLOR_LLM,   label=f"LLM (n={len(llm_df):,})")
        ax.barh(y - height / 2, human_v, height, color=COLOR_HUMAN, label=f"Human (n={len(human_df):,})")

        ax.set_yticks(y)
        ax.set_yticklabels(all_cats, fontsize=8)
        ax.set_xlabel("% of studies", fontsize=9)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.invert_yaxis()

    # Shared legend
    llm_patch   = mpatches.Patch(color=COLOR_LLM,   label=f"LLM (n={len(llm_df):,})")
    human_patch = mpatches.Patch(color=COLOR_HUMAN, label=f"Human (n={len(human_df):,})")
    fig.legend(handles=[llm_patch, human_patch], loc="upper center",
               ncol=2, fontsize=10, bbox_to_anchor=(0.5, 1.01))

    fig.suptitle("LLM vs Human Coding — Distribution Comparison", fontsize=13,
                 fontweight="bold", y=1.04)
    plt.tight_layout()
    _save(fig, out_dir / "llm_vs_human.png")


# =============================================================================
# Human-only and LLM-vs-Human comparison interactive figures
# =============================================================================

# Column mapping: human CSV col → LLM _value col
_HUMAN_COL_MAP = {
    "country_region":              "country_region_value",
    "methodological_approach":     "methodological_approach_value",
    "producer_type":               "producer_type_value",
    "process_outcome_domains":     "process_outcome_domains_value",
    "marginalized_subpopulations": "marginalized_subpopulations_value",
    "publication_year":            "publication_year_value",
    "geographic_scale":            "geographic_scale_value",
    "adaptation_focus":            "adaptation_focus_value",
}

_SKIP_H = {"nan", "", "not_found", "n/a", "none", "unknown", "unclear", "not_reported"}
_SKIP_GEO = {
    "global", "sub-saharan africa", "east africa", "west africa",
    "central africa", "southern africa", "north africa",
    "south asia", "south-east asia", "southeast asia", "latin america",
    "caribbean", "lmics", "low-income countries", "not_reported", "",
}


def _normalize_human_df(df: pd.DataFrame) -> pd.DataFrame:
    """Rename human CSV columns to _value equivalents for reuse of figure logic."""
    rename = {k: v for k, v in _HUMAN_COL_MAP.items()
              if k in df.columns and v not in df.columns}
    return df.rename(columns=rename)


def _split_h(series: pd.Series) -> List[str]:
    out: List[str] = []
    for val in series.dropna().astype(str):
        for p in val.split(";"):
            p = p.strip()
            if p and p.lower() not in _SKIP_H:
                out.append(p)
    return out


_COUNTRY_TYPO_MAP = {
    # typos / misspellings
    "euthopia": "Ethiopia", "ethopia": "Ethiopia", "ethipia": "Ethiopia",
    "ndia": "India",
    "vietam": "Vietnam",
    # case variants handled by title() but keep explicit ones too
    "china": "China", "vietnam": "Vietnam", "zimbabwe": "Zimbabwe",
    "ghana": "Ghana", "uganda": "Uganda", "zambia": "Zambia",
    "indonesia": "Indonesia", "tunisia": "Tunisia", "brazil": "Brazil",
    "nicaragua": "Nicaragua", "south_africa": "South Africa",
    # sub-national → country
    "susa county": "Iran", "bakian village": "Iran",
    "khuzestan province (southwestern iran)": "Iran",
    "fars province": "Iran",
    "northwest (battambang province and pailin province)": "Cambodia",
}

_COUNTRY_REGION_SKIP = {
    "east africa", "west africa", "central africa", "southern africa",
    "north africa", "west_african", "africa", "sub-saharan africa",
    "south asia", "south-east asia", "southeast asia", "latin america",
    "central america", "caribbean", "lmics", "low-income countries",
    "not_reported", "global", "",
}


import re as _re

_KNOWN_COUNTRIES = {
    "afghanistan","albania","algeria","angola","argentina","armenia","australia",
    "austria","azerbaijan","bangladesh","benin","bolivia","botswana","brazil",
    "burkina faso","burundi","cambodia","cameroon","chad","chile","china","colombia",
    "congo","costa rica","côte d'ivoire","cote d'ivoire","cuba","ecuador","egypt",
    "eritrea","eswatini","ethiopia","gambia","ghana","guatemala","guinea","haiti",
    "honduras","india","indonesia","iran","iraq","jordan","kenya","laos","lesotho",
    "liberia","madagascar","malawi","malaysia","mali","mauritania","mexico","mongolia",
    "morocco","mozambique","myanmar","namibia","nepal","nicaragua","niger","nigeria",
    "pakistan","panama","papua new guinea","peru","philippines","rwanda","senegal",
    "sierra leone","somalia","south africa","sri lanka","sudan","tanzania","thailand",
    "timor-leste","togo","tunisia","turkey","uganda","ukraine","uruguay","vietnam",
    "zambia","zimbabwe",
}

_COUNTRY_TYPO_MAP = {
    "euthopia": "Ethiopia", "ethopia": "Ethiopia", "ethipia": "Ethiopia",
    "euthiopia": "Ethiopia",
    "ndia": "India",
    "vietam": "Vietnam",
    "china": "China", "vietnam": "Vietnam", "zimbabwe": "Zimbabwe",
    "ghana": "Ghana", "uganda": "Uganda", "zambia": "Zambia",
    "indonesia": "Indonesia", "tunisia": "Tunisia", "brazil": "Brazil",
    "nicaragua": "Nicaragua", "south_africa": "South Africa",
}

_COUNTRY_REGION_SKIP = {
    "east africa", "west africa", "central africa", "southern africa",
    "north africa", "west_african", "africa", "sub-saharan africa",
    "south asia", "south-east asia", "southeast asia", "latin america",
    "central america", "caribbean", "lmics", "low-income countries",
    "not_reported", "global", "east", "horn of africa", "eritrea horn of africa",
    "",
}

_SUBNATIONAL_SIGNALS = [
    "province", "region", "district", "county", "village", "division",
    "basin", "valley", "plain", "zone", "state", "municipality", "prefecture",
    "department", "commune", "ward", "highland", "lowland", "savannah",
]


def _is_subnational(token: str) -> bool:
    t = token.lower()
    return any(s in t for s in _SUBNATIONAL_SIGNALS)


def _extract_countries(raw: str) -> list:
    """Extract canonical country names from a raw country_region string."""
    import re
    results = []
    # Remove parenthetical sub-national content
    cleaned = re.sub(r'\([^)]*\)', '', raw)
    # Split on ; , and (case-insensitive)
    parts = re.split(r'[;,]|\band\b', cleaned, flags=re.IGNORECASE)
    for part in parts:
        part = part.strip(" ._-")
        if not part or len(part) < 3:
            continue
        # Remove underscores — treat first segment as country
        if '_' in part:
            part = part.split('_')[0].strip()
        lower = part.lower().strip()
        # Skip regional/supranational labels
        if lower in _COUNTRY_REGION_SKIP:
            continue
        # Typo map
        if lower in _COUNTRY_TYPO_MAP:
            results.append(_COUNTRY_TYPO_MAP[lower])
            continue
        # Skip obvious sub-national entries
        if _is_subnational(lower):
            continue
        # Match against known country list
        title = part.title().replace("'S", "'s")
        if lower in _KNOWN_COUNTRIES:
            results.append(title)
        elif lower.replace("-", " ") in _KNOWN_COUNTRIES:
            results.append(title)
    return results


def _geo_counts(df: pd.DataFrame, col: str) -> Dict[str, int]:
    cc: Dict[str, int] = {}
    if col not in df.columns:
        return cc
    for val in df[col].dropna().astype(str):
        for country in _extract_countries(val):
            if country.lower() not in _COUNTRY_REGION_SKIP:
                cc[country] = cc.get(country, 0) + 1
    return cc


def _cell_counts_egm(df: pd.DataFrame) -> Dict[Tuple[str, str], int]:
    cc: Dict[Tuple[str, str], int] = {}
    dom_col  = "process_outcome_domains_value"
    prod_col = "producer_type_value"
    if dom_col not in df.columns or prod_col not in df.columns:
        return cc
    for _, row in df.iterrows():
        domains   = [d.strip() for d in str(row.get(dom_col, "")).split(";")
                     if d.strip() and d.strip().lower() not in _SKIP_H]
        prodtypes = [p.strip() for p in str(row.get(prod_col, "")).split(";")
                     if p.strip() and p.strip().lower() not in _SKIP_H]
        for d in domains:
            for p in prodtypes:
                cc[(d, p)] = cc.get((d, p), 0) + 1
    return cc


def _human_figures_all(human_df: pd.DataFrame, out_dir: Path) -> None:
    """Generate all human-only interactive Plotly figures to out_dir/interactive/human/."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return

    h = _normalize_human_df(human_df)
    n = len(h)
    target = out_dir / "interactive" / "human"
    target.mkdir(parents=True, exist_ok=True)
    print(f"[step16] Generating human figures (n={n}) -> {target.name}")

    # ── Temporal ──────────────────────────────────────────────────────────────
    col = "publication_year_value"
    if col in h.columns:
        years = pd.to_numeric(h[col], errors="coerce").dropna().astype(int)
        years = years[(years >= 2005) & (years <= 2027)]
        if not years.empty:
            counts = years.value_counts().sort_index()
            fig = go.Figure(go.Bar(
                x=counts.index.tolist(), y=counts.values.tolist(),
                marker_color=HUMAN_COLOR,
                hovertemplate="<b>%{x}</b><br>Studies: %{y}<extra></extra>",
            ))
            fig.update_layout(
                title=dict(text=f"<b>Publications Over Time (Human-coded)</b><br><sup>n={len(years):,}</sup>",
                           x=0.5, xanchor="center", font=dict(size=14)),
                height=380, xaxis_title="Publication Year", yaxis_title="Number of Studies",
                plot_bgcolor="white", paper_bgcolor="white",
                font=dict(family="Lato, Arial, sans-serif", size=11, color=DKGREY),
                margin=dict(l=60, r=40, t=100, b=60),
            )
            _save_json_to(fig, target / "temporal_trends.json")

    # ── Producer type ─────────────────────────────────────────────────────────
    col = "producer_type_value"
    if col in h.columns:
        items = _split_h(h[col])
        if items:
            raw_counts = pd.Series(items).value_counts()
            merged: Dict[str, int] = {}
            for raw, nv in raw_counts.items():
                label = PRODUCER_LABELS.get(raw.lower().strip(), None)
                if label is None:
                    continue  # skip undefined / unrecognised
                merged[label] = merged.get(label, 0) + nv
            counts = pd.Series(merged).sort_values(ascending=False)
            if counts.empty:
                counts = raw_counts.head(8).rename(
                    lambda r: r.replace("_", " ").title())
            fig = go.Figure(go.Bar(
                x=counts.values.tolist(), y=counts.index.tolist(), orientation="h",
                marker_color=HUMAN_COLOR,
                hovertemplate="<b>%{y}</b><br>Studies: %{x}<extra></extra>",
            ))
            fig.update_layout(
                title=dict(text=f"<b>Producer Types Studied (Human-coded)</b><br><sup>n={n:,}</sup>",
                           x=0.5, xanchor="center", font=dict(size=14)),
                height=380, xaxis_title="Number of Studies",
                yaxis=dict(autorange="reversed"),
                plot_bgcolor="white", paper_bgcolor="white",
                font=dict(family="Lato, Arial, sans-serif", size=11, color=DKGREY),
                margin=dict(l=180, r=40, t=100, b=60),
            )
            _save_json_to(fig, target / "producer_type.json")

    # ── Methodology ───────────────────────────────────────────────────────────
    col = "methodological_approach_value"
    if col in h.columns:
        items = _split_h(h[col])
        if items:
            raw_counts = pd.Series(items).value_counts()
            merged_m: Dict[str, int] = {}
            for raw, nv in raw_counts.items():
                label = _method_label(raw)
                merged_m[label] = merged_m.get(label, 0) + nv

            # Stacked overlay: parent categories with subtypes shown inside
            # Parent bars (full width, light colour)
            parents = {
                "Quantitative":   merged_m.get("Quantitative", 0),
                "Qualitative":    merged_m.get("Qualitative",  0),
                "Modelling":      merged_m.get("Modelling",    0),
                "Mixed methods":  merged_m.get("Mixed methods",0),
            }
            # Subtype bars (overlaid at base, dark colour, same y)
            subtypes = {
                "Quantitative":  merged_m.get("Experimental / RCT (quantitative)", 0),
                "Qualitative":   merged_m.get("Participatory (qualitative)",        0),
                "Modelling":     0,
                "Mixed methods": 0,
            }
            sub_labels = {
                "Quantitative": "Experimental / RCT",
                "Qualitative":  "Participatory",
                "Modelling":    "",
                "Mixed methods": "",
            }
            ys = list(parents.keys())
            parent_vals = [parents[y] for y in ys]
            sub_vals    = [subtypes[y] for y in ys]

            LIGHT_BLUE  = "#93c5fd"
            DARK_BLUE   = "#1d4ed8"
            LIGHT_GREEN = "#86efac"
            DARK_GREEN  = "#15803d"
            ORANGE      = "#f97316"
            PURPLE      = "#a855f7"

            parent_colors = [LIGHT_BLUE, LIGHT_GREEN, ORANGE, PURPLE]
            sub_colors    = [DARK_BLUE,  DARK_GREEN,  "rgba(0,0,0,0)", "rgba(0,0,0,0)"]

            hover_parent = [
                f"<b>{y}</b><br>Total: {parents[y]} studies" +
                (f"<br>incl. {subtypes[y]} {sub_labels[y]}" if subtypes[y] else "") +
                "<extra></extra>"
                for y in ys
            ]
            hover_sub = [
                f"<b>{sub_labels[y]}</b> (subset of {y})<br>{subtypes[y]} studies<extra></extra>"
                if subtypes[y] else "<extra></extra>"
                for y in ys
            ]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Parent category",
                x=parent_vals, y=ys, orientation="h",
                marker_color=parent_colors,
                hovertemplate=hover_parent,
                showlegend=False,
            ))
            fig.add_trace(go.Bar(
                name="Subtype",
                x=sub_vals, y=ys, orientation="h",
                marker_color=sub_colors,
                hovertemplate=hover_sub,
                showlegend=False,
            ))

            # Labels for both portions of each bar
            for i, y in enumerate(ys):
                p = parents[y]
                s = subtypes[y]
                rest = p - s
                if s > 0:
                    # Dark subtype portion: centered if wide enough, else outside right
                    sub_text = f"Exp./RCT ({s})" if y == "Quantitative" else f"{sub_labels[y]} ({s})"
                    if s >= 18:
                        fig.add_annotation(x=s/2, y=y, text=sub_text,
                                           font=dict(size=10, color="white"), showarrow=False,
                                           xanchor="center", yanchor="middle")
                    else:
                        fig.add_annotation(x=s + 1, y=y, text=sub_text,
                                           font=dict(size=10, color="#1e3a8a"), showarrow=False,
                                           xanchor="left", yanchor="middle")
                    # Light remaining portion: centered
                    if rest > 12:
                        rest_label = f"Other {y.lower()} ({rest})"
                        fig.add_annotation(x=s + rest/2, y=y, text=rest_label,
                                           font=dict(size=10, color="#374151"), showarrow=False,
                                           xanchor="center", yanchor="middle")
                else:
                    # Single bar: count at end
                    if p > 0:
                        fig.add_annotation(x=p + 1, y=y, text=str(p),
                                           font=dict(size=10, color="#374151"), showarrow=False,
                                           xanchor="left", yanchor="middle")
            fig.update_layout(
                barmode="overlay",
                title=dict(
                    text=f"<b>Methodological Approach (Human-coded)</b><br>"
                         f"<sup>n={n:,} · studies may appear in multiple categories · "
                         f"dark shading shows subtype within parent category</sup>",
                    x=0.5, xanchor="center", font=dict(size=14)),
                height=380, xaxis_title="Number of Studies",
                yaxis=dict(autorange="reversed"),
                plot_bgcolor="white", paper_bgcolor="white",
                font=dict(family="Lato, Arial, sans-serif", size=11, color=DKGREY),
                margin=dict(l=160, r=40, t=100, b=60),
            )
            _save_json_to(fig, target / "methodology.json")

    # ── Process/Outcome domains ───────────────────────────────────────────────
    col = "process_outcome_domains_value"
    if col in h.columns:
        items = _split_h(h[col])
        if items:
            raw_counts = pd.Series(items).value_counts()
            merged_d: Dict[str, int] = {}
            for raw, nv in raw_counts.items():
                label = DOMAIN_LABELS.get(raw.lower().strip(),
                                          raw.replace("_", " ").title())
                merged_d[label] = merged_d.get(label, 0) + nv
            counts = pd.Series(merged_d)
            # Preserve canonical domain order
            ordered = [DOMAIN_LABELS[d] for d in ALL_DOMAINS if DOMAIN_LABELS[d] in merged_d]
            counts = counts.reindex(ordered).fillna(0).astype(int)
            bar_colors = [BLUE if d in [DOMAIN_LABELS[x] for x in PROCESS_DOMAINS]
                          else GREEN for d in counts.index]
            fig = go.Figure(go.Bar(
                x=counts.index.tolist(), y=counts.values.tolist(),
                marker_color=bar_colors,
                hovertemplate="<b>%{x}</b><br>Studies: %{y}<extra></extra>",
            ))
            fig.update_layout(
                title=dict(text=f"<b>Process/Outcome Domains (Human-coded)</b><br><sup>n={n:,}</sup>",
                           x=0.5, xanchor="center", font=dict(size=14)),
                height=420, yaxis_title="Number of Studies",
                xaxis=dict(tickangle=-35),
                plot_bgcolor="white", paper_bgcolor="white",
                font=dict(family="Lato, Arial, sans-serif", size=11, color=DKGREY),
                margin=dict(l=60, r=40, t=110, b=120),
            )
            _save_json_to(fig, target / "domain_type.json")

    # ── Equity ────────────────────────────────────────────────────────────────
    col = "marginalized_subpopulations_value"
    if col in h.columns:
        raw_eq = [i for i in _split_h(h[col]) if i.lower() not in _EQUITY_SKIP]
        if raw_eq:
            merged_eq_h: Dict[str, int] = {}
            for raw in raw_eq:
                label = _equity_label(raw) or raw.replace("_", " ").title()
                merged_eq_h[label] = merged_eq_h.get(label, 0) + 1
            counts = pd.Series(merged_eq_h).sort_values(ascending=False)
        items = raw_eq  # for the `if items:` guard below
        if items:
            eq_colors = [RED if c == "No marginalized groups" else PURPLE
                         for c in counts.index]
            fig = go.Figure(go.Bar(
                x=counts.index.tolist(), y=counts.values.tolist(),
                marker_color=eq_colors,
                hovertemplate="<b>%{x}</b><br>Studies: %{y}<extra></extra>",
            ))
            fig.update_layout(
                title=dict(text=f"<b>Equity & Inclusion (Human-coded)</b><br>"
                               f"<sup>Red = no marginalized group focus. Multi-select (n={n:,})</sup>",
                           x=0.5, xanchor="center", font=dict(size=14)),
                height=380, yaxis_title="Number of Studies",
                plot_bgcolor="white", paper_bgcolor="white",
                font=dict(family="Lato, Arial, sans-serif", size=11, color=DKGREY),
                margin=dict(l=60, r=40, t=100, b=80),
            )
            _save_json_to(fig, target / "equity.json")

    # ── Domain heatmap ────────────────────────────────────────────────────────
    dom_col = "process_outcome_domains_value"
    prod_col = "producer_type_value"
    if dom_col in h.columns and prod_col in h.columns:
        rows: List[dict] = []
        for _, row in h.iterrows():
            domains   = [DOMAIN_LABELS.get(d.strip().lower(), None)
                         for d in str(row.get(dom_col, "")).split(";")
                         if d.strip() and d.strip().lower() not in _SKIP_H]
            prodtypes = [PRODUCER_LABELS.get(p.strip().lower(), None)
                         for p in str(row.get(prod_col, "")).split(";")
                         if p.strip() and p.strip().lower() not in _SKIP_H]
            for d in domains:
                for p in prodtypes:
                    if d and p:
                        rows.append({"domain": d, "producer": p})
        if rows:
            ct = pd.DataFrame(rows).pivot_table(
                index="domain", columns="producer", aggfunc="size", fill_value=0)
            # Reindex to canonical order
            d_order = [DOMAIN_LABELS[d] for d in ALL_DOMAINS if DOMAIN_LABELS[d] in ct.index]
            p_order = [PRODUCER_LABELS[p] for p in PRODUCER_TYPES if PRODUCER_LABELS[p] in ct.columns]
            ct = ct.reindex(index=d_order, columns=p_order, fill_value=0)
            if not ct.empty:
                fig = go.Figure(go.Heatmap(
                    z=ct.values.tolist(), x=ct.columns.tolist(), y=ct.index.tolist(),
                    colorscale="YlOrRd",
                    hovertemplate="<b>%{y}</b><br>%{x}<br>Studies: %{z}<extra></extra>",
                    text=ct.values.tolist(), texttemplate="%{text}",
                ))
                fig.update_layout(
                    title=dict(text=f"<b>Domains × Producer Type (Human-coded)</b><br><sup>n={n:,}</sup>",
                               x=0.5, xanchor="center", font=dict(size=14)),
                    height=max(400, ct.shape[0] * 28 + 150),
                    xaxis=dict(tickangle=-30, side="top"),
                    yaxis=dict(autorange="reversed"),
                    plot_bgcolor="white", paper_bgcolor="white",
                    font=dict(family="Lato, Arial, sans-serif", size=10, color=DKGREY),
                    margin=dict(l=240, r=40, t=130, b=60),
                )
                _save_json_to(fig, target / "domain_heatmap.json")

    # ── Geographic map + bar ──────────────────────────────────────────────────
    cc = _geo_counts(h, "country_region_value")
    if cc:
        fig_map = go.Figure(go.Choropleth(
            locations=list(cc.keys()), z=list(cc.values()),
            locationmode="country names", colorscale="YlOrRd", colorbar_title="Studies",
            hovertemplate="<b>%{location}</b><br>Studies: %{z}<extra></extra>",
        ))
        fig_map.update_layout(
            title=dict(text=f"<b>Geographic Distribution (Human-coded)</b><br><sup>n={n:,}</sup>",
                       x=0.5, xanchor="center", font=dict(size=14)),
            geo=dict(showframe=False, showcoastlines=True, projection_type="natural earth",
                     showland=True, landcolor="#F5F5F5", showocean=True, oceancolor="#EBF4FB",
                     showcountries=True, countrycolor="#D0D0D0"),
            height=480, paper_bgcolor="white",
            font=dict(family="Lato, Arial, sans-serif", size=11, color=DKGREY),
            margin=dict(l=0, r=0, t=90, b=0),
        )
        _save_json_to(fig_map, target / "geographic_map.json")

        sorted_c = sorted(cc.items(), key=lambda x: -x[1])
        countries, values = [c for c, _ in sorted_c], [v for _, v in sorted_c]
        fig_bar = go.Figure(go.Bar(
            x=values, y=countries, orientation="h",
            marker_color=HUMAN_COLOR,
            hovertemplate="<b>%{y}</b><br>Studies: %{x}<extra></extra>",
            text=values, textposition="outside",
        ))
        fig_bar.update_layout(
            title=dict(text=f"<b>Studies by Country (Human-coded)</b><br><sup>n={n:,}</sup>",
                       x=0.5, xanchor="center", font=dict(size=14)),
            height=max(400, len(countries) * 22 + 120),
            xaxis_title="Number of Studies",
            yaxis=dict(autorange="reversed"),
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Lato, Arial, sans-serif", size=11, color=DKGREY),
            margin=dict(l=160, r=60, t=80, b=40),
            showlegend=False,
        )
        _save_json_to(fig_bar, target / "geographic_bar.json")

    # ── Evidence Gap Map ──────────────────────────────────────────────────────
    cell_cc = _cell_counts_egm(h)
    if cell_cc:
        max_n = max(cell_cc.values(), default=1)
        d_labels = [DOMAIN_LABELS.get(d, d) for d in ALL_DOMAINS]
        p_labels  = [PRODUCER_LABELS.get(p, p) for p in PRODUCER_TYPES]
        n_cont = sum(1 for _, row in h.iterrows() if any(
            d.strip() for d in str(row.get("process_outcome_domains_value", "")).split(";")
            if d.strip() and d.strip().lower() not in _SKIP_H))

        traces = []
        for d_set, color, name_sfx in [
            (PROCESS_DOMAINS, BLUE,  "Process domain"),
            (OUTCOME_DOMAINS, GREEN, "Outcome domain"),
        ]:
            xs, ys, sizes, texts = [], [], [], []
            for d in d_set:
                dl = DOMAIN_LABELS.get(d, d)
                for p in PRODUCER_TYPES:
                    pl = PRODUCER_LABELS.get(p, p)
                    nv = cell_cc.get((d, p), 0)
                    if nv > 0:
                        xs.append(pl); ys.append(dl); sizes.append(nv)
                        texts.append(f"<b>{dl}</b><br>Producer: {pl}<br>Studies: {nv}")
            if xs:
                traces.append(go.Scatter(
                    x=xs, y=ys, mode="markers+text", name=name_sfx,
                    marker=dict(size=sizes, sizemode="area", sizeref=max_n/(38**2), sizemin=10,
                                color=color, opacity=0.85, line=dict(color="white", width=1.5)),
                    text=[str(s) for s in sizes],
                    textposition="middle center", textfont=dict(size=10, color="white"),
                    hovertext=texts, hoverinfo="text", showlegend=False,
                ))

        gap_x, gap_y, gap_t = [], [], []
        for d in ALL_DOMAINS:
            dl = DOMAIN_LABELS.get(d, d)
            for p in PRODUCER_TYPES:
                pl = PRODUCER_LABELS.get(p, p)
                if cell_cc.get((d, p), 0) == 0:
                    gap_x.append(pl); gap_y.append(dl)
                    gap_t.append(f"<b>{dl}</b><br>Producer: {pl}<br><i>No studies — evidence gap</i>")
        if gap_x:
            traces.append(go.Scatter(x=gap_x, y=gap_y, mode="markers", name="Evidence gap",
                marker=dict(size=14, color="#E0E0E0", line=dict(color="#BDBDBD", width=1)),
                hovertext=gap_t, hoverinfo="text", showlegend=False))

        for cat, color in [("Process domain", BLUE), ("Outcome domain", GREEN)]:
            traces.append(go.Scatter(x=[None], y=[None], name=cat, mode="markers",
                marker=dict(size=14, color=color, opacity=0.85, line=dict(color="white", width=1.5)),
                showlegend=True))
        traces.append(go.Scatter(x=[None], y=[None], name="Evidence gap", mode="markers",
            marker=dict(size=10, color="#E0E0E0", line=dict(color="#BDBDBD", width=1)),
            showlegend=True))

        fig = go.Figure(data=traces)
        _n_out = len(OUTCOME_DOMAINS)
        fig.add_shape(type="line", xref="paper", yref="y", x0=0, x1=1,
            y0=_n_out - 0.5, y1=_n_out - 0.5,
            line=dict(color="#9E9E9E", width=1.5, dash="dot"))
        fig.update_layout(
            title=dict(
                text=f"<b>Evidence Gap Map (Human-coded)</b><br><sup>Bubble size = number of studies (n={n_cont:,})</sup>",
                x=0.5, xanchor="center", font=dict(size=14)),
            height=700,
            xaxis=dict(title=dict(text="Producer Type", standoff=20), side="top",
                       categoryorder="array", categoryarray=p_labels,
                       showgrid=True, gridcolor="#F0F0F0", tickfont=dict(size=12, color=DKGREY)),
            yaxis=dict(categoryorder="array", categoryarray=list(reversed(d_labels)),
                       showgrid=True, gridcolor="#F0F0F0", tickfont=dict(size=11, color=DKGREY)),
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Lato, Arial, sans-serif", color=DKGREY),
            legend=dict(orientation="h", yanchor="bottom", y=-0.14, xanchor="center", x=0.5),
            hovermode="closest", margin=dict(l=210, r=80, t=130, b=80),
        )
        _save_json_to(fig, target / "evidence_gap_map.json")

    print(f"[step16] Human figures done: {len(list(target.glob('*.json')))} files")


def _compare_figures_all(llm_df: pd.DataFrame, human_df: pd.DataFrame, out_dir: Path) -> None:
    """Generate LLM-vs-Human comparison Plotly figures to out_dir/interactive/compare/."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return

    h = _normalize_human_df(human_df)
    n_llm = len(llm_df)
    n_hum = len(h)
    target = out_dir / "interactive" / "compare"
    target.mkdir(parents=True, exist_ok=True)
    print(f"[step16] Generating compare figures (LLM n={n_llm}, Human n={n_hum}) -> {target.name}")

    def vcounts(df: pd.DataFrame, col: str, multi: bool = True) -> pd.Series:
        if col not in df.columns:
            return pd.Series(dtype=int)
        items = _split_h(df[col]) if multi else [
            v.strip() for v in df[col].dropna().astype(str) if v.strip().lower() not in _SKIP_H]
        return pd.Series(items).value_counts() if items else pd.Series(dtype=int)

    def to_pct(s: pd.Series) -> pd.Series:
        """Convert counts to % of total (rounded to 1 dp)."""
        total = s.sum()
        return (s / total * 100).round(1) if total > 0 else s * 0.0

    def grouped_h_bar_pct(llm_c: pd.Series, hum_c: pd.Series, title: str, hl: int = 380) -> "go.Figure":
        llm_p = to_pct(llm_c)
        hum_p = to_pct(hum_c)
        all_cats = sorted(set(llm_c.index.tolist()) | set(hum_c.index.tolist()),
                          key=lambda c: -(hum_p.get(c, 0) + llm_p.get(c, 0)))[:15]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name=f"Human (n={n_hum:,})", y=all_cats,
            x=[round(float(hum_p.get(c, 0)), 1) for c in all_cats],
            orientation="h", marker_color=HUMAN_COLOR,
            hovertemplate="<b>%{y}</b><br>Human: %{x:.1f}%<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            name=f"LLM (n={n_llm:,})", y=all_cats,
            x=[round(float(llm_p.get(c, 0)), 1) for c in all_cats],
            orientation="h", marker_color=TEAL,
            hovertemplate="<b>%{y}</b><br>LLM: %{x:.1f}%<extra></extra>",
        ))
        fig.update_layout(
            title=dict(text=f"<b>{title}</b><br><sup>Human (n={n_hum:,}) vs LLM (n={n_llm:,}) — % of studies in each corpus</sup>",
                       x=0.5, xanchor="center", font=dict(size=14)),
            barmode="group", height=hl,
            xaxis_title="% of Studies",
            xaxis=dict(ticksuffix="%"),
            yaxis=dict(autorange="reversed"),
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Lato, Arial, sans-serif", size=11, color=DKGREY),
            legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5),
            margin=dict(l=200, r=40, t=110, b=90),
        )
        return fig

    # ── Grouped bar comparisons (%) ───────────────────────────────────────────
    for llm_col, hum_col, fig_name, title, hl in [
        ("producer_type_value",              "producer_type_value",              "producer_type",
         "Producer Types — Human vs LLM", 400),
        ("methodological_approach_value",    "methodological_approach_value",    "methodology",
         "Methodology — Human vs LLM", 420),
        ("marginalized_subpopulations_value","marginalized_subpopulations_value","equity",
         "Equity & Inclusion — Human vs LLM", 400),
        ("process_outcome_domains_value",    "process_outcome_domains_value",    "domain_type",
         "Process/Outcome Domains — Human vs LLM", 500),
    ]:
        lc = vcounts(llm_df, llm_col)
        hc = vcounts(h,      hum_col)
        if not lc.empty or not hc.empty:
            _save_json_to(grouped_h_bar_pct(lc, hc, title, hl=hl), target / f"{fig_name}.json")

    # ── Temporal ──────────────────────────────────────────────────────────────
    def yr_counts(df: pd.DataFrame, col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series(dtype=int)
        years = pd.to_numeric(df[col], errors="coerce").dropna().astype(int)
        years = years[(years >= 2005) & (years <= 2027)]
        return years.value_counts().sort_index() if not years.empty else pd.Series(dtype=int)

    ly = yr_counts(llm_df, "publication_year_value")
    hy = yr_counts(h,      "publication_year_value")
    all_yrs = sorted(set(ly.index.tolist()) | set(hy.index.tolist()))
    if all_yrs:
        ly_p = to_pct(ly); hy_p = to_pct(hy)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name=f"Human (n={n_hum:,})", x=all_yrs,
            y=[round(float(hy_p.get(y, 0)), 1) for y in all_yrs], marker_color=HUMAN_COLOR,
            hovertemplate="<b>%{x}</b><br>Human: %{y:.1f}%<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            name=f"LLM (n={n_llm:,})", x=all_yrs,
            y=[round(float(ly_p.get(y, 0)), 1) for y in all_yrs], marker_color=TEAL,
            hovertemplate="<b>%{x}</b><br>LLM: %{y:.1f}%<extra></extra>",
        ))
        fig.update_layout(
            title=dict(text=f"<b>Publications Over Time — Human vs LLM</b><br><sup>Human (n={n_hum:,}) vs LLM (n={n_llm:,}) — % of studies per year</sup>",
                       x=0.5, xanchor="center", font=dict(size=14)),
            barmode="group", height=400,
            xaxis_title="Publication Year",
            yaxis_title="% of Studies", yaxis=dict(ticksuffix="%"),
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Lato, Arial, sans-serif", size=11, color=DKGREY),
            legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5),
            margin=dict(l=60, r=40, t=110, b=90),
        )
        _save_json_to(fig, target / "temporal_trends.json")

    # ── Domain heatmap (side-by-side subplots) ────────────────────────────────
    dom_col  = "process_outcome_domains_value"
    prod_col = "producer_type_value"
    if dom_col in llm_df.columns and prod_col in llm_df.columns and dom_col in h.columns:
        def heatmap_ct(df: pd.DataFrame) -> pd.DataFrame:
            rws: List[dict] = []
            for _, row in df.iterrows():
                doms  = [d.strip() for d in str(row.get(dom_col, "")).split(";")
                         if d.strip() and d.strip().lower() not in _SKIP_H]
                prods = [p.strip() for p in str(row.get(prod_col, "")).split(";")
                         if p.strip() and p.strip().lower() not in _SKIP_H]
                for d in doms:
                    for p in prods:
                        rws.append({"domain": d, "producer": p})
            if not rws:
                return pd.DataFrame()
            return pd.DataFrame(rws).pivot_table(
                index="domain", columns="producer", aggfunc="size", fill_value=0)

        ct_l = heatmap_ct(llm_df)
        ct_h = heatmap_ct(h)

        if not ct_l.empty or not ct_h.empty:
            all_d = sorted(set(ct_l.index.tolist() if not ct_l.empty else []) |
                           set(ct_h.index.tolist() if not ct_h.empty else []))
            all_p = sorted(set(ct_l.columns.tolist() if not ct_l.empty else []) |
                           set(ct_h.columns.tolist() if not ct_h.empty else []))

            def _pct(ct: pd.DataFrame, d: str, p: str, n_tot: int) -> float:
                if ct.empty or d not in ct.index or p not in ct.columns or n_tot == 0:
                    return 0.0
                return float(ct.loc[d, p]) / n_tot * 100

            # Difference matrix: LLM% − Human%
            diff_z = [[round(_pct(ct_l, d, p, n_llm) - _pct(ct_h, d, p, n_hum), 2)
                        for p in all_p] for d in all_d]
            # customdata for rich hover: [[llm_pct, hum_pct], ...]
            cdata = [[[round(_pct(ct_l, d, p, n_llm), 1), round(_pct(ct_h, d, p, n_hum), 1)]
                       for p in all_p] for d in all_d]
            abs_max = max((abs(v) for row in diff_z for v in row), default=1.0)
            diff_cs = [[0.0, HUMAN_COLOR], [0.5, "#FFFFFF"], [1.0, TEAL]]

            fig_diff = go.Figure(go.Heatmap(
                z=diff_z, x=all_p, y=all_d,
                colorscale=diff_cs, zmid=0, zmin=-abs_max, zmax=abs_max,
                colorbar=dict(title="LLM%−Human%", ticksuffix="%"),
                customdata=cdata,
                hovertemplate=(
                    "<b>%{y}</b><br>%{x}<br>"
                    "LLM: %{customdata[0]:.1f}%<br>"
                    "Human: %{customdata[1]:.1f}%<br>"
                    "Diff: %{z:+.1f}%<extra></extra>"
                ),
            ))
            fig_diff.update_layout(
                title=dict(
                    text=(f"<b>Coverage Difference: LLM minus Human</b><br>"
                          f"<sup>Teal = LLM covers more · amber = Human covers more"
                          f" · hover for exact % (LLM n={n_llm:,}, Human n={n_hum:,})</sup>"),
                    x=0.5, xanchor="center", font=dict(size=14)),
                height=max(500, len(all_d) * 35 + 220),
                plot_bgcolor="white", paper_bgcolor="white",
                font=dict(family="Lato, Arial, sans-serif", size=10, color=DKGREY),
                margin=dict(l=210, r=80, t=140, b=60),
            )
            fig_diff.update_xaxes(tickangle=-30, side="top")
            fig_diff.update_yaxes(autorange="reversed")
            _save_json_to(fig_diff, target / "domain_heatmap.json")

    # ── Geographic map (two choropleth subplots) ──────────────────────────────
    cc_l = _geo_counts(llm_df, "country_region_value")
    cc_h = _geo_counts(h,      "country_region_value")
    if cc_l or cc_h:
        tot_h = sum(cc_h.values()) or 1
        tot_l = sum(cc_l.values()) or 1
        all_geo_countries = set(cc_l.keys()) | set(cc_h.keys())
        max_geo_pct = max(
            max((v / tot_l * 100 for v in cc_l.values()), default=1.0),
            max((v / tot_h * 100 for v in cc_h.values()), default=1.0),
            1.0,
        )
        geo_sref = max_geo_pct / (50 ** 2)

        fig_gmap = go.Figure()
        # LLM bubbles first (background) — teal
        if cc_l:
            fig_gmap.add_trace(go.Scattergeo(
                locations=list(cc_l.keys()),
                locationmode="country names",
                mode="markers",
                name=f"LLM (n={n_llm:,})",
                marker=dict(
                    size=[v / tot_l * 100 for v in cc_l.values()],
                    sizemode="area", sizeref=geo_sref, sizemin=5,
                    color=TEAL, opacity=0.60,
                    line=dict(color="white", width=0.5),
                ),
                text=[f"LLM: {v/tot_l*100:.1f}% (n={v})" for v in cc_l.values()],
                hovertemplate="<b>%{location}</b><br>%{text}<extra></extra>",
            ))
        # Human bubbles on top — amber
        if cc_h:
            fig_gmap.add_trace(go.Scattergeo(
                locations=list(cc_h.keys()),
                locationmode="country names",
                mode="markers",
                name=f"Human (n={n_hum:,})",
                marker=dict(
                    size=[v / tot_h * 100 for v in cc_h.values()],
                    sizemode="area", sizeref=geo_sref, sizemin=5,
                    color=HUMAN_COLOR, opacity=0.85,
                    line=dict(color="white", width=1),
                ),
                text=[f"Human: {v/tot_h*100:.1f}% (n={v})" for v in cc_h.values()],
                hovertemplate="<b>%{location}</b><br>%{text}<extra></extra>",
            ))
        fig_gmap.update_geos(
            showframe=False, showcoastlines=True, projection_type="natural earth",
            showland=True, landcolor="#F5F5F5", showocean=True, oceancolor="#EBF4FB",
            showcountries=True, countrycolor="#D0D0D0",
        )
        fig_gmap.update_layout(
            title=dict(
                text=(f"<b>Geographic Distribution — Human vs LLM</b><br>"
                      f"<sup>Bubble size = % of corpus on the same scale · "
                      f"amber = Human (n={n_hum:,}) · teal = LLM (n={n_llm:,})</sup>"),
                x=0.5, xanchor="center", font=dict(size=14)),
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
            height=460, paper_bgcolor="white",
            font=dict(family="Lato, Arial, sans-serif", size=11, color=DKGREY),
            margin=dict(l=0, r=0, t=100, b=40),
        )
        _save_json_to(fig_gmap, target / "geographic_map.json")

        # Bar compare (% of each corpus)
        cc_l_s = pd.Series(cc_l); cc_h_s = pd.Series(cc_h)
        cc_l_p = (cc_l_s / cc_l_s.sum() * 100).round(1) if cc_l_s.sum() > 0 else cc_l_s
        cc_h_p = (cc_h_s / cc_h_s.sum() * 100).round(1) if cc_h_s.sum() > 0 else cc_h_s
        all_c = sorted(set(cc_l.keys()) | set(cc_h.keys()),
                       key=lambda c: -(cc_l_p.get(c, 0.0) + cc_h_p.get(c, 0.0)))[:30]
        bar_fig = go.Figure()
        bar_fig.add_trace(go.Bar(
            name=f"Human (n={n_hum:,})", y=all_c,
            x=[round(float(cc_h_p.get(c, 0.0)), 1) for c in all_c],
            orientation="h", marker_color=HUMAN_COLOR,
            hovertemplate="<b>%{y}</b><br>Human: %{x:.1f}%<extra></extra>",
        ))
        bar_fig.add_trace(go.Bar(
            name=f"LLM (n={n_llm:,})", y=all_c,
            x=[round(float(cc_l_p.get(c, 0.0)), 1) for c in all_c],
            orientation="h", marker_color=TEAL,
            hovertemplate="<b>%{y}</b><br>LLM: %{x:.1f}%<extra></extra>",
        ))
        bar_fig.update_layout(
            title=dict(text=f"<b>Studies by Country — Human vs LLM</b><br><sup>Human (n={n_hum:,}) vs LLM (n={n_llm:,}) — % of studies in each corpus</sup>",
                       x=0.5, xanchor="center", font=dict(size=14)),
            barmode="group", height=max(500, len(all_c) * 32 + 160),
            xaxis_title="% of Studies", xaxis=dict(ticksuffix="%"),
            yaxis=dict(autorange="reversed"),
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Lato, Arial, sans-serif", size=11, color=DKGREY),
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
            margin=dict(l=160, r=40, t=110, b=80),
        )
        _save_json_to(bar_fig, target / "geographic_bar.json")

    # ── Evidence Gap Map (side-by-side bubbles per cell) ─────────────────────
    dom_col  = "process_outcome_domains_value"
    prod_col = "producer_type_value"
    if (dom_col in llm_df.columns and prod_col in llm_df.columns
            and dom_col in h.columns and prod_col in h.columns):
        cc_l_egm = _cell_counts_egm(llm_df)
        cc_h_egm = _cell_counts_egm(h)
        max_pct_egm = max(
            max((v / n_llm * 100 for v in cc_l_egm.values()), default=1.0),
            max((v / n_hum * 100 for v in cc_h_egm.values()), default=1.0),
            1.0,
        )
        sref_egm = max_pct_egm / (36 ** 2)

        d_labels = [DOMAIN_LABELS.get(d, d) for d in ALL_DOMAINS]
        p_labels  = [PRODUCER_LABELS.get(p, p) for p in PRODUCER_TYPES]
        # Numeric x positions so LLM and Human can be offset within each cell
        p_pos = {p: i for i, p in enumerate(PRODUCER_TYPES)}
        OFFSET = 0.22
        BLUE_LIGHT  = "#90CAF9"   # lighter blue  — Human process
        GREEN_LIGHT = "#A5D6A7"   # lighter green — Human outcome

        traces_cmp = []
        for d_set, color_llm, color_hum, domain_lbl in [
            (PROCESS_DOMAINS, BLUE,  BLUE_LIGHT,  "Process"),
            (OUTCOME_DOMAINS, GREEN, GREEN_LIGHT, "Outcome"),
        ]:
            xs_l, ys_l, szs_l, txts_l = [], [], [], []
            xs_h, ys_h, szs_h, txts_h = [], [], [], []
            for d in d_set:
                dl = DOMAIN_LABELS.get(d, d)
                for p in PRODUCER_TYPES:
                    pl = PRODUCER_LABELS.get(p, p)
                    pos = p_pos[p]
                    nv_l = cc_l_egm.get((d, p), 0)
                    nv_h = cc_h_egm.get((d, p), 0)
                    pct_h = round(nv_h / n_hum * 100, 1)
                    pct_l = round(nv_l / n_llm * 100, 1)
                    if nv_h > 0:
                        xs_h.append(pos - OFFSET); ys_h.append(dl)
                        szs_h.append(pct_h)
                        txts_h.append(f"<b>{dl}</b><br>{pl}<br>Human: {pct_h}% (n={nv_h})")
                    if nv_l > 0:
                        xs_l.append(pos + OFFSET); ys_l.append(dl)
                        szs_l.append(pct_l)
                        txts_l.append(f"<b>{dl}</b><br>{pl}<br>LLM: {pct_l}% (n={nv_l})")
            if xs_h:
                traces_cmp.append(go.Scatter(
                    x=xs_h, y=ys_h, mode="markers+text",
                    name=f"Human — {domain_lbl}",
                    marker=dict(size=szs_h, sizemode="area", sizeref=sref_egm, sizemin=9,
                                color=color_hum, opacity=0.85, line=dict(color="white", width=1.5)),
                    text=[f"{s:.1f}%" for s in szs_h],
                    textposition="middle center", textfont=dict(size=9, color=DKGREY),
                    hovertext=txts_h, hoverinfo="text", showlegend=True,
                ))
            if xs_l:
                traces_cmp.append(go.Scatter(
                    x=xs_l, y=ys_l, mode="markers+text",
                    name=f"LLM — {domain_lbl}",
                    marker=dict(size=szs_l, sizemode="area", sizeref=sref_egm, sizemin=9,
                                color=color_llm, opacity=0.80, line=dict(color="white", width=1)),
                    text=[f"{s:.1f}%" for s in szs_l],
                    textposition="middle center", textfont=dict(size=9, color="white"),
                    hovertext=txts_l, hoverinfo="text", showlegend=True,
                ))

        # Gaps (neither LLM nor Human)
        all_egm = set(cc_l_egm.keys()) | set(cc_h_egm.keys())
        gx, gy, gt = [], [], []
        for d in ALL_DOMAINS:
            dl = DOMAIN_LABELS.get(d, d)
            for p in PRODUCER_TYPES:
                pl = PRODUCER_LABELS.get(p, p)
                if (d, p) not in all_egm:
                    gx.append(p_pos[p]); gy.append(dl)
                    gt.append(f"<b>{dl}</b><br>{pl}<br><i>No studies in either</i>")
        if gx:
            traces_cmp.append(go.Scatter(x=gx, y=gy, mode="markers", name="Gap (both)",
                marker=dict(size=10, color="#E0E0E0", line=dict(color="#BDBDBD", width=1)),
                hovertext=gt, hoverinfo="text", showlegend=True))

        fig = go.Figure(data=traces_cmp)
        _n_out = len(OUTCOME_DOMAINS)
        fig.add_shape(type="line", xref="paper", yref="y", x0=0, x1=1,
            y0=_n_out - 0.5, y1=_n_out - 0.5,
            line=dict(color="#9E9E9E", width=1.5, dash="dot"))
        fig.update_layout(
            title=dict(
                text=(f"<b>Evidence Gap Map — LLM vs Human</b><br>"
                      f"<sup>Left bubble = Human (n={n_hum:,}, lighter) · right = LLM (n={n_llm:,}, darker) · blue = process · green = outcome</sup>"),
                x=0.5, xanchor="center", font=dict(size=14)),
            height=740,
            xaxis=dict(
                title=dict(text="Producer Type", standoff=20), side="top",
                tickmode="array",
                tickvals=list(range(len(PRODUCER_TYPES))),
                ticktext=p_labels,
                range=[-0.6, len(PRODUCER_TYPES) - 0.4],
                showgrid=True, gridcolor="#F0F0F0", tickfont=dict(size=12, color=DKGREY)),
            yaxis=dict(categoryorder="array", categoryarray=list(reversed(d_labels)),
                       showgrid=True, gridcolor="#F0F0F0", tickfont=dict(size=11, color=DKGREY)),
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Lato, Arial, sans-serif", color=DKGREY),
            legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5),
            hovermode="closest", margin=dict(l=210, r=80, t=140, b=110),
        )
        _save_json_to(fig, target / "evidence_gap_map.json")

    print(f"[step16] Compare figures done: {len(list(target.glob('*.json')))} files")


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
        # Static matplotlib PNGs (LLM corpus) intentionally skipped —
        # use interactive/human/ Plotly PNGs for all deliverables.

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

        try:
            print("[step16] Producing LLM vs human comparison figure...")
            _llm_vs_human_comparison(df, out_root, out_dir)
            figures_saved.append("llm_vs_human.png")
        except Exception as e:
            print(f"[step16] WARNING: llm_vs_human.png failed — {type(e).__name__}: {e}")

        # Human-only and comparison interactive figures
        _human_path = out_root / "step15" / "step15_human.csv"
        if _human_path.exists():
            try:
                _human_df = pd.read_csv(_human_path, dtype=str).fillna("")
                print(f"[step16] Producing human interactive figures (n={len(_human_df)})...")
                _human_figures_all(_human_df, out_dir)
                figures_saved.append("human/")
            except Exception as e:
                print(f"[step16] WARNING: human figures failed — {type(e).__name__}: {e}")
            try:
                print("[step16] Producing LLM-vs-human compare interactive figures...")
                _compare_figures_all(df, _human_df, out_dir)
                figures_saved.append("compare/")
            except Exception as e:
                print(f"[step16] WARNING: compare figures failed — {type(e).__name__}: {e}")
        else:
            print("[step16] step15_human.csv not found — skipping human/compare interactive figures")

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
