#!/usr/bin/env python3
"""
step15c_saturation.py

Compute information saturation across human-coded batches and produce
a Plotly JSON figure for the methodology page.

For each cumulative batch, counts the number of unique categories in:
  - process_outcome_domains
  - methodological_approach
  - producer_type

Outputs a two-panel Plotly figure:
  1. Cumulative unique categories (lines) vs papers coded
  2. New categories added per batch (bars)

Saved to:
  scripts/outputs/step15c/saturation.json
  frontend/public/map/data/saturation.json
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
STEP15_CSV = ROOT / "scripts" / "outputs" / "step15" / "step15_human.csv"
OUT_DIR = ROOT / "scripts" / "outputs" / "step15c"
FRONTEND_DIR = ROOT / "frontend" / "public" / "map" / "data"

BATCH_ORDER = ["FT-R2a", "FT-R2b", "FT-R2c", "FT-R2d", "FT-R3"]

# Canonical codebook values per field (from CODEBOOK_FT.md).
# Anything not in this set is off-codebook free text — mapped to "other".
CANONICAL: dict[str, set[str]] = {
    "process_outcome_domains": {
        "knowledge_awareness_learning",
        "decision_making_planning",
        "uptake_adoption",
        "behavioral_change",
        "participation_coproduction",
        "institutional_governance",
        "access_information_services",
        "yields_productivity",
        "income_assets",
        "livelihoods",
        "wellbeing",
        "risk_reduction",
        "resilience_adaptive_capacity",
    },
    "methodological_approach": {
        "qualitative",
        "quantitative",
        "participatory",
        "modeling_with_empirical_validation",
        "experimental",
    },
    "producer_type": {
        "crop",
        "livestock",
        "fisheries_aquaculture",
        "agroforestry",
        "mixed",
    },
}

# Aliases: common typos / alternate spellings → canonical value
ALIASES: dict[str, dict[str, str]] = {
    "process_outcome_domains": {
        "resilience_adaptative_capacity":  "resilience_adaptive_capacity",
        "resilience_adaptation_capacity_building": "resilience_adaptive_capacity",
        "resilience":                      "resilience_adaptive_capacity",
        "adaptation_capacity":             "resilience_adaptive_capacity",
        "reduction_of_risk":               "risk_reduction",
        "adaptaation_reduction_of_risk":   "risk_reduction",
        "livelihood":                      "livelihoods",
        "livelihood_sustainability":       "livelihoods",
        "sustainable_livelihood":          "livelihoods",
        "adoptation_uptake":               "uptake_adoption",
        "decision_making_plans":           "decision_making_planning",
        "planning_dicision_making":        "decision_making_planning",
        "planning_the_decision_making":    "decision_making_planning",
        "planning_decision_making":        "decision_making_planning",
        "decision_makingplanning":         "decision_making_planning",
        "decision_making":                 "decision_making_planning",
        "acces_information_services":      "access_information_services",
        "yield_productivity":              "yields_productivity",
        "participation_coproduction i":    "participation_coproduction",
        "uptake adoption":                 "uptake_adoption",
        "food_security_nutrition":         "livelihoods",
    },
    "methodological_approach": {
        "mixed methods":                    "mixed_methods_other",  # not in canonical → other
        "mixed_method":                     "mixed_methods_other",
        "interviews":                       "qualitative",
        "semi-structured interviews":       "qualitative",
        "sureys":                           "quantitative",
        "experimentation":                  "experimental",
        "modelling with emperical validation": "modeling_with_empirical_validation",
        "modeling with emperical validation":  "modeling_with_empirical_validation",
        "emperical_validation_modeling":    "modeling_with_empirical_validation",
        "constant_comparative_methods":     "qualitative",
    },
    "producer_type": {
        "fisheries":        "fisheries_aquaculture",
        "aqua_culture":     "fisheries_aquaculture",
        "crop:livestock":   "mixed",
        "smallholder farmers": "other",
    },
}

TRACK_FIELDS = [
    ("process_outcome_domains",  "Process/outcome domains",  "#2E7D5A"),
    ("methodological_approach",  "Methodological approaches", "#E07B39"),
    ("producer_type",            "Producer types",            "#7C3AED"),
]


def split_vals(val: str) -> list[str]:
    if not val or (isinstance(val, float)):
        return []
    return [v.strip().lower() for v in re.split(r"[;,]", str(val)) if v.strip()]


def normalize(field: str, raw: str) -> str:
    """Map a raw coded value to its canonical form (or 'other' if off-codebook)."""
    val = raw.strip().lower()
    # Apply alias mapping first
    val = ALIASES.get(field, {}).get(val, val)
    # If in canonical set, return it; otherwise collapse to 'other'
    if val in CANONICAL.get(field, set()):
        return val
    return "other"


def compute_saturation(df: pd.DataFrame) -> list[dict]:
    rows = []
    seen: dict[str, set] = defaultdict(set)
    cum_papers = 0
    for batch in BATCH_ORDER:
        bdf = df[df["batch"] == batch]
        if len(bdf) == 0:
            continue
        cum_papers += len(bdf)
        row: dict = {"batch": batch, "papers": cum_papers}
        for field, _, _ in TRACK_FIELDS:
            new_cats = 0
            for v in bdf[field].dropna():
                for raw in split_vals(v):
                    cat = normalize(field, raw)
                    if cat not in seen[field]:
                        seen[field].add(cat)
                        new_cats += 1
            row[f"{field}_total"] = len(seen[field])
            row[f"{field}_new"] = new_cats
        rows.append(row)
    return rows


def build_figure(rows: list[dict]) -> dict:
    papers = [r["papers"] for r in rows]
    batches = [r["batch"] for r in rows]

    # Express cumulative total as % of max (final batch)
    finals = {
        field: rows[-1][f"{field}_total"]
        for field, _, _ in TRACK_FIELDS
    }

    traces = []

    # ── Top panel: cumulative % reached ──────────────────────────────────────
    for field, label, color in TRACK_FIELDS:
        final = finals[field]
        pcts = [r[f"{field}_total"] / final * 100 for r in rows]
        totals = [r[f"{field}_total"] for r in rows]
        traces.append({
            "type": "scatter",
            "xaxis": "x",
            "yaxis": "y",
            "x": papers,
            "y": pcts,
            "mode": "lines+markers",
            "name": label,
            "line": {"color": color, "width": 2.5},
            "marker": {"size": 7, "color": color},
            "customdata": [
                [r["batch"], t, final]
                for r, t in zip(rows, totals)
            ],
            "hovertemplate": (
                "<b>%{customdata[0]}</b><br>"
                "Papers coded: %{x}<br>"
                f"{label}: %{{customdata[1]}} of %{{customdata[2]}} (%{{y:.0f}}%)"
                "<extra></extra>"
            ),
        })

    # ── Saturation threshold line at 95% ────────────────────────────────────
    traces.append({
        "type": "scatter",
        "xaxis": "x",
        "yaxis": "y",
        "x": [0, max(papers) + 15],
        "y": [95, 95],
        "mode": "lines",
        "name": "95% saturation threshold",
        "line": {"color": "#9CA3AF", "width": 1.5, "dash": "dash"},
        "showlegend": True,
        "hoverinfo": "skip",
    })

    # ── Bottom panel: new categories per batch (stacked bars) ─────────────────
    for field, label, color in TRACK_FIELDS:
        new_vals = [r[f"{field}_new"] for r in rows]
        traces.append({
            "type": "bar",
            "xaxis": "x2",
            "yaxis": "y2",
            "x": batches,
            "y": new_vals,
            "name": label,
            "marker": {"color": color, "opacity": 0.75},
            "showlegend": False,
            "hovertemplate": (
                f"<b>%{{x}}</b> — {label}<br>"
                "New categories: %{y}"
                "<extra></extra>"
            ),
        })

    layout = {
        "grid": {"rows": 2, "columns": 1, "pattern": "independent"},
        "xaxis": {
            "title": {"text": "Cumulative papers coded", "font": {"size": 11}},
            "showgrid": True,
            "gridcolor": "#F3F4F6",
            "zeroline": False,
            "domain": [0, 1],
        },
        "yaxis": {
            "title": {"text": "% of final unique categories", "font": {"size": 11}},
            "range": [0, 108],
            "showgrid": True,
            "gridcolor": "#F3F4F6",
            "domain": [0.42, 1.0],
        },
        "xaxis2": {
            "title": {"text": "Batch", "font": {"size": 11}},
            "showgrid": False,
            "domain": [0, 1],
            "tickangle": -30,
            "tickfont": {"size": 11},
        },
        "yaxis2": {
            "title": {"text": "New categories", "font": {"size": 11}},
            "showgrid": True,
            "gridcolor": "#F3F4F6",
            "domain": [0, 0.33],
            "dtick": 1,
        },
        "legend": {
            "x": 0.55,
            "y": 0.95,
            "bgcolor": "rgba(255,255,255,0.9)",
            "bordercolor": "#E5E7EB",
            "borderwidth": 1,
            "font": {"size": 11},
        },
        "annotations": [
            {
                "xref": "paper", "yref": "paper",
                "x": 0.01, "y": 1.01,
                "text": "Cumulative unique categories (% of final)",
                "showarrow": False,
                "font": {"size": 12, "color": "#374151"},
                "xanchor": "left",
            },
            {
                "xref": "paper", "yref": "paper",
                "x": 0.01, "y": 0.27,
                "text": "New categories added per batch",
                "showarrow": False,
                "font": {"size": 12, "color": "#374151"},
                "xanchor": "left",
            },
        ],
        "margin": {"l": 60, "r": 20, "t": 30, "b": 50},
        "plot_bgcolor": "white",
        "paper_bgcolor": "white",
        "hovermode": "closest",
        "height": 520,
        "barmode": "group",
    }

    return {"data": traces, "layout": layout}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(STEP15_CSV, dtype=str)
    print(f"Loaded {len(df)} rows from step15_human.csv")

    rows = compute_saturation(df)

    print("\nSaturation by batch:")
    for r in rows:
        parts = [f"{r['batch']} ({r['papers']} papers)"]
        for field, label, _ in TRACK_FIELDS:
            parts.append(f"  {label}: {r[f'{field}_total']} (+{r[f'{field}_new']})")
        print("\n".join(parts))

    fig = build_figure(rows)

    out_path = OUT_DIR / "saturation.json"
    with open(out_path, "w") as f:
        json.dump(fig, f, separators=(",", ":"))
    print(f"\nSaved: {out_path}")

    if FRONTEND_DIR.exists():
        dest = FRONTEND_DIR / "saturation.json"
        with open(dest, "w") as f:
            json.dump(fig, f, separators=(",", ":"))
        print(f"Synced: {dest}")


if __name__ == "__main__":
    main()
