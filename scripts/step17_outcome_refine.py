#!/usr/bin/env python3
"""
step17_outcome_refine.py

Step 17: Refine outcome-domain scope ahead of indicator/rating design.

This step does NOT score or rate anything, and is not tied to any particular
source tool — that choice (and the rating approach) will be defined in the
forthcoming codebook. All this step does is:

  1. Reproduce the step16 process-vs-outcome Venn (facet 1, unchanged).
  2. Add a second facet (facet 2) that zooms into the outcome-tagged studies
     (both + outcome_only = 97 + 16 = 113) and splits them three ways:
       - Yields & Productivity
       - Income & Assets
       - all other outcome domains (livelihoods, wellbeing, risk_reduction,
         resilience_adaptive_capacity)
     showing how much these three groups overlap.
  3. Export a full per-study CSV: one row per human-coded study, one
     True/False column per process domain and per outcome domain.
  4. Export a subset of that CSV limited to studies tagged with Yields &
     Productivity and/or Income & Assets — the candidate population for the
     indicator/rating work (rows removed only when BOTH are False; all
     domain columns kept for reference).

Inputs:
  - outputs/step15/step15_human.csv

Outputs (under outputs/step17/):
  - domain_venn_dual.png / .json     — two-facet Venn (facet 1 + facet 2)
  - outcome_focus_counts.csv         — facet-2 region counts (Y/I/O overlaps)
  - studies_domain_flags.csv         — all studies x all domain flags
  - studies_domain_flags_yield_income_focus.csv — subset (Y or I True)
  - step17_meta.json

Run:
  conda run -n ilri01 python scripts/step17_outcome_refine.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from step16_map_visualise import (  # noqa: E402
    ALL_DOMAINS,
    DOMAIN_LABELS,
    OUTCOME_DOMAINS,
    BLUE,
    GREEN,
    ORANGE,
    GREY,
    DKGREY,
    _domain_tokens,
    _process_outcome_label,
    _process_outcome_classify,
    _normalize_human_df,
)

OUT_ROOT = _HERE / "outputs"
STEP17_DIR = OUT_ROOT / "step17"

FOCUS_OUTCOMES = ["yields_productivity", "income_assets"]
OTHER_OUTCOMES = [d for d in OUTCOME_DOMAINS if d not in FOCUS_OUTCOMES]
DOM_COL = "process_outcome_domains_value"


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _load_human() -> pd.DataFrame:
    path = OUT_ROOT / "step15" / "step15_human.csv"
    if not path.exists():
        raise FileNotFoundError(f"step15_human.csv not found at {path}")
    df = pd.read_csv(path, dtype=str).fillna("")
    return _normalize_human_df(df)


def _domain_flags_df(h: pd.DataFrame) -> pd.DataFrame:
    """One row per study: doi, title, a bool column per domain code,
    domain_type (process_only/outcome_only/both/neither), and the raw
    process_outcome_domains string for traceability."""
    recs = []
    for _, row in h.iterrows():
        raw = row.get(DOM_COL, "")
        toks = set(_domain_tokens(raw))
        rec = {"doi": row.get("doi", ""), "title": row.get("title", "")}
        for d in ALL_DOMAINS:
            rec[d] = d in toks
        rec["domain_type"] = _process_outcome_label(raw)
        rec["process_outcome_domains_raw"] = raw
        recs.append(rec)
    return pd.DataFrame(recs)


def _three_set_counts(flags: pd.DataFrame) -> dict:
    """Y/I/O overlap counts within the outcome-tagged subset (both + outcome_only)."""
    scope = flags[flags["domain_type"].isin(["both", "outcome_only"])]
    Y = scope["yields_productivity"]
    I = scope["income_assets"]
    O = scope[OTHER_OUTCOMES].any(axis=1)
    counts = {
        "total_outcome_tagged": int(len(scope)),
        "yields_only":              int((Y & ~I & ~O).sum()),
        "income_only":              int((~Y & I & ~O).sum()),
        "other_only":               int((~Y & ~I & O).sum()),
        "yields_and_income":        int((Y & I & ~O).sum()),
        "yields_and_other":         int((Y & ~I & O).sum()),
        "income_and_other":         int((~Y & I & O).sum()),
        "yields_and_income_and_other": int((Y & I & O).sum()),
        "none_of_the_three":        int((~Y & ~I & ~O).sum()),  # sanity check, should be 0
    }
    return counts


def _dual_venn_figure(counts2: dict, counts3: dict, yield_income_total: int, n_total: int):
    """Two-facet Venn: facet 1 = process vs outcome (step16, unchanged),
    facet 2 = Yields vs Income vs Other-outcomes, among outcome-tagged studies."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from shapely.geometry import Point

    total2 = sum(counts2[k] for k in ("process_only", "outcome_only", "both", "neither"))

    def pct2(v):
        return f"{v / total2 * 100:.1f}%" if total2 else "0%"

    total3 = counts3["total_outcome_tagged"]

    def pct3(v):
        return f"{v / total3 * 100:.1f}%" if total3 else "0%"

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"Process vs. Outcome Domains<br><sup>n={total2:,}</sup>",
            f"Outcome Focus: Yields & Productivity vs. Income & Assets vs. Other<br>"
            f"<sup>outcome-tagged studies only (both + outcome-only), n={total3:,}</sup>",
        ),
    )

    # ---- Facet 1: replicate step16's two-circle schematic Venn --------------
    p_only, o_only, both, neither = (
        counts2["process_only"], counts2["outcome_only"], counts2["both"], counts2["neither"],
    )
    fig.add_shape(type="circle", xref="x", yref="y", row=1, col=1,
                  x0=0.0, x1=2.0, y0=0.0, y1=2.0,
                  fillcolor=BLUE, opacity=0.40, line=dict(color=BLUE, width=2))
    fig.add_shape(type="circle", xref="x", yref="y", row=1, col=1,
                  x0=1.15, x1=3.15, y0=0.0, y1=2.0,
                  fillcolor=GREEN, opacity=0.40, line=dict(color=GREEN, width=2))
    fig.add_annotation(x=0.55, y=1.0, row=1, col=1, showarrow=False,
                        text=f"<b>Process only</b><br>{p_only:,}<br>({pct2(p_only)})",
                        font=dict(size=12, color=DKGREY))
    fig.add_annotation(x=2.6, y=1.0, row=1, col=1, showarrow=False,
                        text=f"<b>Outcome only</b><br>{o_only:,}<br>({pct2(o_only)})",
                        font=dict(size=12, color=DKGREY))
    fig.add_annotation(x=1.575, y=1.0, row=1, col=1, showarrow=False,
                        text=f"<b>Both</b><br>{both:,}<br>({pct2(both)})",
                        font=dict(size=12, color="white"))
    fig.add_annotation(x=0.55, y=2.18, row=1, col=1, showarrow=False,
                        text="<b>PROCESS DOMAINS</b>", font=dict(size=10, color=BLUE))
    fig.add_annotation(x=2.6, y=2.18, row=1, col=1, showarrow=False,
                        text="<b>OUTCOME DOMAINS</b>", font=dict(size=10, color=GREEN))
    fig.add_annotation(x=1.575, y=-0.35, row=1, col=1, showarrow=False,
                        text=f"Neither tagged: {neither:,} ({pct2(neither)})",
                        font=dict(size=10, color=GREY))
    fig.update_xaxes(visible=False, range=[-0.3, 3.45], row=1, col=1)
    fig.update_yaxes(visible=False, range=[-0.7, 2.55], scaleanchor="x", scaleratio=1, row=1, col=1)

    # ---- Facet 2: three-circle Venn (Yields / Income / Other-outcomes) -----
    R = 1.3
    centers = {"Y": (0.0, 1.05), "I": (-1.05, -0.55), "O": (1.05, -0.55)}
    circles = {k: Point(v).buffer(R, resolution=128) for k, v in centers.items()}

    def rep(poly):
        return (None, None) if poly.is_empty else (poly.representative_point().x,
                                                     poly.representative_point().y)

    regions = {
        "yields_only":                 circles["Y"].difference(circles["I"]).difference(circles["O"]),
        "income_only":                 circles["I"].difference(circles["Y"]).difference(circles["O"]),
        "other_only":                  circles["O"].difference(circles["Y"]).difference(circles["I"]),
        "yields_and_income":           circles["Y"].intersection(circles["I"]).difference(circles["O"]),
        "yields_and_other":            circles["Y"].intersection(circles["O"]).difference(circles["I"]),
        "income_and_other":            circles["I"].intersection(circles["O"]).difference(circles["Y"]),
        "yields_and_income_and_other": circles["Y"].intersection(circles["I"]).intersection(circles["O"]),
    }
    region_labels = {
        "yields_only":                 "Yields &<br>Productivity only",
        "income_only":                 "Income &<br>Assets only",
        "other_only":                  "Other outcome<br>domains only",
        "yields_and_income":           "Yields +<br>Income",
        "yields_and_other":            "Yields +<br>Other",
        "income_and_other":            "Income +<br>Other",
        "yields_and_income_and_other": "All three",
    }

    for key, poly in ((k, circles[k]) for k in ("Y", "I", "O")):
        color = {"Y": BLUE, "I": GREEN, "O": ORANGE}[key]
        cx, cy = centers[key]
        fig.add_shape(type="circle", xref="x2", yref="y2", row=1, col=2,
                      x0=cx - R, x1=cx + R, y0=cy - R, y1=cy + R,
                      fillcolor=color, opacity=0.35, line=dict(color=color, width=2))

    for key, poly in regions.items():
        x, y = rep(poly)
        n = counts3[key]
        if x is None:
            continue
        text_color = "white" if key == "yields_and_income_and_other" else DKGREY
        fig.add_annotation(x=x, y=y, row=1, col=2, showarrow=False,
                            text=f"<b>{region_labels[key]}</b><br>{n:,}<br>({pct3(n)})",
                            font=dict(size=10.5, color=text_color))

    fig.add_annotation(x=0.0, y=2.55, row=1, col=2, showarrow=False,
                        text="<b>YIELDS & PRODUCTIVITY</b>", font=dict(size=10, color=BLUE))
    fig.add_annotation(x=-1.85, y=-1.75, row=1, col=2, showarrow=False,
                        text="<b>INCOME & ASSETS</b>", font=dict(size=10, color=GREEN))
    fig.add_annotation(x=1.85, y=-1.75, row=1, col=2, showarrow=False,
                        text="<b>OTHER OUTCOME<br>DOMAINS</b>", font=dict(size=10, color=ORANGE))
    none3 = counts3["none_of_the_three"]
    if none3:
        fig.add_annotation(x=0.0, y=-2.65, row=1, col=2, showarrow=False,
                            text=f"Outcome-tagged but none of the three matched "
                                 f"(check aliases): {none3:,}",
                            font=dict(size=9, color=GREY))

    pct_focus = f"{yield_income_total / n_total * 100:.1f}%" if n_total else "0%"
    fig.add_annotation(x=0.0, y=-2.35, row=1, col=2, showarrow=False,
                        text=(f"<b>Studies including Yields & Productivity and/or "
                              f"Income & Assets: {yield_income_total:,} of {n_total:,} "
                              f"({pct_focus})</b>"),
                        font=dict(size=11, color=DKGREY))

    fig.update_xaxes(visible=False, range=[-2.6, 2.6], row=1, col=2)
    fig.update_yaxes(visible=False, range=[-3.15, 2.9], scaleanchor="x2", scaleratio=1, row=1, col=2)

    fig.update_layout(
        title=dict(
            text="<b>Domain Coverage — Human-coded Primary Studies</b>",
            x=0.5, xanchor="center", font=dict(size=16),
        ),
        height=660, width=1500,
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=40, r=40, t=140, b=60),
        font=dict(family="Lato, Arial, sans-serif", color=DKGREY),
        showlegend=False,
    )
    return fig


def main() -> dict:
    STEP17_DIR.mkdir(parents=True, exist_ok=True)
    h = _load_human()
    print(f"[step17] Loaded {len(h)} human-coded studies")

    flags = _domain_flags_df(h)
    counts2 = _process_outcome_classify(h[DOM_COL].tolist())
    counts3 = _three_set_counts(flags)
    print(f"[step17] Process/outcome counts: {counts2}")
    print(f"[step17] Yields/Income/Other counts: {counts3}")

    # ---- Full per-study domain-flag CSV ----------------------------------
    label_cols = [DOMAIN_LABELS[d] for d in ALL_DOMAINS]
    full_df = flags.rename(columns=DOMAIN_LABELS)
    full_cols = ["doi", "title", "domain_type"] + label_cols + ["process_outcome_domains_raw"]
    full_df = full_df[full_cols]
    full_df.to_csv(STEP17_DIR / "studies_domain_flags.csv", index=False)
    print(f"[step17] -> studies_domain_flags.csv ({len(full_df)} studies)")

    # ---- Subset: studies tagged with Yields and/or Income ----------------
    # Kept if EITHER Yields & Productivity OR Income & Assets is True;
    # removed only when both are False. All domain columns are kept.
    y_label = DOMAIN_LABELS["yields_productivity"]
    i_label = DOMAIN_LABELS["income_assets"]
    subset_mask = full_df[y_label] | full_df[i_label]
    subset_df = full_df[subset_mask].copy()
    subset_df.to_csv(STEP17_DIR / "studies_domain_flags_yield_income_focus.csv", index=False)
    yield_income_total = len(subset_df)
    print(f"[step17] -> studies_domain_flags_yield_income_focus.csv ({yield_income_total} studies)")

    # ---- Dual-facet Venn ------------------------------------------------
    fig = _dual_venn_figure(counts2, counts3, yield_income_total, len(h))
    fig.write_json(str(STEP17_DIR / "domain_venn_dual.json"))
    print("[step17] -> domain_venn_dual.json")
    try:
        fig.write_image(str(STEP17_DIR / "domain_venn_dual.png"), scale=2, width=1500, height=660)
        print("[step17] -> domain_venn_dual.png")
    except Exception as e:
        print(f"[step17] WARNING: kaleido PNG export failed ({e})")

    pd.DataFrame([counts3]).to_csv(STEP17_DIR / "outcome_focus_counts.csv", index=False)
    print("[step17] -> outcome_focus_counts.csv")

    meta = {
        "timestamp_utc": _now_utc(),
        "source": "outputs/step15/step15_human.csv",
        "n_studies_total": len(h),
        "process_outcome_counts": counts2,
        "outcome_focus_counts": counts3,
        "yield_income_subset_n": int(len(subset_df)),
        "subset_rule": "kept if yields_productivity OR income_assets is True",
        "note": "Scoping/visualisation only — no indicator or rating logic here; "
                "that will be defined in the forthcoming codebook.",
    }
    with open(STEP17_DIR / "step17_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[step17] -> step17_meta.json")
    print("[step17] Done.")
    return meta


if __name__ == "__main__":
    main()
