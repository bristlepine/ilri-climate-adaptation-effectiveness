#!/usr/bin/env python3
"""
step16c_crosstabs.py

Generate two cross-tabulation heatmaps for C3 (Neal Haddaway reviewer request):
  1. Domain × time period  — how research focus shifted over time
  2. Producer type × region — geographic concentration by producer type

Outputs to: scripts/outputs/step16/interactive/human/
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd
import plotly.graph_objects as go

OUT = ROOT / "scripts" / "outputs" / "step16" / "interactive" / "human"
OUT.mkdir(parents=True, exist_ok=True)

DOMAIN_LABELS = {
    "knowledge_awareness_learning":   "Knowledge & Awareness",
    "decision_making_planning":        "Decision-making & Planning",
    "uptake_adoption":                 "Uptake & Adoption",
    "behavioral_change":               "Behavioral Change",
    "participation_coproduction":      "Participation & Co-production",
    "institutional_governance":        "Institutional Governance",
    "access_information_services":     "Access to Services",
    "acces_information_services":      "Access to Services",
    "yields_productivity":             "Yields & Productivity",
    "income_assets":                   "Income & Assets",
    "livelihoods":                     "Livelihoods",
    "wellbeing":                       "Wellbeing",
    "risk_reduction":                  "Risk Reduction",
    "resilience_adaptive_capacity":    "Resilience & Adaptive Capacity",
    "resilience_adaptative_capacity":  "Resilience & Adaptive Capacity",
}

PROCESS_DOMAINS = {
    "Knowledge & Awareness", "Decision-making & Planning", "Uptake & Adoption",
    "Behavioral Change", "Participation & Co-production", "Institutional Governance",
    "Access to Services",
}

PRODUCER_LABELS = {
    "crop":                  "Crop farmers",
    "livestock":             "Livestock",
    "fisheries_aquaculture": "Fisheries/Aquaculture",
    "fisheries":             "Fisheries/Aquaculture",
    "agroforestry":          "Agroforestry",
    "mixed":                 "Mixed/General",
    "undefined":             "Mixed/General",
}

COUNTRY_REGION = {
    # Sub-Saharan Africa
    "ghana": "Sub-Saharan Africa", "kenya": "Sub-Saharan Africa",
    "ethiopia": "Sub-Saharan Africa", "tanzania": "Sub-Saharan Africa",
    "nigeria": "Sub-Saharan Africa", "uganda": "Sub-Saharan Africa",
    "zimbabwe": "Sub-Saharan Africa", "zambia": "Sub-Saharan Africa",
    "malawi": "Sub-Saharan Africa", "mozambique": "Sub-Saharan Africa",
    "botswana": "Sub-Saharan Africa", "senegal": "Sub-Saharan Africa",
    "mali": "Sub-Saharan Africa", "burkina faso": "Sub-Saharan Africa",
    "benin": "Sub-Saharan Africa", "eritrea": "Sub-Saharan Africa",
    "angola": "Sub-Saharan Africa", "rwanda": "Sub-Saharan Africa",
    "south africa": "Sub-Saharan Africa",
    # South Asia
    "india": "South Asia", "bangladesh": "South Asia",
    "pakistan": "South Asia", "nepal": "South Asia",
    "sri lanka": "South Asia", "afghanistan": "South Asia",
    # East & SE Asia
    "china": "East & SE Asia", "vietnam": "East & SE Asia",
    "indonesia": "East & SE Asia", "philippines": "East & SE Asia",
    "cambodia": "East & SE Asia", "thailand": "East & SE Asia",
    "myanmar": "East & SE Asia",
    # Latin America
    "brazil": "Latin America", "nicaragua": "Latin America",
    "peru": "Latin America", "bolivia": "Latin America",
    "mexico": "Latin America", "colombia": "Latin America",
    "ecuador": "Latin America", "haiti": "Latin America",
    # MENA
    "iran": "MENA", "morocco": "MENA", "tunisia": "MENA",
    "egypt": "MENA", "iraq": "MENA", "jordan": "MENA",
}


def get_region(country_raw):
    """Map raw country_region string to broad region."""
    for part in country_raw.replace(";", ",").split(","):
        part = part.strip().lower()
        if part in COUNTRY_REGION:
            return COUNTRY_REGION[part]
        # Try prefix match for Country_SubRegion patterns
        for key, region in COUNTRY_REGION.items():
            if part.startswith(key):
                return region
    return None


def period_label(year):
    if pd.isna(year):
        return None
    y = int(year)
    if y <= 2014:
        return "2005–2014"
    elif y <= 2019:
        return "2015–2019"
    else:
        return "2020–2025"


def fig1_domain_time(df, n=None):
    """Heatmap: domain (rows) × time period (cols)."""
    PERIODS = ["2005–2014", "2015–2019", "2020–2025"]

    df["period"] = pd.to_numeric(df["publication_year"], errors="coerce").apply(period_label)

    rows = []
    for _, row in df.iterrows():
        p = row["period"]
        if not p:
            continue
        for d in str(row["process_outcome_domains"]).replace(";", ",").split(","):
            d = d.strip()
            label = DOMAIN_LABELS.get(d)
            if label:
                rows.append({"period": p, "domain": label})

    ct = (pd.DataFrame(rows)
          .groupby(["domain", "period"])
          .size()
          .unstack(fill_value=0)
          .reindex(columns=PERIODS, fill_value=0))

    # Order: process domains first, then outcome
    process_order = [d for d in [
        "Knowledge & Awareness", "Decision-making & Planning", "Uptake & Adoption",
        "Behavioral Change", "Participation & Co-production", "Institutional Governance",
        "Access to Services",
    ] if d in ct.index]
    outcome_order = [d for d in [
        "Yields & Productivity", "Income & Assets", "Livelihoods",
        "Wellbeing", "Risk Reduction", "Resilience & Adaptive Capacity",
    ] if d in ct.index]
    ordered = process_order + outcome_order
    ct = ct.reindex(ordered)

    # Colour: process=blue, outcome=green (split colorscale via annotations)
    z = ct.values.tolist()
    text = [[str(v) if v > 0 else "" for v in row] for row in z]

    colors = ["#3b82f6" if d in PROCESS_DOMAINS else "#22c55e" for d in ct.index]

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=z,
        x=PERIODS,
        y=list(ct.index),
        text=text,
        texttemplate="%{text}",
        colorscale=[[0, "#f0f9ff"], [1, "#1e40af"]],
        showscale=True,
        colorbar=dict(title="Studies"),
        hovertemplate="<b>%{y}</b><br>%{x}: %{z} studies<extra></extra>",
    ))

    # Add divider line between process and outcome domains
    fig.add_shape(type="line",
                  x0=-0.5, x1=2.5,
                  y0=len(process_order) - 0.5, y1=len(process_order) - 0.5,
                  line=dict(color="gray", width=1.5, dash="dash"))

    fig.add_annotation(x=2.5, y=len(process_order) - 0.5,
                       text="← Process domains  |  Outcome domains →",
                       showarrow=False, xanchor="right", yanchor="bottom",
                       font=dict(size=9, color="gray"))

    fig.update_layout(
        title=dict(
            text=f"<b>Research Focus Over Time</b><br>"
                 f"<sup>Number of studies per domain × publication period · n={n} · "
                 f"studies may appear in multiple domains</sup>",
            x=0.5, xanchor="center", font=dict(size=14),
        ),
        height=520,
        yaxis=dict(autorange="reversed"),
        xaxis=dict(side="top"),
        margin=dict(l=220, r=40, t=120, b=40),
    )
    return fig


def fig2_producer_region(df, n=None):
    """Heatmap: producer type (cols) × region (rows)."""
    REGIONS = ["Sub-Saharan Africa", "South Asia", "East & SE Asia",
               "Latin America", "MENA"]
    PRODUCERS = ["Crop farmers", "Livestock", "Fisheries/Aquaculture",
                 "Agroforestry", "Mixed/General"]

    rows = []
    for _, row in df.iterrows():
        region = get_region(str(row.get("country_region", "")))
        if not region or region not in REGIONS:
            continue
        for p in str(row["producer_type"]).replace(";", ",").split(","):
            p = p.strip().lower()
            label = PRODUCER_LABELS.get(p)
            if label:
                rows.append({"region": region, "producer": label})

    ct = (pd.DataFrame(rows)
          .groupby(["region", "producer"])
          .size()
          .unstack(fill_value=0)
          .reindex(index=REGIONS, columns=PRODUCERS, fill_value=0))

    z = ct.values.tolist()
    text = [[str(v) if v > 0 else "" for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=PRODUCERS,
        y=REGIONS,
        text=text,
        texttemplate="%{text}",
        colorscale=[[0, "#fff7ed"], [1, "#c2410c"]],
        showscale=True,
        colorbar=dict(title="Studies"),
        hovertemplate="<b>%{y} · %{x}</b><br>%{z} studies<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>Producer Types by Region</b><br>"
                 f"<sup>Number of studies per producer type × region · n={n} · "
                 f"studies may appear in multiple categories</sup>",
            x=0.5, xanchor="center", font=dict(size=14),
        ),
        height=420,
        xaxis=dict(side="bottom", tickangle=-30),
        margin=dict(l=160, r=60, t=100, b=100),
    )
    return fig


def save(fig, name):
    import plotly.io as pio
    json_path = OUT / f"{name}.json"
    png_path  = OUT / f"{name}.png"
    fig.write_json(str(json_path))
    try:
        pio.write_image(fig, str(png_path), scale=2)
        print(f"Saved: {png_path.name}  ({png_path.stat().st_size//1024} KB)")
    except Exception as e:
        print(f"PNG failed ({e}) — JSON saved: {json_path.name}")


if __name__ == "__main__":
    df = pd.read_csv(ROOT / "scripts" / "outputs" / "step15" / "step15_human.csv")
    n = len(df)
    print(f"Loaded {n} studies")

    f1 = fig1_domain_time(df, n=n)
    save(f1, "crosstab_domain_time")

    f2 = fig2_producer_region(df, n=n)
    save(f2, "crosstab_producer_region")

    print("Done.")
