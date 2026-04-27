"""
compare_r1a.py

Side-by-side comparison of AZ, CGS, and LLM coding for FT-R1a.

Outputs:
  comparison_r1a.csv   — long-format: doi × field → values + agree flags
  summary_r1a.csv      — agreement rates by field
  comparison_r1a.html  — human-readable colour-coded table

Run: python documentation/coding/systematic-map/rounds/FT-R1a/compare_r1a.py
"""
from __future__ import annotations

from pathlib import Path
import re
import pandas as pd

HERE = Path(__file__).resolve().parent

# ── load ──────────────────────────────────────────────────────────────────────

def load(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)

llm = load(HERE.parent / "FT-R1b" / "coding_ft_r1b_LLM.csv")
cgs = load(HERE / "coding_ft_r1a_CGS.xlsx")
az  = load(HERE / "coding_ft_r1a_AZ.xlsx")

# Align all on DOI
llm = llm.set_index("doi")
cgs = cgs.set_index("doi")
az  = az.set_index("doi")

# Fields to compare (union of all cols except metadata)
SKIP = {"filename", "coder_id", "notes"}

CATEGORICAL = [
    "publication_year", "publication_type", "geographic_scale",
    "producer_type", "temporal_coverage", "cost_data_reported",
    "purpose_of_assessment",
]

MULTIVALUE = [
    "marginalized_subpopulations", "methodological_approach",
    "process_outcome_domains", "data_sources",
]

FREETEXT = [
    "country_region", "adaptation_focus", "indicators_measured",
    "strengths_and_limitations", "lessons_learned",
]

ALL_FIELDS = CATEGORICAL + MULTIVALUE + FREETEXT

# ── normalise helpers ─────────────────────────────────────────────────────────

def norm_str(v) -> str:
    if pd.isna(v):
        return ""
    return str(v).strip().lower()

def norm_set(v) -> frozenset:
    """Split semicolon/comma-separated multi-value fields into a frozenset."""
    if pd.isna(v):
        return frozenset()
    parts = re.split(r"[;,]", str(v))
    return frozenset(p.strip().lower() for p in parts if p.strip())

def agree_categorical(a, b) -> bool:
    return norm_str(a) == norm_str(b)

def agree_multivalue(a, b) -> str:
    """Return 'yes', 'partial', or 'no'."""
    sa, sb = norm_set(a), norm_set(b)
    if not sa and not sb:
        return "both_empty"
    if sa == sb:
        return "yes"
    if sa & sb:
        return "partial"
    return "no"

# ── build comparison rows ─────────────────────────────────────────────────────

rows = []
dois = llm.index.tolist()

for doi in dois:
    for field in ALL_FIELDS:
        v_llm = llm.at[doi, field] if field in llm.columns else None
        v_cgs = cgs.at[doi, field] if field in cgs.columns else None
        v_az  = az.at[doi,  field] if field in az.columns  else None

        if field in CATEGORICAL:
            cgs_llm = "yes" if agree_categorical(v_cgs, v_llm) else "no"
            az_llm  = "yes" if agree_categorical(v_az,  v_llm) else "no"
            cgs_az  = "yes" if agree_categorical(v_cgs, v_az)  else "no"
        elif field in MULTIVALUE:
            cgs_llm = agree_multivalue(v_cgs, v_llm)
            az_llm  = agree_multivalue(v_az,  v_llm)
            cgs_az  = agree_multivalue(v_cgs, v_az)
        else:
            cgs_llm = az_llm = cgs_az = "—"

        rows.append({
            "doi":       doi,
            "field":     field,
            "field_type": ("categorical" if field in CATEGORICAL
                           else "multivalue" if field in MULTIVALUE
                           else "freetext"),
            "CGS":       "" if v_cgs is None or (isinstance(v_cgs, float) and pd.isna(v_cgs)) else str(v_cgs),
            "AZ":        "" if v_az  is None or (isinstance(v_az,  float) and pd.isna(v_az))  else str(v_az),
            "LLM":       "" if v_llm is None or (isinstance(v_llm, float) and pd.isna(v_llm)) else str(v_llm),
            "CGS_vs_LLM": cgs_llm,
            "AZ_vs_LLM":  az_llm,
            "CGS_vs_AZ":  cgs_az,
        })

comp = pd.DataFrame(rows)
comp.to_csv(HERE / "comparison_r1a.csv", index=False)
print(f"Saved comparison_r1a.csv ({len(comp)} rows)")

# ── summary by field ──────────────────────────────────────────────────────────

def pct_yes(series):
    total = len(series)
    if total == 0:
        return "—"
    yes = (series == "yes").sum()
    return f"{yes}/{total} ({100*yes//total}%)"

summary_rows = []
for field in CATEGORICAL + MULTIVALUE:
    sub = comp[comp["field"] == field]
    summary_rows.append({
        "field":       field,
        "type":        "categorical" if field in CATEGORICAL else "multivalue",
        "CGS_vs_LLM":  pct_yes(sub["CGS_vs_LLM"]),
        "AZ_vs_LLM":   pct_yes(sub["AZ_vs_LLM"]),
        "CGS_vs_AZ":   pct_yes(sub["CGS_vs_AZ"]),
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(HERE / "summary_r1a.csv", index=False)
print(f"Saved summary_r1a.csv")
print()
print(summary.to_string(index=False))

# ── HTML report ───────────────────────────────────────────────────────────────

COLORS = {
    "yes":        "#d4edda",
    "partial":    "#fff3cd",
    "no":         "#f8d7da",
    "both_empty": "#e9ecef",
    "—":          "#ffffff",
}

short_doi = {d: d.split("/")[-1][:30] for d in dois}

html_parts = ["""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>FT-R1a Comparison: AZ vs CGS vs LLM</title>
<style>
  body { font-family: Arial, sans-serif; font-size: 12px; margin: 20px; }
  h2 { color: #21472E; }
  h3 { color: #3c3c3c; margin-top: 30px; border-bottom: 2px solid #21472E; padding-bottom: 4px; }
  table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
  th { background: #21472E; color: white; padding: 6px 8px; text-align: left; position: sticky; top: 0; }
  td { padding: 5px 8px; border: 1px solid #ddd; vertical-align: top; max-width: 300px; word-wrap: break-word; }
  .field { font-weight: bold; color: #555; white-space: nowrap; }
  .type { color: #888; font-size: 10px; }
  .agree { background: #d4edda; }
  .partial { background: #fff3cd; }
  .disagree { background: #f8d7da; }
  .empty { background: #e9ecef; color: #aaa; }
  .summary-table td, .summary-table th { padding: 4px 10px; }
</style></head><body>
<h2>FT-R1a Calibration Round — Coder Comparison</h2>
<p>Comparing <b>CGS</b>, <b>AZ</b>, and <b>LLM</b> across 5 papers and all fields.<br>
<span style="background:#d4edda;padding:2px 6px">Green = agree</span>&nbsp;
<span style="background:#fff3cd;padding:2px 6px">Yellow = partial</span>&nbsp;
<span style="background:#f8d7da;padding:2px 6px">Red = disagree</span>
</p>
"""]

# Summary table
html_parts.append("<h3>Agreement Summary (categorical + multivalue fields)</h3>")
html_parts.append('<table class="summary-table"><tr><th>Field</th><th>Type</th>'
                  '<th>CGS vs LLM</th><th>AZ vs LLM</th><th>CGS vs AZ</th></tr>')
for _, r in summary.iterrows():
    html_parts.append(f"<tr><td class='field'>{r['field']}</td><td class='type'>{r['type']}</td>"
                      f"<td>{r['CGS_vs_LLM']}</td><td>{r['AZ_vs_LLM']}</td><td>{r['CGS_vs_AZ']}</td></tr>")
html_parts.append("</table>")

# Per-paper detail
for doi in dois:
    html_parts.append(f"<h3>{doi}</h3>")
    html_parts.append("<table><tr><th>Field</th><th>Type</th>"
                      "<th>CGS</th><th>AZ</th><th>LLM</th>"
                      "<th>CGS vs LLM</th><th>AZ vs LLM</th><th>CGS vs AZ</th></tr>")
    paper = comp[comp["doi"] == doi]
    for _, row in paper.iterrows():
        def cell_class(val):
            if val == "yes":      return "agree"
            if val == "partial":  return "partial"
            if val == "no":       return "disagree"
            if val == "both_empty": return "empty"
            return ""

        html_parts.append(
            f"<tr>"
            f"<td class='field'>{row['field']}</td>"
            f"<td class='type'>{row['field_type']}</td>"
            f"<td>{row['CGS']}</td>"
            f"<td>{row['AZ']}</td>"
            f"<td>{row['LLM']}</td>"
            f"<td class='{cell_class(row['CGS_vs_LLM'])}'>{row['CGS_vs_LLM']}</td>"
            f"<td class='{cell_class(row['AZ_vs_LLM'])}'>{row['AZ_vs_LLM']}</td>"
            f"<td class='{cell_class(row['CGS_vs_AZ'])}'>{row['CGS_vs_AZ']}</td>"
            f"</tr>"
        )
    html_parts.append("</table>")

html_parts.append("</body></html>")

html_path = HERE / "comparison_r1a.html"
html_path.write_text("\n".join(html_parts), encoding="utf-8")
print(f"\nSaved comparison_r1a.html — open in browser for full review")
print(f"  file://{html_path}")

# ── PDF report ────────────────────────────────────────────────────────────────

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
)

GREEN    = colors.HexColor("#21472E")
RED      = colors.HexColor("#f8d7da")
YELLOW   = colors.HexColor("#fff3cd")
LGREEN   = colors.HexColor("#d4edda")
LGREY    = colors.HexColor("#e9ecef")
WHITE    = colors.white
OFFWHITE = colors.HexColor("#f5f5f5")

styles = getSampleStyleSheet()
body   = ParagraphStyle("body",   fontSize=7,  leading=9)
bold   = ParagraphStyle("bold",   fontSize=7,  leading=9,  fontName="Helvetica-Bold")
title  = ParagraphStyle("title",  fontSize=14, leading=18, fontName="Helvetica-Bold", textColor=GREEN)
h3     = ParagraphStyle("h3",     fontSize=10, leading=14, fontName="Helvetica-Bold", textColor=GREEN,
                         spaceBefore=14, spaceAfter=4)
small  = ParagraphStyle("small",  fontSize=6,  leading=8,  textColor=colors.grey)

def esc(text, maxlen=220) -> str:
    if not text:
        return "—"
    s = str(text)
    if len(s) > maxlen:
        s = s[:maxlen] + "…"
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;"))

def P(text, style=body, maxlen=220):
    return Paragraph(esc(text, maxlen), style)

def agree_color(val):
    if val == "yes":        return LGREEN
    if val == "partial":    return YELLOW
    if val == "no":         return RED
    if val == "both_empty": return LGREY
    return WHITE

pdf_path = HERE / "comparison_r1a.pdf"
doc = SimpleDocTemplate(
    str(pdf_path),
    pagesize=landscape(A4),
    leftMargin=0.8*cm, rightMargin=0.8*cm,
    topMargin=1.0*cm,  bottomMargin=1.0*cm,
)

story = [
    Paragraph("FT-R1a Calibration Round — Coder Comparison", title),
    Paragraph("CGS · AZ · LLM — 5 papers, all fields", small),
    Spacer(1, 0.4*cm),
]

# ── Summary table ─────────────────────────────────────────────────────────────
story.append(Paragraph("Agreement Summary (categorical + multivalue fields)", h3))

sum_header = ["Field", "Type", "CGS vs LLM", "AZ vs LLM", "CGS vs AZ"]
sum_data   = [sum_header]
for _, r in summary.iterrows():
    sum_data.append([P(r["field"], bold), P(r["type"], small),
                     P(r["CGS_vs_LLM"]), P(r["AZ_vs_LLM"]), P(r["CGS_vs_AZ"])])

sum_table = Table(sum_data, colWidths=[6*cm, 2.5*cm, 3*cm, 3*cm, 3*cm])
sum_table.setStyle(TableStyle([
    ("BACKGROUND",  (0,0), (-1,0),  GREEN),
    ("TEXTCOLOR",   (0,0), (-1,0),  WHITE),
    ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
    ("FONTSIZE",    (0,0), (-1,-1), 7),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [WHITE, OFFWHITE]),
    ("GRID",        (0,0), (-1,-1), 0.3, colors.HexColor("#cccccc")),
    ("VALIGN",      (0,0), (-1,-1), "TOP"),
    ("TOPPADDING",  (0,0), (-1,-1), 3),
    ("BOTTOMPADDING",(0,0),(-1,-1), 3),
]))
story.append(sum_table)
story.append(PageBreak())

# ── Per-paper detail tables ───────────────────────────────────────────────────
# Landscape A4 usable width: 29.7 - 1.6 margins = 28.1cm
COL_W = [3.0*cm, 1.8*cm, 6.2*cm, 6.2*cm, 6.2*cm, 1.6*cm, 1.6*cm, 1.5*cm]  # = 28.1cm
HEADER = ["Field", "Type", "CGS", "AZ", "LLM", "CGS\nvs LLM", "AZ\nvs LLM", "CGS\nvs AZ"]

for doi in dois:
    story.append(Paragraph(doi, h3))
    paper = comp[comp["doi"] == doi]

    tdata = [HEADER]
    tstyle_cmds = [
        ("BACKGROUND",   (0,0), (-1,0),  GREEN),
        ("TEXTCOLOR",    (0,0), (-1,0),  WHITE),
        ("FONTNAME",     (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,-1), 6),
        ("GRID",         (0,0), (-1,-1), 0.3, colors.HexColor("#cccccc")),
        ("VALIGN",       (0,0), (-1,-1), "TOP"),
        ("TOPPADDING",   (0,0), (-1,-1), 3),
        ("BOTTOMPADDING",(0,0), (-1,-1), 3),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [WHITE, OFFWHITE]),
    ]

    for i, (_, row) in enumerate(paper.iterrows(), start=1):
        tdata.append([
            P(row["field"], bold),
            P(row["field_type"], small),
            P(row["CGS"]),
            P(row["AZ"]),
            P(row["LLM"]),
            P(row["CGS_vs_LLM"]),
            P(row["AZ_vs_LLM"]),
            P(row["CGS_vs_AZ"]),
        ])
        for col_idx, key in [(5, "CGS_vs_LLM"), (6, "AZ_vs_LLM"), (7, "CGS_vs_AZ")]:
            c = agree_color(row[key])
            if c != WHITE:
                tstyle_cmds.append(("BACKGROUND", (col_idx, i), (col_idx, i), c))

    t = Table(tdata, colWidths=COL_W, repeatRows=1)
    t.setStyle(TableStyle(tstyle_cmds))
    story.append(t)
    story.append(Spacer(1, 0.3*cm))

doc.build(story)
print(f"Saved comparison_r1a.pdf")
print(f"  {pdf_path}")
