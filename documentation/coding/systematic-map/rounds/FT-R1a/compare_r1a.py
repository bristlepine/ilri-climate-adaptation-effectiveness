"""
compare_r1a.py

Side-by-side comparison of all human coders and LLM for FT-R1a.

Outputs:
  comparison_r1a.csv   — long-format: doi × field → values + agree flags
  summary_r1a.csv      — agreement rates by field
  comparison_r1a.html  — human-readable colour-coded table
  comparison_r1a.pdf   — PDF version

Run: python documentation/coding/systematic-map/rounds/FT-R1a/compare_r1a.py
"""
from __future__ import annotations

from pathlib import Path
from itertools import combinations
import re
import pandas as pd

HERE = Path(__file__).resolve().parent

# ── load ──────────────────────────────────────────────────────────────────────

def load(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)

# Human coders — add new coders here
HUMAN_FILES = {
    "AZ":  HERE / "coding_ft_r1a_AZ.xlsx",
    "CGS": HERE / "coding_ft_r1a_CGS.xlsx",
    "SZC": HERE / "coding_ft_r1a_SZC.csv",
}

llm = load(HERE.parent / "FT-R1b" / "coding_ft_r1b_LLM.csv").set_index("doi")
humans_raw = {name: load(path) for name, path in HUMAN_FILES.items()}

humans = {name: df.set_index("doi") for name, df in humans_raw.items()}
CODERS = {"CGS": humans["CGS"], "LLM": llm, "AZ": humans["AZ"], "SZC": humans["SZC"]}
HUMAN_NAMES = ["CGS", "AZ", "SZC"]

# ── field definitions ─────────────────────────────────────────────────────────

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
    if pd.isna(v):
        return frozenset()
    parts = re.split(r"[;,]", str(v))
    return frozenset(p.strip().lower() for p in parts if p.strip())

def agree_categorical(a, b) -> bool:
    return norm_str(a) == norm_str(b)

def agree_multivalue(a, b) -> str:
    sa, sb = norm_set(a), norm_set(b)
    if not sa and not sb:
        return "both_empty"
    if sa == sb:
        return "yes"
    if sa & sb:
        return "partial"
    return "no"

def get_val(df, doi, field):
    try:
        v = df.at[doi, field] if field in df.columns else None
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
        return str(v)
    except KeyError:
        return ""

# ── comparison pairs ──────────────────────────────────────────────────────────
# All human-vs-LLM pairs + all human-vs-human pairs

PAIRS = [
    ("CGS", "LLM"),
    ("CGS", "AZ"),
    ("CGS", "SZC"),
    ("LLM", "AZ"),
    ("LLM", "SZC"),
    ("AZ", "SZC"),
]
PAIR_COLS = [f"{a}_vs_{b}" for a, b in PAIRS]

# ── build comparison rows ─────────────────────────────────────────────────────

dois = llm.index.tolist()
rows = []

for doi in dois:
    for field in ALL_FIELDS:
        vals = {name: get_val(df, doi, field) for name, df in CODERS.items()}

        agrees = {}
        for (a, b), col in zip(PAIRS, PAIR_COLS):
            va, vb = vals[a], vals[b]
            if field in CATEGORICAL:
                agrees[col] = "yes" if agree_categorical(va, vb) else "no"
            elif field in MULTIVALUE:
                agrees[col] = agree_multivalue(va, vb)
            else:
                agrees[col] = "—"

        rows.append({
            "doi": doi,
            "field": field,
            "field_type": (
                "categorical" if field in CATEGORICAL
                else "multivalue" if field in MULTIVALUE
                else "freetext"
            ),
            **vals,
            **agrees,
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
    row = {"field": field, "type": "categorical" if field in CATEGORICAL else "multivalue"}
    for col in PAIR_COLS:
        row[col] = pct_yes(sub[col])
    summary_rows.append(row)

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

html_parts = [f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>FT-R1a Comparison</title>
<style>
  body {{ font-family: Arial, sans-serif; font-size: 12px; margin: 20px; }}
  h2 {{ color: #21472E; }}
  h3 {{ color: #3c3c3c; margin-top: 30px; border-bottom: 2px solid #21472E; padding-bottom: 4px; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
  th {{ background: #21472E; color: white; padding: 6px 8px; text-align: left; position: sticky; top: 0; }}
  td {{ padding: 5px 8px; border: 1px solid #ddd; vertical-align: top; max-width: 260px; word-wrap: break-word; }}
  .field {{ font-weight: bold; color: #555; white-space: nowrap; }}
  .type {{ color: #888; font-size: 10px; }}
  .agree {{ background: #d4edda; }}
  .partial {{ background: #fff3cd; }}
  .disagree {{ background: #f8d7da; }}
  .empty {{ background: #e9ecef; color: #aaa; }}
  .summary-table td, .summary-table th {{ padding: 4px 10px; }}
</style></head><body>
<h2>FT-R1a Calibration Round — Coder Comparison</h2>
<p>Comparing <b>{", ".join(HUMAN_NAMES)}</b> and <b>LLM</b> across {len(dois)} papers.<br>
<span style="background:#d4edda;padding:2px 6px">Green = agree</span>&nbsp;
<span style="background:#fff3cd;padding:2px 6px">Yellow = partial</span>&nbsp;
<span style="background:#f8d7da;padding:2px 6px">Red = disagree</span>
</p>
"""]

# Summary table with heatmap
def pct_to_color(pct_str):
    """Convert '5/5 (100%)' to color: green (100%) → yellow (≥20%) → red (<20%)"""
    if not pct_str or pct_str == "—":
        return "#ffffff"
    try:
        pct = int(pct_str.split("(")[1].split("%")[0])
    except (IndexError, ValueError):
        return "#ffffff"
    if pct == 100:
        return "#d4edda"  # green
    elif pct >= 20:
        return "#fff3cd"  # yellow
    else:
        return "#f8d7da"  # red

html_parts.append("<h3>Agreement Summary (categorical + multivalue fields)</h3>")
header_cells = "".join(f"<th>{c}</th>" for c in ["Field", "Type"] + PAIR_COLS)
html_parts.append(f'<table class="summary-table"><tr>{header_cells}</tr>')
for _, r in summary.iterrows():
    cells = f"<td class='field'>{r['field']}</td><td class='type'>{r['type']}</td>"
    for col in PAIR_COLS:
        color = pct_to_color(r[col])
        cells += f"<td style='background-color:{color};'>{r[col]}</td>"
    html_parts.append(f"<tr>{cells}</tr>")
html_parts.append("</table>")

# Per-paper detail
for doi in dois:
    html_parts.append(f"<h3>{doi}</h3>")
    val_headers = "".join(f"<th>{n}</th>" for n in list(CODERS.keys()))
    agree_headers = "".join(f"<th>{c}</th>" for c in PAIR_COLS)
    html_parts.append(f"<table><tr><th>Field</th><th>Type</th>{val_headers}{agree_headers}</tr>")
    paper = comp[comp["doi"] == doi]
    for _, row in paper.iterrows():
        def cell_class(val):
            if val == "yes":        return "agree"
            if val == "partial":    return "partial"
            if val == "no":         return "disagree"
            if val == "both_empty": return "empty"
            return ""

        val_cells = "".join(f"<td>{row[n]}</td>" for n in CODERS.keys())
        agree_cells = "".join(
            f"<td class='{cell_class(row[col])}'>{row[col]}</td>"
            for col in PAIR_COLS
        )
        html_parts.append(
            f"<tr>"
            f"<td class='field'>{row['field']}</td>"
            f"<td class='type'>{row['field_type']}</td>"
            f"{val_cells}{agree_cells}"
            f"</tr>"
        )
    html_parts.append("</table>")

html_parts.append("</body></html>")
html_path = HERE / "comparison_r1a.html"
html_path.write_text("\n".join(html_parts), encoding="utf-8")
print(f"\nSaved comparison_r1a.html")
print(f"  file://{html_path}")

# ── PDF report ────────────────────────────────────────────────────────────────

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak

    GREEN    = colors.HexColor("#21472E")
    RED      = colors.HexColor("#f8d7da")
    YELLOW   = colors.HexColor("#fff3cd")
    LGREEN   = colors.HexColor("#d4edda")
    LGREY    = colors.HexColor("#e9ecef")
    WHITE    = colors.white
    OFFWHITE = colors.HexColor("#f5f5f5")

    body  = ParagraphStyle("body",  fontSize=6,  leading=8)
    bold  = ParagraphStyle("bold",  fontSize=6,  leading=8,  fontName="Helvetica-Bold")
    title = ParagraphStyle("title", fontSize=13, leading=16, fontName="Helvetica-Bold", textColor=GREEN)
    h3    = ParagraphStyle("h3",    fontSize=9,  leading=12, fontName="Helvetica-Bold", textColor=GREEN,
                            spaceBefore=12, spaceAfter=4)
    small = ParagraphStyle("small", fontSize=5,  leading=7,  textColor=colors.grey)

    def esc(text, maxlen=180):
        if not text:
            return "—"
        s = str(text)[:maxlen]
        return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

    def P(text, style=body, maxlen=180):
        return Paragraph(esc(text, maxlen), style)

    def agree_color(val):
        if val == "yes":        return LGREEN
        if val == "partial":    return YELLOW
        if val == "no":         return RED
        if val == "both_empty": return LGREY
        return WHITE

    pdf_path = HERE / "comparison_r1a.pdf"
    n_coders = len(CODERS)
    n_pairs  = len(PAIRS)

    # Landscape A4 usable width: ~28cm
    # Layout: Field(2.5) + Type(1.5) + N coder cols + N pair cols
    coder_w = 4.5
    pair_w  = 1.4
    total_w = 2.5 + 1.5 + n_coders * coder_w + n_pairs * pair_w
    # Scale down if too wide
    scale = min(1.0, 28.0 / total_w)
    COL_W = (
        [2.5*scale*cm, 1.5*scale*cm]
        + [coder_w*scale*cm] * n_coders
        + [pair_w*scale*cm]  * n_pairs
    )
    HEADER_ROW = (
        ["Field", "Type"]
        + list(CODERS.keys())
        + [c.replace("_vs_", "\nvs\n") for c in PAIR_COLS]
    )

    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=landscape(A4),
        leftMargin=0.8*cm, rightMargin=0.8*cm,
        topMargin=1.0*cm,  bottomMargin=1.0*cm,
    )
    story = [
        Paragraph("FT-R1a Calibration Round — Coder Comparison", title),
        Paragraph(f"{', '.join(HUMAN_NAMES)} · LLM — {len(dois)} papers, all fields", small),
        Spacer(1, 0.4*cm),
    ]

    # Summary table with heatmap coloring
    def pct_to_color(pct_str):
        """Convert '3/5 (60%)' to a color: green (100%) → yellow → red (0%)"""
        if not pct_str or pct_str == "—":
            return WHITE
        try:
            pct = int(pct_str.split("(")[1].split("%")[0])
        except (IndexError, ValueError):
            return WHITE
        if pct == 100:
            return colors.HexColor("#d4edda")  # green
        elif pct >= 20:
            return colors.HexColor("#fff3cd")  # yellow
        else:
            return colors.HexColor("#f8d7da")  # red

    story.append(Paragraph("Agreement Summary (categorical + multivalue fields)", h3))
    sum_header = ["Field", "Type"] + PAIR_COLS
    sum_data = [sum_header]
    for _, r in summary.iterrows():
        sum_data.append([P(r["field"], bold), P(r["type"], small)]
                        + [P(r[col]) for col in PAIR_COLS])
    # Bigger columns — use available page width
    sum_col_w = [5.5*cm, 2.2*cm] + [3.2*cm]*n_pairs
    sum_table = Table(sum_data, colWidths=sum_col_w)

    # Build table style with heatmap coloring on agreement columns
    style_cmds = [
        ("BACKGROUND",  (0,0), (-1,0),  GREEN),
        ("TEXTCOLOR",   (0,0), (-1,0),  WHITE),
        ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 7),
        ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("ALIGN",       (0,0), (-1,-1), "CENTER"),
        ("TOPPADDING",  (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
    ]
    # Color agreement columns: columns 2 onward (0=Field, 1=Type, 2+=pairs)
    for i, (_, r) in enumerate(summary.iterrows(), start=1):
        for j, col in enumerate(PAIR_COLS):
            bg_color = pct_to_color(r[col])
            col_idx = 2 + j  # columns 0,1 are Field,Type; pairs start at col 2
            style_cmds.append(("BACKGROUND", (col_idx, i), (col_idx, i), bg_color))
    # Stripe alternating rows for readability
    for i in range(1, len(summary) + 1):
        if i % 2 == 0:
            style_cmds.append(("BACKGROUND", (0, i), (1, i), OFFWHITE))

    sum_table.setStyle(TableStyle(style_cmds))
    story.append(sum_table)
    story.append(Spacer(1, 1*cm))

    # Per-paper tables
    for doi in dois:
        story.append(Paragraph(doi, h3))
        paper = comp[comp["doi"] == doi]
        tdata = [HEADER_ROW]
        tstyle_cmds = [
            ("BACKGROUND",   (0,0), (-1,0),  GREEN),
            ("TEXTCOLOR",    (0,0), (-1,0),  WHITE),
            ("FONTNAME",     (0,0), (-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",     (0,0), (-1,-1), 5),
            ("GRID",         (0,0), (-1,-1), 0.3, colors.HexColor("#cccccc")),
            ("VALIGN",       (0,0), (-1,-1), "TOP"),
            ("TOPPADDING",   (0,0), (-1,-1), 2),
            ("BOTTOMPADDING",(0,0), (-1,-1), 2),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [WHITE, OFFWHITE]),
        ]
        for i, (_, row) in enumerate(paper.iterrows(), start=1):
            tdata.append(
                [P(row["field"], bold), P(row["field_type"], small)]
                + [P(row[n]) for n in CODERS.keys()]
                + [P(row[col]) for col in PAIR_COLS]
            )
            for j, col in enumerate(PAIR_COLS):
                c = agree_color(row[col])
                col_idx = 2 + n_coders + j
                if c != WHITE:
                    tstyle_cmds.append(("BACKGROUND", (col_idx, i), (col_idx, i), c))

        t = Table(tdata, colWidths=COL_W, repeatRows=1)
        t.setStyle(TableStyle(tstyle_cmds))
        story.append(t)
        story.append(Spacer(1, 0.3*cm))

    doc.build(story)
    print(f"Saved comparison_r1a.pdf")
    print(f"  {pdf_path}")

except ImportError:
    print("reportlab not available — skipping PDF")
