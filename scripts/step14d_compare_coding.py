#!/usr/bin/env python3
"""
step14d_compare_coding.py

Compare human coder outputs against LLM coding for any round.
Generalises compare_r1a.py to work with all rounds (calibration and production).

Human coder files are discovered automatically:
  New rounds  (FT-R2x): scripts/outputs/step14b/{round}/coding_*_INITIALS.csv
  Legacy rounds (FT-R1x): documentation/coding/systematic-map/rounds/{round}/coding_*_INITIALS.*

LLM data comes from:
  scripts/outputs/step14/step14_results.csv   — screening decision
  scripts/outputs/step15/step15_coded.csv     — extraction fields

Outputs per round go to: scripts/outputs/step14d/{round}/
  comparison_{round}.csv   — long-format rows: doi × field → values + agree flags
  summary_{round}.csv      — agreement rates by field
  comparison_{round}.html  — colour-coded HTML table
  comparison_{round}.pdf   — PDF version

Cross-round summary: scripts/outputs/step14d/summary_all_rounds.csv/.html

Usage:
  python scripts/step14d_compare_coding.py --round FT-R1a
  python scripts/step14d_compare_coding.py --round FT-R2a --round FT-R2b
  python scripts/step14d_compare_coding.py --all
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────────

REPO_ROOT      = Path(__file__).resolve().parent.parent
OUTPUTS_DIR    = REPO_ROOT / "scripts" / "outputs"
ROUNDS_DOC_DIR = REPO_ROOT / "documentation" / "coding" / "systematic-map" / "rounds"

STEP14_CSV = OUTPUTS_DIR / "step14" / "step14_results.csv"
STEP15_CSV = OUTPUTS_DIR / "step15" / "step15_coded.csv"
OUT_DIR    = OUTPUTS_DIR / "step14d"

# ── field definitions ──────────────────────────────────────────────────────────

INCLUSION_FIELD = "confirmed_include"

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
ALL_FIELDS     = [INCLUSION_FIELD] + CATEGORICAL + MULTIVALUE + FREETEXT
SCORED_FIELDS  = [INCLUSION_FIELD] + CATEGORICAL + MULTIVALUE  # used in summary stats


# ── normalisation helpers ──────────────────────────────────────────────────────

def norm_str(v) -> str:
    if pd.isna(v) or v is None:
        return ""
    return str(v).strip().lower()

def norm_set(v) -> frozenset:
    if pd.isna(v) or v is None:
        return frozenset()
    return frozenset(p.strip().lower() for p in re.split(r"[;,]", str(v)) if p.strip())

def norm_include(v) -> str:
    """Normalise confirmed_include to yes / no / unclear."""
    s = norm_str(v)
    if s in ("yes", "include", "1", "true"):
        return "yes"
    if s in ("no", "exclude", "0", "false"):
        return "no"
    if s in ("unclear", "uncertain", "maybe"):
        return "unclear"
    return s  # pass through if unrecognised

def agree_categorical(a, b) -> str:
    na, nb = norm_str(a), norm_str(b)
    if not na and not nb:
        return "both_empty"
    return "yes" if na == nb else "no"

def agree_multivalue(a, b) -> str:
    sa, sb = norm_set(a), norm_set(b)
    if not sa and not sb:
        return "both_empty"
    if sa == sb:
        return "yes"
    if sa & sb:
        return "partial"
    return "no"

def get_val(df: pd.DataFrame, doi: str, field: str) -> str:
    try:
        v = df.at[doi, field]
        if pd.isna(v):
            return ""
        if field == INCLUSION_FIELD:
            return norm_include(str(v))
        return str(v)
    except KeyError:
        return ""


# ── LLM data loading ───────────────────────────────────────────────────────────

_llm_cache: pd.DataFrame | None = None

def load_llm() -> pd.DataFrame:
    """
    Build a unified LLM coding frame indexed by doi.

    confirmed_include  — from s14_decision (Include → yes, Exclude → no)
    <field>            — from step15 *_value columns (stripped suffix)
    """
    global _llm_cache
    if _llm_cache is not None:
        return _llm_cache

    value_cols = [f"{f}_value" for f in CATEGORICAL + MULTIVALUE + FREETEXT]

    # Step14: screening decision for all papers
    s14 = pd.read_csv(STEP14_CSV, usecols=["doi", "s14_decision"], low_memory=False)
    s14 = s14.dropna(subset=["doi"]).drop_duplicates("doi").set_index("doi")
    s14[INCLUSION_FIELD] = s14["s14_decision"].apply(
        lambda x: "yes" if str(x).strip().lower() == "include" else
                  "no"  if str(x).strip().lower() == "exclude" else "unclear"
    )

    # Step15: extraction for included papers
    available = [c for c in value_cols if c in
                 pd.read_csv(STEP15_CSV, nrows=0).columns]
    usecols = ["doi"] + available
    s15 = pd.read_csv(STEP15_CSV, usecols=usecols, low_memory=False)
    s15 = s15.dropna(subset=["doi"]).drop_duplicates("doi")
    # Strip _value suffix
    s15 = s15.rename(columns={f"{field}_value": field
                               for field in CATEGORICAL + MULTIVALUE + FREETEXT})
    s15 = s15.set_index("doi")

    llm = s14[[INCLUSION_FIELD]].join(s15, how="left")
    _llm_cache = llm
    return llm


# ── human coder discovery ──────────────────────────────────────────────────────

_CODER_RE = re.compile(r"coding_.+_([A-Za-z]{1,5})\.(csv|xlsx)$", re.IGNORECASE)

def discover_coder_files(round_name: str) -> dict[str, Path]:
    """
    Return {initials: path} for all human coder files found for this round.
    Looks in both the new step14b output folder and the legacy documentation folder.
    Skips templates, LLM files, and reconciled files.
    """
    skip = {"template", "llm", "reconciled", "fixed", "papers"}
    locations = [
        OUTPUTS_DIR / "step14b" / round_name,
        ROUNDS_DOC_DIR / round_name,
    ]
    files: dict[str, Path] = {}
    for loc in locations:
        if not loc.exists():
            continue
        for f in sorted(loc.iterdir()):
            if not f.is_file():
                continue
            if any(s in f.name.lower() for s in skip):
                continue
            m = _CODER_RE.search(f.name)
            if m:
                initials = m.group(1).upper()
                if initials not in files:
                    files[initials] = f
    return files


def discover_all_rounds() -> list[str]:
    """Return all rounds that have at least one human coder file."""
    rounds: set[str] = set()
    round_re = re.compile(r"^FT-R\d+[a-z]?$")
    for loc in [OUTPUTS_DIR / "step14b", ROUNDS_DOC_DIR]:
        if not loc.exists():
            continue
        for child in loc.iterdir():
            if child.is_dir() and round_re.match(child.name):
                if discover_coder_files(child.name):
                    rounds.add(child.name)
    return sorted(rounds)


# ── coder file loading ─────────────────────────────────────────────────────────

def load_coder(path: Path) -> pd.DataFrame:
    """Load a human coder file, normalise confirmed_include, index by doi."""
    df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_excel(path)
    if "doi" not in df.columns:
        raise ValueError(f"{path.name}: no 'doi' column found")
    df = df.dropna(subset=["doi"]).drop_duplicates("doi").set_index("doi")
    if INCLUSION_FIELD in df.columns:
        df[INCLUSION_FIELD] = df[INCLUSION_FIELD].apply(
            lambda v: norm_include(v) if pd.notna(v) else ""
        )
    return df


# ── comparison logic ───────────────────────────────────────────────────────────

def build_comparison(coders: dict[str, pd.DataFrame], dois: list[str]) -> pd.DataFrame:
    """
    Build long-format comparison: one row per (doi, field).
    Includes values from each coder and pairwise agreement flags.
    """
    coder_names = list(coders.keys())
    from itertools import combinations
    pairs      = list(combinations(coder_names, 2))
    pair_cols  = [f"{a}_vs_{b}" for a, b in pairs]

    rows = []
    for doi in dois:
        for field in ALL_FIELDS:
            vals = {name: get_val(df, doi, field) for name, df in coders.items()}
            agrees = {}
            for (a, b), col in zip(pairs, pair_cols):
                va, vb = vals[a], vals[b]
                if field in [INCLUSION_FIELD] + CATEGORICAL:
                    agrees[col] = agree_categorical(va, vb)
                elif field in MULTIVALUE:
                    agrees[col] = agree_multivalue(va, vb)
                else:
                    agrees[col] = "—"

            rows.append({
                "doi":        doi,
                "field":      field,
                "field_type": ("inclusion"   if field == INCLUSION_FIELD else
                               "categorical" if field in CATEGORICAL      else
                               "multivalue"  if field in MULTIVALUE        else
                               "freetext"),
                **vals,
                **agrees,
            })

    return pd.DataFrame(rows)


def build_summary(comp: pd.DataFrame, coder_names: list[str]) -> pd.DataFrame:
    from itertools import combinations
    pairs     = list(combinations(coder_names, 2))
    pair_cols = [f"{a}_vs_{b}" for a, b in pairs]

    def pct_yes(series: pd.Series) -> str:
        valid = series[series.isin(["yes", "no", "partial", "both_empty"])]
        if len(valid) == 0:
            return "—"
        yes = (valid == "yes").sum()
        return f"{yes}/{len(valid)} ({100 * yes // len(valid)}%)"

    rows = []
    for field in SCORED_FIELDS:
        sub = comp[comp["field"] == field]
        row = {
            "field": field,
            "type":  ("inclusion" if field == INCLUSION_FIELD else
                      "categorical" if field in CATEGORICAL else "multivalue"),
        }
        for col in pair_cols:
            row[col] = pct_yes(sub[col]) if col in sub.columns else "—"
        rows.append(row)

    return pd.DataFrame(rows)


# ── HTML report ────────────────────────────────────────────────────────────────

AGREE_COLORS = {
    "yes":        "#d4edda",
    "partial":    "#fff3cd",
    "no":         "#f8d7da",
    "both_empty": "#e9ecef",
    "—":          "#ffffff",
}

def pct_to_color(pct_str: str) -> str:
    if not pct_str or pct_str == "—":
        return "#ffffff"
    try:
        pct = int(pct_str.split("(")[1].split("%")[0])
    except (IndexError, ValueError):
        return "#ffffff"
    if pct >= 80:
        return "#d4edda"
    if pct >= 50:
        return "#fff3cd"
    return "#f8d7da"

def cell_class(val: str) -> str:
    return {"yes": "agree", "partial": "partial", "no": "disagree",
            "both_empty": "empty"}.get(val, "")

def write_html(
    round_name: str,
    comp: pd.DataFrame,
    summary: pd.DataFrame,
    coder_names: list[str],
    dois: list[str],
    out_path: Path,
) -> None:
    from itertools import combinations
    pairs     = list(combinations(coder_names, 2))
    pair_cols = [f"{a}_vs_{b}" for a, b in pairs]

    parts = [f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>{round_name} Comparison</title>
<style>
  body {{ font-family: Arial, sans-serif; font-size: 12px; margin: 20px; }}
  h2 {{ color: #21472E; }}
  h3 {{ color: #3c3c3c; margin-top: 30px; border-bottom: 2px solid #21472E; padding-bottom: 4px; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
  th {{ background: #21472E; color: white; padding: 6px 8px; text-align: left; position: sticky; top: 0; }}
  td {{ padding: 5px 8px; border: 1px solid #ddd; vertical-align: top; max-width: 260px; word-wrap: break-word; }}
  .field {{ font-weight: bold; color: #555; white-space: nowrap; }}
  .type  {{ color: #888; font-size: 10px; }}
  .agree {{ background: #d4edda; }}
  .partial {{ background: #fff3cd; }}
  .disagree {{ background: #f8d7da; }}
  .empty {{ background: #e9ecef; color: #aaa; }}
</style></head><body>
<h2>{round_name} — Coder Comparison</h2>
<p>Coders: <b>{", ".join(coder_names)}</b> · {len(dois)} papers<br>
<span style="background:#d4edda;padding:2px 6px">Green = agree</span>&nbsp;
<span style="background:#fff3cd;padding:2px 6px">Yellow = partial</span>&nbsp;
<span style="background:#f8d7da;padding:2px 6px">Red = disagree</span>
</p>"""]

    # Summary table
    parts.append("<h3>Agreement Summary</h3>")
    hcells = "".join(f"<th>{c}</th>" for c in ["Field", "Type"] + pair_cols)
    parts.append(f"<table><tr>{hcells}</tr>")
    for _, r in summary.iterrows():
        cells = f"<td class='field'>{r['field']}</td><td class='type'>{r['type']}</td>"
        for col in pair_cols:
            v = r.get(col, "—")
            cells += f"<td style='background:{pct_to_color(v)}'>{v}</td>"
        parts.append(f"<tr>{cells}</tr>")
    parts.append("</table>")

    # Per-paper detail
    for doi in dois:
        parts.append(f"<h3>{doi}</h3>")
        vhdrs = "".join(f"<th>{n}</th>" for n in coder_names)
        ahdrs = "".join(f"<th>{c}</th>" for c in pair_cols)
        parts.append(f"<table><tr><th>Field</th><th>Type</th>{vhdrs}{ahdrs}</tr>")
        for _, row in comp[comp["doi"] == doi].iterrows():
            vcells = "".join(f"<td>{row.get(n, '')}</td>" for n in coder_names)
            acells = "".join(
                f"<td class='{cell_class(row.get(col, ''))}'>{row.get(col, '')}</td>"
                for col in pair_cols
            )
            parts.append(
                f"<tr><td class='field'>{row['field']}</td>"
                f"<td class='type'>{row['field_type']}</td>"
                f"{vcells}{acells}</tr>"
            )
        parts.append("</table>")

    parts.append("</body></html>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


# ── PDF report ─────────────────────────────────────────────────────────────────

def write_pdf(
    round_name: str,
    comp: pd.DataFrame,
    summary: pd.DataFrame,
    coder_names: list[str],
    dois: list[str],
    out_path: Path,
) -> None:
    try:
        from itertools import combinations
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        )
    except ImportError:
        print("  reportlab not available — skipping PDF")
        return

    pairs     = list(combinations(coder_names, 2))
    pair_cols = [f"{a}_vs_{b}" for a, b in pairs]

    GREEN   = colors.HexColor("#21472E")
    RED     = colors.HexColor("#f8d7da")
    YELLOW  = colors.HexColor("#fff3cd")
    LGREEN  = colors.HexColor("#d4edda")
    LGREY   = colors.HexColor("#e9ecef")
    WHITE   = colors.white
    OFFWHT  = colors.HexColor("#f5f5f5")

    body  = ParagraphStyle("body",  fontSize=6, leading=8)
    bold  = ParagraphStyle("bold",  fontSize=6, leading=8,  fontName="Helvetica-Bold")
    title = ParagraphStyle("title", fontSize=13, leading=16, fontName="Helvetica-Bold",
                           textColor=GREEN)
    h3    = ParagraphStyle("h3",    fontSize=9, leading=12, fontName="Helvetica-Bold",
                           textColor=GREEN, spaceBefore=12, spaceAfter=4)
    small = ParagraphStyle("small", fontSize=5, leading=7, textColor=colors.grey)

    def esc(text, maxlen=180):
        s = str(text or "")[:maxlen]
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") or "—"

    def P(text, style=body):
        return Paragraph(esc(text), style)

    def agree_bg(val):
        return {"yes": LGREEN, "partial": YELLOW, "no": RED,
                "both_empty": LGREY}.get(val, WHITE)

    def pct_bg(pct_str):
        if not pct_str or pct_str == "—":
            return WHITE
        try:
            pct = int(pct_str.split("(")[1].split("%")[0])
        except (IndexError, ValueError):
            return WHITE
        return LGREEN if pct >= 80 else YELLOW if pct >= 50 else RED

    n_c = len(coder_names)
    n_p = len(pairs)
    scale = min(1.0, 27.0 / (2.5 + 1.5 + n_c * 4.5 + n_p * 1.4))
    col_w = ([2.5 * scale * cm, 1.5 * scale * cm]
             + [4.5 * scale * cm] * n_c
             + [1.4 * scale * cm] * n_p)
    hdr_row = (["Field", "Type"] + coder_names
               + [c.replace("_vs_", "\nvs\n") for c in pair_cols])

    BASE_STYLE = [
        ("BACKGROUND",   (0, 0), (-1, 0),  GREEN),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  WHITE),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 5),
        ("GRID",         (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",   (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 2),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, OFFWHT]),
    ]

    doc = SimpleDocTemplate(
        str(out_path), pagesize=landscape(A4),
        leftMargin=0.8 * cm, rightMargin=0.8 * cm,
        topMargin=1.0 * cm,  bottomMargin=1.0 * cm,
    )
    story = [
        Paragraph(f"{round_name} — Coder Comparison", title),
        Paragraph(f"{', '.join(coder_names)} — {len(dois)} papers", small),
        Spacer(1, 0.4 * cm),
    ]

    # Summary table
    story.append(Paragraph("Agreement Summary (scored fields)", h3))
    sum_data  = [["Field", "Type"] + pair_cols]
    sum_style = list(BASE_STYLE)
    for i, (_, r) in enumerate(summary.iterrows(), start=1):
        sum_data.append([P(r["field"], bold), P(r["type"], small)]
                        + [P(r.get(col, "—")) for col in pair_cols])
        for j, col in enumerate(pair_cols):
            sum_style.append(("BACKGROUND", (2 + j, i), (2 + j, i),
                               pct_bg(r.get(col, "—"))))
    # Scale summary columns so total fits within landscape A4 page width
    sum_avail = 27.0
    sum_raw   = 5.5 + 2.2 + n_p * 3.2
    sum_scale = min(1.0, sum_avail / sum_raw) if sum_raw > 0 else 1.0
    sum_col_w = ([5.5 * sum_scale * cm, 2.2 * sum_scale * cm]
                 + [3.2 * sum_scale * cm] * n_p)
    sum_t = Table(sum_data, colWidths=sum_col_w)
    sum_t.setStyle(TableStyle(sum_style))
    story += [sum_t, Spacer(1, 0.8 * cm)]

    # Per-paper tables
    for doi in dois:
        story.append(Paragraph(doi, h3))
        paper  = comp[comp["doi"] == doi]
        tdata  = [hdr_row]
        tstyle = list(BASE_STYLE)
        for i, (_, row) in enumerate(paper.iterrows(), start=1):
            tdata.append(
                [P(row["field"], bold), P(row["field_type"], small)]
                + [P(row.get(n, "")) for n in coder_names]
                + [P(row.get(col, "")) for col in pair_cols]
            )
            for j, col in enumerate(pair_cols):
                bg = agree_bg(row.get(col, ""))
                if bg != WHITE:
                    tstyle.append(("BACKGROUND", (2 + n_c + j, i), (2 + n_c + j, i), bg))
        t = Table(tdata, colWidths=col_w, repeatRows=1)
        t.setStyle(TableStyle(tstyle))
        story += [t, Spacer(1, 0.3 * cm)]

    try:
        doc.build(story)
    except Exception as e:
        print(f"  ⚠️  PDF build failed for {round_name} ({e.__class__.__name__}: {e}) — skipping PDF")


# ── per-round runner ───────────────────────────────────────────────────────────

def run_round(round_name: str, llm: pd.DataFrame) -> dict | None:
    """
    Run comparison for one round. Returns summary dict for cross-round table,
    or None if no coder files found.
    """
    coder_files = discover_coder_files(round_name)
    if not coder_files:
        print(f"  {round_name}: no human coder files found — skipping")
        return None

    print(f"\n{'─'*60}")
    print(f"  {round_name}: {list(coder_files.keys())}")
    print(f"{'─'*60}")

    # Load human coders
    coders: dict[str, pd.DataFrame] = {}
    for initials, path in coder_files.items():
        try:
            coders[initials] = load_coder(path)
            print(f"  Loaded {initials}: {path.name}  ({len(coders[initials])} papers)")
        except Exception as e:
            print(f"  WARNING: could not load {path.name}: {e}")

    if not coders:
        return None

    # Get DOIs for this round (union of all coders' papers)
    all_dois = sorted(set().union(*[set(df.index) for df in coders.values()]))

    # Add LLM as a coder
    llm_round = llm.reindex(all_dois)
    coders["LLM"] = llm_round
    coder_names = list(coders.keys())

    # Run comparison
    comp    = build_comparison(coders, all_dois)
    summary = build_summary(comp, coder_names)

    # Write outputs
    out = OUT_DIR / round_name
    out.mkdir(parents=True, exist_ok=True)

    comp.to_csv(out / f"comparison_{round_name}.csv", index=False)
    summary.to_csv(out / f"summary_{round_name}.csv", index=False)
    print(f"  Saved: comparison_{round_name}.csv  ({len(comp)} rows)")
    print(f"  Saved: summary_{round_name}.csv")

    write_html(round_name, comp, summary, coder_names, all_dois,
               out / f"comparison_{round_name}.html")
    print(f"  Saved: comparison_{round_name}.html")

    write_pdf(round_name, comp, summary, coder_names, all_dois,
              out / f"comparison_{round_name}.pdf")
    print(f"  Saved: comparison_{round_name}.pdf")

    # Return headline stats for cross-round summary
    from itertools import combinations
    pairs     = list(combinations(coder_names, 2))
    pair_cols = [f"{a}_vs_{b}" for a, b in pairs]
    headline  = {"round": round_name, "n_papers": len(all_dois),
                 "coders": ", ".join(c for c in coder_names if c != "LLM")}
    for col in pair_cols:
        sub   = summary[summary["field"] == INCLUSION_FIELD]
        headline[f"include_agree_{col}"] = sub[col].values[0] if len(sub) else "—"
    return headline


# ── cross-round summary ────────────────────────────────────────────────────────

def write_summary_all(headlines: list[dict]) -> None:
    if not headlines:
        return
    df = pd.DataFrame(headlines)
    out = OUT_DIR / "summary_all_rounds.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved cross-round summary: {out.relative_to(REPO_ROOT)}")

    # Simple HTML
    html = ["<!DOCTYPE html><html><head><meta charset='utf-8'>",
            "<title>All Rounds Summary</title>",
            "<style>body{font-family:Arial,sans-serif;font-size:12px;margin:20px}",
            "h2{color:#21472E}table{border-collapse:collapse}",
            "th{background:#21472E;color:white;padding:6px 10px}",
            "td{padding:5px 10px;border:1px solid #ddd}</style></head><body>",
            "<h2>Coding Rounds — Agreement Summary</h2>",
            "<table><tr>" +
            "".join(f"<th>{c}</th>" for c in df.columns) + "</tr>"]
    for _, row in df.iterrows():
        html.append("<tr>" + "".join(f"<td>{row[c]}</td>" for c in df.columns) + "</tr>")
    html.append("</table></body></html>")
    html_out = OUT_DIR / "summary_all_rounds.html"
    html_out.write_text("\n".join(html), encoding="utf-8")
    print(f"Saved: {html_out.relative_to(REPO_ROOT)}")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare human coder outputs against LLM coding."
    )
    parser.add_argument("--round",  action="append", dest="rounds", metavar="ROUND",
                        help="Round(s) to compare, e.g. --round FT-R1a --round FT-R2a")
    parser.add_argument("--all",    action="store_true",
                        help="Run all rounds that have human coder files")
    args = parser.parse_args()

    if not args.rounds and not args.all:
        parser.error("provide --round ROUND_NAME or --all")

    rounds = discover_all_rounds() if args.all else args.rounds

    if not rounds:
        print("No rounds with coder files found.")
        sys.exit(0)

    print(f"\n{'='*60}")
    print(f"  step14d_compare_coding")
    print(f"  Rounds: {', '.join(rounds)}")
    print(f"{'='*60}")

    llm = load_llm()
    print(f"LLM data loaded: {len(llm):,} papers")

    headlines = []
    for round_name in rounds:
        result = run_round(round_name, llm)
        if result:
            headlines.append(result)

    if len(headlines) > 1 or args.all:
        write_summary_all(headlines)

    print(f"\n{'='*60}")
    print(f"  Done — {len(headlines)} round(s) processed")
    print(f"  Outputs: {(OUT_DIR).relative_to(REPO_ROOT)}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
