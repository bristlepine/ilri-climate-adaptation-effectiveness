#!/usr/bin/env python3
"""
step7_scopus_check.py

Eligible-benchmark vs Scopus retrieval comparison WITH iteration tracking.

Compares Scopus retrieval (Step 2) against benchmark lists and eligible sets:

Benchmark list (FULL):
- Step5 wide eligibility file (step5_eligibility_wide.csv) merged with benchmark metadata
- Plotted as: step7_retrieval_analysis_all_2x2.png
- Archived as: history/iterXXX_benchmark_all_2x2.png

Secondary check list (FULL LIST OF SECONDARY CHECKS; ALL DECISIONS):
- From step6/eligibility_report_secondary_check_final.csv (this is the "secondary list")
- Plotted as: step7a_retrieval_analysis_all_2x2_secondary.png  ✅ (full secondary list; 2x2 like benchmark)
- Archived as: history/iterXXX_secondary_benchmark_all_2x2.png ✅

Eligible sets (OVERALL-ONLY bars):
- Baseline eligible (from step6 excel; fallback step5 passed): step7_retrieval_analysis_eligible_2x2.png
- Secondary eligible (from step6 secondary csv includes): step7a_retrieval_analysis_secondary_2x2.png

REMOVED (per request):
- history/iterXXX_full_list_2x2.png
- step7_retrieval_analysis_final_target_2x2.png
- step7_retrieval_analysis_all_2x2_secondary.png
- history/iterXXX_benchmark_all_2x2_secondary.png

Suggestions:
- step7_searchstring_suggestions.txt includes:
  - per-block routing
  - FINAL SUMMARY (deduped) WITH title_hits / abstract_hits / score ✅
"""

from __future__ import annotations

import os
import re
import json
import time
import html
import hashlib
import shutil
from typing import Any, Optional, List, Tuple, Dict
from collections import Counter, defaultdict

import requests
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    from docx import Document
    from docx.shared import Pt, Inches
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False
    print("⚠️ 'python-docx' not installed. Word files will not be generated. Run: pip install python-docx")


# -------------------------
# Visual palette
PALETTE_RETRIEVAL = {True: "#93c47d", False: "#e06666"}  # in scopus / missed
CATS_SOURCE = ["Peer Reviewed", "Grey Literature"]

# Inputs
SECONDARY_RESULTS_CSV_BASENAME = "eligibility_report_secondary_check_final.csv"
BASELINE_ELIGIBILITY_XLSX_BASENAME = "eligibility_report.xlsx"
BASELINE_ELIGIBILITY_SHEETNAME = "All Papers"
STEP1_QUERIES_JSON_BASENAME = "step1_queries.json"

# Crossref
CROSSREF_BASE = "https://api.crossref.org/works/"
CROSSREF_TIMEOUT = 7

# Suggestion filtering
STOPWORDS = {
    "the", "and", "for", "with", "from", "into", "over", "under", "between", "within",
    "towards", "toward", "among", "using", "use", "used", "based", "study", "studies",
    "analysis", "effects", "effect", "impact", "impacts", "results", "evidence",
    "case", "cases", "approach", "methods", "method", "data", "paper", "review",
    "climate", "change", "adaptation", "adaptive", "agriculture", "farm", "farming",
    "farmer", "farmers", "rural", "smallholder", "smallholders",
    "this", "that", "these", "those", "their", "there", "here", "where", "when",
    "what", "which", "who", "whom", "whose", "been", "being", "were", "was", "are",
    "also", "very", "more", "most", "less", "many", "some", "such", "than", "then",
    "onto", "across", "via", "per", "without",
}

GENERIC_BAD_SUGGESTIONS = {
    "journal", "international", "understanding", "measures", "determinants", "strategies",
    "central", "towards", "based", "using", "study", "studies", "analysis", "results",
    "evidence", "review", "effects", "effect", "impacts", "impact", "data", "paper",
}

# Domain / eligibility-ish terms: used to BOOST relevance (not hard-filter)
ELIGIBILITY_BOOST_TERMS = {
    # evaluation / impact
    "evaluation", "evaluations", "impact", "impacts", "effect", "effects", "effectiveness", "efficacy",
    "outcome", "outcomes", "performance", "assessment", "assessments", "evidence", "results",
    "randomized", "randomised", "trial", "trials", "quasi", "baseline",
    "counterfactual", "matching", "propensity", "regression", "instrumental",
    "difference", "differences", "rct", "experiment", "experimental",
    # interventions / adaptation
    "intervention", "interventions", "program", "programme", "programs", "policy", "policies",
    "insurance", "index", "irrigation", "extension", "advisory", "forecast", "forecasts",
    "climatesmart", "climate-smart", "agroforestry", "conservation", "mulching", "fertilizer", "fertiliser",
    "seed", "seeds", "varieties", "drought", "droughts", "heat", "flood", "floods",
    "resilience", "resilient", "risk", "risks", "coping", "mitigation",
    # agriculture / livelihoods
    "yield", "yields", "income", "incomes", "profit", "profits", "livelihood", "livelihoods",
    "nutrition", "food", "security", "livestock", "crop", "crops", "pastoral", "pastoralist",
    "smallholder", "smallholders", "farmer", "farmers", "household", "households",
}

# Routing dictionaries (light heuristics to suggest WHICH step1 block to edit)
ROUTE_TOKENS = {
    "M__measurement_eval": {
        "evaluation", "evaluations", "impact", "impacts", "effect", "effects", "effectiveness",
        "assessment", "assessments", "indicator", "indicators", "metric", "metrics", "monitor", "monitoring",
        "survey", "surveys", "randomized", "randomised", "trial", "trials", "rct", "experiment",
        "counterfactual", "matching", "propensity", "baseline", "difference", "differences", "regression",
        "quasi", "method", "methods", "mel", "m&e", "evaluation", "evaluat",
    },
    "C_context_climate__hazards": {
        "drought", "droughts", "flood", "floods", "heat", "heatwave", "heatwaves", "rainfall",
        "aridity", "storm", "storms", "cyclone", "cyclones", "hurricane", "typhoon", "salinity",
        "intrusion", "shock", "shocks", "hazard", "hazards",
    },
    "C_context_climate__climate_framing": {
        "climate", "variability", "resilient", "resilience", "risk", "risks", "hazard", "hazards",
        "exposure", "shock", "shocks", "climatesmart", "csa",
    },
    "C_context_agriculture__ag_systems": {
        "agriculture", "agricultural", "farm", "farming", "crop", "crops", "livestock",
        "irrigation", "rainfed", "rangeland", "fisheries", "aquaculture", "forestry",
        "agroforestry", "silvopastoral", "pastoral", "pastoralist",
    },
    "C_concept__adaptation_actions": {
        "adoption", "uptake", "participatory", "governance", "learning", "practice", "practices",
        "behaviour", "behavior", "decision", "decisions", "strategy", "strategies", "adjustment",
        "adjust", "change", "changes", "coping", "insurance", "extension", "advisory",
    },
    "C_concept__capacities_knowledge": {
        "capacity", "capacities", "knowledge", "information", "awareness", "skills",
        "training", "literacy",
    },
    "C_concept__livelihood_productivity": {
        "income", "incomes", "yield", "yields", "productivity", "profit", "profits", "livelihood",
        "livelihoods",
    },
    "C_concept__resilience_outcomes": {
        "resilience", "resilient", "wellbeing", "well-being", "vulnerability", "exposure",
        "risk", "risks", "outcome", "outcomes",
    },
    "C_concept__maladaptation": {
        "maladaptation", "maladaptive", "maladapt",
    },
    "P__smallholder_small_scale": {
        "smallholder", "smallholders", "subsistence", "resourcepoor", "resource-poor",
        "marginal", "smallscale", "small-scale",
    },
    "P__households_poverty": {
        "household", "households", "poverty", "poor", "lowincome", "low-income", "family",
    },
    "P__livestock_pastoral": {
        "pastoral", "pastoralist", "herder", "livestock", "dairy",
    },
    "P__fisheries_aquaculture": {
        "fishery", "fisheries", "fisher", "fisherfolk", "aquaculture", "shrimp",
    },
    "P__agroforestry_tree": {
        "agroforestry", "silvopastoral", "tree", "trees", "forest", "fruit",
    },
    "P__marginalized_groups": {
        "women", "female", "youth", "indigenous", "tribal", "ethnic", "migrant",
        "landless", "tenant", "sharecropper",
    },
}


# -------------------------
# Helpers
def _now_local_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _safe_str(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, float) and pd.isna(val):
        return ""
    return html.unescape(str(val)).strip()


def clean_text(text: Any) -> str:
    if not isinstance(text, str):
        text = _safe_str(text)
    return re.sub(r"[^a-z0-9]", "", text.lower())


def find_col(df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
    kw_lower = [k.lower() for k in keywords]
    for col in df.columns:
        if str(col).strip().lower() in kw_lower:
            return col
    for col in df.columns:
        col_l = str(col).lower()
        for k in keywords:
            k_l = k.lower()
            if len(k_l) > 2 and k_l in col_l:
                return col
    return None


def _clean_doi(val: Any) -> str:
    s = _safe_str(val).lower().strip()
    for p in ("https://doi.org/", "http://doi.org/", "doi:"):
        if s.startswith(p):
            s = s[len(p):]
    return s.strip().rstrip(" .),;]}>").strip()


def _decision_is_include(val: Any) -> bool:
    v = str(val or "").strip().lower()
    return bool(re.search(r"\b(include|included|pass|yes)\b", v))


def _map_source_type(val: Any) -> str:
    v = str(val or "").lower()
    if any(k in v for k in ["grey", "report", "thesis", "working paper", "brief"]):
        return "Grey Literature"
    return "Peer Reviewed"


def _simple_status(val: Any) -> str:
    v = str(val or "").lower()
    if "include" in v or "pass" in v or re.fullmatch(r"\s*yes\s*", v):
        return "Passed"
    if "unclear" in v or "pending" in v or "no_abstract" in v:
        return "Unclear"
    return "Failed"


def _read_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


def _load_jsonl(path: str) -> List[dict]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _write_jsonl_append(path: str, obj: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _resolve_benchmark_csv(out_root: str, configured: Optional[str]) -> str:
    if configured:
        p = str(configured).strip()
        if p and os.path.exists(p):
            return p
        p2 = os.path.join(out_root, p)
        if os.path.exists(p2):
            return p2

    fallback = os.path.join(out_root, "step3", "step3_benchmark_list.csv")
    if os.path.exists(fallback):
        return fallback
    return ""


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _archive_copy(src_path: str, dst_path: str) -> None:
    try:
        if src_path and os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
    except Exception:
        pass


def _file_fingerprint(path: Optional[str]) -> str:
    if not path or not os.path.exists(path):
        return "missing"
    try:
        st = os.stat(path)
        return f"{os.path.basename(path)}|{st.st_size}|{int(st.st_mtime)}"
    except Exception:
        return f"{os.path.basename(path)}|unstatable"


# -------------------------
# Crossref fetch + citation builder
def fetch_crossref_metadata(doi: str) -> dict:
    clean_doi = _clean_doi(doi)
    if not clean_doi:
        return {}

    url = f"{CROSSREF_BASE}{clean_doi}"
    try:
        r = requests.get(url, timeout=CROSSREF_TIMEOUT)
        if r.status_code != 200:
            return {}
        data = r.json().get("message", {}) or {}

        authors_list = data.get("author", []) or []
        author_parts = []
        for a in authors_list:
            family = a.get("family")
            given = a.get("given")
            if family:
                name = family
                if given:
                    name += f", {given[0]}."
                author_parts.append(name)
        if len(author_parts) > 3:
            author_str = ", ".join(author_parts[:3]) + " et al."
        else:
            author_str = ", ".join(author_parts)

        container = data.get("container-title", []) or []
        journal = container[0] if container else ""

        issued = (data.get("issued", {}) or {}).get("date-parts", [[None]])[0][0]
        year = str(issued) if issued else ""

        return {"fetched_authors": author_str, "fetched_journal": journal, "fetched_year": year}
    except Exception:
        return {}


def build_citation(row: dict) -> str:
    author = _safe_str(row.get("fetched_authors")) or _safe_str(row.get("author_names")) or _safe_str(row.get("authors")) or "Unknown Author"

    year_raw = _safe_str(row.get("fetched_year")) or _safe_str(row.get("year_scopus")) or _safe_str(row.get("year"))
    y_match = re.search(r"\d{4}", year_raw)
    year_str = y_match.group(0) if y_match else "n.d."

    title = (_safe_str(row.get("title")) or _safe_str(row.get("title_scopus"))).rstrip(". ")

    journal = _safe_str(row.get("fetched_journal")) or _safe_str(row.get("publicationName"))

    doi = _safe_str(row.get("doi")) or _safe_str(row.get("doi_scopus"))
    if doi and not doi.lower().startswith("http"):
        doi = f"https://doi.org/{_clean_doi(doi)}"

    parts = [author, f"({year_str})"]
    if title:
        parts.append(f"'{title}'.")
    if journal:
        parts.append(f"{journal}.")
    else:
        st = _safe_str(row.get("source_type"))
        parts.append("[Grey Literature]." if "Grey" in st else "[Journal unavailable].")
    if doi:
        parts.append(doi)

    return " ".join([p for p in parts if p]).strip()


# -------------------------
# Word export helpers
def export_to_word_citations(df: pd.DataFrame, title: str, filename: str) -> None:
    if not HAS_DOCX:
        return
    doc = Document()
    doc.add_heading(title, level=1)

    if "citation" not in df.columns:
        doc.add_paragraph("No citation column found.")
        doc.save(filename)
        print(f"📄 Saved Word document: {filename}")
        return

    for _, row in df.iterrows():
        c = _safe_str(row.get("citation"))
        if c:
            p = doc.add_paragraph(c, style="List Number")
            p.paragraph_format.space_after = Pt(6)

    doc.save(filename)
    print(f"📄 Saved Word document: {filename}")


def _build_appendix_from_evolution(
    appendix_path: str,
    evo_rows: List[dict],
    progress_chart_path: str,
) -> None:
    if not HAS_DOCX:
        return

    doc = Document()
    doc.add_heading("Appendix: Search String Evolution Log", level=1)

    doc.add_paragraph("Running summary (updated each iteration):")
    if os.path.exists(progress_chart_path):
        try:
            doc.add_picture(progress_chart_path, width=Inches(6.5))
        except Exception:
            doc.add_paragraph("[Could not embed progress chart]")
    doc.add_paragraph("")

    for r in evo_rows:
        ts = r.get("timestamp_local", "")
        it = r.get("iteration", "")
        ss = r.get("search_string", "")
        src = r.get("search_source", "")
        sh = r.get("search_hash", "")
        elig = r.get("eligibility_sources", {}) or {}
        cov = r.get("coverage", {}) or {}
        cov_final = cov.get("final_target", {}) or {}
        missed_n = cov.get("final_missed_n", 0)
        sugg = r.get("suggestion_text", "")

        figs = (r.get("figures", {}) or {})
        fig_comp = figs.get("coverage_comparison", "")
        fig_base = figs.get("baseline_eligible_plot", "")
        fig_sec = figs.get("secondary_eligible_plot", "")
        fig_benchmark = figs.get("benchmark_all_plot", "")
        fig_secondary_full = figs.get("secondary_benchmark_all_plot", "")

        doc.add_page_break()
        doc.add_heading(f"Iteration {it} — {ts}", level=2)

        doc.add_paragraph("Summary:")
        doc.add_paragraph(f"Search hash: {sh} (source: {src})", style="List Bullet")
        doc.add_paragraph(f"Final target: {elig.get('final_target','')}", style="List Bullet")
        doc.add_paragraph(
            f"Final coverage: {cov_final.get('n_in_overall',0)}/{cov_final.get('n_total_overall',0)} "
            f"({float(cov_final.get('pct_overall',0.0)):.1f}%)",
            style="List Bullet",
        )
        doc.add_paragraph(f"Final missed count: {missed_n}", style="List Bullet")

        doc.add_paragraph("Search string used:")
        doc.add_paragraph(ss or "[missing]")

        doc.add_paragraph("Suggested improvements:")
        doc.add_paragraph(sugg or "[none]")

        for cap, path in [
            ("Figure: Coverage comparison (baseline vs secondary) — Overall only", fig_comp),
            ("Figure: Baseline eligible retrieval (overall only)", fig_base),
            ("Figure: Secondary eligible retrieval (overall only)", fig_sec),
            ("Figure: Full benchmark list retrieval (2x2 by status)", fig_benchmark),
            ("Figure: Full secondary check list retrieval (2x2 by status)", fig_secondary_full),
        ]:
            if path and os.path.exists(path):
                doc.add_paragraph(cap)
                try:
                    doc.add_picture(path, width=Inches(6.5))
                except Exception:
                    doc.add_paragraph(f"[Could not embed figure: {os.path.basename(path)}]")
                doc.add_paragraph("")

    doc.save(appendix_path)
    print(f"📄 Rebuilt appendix: {appendix_path}")


# -------------------------
# Eligibility loaders + secondary list loader
def _eligibility_key_from_row(doi_val: Any, title_val: Any) -> str:
    doi = _clean_doi(doi_val)
    if doi:
        return f"doi:{doi}"
    t = _safe_str(title_val)
    if t:
        return f"title:{clean_text(t)}"
    return ""


def _load_baseline_eligible_keys(step6_dir: str, fallback_step5_df: pd.DataFrame) -> Tuple[set, str]:
    xlsx_path = os.path.join(step6_dir, BASELINE_ELIGIBILITY_XLSX_BASENAME)
    if os.path.exists(xlsx_path):
        try:
            df = pd.read_excel(xlsx_path, sheet_name=BASELINE_ELIGIBILITY_SHEETNAME)
        except Exception:
            df = pd.read_excel(xlsx_path)

        doi_col = find_col(df, ["doi", "DOI"])
        title_col = find_col(df, ["title", "Title"])
        dec_col = find_col(df, ["final_decision", "Final Decision", "decision"])

        if dec_col:
            elig = df[df[dec_col].apply(_decision_is_include)].copy()
            keys = set()
            for _, r in elig.iterrows():
                k = _eligibility_key_from_row(r.get(doi_col, ""), r.get(title_col, ""))
                if k:
                    keys.add(k)
            return keys, f"baseline_from_step6_excel:{os.path.basename(xlsx_path)}"

    # fallback: Step5 passed
    s5_title_col = find_col(fallback_step5_df, ["title", "study"])
    s5_doi_col = find_col(fallback_step5_df, ["doi", "DOI"])
    s5_status_col = find_col(fallback_step5_df, ["final_decision", "decision"])
    if not s5_status_col:
        return set(), "baseline_fallback:missing_status_col"

    passed = fallback_step5_df[fallback_step5_df[s5_status_col].apply(_decision_is_include)].copy()
    keys = set()
    for _, r in passed.iterrows():
        k = _eligibility_key_from_row(r.get(s5_doi_col, ""), r.get(s5_title_col, ""))
        if k:
            keys.add(k)
    return keys, "baseline_fallback:step5_passed"


def _load_secondary_list_decisions(step6_dir: str) -> Tuple[Dict[str, str], str, Optional[str]]:
    """
    Loads the FULL secondary check list (all decisions), returning:
      - key -> decision_str (raw)
      - source label
      - csv path
    key is doi:... if present else title:clean_title
    """
    csv_path = os.path.join(step6_dir, SECONDARY_RESULTS_CSV_BASENAME)
    if not os.path.exists(csv_path):
        return {}, "secondary_missing", None

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return {}, f"secondary_unreadable:{os.path.basename(csv_path)}", csv_path

    doi_col = find_col(df, ["doi", "DOI"])
    title_col = find_col(df, ["title", "Title"])
    dec_col = find_col(df, ["final_decision", "Final Decision", "decision", "Decision"])

    if not title_col:
        return {}, f"secondary_no_title_col:{os.path.basename(csv_path)}", csv_path
    if not dec_col:
        # still return membership (but no decision)
        dec_col = None

    out: Dict[str, str] = {}
    for _, r in df.iterrows():
        doi = _clean_doi(r.get(doi_col)) if doi_col else ""
        title = _safe_str(r.get(title_col))
        key = f"doi:{doi}" if doi else f"title:{clean_text(title)}"
        if not key or key.endswith(":"):
            continue
        out[key] = _safe_str(r.get(dec_col)) if dec_col else ""
    return out, f"secondary_from_csv:{os.path.basename(csv_path)}", csv_path


def _load_secondary_eligible_keys(step6_dir: str) -> Tuple[set, str, Optional[str]]:
    decisions, source, csv_path = _load_secondary_list_decisions(step6_dir)
    if not decisions:
        return set(), source, csv_path
    keys = {k for k, v in decisions.items() if _decision_is_include(v)}
    return keys, source, csv_path


# -------------------------
# Abstract lookup (Step 4)
def _load_abstract_lookup(out_root: str) -> Tuple[Dict[str, str], Dict[str, str], str]:
    step4_path = os.path.join(out_root, "step4", "step4_abstracts.csv")
    if not os.path.exists(step4_path):
        return {}, {}, "abstracts_missing"

    try:
        df = pd.read_csv(step4_path)
    except Exception:
        return {}, {}, f"abstracts_unreadable:{os.path.basename(step4_path)}"

    title_col = find_col(df, ["title", "Title", "study"])
    doi_col = find_col(df, ["doi", "DOI"])
    abs_col = find_col(df, ["abstract", "Abstract", "description", "summary"])

    if not abs_col:
        return {}, {}, f"abstracts_no_abstract_col:{os.path.basename(step4_path)}"

    doi_to_abs: Dict[str, str] = {}
    title_to_abs: Dict[str, str] = {}

    for _, r in df.iterrows():
        abs_txt = _safe_str(r.get(abs_col))
        if not abs_txt:
            continue

        doi = _clean_doi(r.get(doi_col)) if doi_col else ""
        if doi:
            doi_to_abs[doi] = abs_txt

        title = _safe_str(r.get(title_col)) if title_col else ""
        if title:
            title_to_abs[clean_text(title)] = abs_txt

    return doi_to_abs, title_to_abs, f"abstracts_from:{os.path.basename(step4_path)}"


# -------------------------
# Step1 query blocks loader (for suggestion routing)
def _load_step1_blocks(out_root: str) -> Tuple[Dict[str, str], str]:
    p = os.path.join(out_root, "step1", STEP1_QUERIES_JSON_BASENAME)
    if not os.path.exists(p):
        return {}, "step1_queries_missing"

    try:
        with open(p, "r", encoding="utf-8") as f:
            rows = json.load(f)
    except Exception:
        return {}, "step1_queries_unreadable"

    blocks: Dict[str, str] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        name = r.get("name")
        query = r.get("query")
        if isinstance(name, str) and isinstance(query, str) and name.strip():
            blocks[name.strip()] = query.strip()

    return blocks, f"step1_queries_from:{os.path.basename(p)}"


# -------------------------
# Search string detection + suggestions (title + abstract; eligibility-aware; per-block routing)
def _get_search_string(config: dict, out_root: str) -> Tuple[str, str]:
    if config.get("search_string"):
        return str(config["search_string"]).strip(), "config.search_string"

    step1_txt = os.path.join(out_root, "step1", "step1_total_query.txt")
    s = _read_text_file(step1_txt)
    if s:
        return s, f"file:{step1_txt}"

    return "", "missing"


def _tokenize(text: str) -> List[str]:
    t = re.sub(r"[^a-z0-9\s\-]", " ", (text or "").lower())
    toks = []
    for x in t.split():
        if len(x) < 4:
            continue
        toks.append(x.strip("-"))
    return toks


def _extract_author_tokens(df: pd.DataFrame) -> set:
    toks = set()
    for col in ["fetched_authors", "author_names"]:
        if col not in df.columns:
            continue
        for v in df[col].dropna().astype(str).tolist():
            for t in re.findall(r"[a-z]{4,}", v.lower()):
                toks.add(t)
    return toks


def _anchors_from_search_string(search_string: str) -> set:
    if not search_string:
        return set()
    anchors = set(re.findall(r"[a-z]{4,}", search_string.lower()))
    anchors -= STOPWORDS
    anchors -= GENERIC_BAD_SUGGESTIONS
    return anchors


def _route_term_to_block(term: str, step1_blocks: Dict[str, str]) -> str:
    t = (term or "").lower().strip()
    if not t:
        return "UNASSIGNED"

    for block, vocab in ROUTE_TOKENS.items():
        if t in vocab and (block in step1_blocks):
            return block

    for block_name, q in step1_blocks.items():
        if t in (q or "").lower():
            return block_name

    return "UNASSIGNED"


def _suggest_improvements_by_block(
    search_string: str,
    missed_records: List[Dict[str, str]],
    author_tokens: set,
    step1_blocks: Dict[str, str],
    top_k: int = 18,
) -> Tuple[str, List[str]]:
    ss = (search_string or "").lower()
    anchors = _anchors_from_search_string(search_string)

    counter = Counter()
    evidence: Dict[str, Dict[str, int]] = {}

    for rec in missed_records:
        title = (rec.get("title") or "").lower()
        abstract = (rec.get("abstract") or "").lower()

        record_text = f"{title} {abstract}".strip()
        anchor_hit = any(a in record_text for a in anchors) if anchors else True
        anchor_mult = 2.0 if anchor_hit else 1.0

        title_toks = _tokenize(title)
        abs_toks = _tokenize(abstract)

        def _add(tok: str, base_w: float, where: str):
            if not tok:
                return
            if not re.fullmatch(r"[a-z]+", tok):
                return
            if tok in STOPWORDS or tok in GENERIC_BAD_SUGGESTIONS:
                return
            if tok in author_tokens:
                return
            if tok in ss:
                return

            boost = 1.0
            if tok in ELIGIBILITY_BOOST_TERMS:
                boost *= 2.25
            if anchors and any((tok.startswith(a[:5]) or a.startswith(tok[:5])) for a in anchors if len(a) >= 5 and len(tok) >= 5):
                boost *= 1.35

            w = base_w * anchor_mult * boost
            counter[tok] += w

            if tok not in evidence:
                evidence[tok] = {"title_hits": 0, "abstract_hits": 0}
            if where == "title":
                evidence[tok]["title_hits"] += 1
            else:
                evidence[tok]["abstract_hits"] += 1

        for t in title_toks:
            _add(t, base_w=2.0, where="title")
        for t in abs_toks:
            _add(t, base_w=1.0, where="abstract")

    if not counter:
        return "No eligible, non-noise missed-title/abstract keywords found to suggest additions.", []

    picks = counter.most_common(max(60, top_k * 3))

    by_block: Dict[str, List[dict]] = defaultdict(list)
    unassigned: List[dict] = []

    for tok, score in picks:
        ev = evidence.get(tok, {"title_hits": 0, "abstract_hits": 0})
        routed = _route_term_to_block(tok, step1_blocks)
        entry = {
            "term": tok,
            "score": float(score),
            "title_hits": int(ev.get("title_hits", 0)),
            "abstract_hits": int(ev.get("abstract_hits", 0)),
            "block": routed,
        }
        if routed == "UNASSIGNED":
            unassigned.append(entry)
        else:
            by_block[routed].append(entry)

    PER_BLOCK_MAX = 8
    for b in list(by_block.keys()):
        by_block[b] = sorted(by_block[b], key=lambda x: x["score"], reverse=True)[:PER_BLOCK_MAX]

    ordered_blocks = [
        "M__measurement_eval",
        "C_concept__adaptation_actions",
        "C_concept__resilience_outcomes",
        "C_concept__capacities_knowledge",
        "C_concept__livelihood_productivity",
        "C_context_climate__hazards",
        "C_context_climate__climate_framing",
        "C_context_agriculture__ag_systems",
        "C_concept__maladaptation",
        "P__smallholder_small_scale",
        "P__households_poverty",
        "P__livestock_pastoral",
        "P__fisheries_aquaculture",
        "P__agroforestry_tree",
        "P__marginalized_groups",
    ]
    for b in step1_blocks.keys():
        if b not in ordered_blocks and not b.endswith("__ALL"):
            ordered_blocks.append(b)

    flat: List[dict] = []
    for b in ordered_blocks:
        flat.extend(by_block.get(b, []))
        if len(flat) >= top_k:
            break
    flat = flat[:top_k]

    capped_by_block: Dict[str, List[dict]] = defaultdict(list)
    capped_unassigned: List[dict] = []
    for e in flat:
        if e["block"] == "UNASSIGNED":
            capped_unassigned.append(e)
        else:
            capped_by_block[e["block"]].append(e)

    # deduped terms + keep their hits for final summary
    seen = set()
    deduped_terms: List[str] = []
    deduped_meta: Dict[str, Dict[str, float]] = {}
    for e in flat:
        t = e["term"]
        if t in seen:
            continue
        seen.add(t)
        deduped_terms.append(t)
        deduped_meta[t] = {
            "score": float(e["score"]),
            "title_hits": int(e["title_hits"]),
            "abstract_hits": int(e["abstract_hits"]),
        }

    lines: List[str] = []
    lines.append("Candidate terms from missed eligible titles + abstracts (filtered + eligibility-weighted)")
    lines.append("and routed to your Step1 query blocks (so you know WHERE to add them).")
    lines.append("")

    def _fmt_block_snippet(_block_name: str, terms: List[str]) -> str:
        return " OR ".join([f"{t}*" for t in terms])

    any_block_output = False
    for b in ordered_blocks:
        entries = capped_by_block.get(b, [])
        if not entries:
            continue
        any_block_output = True

        terms = [e["term"] for e in entries]
        lines.append(f"BLOCK: {b}")
        if b in step1_blocks:
            lines.append("  Where to add: inside this block's TITLE-ABS-KEY(...) OR list in step1_queries.json")
        else:
            lines.append("  Where to add: (block not found in step1_queries.json; add to the closest concept block)")
        lines.append("  Suggested terms:")
        lines.append("    " + ", ".join(terms))
        lines.append("  Paste-ready OR-snippet:")
        lines.append("    " + _fmt_block_snippet(b, terms))
        lines.append("  Rationale (weighted score; where they occur):")
        for e in entries:
            lines.append(f"    - {e['term']}: {e['score']:.2f} (title_hits={e['title_hits']}, abstract_hits={e['abstract_hits']})")
        lines.append("")

    if not any_block_output:
        lines.append("No routed terms found (unexpected).")
        lines.append("")

    if capped_unassigned:
        terms = [e["term"] for e in capped_unassigned]
        lines.append("UNASSIGNED (could not confidently map to a block; review manually):")
        lines.append("  " + ", ".join(terms))
        lines.append("")

    if deduped_terms:
        lines.append("FINAL SUMMARY (deduped across all blocks; with title/abstract hit counts):")
        for t in deduped_terms:
            m = deduped_meta.get(t, {})
            lines.append(f"  - {t}: title_hits={int(m.get('title_hits',0))}, abstract_hits={int(m.get('abstract_hits',0))}, score={float(m.get('score',0.0)):.2f}")
        lines.append("")
        lines.append("FINAL OR-SNIPPET (wildcarded; paste terms into the appropriate blocks above, not necessarily into TOTAL__ALL directly):")
        lines.append("  " + " OR ".join([f"{t}*" for t in deduped_terms]))
        lines.append("")

    return "\n".join(lines).strip(), deduped_terms


# -------------------------
# Plotting
def _stamp_figure(fig, stamp: str) -> None:
    fig.text(0.99, 0.006, stamp, ha="right", va="bottom", fontsize=8, color="black")


def _plot_eligible_overall(df: pd.DataFrame, out_path: str, title: str, timestamp: str) -> None:
    n_total = int(len(df))
    n_in = int(df["in_scopus_retrieval"].sum()) if n_total else 0
    n_missed = n_total - n_in
    pct = (100.0 * n_in / n_total) if n_total else 0.0

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.bar(["Eligible"], [n_in], label="In Scopus", color=PALETTE_RETRIEVAL[True], alpha=0.9)
    ax.bar(["Eligible"], [n_missed], bottom=[n_in], label="Missed", color=PALETTE_RETRIEVAL[False], alpha=0.9)

    ymax = max(1, int((n_total * 1.35)))
    ax.set_ylim(0, ymax)
    ax.set_ylabel("Count")
    ax.set_title(title, fontweight="bold")

    if n_in > 0:
        ax.text(0, n_in / 2, f"{n_in}", ha="center", va="center", color="white", fontweight="bold")
    if n_missed > 0:
        ax.text(0, n_in + n_missed / 2, f"{n_missed}", ha="center", va="center", color="white", fontweight="bold")
    ax.text(0, n_total + max(0.5, n_total * 0.03), f"{pct:.1f}%", ha="center", va="bottom", fontweight="bold")

    ax.legend(frameon=False, loc="upper right")
    _stamp_figure(fig, f"Generated: {timestamp}")

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.16)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"📊 Chart saved: {out_path}")


def _plot_retrieval_2x2(df: pd.DataFrame, out_path: str, title: str, timestamp: str) -> None:
    plot_configs = [
        {"title": "All Studies", "filter": lambda d: d},
        {"title": "Passed Studies", "filter": lambda d: d[d["simple_status"] == "Passed"]},
        {"title": "Unclear Studies", "filter": lambda d: d[d["simple_status"] == "Unclear"]},
        {"title": "Failed Studies", "filter": lambda d: d[d["simple_status"] == "Failed"]},
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()

    for i, cfg in enumerate(plot_configs):
        ax = axes[i]
        subset = cfg["filter"](df)
        counts = subset.groupby(["source_type", "in_scopus_retrieval"]).size().unstack(fill_value=0)
        counts = counts.reindex(CATS_SOURCE).fillna(0)
        if True not in counts.columns:
            counts[True] = 0
        if False not in counts.columns:
            counts[False] = 0

        p1 = ax.bar(counts.index, counts[True], label="In Scopus", color=PALETTE_RETRIEVAL[True], alpha=0.9)
        p2 = ax.bar(counts.index, counts[False], bottom=counts[True], label="Missed", color=PALETTE_RETRIEVAL[False], alpha=0.9)

        max_height = (counts[True] + counts[False]).max()
        if max_height > 0:
            ax.set_ylim(0, max_height * 1.20)

        ax.set_title(f"{cfg['title']} (n={len(subset)})", fontweight="bold")

        for bar_group in (p1, p2):
            for bar in bar_group:
                h = bar.get_height()
                if h > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        bar.get_y() + h / 2.0,
                        f"{int(h)}",
                        ha="center",
                        va="center",
                        color="white",
                        fontweight="bold",
                    )

        totals = counts[True] + counts[False]
        for idx, cat in enumerate(counts.index):
            total = totals.loc[cat]
            found = counts.loc[cat, True]
            if total > 0:
                pct = (found / total) * 100
                ax.text(
                    idx,
                    total + (max_height * 0.03 if max_height else 0.5),
                    f"{pct:.1f}%",
                    ha="center",
                    va="bottom",
                    color="black",
                    fontweight="bold",
                    fontsize=10,
                )

    fig.suptitle(title, fontsize=16, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=2, frameon=False)

    _stamp_figure(fig, f"Generated: {timestamp}")

    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.12)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"📊 Chart saved: {out_path}")


def _plot_coverage_comparison_overall_only(
    base_stats: dict,
    sec_stats: Optional[dict],
    out_path: str,
    timestamp: str,
) -> None:
    base_pct = float(base_stats.get("pct_overall", 0.0))
    sec_pct = float(sec_stats.get("pct_overall", 0.0)) if sec_stats else None

    fig, ax = plt.subplots(figsize=(8.8, 5.2))

    labels = ["Overall"]
    x = [0]
    width = 0.35 if sec_pct is not None else 0.6

    b1 = ax.bar(
        [i - (width / 2 if sec_pct is not None else 0) for i in x],
        [base_pct],
        width,
        label="Baseline eligible",
        edgecolor="black",
    )
    b2 = None
    if sec_pct is not None:
        b2 = ax.bar(
            [i + width / 2 for i in x],
            [sec_pct],
            width,
            label="Secondary eligible",
            edgecolor="black",
        )

    ax.set_ylim(0, 105)
    ax.set_ylabel("Coverage in Scopus (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    def _label_bars(bars):
        if not bars:
            return
        for br in bars:
            v = br.get_height()
            ax.text(
                br.get_x() + br.get_width() / 2,
                v + 2,
                f"{v:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    _label_bars(b1)
    if b2 is not None:
        _label_bars(b2)

    title = "Eligible Benchmark Coverage in Scopus (Baseline vs Secondary)" if sec_pct is not None else "Eligible Benchmark Coverage in Scopus (Baseline)"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    sub = f"Baseline: {base_stats['n_in_overall']}/{base_stats['n_total_overall']}"
    if sec_pct is not None:
        sub += f" | Secondary: {sec_stats['n_in_overall']}/{sec_stats['n_total_overall']}"
    ax.text(0.5, 1.02, sub, transform=ax.transAxes, ha="center", va="bottom", fontsize=10)

    ax.legend(frameon=False, loc="upper right")
    _stamp_figure(fig, f"Generated: {timestamp}")

    plt.tight_layout(rect=[0, 0.08, 1, 0.92])
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"📊 Chart saved: {out_path}")


def _plot_progress_over_iterations(evo_rows: List[dict], out_path: str, timestamp: str) -> None:
    if not evo_rows:
        return

    xs, ys = [], []
    for r in evo_rows:
        it = r.get("iteration")
        cov_final = (r.get("coverage", {}) or {}).get("final_target", {}) or {}
        xs.append(it)
        ys.append(float(cov_final.get("pct_overall", 0.0)))

    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    ax.plot(xs, ys, marker="o")
    ax.set_ylim(0, 105)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Final Target Coverage (%)")
    ax.set_title("Scopus Coverage Progress Over Iterations", fontweight="bold")

    if xs and ys:
        ax.text(xs[-1], min(105, ys[-1] + 3), f"{ys[-1]:.1f}%", ha="center", va="bottom", fontweight="bold")

    _stamp_figure(fig, f"Generated: {timestamp}")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.16)
    plt.savefig(out_path, dpi=300)
    plt.close()


# -------------------------
# Core logic
def _merge_scopus_metadata(step_df: pd.DataFrame, scopus_df: pd.DataFrame) -> Tuple[pd.DataFrame, set, set]:
    sc_title_col = find_col(scopus_df, ["Title", "Article Title"])
    sc_doi_col = find_col(scopus_df, ["DOI", "doi"])
    if not sc_title_col:
        raise RuntimeError("Scopus file missing a title column (expected Title / Article Title).")

    scopus_df = scopus_df.copy()
    scopus_df["clean_title"] = scopus_df[sc_title_col].apply(clean_text)
    scopus_titles = set(scopus_df["clean_title"].dropna().unique())
    scopus_titles.discard("")

    scopus_dois = set()
    if sc_doi_col:
        scopus_df["clean_doi"] = scopus_df[sc_doi_col].apply(clean_text)
        scopus_dois = set(scopus_df["clean_doi"].dropna().unique())
        scopus_dois.discard("")

    sc_author_col = find_col(scopus_df, ["Authors", "Author", "author_names", "Creator"])
    sc_journal_col = find_col(scopus_df, ["Source title", "publicationName", "Journal"])
    sc_date_col = find_col(scopus_df, ["Year", "coverDate", "Date"])

    needed_cols = [sc_title_col]
    if sc_author_col:
        needed_cols.append(sc_author_col)
    if sc_journal_col:
        needed_cols.append(sc_journal_col)
    if sc_date_col:
        needed_cols.append(sc_date_col)
    if sc_doi_col:
        needed_cols.append(sc_doi_col)

    scopus_meta = scopus_df[needed_cols].dropna(subset=[sc_title_col]).copy()
    rename_map = {sc_title_col: "title_scopus"}
    if sc_author_col:
        rename_map[sc_author_col] = "author_names"
    if sc_journal_col:
        rename_map[sc_journal_col] = "publicationName"
    if sc_date_col:
        rename_map[sc_date_col] = "year_scopus"
    if sc_doi_col:
        rename_map[sc_doi_col] = "doi_scopus"
    scopus_meta = scopus_meta.rename(columns=rename_map)
    scopus_meta["match_title"] = scopus_meta["title_scopus"].apply(clean_text)

    out = step_df.merge(scopus_meta, how="left", on="match_title", suffixes=("", "_scopus_merge"))
    return out, scopus_titles, scopus_dois


def _compute_in_scopus(row: pd.Series, scopus_titles: set, scopus_dois: set) -> bool:
    md = _safe_str(row.get("match_doi"))
    mt = _safe_str(row.get("match_title"))
    return (bool(md) and md in scopus_dois) or (bool(mt) and mt in scopus_titles)


def _compute_coverage_stats(df: pd.DataFrame) -> dict:
    def _pct(n_in: int, n_total: int) -> float:
        return (100.0 * n_in / n_total) if n_total > 0 else 0.0

    out = {}
    out["n_total_overall"] = int(len(df))
    out["n_in_overall"] = int(df["in_scopus_retrieval"].sum())
    out["pct_overall"] = _pct(out["n_in_overall"], out["n_total_overall"])

    peer = df[df["source_type"] == "Peer Reviewed"]
    grey = df[df["source_type"] == "Grey Literature"]

    out["n_total_peer"] = int(len(peer))
    out["n_in_peer"] = int(peer["in_scopus_retrieval"].sum())
    out["pct_peer"] = _pct(out["n_in_peer"], out["n_total_peer"])

    out["n_total_grey"] = int(len(grey))
    out["n_in_grey"] = int(grey["in_scopus_retrieval"].sum())
    out["pct_grey"] = _pct(out["n_in_grey"], out["n_total_grey"])
    return out


def run(config: dict) -> None:
    print("--- Running Step 7: Eligible Benchmark vs. Scopus Retrieval ---")

    out_root = config.get("out_dir") or "scripts/outputs"
    step7_dir = os.path.join(out_root, "step7")
    _ensure_dir(step7_dir)
    step6_dir = os.path.join(out_root, "step6")

    history_dir = os.path.join(step7_dir, "history")
    _ensure_dir(history_dir)

    scopus_file = os.path.join(out_root, "step2", "step2_total_records.csv")
    step5_file = os.path.join(out_root, "step5", "step5_eligibility_wide.csv")
    bm_file = _resolve_benchmark_csv(out_root, config.get("benchmark_csv"))

    missing = []
    for p in [scopus_file, step5_file, bm_file]:
        if not p or not os.path.exists(p):
            missing.append(p)
    if missing:
        raise RuntimeError("Missing one or more input files:\n" + "\n".join([f"- {m}" for m in missing]))

    timestamp = _now_local_str()

    # Search string
    search_string, search_source = _get_search_string(config, out_root)
    search_hash = _sha1(search_string or "")[:12]

    # Abstract lookup
    doi_to_abs, title_to_abs, abs_source = _load_abstract_lookup(out_root)

    # Step1 blocks for routing
    step1_blocks, step1_blocks_source = _load_step1_blocks(out_root)

    # Iteration tracking files
    evolution_jsonl = os.path.join(step7_dir, "step7_searchstring_evolution.jsonl")
    evolution_csv = os.path.join(step7_dir, "step7_searchstring_evolution.csv")
    appendix_docx = os.path.join(step7_dir, "step7_appendix_searchstring_evolution.docx")
    suggestions_txt = os.path.join(step7_dir, "step7_searchstring_suggestions.txt")
    progress_png = os.path.join(step7_dir, "step7_progress_over_iterations.png")

    previous = _load_jsonl(evolution_jsonl)

    # run fingerprint (for de-dup reruns)
    baseline_xlsx = os.path.join(step6_dir, BASELINE_ELIGIBILITY_XLSX_BASENAME)
    secondary_csv = os.path.join(step6_dir, SECONDARY_RESULTS_CSV_BASENAME)
    step1_queries_json = os.path.join(out_root, "step1", STEP1_QUERIES_JSON_BASENAME)

    run_fingerprint_src = "|".join([
        f"search_hash:{search_hash}",
        _file_fingerprint(scopus_file),
        _file_fingerprint(step5_file),
        _file_fingerprint(bm_file),
        _file_fingerprint(baseline_xlsx),
        _file_fingerprint(secondary_csv),
        _file_fingerprint(step1_queries_json),
    ])
    run_fingerprint = _sha1(run_fingerprint_src)[:12]

    deduped = False
    if previous and previous[-1].get("run_fingerprint") == run_fingerprint:
        deduped = True
        iteration = int(previous[-1].get("iteration", len(previous)))
        print(f"ℹ️ Detected identical rerun (run_fingerprint={run_fingerprint}). Not creating a new iteration.")
    else:
        iteration = len(previous) + 1

    iter_tag = f"iter{iteration:03d}"

    # Load data
    scopus_df = pd.read_csv(scopus_file)
    step5_df = pd.read_csv(step5_file)
    bm_df = pd.read_csv(bm_file)

    # Prepare Step5
    s5_title_col = find_col(step5_df, ["title", "study"])
    s5_doi_col = find_col(step5_df, ["doi", "DOI"])
    s5_status_col = find_col(step5_df, ["final_decision", "decision"])
    if not s5_title_col or not s5_status_col:
        raise RuntimeError("Step 5 missing required columns (need title and final_decision).")

    step5_df = step5_df.copy()
    step5_df["title"] = step5_df[s5_title_col].apply(_safe_str)
    step5_df["doi"] = step5_df[s5_doi_col].apply(_clean_doi) if s5_doi_col else ""
    step5_df["match_title"] = step5_df["title"].apply(clean_text)
    step5_df["match_doi"] = step5_df["doi"].apply(clean_text) if s5_doi_col else ""
    step5_df["simple_status"] = step5_df[s5_status_col].apply(_simple_status)

    # Restore source types from benchmark CSV (Step3 output)
    bm_title_col = find_col(bm_df, ["Study", "Title", "title"])
    bm_type_col = find_col(bm_df, ["Type", "Source Type", "source_type"])
    if bm_title_col and bm_type_col:
        bm_df = bm_df.copy()
        bm_df["match_title_bm"] = bm_df[bm_title_col].apply(clean_text)
        type_lookup = bm_df.drop_duplicates("match_title_bm").set_index("match_title_bm")[bm_type_col].to_dict()
        step5_df["restored_type"] = step5_df["match_title"].map(type_lookup).fillna("Unknown")
    else:
        step5_df["restored_type"] = "Unknown"
    step5_df["source_type"] = step5_df["restored_type"].apply(_map_source_type)

    # Merge Scopus metadata + compute in_scopus flags
    step5_df, scopus_titles, scopus_dois = _merge_scopus_metadata(step5_df, scopus_df)
    if "author_names" not in step5_df.columns:
        step5_df["author_names"] = ""
    step5_df["in_scopus_retrieval"] = step5_df.apply(lambda r: _compute_in_scopus(r, scopus_titles, scopus_dois), axis=1)

    # Eligible keys from baseline + secondary decisions
    baseline_keys, baseline_source = _load_baseline_eligible_keys(step6_dir, step5_df)
    secondary_keys, secondary_source, _secondary_path = _load_secondary_eligible_keys(step6_dir)

    # FULL SECONDARY LIST decisions (ALL decisions) for secondary "benchmark-style" 2x2
    secondary_decisions_map, _secondary_list_source, _ = _load_secondary_list_decisions(step6_dir)
    secondary_all_keys = set(secondary_decisions_map.keys())

    def _row_key(r: pd.Series) -> str:
        d = _clean_doi(r.get("doi"))
        if d:
            return f"doi:{d}"
        return f"title:{clean_text(_safe_str(r.get('title')))}"

    step5_df["eligible_baseline"] = step5_df.apply(lambda r: _row_key(r) in baseline_keys, axis=1)
    step5_df["eligible_secondary"] = step5_df.apply(lambda r: _row_key(r) in secondary_keys, axis=1) if secondary_keys else False

    # Repair citations for UNION of eligible sets
    print("\n--- repairing missing metadata for Eligible benchmark records (Crossref) ---")
    has_scopus_author = step5_df["author_names"].notna() & (step5_df["author_names"] != "")
    has_doi = step5_df["doi"].astype(str).str.strip() != ""
    mask_needs_repair = (
        (step5_df["eligible_baseline"] | step5_df["eligible_secondary"])
        & has_doi
        & ((step5_df["in_scopus_retrieval"] == False) | (~has_scopus_author))
    )

    to_repair = step5_df[mask_needs_repair].index.tolist()
    print(f"Fetching metadata for {len(to_repair)} records...")
    for idx in tqdm(to_repair, total=len(to_repair), unit="rec"):
        doi = step5_df.at[idx, "doi"]
        meta = fetch_crossref_metadata(doi)
        if meta:
            step5_df.at[idx, "fetched_authors"] = meta.get("fetched_authors", "")
            step5_df.at[idx, "fetched_journal"] = meta.get("fetched_journal", "")
            step5_df.at[idx, "fetched_year"] = meta.get("fetched_year", "")
        time.sleep(0.1)

    step5_df["citation"] = step5_df.apply(lambda r: build_citation(r.to_dict()), axis=1)

    # Full status
    full_csv = os.path.join(step7_dir, "step7_full_benchmark_status.csv")
    step5_df.to_csv(full_csv, index=False, encoding="utf-8")
    print(f"✅ Saved: {full_csv}")

    # FULL BENCHMARK LIST plot (2x2) — this is your "initial" reference
    benchmark_plot = os.path.join(step7_dir, "step7_retrieval_analysis_all_2x2.png")
    _plot_retrieval_2x2(step5_df, benchmark_plot, "Scopus Retrieval Analysis — Full Benchmark List (All Decisions)", timestamp)

    # ✅ FULL SECONDARY LIST plot (2x2), using SECONDARY decisions for status (All/Passed/Unclear/Failed)
    secondary_full_plot = ""
    secondary_full_df = pd.DataFrame()
    if secondary_all_keys:
        step5_df["secondary_decision_raw"] = step5_df.apply(lambda r: secondary_decisions_map.get(_row_key(r), ""), axis=1)
        secondary_full_df = step5_df[step5_df["secondary_decision_raw"].astype(str) != ""].copy()
        # IMPORTANT: override status using secondary decisions (so quadrants match the secondary check list)
        secondary_full_df["simple_status"] = secondary_full_df["secondary_decision_raw"].apply(_simple_status)

        secondary_full_plot = os.path.join(step7_dir, "step7a_retrieval_analysis_all_2x2_secondary.png")
        _plot_retrieval_2x2(
            secondary_full_df,
            secondary_full_plot,
            "Scopus Retrieval Analysis — Full Secondary Check List (All Decisions)",
            timestamp,
        )
    else:
        print("ℹ️ No secondary full list found; skipping full-secondary 2x2 plot.")

    # Baseline eligible outputs
    base_elig_df = step5_df[step5_df["eligible_baseline"]].copy()
    base_missed_df = base_elig_df[base_elig_df["in_scopus_retrieval"] == False].copy()

    out_base_all_csv = os.path.join(step7_dir, "step7_all_eligible_studies_baseline.csv")
    out_base_missed_csv = os.path.join(step7_dir, "step7_missed_eligible_studies_baseline.csv")
    base_elig_df.to_csv(out_base_all_csv, index=False, encoding="utf-8")
    base_missed_df.to_csv(out_base_missed_csv, index=False, encoding="utf-8")
    print(f"✅ Saved: {out_base_all_csv}")
    print(f"✅ Saved: {out_base_missed_csv}")

    if HAS_DOCX:
        export_to_word_citations(base_missed_df, "Eligible Benchmark (Baseline): Missed in Scopus", os.path.join(step7_dir, "step7_missed_eligible_studies_baseline.docx"))
        export_to_word_citations(base_elig_df, "Eligible Benchmark (Baseline): Full List", os.path.join(step7_dir, "step7_all_eligible_studies_baseline.docx"))

    base_overall_plot = ""
    if not base_elig_df.empty:
        base_overall_plot = os.path.join(step7_dir, "step7_retrieval_analysis_eligible_2x2.png")
        _plot_eligible_overall(base_elig_df, base_overall_plot, "Eligible Benchmark (Baseline): In Scopus vs Missed (Overall)", timestamp)

    # Secondary eligible outputs (optional)
    sec_elig_df = step5_df[step5_df["eligible_secondary"]].copy() if secondary_keys else pd.DataFrame()
    sec_missed_df = pd.DataFrame()
    sec_overall_plot = ""

    if secondary_keys and not sec_elig_df.empty:
        sec_missed_df = sec_elig_df[sec_elig_df["in_scopus_retrieval"] == False].copy()

        out_sec_all_csv = os.path.join(step7_dir, "step7a_all_eligible_studies_secondary.csv")
        out_sec_missed_csv = os.path.join(step7_dir, "step7a_missed_eligible_studies_secondary.csv")
        sec_elig_df.to_csv(out_sec_all_csv, index=False, encoding="utf-8")
        sec_missed_df.to_csv(out_sec_missed_csv, index=False, encoding="utf-8")
        print(f"✅ Saved: {out_sec_all_csv}")
        print(f"✅ Saved: {out_sec_missed_csv}")

        if HAS_DOCX:
            export_to_word_citations(sec_missed_df, "Eligible Benchmark (Secondary): Missed in Scopus", os.path.join(step7_dir, "step7a_missed_eligible_studies_secondary.docx"))
            export_to_word_citations(sec_elig_df, "Eligible Benchmark (Secondary): Full List", os.path.join(step7_dir, "step7a_all_eligible_studies_secondary.docx"))

        sec_overall_plot = os.path.join(step7_dir, "step7a_retrieval_analysis_secondary_2x2.png")
        _plot_eligible_overall(sec_elig_df, sec_overall_plot, "Eligible Benchmark (Secondary): In Scopus vs Missed (Overall)", timestamp)
    else:
        print("ℹ️ No usable secondary eligible set found; skipping secondary-eligible outputs.")

    # Coverage comparison figure (OVERALL ONLY)
    base_stats = _compute_coverage_stats(base_elig_df) if not base_elig_df.empty else {
        "n_total_overall": 0, "n_in_overall": 0, "pct_overall": 0.0,
        "n_total_peer": 0, "n_in_peer": 0, "pct_peer": 0.0,
        "n_total_grey": 0, "n_in_grey": 0, "pct_grey": 0.0,
    }
    sec_stats = _compute_coverage_stats(sec_elig_df) if (secondary_keys and not sec_elig_df.empty) else None

    comparison_plot = os.path.join(step7_dir, "step7a_retrieval_analysis_comparison.png")
    _plot_coverage_comparison_overall_only(base_stats, sec_stats, comparison_plot, timestamp)

    # Final target = secondary eligible if present else baseline eligible (used for missed list + suggestions)
    final_target_df = sec_elig_df if (secondary_keys and not sec_elig_df.empty) else base_elig_df
    final_target_label = "secondary" if (secondary_keys and not sec_elig_df.empty) else "baseline"

    final_missed_df = final_target_df[final_target_df["in_scopus_retrieval"] == False].copy()
    final_missed_csv = os.path.join(step7_dir, "step7_benchmark_scopus_missed.csv")
    final_missed_df.to_csv(final_missed_csv, index=False, encoding="utf-8")
    print(f"✅ Saved: {final_missed_csv}")

    if HAS_DOCX:
        export_to_word_citations(final_missed_df, f"Final Eligible Benchmark ({final_target_label}): Missed in Scopus", os.path.join(step7_dir, "step7_benchmark_scopus_missed.docx"))

    # Suggestions from MISSED eligible titles + abstracts
    missed_records: List[Dict[str, str]] = []
    for _, r in final_missed_df.iterrows():
        title = _safe_str(r.get("title"))
        doi = _clean_doi(r.get("doi"))
        abs_txt = ""
        if doi and doi in doi_to_abs:
            abs_txt = doi_to_abs.get(doi, "")
        if not abs_txt and title:
            abs_txt = title_to_abs.get(clean_text(title), "")
        missed_records.append({"title": title, "abstract": abs_txt})

    author_tokens = _extract_author_tokens(final_missed_df)
    suggestion_text, deduped_terms = _suggest_improvements_by_block(
        search_string,
        missed_records,
        author_tokens=author_tokens,
        step1_blocks=step1_blocks,
        top_k=18,
    )

    with open(suggestions_txt, "w", encoding="utf-8") as f:
        f.write(f"(Abstract source: {abs_source})\n")
        f.write(f"(Step1 blocks source: {step1_blocks_source})\n")
        f.write(suggestion_text + "\n")
    print(f"✅ Saved: {suggestions_txt}")

    # Coverage CSV (overwrite)
    coverage_csv = os.path.join(step7_dir, "step7_benchmark_scopus_coverage.csv")
    final_stats = _compute_coverage_stats(final_target_df) if not final_target_df.empty else {
        "n_total_overall": 0, "n_in_overall": 0, "pct_overall": 0.0,
        "n_total_peer": 0, "n_in_peer": 0, "pct_peer": 0.0,
        "n_total_grey": 0, "n_in_grey": 0, "pct_grey": 0.0,
    }
    coverage_row = {
        "timestamp_local": timestamp,
        "iteration": iteration,
        "run_fingerprint": run_fingerprint,
        "search_source": search_source,
        "search_hash": search_hash,
        "baseline_eligibility_source": baseline_source,
        "secondary_eligibility_source": secondary_source,
        "final_target": final_target_label,
        "baseline_n_total": base_stats["n_total_overall"],
        "baseline_n_in_scopus": base_stats["n_in_overall"],
        "baseline_pct": round(base_stats["pct_overall"], 3),
        "secondary_n_total": (sec_stats["n_total_overall"] if sec_stats else 0),
        "secondary_n_in_scopus": (sec_stats["n_in_overall"] if sec_stats else 0),
        "secondary_pct": (round(sec_stats["pct_overall"], 3) if sec_stats else 0.0),
        "final_n_total": final_stats["n_total_overall"],
        "final_n_in_scopus": final_stats["n_in_overall"],
        "final_pct": round(final_stats["pct_overall"], 3),
        "final_missed_n": int(len(final_missed_df)),
        "abstract_source": abs_source,
        "step1_blocks_source": step1_blocks_source,
        "suggested_terms_deduped": ", ".join(deduped_terms),
    }
    pd.DataFrame([coverage_row]).to_csv(coverage_csv, index=False, encoding="utf-8")
    print(f"✅ Saved: {coverage_csv}")

    # Archive figures per iteration (only if NEW iteration)
    fig_benchmark_arch = os.path.join(history_dir, f"{iter_tag}_benchmark_all_2x2.png")
    fig_secondary_full_arch = os.path.join(history_dir, f"{iter_tag}_secondary_benchmark_all_2x2.png")
    fig_comp_arch = os.path.join(history_dir, f"{iter_tag}_coverage_comparison_overall.png")
    fig_base_arch = os.path.join(history_dir, f"{iter_tag}_baseline_eligible_overall.png")
    fig_sec_arch = os.path.join(history_dir, f"{iter_tag}_secondary_eligible_overall.png")

    if not deduped:
        _archive_copy(benchmark_plot, fig_benchmark_arch)
        if secondary_full_plot:
            _archive_copy(secondary_full_plot, fig_secondary_full_arch)

        _archive_copy(comparison_plot, fig_comp_arch)
        if base_overall_plot:
            _archive_copy(base_overall_plot, fig_base_arch)
        if sec_overall_plot:
            _archive_copy(sec_overall_plot, fig_sec_arch)

    # Evolution log (append-only; de-dup reruns)
    if not deduped:
        evolution_rec = {
            "timestamp_local": timestamp,
            "iteration": iteration,
            "run_fingerprint": run_fingerprint,
            "search_source": search_source,
            "search_string": search_string,
            "search_hash": search_hash,
            "inputs": {
                "benchmark_csv": bm_file,
                "scopus_csv": scopus_file,
                "step5_csv": step5_file,
                "abstract_source": abs_source,
                "step1_blocks_source": step1_blocks_source,
            },
            "eligibility_sources": {"baseline": baseline_source, "secondary": secondary_source, "final_target": final_target_label},
            "coverage": {"baseline": base_stats, "secondary": (sec_stats or {}), "final_target": final_stats, "final_missed_n": int(len(final_missed_df))},
            "missed_titles_sample": [m.get("title", "") for m in missed_records[:10]],
            "suggestion_text": suggestion_text,
            "figures": {
                "benchmark_all_plot": fig_benchmark_arch if os.path.exists(fig_benchmark_arch) else "",
                "secondary_benchmark_all_plot": fig_secondary_full_arch if os.path.exists(fig_secondary_full_arch) else "",
                "coverage_comparison": fig_comp_arch if os.path.exists(fig_comp_arch) else "",
                "baseline_eligible_plot": fig_base_arch if os.path.exists(fig_base_arch) else "",
                "secondary_eligible_plot": fig_sec_arch if os.path.exists(fig_sec_arch) else "",
            },
        }
        _write_jsonl_append(evolution_jsonl, evolution_rec)
        print(f"✅ Appended evolution log: {evolution_jsonl}")

    # Derived evolution CSV + progress chart + appendix rebuilt each run
    evo_rows = _load_jsonl(evolution_jsonl)

    flat = []
    for r in evo_rows:
        cov_final = (r.get("coverage", {}) or {}).get("final_target", {}) or {}
        flat.append({
            "iteration": r.get("iteration"),
            "timestamp_local": r.get("timestamp_local"),
            "search_hash": r.get("search_hash"),
            "final_target": (r.get("eligibility_sources", {}) or {}).get("final_target"),
            "final_n_total": cov_final.get("n_total_overall", 0),
            "final_n_in_scopus": cov_final.get("n_in_overall", 0),
            "final_pct": cov_final.get("pct_overall", 0.0),
        })
    pd.DataFrame(flat).to_csv(evolution_csv, index=False, encoding="utf-8")
    print(f"✅ Updated evolution CSV: {evolution_csv}")

    _plot_progress_over_iterations(evo_rows, progress_png, timestamp)
    print(f"✅ Saved: {progress_png}")

    if HAS_DOCX:
        _build_appendix_from_evolution(appendix_docx, evo_rows, progress_png)

    print("\n--- SUMMARY ---")
    print(f"Benchmark CSV used:        {bm_file}")
    print(f"Baseline eligible source:  {baseline_source}")
    print(f"Secondary eligible source: {secondary_source}")
    print(f"Secondary full list size:  {len(secondary_full_df) if not secondary_full_df.empty else 0}")
    print(f"Final target:             {final_target_label}")
    print(f"Final coverage:           {final_stats['n_in_overall']}/{final_stats['n_total_overall']} ({final_stats['pct_overall']:.1f}%)")
    print(f"Final missed:             {len(final_missed_df)}")
    print(f"Abstract source:          {abs_source}")
    print(f"Step1 blocks source:      {step1_blocks_source}")
    print(f"Appendix:                 {appendix_docx if HAS_DOCX else '[docx not available]'}")
    if deduped:
        print("NOTE: This run was deduped (no new iteration appended).")


if __name__ == "__main__":
    run({"out_dir": "scripts/outputs"})
