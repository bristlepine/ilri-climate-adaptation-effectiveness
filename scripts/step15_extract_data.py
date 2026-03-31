#!/usr/bin/env python3
"""
step15_extract_data.py

Step 15: Structured data extraction (coding) for all records that passed
screening, using the coding schema defined in the systematic map protocol
(Table 3, Deliverable 3).

Coding source hierarchy (tracked per record):
  full_text        — coded from downloaded PDF/HTML (step 13 + step 14 Include)
  abstract_only    — coded from abstract only (no full text retrieved)
  missing_abstract — no abstract and no full text; titles only
  needs_manual     — full text extraction failed (scanned/image PDF etc.)

Cache key = stable_key + "::" + coding_source  so that when a paper is
upgraded (e.g. full text becomes available), it is automatically re-extracted
on the next run without re-processing papers that already have full text.

Inputs:
  - outputs/step14/step14_results.csv     (all step12 Include records + s14 decisions)
  - outputs/step13/fulltext/              (downloaded full text files)
  - outputs/step9a/step9a_scopus_enriched.csv  (abstracts)

Outputs (under outputs/step15/):
  - step15_coded.csv          one row per coded study, all 20 coding columns
  - step15_coded.meta.json    counts by coding_source, completion rates, timing
  - step15_needs_review.csv   records with low-confidence or missing fields
  - step15_summary.png        summary figure

Run:
  python step15_extract_data.py
  (or via run.py with run_step15 = 1)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import yaml

try:
    from json_repair import repair_json as _repair_json
    _HAS_JSON_REPAIR = True
except ImportError:
    _HAS_JSON_REPAIR = False


# =============================================================================
# SETTINGS
# =============================================================================

OLLAMA_URL       = "http://localhost:11434/api/generate"
DEFAULT_MODEL    = "qwen2.5:14b"
TEMPERATURE      = 0.0
OLLAMA_KEEP_ALIVE = "30m"
OLLAMA_THINK     = False

# Characters of full text to send to LLM (more than screening — we need detail)
FULLTEXT_MAX_CHARS = 12_000
ABSTRACT_MAX_CHARS = 4_000

PRINT_EVERY = 50


# =============================================================================
# CODING SCHEMA  (protocol Table 3, Deliverable 3)
# =============================================================================

# Each entry: field_name -> (description, valid_values_hint)
# valid_values_hint is included verbatim in the LLM prompt.
CODING_SCHEMA: Dict[str, Tuple[str, str]] = {
    "publication_year": (
        "Year of publication (use most recent if multiple versions exist).",
        "integer e.g. 2021",
    ),
    "publication_type": (
        "Type of publication.",
        "journal_article | report | working_paper | thesis | other",
    ),
    "country_region": (
        "Country or countries covered. Use region label only if countries are not specified. "
        "Separate multiple entries with semicolons.",
        "e.g. Kenya; Ethiopia  or  Sub-Saharan Africa",
    ),
    "geographic_scale": (
        "Spatial scale of analysis.",
        "local | sub-national | national | multi-country | regional",
    ),
    "producer_type": (
        "Type(s) of agricultural producers studied. List all that apply, separated by semicolons.",
        "crop | livestock | fisheries_aquaculture | agroforestry | mixed",
    ),
    "marginalized_subpopulations": (
        "Whether the study explicitly addresses any marginalized groups. "
        "List all that apply, or 'none_reported'.",
        "women | youth | landless | indigenous_peoples | ethnic_minorities | "
        "migrant_seasonal_workers | none_reported",
    ),
    "adaptation_focus": (
        "Describe the specific climate adaptation action, intervention, or practice assessed. "
        "This is emergent — use free text that best captures what the study is about.",
        "free text e.g. 'drought-tolerant maize varieties' or 'seasonal climate forecasts for farmers'",
    ),
    "domain_type": (
        "Whether the study assesses adaptation processes, outcomes, or both.",
        "adaptation_process | adaptation_outcome | both",
    ),
    "process_outcome_domains": (
        "Specific process and/or outcome domains assessed. List all that apply, separated by semicolons.",
        "Process: knowledge_awareness_learning | decision_making_planning | uptake_adoption | "
        "behavioral_change | participation_coproduction | institutional_governance | "
        "access_information_services. "
        "Outcome: yields_productivity | income_assets | livelihoods | wellbeing | "
        "risk_reduction | resilience_adaptive_capacity",
    ),
    "indicators_measured": (
        "Types of indicators or metrics used in the study. "
        "This is emergent — summarise the key indicators in free text.",
        "free text e.g. 'yield (kg/ha), household income, self-reported adaptive capacity index'",
    ),
    "methodological_approach": (
        "Primary methodological design of the study.",
        "qualitative | quantitative | mixed_methods | participatory | "
        "modeling_with_empirical_validation",
    ),
    "purpose_of_assessment": (
        "Primary purpose of the assessment as stated or implied in the study.",
        "project_learning | program_evaluation | donor_reporting | national_reporting | research",
    ),
    "data_sources": (
        "Data sources used. List all that apply, separated by semicolons.",
        "surveys | administrative_data | remote_sensing | participatory_methods | secondary_data",
    ),
    "temporal_coverage": (
        "Time horizon of data collection.",
        "cross_sectional | seasonal | longitudinal",
    ),
    "cost_data_reported": (
        "Whether cost or resource requirement data are reported.",
        "yes | no",
    ),
    "equity_inclusion": (
        "Whether equity dimensions are explicitly addressed. List all that apply, or 'none_reported'.",
        "gender | youth | land_tenure | disability | other | none_reported",
    ),
    "strengths": (
        "Author-reported strengths of the study. Extract verbatim where possible.",
        "free text",
    ),
    "limitations": (
        "Author-reported limitations of the study. Extract verbatim where possible.",
        "free text",
    ),
    "lessons_learned": (
        "Key lessons or recommendations reported by the authors.",
        "free text",
    ),
    "validity_notes": (
        "Any notes relevant to internal or external validity "
        "(e.g. small sample, single site, self-reported outcomes).",
        "free text",
    ),
}


# =============================================================================
# Coding source labels
# =============================================================================

SOURCE_FULL_TEXT        = "full_text"
SOURCE_ABSTRACT_ONLY    = "abstract_only"
SOURCE_MISSING_ABSTRACT = "missing_abstract"
SOURCE_NEEDS_MANUAL     = "needs_manual"


# =============================================================================
# Helpers
# =============================================================================

def safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    s = str(x).replace("\r", " ").replace("\n", " ")
    return " ".join(s.split()).strip()


def normalize_doi(doi: Any) -> str:
    s = safe_str(doi)
    if not s:
        return ""
    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s, flags=re.IGNORECASE).strip()
    return s.rstrip(" .;,)\t").lower()


def year_from_any(x: Any) -> str:
    s = safe_str(x)
    m = re.search(r"\b(19|20)\d{2}\b", s)
    return m.group(0) if m else ""


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stable_key(row: pd.Series, i: int) -> str:
    dk = safe_str(row.get("dedupe_key", ""))
    if dk:
        return dk
    doi = normalize_doi(row.get("doi", ""))
    if doi:
        return f"doi:{doi}"
    title = safe_str(row.get("title", ""))
    year = year_from_any(row.get("year", "") or row.get("coverDate", ""))
    if title and year:
        clean = re.sub(r"[^a-z0-9]+", "", title.lower())
        return f"ty:{clean}:{year}"
    if title:
        clean = re.sub(r"[^a-z0-9]+", "", title.lower())
        return f"t:{clean}"
    blob = f"{i}|{title}|{doi}".encode("utf-8", errors="ignore")
    return "row:" + hashlib.sha1(blob).hexdigest()[:16]


def _cache_key(base_key: str, coding_source: str) -> str:
    """Include source in cache key so upgrading source triggers re-extraction."""
    return f"{base_key}::{coding_source}"


def _run_signature(*, model: str) -> str:
    blob = f"model={model}\nschema_version=1\n"
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


# =============================================================================
# Text extraction from files
# =============================================================================

def _extract_pdf_text(path: Path, max_chars: int) -> Tuple[str, str]:
    """Returns (text, note). note is empty on success."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        parts: List[str] = []
        for page in reader.pages:
            t = page.extract_text() or ""
            parts.append(t)
            if sum(len(p) for p in parts) >= max_chars * 2:
                break
        text = " ".join(parts)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars], ""
    except Exception as e:
        return "", f"pdf_error:{type(e).__name__}"


def _extract_html_text(path: Path, max_chars: int) -> Tuple[str, str]:
    try:
        import trafilatura
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        text = trafilatura.extract(raw) or ""
        if not text.strip():
            raise ValueError("trafilatura returned empty")
        return text[:max_chars], ""
    except Exception:
        pass
    try:
        from bs4 import BeautifulSoup
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        soup = BeautifulSoup(raw, "html.parser")
        text = soup.get_text(separator=" ")
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars], ""
    except Exception as e:
        return "", f"html_error:{type(e).__name__}"


def extract_text(file_path: str, max_chars: int = FULLTEXT_MAX_CHARS) -> Tuple[str, str]:
    """Extract text from a file. Returns (text, note)."""
    if not file_path:
        return "", "no_file"
    p = Path(file_path)
    if not p.exists():
        return "", "file_not_found"
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf_text(p, max_chars)
    if suffix in (".html", ".htm"):
        return _extract_html_text(p, max_chars)
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
        return text[:max_chars], ""
    except Exception as e:
        return "", f"read_error:{type(e).__name__}"


# =============================================================================
# Paths
# =============================================================================

def _out_dir(out_root: Path) -> Path:
    d = out_root / "step15"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _step14_csv(out_root: Path) -> Path:
    return out_root / "step14" / "step14_results.csv"

def _step9a_csv(out_root: Path) -> Path:
    p = out_root / "step9a" / "step9a_scopus_enriched.csv"
    if p.exists():
        return p
    return out_root / "step9" / "step9_scopus_enriched.csv"


# =============================================================================
# Load and merge inputs
# =============================================================================

def load_inputs(out_root: Path) -> pd.DataFrame:
    """
    Load step14 results (Include + Needs_Manual) merged with step9a abstracts.
    Assigns coding_source per record based on available text.
    """
    s14 = pd.read_csv(_step14_csv(out_root), engine="python", on_bad_lines="skip")
    print(f"[step15] Step14 records loaded: {len(s14):,}")

    # Only code records that were not excluded
    codeable = s14[s14["s14_decision"].str.strip().isin(["Include", "Needs_Manual"])].copy()
    print(f"[step15] Codeable records (Include + Needs_Manual): {len(codeable):,}")

    # Load abstracts from step9a to fill abstract_only records
    s9a_path = _step9a_csv(out_root)
    if s9a_path.exists():
        s9a = pd.read_csv(s9a_path, engine="python", on_bad_lines="skip")
        for col in ["dedupe_key", "doi", "abstract", "xref_abstract"]:
            if col not in s9a.columns:
                s9a[col] = ""
        s9a["dedupe_key"] = s9a["dedupe_key"].apply(safe_str)
        s9a["doi"] = s9a["doi"].apply(normalize_doi)
        # Best abstract from step9a
        def _best_abs(row: pd.Series) -> str:
            for col in ["abstract", "xref_abstract"]:
                v = safe_str(row.get(col, ""))
                if v:
                    return v
            return ""
        s9a["_s9a_abstract"] = s9a.apply(_best_abs, axis=1)
        s9a_abs = s9a[["dedupe_key", "_s9a_abstract"]].copy()
        codeable["dedupe_key"] = codeable["dedupe_key"].apply(safe_str)
        codeable = codeable.merge(s9a_abs, on="dedupe_key", how="left")
    else:
        codeable["_s9a_abstract"] = ""

    # Determine coding_source per record
    def _coding_source(row: pd.Series) -> str:
        file_path = safe_str(row.get("ft_file_path", ""))
        s14_note  = safe_str(row.get("s14_fulltext_note", ""))
        s12_reason = safe_str(row.get("screen_reasons", ""))
        abstract  = safe_str(row.get("_s9a_abstract", ""))

        # Has a full text file and step14 extracted it (Include decision)
        if file_path and row.get("s14_decision") == "Include":
            return SOURCE_FULL_TEXT

        # Had a file but extraction failed
        if file_path and s14_note in ("pdf_error", "html_error", "empty"):
            return SOURCE_NEEDS_MANUAL

        # No full text — check for abstract
        is_missing_abstract = "missing abstract" in s12_reason.lower()
        if is_missing_abstract and not abstract:
            return SOURCE_MISSING_ABSTRACT

        if abstract:
            return SOURCE_ABSTRACT_ONLY

        return SOURCE_NEEDS_MANUAL

    codeable["coding_source"] = codeable.apply(_coding_source, axis=1)

    counts = codeable["coding_source"].value_counts().to_dict()
    print(f"[step15] Coding source breakdown: {counts}")

    return codeable.reset_index(drop=True)


# =============================================================================
# LLM prompt
# =============================================================================

def _build_extraction_prompt() -> str:
    lines: List[str] = []
    for field, (desc, valid) in CODING_SCHEMA.items():
        lines.append(f'  "{field}": {{')
        lines.append(f'    "value": <{valid}>,')
        lines.append(f'    "confidence": "high | low | not_found",')
        lines.append(f'    "note": "<brief justification or direct quote>"')
        lines.append("  },")
    return "\n".join(lines)


SYSTEM_PROMPT_TEMPLATE = """You are a systematic review research assistant. Your task is to extract \
structured data from a research paper for a systematic map of climate adaptation evidence for \
smallholder producers in low- and middle-income countries.

Extract ALL of the following fields. For each field:
- Set "value" to the best answer based on the text provided.
- Set "confidence" to "high" if clearly stated, "low" if inferred, or "not_found" if the \
information is absent.
- Set "note" to a brief justification or a direct quote from the text.

If you are extracting from an ABSTRACT ONLY, many fields will be "not_found" — that is \
expected. Extract what you can.

CODING FIELDS:
{schema_block}

OUTPUT FORMAT:
Return ONLY a valid JSON object with exactly the fields above. No markdown, no commentary."""


def build_system_prompt() -> str:
    schema_block = _build_extraction_prompt()
    return SYSTEM_PROMPT_TEMPLATE.format(schema_block=schema_block)


def _user_prompt(title: str, text: str, coding_source: str) -> str:
    source_label = {
        SOURCE_FULL_TEXT:        "FULL TEXT (extract)",
        SOURCE_ABSTRACT_ONLY:    "ABSTRACT ONLY",
        SOURCE_MISSING_ABSTRACT: "TITLE ONLY (no abstract available)",
        SOURCE_NEEDS_MANUAL:     "TITLE + PARTIAL TEXT",
    }.get(coding_source, "TEXT")
    return f"TITLE: {title}\n\n{source_label}:\n{text}"


# =============================================================================
# Ollama
# =============================================================================

def _ollama_fail_fast(model: str) -> None:
    base = OLLAMA_URL.split("/api/")[0]
    try:
        r = requests.get(f"{base}/api/version", timeout=5)
        r.raise_for_status()
    except Exception as e:
        raise SystemExit(f"Ollama not reachable at {base} ({type(e).__name__}: {e})")
    try:
        r = requests.get(f"{base}/api/tags", timeout=8)
        r.raise_for_status()
        names = {safe_str(m.get("name", "")) for m in (r.json().get("models") or [])}
        if model not in names:
            raise SystemExit(
                f"Ollama: model '{model}' not installed. Fix: ollama pull {model}"
            )
    except SystemExit:
        raise
    except Exception:
        pass


def call_ollama(
    session: requests.Session,
    *,
    title: str,
    text: str,
    coding_source: str,
    system_prompt: str,
    model: str,
) -> str:
    payload = {
        "model":      model,
        "system":     system_prompt,
        "prompt":     _user_prompt(title, text, coding_source),
        "stream":     False,
        "format":     "json",
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options":    {"temperature": TEMPERATURE},
        "think":      OLLAMA_THINK,
    }
    try:
        r = session.post(OLLAMA_URL, json=payload, timeout=300)
        if r.status_code == 200:
            return r.json().get("response", "")
        return json.dumps({"error": f"HTTP {r.status_code}", "body": (r.text or "")[:500]})
    except Exception as e:
        return json.dumps({"error": str(e)})


# =============================================================================
# Parse LLM response
# =============================================================================

def _parse_response(raw: str) -> Tuple[Dict[str, Any], bool]:
    """
    Returns (coded_fields, parse_ok).
    coded_fields: {field: {"value": ..., "confidence": ..., "note": ...}}
    """
    text = raw.strip()
    try:
        data = json.loads(text)
        return data, True
    except json.JSONDecodeError:
        pass
    if _HAS_JSON_REPAIR:
        try:
            data = json.loads(_repair_json(text))
            return data, True
        except Exception:
            pass
    # Try to extract JSON object from text
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            return data, True
        except Exception:
            pass
    return {}, False


def flatten_coded(
    coded: Dict[str, Any],
    parse_ok: bool,
    coding_source: str,
) -> Dict[str, Any]:
    """
    Flatten structured LLM output into one row dict.
    Columns: <field>_value, <field>_confidence, <field>_note for each coding field.
    Plus: coding_source, needs_review, parse_ok.
    """
    row: Dict[str, Any] = {
        "coding_source": coding_source,
        "parse_ok":      parse_ok,
    }
    low_confidence_fields: List[str] = []

    for field in CODING_SCHEMA:
        entry = coded.get(field, {})
        if isinstance(entry, dict):
            value      = safe_str(entry.get("value", ""))
            confidence = safe_str(entry.get("confidence", "not_found"))
            note       = safe_str(entry.get("note", ""))
        else:
            # LLM returned a scalar instead of dict
            value      = safe_str(entry)
            confidence = "low"
            note       = ""

        row[f"{field}_value"]      = value
        row[f"{field}_confidence"] = confidence
        row[f"{field}_note"]       = note

        if confidence in ("low", "not_found") and field not in (
            "strengths", "limitations", "lessons_learned", "validity_notes"
        ):
            low_confidence_fields.append(field)

    # needs_review: True if >3 structured fields are low/not_found, or parse failed
    needs_review = not parse_ok or len(low_confidence_fields) > 3
    # Abstract/title-only sourced records always need review
    if coding_source in (SOURCE_ABSTRACT_ONLY, SOURCE_MISSING_ABSTRACT, SOURCE_NEEDS_MANUAL):
        needs_review = True
    row["needs_review"]          = "true" if needs_review else "false"
    row["low_confidence_fields"] = "; ".join(low_confidence_fields)

    return row


# =============================================================================
# JSONL cache
# =============================================================================

def _load_jsonl_cache(path: Path) -> Dict[str, dict]:
    last: Dict[str, dict] = {}
    if not path.exists():
        return last
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
                k = safe_str(j.get("cache_key"))
                if k:
                    last[k] = j
            except Exception:
                continue
    return last


def _rewrite_jsonl(path: Path, cache: Dict[str, dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for j in cache.values():
            f.write(json.dumps(j, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


# =============================================================================
# Main extraction loop
# =============================================================================

def extract_all(
    df: pd.DataFrame,
    *,
    out_dir: Path,
    model: str,
    run_limit: Optional[int] = None,
) -> pd.DataFrame:

    _ollama_fail_fast(model)
    system_prompt = build_system_prompt()
    sig = _run_signature(model=model)

    jsonl_path = out_dir / "step15_results_details.jsonl"
    cache = _load_jsonl_cache(jsonl_path)
    if cache:
        _rewrite_jsonl(jsonl_path, cache)
    print(f"[step15] Cache warm: {len(cache):,} records")

    df = df.copy()
    n_total = len(df)
    n_run   = min(int(run_limit), n_total) if run_limit else n_total

    # Pre-build column list
    # coding_source is already set correctly by load_inputs — do NOT overwrite it
    result_cols = (
        ["coding_source", "parse_ok", "needs_review", "low_confidence_fields"]
        + [f"{f}_{s}" for f in CODING_SCHEMA for s in ("value", "confidence", "note")]
    )
    for col in result_cols:
        if col != "coding_source":
            df[col] = ""

    t0 = time.time()
    n_full = n_abstract = n_missing = n_manual = n_cached = n_error = 0

    print(f"[step15] Start extraction | total={n_total:,} | running={n_run:,} | model={model}")

    def _to_str(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, bool):
            return "true" if v else "false"
        return str(v)

    with requests.Session() as session:
        with open(jsonl_path, "a", encoding="utf-8", buffering=1) as jf:
            for i in range(n_run):
                row    = df.iloc[i]
                base_k = stable_key(row, i)
                src    = safe_str(row.get("coding_source", SOURCE_NEEDS_MANUAL))
                ck     = _cache_key(base_k, src)
                title  = safe_str(row.get("title", ""))

                if i == 0 or (i + 1) % PRINT_EVERY == 0 or (i + 1) == n_run:
                    elapsed = time.time() - t0
                    rate    = (i + 1) / elapsed if elapsed > 0 else 0
                    eta     = (n_run - i - 1) / rate if rate > 0 else 0
                    print(
                        f"[step15] {i+1:,}/{n_run:,} | "
                        f"full={n_full} abstract={n_abstract} missing={n_missing} "
                        f"manual={n_manual} cached={n_cached} err={n_error} | "
                        f"elapsed={elapsed:,.0f}s ETA={eta:,.0f}s"
                    )

                rec: Dict[str, Any] = {
                    "run_signature": sig,
                    "timestamp_utc": _now_utc(),
                    "cache_key":     ck,
                    "base_key":      base_k,
                    "coding_source": src,
                    "title":         title,
                }

                # --- Cache hit (same source) ---
                if ck in cache:
                    cached = cache[ck]
                    if safe_str(cached.get("run_signature")) == sig:
                        n_cached += 1
                        for col in result_cols:
                            df.at[i, col] = _to_str(cached.get(col, ""))
                        continue

                # --- Determine text to send ---
                if src == SOURCE_FULL_TEXT:
                    file_path = safe_str(row.get("ft_file_path", ""))
                    text, note = extract_text(file_path, FULLTEXT_MAX_CHARS)
                    if not text.strip():
                        # File present but extraction failed — downgrade
                        src = SOURCE_NEEDS_MANUAL
                        ck  = _cache_key(base_k, src)
                        text = safe_str(row.get("_s9a_abstract", ""))[:ABSTRACT_MAX_CHARS]
                    n_full += 1

                elif src == SOURCE_ABSTRACT_ONLY:
                    # Abstract-only records are not coded — pending full text retrieval
                    n_abstract += 1
                    flat = flatten_coded({}, True, src)
                    rec.update(flat)
                    jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    cache[_cache_key(base_k, src)] = rec
                    for col in result_cols:
                        df.at[i, col] = _to_str(rec.get(col, ""))
                    continue

                elif src == SOURCE_MISSING_ABSTRACT:
                    # No text at all — skip LLM, mark all fields not_found
                    n_missing += 1
                    flat = flatten_coded({}, True, src)
                    rec.update(flat)
                    jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    cache[_cache_key(base_k, src)] = rec
                    for col in result_cols:
                        df.at[i, col] = _to_str(rec.get(col, ""))
                    continue

                else:  # needs_manual — pending full text retrieval, skip LLM
                    n_manual += 1
                    flat = flatten_coded({}, True, src)
                    rec.update(flat)
                    jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    cache[_cache_key(base_k, src)] = rec
                    for col in result_cols:
                        df.at[i, col] = _to_str(rec.get(col, ""))
                    continue

                # --- LLM call ---
                llm_resp = call_ollama(
                    session,
                    title=title,
                    text=text,
                    coding_source=src,
                    system_prompt=system_prompt,
                    model=model,
                )

                coded, parse_ok = _parse_response(llm_resp)
                if not parse_ok:
                    n_error += 1

                flat = flatten_coded(coded, parse_ok, src)
                rec.update(flat)

                # Write to cache
                jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                cache[ck] = rec

                for col in result_cols:
                    df.at[i, col] = _to_str(rec.get(col, ""))

    return df


# =============================================================================
# Write outputs
# =============================================================================

def write_outputs(df: pd.DataFrame, out_dir: Path, elapsed: float) -> dict:
    # Value columns only (not _confidence/_note) for the main CSV
    value_cols = [f"{f}_value" for f in CODING_SCHEMA]
    id_cols    = [c for c in ["dedupe_key", "doi", "title", "year",
                               "publicationName", "author_names",
                               "screen_decision", "screen_reasons",
                               "s14_decision", "s14_reasons",
                               "coding_source", "needs_review",
                               "low_confidence_fields", "parse_ok"]
                  if c in df.columns]
    conf_cols  = [f"{f}_confidence" for f in CODING_SCHEMA]
    note_cols  = [f"{f}_note" for f in CODING_SCHEMA]

    all_cols = id_cols + value_cols + conf_cols + note_cols
    out_cols = [c for c in all_cols if c in df.columns]

    coded_csv = out_dir / "step15_coded.csv"
    df[out_cols].to_csv(coded_csv, index=False)
    print(f"[step15] Coded CSV -> {coded_csv}  ({len(df):,} rows)")

    # Needs review CSV
    review_df = df[df["needs_review"].astype(str).str.lower() == "true"]
    review_csv = out_dir / "step15_needs_review.csv"
    review_df[out_cols].to_csv(review_csv, index=False)
    print(f"[step15] Needs-review CSV -> {review_csv}  ({len(review_df):,} rows)")

    # Meta
    src_counts = df["coding_source"].value_counts(dropna=False).to_dict()
    meta = {
        "rows_total":          len(df),
        "rows_needs_review":   int(len(review_df)),
        "coding_source_counts": {str(k): int(v) for k, v in src_counts.items()},
        "elapsed_seconds":     round(elapsed, 1),
        "elapsed_hms":         time.strftime("%H:%M:%S", time.gmtime(elapsed)),
        "timestamp_utc":       _now_utc(),
    }
    meta_path = out_dir / "step15_coded.meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[step15] Meta -> {meta_path}")

    _plot_summary(meta, df, out_dir)

    return meta


# =============================================================================
# Summary figure
# =============================================================================

def _plot_summary(meta: dict, df: pd.DataFrame, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[step15] matplotlib not installed — skipping summary figure")
        return

    BLUE   = "#2196F3"
    ORANGE = "#FF9800"
    RED    = "#F44336"
    GREY   = "#9E9E9E"

    src_counts = meta.get("coding_source_counts", {})
    labels = [SOURCE_FULL_TEXT, SOURCE_ABSTRACT_ONLY,
              SOURCE_MISSING_ABSTRACT, SOURCE_NEEDS_MANUAL]
    display = ["Full Text", "Abstract Only", "Missing Abstract", "Needs Manual"]
    colors  = [BLUE, ORANGE, RED, GREY]
    values  = [src_counts.get(s, 0) for s in labels]
    total   = sum(values)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Step 15 — Data Extraction Summary", fontsize=14, fontweight="bold", y=1.01)

    # Panel 1: coding source breakdown
    ax1 = axes[0]
    bars = ax1.bar(display, values, color=colors, edgecolor="white", linewidth=0.8, width=0.5)
    for bar, val in zip(bars, values):
        pct = val / total * 100 if total else 0
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + total * 0.005,
                 f"{val:,}\n({pct:.1f}%)",
                 ha="center", va="bottom", fontsize=9)
    ax1.set_title(f"Records by Coding Source  (n={total:,})", fontsize=11)
    ax1.set_ylabel("Records")
    ax1.set_ylim(0, max(values) * 1.2 if values else 1)
    ax1.tick_params(axis="x", labelsize=9)
    ax1.spines[["top", "right"]].set_visible(False)

    # Panel 2: needs_review breakdown
    ax2 = axes[1]
    needs_rev = int(meta.get("rows_needs_review", 0))
    ok        = total - needs_rev
    ax2.bar(["Extraction OK", "Needs Review"], [ok, needs_rev],
            color=[BLUE, ORANGE], edgecolor="white", linewidth=0.8, width=0.4)
    for val, label in zip([ok, needs_rev], ["Extraction OK", "Needs Review"]):
        pct = val / total * 100 if total else 0
        ax2.text(["Extraction OK", "Needs Review"].index(label),
                 val + total * 0.005,
                 f"{val:,}\n({pct:.1f}%)",
                 ha="center", va="bottom", fontsize=10)
    ax2.set_title("Human Review Required", fontsize=11)
    ax2.set_ylabel("Records")
    ax2.set_ylim(0, max(ok, needs_rev) * 1.2 if total else 1)
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out_path = out_dir / "step15_summary.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[step15] Summary figure -> {out_path}")


# =============================================================================
# Entrypoint
# =============================================================================

def run(config: dict) -> dict:
    import config as cfg_module
    out_root   = Path(config.get("out_dir", "outputs"))
    out_dir    = _out_dir(out_root)
    model      = config.get("step15_model", "") or DEFAULT_MODEL
    run_limit  = config.get("step15_run_limit", None)

    print(f"[step15] Model: {model}")
    print(f"[step15] Output dir: {out_dir}")

    t0 = time.time()
    df = load_inputs(out_root)

    if df.empty:
        print("[step15] No records to code — check step14 outputs.")
        return {"status": "empty"}

    df = extract_all(df, out_dir=out_dir, model=model, run_limit=run_limit)
    elapsed = time.time() - t0
    meta = write_outputs(df, out_dir, elapsed)

    print(f"[step15] Done in {meta['elapsed_hms']}.")
    return meta


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    run({"out_dir": str(here / "outputs")})
