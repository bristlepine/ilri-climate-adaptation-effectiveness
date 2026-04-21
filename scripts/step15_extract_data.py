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
  - step15_coded.csv          one row per coded study, all 19 coding fields
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
import signal
import sys
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

_here = Path(__file__).resolve().parent
_REPO_ROOT = _here.parent


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

# Default criteria YAML — versioned alongside codebook in documentation/
DEFAULT_CRITERIA_YML = (
    _REPO_ROOT
    / "documentation" / "coding" / "systematic-map" / "llm-criteria" / "criteria_sysmap_v1.yml"
)


# =============================================================================
# CODING SCHEMA  (protocol Table 3, Deliverable 3)
# =============================================================================
#
# Loaded at startup from DEFAULT_CRITERIA_YML (or a path set in config/CLI).
# Override per-round by pointing step15_criteria_yml in config.py to a newer
# criteria file (e.g. criteria_v2.yml after FT-R1a reconciliation).
#
# Hardcoded fallback below is used only when the YAML file is missing.
# Each entry: field_name -> (description, valid_values_hint)

_HARDCODED_SCHEMA: Dict[str, Tuple[str, str]] = {
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
    "strengths_and_limitations": (
        "Author-reported strengths and limitations combined. "
        "Label each as 'Strength: ...' or 'Limitation: ...'. Extract verbatim where possible.",
        "free text e.g. 'Strength: large sample. Limitation: single-season follow-up.'",
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

CODING_SCHEMA: Dict[str, Tuple[str, str]] = {}
_CRITERIA_DATA: dict = {}   # raw YAML — used by build_system_prompt() for full prompt


def _load_coding_schema(criteria_path: Path) -> bool:
    """
    Load CODING_SCHEMA and _CRITERIA_DATA from a YAML criteria file.
    Returns True on success, False on error (falls back to hardcoded schema).

    YAML format (mirrors abstract-screening/criteria/criteria.yml):
        fields:
          field_name:
            name: "Display name"
            extract: "What to look for and how to decide"
            valid_values: "..."
            r1_further_guidance: >   # optional, added after each round
              ...
    """
    global CODING_SCHEMA, _CRITERIA_DATA
    try:
        with open(criteria_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        fields = data.get("fields", {})
        if not fields:
            raise ValueError("No 'fields' key in criteria YAML")
        # CODING_SCHEMA: field → (extract text, valid_values) — used for output column building
        loaded = {
            name: (attrs.get("extract", attrs.get("description", "")), attrs.get("valid_values", ""))
            for name, attrs in fields.items()
        }
        CODING_SCHEMA.clear()
        CODING_SCHEMA.update(loaded)
        _CRITERIA_DATA.clear()
        _CRITERIA_DATA.update(data)
        print(f"[step15] Criteria: {criteria_path.name}  ({len(loaded)} fields)")
        return True
    except FileNotFoundError:
        print(f"[step15] Criteria file not found: {criteria_path} — using hardcoded defaults")
    except Exception as e:
        print(f"[step15] Error loading criteria ({criteria_path.name}): {e} — using hardcoded defaults")
    CODING_SCHEMA.clear()
    CODING_SCHEMA.update(_HARDCODED_SCHEMA)
    _CRITERIA_DATA.clear()
    return False


# Load at import time from default path (populated before any function runs)
_load_coding_schema(DEFAULT_CRITERIA_YML)


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


def _run_signature(*, model: str, criteria_path: Optional[Path] = None) -> str:
    """Hash of model + criteria file — changes when criteria are updated, invalidating cache."""
    criteria_hash = "hardcoded"
    if criteria_path and criteria_path.exists():
        with open(criteria_path, "rb") as f:
            criteria_hash = hashlib.sha1(f.read()).hexdigest()[:12]
    blob = f"model={model}\ncriteria={criteria_hash}\n"
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


# =============================================================================
# Text extraction from files
# =============================================================================

def _extract_pdf_text(path: Path, max_chars: int) -> Tuple[str, str]:
    """Returns (text, note). note is empty on success.
    Runs in a thread with a 45-second timeout so a hung PDF never blocks the loop.
    """
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FuturesTimeout

    def _read() -> Tuple[str, str]:
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

    with ThreadPoolExecutor(max_workers=1) as _ex:
        _fut = _ex.submit(_read)
        try:
            return _fut.result(timeout=45)
        except _FuturesTimeout:
            print(f"[step15] PDF timeout (45s): {path.name} — treating as needs_manual")
            return "", "pdf_timeout"
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

    # Only code from full text — abstract-only extraction is not reliable enough
    # to report and wastes compute. Skip abstract_only and missing_abstract records.
    before = len(codeable)
    codeable = codeable[codeable["coding_source"] == SOURCE_FULL_TEXT].reset_index(drop=True)
    print(f"[step15] Full-text only: {len(codeable):,} records (skipped {before - len(codeable):,} abstract-only / missing)")

    return codeable


# =============================================================================
# LLM prompt
# =============================================================================

import re as _re
_ROUND_GUIDANCE_PAT = _re.compile(r'^r\d+[a-z]*_further_guidance$')


def _build_extraction_prompt() -> str:
    """
    Build the EXTRACTION FIELDS block for the LLM prompt.
    Mirrors step10's _build_criteria_prompt(): reads field name, extract guidance,
    valid values, and any round-specific further guidance from _CRITERIA_DATA.
    Falls back to CODING_SCHEMA tuples when _CRITERIA_DATA is not loaded.
    """
    fields_data = _CRITERIA_DATA.get("fields", {})
    lines: List[str] = []

    for field, (extract_text, valid) in CODING_SCHEMA.items():
        attrs = fields_data.get(field, {}) if fields_data else {}
        display_name = attrs.get("name", field) if attrs else field

        lines.append(f"{display_name} ({field}):")
        lines.append(f"   - Extract: {extract_text.strip()}")
        lines.append(f"   - Valid values: {valid}")

        # Append any round-specific further guidance in round order
        if attrs:
            for key in sorted(k for k in attrs if _ROUND_GUIDANCE_PAT.match(k)):
                if extra := str(attrs[key] or "").strip():
                    lines.append(f"   - Further guidance [{key}]: {extra}")

        lines.append("")

    return "\n".join(lines).strip()


def build_system_prompt() -> str:
    """Build the complete LLM system prompt from loaded criteria."""
    fields_block = _build_extraction_prompt()

    # Build output JSON structure from CODING_SCHEMA field names
    json_fields = "\n".join(
        f'  "{f}": {{"value": <answer>, "confidence": "high|low|not_found", "note": "<justification>"}}'
        for f in CODING_SCHEMA
    )

    return f"""You are a systematic review research assistant. Your task is to extract structured \
data from a research paper for a systematic map of climate adaptation evidence for smallholder \
producers in low- and middle-income countries (LMICs).

For each field below:
- Extract the best answer from the text.
- Set confidence to "high" if clearly stated, "low" if inferred, "not_found" if absent.
- Set note to a brief justification or a direct quote.

If extracting from an ABSTRACT ONLY, many fields will be "not_found" — that is expected.

EXTRACTION FIELDS:
{fields_block}

OUTPUT FORMAT:
Return ONLY a valid JSON object with exactly these fields. No markdown, no commentary.
{{
{json_fields}
}}"""


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
    criteria_path: Optional[Path] = None,
) -> pd.DataFrame:

    _ollama_fail_fast(model)
    system_prompt = build_system_prompt()
    sig = _run_signature(model=model, criteria_path=criteria_path)

    jsonl_path = out_dir / "step15_results_details.jsonl"
    cache = _load_jsonl_cache(jsonl_path)
    if cache:
        _rewrite_jsonl(jsonl_path, cache)
    print(f"[step15] Cache warm: {len(cache):,} records")

    df = df.copy()
    _ckpt["df"] = df   # point checkpoint at the COPY being mutated by the loop
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

                # Periodic checkpoint — write CSV every N records so step16
                # can be run at any time without waiting for full completion
                if (i + 1) % CHECKPOINT_EVERY == 0:
                    _flush_checkpoint(f"checkpoint at {i+1:,}/{n_run:,}")

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
                    fname = Path(file_path).name if file_path else "no_file"
                    print(f"[step15] record {i+1}: reading PDF {fname!r} …")
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
                print(f"[step15] record {i+1}: LLM call ({src}) for {title[:60]!r} …")
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

def write_outputs(df: pd.DataFrame, out_dir: Path, elapsed: float, *, skip_figure: bool = False) -> dict:
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
    if "needs_review" not in df.columns:
        df["needs_review"] = False
    review_df = df[df["needs_review"].astype(str).str.lower() == "true"]
    review_csv = out_dir / "step15_needs_review.csv"
    review_df[out_cols].to_csv(review_csv, index=False)
    print(f"[step15] Needs-review CSV -> {review_csv}  ({len(review_df):,} rows)")

    # Meta
    src_counts = df["coding_source"].value_counts(dropna=False).to_dict()

    # rows_llm_coded = records where LLM actually ran and returned data
    # (publication_year_value non-empty is a reliable proxy)
    yr_col = "publication_year_value"
    if yr_col in df.columns:
        rows_llm_coded = int((df[yr_col].astype(str).str.strip() != "").sum())
    else:
        rows_llm_coded = 0

    meta = {
        "rows_total":           len(df),
        "rows_needs_review":    int(len(review_df)),
        "rows_llm_coded":       rows_llm_coded,   # actually coded by LLM (accurate progress)
        "coding_source_counts": {str(k): int(v) for k, v in src_counts.items()},
        "elapsed_seconds":      round(elapsed, 1),
        "elapsed_hms":          time.strftime("%H:%M:%S", time.gmtime(elapsed)),
        "timestamp_utc":        _now_utc(),
    }
    meta_path = out_dir / "step15_coded.meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[step15] Meta -> {meta_path}")

    if not skip_figure:
        print(f"[step15] Generating summary figure (this takes ~5s) …")
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
# Checkpoint / signal handling
# =============================================================================

CHECKPOINT_EVERY = 25   # write CSV every N records processed (cheap — just CSV+JSON, no figure)

# Shared mutable state so the SIGINT handler can flush on Ctrl+C
_ckpt: Dict[str, Any] = {"df": None, "out_dir": None, "t0": None}


def _flush_checkpoint(reason: str = "checkpoint") -> None:
    """Write step15_coded.csv with whatever has been coded so far."""
    df      = _ckpt.get("df")
    out_dir = _ckpt.get("out_dir")
    t0      = _ckpt.get("t0")
    if df is None or out_dir is None:
        return
    elapsed = time.time() - (t0 or time.time())
    t_ck = time.time()
    yr_col = "publication_year_value"
    n_actually_coded = int((df[yr_col].astype(str).str.strip() != "").sum()) if yr_col in df.columns else 0
    print(f"\n[step15] {reason} — {n_actually_coded:,} actually coded / {len(df):,} total — saving …")
    try:
        write_outputs(df, out_dir, elapsed, skip_figure=True)
        print(f"[step15] Checkpoint done in {time.time()-t_ck:.1f}s — run `python scripts/step16_map_visualise.py` anytime to update figures\n")
    except Exception as e:
        print(f"[step15] Checkpoint write failed: {e}")


def _sigint_handler(sig, frame):
    _flush_checkpoint("Interrupted (Ctrl+C)")
    sys.exit(0)


signal.signal(signal.SIGINT, _sigint_handler)


# =============================================================================
# Entrypoint
# =============================================================================

def run(config: dict) -> dict:
    model     = config.get("step15_model", "") or DEFAULT_MODEL
    run_limit = config.get("step15_run_limit", None)

    # Criteria YAML
    crit_str      = str(config.get("step15_criteria_yml", "")).strip()
    criteria_path = Path(crit_str) if crit_str else DEFAULT_CRITERIA_YML
    if criteria_path != DEFAULT_CRITERIA_YML or not CODING_SCHEMA:
        _load_coding_schema(criteria_path)

    # --- Calibration-round mode ---
    tmpl_str = str(config.get("step15_round_template", "")).strip()
    if tmpl_str:
        tmpl_path = Path(tmpl_str)
        pdfs_str  = str(config.get("step15_pdfs_dir", "")).strip()
        out_str   = str(config.get("step15_round_out_csv", "")).strip()

        # Infer pdfs dir and output path from template if not set
        stem = tmpl_path.stem
        m    = re.search(r"ft_r(\d+[a-z]?)", stem, re.IGNORECASE)
        round_label = f"FT-R{m.group(1).upper()}" if m else "FT"

        pdfs_path = Path(pdfs_str) if pdfs_str else tmpl_path.parent / f"{round_label} pdfs"
        if out_str:
            out_path = Path(out_str)
        else:
            new_stem = stem.replace("_XX", "_LLM").replace("_xx", "_LLM")
            out_path = tmpl_path.parent / (new_stem + ".csv")

        return run_calibration_round(
            tmpl_path,
            pdfs_path,
            out_path,
            model=model,
            criteria_path=criteria_path,
            run_limit=run_limit,
        )

    # --- Full-corpus mode ---
    out_root = Path(config.get("out_dir", "outputs"))
    out_dir  = _out_dir(out_root)
    print(f"[step15] Model: {model}")
    print(f"[step15] Output dir: {out_dir}")

    t0 = time.time()
    df = load_inputs(out_root)

    if df.empty:
        print("[step15] No records to code — check step14 outputs.")
        return {"status": "empty"}

    # Register shared state for SIGINT checkpoint handler
    _ckpt["df"]      = df
    _ckpt["out_dir"] = out_dir
    _ckpt["t0"]      = t0

    df = extract_all(df, out_dir=out_dir, model=model, run_limit=run_limit,
                     criteria_path=criteria_path)
    elapsed = time.time() - t0
    print(f"[step15] Extraction loop complete — writing final outputs …")
    meta = write_outputs(df, out_dir, elapsed)

    print(f"[step15] Done in {meta['elapsed_hms']}.")
    return meta


# =============================================================================
# Calibration-round mode
# =============================================================================

def run_calibration_round(
    template_csv: Path,
    pdfs_dir: Path,
    out_csv: Path,
    *,
    model: str = DEFAULT_MODEL,
    criteria_path: Optional[Path] = None,
    coder_id: str = "LLM",
    run_limit: Optional[int] = None,
) -> dict:
    """
    Run LLM coding for a single calibration round (FT-R1a, FT-R2a, etc.).

    Reads doi + filename from template_csv, finds each full-text file in
    pdfs_dir (falls back to outputs/step13/fulltext/ if not found there),
    calls the LLM, and writes results to out_csv in human coding sheet format:

        doi, filename, <19 coding fields>, coder_id, notes

    This is the same column layout as coding_ft_r1a_XX.csv so the LLM sheet
    can be placed next to the human sheets for reconciliation.

    Usage:
        python step15_extract_data.py \\
            --round-template documentation/coding/systematic-map/rounds/FT-R1a/coding_ft_r1a_XX.csv \\
            --pdfs-dir "documentation/coding/systematic-map/rounds/FT-R1a/FT-R1a pdfs" \\
            --out-csv documentation/coding/systematic-map/rounds/FT-R1a/coding_ft_r1a_LLM.csv
    """
    _load_coding_schema(criteria_path or DEFAULT_CRITERIA_YML)
    _ollama_fail_fast(model)
    system_prompt = build_system_prompt()
    sig = _run_signature(model=model, criteria_path=criteria_path or DEFAULT_CRITERIA_YML)

    tmpl = pd.read_csv(template_csv)
    n_total = len(tmpl)
    n_run = min(int(run_limit), n_total) if run_limit else n_total

    jsonl_path = out_csv.with_suffix(".jsonl")
    cache = _load_jsonl_cache(jsonl_path)

    print(f"[step15-cal] Template : {template_csv}")
    print(f"[step15-cal] PDFs dir : {pdfs_dir}")
    print(f"[step15-cal] Output   : {out_csv}")
    print(f"[step15-cal] Model    : {model}")
    print(f"[step15-cal] Papers   : {n_run}/{n_total}  |  cache warm: {len(cache)}")

    ft_fallback = _here / "outputs" / "step13" / "fulltext"
    rows: list = []
    t0 = time.time()
    n_hit = n_ok = n_err = 0

    with requests.Session() as session:
        with open(jsonl_path, "a", encoding="utf-8", buffering=1) as jf:
            for i in range(n_run):
                row      = tmpl.iloc[i]
                doi      = safe_str(row.get("doi", ""))
                filename = safe_str(row.get("filename", ""))
                ck       = f"cal:{doi}::{sig}"

                if i == 0 or (i + 1) % PRINT_EVERY == 0 or (i + 1) == n_run:
                    elapsed = time.time() - t0
                    rate    = (i + 1) / elapsed if elapsed > 0 else 0
                    eta     = (n_run - i - 1) / rate if rate > 0 else 0
                    print(
                        f"[step15-cal] {i+1}/{n_run} | "
                        f"ok={n_ok} err={n_err} cached={n_hit} | "
                        f"elapsed={elapsed:.0f}s ETA={eta:.0f}s"
                    )

                # --- Cache hit ---
                if ck in cache and safe_str(cache[ck].get("run_signature")) == sig:
                    cached = cache[ck]
                    out_row: dict = {"doi": doi, "filename": filename}
                    for field in CODING_SCHEMA:
                        out_row[field] = cached.get(f"{field}_value", "")
                    out_row["coder_id"] = coder_id
                    out_row["notes"]    = cached.get("_cal_note", "")
                    rows.append(out_row)
                    n_hit += 1
                    continue

                # --- Find full-text file ---
                file_path = pdfs_dir / filename
                text, note = extract_text(file_path, FULLTEXT_MAX_CHARS)
                if not text.strip() and (ft_fallback / filename).exists():
                    text, note = extract_text(ft_fallback / filename, FULLTEXT_MAX_CHARS)
                    note = f"fallback:{note}" if note else "fallback"

                coding_source = SOURCE_FULL_TEXT if text.strip() else SOURCE_NEEDS_MANUAL

                # --- LLM call (or skip if no text) ---
                if text.strip():
                    llm_resp = call_ollama(
                        session,
                        title=doi,
                        text=text,
                        coding_source=coding_source,
                        system_prompt=system_prompt,
                        model=model,
                    )
                    coded, parse_ok = _parse_response(llm_resp)
                    if parse_ok:
                        n_ok += 1
                    else:
                        n_err += 1
                        note = (note + " parse_error").strip()
                else:
                    coded, parse_ok = {}, False
                    n_err += 1
                    note = (note or "no_text").strip()

                flat = flatten_coded(coded, parse_ok, coding_source)

                # Build cache record
                rec: dict = {
                    "run_signature": sig,
                    "timestamp_utc": _now_utc(),
                    "cache_key":     ck,
                    "doi":           doi,
                    "filename":      filename,
                    "_cal_note":     note,
                    **flat,
                }
                cache[ck] = rec
                jf.write(json.dumps(rec, ensure_ascii=False) + "\n")

                # Build output row (coding-sheet format)
                out_row = {"doi": doi, "filename": filename}
                for field in CODING_SCHEMA:
                    out_row[field] = flat.get(f"{field}_value", "")
                out_row["coder_id"] = coder_id
                out_row["notes"]    = note
                rows.append(out_row)

    # Write coding sheet CSV
    cols = ["doi", "filename"] + list(CODING_SCHEMA.keys()) + ["coder_id", "notes"]
    out_df = pd.DataFrame(rows, columns=cols)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    elapsed = time.time() - t0
    print(
        f"[step15-cal] Done in {time.strftime('%H:%M:%S', time.gmtime(elapsed))} | "
        f"ok={n_ok} err={n_err} cached={n_hit} | output={out_csv}"
    )
    return {"rows": len(out_df), "ok": n_ok, "err": n_err, "cached": n_hit,
            "elapsed_seconds": round(elapsed, 1)}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Step 15 — data extraction")
    ap.add_argument(
        "--round-template",
        metavar="CSV",
        help="Calibration-round template CSV (e.g. coding_ft_r1a_XX.csv). "
             "Runs calibration mode instead of full-corpus mode.",
    )
    ap.add_argument(
        "--pdfs-dir",
        metavar="DIR",
        help="Folder containing PDFs/HTMLs for the calibration round. "
             "Defaults to '<template-dir>/<round> pdfs/' if omitted.",
    )
    ap.add_argument(
        "--out-csv",
        metavar="CSV",
        help="Output CSV path. Defaults to coding_ft_r1a_LLM.csv next to template.",
    )
    ap.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model (default: {DEFAULT_MODEL})",
    )
    ap.add_argument(
        "--criteria-yml",
        metavar="YML",
        help="Criteria YAML file (default: criteria_sysmap_v1.yml)",
    )
    ap.add_argument(
        "--run-limit",
        type=int,
        metavar="N",
        help="Process only first N papers (for testing)",
    )
    args = ap.parse_args()

    if args.round_template:
        tmpl_path = Path(args.round_template).resolve()

        # Infer round label from template name (e.g. coding_ft_r1a_XX -> FT-R1a)
        stem = tmpl_path.stem  # e.g. coding_ft_r1a_XX
        # Extract round identifier: ft_r1a → FT-R1a
        m = re.search(r"ft_r(\d+[a-z]?)", stem, re.IGNORECASE)
        round_label = f"FT-R{m.group(1).upper()}" if m else "FT"

        if args.pdfs_dir:
            pdfs = Path(args.pdfs_dir).resolve()
        else:
            pdfs = tmpl_path.parent / f"{round_label} pdfs"

        if args.out_csv:
            out = Path(args.out_csv).resolve()
        else:
            new_stem = stem.replace("_XX", "_LLM").replace("_xx", "_LLM")
            out = tmpl_path.parent / (new_stem + ".csv")

        crit = Path(args.criteria_yml).resolve() if args.criteria_yml else None

        run_calibration_round(
            tmpl_path,
            pdfs,
            out,
            model=args.model,
            criteria_path=crit,
            run_limit=args.run_limit,
        )
    else:
        run({"out_dir": str(_here / "outputs")})
