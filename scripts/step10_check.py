#!/usr/bin/env python3
"""
step10_check.py

Step 10: Check Eligibility + Ollama (Step9 matching + fail-fast Ollama preflight + progress logs)

Reads:
  1) Step 9 enriched CSV (preferred): outputs/step9/step9_scopus_enriched.csv
     (fallback: scripts/outputs/step9/step9_scopus_enriched.csv relative to this file)
  2) A fixed calibration Excel: scripts/data/calibration_r1_205.xlsx

Then:
  - Deterministic row-wise matching (preserve Excel row order) using priority:
      scopus_id -> eid -> dedupe_key -> doi -> title+year -> title
  - Uses BEST available abstract from Step9:
      abstract -> xref_abstract
  - Fill screen_decision + screen_reasons using Ollama + criteria.yml
  - If abstract missing: MAYBE/UNCLEAR (no model call)
  - Resume/caching via JSONL (key-based; last record wins)

Outputs (under outputs/step10/):
  - step10_check.csv
  - step10_check.meta.json
  - step10_check_details.jsonl
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter, defaultdict

import pandas as pd
import requests
import yaml
from difflib import SequenceMatcher


# =============================================================================
# SETTINGS (deterministic)
# =============================================================================

CHECK_RIS = (Path(__file__).resolve().parent / "data" / "calibration_r1_205.ris.txt").resolve()

# If None -> run all rows, else run first N rows deterministically
RUN_LIMIT: Optional[int] = None

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:14b"
TEMPERATURE = 0.0
OLLAMA_KEEP_ALIVE = "30m"
OLLAMA_THINK = False

# Progress logging knobs
PRINT_EVERY = 25          # progress summary every N rows
PRINT_MATCH_DEBUG = 15    # print first N non-matches in detail
PRINT_BEFORE_OLLAMA = True


# =============================================================================
# Helpers
# =============================================================================

def _parse_ris_records(text: str) -> List[Dict[str, Any]]:
    """
    Minimal RIS parser.
    Returns list of records as dict(tag -> value or list of values for repeated tags like AU).
    """
    records: List[Dict[str, Any]] = []
    cur: Dict[str, Any] = {}
    for line in text.splitlines():
        line = line.rstrip("\n")
        if line.startswith("ER  -"):
            if cur:
                records.append(cur)
            cur = {}
            continue

        m = re.match(r"^([A-Z0-9]{2})\s{2}-\s?(.*)$", line)
        if not m:
            # continuation line: append to last tag if possible
            if cur and "_last_tag" in cur:
                t = cur["_last_tag"]
                if isinstance(cur.get(t), list):
                    cur[t][-1] = (cur[t][-1] + " " + line.strip()).strip()
                else:
                    cur[t] = (safe_str(cur.get(t)) + " " + line.strip()).strip()
            continue

        tag, val = m.group(1), m.group(2)
        val = safe_str(val)
        cur["_last_tag"] = tag

        # repeated tags become lists (AU, KW, etc.)
        if tag in cur and tag != "_last_tag":
            if not isinstance(cur[tag], list):
                cur[tag] = [cur[tag]]
            cur[tag].append(val)
        else:
            cur[tag] = val

    if cur:
        records.append(cur)
    # remove internal helper key
    for r in records:
        r.pop("_last_tag", None)
    return records


def _first_tag(rec: Dict[str, Any], tags: List[str]) -> str:
    for t in tags:
        v = rec.get(t)
        if isinstance(v, list):
            v = v[0] if v else ""
        v = safe_str(v)
        if v:
            return v
    return ""


def _all_tag_join(rec: Dict[str, Any], tag: str, sep: str = "; ") -> str:
    v = rec.get(tag)
    if isinstance(v, list):
        return sep.join([safe_str(x) for x in v if safe_str(x)])
    return safe_str(v)


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    s = str(x)
    s = s.replace("\r", " ").replace("\n", " ")
    s = " ".join(s.split()).strip()
    return s


def normalize_doi(doi: Any) -> str:
    s = safe_str(doi)
    if not s:
        return ""
    s = s.strip()
    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s, flags=re.IGNORECASE).strip()
    s = s.rstrip(" .;,)\t")
    return s.lower()


def year_from_any(x: Any) -> str:
    s = safe_str(x)
    if not s:
        return ""
    m = re.search(r"\b(19|20)\d{2}\b", s)
    return m.group(0) if m else ""


def clean_for_key_title(s: Any) -> str:
    s = safe_str(s).lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def clean_id_digits(x: Any) -> str:
    """Excel IDs sometimes come as floats/scientific notation; normalize to digits-only."""
    s = safe_str(x)
    if not s:
        return ""
    s = s.replace(",", "")
    # handle scientific notation like 1.259E+08
    try:
        if re.search(r"e\+?\d+", s, flags=re.IGNORECASE):
            n = int(float(s))
            return str(n)
    except Exception:
        pass
    digits = re.sub(r"\D+", "", s)
    return digits


def dedupe_key_like_step8(row: pd.Series) -> str:
    """
    Mirror Step8-style key behavior:
      1) doi:<doi_core>
      2) ty:<clean_title>:<year>
      3) t:<clean_title>
      4) eid:<eid> / sid:<scopus_id>
    """
    doi = normalize_doi(row.get("doi", ""))
    if doi:
        return f"doi:{doi}"

    t = clean_for_key_title(row.get("title", ""))
    y = year_from_any(row.get("coverDate", "")) or year_from_any(row.get("year", "")) or ""
    if t and y:
        return f"ty:{t}:{y}"
    if t:
        return f"t:{t}"

    eid = safe_str(row.get("eid", ""))
    sid = clean_id_digits(row.get("scopus_id", ""))
    if eid:
        return f"eid:{eid}"
    if sid:
        return f"sid:{sid}"
    return ""


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find a column by exact match (case-insensitive) then substring match."""
    cand_l = [c.lower() for c in candidates]
    for col in df.columns:
        if col.strip().lower() in cand_l:
            return col
    for col in df.columns:
        cl = col.lower()
        for c in cand_l:
            if len(c) >= 3 and c in cl:
                return col
    return None


def similarity(a: Any, b: Any) -> float:
    a = safe_str(a).lower()
    b = safe_str(b).lower()
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def best_abstract_from_row(row: pd.Series, *, candidates: List[str]) -> Tuple[str, str]:
    """Return (abstract_text, source_col) using first non-empty candidate field."""
    for c in candidates:
        if c in row.index:
            v = safe_str(row.get(c))
            if v.strip():
                return v, c
    return "", ""


# =============================================================================
# Ollama preflight (FAIL FAST)
# =============================================================================

def _ollama_base_url() -> str:
    # OLLAMA_URL is .../api/generate
    return OLLAMA_URL.split("/api/")[0]


def _ollama_check_ready(*, timeout: int = 5) -> None:
    base = _ollama_base_url()
    try:
        r = requests.get(f"{base}/api/version", timeout=timeout)
        r.raise_for_status()
    except Exception as e:
        raise SystemExit(f"Ollama not reachable at {base} ({type(e).__name__}: {e})")


def _ollama_model_exists(model: str, *, timeout: int = 8) -> bool:
    base = _ollama_base_url()
    try:
        r = requests.get(f"{base}/api/tags", timeout=timeout)
        r.raise_for_status()
        data = r.json() or {}
        models = data.get("models") or []
        names = {safe_str(m.get("name", "")).strip() for m in models}
        return safe_str(model).strip() in names
    except Exception:
        return False


def _ollama_fail_fast(model: str) -> None:
    _ollama_check_ready()
    if not _ollama_model_exists(model):
        base = _ollama_base_url()
        raise SystemExit(
            f"Ollama is running at {base}, but model '{model}' is not installed.\n"
            f"Fix: run `ollama pull {model}` or change DEFAULT_MODEL / config to an installed model."
        )


# =============================================================================
# Screening helpers
# =============================================================================

def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing criteria file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _build_criteria_prompt(criteria_dict: dict) -> str:
    prompt_lines: List[str] = []
    for key in sorted(criteria_dict.keys()):
        val = criteria_dict[key] or {}
        prompt_lines.append(f"{val.get('name', key)}:")
        prompt_lines.append(f"   - INCLUDE: {val.get('include','')}")
        prompt_lines.append(f"   - EXCLUDE: {val.get('exclude','')}")
        prompt_lines.append("")
    return "\n".join(prompt_lines).strip()


def _run_signature(*, model: str, criteria_text: str, min_year: int) -> str:
    blob = f"model={model}\nmin_year={min_year}\ncriteria=\n{criteria_text}\n"
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def call_ollama(
    session: requests.Session,
    *,
    title: str,
    abstract: str,
    criteria_prompt: str,
    model: str,
) -> str:
    system_prompt = f"""You are a strict research assistant screening papers for a systematic review.
Analyze the Title and Abstract against these eligibility criteria:
{criteria_prompt}

OUTPUT FORMAT:
Return ONLY a valid JSON object. Do not use markdown blocks.
Structure:
{{
  "1_population": {{ "decision": "yes|no|unclear", "reason": "brief reason", "quote": "exact substring from text" }},
  "2_concept": {{ "decision": "yes|no|unclear", "reason": "brief reason", "quote": "exact substring from text" }},
  "3_context": {{ "decision": "yes|no|unclear", "reason": "brief reason", "quote": "exact substring from text" }},
  "4_methodology": {{ "decision": "yes|no|unclear", "reason": "brief reason", "quote": "exact substring from text" }},
  "5_geography": {{ "decision": "yes|no|unclear", "reason": "brief reason", "quote": "exact substring from text" }}
}}
""".strip()

    user_message = f"TITLE: {title}\n\nABSTRACT: {abstract}"

    payload = {
        "model": model,
        "system": system_prompt,
        "prompt": user_message,
        "stream": False,
        "format": "json",
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {"temperature": TEMPERATURE},
        "think": OLLAMA_THINK,
    }

    try:
        r = session.post(OLLAMA_URL, json=payload, timeout=180)
        if r.status_code == 200:
            return r.json().get("response", "")
        return json.dumps({"error": f"HTTP {r.status_code}", "body": (r.text or "")[:500]})
    except Exception as e:
        return json.dumps({"error": str(e)})


def verify_quote(full_text: str, quote: str) -> Dict[str, Any]:
    if not quote or not full_text:
        return {"verified": False, "start": -1}
    txt_clean = " ".join(str(full_text).lower().split())
    qt_clean = " ".join(str(quote).lower().split())
    idx = txt_clean.find(qt_clean)
    return {"verified": True, "start": idx} if idx != -1 else {"verified": False, "start": -1}


def _classify_decision(val: Any) -> str:
    """Robust classifier: returns yes/no/unclear only."""
    v = str(val or "").strip().lower()
    if v == "yes":
        return "yes"
    if v == "no":
        return "no"
    if v.startswith("include"):
        return "yes"
    if v.startswith("exclude"):
        return "no"
    if v.startswith("unclear") or v in ("pending", "skipped"):
        return "unclear"
    return "unclear"


def _load_jsonl_last_by_key(path: Path) -> dict[str, dict]:
    last: dict[str, dict] = {}
    if not path.exists():
        return last
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
            except Exception:
                continue
            k = safe_str(j.get("key"))
            if k:
                last[k] = j  # last wins
    return last


def _rewrite_jsonl(path: Path, last_by_key: dict[str, dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as wf:
        for _, j in last_by_key.items():
            wf.write(json.dumps(j, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


# =============================================================================
# IO paths
# =============================================================================

def default_outputs_root() -> Path:
    return Path("outputs")


def step9_path(out_root: Path) -> Path:
    """
    Prefer:
      outputs/step9/step9_scopus_enriched.csv
    Fallback:
      <thisfile>/outputs/step9/step9_scopus_enriched.csv
    """
    p1 = out_root / "step9" / "step9_scopus_enriched.csv"
    if p1.exists() and p1.stat().st_size > 0:
        return p1

    here = Path(__file__).resolve().parent
    p2 = here / "outputs" / "step9" / "step9_scopus_enriched.csv"
    if p2.exists() and p2.stat().st_size > 0:
        return p2

    raise SystemExit(f"Missing Step 9 enriched CSV at {p1} or {p2}")


def check_paths(out_root: Path) -> Tuple[Path, Path]:
    d = out_root / "step10"
    return d / "step10_check.csv", d / "step10_check.meta.json"


def check_jsonl_path(out_root: Path) -> Path:
    return out_root / "step10" / "step10_check_details.jsonl"


# =============================================================================
# Load Step 9 enriched
# =============================================================================

def load_step9_enriched(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    if df.empty:
        raise SystemExit(f"Step 9 enriched CSV is empty: {path}")

    # Ensure expected fields exist
    for c in [
        "dedupe_key", "title", "publicationName", "year", "coverDate", "doi",
        "prism_url", "author_names", "creator", "authors_joined",
        "xref_abstract", "abstract",
        "eid", "scopus_id",
        "abstract_source", "abstract_status",
    ]:
        if c not in df.columns:
            df[c] = ""

    # Normalize
    df["dedupe_key"] = df["dedupe_key"].apply(safe_str)
    df["doi"] = df["doi"].apply(normalize_doi)
    df["title"] = df["title"].apply(safe_str)
    df["coverDate"] = df["coverDate"].apply(safe_str)
    df["year"] = df["year"].apply(year_from_any)
    df["scopus_id"] = df["scopus_id"].apply(clean_id_digits)
    df["eid"] = df["eid"].apply(safe_str)

    # Fill missing dedupe_key if needed
    needs = df["dedupe_key"] == ""
    if needs.any():
        df.loc[needs, "dedupe_key"] = df.loc[needs].apply(dedupe_key_like_step8, axis=1)

    df = df[df["dedupe_key"] != ""].copy()

    # Alternate title keys
    df["_alt_key_ty"] = df.apply(
        lambda r: (
            f"ty:{clean_for_key_title(r.get('title',''))}:{safe_str(r.get('year',''))}"
            if clean_for_key_title(r.get("title", "")) and safe_str(r.get("year", ""))
            else ""
        ),
        axis=1,
    )
    df["_alt_key_t"] = df.apply(
        lambda r: (f"t:{clean_for_key_title(r.get('title',''))}" if clean_for_key_title(r.get("title","")) else ""),
        axis=1,
    )

    # Best abstract selection
    abs_candidates = ["abstract", "xref_abstract"]
    best_abs, best_src = [], []
    for _, row in df.iterrows():
        a, src = best_abstract_from_row(row, candidates=abs_candidates)
        best_abs.append(a)
        best_src.append(src)
    df["_abstract_best"] = best_abs
    df["_abstract_best_src"] = best_src

    return df.reset_index(drop=True)


# =============================================================================
# Load calibration Excel
# =============================================================================

def load_check_data_ris(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"Missing calibration RIS: {path}")

    text = path.read_text(encoding="utf-8", errors="ignore")
    recs = _parse_ris_records(text)
    if not recs:
        raise SystemExit(f"No RIS records parsed from: {path}")

    rows: List[Dict[str, Any]] = []
    for idx, r in enumerate(recs, start=1):
        # Common RIS tags:
        # ID/AN = identifier (often numeric in EPPI exports)
        # TI/T1 = title
        # PY/Y1 = year/date
        # DO = DOI
        # AU = authors (repeatable)
        # AB = abstract
        rid = _first_tag(r, ["ID", "AN"])
        title = _first_tag(r, ["TI", "T1"])
        year = year_from_any(_first_tag(r, ["PY", "Y1"]))
        doi = normalize_doi(_first_tag(r, ["DO"]))
        authors = _all_tag_join(r, "AU")

        abstract = safe_str(r.get("AB", ""))

        # Keep some original-ish columns for traceability
        rows.append(
            {
                "ID": rid,
                "Title": title,
                "Year": year,
                "DOI": doi,
                "Authors": authors,
                "Abstract": abstract,
                "Ref. Type": _first_tag(r, ["TY"]),
                "_check_row_id": idx,
            }
        )

    df = pd.DataFrame(rows)
    # Preserve "original columns" concept (so your output ordering logic still works)
    df.attrs["original_cols"] = list(df.columns)

    # Standardize helper fields for matching (same names your code expects)
    df["_check_scopus_id"] = df["ID"].apply(clean_id_digits)
    df["_check_title"] = df["Title"].apply(safe_str)
    df["_check_year"] = df["Year"].apply(year_from_any)
    df["_check_doi"] = df["DOI"].apply(normalize_doi)
    df["_check_author_hint"] = df["Authors"].apply(safe_str)
    df["_check_abstract"] = df["Abstract"].apply(safe_str)

    # Precompute title keys (same as Excel path)
    df["_check_key_doi"] = df["_check_doi"].apply(lambda d: f"doi:{d}" if d else "")
    df["_check_key_ty"] = df.apply(
        lambda r: (
            f"ty:{clean_for_key_title(r.get('_check_title',''))}:{r.get('_check_year','')}"
            if clean_for_key_title(r.get("_check_title","")) and r.get("_check_year","")
            else ""
        ),
        axis=1,
    )
    df["_check_key_t"] = df["_check_title"].apply(
        lambda t: f"t:{clean_for_key_title(t)}" if clean_for_key_title(t) else ""
    )

    return df

# =============================================================================
# Matching Step9 -> Excel rows
# =============================================================================

def build_step9_indexes(step9: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    idx_scopus: Dict[str, int] = {}
    idx_eid: Dict[str, int] = {}
    idx_dedupe: Dict[str, int] = {}
    idx_doi: Dict[str, int] = {}
    idx_ty: Dict[str, int] = {}
    idx_t: Dict[str, int] = {}

    for i, r in step9.iterrows():
        sid = clean_id_digits(r.get("scopus_id", ""))
        if sid and sid not in idx_scopus:
            idx_scopus[sid] = i

        eid = safe_str(r.get("eid", ""))
        if eid and eid not in idx_eid:
            idx_eid[eid] = i

        dk = safe_str(r.get("dedupe_key", ""))
        if dk and dk not in idx_dedupe:
            idx_dedupe[dk] = i

        doi = normalize_doi(r.get("doi", ""))
        if doi:
            kdoi = f"doi:{doi}"
            if kdoi not in idx_doi:
                idx_doi[kdoi] = i

        ty = safe_str(r.get("_alt_key_ty", ""))
        if ty and ty not in idx_ty:
            idx_ty[ty] = i

        tt = safe_str(r.get("_alt_key_t", ""))
        if tt and tt not in idx_t:
            idx_t[tt] = i

    return {"scopus_id": idx_scopus, "eid": idx_eid, "dedupe": idx_dedupe, "doi": idx_doi, "ty": idx_ty, "t": idx_t}


def match_one_row(
    check_row: pd.Series,
    *,
    step9: pd.DataFrame,
    idx: Dict[str, Dict[str, int]],
) -> Tuple[dict, Optional[pd.Series]]:
    sid = clean_id_digits(check_row.get("_check_scopus_id", ""))
    doi_key = safe_str(check_row.get("_check_key_doi", ""))
    ty_key = safe_str(check_row.get("_check_key_ty", ""))
    t_key = safe_str(check_row.get("_check_key_t", ""))

    if sid and sid in idx["scopus_id"]:
        j = idx["scopus_id"][sid]
        return {"match_source": "scopus_id", "match_value": sid}, step9.iloc[j]

    # if Excel ID is actually an EID:
    eid_guess = safe_str(check_row.get("_check_scopus_id", ""))
    if eid_guess and eid_guess in idx["eid"]:
        j = idx["eid"][eid_guess]
        return {"match_source": "eid", "match_value": eid_guess}, step9.iloc[j]

    excel_dk = safe_str(check_row.get("dedupe_key", ""))
    if excel_dk and excel_dk in idx["dedupe"]:
        j = idx["dedupe"][excel_dk]
        return {"match_source": "dedupe_key", "match_value": excel_dk}, step9.iloc[j]

    if doi_key and doi_key in idx["doi"]:
        j = idx["doi"][doi_key]
        return {"match_source": "doi", "match_value": doi_key}, step9.iloc[j]

    if ty_key and ty_key in idx["ty"]:
        j = idx["ty"][ty_key]
        return {"match_source": "title_year", "match_value": ty_key}, step9.iloc[j]

    if t_key and t_key in idx["t"]:
        j = idx["t"][t_key]
        return {"match_source": "title_only", "match_value": t_key}, step9.iloc[j]

    return {"match_source": "no_match", "match_value": ""}, None


def build_check_table_from_step9(step9: pd.DataFrame, check: pd.DataFrame) -> pd.DataFrame:
    idx = build_step9_indexes(step9)

    out = check.copy()

    # Screening outputs
    out["screen_decision"] = ""
    out["screen_reasons"] = ""
    out["screen_rule_hits"] = ""
    out["screen_checked_at_utc"] = ""

    # Audit
    out["match_source"] = ""
    out["match_value"] = ""
    out["match_title_score"] = 0.0
    out["match_author_hint"] = ""
    out["abstract_used_from"] = ""  # which Step9 col

    bring_cols = [
        "dedupe_key", "title", "publicationName", "year", "coverDate", "doi", "prism_url",
        "author_names", "creator", "authors_joined", "eid", "scopus_id",
        "abstract_source", "abstract_status",
        "_abstract_best", "_abstract_best_src",
    ]
    for c in bring_cols:
        out[f"step9__{c}"] = ""

    n = len(out)
    t0 = time.time()
    mismatches_printed = 0
    matched = 0
    with_abs = 0

    for i in range(n):
        row = out.iloc[i]
        meta, m = match_one_row(row, step9=step9, idx=idx)

        out.at[i, "match_source"] = meta["match_source"]
        out.at[i, "match_value"] = meta["match_value"]

        if m is not None:
            matched += 1
            for c in bring_cols:
                out.at[i, f"step9__{c}"] = safe_str(m.get(c, ""))

            excel_title = safe_str(row.get("_check_title", ""))
            step9_title = safe_str(m.get("title", ""))
            out.at[i, "match_title_score"] = float(similarity(excel_title, step9_title))

            hint = safe_str(row.get("_check_author_hint", ""))
            step9_auth_blob = " ".join(
                [safe_str(m.get("creator", "")), safe_str(m.get("author_names", "")), safe_str(m.get("authors_joined", ""))]
            ).lower()
            author_ok = (hint.lower().split("(")[0].strip() in step9_auth_blob) if hint else False
            out.at[i, "match_author_hint"] = "ok" if author_ok else ("no_hint" if not hint else "mismatch")

            best_abs = safe_str(m.get("_abstract_best", ""))
            best_src = safe_str(m.get("_abstract_best_src", ""))
            if best_abs.strip():
                with_abs += 1
                out.at[i, "abstract_used_from"] = best_src

        else:
            if mismatches_printed < PRINT_MATCH_DEBUG:
                mismatches_printed += 1
                print(
                    f"[step10] NO MATCH row {i+1}/{n} | "
                    f"ID={safe_str(row.get('_check_scopus_id',''))} | "
                    f"DOI={safe_str(row.get('_check_doi',''))} | "
                    f"YEAR={safe_str(row.get('_check_year',''))} | "
                    f"TITLE={safe_str(row.get('_check_title',''))[:120]}"
                )

        if (i + 1) % PRINT_EVERY == 0 or (i + 1) == n:
            elapsed = time.time() - t0
            left = n - (i + 1)
            print(
                f"[step10] Match progress {i+1:,}/{n:,} | matched={matched:,} | with_abstract={with_abs:,} | "
                f"left={left:,} | elapsed={elapsed:,.1f}s"
            )

    # Column order: decision cols first, then original Excel cols, then rest
    original_cols = list(check.attrs.get("original_cols") or [])
    original_cols = [c for c in original_cols if c in out.columns]
    decision_cols = ["screen_decision", "screen_reasons"]
    used = set(decision_cols + original_cols)
    rest_cols = [c for c in out.columns if c not in used]
    return out[decision_cols + original_cols + rest_cols].copy()


# =============================================================================
# Apply Ollama screening
# =============================================================================

def apply_ollama_screening(check_df: pd.DataFrame, *, out_root: Path, model: str, run_limit: Optional[int]) -> pd.DataFrame:
    here = Path(__file__).resolve().parent
    criteria_yml = here / "criteria.yml"
    cfg = _load_yaml(criteria_yml)

    min_year = int((cfg.get("hard_filters", {}) or {}).get("min_year", 2005))
    criteria_text = _build_criteria_prompt(cfg.get("criteria", {}) or {})
    crit_keys = sorted((cfg.get("criteria", {}) or {}).keys())
    sig = _run_signature(model=model, criteria_text=criteria_text, min_year=min_year)

    def _print_row_result(*, i: int, n_run: int, k: str, title: str, decision: str, reasons: str, tag: str) -> None:
        # Single-line, easy to grep
        k_short = (k or "")[:50]
        t_short = (title or "")[:70].replace("\n", " ").replace("\r", " ")
        r_short = (reasons or "")[:140].replace("\n", " ").replace("\r", " ")
        print(f"[step10]  -> [{tag}] {i+1}/{n_run} | {decision} | key={k_short} | title='{t_short}' | {r_short}")


    # FAIL FAST before doing anything expensive
    _ollama_fail_fast(model)

    jsonl_path = check_jsonl_path(out_root)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    last_by_key = _load_jsonl_last_by_key(jsonl_path)
    if jsonl_path.exists():
        _rewrite_jsonl(jsonl_path, last_by_key)
    processed = set(last_by_key.keys())
    print(f"[step10] Cache warm: {len(processed):,} records loaded from {jsonl_path} (exists={jsonl_path.exists()})")


    title_col = "step9__title" if "step9__title" in check_df.columns else "_check_title"

    # Prefer RIS Abstract; fallback to Step9 best; then anything else
    if "_check_abstract" in check_df.columns:
        abs_col = "_check_abstract"
    elif "step9___abstract_best" in check_df.columns:
        abs_col = "step9___abstract_best"
    else:
        abs_col = ""

    def _fallback_key(*, i: int, title: str, doi: str) -> str:
        base = f"{i}|{title}|{doi}".encode("utf-8", errors="ignore")
        return "row:" + hashlib.sha1(base).hexdigest()[:16]


    def stable_key(i: int) -> str:
        k = safe_str(check_df.at[i, "step9__dedupe_key"]) if "step9__dedupe_key" in check_df.columns else ""
        if not k:
            sid = safe_str(check_df.at[i, "_check_scopus_id"]) if "_check_scopus_id" in check_df.columns else ""
            if sid:
                k = f"sid:{clean_id_digits(sid)}"
        if not k and "_check_key_doi" in check_df.columns:
            k = safe_str(check_df.at[i, "_check_key_doi"])
        if not k and "_check_key_ty" in check_df.columns:
            k = safe_str(check_df.at[i, "_check_key_ty"])
        if not k and "_check_key_t" in check_df.columns:
            k = safe_str(check_df.at[i, "_check_key_t"])

        # FINAL fallback so caching always works
        if not k:
            title_tmp = safe_str(check_df.at[i, title_col]) if title_col in check_df.columns else ""
            doi_tmp = safe_str(check_df.at[i, "_check_doi"]) if "_check_doi" in check_df.columns else ""
            k = _fallback_key(i=i, title=title_tmp, doi=doi_tmp)
        return k

    n_total = len(check_df)
    n_run = int(run_limit) if (run_limit is not None and int(run_limit) > 0) else n_total
    n_run = min(n_run, n_total)

    t0 = time.time()
    print(f"[step10] Screening start | rows={n_total:,} | run_limit={run_limit} -> running={n_run:,} | model={model}")
    print(f"[step10] Using title_col='{title_col}' | abs_col='{abs_col}' | cache='{jsonl_path}'")

    with requests.Session() as session:
        with open(jsonl_path, "a", encoding="utf-8", buffering=1) as jf:
            for i in range(n_run):
                k = stable_key(i)
                title = safe_str(check_df.at[i, title_col]) if title_col in check_df.columns else safe_str(check_df.at[i, "_check_title"])
                abstract = safe_str(check_df.at[i, abs_col]) if abs_col in check_df.columns else ""

                if i == 0 or (i + 1) % PRINT_EVERY == 0 or (i + 1) == n_run:
                    elapsed = time.time() - t0
                    left = n_run - (i + 1)
                    print(f"[step10] Row {i+1:,}/{n_run:,} | left={left:,} | key={k[:60]} | abstract_len={len(abstract):,} | elapsed={elapsed:,.1f}s")

                # Missing abstract -> no model call
                if not abstract.strip():
                    check_df.at[i, "screen_decision"] = "MAYBE/UNCLEAR"
                    check_df.at[i, "screen_reasons"] = "Missing abstract"
                    check_df.at[i, "screen_rule_hits"] = ""
                    check_df.at[i, "screen_checked_at_utc"] = ""

                    rec = {
                        "run_signature": sig,
                        "timestamp_utc": _now_utc(),
                        "key": k,
                        "title": title,
                        "abstract_present": False,
                        "min_year": min_year,
                        "criteria_keys": crit_keys,
                        "screen_decision": "MAYBE/UNCLEAR",
                        "screen_reasons": "Missing abstract",
                        "screen_rule_hits": "",
                    }
                    jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    jf.flush()
                    if k:
                        processed.add(k)
                        last_by_key[k] = rec

                    _print_row_result(
                        i=i, n_run=n_run, k=k, title=title,
                        decision=rec["screen_decision"],
                        reasons=rec["screen_reasons"],
                        tag="NO_ABSTRACT",
                    )

                    continue

                # Cache hit (only if signature matches current run)
                if k and k in processed:
                    cached = last_by_key.get(k, {})
                    if safe_str(cached.get("run_signature")) == sig:
                        cd = safe_str(cached.get("screen_decision"))
                        cr = safe_str(cached.get("screen_reasons"))
                        ch = safe_str(cached.get("screen_rule_hits"))

                        check_df.at[i, "screen_decision"] = cd
                        check_df.at[i, "screen_reasons"] = cr
                        check_df.at[i, "screen_rule_hits"] = ch
                        check_df.at[i, "screen_checked_at_utc"] = ""

                        _print_row_result(
                            i=i, n_run=n_run, k=k, title=title,
                            decision=cd, reasons=cr, tag="CACHED",
                        )
                        continue



                if PRINT_BEFORE_OLLAMA:
                    print(f"[step10]  - calling ollama: row {i+1}/{n_run} | key={k[:60]} | title='{title[:80]}'")

                llm_resp = call_ollama(
                    session,
                    title=title,
                    abstract=abstract,
                    criteria_prompt=criteria_text,
                    model=model,
                )

                rec = {
                    "run_signature": sig,
                    "timestamp_utc": _now_utc(),
                    "key": k,
                    "title": title,
                    "abstract_present": True,
                    "min_year": min_year,
                    "criteria_keys": crit_keys,
                    "screen_decision": "MAYBE/UNCLEAR",
                    "screen_reasons": "",
                    "screen_rule_hits": "",
                }

                try:
                    data = json.loads(llm_resp)
                    full_text = f"{title}\n{abstract}"

                    # --- NEW: detect Ollama failure payloads and mark explicitly
                    if isinstance(data, dict) and "error" in data:
                        rec["screen_decision"] = "MAYBE/UNCLEAR"
                        rec["screen_reasons"] = f"MODEL_ERROR: {safe_str(data.get('error'))}"
                        rec["screen_rule_hits"] = json.dumps({"raw": data}, ensure_ascii=False)
                        # write + continue like a normal record
                        jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        jf.flush()
                        if k:
                            processed.add(k)
                            last_by_key[k] = rec

                        check_df.at[i, "screen_decision"] = rec["screen_decision"]
                        check_df.at[i, "screen_reasons"] = rec["screen_reasons"]
                        check_df.at[i, "screen_rule_hits"] = rec["screen_rule_hits"]
                        check_df.at[i, "screen_checked_at_utc"] = ""
                        _print_row_result(
                            i=i, n_run=n_run, k=k, title=title,
                            decision=rec["screen_decision"],
                            reasons=rec["screen_reasons"],
                            tag="MODEL_ERROR",
                        )

                        continue


                    decisions: List[str] = []
                    reasons_no: List[str] = []
                    reasons_unc: List[str] = []
                    per_crit: Dict[str, Any] = {}

                    for ck in crit_keys:
                        item = data.get(ck, {}) if isinstance(data, dict) else {}
                        decision_raw = safe_str(item.get("decision", "unclear")).lower().strip()
                        reason = safe_str(item.get("reason", ""))
                        quote = safe_str(item.get("quote", ""))

                        v = verify_quote(full_text, quote)

                        decision_norm = _classify_decision(decision_raw)

                        # Don't downgrade "yes" to "unclear" just because quote isn't exact.
                        # Instead, keep the decision but flag it for audit.
                        quote_problem = (decision_norm == "yes" and quote and not v["verified"])

                        decisions.append(decision_norm)

                        if decision_norm == "no":
                            reasons_no.append(f"{ck}: {reason}".strip())
                        elif decision_norm == "unclear":
                            reasons_unc.append(f"{ck}: {reason}".strip())

                        per_crit[ck] = {
                            "decision": decision_raw,
                            "decision_norm": decision_norm,
                            "reason": reason,
                            "quote": quote,
                            "quote_verified": bool(v["verified"]),
                            "quote_problem": bool(quote_problem),
                        }


                    if "no" in decisions:
                        final = "Exclude"
                        why = "; ".join([r for r in reasons_no if r]) or "Failed one or more criteria"
                    elif "unclear" in decisions:
                        # Minimize maybes: include on uncertainty at title/abstract stage
                        final = "Include"
                        why = "; ".join([r for r in reasons_unc if r]) or "Included despite some uncertainty"
                    else:
                        final = "Include"
                        why = "Meets all criteria"

                    rec["screen_decision"] = final
                    rec["screen_reasons"] = why
                    rec["screen_rule_hits"] = json.dumps({"raw": data, "per_criteria": per_crit}, ensure_ascii=False)

                except Exception as e:
                    rec["screen_decision"] = "MAYBE/UNCLEAR"
                    rec["screen_reasons"] = f"LLM parse/error: {type(e).__name__}"
                    rec["screen_rule_hits"] = json.dumps({"error": str(e), "raw_response": llm_resp[:2000]}, ensure_ascii=False)

                jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                jf.flush()
                if k:
                    processed.add(k)
                    last_by_key[k] = rec

                check_df.at[i, "screen_decision"] = rec["screen_decision"]
                check_df.at[i, "screen_reasons"] = rec["screen_reasons"]
                check_df.at[i, "screen_rule_hits"] = rec["screen_rule_hits"]
                check_df.at[i, "screen_checked_at_utc"] = ""

                _print_row_result(
                    i=i, n_run=n_run, k=k, title=title,
                    decision=rec["screen_decision"],
                    reasons=rec["screen_reasons"],
                    tag="LLM",
                )


    print("[step10] Screening done.")
    return check_df


# =============================================================================
# Write outputs
# =============================================================================

def _canonical_decision(x: Any) -> str:
    v = safe_str(x).strip().lower()
    if v in ("include",):
        return "Include"
    if v in ("exclude",):
        return "Exclude"
    if v in ("maybe/unclear", "maybe", "unclear"):
        return "MAYBE/UNCLEAR"
    if v == "":
        return "BLANK"
    return str(x)


def summarize_existing_outputs(*, out_csv: Path) -> Dict[str, Any]:
    if not out_csv.exists() or out_csv.stat().st_size == 0:
        return {"error": f"Missing output CSV for summary: {out_csv}"}

    df = pd.read_csv(out_csv, engine="python", on_bad_lines="skip")
    if df.empty:
        return {"error": f"Output CSV is empty: {out_csv}"}

    decisions = df["screen_decision"].fillna("").astype(str).apply(_canonical_decision)
    decision_counts = decisions.value_counts(dropna=False).to_dict()

    # Count excluded criteria by scanning "1_population:" etc in screen_reasons
    excluded_by_criterion = Counter()
    model_errors = 0
    parse_errors = 0

    reasons = df["screen_reasons"].fillna("").astype(str)
    for d, r in zip(decisions.tolist(), reasons.tolist()):
        r = r.strip()
        if "MODEL_ERROR:" in r:
            model_errors += 1
        if r.startswith("LLM parse/error:"):
            parse_errors += 1
        if d == "Exclude":
            for m in re.finditer(r"\b([1-5]_[a-zA-Z0-9]+)\s*:", r):
                excluded_by_criterion[m.group(1)] += 1

    # Missing abstracts (based on screen_reasons == "Missing abstract")
    missing_mask = reasons.str.strip().eq("Missing abstract")
    missing_count = int(missing_mask.sum())

    missing_list = []
    if missing_count:
        sub = df.loc[missing_mask, ["_check_row_id", "ID", "Title", "DOI"]].copy()
        for _, row in sub.iterrows():
            missing_list.append(
                {
                    "row_id": int(row.get("_check_row_id")) if str(row.get("_check_row_id", "")).isdigit() else safe_str(row.get("_check_row_id")),
                    "id": safe_str(row.get("ID")),
                    "title": safe_str(row.get("Title")),
                    "doi": safe_str(row.get("DOI")),
                }
            )

    return {
        "rows": int(len(df)),
        "decision_counts": {k: int(v) for k, v in decision_counts.items()},
        "excluded_by_criterion": dict(sorted(excluded_by_criterion.items(), key=lambda kv: (-kv[1], kv[0]))),
        "model_error_count": int(model_errors),
        "parse_error_count": int(parse_errors),
        "missing_abstract_count": int(missing_count),
        "missing_abstract_list": missing_list,
    }
def write_outputs(
    out_root: Path,
    check_df: pd.DataFrame,
    *,
    check_ris: Path,
    step9_csv: Path,
    elapsed_seconds: Optional[float] = None,
) -> Dict[str, object]:
    out_csv, out_meta = check_paths(out_root)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Write CSV (current behavior)
    check_df.to_csv(out_csv, index=False)

    missing_csv = out_csv.parent / "step10_missing_abstracts.csv"
    missing_mask = check_df.get("screen_reasons", "").fillna("").astype(str).str.strip().eq("Missing abstract")
    check_df.loc[missing_mask].to_csv(missing_csv, index=False)

    # Summary from the written CSV + JSONL cache
    jsonl_path = check_jsonl_path(out_root)
    summary = summarize_existing_outputs(out_csv=out_csv)

    meta: Dict[str, object] = {
        "input_step9_csv": str(step9_csv),
        "input_check_ris": str(check_ris),
        "output_csv": str(out_csv),
        "output_meta_json": str(out_meta),
        "missing_abstracts_csv": str(missing_csv),
        "rows_output": int(len(check_df)),
        "timestamp_utc": _now_utc(),
        "elapsed_seconds": float(elapsed_seconds) if elapsed_seconds is not None else None,
        "elapsed_hms": (
            time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds))
            if elapsed_seconds is not None
            else None
        ),
        "summary": summary,
        "notes": "Step10 matches against Step9 enriched CSV and uses abstract/xref_abstract as best abstract. Meta includes post-run summary from CSV + JSONL cache.",
    }

    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[step10] Wrote: {out_csv}")
    print(f"[step10] Wrote: {out_meta}")

    # Optional: show a tiny terminal summary too
    if isinstance(summary, dict) and "decision_counts" in summary:
        dc = summary["decision_counts"]
        print(f"[step10] Summary decisions: {dc}")

    return meta

# =============================================================================
# Entrypoints
# =============================================================================

def run(config: dict) -> dict:
    t_start = time.time()

    out_root = Path(safe_str((config or {}).get("out_dir", "")) or "outputs")
    model = safe_str((config or {}).get("ollama_model", "")) or DEFAULT_MODEL

    step9_csv = step9_path(out_root)
    step9 = load_step9_enriched(step9_csv)
    check = load_check_data_ris(CHECK_RIS)

    check_df = build_check_table_from_step9(step9=step9, check=check)
    check_df = apply_ollama_screening(check_df, out_root=out_root, model=model, run_limit=RUN_LIMIT)

    elapsed = time.time() - t_start
    return write_outputs(
        out_root=out_root,
        check_df=check_df,
        check_ris=CHECK_RIS,
        step9_csv=step9_csv,
        elapsed_seconds=elapsed,
    )

def run_check(config: dict) -> dict:
    return run(config)


def main() -> int:
    t_start = time.time()  # <-- ADD THIS

    out_root = default_outputs_root()
    model = DEFAULT_MODEL

    step9_csv = step9_path(out_root)
    step9 = load_step9_enriched(step9_csv)
    check = load_check_data_ris(CHECK_RIS)

    check_df = build_check_table_from_step9(step9=step9, check=check)
    check_df = apply_ollama_screening(check_df, out_root=out_root, model=model, run_limit=RUN_LIMIT)

    elapsed = time.time() - t_start  # <-- ADD THIS

    meta = write_outputs(
        out_root=out_root,
        check_df=check_df,
        check_ris=CHECK_RIS,
        step9_csv=step9_csv,
        elapsed_seconds=elapsed,  # <-- ADD THIS
    )
    return 0 if meta else 1

if __name__ == "__main__":
    raise SystemExit(main())
