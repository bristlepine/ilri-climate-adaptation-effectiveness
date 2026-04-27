#!/usr/bin/env python3
"""
step14_screen_fulltext.py

Step 14: Full-text screening of records included in step 12.

For each record that passed abstract screening, reads the downloaded full text
(from step 13) and applies the PCCM eligibility criteria for a second, more
stringent screen using the actual paper content.

Decision logic (same binary as step 12 — no MAYBE):
  - Any criterion "no"              -> Exclude
  - All "yes" or any "unclear"      -> Include
  - No full text available          -> Skipped (saved to step14_no_fulltext.csv; rerun after manual retrieval)
  - PDF extraction fails            -> Needs_Manual (flagged for manual review)

Text extraction:
  - PDF  : pypdf (fast, no native deps)
  - HTML : trafilatura if available, else BeautifulSoup, else raw read
  - Text is truncated to FULLTEXT_CHAR_LIMIT chars (intro + methods = most informative)

Inputs:
  - outputs/step12/step12_results.csv      (abstract screening decisions)
  - outputs/step13/step13_manifest.csv     (file paths per included record)
  - scripts/criteria.yml

Outputs (under outputs/step14/):
  - step14_results.csv
  - step14_results.meta.json
  - step14_results_details.jsonl    (JSONL cache; resume-safe)
  - step14_no_fulltext.csv          (records skipped — no full text retrieved; rerun after manual retrieval)
  - step14_extraction_failed.csv    (records where file path existed but text extraction failed)
  - step14_summary.png              (summary figure)

Run:
  python step14_screen_fulltext.py
  (or via run.py with run_step14 = 1)
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

OLLAMA_URL      = "http://localhost:11434/api/generate"
DEFAULT_MODEL   = "qwen2.5:14b"
TEMPERATURE     = 0.0
OLLAMA_KEEP_ALIVE = "30m"
OLLAMA_THINK    = False

FULLTEXT_CHAR_LIMIT = 12_000   # ~3k tokens; covers abstract + intro + methods
PRINT_EVERY     = 50

RUN_LIMIT: Optional[int] = None


# =============================================================================
# Shared helpers (mirrored from step12)
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
    year = safe_str(row.get("year", ""))
    if title and year:
        clean = re.sub(r"[^a-z0-9]+", "", title.lower())
        return f"ty:{clean}:{year}"
    if title:
        clean = re.sub(r"[^a-z0-9]+", "", title.lower())
        return f"t:{clean}"
    blob = f"{i}|{title}|{doi}".encode("utf-8", errors="ignore")
    return "row:" + hashlib.sha1(blob).hexdigest()[:16]


# =============================================================================
# IO paths
# =============================================================================

def _step12_csv(out_root: Path) -> Path:
    p = out_root / "step12" / "step12_results.csv"
    if not p.exists():
        raise SystemExit(f"Step 12 results not found: {p}")
    return p


def _step13_manifest(out_root: Path) -> Path:
    p = out_root / "step13" / "step13_manifest.csv"
    if not p.exists():
        raise SystemExit(f"Step 13 manifest not found: {p}")
    return p


def _out_dir(out_root: Path) -> Path:
    d = out_root / "step14"
    d.mkdir(parents=True, exist_ok=True)
    return d


# =============================================================================
# Full-text extraction
# =============================================================================

def _extract_pdf(path: Path) -> Tuple[str, str]:
    """Return (text, note). note is empty on success."""
    try:
        import pypdf  # type: ignore
    except ImportError:
        return "", "pypdf not installed — pip install pypdf"

    try:
        reader = pypdf.PdfReader(str(path))
        parts: List[str] = []
        for page in reader.pages:
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                continue
        text = "\n".join(parts)
        if not text.strip():
            return "", "PDF extracted but no text (possibly scanned/image-only)"
        return text, ""
    except Exception as e:
        return "", f"pypdf error: {type(e).__name__}: {e}"


def _extract_html(path: Path) -> Tuple[str, str]:
    """Return (text, note)."""
    raw = path.read_text(encoding="utf-8", errors="ignore")

    # Try trafilatura first (best quality)
    try:
        import trafilatura  # type: ignore
        text = trafilatura.extract(raw) or ""
        if text.strip():
            return text, ""
    except ImportError:
        pass

    # Fallback: BeautifulSoup
    try:
        from bs4 import BeautifulSoup  # type: ignore
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator=" ")
        text = " ".join(text.split())
        if text.strip():
            return text, ""
    except ImportError:
        pass

    # Last resort: strip HTML tags with regex
    text = re.sub(r"<[^>]+>", " ", raw)
    text = " ".join(text.split())
    return text, "html-regex-stripped"


def extract_fulltext(file_path: str) -> Tuple[str, str]:
    """
    Extract readable text from a downloaded file.
    Returns (text, note) — note is empty on clean success.
    """
    if not file_path:
        return "", "no_file_path"

    p = Path(file_path)
    if not p.exists():
        return "", f"file_not_found: {p}"

    suffix = p.suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf(p)
    elif suffix in (".html", ".htm"):
        return _extract_html(p)
    elif suffix in (".xml",):
        # XML full texts (Elsevier) — strip tags
        raw = p.read_text(encoding="utf-8", errors="ignore")
        text = re.sub(r"<[^>]+>", " ", raw)
        text = " ".join(text.split())
        return text, "xml-tag-stripped"
    elif suffix == ".txt":
        text = p.read_text(encoding="utf-8", errors="ignore")
        return text, ""
    else:
        # Try reading as text
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
            return text, f"unknown-ext:{suffix}"
        except Exception as e:
            return "", f"read_error: {e}"


def truncate_text(text: str, limit: int = FULLTEXT_CHAR_LIMIT) -> str:
    """Truncate to limit chars, keeping from the start of the document."""
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n[... truncated ...]"


# =============================================================================
# YAML / criteria (mirrored from step12)
# =============================================================================

def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing criteria file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _build_criteria_prompt(criteria_dict: dict) -> str:
    _guidance_pat = re.compile(r'^r\d+[a-z]*_(include|exclude)_further_guidelines$')
    lines: List[str] = []
    for key in sorted(criteria_dict.keys()):
        val = criteria_dict[key] or {}
        lines.append(f"{val.get('name', key)}:")
        lines.append(f"   - INCLUDE: {val.get('include', '')}")
        for fk in sorted(k for k in val if _guidance_pat.match(k) and "_include_" in k):
            if extra := (val[fk] or "").strip():
                lines.append(f"     (further guidance: {extra})")
        lines.append(f"   - EXCLUDE: {val.get('exclude', '')}")
        for fk in sorted(k for k in val if _guidance_pat.match(k) and "_exclude_" in k):
            if extra := (val[fk] or "").strip():
                lines.append(f"     (further guidance: {extra})")
        lines.append("")
    return "\n".join(lines).strip()


def _run_signature(*, model: str, criteria_text: str, step: str = "14") -> str:
    blob = f"step={step}\nmodel={model}\ncriteria=\n{criteria_text}\n"
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


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
                f"Ollama model '{model}' not installed.\nFix: ollama pull {model}"
            )
    except SystemExit:
        raise
    except Exception:
        pass


def call_ollama(
    session: requests.Session,
    *,
    title: str,
    fulltext: str,
    criteria_prompt: str,
    model: str,
) -> str:
    system_prompt = f"""You are a strict research assistant screening papers for a systematic review.
Analyze the Title and Full Text against these eligibility criteria:
{criteria_prompt}

DECISION RULES:
- Return "no" when you can clearly identify that a criterion is not met.
- Return "unclear" ONLY when there is genuine ambiguity that cannot be resolved from the full text.
- Do NOT return "unclear" when you have identified a clear exclusion signal. Commit to "no".
- A single "no" on any criterion results in EXCLUDE.

OUTPUT FORMAT:
Return ONLY a valid JSON object. Do not use markdown blocks.
Structure:
{{
  "1_population": {{ "decision": "yes|no|unclear", "reason": "brief reason", "quote": "exact substring from text" }},
  "2_concept": {{ "decision": "yes|no|unclear", "reason": "brief reason", "quote": "exact substring from text" }},
  "3_context": {{ "decision": "yes|no|unclear", "reason": "brief reason", "quote": "exact substring from text" }},
  "4_methodology": {{ "decision": "yes|no|unclear", "reason": "brief reason", "quote": "exact substring from text" }},
  "5_geography": {{ "decision": "yes|no|unclear", "reason": "brief reason", "quote": "exact substring from text" }}
}}""".strip()

    payload = {
        "model": model,
        "system": system_prompt,
        "prompt": f"TITLE: {title}\n\nFULL TEXT:\n{fulltext}",
        "stream": False,
        "format": "json",
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {"temperature": TEMPERATURE},
        "think": OLLAMA_THINK,
    }

    try:
        r = session.post(OLLAMA_URL, json=payload, timeout=300)
        if r.status_code == 200:
            return r.json().get("response", "")
        return json.dumps({"error": f"HTTP {r.status_code}", "body": (r.text or "")[:500]})
    except Exception as e:
        return json.dumps({"error": str(e)})


# =============================================================================
# Decision parsing (mirrored from step12)
# =============================================================================

def _classify_decision(val: Any) -> str:
    v = str(val or "").strip().lower()
    if v == "yes" or v.startswith("include"):
        return "yes"
    if v == "no" or v.startswith("exclude"):
        return "no"
    return "unclear"


def parse_llm_response(llm_resp: str, *, crit_keys: List[str]) -> Dict[str, Any]:
    try:
        try:
            data = json.loads(llm_resp)
        except json.JSONDecodeError:
            if _HAS_JSON_REPAIR:
                repaired = _repair_json(llm_resp, return_objects=True)
                if not isinstance(repaired, dict):
                    raise ValueError(f"json_repair returned {type(repaired).__name__}")
                data = repaired
            else:
                raise

        if isinstance(data, dict) and "error" in data and not any(k in data for k in crit_keys):
            return {
                "screen_decision": "Include",
                "screen_reasons": f"MODEL_ERROR: {safe_str(data.get('error'))}",
                "screen_rule_hits": json.dumps({"raw": data}, ensure_ascii=False),
                "parse_ok": False,
            }

        decisions: List[str] = []
        reasons_no: List[str] = []
        reasons_unc: List[str] = []
        per_crit: Dict[str, Any] = {}

        for ck in crit_keys:
            item = data.get(ck, {}) if isinstance(data, dict) else {}
            decision_raw = safe_str(item.get("decision", "unclear")).lower().strip()
            reason = safe_str(item.get("reason", ""))
            quote = safe_str(item.get("quote", ""))
            decision_norm = _classify_decision(decision_raw)
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
            }

        if "no" in decisions:
            final = "Exclude"
            why = "; ".join(r for r in reasons_no if r) or "Failed one or more criteria"
        else:
            final = "Include"
            why = "; ".join(r for r in reasons_unc if r) if reasons_unc else "Meets all criteria"

        return {
            "screen_decision": final,
            "screen_reasons": why,
            "screen_rule_hits": json.dumps({"raw": data, "per_criteria": per_crit}, ensure_ascii=False),
            "parse_ok": True,
        }

    except Exception as e:
        return {
            "screen_decision": "Include",
            "screen_reasons": f"LLM parse/error: {type(e).__name__}",
            "screen_rule_hits": json.dumps({"error": str(e), "raw_response": llm_resp[:2000]}, ensure_ascii=False),
            "parse_ok": False,
        }


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
                k = safe_str(j.get("key"))
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
# Load inputs
# =============================================================================

def load_inputs(out_root: Path) -> pd.DataFrame:
    """
    Load step12 + step12b Include records and join with step13 manifest to get file paths.
    Returns a DataFrame with one row per included record, with file_path column.
    """
    step12 = pd.read_csv(_step12_csv(out_root), engine="python", on_bad_lines="skip")
    step13 = pd.read_csv(_step13_manifest(out_root), engine="python", on_bad_lines="skip")

    # Merge step12b includes if available
    step12b_csv = out_root / "step12b" / "step12b_results.csv"
    if step12b_csv.exists() and step12b_csv.stat().st_size > 0:
        step12b = pd.read_csv(step12b_csv, engine="python", on_bad_lines="skip")
        # Align column name: step12b uses s12b_decision, step12 uses screen_decision
        if "s12b_decision" in step12b.columns and "screen_decision" not in step12b.columns:
            step12b = step12b.rename(columns={"s12b_decision": "screen_decision"})
        step12 = pd.concat([step12, step12b], ignore_index=True)
        print(f"[step14] Loaded step12 + step12b: {len(step12):,} records combined")

    # Filter to Include only
    inc = step12[step12["screen_decision"].str.strip().str.lower() == "include"].copy()
    print(f"[step14] Total included records: {len(inc):,}")

    # Normalise join keys
    for df in [inc, step13]:
        for col in ["dedupe_key", "doi"]:
            if col in df.columns:
                df[col] = df[col].apply(safe_str)

    # Join on dedupe_key first, fall back to doi
    s13_cols = ["dedupe_key", "doi", "file_path", "status", "source", "missing_abstract"]
    s13_cols = [c for c in s13_cols if c in step13.columns]
    s13 = step13[s13_cols].copy()
    s13 = s13.rename(columns={
        "file_path": "s13_file_path",
        "status": "s13_status",
        "source": "s13_source",
    })

    # Merge on dedupe_key
    merged = inc.merge(
        s13.add_suffix("_s13").rename(columns=lambda c: c.replace("_s13_s13", "_s13")),
        left_on="dedupe_key", right_on="dedupe_key_s13", how="left"
    ) if "dedupe_key" in s13.columns else inc.copy()

    # Fill file_path from merge result
    if "s13_file_path_s13" in merged.columns:
        merged = merged.rename(columns={"s13_file_path_s13": "ft_file_path"})
    elif "s13_file_path" in merged.columns:
        merged = merged.rename(columns={"s13_file_path": "ft_file_path"})
    else:
        merged["ft_file_path"] = ""

    if "ft_file_path" not in merged.columns:
        merged["ft_file_path"] = ""

    merged["ft_file_path"] = merged["ft_file_path"].apply(safe_str)

    # Validate that recorded file paths actually exist — step13 occasionally records
    # a corrupt path (e.g. doi_10.pdf) for papers that failed to download (HTTP 403 etc).
    # Clear any path that doesn't resolve to a real file so step14 treats them correctly.
    def _validate_path(fp):
        if not fp:
            return ""
        from pathlib import Path as _P
        return fp if _P(fp).exists() else ""
    merged["ft_file_path"] = merged["ft_file_path"].apply(_validate_path)

    n_with = (merged["ft_file_path"] != "").sum()
    n_without = (merged["ft_file_path"] == "").sum()
    print(f"[step14] Full texts available: {n_with:,}  |  Skipped (no full text): {n_without:,}")

    # Save skipped records for later — do not screen them now
    no_ft = merged[merged["ft_file_path"] == ""]
    no_ft_path = out_root / "step14" / "step14_no_fulltext.csv"
    no_ft_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in ["dedupe_key", "doi", "title", "year", "ft_file_path"] if c in no_ft.columns]
    no_ft[cols].to_csv(no_ft_path, index=False)
    print(f"[step14] Skipped list -> {no_ft_path}")

    # Only return records that have a full text to screen
    return merged[merged["ft_file_path"] != ""].reset_index(drop=True)


# =============================================================================
# Main screening loop
# =============================================================================

def screen_fulltext(
    df: pd.DataFrame,
    *,
    out_dir: Path,
    model: str,
    criteria_yml_path: Path,
    run_limit: Optional[int] = None,
) -> pd.DataFrame:
    cfg = _load_yaml(criteria_yml_path)
    criteria_text = _build_criteria_prompt(cfg.get("criteria", {}) or {})
    crit_keys = sorted((cfg.get("criteria", {}) or {}).keys())
    sig = _run_signature(model=model, criteria_text=criteria_text)

    _ollama_fail_fast(model)

    jsonl_path = out_dir / "step14_results_details.jsonl"
    cache = _load_jsonl_cache(jsonl_path)
    if cache:
        _rewrite_jsonl(jsonl_path, cache)
    processed = set(cache.keys())
    print(f"[step14] Cache warm: {len(processed):,} records from {jsonl_path}")

    df = df.copy()
    df["s14_decision"] = ""
    df["s14_reasons"] = ""
    df["s14_rule_hits"] = ""
    df["s14_fulltext_chars"] = 0
    df["s14_fulltext_note"] = ""
    df["s14_checked_at_utc"] = ""

    n_total = len(df)
    n_run = min(int(run_limit), n_total) if run_limit else n_total

    t0 = time.time()
    n_include = n_exclude = n_no_text = n_cached = n_error = 0

    print(f"[step14] Start full-text screening | total={n_total:,} | running={n_run:,} | model={model}")

    with requests.Session() as session:
        with open(jsonl_path, "a", encoding="utf-8", buffering=1) as jf:
            for i in range(n_run):
                row = df.iloc[i]
                k = stable_key(row, i)
                title = safe_str(row.get("title", ""))
                file_path = safe_str(row.get("ft_file_path", ""))

                if i == 0 or (i + 1) % PRINT_EVERY == 0 or (i + 1) == n_run:
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    eta = (n_run - i - 1) / rate if rate > 0 else 0
                    print(
                        f"[step14] {i+1:,}/{n_run:,} | "
                        f"include={n_include:,} exclude={n_exclude:,} "
                        f"no_text={n_no_text:,} cached={n_cached:,} err={n_error:,} | "
                        f"elapsed={elapsed:,.0f}s ETA={eta:,.0f}s"
                    )

                rec: Dict[str, Any] = {
                    "run_signature": sig,
                    "timestamp_utc": _now_utc(),
                    "key": k,
                    "title": title,
                    "file_path": file_path,
                }

                # Safety net: no file path (should not reach here — filtered in load_inputs)
                if not file_path:
                    rec["s14_decision"] = "Needs_Manual"
                    rec["s14_reasons"] = "No full text retrieved — skipped"
                    rec["s14_rule_hits"] = ""
                    rec["s14_fulltext_chars"] = 0
                    rec["s14_fulltext_note"] = "no_file"
                    n_no_text += 1

                # Cache hit — only valid if signature matches
                elif k in processed and safe_str(cache[k].get("run_signature")) == sig:
                    cached = cache[k]
                    rec["s14_decision"] = safe_str(cached.get("s14_decision"))
                    rec["s14_reasons"] = safe_str(cached.get("s14_reasons"))
                    rec["s14_rule_hits"] = safe_str(cached.get("s14_rule_hits", ""))
                    rec["s14_fulltext_chars"] = cached.get("s14_fulltext_chars", 0)
                    rec["s14_fulltext_note"] = safe_str(cached.get("s14_fulltext_note", ""))
                    n_cached += 1
                    if rec["s14_decision"] == "Include":
                        n_include += 1
                    elif rec["s14_decision"] == "Exclude":
                        n_exclude += 1
                    else:
                        n_no_text += 1

                    df.at[i, "s14_decision"] = rec["s14_decision"]
                    df.at[i, "s14_reasons"] = rec["s14_reasons"]
                    df.at[i, "s14_rule_hits"] = rec["s14_rule_hits"]
                    df.at[i, "s14_fulltext_chars"] = rec["s14_fulltext_chars"]
                    df.at[i, "s14_fulltext_note"] = rec["s14_fulltext_note"]
                    df.at[i, "s14_checked_at_utc"] = safe_str(cached.get("timestamp_utc", ""))
                    continue

                else:
                    # Extract text from file
                    fulltext_raw, extract_note = extract_fulltext(file_path)
                    fulltext = truncate_text(fulltext_raw)

                    rec["s14_fulltext_chars"] = len(fulltext_raw)
                    rec["s14_fulltext_note"] = extract_note

                    if not fulltext.strip():
                        # Extraction failed -> Needs_Manual (cannot screen without text)
                        rec["s14_decision"] = "Needs_Manual"
                        rec["s14_reasons"] = f"Full text extraction failed — manual screening required ({extract_note or 'no text'})"
                        rec["s14_rule_hits"] = ""
                        n_no_text += 1
                    else:
                        print(f"[step14]  -> LLM {i+1}/{n_run} | key={k[:50]} | chars={len(fulltext_raw):,}")
                        llm_resp = call_ollama(
                            session,
                            title=title,
                            fulltext=fulltext,
                            criteria_prompt=criteria_text,
                            model=model,
                        )
                        result = parse_llm_response(llm_resp, crit_keys=crit_keys)
                        rec["s14_decision"] = result["screen_decision"]
                        rec["s14_reasons"] = result["screen_reasons"]
                        rec["s14_rule_hits"] = result["screen_rule_hits"]
                        if not result["parse_ok"]:
                            n_error += 1
                        if result["screen_decision"] == "Include":
                            n_include += 1
                        else:
                            n_exclude += 1

                df.at[i, "s14_decision"] = rec["s14_decision"]
                df.at[i, "s14_reasons"] = rec["s14_reasons"]
                df.at[i, "s14_rule_hits"] = rec.get("s14_rule_hits", "")
                df.at[i, "s14_fulltext_chars"] = rec.get("s14_fulltext_chars", 0)
                df.at[i, "s14_fulltext_note"] = rec.get("s14_fulltext_note", "")
                df.at[i, "s14_checked_at_utc"] = rec["timestamp_utc"]

                jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                jf.flush()
                processed.add(k)
                cache[k] = rec

    elapsed_total = time.time() - t0
    print(
        f"[step14] Done | include={n_include:,} exclude={n_exclude:,} "
        f"no_text={n_no_text:,} errors={n_error:,} | elapsed={elapsed_total:,.1f}s"
    )
    return df


# =============================================================================
# Write outputs
# =============================================================================

def write_outputs(
    df: pd.DataFrame,
    *,
    out_dir: Path,
    out_root: Path,
    criteria_yml: Path,
    elapsed_seconds: Optional[float],
) -> Dict[str, Any]:
    out_csv   = out_dir / "step14_results.csv"
    out_meta  = out_dir / "step14_results.meta.json"
    no_ft_csv = out_dir / "step14_extraction_failed.csv"

    decision_cols = ["s14_decision", "s14_reasons", "s14_fulltext_chars", "s14_fulltext_note"]
    other_cols = [c for c in df.columns if c not in decision_cols and not c.startswith("s14_rule_hits") and not c.startswith("_")]
    rule_cols = [c for c in df.columns if c == "s14_rule_hits"]
    col_order = [c for c in decision_cols + other_cols + rule_cols if c in df.columns]
    df[col_order].to_csv(out_csv, index=False)

    # Extraction-failed sidecar (records that had a file path but text extraction failed)
    no_ft = df[df["s14_fulltext_note"].str.contains("extraction failed", na=False, case=False)]
    cols = [c for c in ["dedupe_key", "doi", "title", "year", "s14_decision", "s14_reasons", "ft_file_path"] if c in df.columns]
    no_ft[cols].to_csv(no_ft_csv, index=False)

    # Counts
    decisions = df["s14_decision"].fillna("").str.strip()
    counts = decisions.value_counts(dropna=False).to_dict()

    from collections import Counter
    excl_by_crit: Counter = Counter()
    for d, r in zip(decisions, df["s14_reasons"].fillna("").astype(str)):
        if d == "Exclude":
            for m in re.finditer(r"\b([1-5]_[a-zA-Z0-9]+)\s*:", r):
                excl_by_crit[m.group(1)] += 1

    n_screened_with_text = int((df["s14_fulltext_chars"] > 0).sum())
    n_extraction_failed = int((df["s14_fulltext_note"].str.contains("extraction failed", na=False, case=False)).sum())

    meta: Dict[str, Any] = {
        "step12_csv": str(_step12_csv(out_root)),
        "step13_manifest": str(_step13_manifest(out_root)),
        "criteria_yml": str(criteria_yml),
        "output_csv": str(out_csv),
        "extraction_failed_csv": str(no_ft_csv),
        "rows_total": int(len(df)),
        "rows_screened_with_fulltext": n_screened_with_text,
        "rows_extraction_failed": n_extraction_failed,
        "decision_counts": {k: int(v) for k, v in counts.items()},
        "excluded_by_criterion": dict(sorted(excl_by_crit.items(), key=lambda kv: (-kv[1], kv[0]))),
        "timestamp_utc": _now_utc(),
        "elapsed_seconds": float(elapsed_seconds) if elapsed_seconds else None,
        "elapsed_hms": time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds)) if elapsed_seconds else None,
    }

    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[step14] Wrote: {out_csv}")
    print(f"[step14] Wrote: {out_meta}")
    print(f"[step14] Decision counts: {counts}")

    _plot_summary(meta, out_dir)
    return meta


# =============================================================================
# Summary figure
# =============================================================================

def _plot_summary(meta: Dict[str, Any], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[step14] matplotlib not installed — skipping summary figure")
        return

    CRIT_LABELS = {
        "1_population":  "1. Population",
        "2_concept":     "2. Concept",
        "3_context":     "3. Context",
        "4_methodology": "4. Methodology",
        "5_geography":   "5. Geography",
    }
    BLUE = "#2196F3"; RED = "#F44336"; ORANGE = "#FF9800"

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("Step 14 — Full-text Screening Summary", fontsize=14, fontweight="bold", y=1.01)

    ax1 = axes[0]
    dec     = meta.get("decision_counts", {})
    n_inc   = dec.get("Include", 0)
    n_exc   = dec.get("Exclude", 0)
    n_man   = dec.get("Needs_Manual", 0)
    total   = meta.get("rows_total", n_inc + n_exc + n_man)

    labels = ["Include\n(screened)", "Needs Manual\n(extraction failed)", "Exclude"]
    values = [n_inc, n_man, n_exc]
    colors = [BLUE, ORANGE, RED]
    bars = ax1.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8, width=0.5)
    for bar, val in zip(bars, values):
        pct = val / total * 100 if total else 0
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.005,
                 f"{val:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10)
    ax1.set_title(f"Full-text Screening Decisions  (n={total:,})", fontsize=11)
    ax1.set_ylabel("Records")
    ax1.set_ylim(0, max(values) * 1.18 if values else 1)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.text(0.98, 0.98, "Needs_Manual = text extraction failed;\nmanual review required",
             transform=ax1.transAxes, fontsize=8, color=ORANGE, ha="right", va="top", style="italic")

    ax2 = axes[1]
    excl_raw = meta.get("excluded_by_criterion", {})
    if excl_raw:
        excl_sorted = sorted(excl_raw.items(), key=lambda kv: -kv[1])
        crit_keys = [k for k, _ in excl_sorted]
        crit_vals = [v for _, v in excl_sorted]
        crit_names = [CRIT_LABELS.get(k, k) for k in crit_keys]
        h_bars = ax2.barh(range(len(crit_names)), crit_vals, color=RED, alpha=0.80, edgecolor="white")
        ax2.set_yticks(range(len(crit_names)))
        ax2.set_yticklabels(crit_names, fontsize=10)
        ax2.invert_yaxis()
        for bar, val in zip(h_bars, crit_vals):
            pct = val / n_exc * 100 if n_exc else 0
            ax2.text(bar.get_width() + max(crit_vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{val:,}  ({pct:.1f}% of excluded)", va="center", fontsize=9)
        ax2.set_xlim(0, max(crit_vals) * 1.38)
    else:
        ax2.text(0.5, 0.5, "No exclusions yet", ha="center", va="center", transform=ax2.transAxes)

    ax2.set_title("Excluded Records by Criterion\n(note: one record may fail multiple criteria)", fontsize=11)
    ax2.set_xlabel("Records failing criterion")
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out_fig = out_dir / "step14_summary.png"
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[step14] Wrote: {out_fig}")


# =============================================================================
# Entrypoint
# =============================================================================

def run(config: dict) -> dict:
    t_start = time.time()
    c = config or {}

    out_root = Path(safe_str(c.get("out_dir", "")) or "outputs")
    model = safe_str(c.get("step14_model", "")) or safe_str(c.get("ollama_model", "")) or DEFAULT_MODEL
    run_limit = c.get("step14_run_limit") or RUN_LIMIT

    here = Path(__file__).resolve().parent
    crit_str = safe_str(c.get("step14_criteria_yml", ""))
    criteria_yml = Path(crit_str) if crit_str else (here / "criteria.yml")

    out_dir = _out_dir(out_root)

    print(f"[step14] Model       : {model}")
    print(f"[step14] Criteria    : {criteria_yml}")
    print(f"[step14] Output dir  : {out_dir}")

    df = load_inputs(out_root)
    df = screen_fulltext(df, out_dir=out_dir, model=model, criteria_yml_path=criteria_yml, run_limit=run_limit)

    return write_outputs(
        df,
        out_dir=out_dir,
        out_root=out_root,
        criteria_yml=criteria_yml,
        elapsed_seconds=time.time() - t_start,
    )


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    try:
        import config as _cfg
        _run_cfg = {k: getattr(_cfg, k) for k in dir(_cfg) if not k.startswith("_")}
    except ImportError:
        _run_cfg = {}
    _run_cfg.setdefault("out_dir", str(here / "outputs"))
    run(_run_cfg)
