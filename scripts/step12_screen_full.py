#!/usr/bin/env python3
"""
step12_screen_full.py

Step 12: Full-corpus LLM screening using criteria.yml.

Reads the step9a enriched CSV (full list) and applies Ollama-based
eligibility screening to every record that has an abstract.

Decision logic (no MAYBE):
  - Any criterion returns "no"    -> Exclude
  - All criteria are "yes"/"unclear" -> Include
  - Missing abstract               -> Include (conservative; flagged in reasons)

Inputs:
  - outputs/step9a/step9a_scopus_enriched.csv
  - scripts/criteria.yml (or config-supplied path)

Outputs (under outputs/step12/):
  - step12_results.csv
  - step12_results.meta.json
  - step12_results_details.jsonl   (JSONL cache; resume-safe)
  - step12_missing_abstracts.csv

Run:
  python step12_screen_full.py
  (or via run.py with run_step12 = 1)
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

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:14b"
TEMPERATURE = 0.0
OLLAMA_KEEP_ALIVE = "30m"
OLLAMA_THINK = False

PRINT_EVERY = 100
PRINT_BEFORE_OLLAMA = True

# If set, only process first N rows (for testing)
RUN_LIMIT: Optional[int] = None


# =============================================================================
# Helpers (shared with step10)
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


def clean_id_digits(x: Any) -> str:
    s = safe_str(x)
    if not s:
        return ""
    try:
        if re.search(r"e\+?\d+", s, flags=re.IGNORECASE):
            return str(int(float(s)))
    except Exception:
        pass
    return re.sub(r"\D+", "", s)


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# =============================================================================
# IO paths
# =============================================================================

def step9a_path(out_root: Path) -> Path:
    p = out_root / "step9a" / "step9a_scopus_enriched.csv"
    if p.exists() and p.stat().st_size > 0:
        return p
    # fallback to step9
    p2 = out_root / "step9" / "step9_scopus_enriched.csv"
    if p2.exists() and p2.stat().st_size > 0:
        print(f"[step12] WARNING: step9a CSV not found, falling back to step9: {p2}")
        return p2
    raise SystemExit(f"No enriched CSV found at {p} or {p2}")


def step12_out_dir(out_root: Path) -> Path:
    d = out_root / "step12"
    d.mkdir(parents=True, exist_ok=True)
    return d


# =============================================================================
# Load enriched CSV
# =============================================================================

def load_enriched(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    if df.empty:
        raise SystemExit(f"Enriched CSV is empty: {path}")

    for c in ["dedupe_key", "title", "abstract", "xref_abstract",
              "doi", "year", "coverDate", "eid", "scopus_id",
              "publicationName", "author_names", "abstract_source", "abstract_status"]:
        if c not in df.columns:
            df[c] = ""

    df["doi"] = df["doi"].apply(normalize_doi)
    df["year"] = df["year"].apply(year_from_any)
    df["title"] = df["title"].apply(safe_str)
    df["dedupe_key"] = df["dedupe_key"].apply(safe_str)

    # Best abstract: prefer abstract, then xref_abstract
    def _best_abs(row: pd.Series) -> Tuple[str, str]:
        for col in ["abstract", "xref_abstract"]:
            v = safe_str(row.get(col, ""))
            if v:
                return v, col
        return "", ""

    best_abs, best_src = zip(*[_best_abs(r) for _, r in df.iterrows()]) if len(df) else ([], [])
    df["_abstract_best"] = list(best_abs)
    df["_abstract_best_src"] = list(best_src)

    return df.reset_index(drop=True)


# =============================================================================
# Stable cache key
# =============================================================================

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
    # last resort: row hash
    blob = f"{i}|{title}|{doi}".encode("utf-8", errors="ignore")
    return "row:" + hashlib.sha1(blob).hexdigest()[:16]


# =============================================================================
# YAML / criteria
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


def _run_signature(*, model: str, criteria_text: str, min_year: int) -> str:
    blob = f"model={model}\nmin_year={min_year}\ncriteria=\n{criteria_text}\n"
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


# =============================================================================
# Ollama preflight
# =============================================================================

def _ollama_base_url() -> str:
    return OLLAMA_URL.split("/api/")[0]


def _ollama_fail_fast(model: str) -> None:
    base = _ollama_base_url()
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
                f"Ollama is running but model '{model}' is not installed.\n"
                f"Fix: ollama pull {model}"
            )
    except SystemExit:
        raise
    except Exception:
        pass  # if tags call fails, proceed and let the first real call surface the error


# =============================================================================
# Ollama call
# =============================================================================

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

DECISION RULES:
- Return "no" when you can clearly identify that a criterion is not met. If your reason states that something is "not explicitly assessed", "not explicitly measured", "does not analyze adaptation", or similar — that is a "no", not "unclear".
- Return "unclear" ONLY when there is genuine ambiguity that cannot be resolved from the abstract (e.g., the abstract is too short or vague to determine).
- Do NOT return "unclear" when you have identified a clear exclusion signal. Commit to "no" in those cases.
- A single "no" on any criterion results in EXCLUDE. Do not soften a clear "no" to "unclear" to avoid excluding.

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
        "prompt": f"TITLE: {title}\n\nABSTRACT: {abstract}",
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


# =============================================================================
# Decision parsing
# =============================================================================

def _classify_decision(val: Any) -> str:
    v = str(val or "").strip().lower()
    if v == "yes" or v.startswith("include"):
        return "yes"
    if v == "no" or v.startswith("exclude"):
        return "no"
    return "unclear"


def parse_llm_response(llm_resp: str, *, crit_keys: List[str], full_text: str) -> Dict[str, Any]:
    """Parse LLM JSON response. Returns dict with screen_decision, screen_reasons, per_criteria."""
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
            # all yes or any unclear -> Include (no MAYBE)
            final = "Include"
            if reasons_unc:
                why = "; ".join(r for r in reasons_unc if r) or "Included despite some uncertainty"
            else:
                why = "Meets all criteria"

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
# Main screening loop
# =============================================================================

def screen_full(
    df: pd.DataFrame,
    *,
    out_dir: Path,
    model: str,
    criteria_yml_path: Path,
    run_limit: Optional[int] = None,
) -> pd.DataFrame:
    cfg = _load_yaml(criteria_yml_path)
    min_year = int((cfg.get("hard_filters", {}) or {}).get("min_year", 2005))
    criteria_text = _build_criteria_prompt(cfg.get("criteria", {}) or {})
    crit_keys = sorted((cfg.get("criteria", {}) or {}).keys())
    sig = _run_signature(model=model, criteria_text=criteria_text, min_year=min_year)

    _ollama_fail_fast(model)

    jsonl_path = out_dir / "step12_results_details.jsonl"
    cache = _load_jsonl_cache(jsonl_path)
    if cache:
        _rewrite_jsonl(jsonl_path, cache)  # compact on start
    processed = set(cache.keys())
    print(f"[step12] Cache warm: {len(processed):,} records from {jsonl_path}")

    # Output columns
    df = df.copy()
    df["screen_decision"] = ""
    df["screen_reasons"] = ""
    df["screen_rule_hits"] = ""
    df["screen_checked_at_utc"] = ""

    n_total = len(df)
    n_run = min(int(run_limit), n_total) if run_limit else n_total

    t0 = time.time()
    n_include = n_exclude = n_missing = n_cached = n_error = 0

    print(f"[step12] Start screening | total={n_total:,} | running={n_run:,} | model={model}")

    with requests.Session() as session:
        with open(jsonl_path, "a", encoding="utf-8", buffering=1) as jf:
            for i in range(n_run):
                row = df.iloc[i]
                k = stable_key(row, i)
                title = safe_str(row.get("title", ""))
                abstract = safe_str(row.get("_abstract_best", ""))

                # Progress log
                if i == 0 or (i + 1) % PRINT_EVERY == 0 or (i + 1) == n_run:
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    eta = (n_run - i - 1) / rate if rate > 0 else 0
                    print(
                        f"[step12] {i+1:,}/{n_run:,} | "
                        f"include={n_include:,} exclude={n_exclude:,} missing={n_missing:,} "
                        f"cached={n_cached:,} err={n_error:,} | "
                        f"elapsed={elapsed:,.0f}s ETA={eta:,.0f}s"
                    )

                rec: Dict[str, Any] = {
                    "run_signature": sig,
                    "timestamp_utc": _now_utc(),
                    "key": k,
                    "title": title,
                    "min_year": min_year,
                    "criteria_keys": crit_keys,
                }

                # Missing abstract -> Include (conservative)
                if not abstract.strip():
                    rec["screen_decision"] = "Include"
                    rec["screen_reasons"] = "Missing abstract - included for manual review"
                    rec["screen_rule_hits"] = ""
                    rec["abstract_present"] = False
                    n_missing += 1
                    n_include += 1

                # Cache hit
                elif k in processed:
                    cached = cache[k]
                    is_parse_error = safe_str(cached.get("screen_reasons", "")).startswith("LLM parse/error:")
                    if safe_str(cached.get("run_signature")) == sig and not is_parse_error:
                        rec["screen_decision"] = safe_str(cached.get("screen_decision"))
                        rec["screen_reasons"] = safe_str(cached.get("screen_reasons"))
                        rec["screen_rule_hits"] = safe_str(cached.get("screen_rule_hits", ""))
                        rec["abstract_present"] = True
                        n_cached += 1
                        if rec["screen_decision"] == "Include":
                            n_include += 1
                        else:
                            n_exclude += 1

                        df.at[i, "screen_decision"] = rec["screen_decision"]
                        df.at[i, "screen_reasons"] = rec["screen_reasons"]
                        df.at[i, "screen_rule_hits"] = rec["screen_rule_hits"]
                        df.at[i, "screen_checked_at_utc"] = safe_str(cached.get("timestamp_utc", ""))
                        continue  # skip writing to JSONL again

                    # stale cache (sig changed or parse error) -> fall through to LLM
                    else:
                        pass

                else:
                    # LLM call
                    if PRINT_BEFORE_OLLAMA:
                        print(f"[step12]  -> LLM row {i+1}/{n_run} | key={k[:60]} | title='{title[:80]}'")

                    rec["abstract_present"] = True
                    llm_resp = call_ollama(
                        session,
                        title=title,
                        abstract=abstract,
                        criteria_prompt=criteria_text,
                        model=model,
                    )
                    result = parse_llm_response(llm_resp, crit_keys=crit_keys, full_text=f"{title}\n{abstract}")
                    rec["screen_decision"] = result["screen_decision"]
                    rec["screen_reasons"] = result["screen_reasons"]
                    rec["screen_rule_hits"] = result["screen_rule_hits"]
                    if not result["parse_ok"]:
                        n_error += 1
                    if rec["screen_decision"] == "Include":
                        n_include += 1
                    else:
                        n_exclude += 1

                df.at[i, "screen_decision"] = rec["screen_decision"]
                df.at[i, "screen_reasons"] = rec["screen_reasons"]
                df.at[i, "screen_rule_hits"] = rec.get("screen_rule_hits", "")
                df.at[i, "screen_checked_at_utc"] = rec["timestamp_utc"]

                jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                jf.flush()
                processed.add(k)
                cache[k] = rec

    elapsed_total = time.time() - t0
    print(
        f"[step12] Done | include={n_include:,} exclude={n_exclude:,} "
        f"missing={n_missing:,} errors={n_error:,} | elapsed={elapsed_total:,.1f}s"
    )
    return df


# =============================================================================
# Write outputs
# =============================================================================

def write_outputs(
    df: pd.DataFrame,
    *,
    out_dir: Path,
    input_csv: Path,
    criteria_yml: Path,
    elapsed_seconds: Optional[float],
) -> Dict[str, Any]:
    out_csv = out_dir / "step12_results.csv"
    out_meta = out_dir / "step12_results.meta.json"
    missing_csv = out_dir / "step12_missing_abstracts.csv"

    # Reorder: decision cols first, then rest
    decision_cols = ["screen_decision", "screen_reasons"]
    other_cols = [c for c in df.columns if c not in decision_cols and not c.startswith("screen_rule_hits") and not c.startswith("_")]
    rule_cols = [c for c in df.columns if c == "screen_rule_hits"]
    private_cols = [c for c in df.columns if c.startswith("_")]
    col_order = decision_cols + other_cols + rule_cols + private_cols
    col_order = [c for c in col_order if c in df.columns]
    df[col_order].to_csv(out_csv, index=False)

    # Missing abstracts sidecar
    missing_mask = df["screen_reasons"].str.startswith("Missing abstract", na=False)
    df.loc[missing_mask, col_order].to_csv(missing_csv, index=False)

    # Summary
    decisions = df["screen_decision"].fillna("").str.strip()
    counts = decisions.value_counts(dropna=False).to_dict()

    from collections import Counter
    excl_by_crit: Counter = Counter()
    for d, r in zip(decisions, df["screen_reasons"].fillna("").astype(str)):
        if d == "Exclude":
            for m in re.finditer(r"\b([1-5]_[a-zA-Z0-9]+)\s*:", r):
                excl_by_crit[m.group(1)] += 1

    meta: Dict[str, Any] = {
        "input_csv": str(input_csv),
        "criteria_yml": str(criteria_yml),
        "output_csv": str(out_csv),
        "missing_abstracts_csv": str(missing_csv),
        "rows_total": int(len(df)),
        "rows_screened": int((decisions != "").sum()),
        "timestamp_utc": _now_utc(),
        "elapsed_seconds": float(elapsed_seconds) if elapsed_seconds is not None else None,
        "elapsed_hms": time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds)) if elapsed_seconds else None,
        "decision_counts": {k: int(v) for k, v in counts.items()},
        "excluded_by_criterion": dict(sorted(excl_by_crit.items(), key=lambda kv: (-kv[1], kv[0]))),
        "missing_abstract_count": int(missing_mask.sum()),
    }

    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[step12] Wrote: {out_csv}")
    print(f"[step12] Wrote: {out_meta}")
    print(f"[step12] Decision counts: {counts}")

    _plot_summary(meta, out_dir)

    return meta


def _plot_summary(meta: Dict[str, Any], out_dir: Path) -> None:
    """Save a two-panel summary figure from the meta dict."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[step12] matplotlib not installed — skipping summary figure")
        return

    CRIT_LABELS = {
        "1_population":  "1. Population",
        "2_concept":     "2. Concept",
        "3_context":     "3. Context",
        "4_methodology": "4. Methodology",
        "5_geography":   "5. Geography",
    }
    BLUE   = "#2196F3"
    RED    = "#F44336"
    ORANGE = "#FF9800"
    GREY   = "#9E9E9E"

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("Step 12 — Abstract Screening Summary", fontsize=14, fontweight="bold", y=1.01)

    # ── Panel 1: Include (screened) / Missing abstract / Exclude ──────────────
    ax1 = axes[0]
    dec   = meta.get("decision_counts", {})
    n_inc_total = dec.get("Include", 0)
    n_exc = dec.get("Exclude", 0)
    n_mis = meta.get("missing_abstract_count", 0)
    n_inc = n_inc_total - n_mis          # screened-and-included only
    total = meta.get("rows_total", n_inc_total + n_exc)

    labels = ["Include\n(screened)", "Missing\nAbstract", "Exclude"]
    values = [n_inc, n_mis, n_exc]
    colors = [BLUE, ORANGE, RED]
    bars   = ax1.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8, width=0.5)

    for bar, val in zip(bars, values):
        pct = val / total * 100 if total else 0
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.005,
            f"{val:,}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=10,
        )

    ax1.set_title(f"Screening Decisions  (n={total:,})", fontsize=11)
    ax1.set_ylabel("Records")
    ax1.set_ylim(0, max(values) * 1.18)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.text(
        0.98, 0.98,
        f"Missing abstracts conservatively\nincluded for manual review",
        transform=ax1.transAxes, fontsize=8, color=ORANGE,
        ha="right", va="top", style="italic",
    )

    # ── Panel 2: Excluded-by-criterion breakdown ──────────────────────────────
    ax2 = axes[1]
    excl_raw = meta.get("excluded_by_criterion", {})
    # sort descending
    excl_sorted = sorted(excl_raw.items(), key=lambda kv: -kv[1])
    crit_keys  = [k for k, _ in excl_sorted]
    crit_vals  = [v for _, v in excl_sorted]
    crit_names = [CRIT_LABELS.get(k, k) for k in crit_keys]

    h_bars = ax2.barh(
        range(len(crit_names)), crit_vals,
        color=RED, alpha=0.80, edgecolor="white", linewidth=0.8,
    )
    ax2.set_yticks(range(len(crit_names)))
    ax2.set_yticklabels(crit_names, fontsize=10)
    ax2.invert_yaxis()

    for bar, val in zip(h_bars, crit_vals):
        pct = val / n_exc * 100 if n_exc else 0
        ax2.text(
            bar.get_width() + max(crit_vals) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,}  ({pct:.1f}% of excluded)",
            va="center", fontsize=9,
        )

    ax2.set_title(
        f"Excluded Records by Criterion\n(note: one record may fail multiple criteria)",
        fontsize=11,
    )
    ax2.set_xlabel("Records failing criterion")
    ax2.set_xlim(0, max(crit_vals) * 1.38)
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out_fig = out_dir / "step12_summary.png"
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[step12] Wrote: {out_fig}")


# =============================================================================
# Entrypoint
# =============================================================================

def run(config: dict) -> dict:
    t_start = time.time()
    c = config or {}

    out_root = Path(safe_str(c.get("out_dir", "")) or "outputs")
    model = safe_str(c.get("step12_model", "")) or safe_str(c.get("ollama_model", "")) or DEFAULT_MODEL
    run_limit = c.get("step12_run_limit") or RUN_LIMIT

    here = Path(__file__).resolve().parent
    crit_str = safe_str(c.get("step12_criteria_yml", "") or c.get("step10_criteria_yml", ""))
    criteria_yml = Path(crit_str) if crit_str else (here / "criteria.yml")

    input_csv = step9a_path(out_root)
    out_dir = step12_out_dir(out_root)

    print(f"[step12] Input CSV  : {input_csv}")
    print(f"[step12] Criteria   : {criteria_yml}")
    print(f"[step12] Output dir : {out_dir}")
    print(f"[step12] Model      : {model}")

    df = load_enriched(input_csv)
    df = screen_full(
        df,
        out_dir=out_dir,
        model=model,
        criteria_yml_path=criteria_yml,
        run_limit=run_limit,
    )

    return write_outputs(
        df,
        out_dir=out_dir,
        input_csv=input_csv,
        criteria_yml=criteria_yml,
        elapsed_seconds=time.time() - t_start,
    )


def main() -> int:
    try:
        import config as _cfg
        cfg_dict = {
            "out_dir": str(getattr(_cfg, "out_dir", "outputs")),
            "step12_criteria_yml": str(getattr(_cfg, "step12_criteria_yml",
                                       getattr(_cfg, "step10_criteria_yml", ""))),
            "step12_model": str(getattr(_cfg, "step12_model", "")),
            "step12_run_limit": getattr(_cfg, "step12_run_limit", None),
        }
    except ImportError:
        cfg_dict = {}
    return 0 if run(cfg_dict) else 1


if __name__ == "__main__":
    raise SystemExit(main())
