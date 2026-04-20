"""
step12b_screen_wos.py

Abstract screening for WOS net-new records (output of step2b_multidatabase_dedupe.py).

Applies the same PCCM eligibility criteria as step12, but reads from
step2b_net_new.csv instead of the Scopus-enriched step9a CSV.

Outputs (under outputs/step12b/):
  - step12b_results.csv          — screening decisions for all 4,683 WOS records
  - step12b_results.meta.json    — counts, elapsed time, criteria breakdown
  - step12b_results_details.jsonl — per-record cache (same format as step12)
  - step12b_summary.png          — decision bar chart

Net-new included records can then be merged with step12 results and fed into
step13 (full-text retrieval) as part of the expanded corpus.

Run: python scripts/step12b_screen_wos.py
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import yaml

# ── resolve paths ─────────────────────────────────────────────────────────────
HERE     = Path(__file__).resolve().parent
OUT_ROOT = HERE / "outputs"
IN_CSV   = OUT_ROOT / "step2b" / "step2b_net_new.csv"
OUT_DIR  = OUT_ROOT / "step12b"

# ── borrow helpers from step12 ────────────────────────────────────────────────
sys.path.insert(0, str(HERE))
from step12_screen_abstracts import (
    DEFAULT_MODEL,
    OLLAMA_URL,
    normalize_doi,
    year_from_any,
    safe_str,
    stable_key,
    _load_yaml,
    _build_criteria_prompt,
    _run_signature,
    _ollama_fail_fast,
    _load_jsonl_cache,
    _rewrite_jsonl,
    _call_llm,
    _parse_response,
    _plot_summary,
)

PRINT_EVERY = 50


# =============================================================================
# Load and align net-new CSV to step12 schema
# =============================================================================

def load_net_new(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"step2b net-new CSV not found: {path}\n"
            "Run step2b_multidatabase_dedupe.py first."
        )
    df = pd.read_csv(path, dtype=str, engine="python", on_bad_lines="skip").fillna("")

    # Rename record_key → dedupe_key (step12 cache key logic uses this)
    if "record_key" in df.columns and "dedupe_key" not in df.columns:
        df = df.rename(columns={"record_key": "dedupe_key"})

    # Ensure all columns step12's load_enriched adds are present
    for col in ["dedupe_key", "title", "abstract", "xref_abstract",
                "doi", "year", "coverDate", "eid", "scopus_id",
                "publicationName", "author_names", "abstract_source",
                "abstract_status"]:
        if col not in df.columns:
            df[col] = ""

    # Normalise
    df["doi"]   = df["doi"].apply(normalize_doi)
    df["year"]  = df["year"].apply(year_from_any)
    df["title"] = df["title"].apply(safe_str)

    # Best abstract (WOS always has abstract in the abstract column)
    def _best_abs(row: pd.Series) -> Tuple[str, str]:
        for col in ["abstract", "xref_abstract"]:
            v = safe_str(row.get(col, ""))
            if v:
                return v, col
        return "", ""

    best_abs, best_src = zip(*[_best_abs(r) for _, r in df.iterrows()]) if len(df) else ([], [])
    df["_abstract_best"]     = list(best_abs)
    df["_abstract_best_src"] = list(best_src)

    return df.reset_index(drop=True)


# =============================================================================
# Screening loop (mirrors step12.screen_full exactly, different output dir)
# =============================================================================

def screen_wos(
    df: pd.DataFrame,
    *,
    out_dir: Path,
    model: str,
    criteria_yml_path: Path,
    run_limit: Optional[int] = None,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg           = _load_yaml(criteria_yml_path)
    min_year      = int((cfg.get("hard_filters", {}) or {}).get("min_year", 2005))
    criteria_text = _build_criteria_prompt(cfg.get("criteria", {}) or {})
    crit_keys     = sorted((cfg.get("criteria", {}) or {}).keys())
    sig           = _run_signature(model=model, criteria_text=criteria_text, min_year=min_year)

    _ollama_fail_fast(model)

    jsonl_path = out_dir / "step12b_results_details.jsonl"
    cache      = _load_jsonl_cache(jsonl_path)
    if cache:
        _rewrite_jsonl(jsonl_path, cache)
    processed = set(cache.keys())
    print(f"[step12b] Cache warm: {len(processed):,} records from {jsonl_path}")

    df = df.copy()
    df["screen_decision"]    = ""
    df["screen_reasons"]     = ""
    df["screen_rule_hits"]   = ""
    df["screen_checked_at_utc"] = ""

    n_total = len(df)
    n_run   = min(int(run_limit), n_total) if run_limit else n_total

    t0 = time.time()
    n_include = n_exclude = n_missing = n_cached = n_error = 0

    print(f"[step12b] Start screening | total={n_total:,} | running={n_run:,} | model={model}")

    with requests.Session() as session:
        with open(jsonl_path, "a", encoding="utf-8", buffering=1) as jf:
            for i in range(n_run):
                row      = df.iloc[i]
                k        = stable_key(row, i)
                title    = safe_str(row.get("title", ""))
                abstract = safe_str(row.get("_abstract_best", ""))

                if i == 0 or (i + 1) % PRINT_EVERY == 0 or (i + 1) == n_run:
                    elapsed = time.time() - t0
                    rate    = (i + 1) / elapsed if elapsed > 0 else 0
                    eta     = int((n_run - i - 1) / rate) if rate > 0 else 0
                    print(
                        f"[step12b] {i+1:,}/{n_run:,} | "
                        f"include={n_include} exclude={n_exclude} "
                        f"missing={n_missing} cached={n_cached} err={n_error} | "
                        f"elapsed={int(elapsed)}s ETA={eta}s"
                    )

                # ── year hard filter ──────────────────────────────────────────
                yr = year_from_any(row.get("year", "") or row.get("coverDate", ""))
                if yr and int(yr) < min_year:
                    decision, reasons, rule_hits = "Exclude", f"year<{min_year}", "{}"
                    df.at[i, "screen_decision"]  = decision
                    df.at[i, "screen_reasons"]   = reasons
                    df.at[i, "screen_rule_hits"] = rule_hits
                    n_exclude += 1
                    rec = {"key": k, "sig": sig, "decision": decision,
                           "reasons": reasons, "rule_hits": rule_hits,
                           "title": title, "source": "year_filter"}
                    jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    continue

                # ── cache hit ─────────────────────────────────────────────────
                if k in cache and cache[k].get("sig") == sig:
                    cached = cache[k]
                    df.at[i, "screen_decision"]  = cached.get("decision", "")
                    df.at[i, "screen_reasons"]   = cached.get("reasons", "")
                    df.at[i, "screen_rule_hits"] = cached.get("rule_hits", "{}")
                    d = cached.get("decision", "")
                    if d == "Include":   n_include += 1
                    elif d == "Exclude": n_exclude += 1
                    else:                n_missing += 1
                    n_cached += 1
                    continue

                # ── missing abstract ──────────────────────────────────────────
                if not abstract:
                    decision = "Needs_Manual"
                    reasons  = "Missing abstract — cannot screen"
                    df.at[i, "screen_decision"]  = decision
                    df.at[i, "screen_reasons"]   = reasons
                    df.at[i, "screen_rule_hits"] = "{}"
                    n_missing += 1
                    rec = {"key": k, "sig": sig, "decision": decision,
                           "reasons": reasons, "rule_hits": "{}",
                           "title": title, "source": "missing_abstract"}
                    jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    cache[k] = rec
                    continue

                # ── LLM call ──────────────────────────────────────────────────
                try:
                    raw = _call_llm(
                        title=title,
                        abstract=abstract,
                        criteria_text=criteria_text,
                        model=model,
                        session=session,
                    )
                    decision, reasons, rule_hits = _parse_response(raw, crit_keys)
                except Exception as e:
                    n_error += 1
                    print(f"[step12b] ERROR row {i}: {e}")
                    df.at[i, "screen_decision"]  = "Needs_Manual"
                    df.at[i, "screen_reasons"]   = f"LLM error: {e}"
                    df.at[i, "screen_rule_hits"] = "{}"
                    continue

                df.at[i, "screen_decision"]     = decision
                df.at[i, "screen_reasons"]      = reasons
                df.at[i, "screen_rule_hits"]    = rule_hits
                df.at[i, "screen_checked_at_utc"] = time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                )

                if decision == "Include":   n_include += 1
                elif decision == "Exclude": n_exclude += 1
                else:                       n_missing += 1

                rec = {"key": k, "sig": sig, "decision": decision,
                       "reasons": reasons, "rule_hits": rule_hits,
                       "title": title, "source": "llm"}
                jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                cache[k] = rec

    print(f"[step12b] Done | include={n_include} exclude={n_exclude} "
          f"missing={n_missing} errors={n_error} | "
          f"elapsed={int(time.time()-t0)}s")
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
    out_csv  = out_dir / "step12b_results.csv"
    out_meta = out_dir / "step12b_results.meta.json"

    decision_cols = ["screen_decision", "screen_reasons"]
    other_cols    = [c for c in df.columns
                     if c not in decision_cols
                     and not c.startswith("screen_rule_hits")
                     and not c.startswith("_")]
    rule_cols     = [c for c in df.columns if c == "screen_rule_hits"]
    col_order     = [c for c in decision_cols + other_cols + rule_cols if c in df.columns]
    df[col_order].to_csv(out_csv, index=False)

    decisions = df["screen_decision"].fillna("").str.strip()
    counts    = decisions.value_counts(dropna=False).to_dict()

    from collections import Counter
    excl_by_crit: Counter = Counter()
    for d, r in zip(decisions, df["screen_reasons"].fillna("").astype(str)):
        if d == "Exclude":
            for m in re.finditer(r"\b([1-5]_[a-zA-Z0-9]+)\s*:", r):
                excl_by_crit[m.group(1)] += 1

    meta: Dict[str, Any] = {
        "source":            "WOS net-new (step2b)",
        "input_csv":         str(input_csv),
        "criteria_yml":      str(criteria_yml),
        "output_csv":        str(out_csv),
        "rows_total":        int(len(df)),
        "rows_screened":     int((decisions != "").sum()),
        "timestamp_utc":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds":   float(elapsed_seconds) if elapsed_seconds else None,
        "elapsed_hms":       time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds)) if elapsed_seconds else None,
        "decision_counts":   {k: int(v) for k, v in counts.items()},
        "excluded_by_criterion": dict(sorted(excl_by_crit.items(), key=lambda kv: (-kv[1], kv[0]))),
    }

    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[step12b] Wrote: {out_csv}")
    print(f"[step12b] Wrote: {out_meta}")
    print(f"[step12b] Decision counts: {counts}")

    # Reuse step12's plot helper (reads from meta dict)
    _plot_summary(meta, out_dir)

    return meta


# =============================================================================
# Entrypoint
# =============================================================================

def run(
    input_csv: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    model: Optional[str] = None,
    criteria_yml: Optional[Path] = None,
    run_limit: Optional[int] = None,
) -> Dict[str, Any]:
    t_start = time.time()

    input_csv    = input_csv    or IN_CSV
    out_dir      = out_dir      or OUT_DIR
    model        = model        or DEFAULT_MODEL
    run_limit    = run_limit    or None

    # Criteria: prefer explicit arg, then config, then default abstract screening criteria
    if criteria_yml is None:
        try:
            import config as _cfg
            crit_str = safe_str(getattr(_cfg, "step12_criteria_yml", "")
                                or getattr(_cfg, "step10_criteria_yml", ""))
            criteria_yml = Path(crit_str) if crit_str else None
        except ImportError:
            pass
    if criteria_yml is None:
        criteria_yml = (
            HERE.parent / "documentation" / "coding" / "abstract-screening"
            / "criteria" / "criteria.yml"
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[step12b] Input CSV  : {input_csv}")
    print(f"[step12b] Criteria   : {criteria_yml}")
    print(f"[step12b] Output dir : {out_dir}")
    print(f"[step12b] Model      : {model}")

    df = load_net_new(input_csv)
    print(f"[step12b] WOS net-new records loaded: {len(df):,}")
    has_abstract = (df["_abstract_best"].str.strip() != "").sum()
    print(f"[step12b] Records with abstract: {has_abstract:,} / {len(df):,}")

    df = screen_wos(
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


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Step 12b — abstract screening for WOS net-new records")
    ap.add_argument("--model",        default=None, help=f"Ollama model (default: {DEFAULT_MODEL})")
    ap.add_argument("--criteria-yml", default=None, help="Criteria YAML (default: same as step12)")
    ap.add_argument("--run-limit",    type=int, default=None, help="Screen only first N records (testing)")
    ap.add_argument("--input-csv",    default=None, help=f"Net-new CSV (default: {IN_CSV})")
    ap.add_argument("--out-dir",      default=None, help=f"Output dir (default: {OUT_DIR})")
    args = ap.parse_args()

    run(
        input_csv    = Path(args.input_csv).resolve()    if args.input_csv    else None,
        out_dir      = Path(args.out_dir).resolve()      if args.out_dir      else None,
        model        = args.model,
        criteria_yml = Path(args.criteria_yml).resolve() if args.criteria_yml else None,
        run_limit    = args.run_limit,
    )
