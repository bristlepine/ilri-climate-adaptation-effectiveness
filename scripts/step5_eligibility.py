#!/usr/bin/env python3
"""
step5_eligibility.py

Step 5 (Eligibility):
  - Merge Step 3 and Step 4
  - Hard Filters (Year, Abstract Length)
  - AI Screening (Ollama)
  - Resume/caching via JSONL
  - Dedupe on the fly
  - Ensures CONSISTENT data (no empty cells)
"""

from __future__ import annotations

import os
import re
import json
import time
import hashlib
from typing import Optional, Dict, Any

import requests
import yaml
import pandas as pd
from tqdm import tqdm

import config as cfg

# --- CONFIGURATION ---
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:14b"
TEMPERATURE = 0.0

# Keep model in memory so it doesn't reload every request (big speed win)
OLLAMA_KEEP_ALIVE = "30m"

# If your model supports it, this can reduce latency (harmless if ignored)
OLLAMA_THINK = False


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _load_yaml(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing criteria file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _normalize_doi(x: Any) -> str:
    if not isinstance(x, str):
        return ""
    d = x.strip().lower()
    for p in ("https://doi.org/", "http://doi.org/", "doi:"):
        if d.startswith(p):
            d = d[len(p) :]
    return d.strip().rstrip(" .),;]}>").strip()


def _build_criteria_prompt(criteria_dict: dict) -> str:
    prompt_lines = []
    for key in sorted(criteria_dict.keys()):
        val = criteria_dict[key] or {}
        prompt_lines.append(f"{val.get('name', key)}:")
        prompt_lines.append(f"   - INCLUDE: {val.get('include','')}")
        prompt_lines.append(f"   - EXCLUDE: {val.get('exclude','')}")
        prompt_lines.append("")
    return "\n".join(prompt_lines).strip()


def _get_year_from_text(text: str) -> Optional[int]:
    if not isinstance(text, str):
        return None
    m = re.search(r"\((\d{4})[a-z]?\)", text)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(19|20)\d{2}\b", text)
    if m:
        return int(m.group(0))
    return None


def _run_signature(model: str, criteria_text: str, min_year: int) -> str:
    blob = f"model={model}\nmin_year={min_year}\ncriteria=\n{criteria_text}\n"
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def call_ollama(
    session: requests.Session,
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
        r = session.post(OLLAMA_URL, json=payload, timeout=120)
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


def _fill_placeholders(rec: dict, crit_keys: list[str], decision_val: str) -> None:
    for k in crit_keys:
        rec[f"{k}_decision"] = decision_val
        rec[f"{k}_quote"] = ""
        rec[f"{k}_verified"] = False


def _load_jsonl_last_by_doi(path: str) -> dict[str, dict]:
    last: dict[str, dict] = {}
    if not os.path.exists(path):
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
            d = _normalize_doi(j.get("doi"))
            if d and d not in ("nan", "none"):
                last[d] = j  # last wins
    return last


def _rewrite_jsonl(path: str, last_by_doi: dict[str, dict]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as wf:
        for _, j in last_by_doi.items():
            wf.write(json.dumps(j, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _classify_decision(val: Any) -> str:
    """
    Robust classifier that avoids substring bugs like:
      'unclear_no_abstract' being counted as 'no' because it contains 'no'
    """
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


def step5_check_eligibility(config: dict) -> dict:
    out_dir = config.get("out_dir") or getattr(cfg, "out_dir", "outputs")
    here = os.path.dirname(os.path.abspath(__file__))
    model = config.get("ollama_model") or getattr(cfg, "OLLAMA_MODEL", DEFAULT_MODEL)

    step3_csv = os.path.join(out_dir, "step3", "step3_benchmark_list.csv")
    step4_csv = os.path.join(out_dir, "step4", "step4_abstracts.csv")
    criteria_yml = os.path.join(here, "criteria.yml")

    step5_dir = os.path.join(out_dir, "step5")
    os.makedirs(step5_dir, exist_ok=True)

    out_wide_csv = os.path.join(step5_dir, "step5_eligibility_wide.csv")
    out_jsonl = os.path.join(step5_dir, "step5_details.jsonl")
    out_meta = os.path.join(step5_dir, "step5_eligibility.meta.json")
    out_pass_csv = os.path.join(step5_dir, "step5_eligibility_passed.csv")


    print(f"[Step 5] Loading criteria from {criteria_yml}")
    cfg_yml = _load_yaml(criteria_yml)
    min_year = int(cfg_yml.get("hard_filters", {}).get("min_year", 2005))
    criteria_text = _build_criteria_prompt(cfg_yml.get("criteria", {}))
    crit_keys = sorted((cfg_yml.get("criteria", {}) or {}).keys())
    sig = _run_signature(model=model, criteria_text=criteria_text, min_year=min_year)

    if not os.path.exists(step3_csv) or not os.path.exists(step4_csv):
        return {"status": "error", "message": "Missing step3/step4 outputs"}

    df3 = pd.read_csv(step3_csv)
    df4 = pd.read_csv(step4_csv)

    df3["doi_clean"] = df3.get("doi", pd.Series([""] * len(df3))).apply(_normalize_doi)
    if "doi_clean" in df4.columns:
        df4["doi_clean"] = df4["doi_clean"].apply(_normalize_doi)
    else:
        df4["doi_clean"] = df4.get("doi", pd.Series([""] * len(df4))).apply(_normalize_doi)

    merged = pd.merge(df3, df4, on="doi_clean", suffixes=("_meta", "_abs"), how="left")
    # --- unify citation_raw/title after merge (handles suffixes) ---
    if "citation_raw" not in merged.columns:
        c_abs = "citation_raw_abs" if "citation_raw_abs" in merged.columns else None
        c_meta = "citation_raw_meta" if "citation_raw_meta" in merged.columns else None
        if c_abs or c_meta:
            merged["citation_raw"] = ""
            if c_abs:
                merged["citation_raw"] = merged[c_abs].fillna("")
            if c_meta:
                merged.loc[merged["citation_raw"].astype(str).str.strip().eq(""), "citation_raw"] = merged[c_meta].fillna("")

    merged["doi_clean"] = merged["doi_clean"].fillna("").astype(str).str.strip().str.lower()
    merged = merged[merged["doi_clean"] != ""].copy()
    merged["abstract"] = merged.get("abstract", pd.Series([""] * len(merged))).fillna("").astype(str)

    merged["_abs_len"] = merged["abstract"].astype(str).str.len()
    merged = (
        merged.sort_values(by=["doi_clean", "_abs_len"], ascending=[True, False])
        .drop_duplicates(subset=["doi_clean"], keep="first")
    )
    merged = merged.drop(columns=["_abs_len"]).copy()

    last_by_doi = _load_jsonl_last_by_doi(out_jsonl)
    if os.path.exists(out_jsonl):
        _rewrite_jsonl(out_jsonl, last_by_doi)
    processed_dois = set(last_by_doi.keys())

    try:
        requests.get(OLLAMA_URL.replace("/api/generate", ""), timeout=5)
    except Exception:
        return {"status": "error", "message": "Ollama not running"}

    total_rows = len(merged)
    print(f"[Step 5] Processing {total_rows} papers... (resume: {len(processed_dois)} already done)")
    pbar = tqdm(total=total_rows, desc="Eligibility Check", unit="paper")

    with requests.Session() as session:
        with open(out_jsonl, "a", encoding="utf-8") as jf:
            for _, row in merged.iterrows():
                doi = str(row.get("doi_clean") or "")
                if doi in processed_dois:
                    pbar.update(1)
                    continue

                title_cand = str(row.get("title_fetched") or row.get("title_benchmark") or row.get("title") or "")
                abstract = str(row.get("abstract") or "")

                year_val = None
                year_source = ""
                y_in = row.get("year", None)
                ys_in = row.get("year_source", None)
                try:
                    y_int = int(str(y_in).strip()) if str(y_in).strip() not in ("", "nan", "none") else None
                except Exception:
                    y_int = None
                if y_int:
                    year_val = y_int
                    year_source = str(ys_in or "").strip() or "carried_forward"
                else:
                    year_val = _get_year_from_text(str(row.get("citation_raw") or title_cand))
                    year_source = "parsed_from_text" if year_val else ""

                rec = {
                    "run_signature": sig,
                    "timestamp_utc": _now_utc(),
                    "doi": doi,
                    "doi_clean": doi,
                    "record_key": str(row.get("record_key") or ""),
                    "citation_raw": str(row.get("citation_raw") or ""),
                    "title_best": title_cand,
                    "title_benchmark": str(row.get("title_benchmark") or ""),
                    "title_fetched": str(row.get("title_fetched") or ""),
                    "abstract": abstract,
                    "year": year_val,
                    "year_source": year_source,
                    "hard_filter_status": "pass",
                    "final_decision": "pending",
                    "notes": "",
                }

                skip = False
                skip_reason = ""
                if rec["year"] and int(rec["year"]) < min_year:
                    skip = True
                    skip_reason = f"exclude_pre_{min_year}"
                elif len(abstract.strip()) < 50 or abstract.strip().lower() == "nan":
                    skip = True
                    skip_reason = "exclude_no_abstract"

                if skip:
                    rec["hard_filter_status"] = skip_reason
                    rec["final_decision"] = "unclear_no_abstract" if "abstract" in skip_reason else "exclude"
                    _fill_placeholders(rec, crit_keys, "unclear_skipped")
                    jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    processed_dois.add(doi)
                    pbar.update(1)
                    continue

                llm_resp = call_ollama(session, title_cand, abstract, criteria_text, model)
                try:
                    data = json.loads(llm_resp)
                    decisions = []
                    full_text = f"{title_cand}\n{abstract}"

                    for k in crit_keys:
                        item = data.get(k, {}) if isinstance(data, dict) else {}
                        decision = str(item.get("decision", "unclear")).lower().strip()
                        quote = str(item.get("quote", "") or "")

                        v = verify_quote(full_text, quote)
                        if decision == "yes" and not v["verified"]:
                            decision = "unclear_bad_quote"

                        rec[f"{k}_decision"] = decision
                        rec[f"{k}_quote"] = quote
                        rec[f"{k}_verified"] = bool(v["verified"])
                        decisions.append(decision)

                    if "no" in decisions:
                        rec["final_decision"] = "exclude"
                    elif any(d.startswith("unclear") for d in decisions):
                        rec["final_decision"] = "unclear"
                    else:
                        rec["final_decision"] = "include"

                except Exception:
                    rec["final_decision"] = "unclear_error"
                    _fill_placeholders(rec, crit_keys, "unclear_error")

                jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                processed_dois.add(doi)
                pbar.update(1)

    pbar.close()

    print("[Step 5] Compiling stats...")
    last_by_doi = _load_jsonl_last_by_doi(out_jsonl)
    df_final = pd.DataFrame(list(last_by_doi.values()))

    if "notes" not in df_final.columns:
        df_final["notes"] = ""
    df_final["notes"] = df_final["notes"].fillna("").astype(str)

    lk = df4[
        [
            "doi_clean",
            "record_key",
            "citation_raw",
            "title_benchmark",
            "title_fetched",
            "type",
            "identified_via",
            "year",
            "year_source",
            "abstract",
            "landing_url",
            "fetch_status",
            "source",
            "status_code",
            "error",
            "fetched_utc",
            "last_attempt_utc",
            "fail_count",
        ]
    ].copy()

    lk["doi_clean"] = lk["doi_clean"].fillna("").astype(str).str.strip().str.lower()
    lk = lk.rename(columns={"doi_clean": "doi"})

    df_final["doi"] = df_final.get("doi", "").fillna("").astype(str).str.strip().str.lower()

    lk = lk.drop_duplicates(subset=["doi"], keep="first")

    df_final = df_final.merge(lk, on="doi", how="left", suffixes=("", "_step4"))

    for c in [
        "doi",
        "record_key",
        "citation_raw",
        "title_benchmark",
        "title_fetched",
        "type",
        "identified_via",
        "year",
        "year_source",
        "abstract",
        "landing_url",
        "fetch_status",
        "source",
        "status_code",
        "error",
        "fetched_utc",
        "last_attempt_utc",
        "fail_count",
    ]:
        sc = f"{c}_step4"
        if sc in df_final.columns:
            if c in df_final.columns:
                mask = df_final[c].isna() | df_final[c].astype(str).str.strip().eq("")
                df_final.loc[mask, c] = df_final.loc[mask, sc]
            else:
                df_final[c] = df_final[sc]
            df_final = df_final.drop(columns=[sc])
    
    if "year" in df_final.columns:
        df_final["year"] = pd.to_numeric(df_final["year"], errors="coerce").astype("Int64")

    if "status_code" in df_final.columns:
        df_final["status_code"] = pd.to_numeric(df_final["status_code"], errors="coerce").astype("Int64")

    df_final["doi_clean"] = (
        df_final.get("doi_clean", pd.Series([None] * len(df_final)))
        .fillna(df_final["doi"])
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if not df_final.empty:
        for c in [c for c in df_final.columns if c.endswith("_decision")]:
            df_final[c] = df_final[c].fillna("unclear_skipped")
        for c in [c for c in df_final.columns if c.endswith("_quote")]:
            df_final[c] = df_final[c].fillna("")
        for c in [c for c in df_final.columns if c.endswith("_verified")]:
            df_final[c] = df_final[c].apply(
                lambda x: bool(x) if isinstance(x, bool) else str(x).strip().lower() in ("true", "1", "yes")
            )

    crit_decision_cols = [c for c in df_final.columns if c.endswith("_decision") and c != "final_decision"]

    stats = {
        "status": "completed",
        "rows": int(len(df_final)),
        "timestamp_utc": _now_utc(),
        "run_signature": sig,
        "criteria_breakdown": {},
        "final_breakdown": {},
    }

    if not df_final.empty:
        for col in crit_decision_cols:
            crit_name = col.replace("_decision", "")
            s = df_final[col].map(_classify_decision)
            yes = int((s == "yes").sum())
            no = int((s == "no").sum())
            unclear = int((s == "unclear").sum())
            stats["criteria_breakdown"][crit_name] = {"yes": yes, "no": no, "unclear": unclear}

        fc = df_final["final_decision"].astype(str).str.lower().value_counts()
        unc_total = sum(int(fc.get(k, 0)) for k in fc.keys() if str(k).startswith("unclear"))
        stats["final_breakdown"] = {
            "include": int(fc.get("include", 0)),
            "exclude": int(fc.get("exclude", 0)),
            "unclear": int(unc_total),
        }

    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    base_cols = [
        "doi",
        "doi_clean",
        "record_key",
        "citation_raw",
        "title_best",
        "title_benchmark",
        "title_fetched",
        "type",
        "identified_via",
        "abstract",
        "landing_url",
        "fetch_status",
        "source",
        "status_code",
        "error",
        "fetched_utc",
        "last_attempt_utc",
        "fail_count",
        "year",
        "year_source",
        "final_decision",
        "notes",
        "hard_filter_status",
        "run_signature",
        "timestamp_utc",
    ]

    extra_cols = sorted(
        [
            c
            for c in df_final.columns
            if (c.endswith("_decision") or c.endswith("_quote") or c.endswith("_verified"))
            and c != "final_decision"
        ]
    )
    cols = [c for c in base_cols + extra_cols if c in df_final.columns]
    seen = set()
    cols = [c for c in cols if not (c in seen or seen.add(c))]
    df_final[cols].to_csv(out_wide_csv, index=False)

    # --- Passed-only export (final_decision == include) ---
    if "final_decision" in df_final.columns and not df_final.empty:
        passed = df_final[df_final["final_decision"].astype(str).str.lower().eq("include")].copy()
    else:
        passed = df_final.iloc[0:0].copy()

    passed[cols].to_csv(out_pass_csv, index=False)


    print("\n" + "=" * 50)
    print("📊 FINAL CONSISTENT SUMMARY")
    print("=" * 50)
    for k, v in stats["criteria_breakdown"].items():
        print(f"🔹 {k:<20} | ✅:{v['yes']:<3} ❌:{v['no']:<3} ⚠️:{v['unclear']:<3} | Σ:{v['yes']+v['no']+v['unclear']}")
    print("-" * 50)
    fb = stats["final_breakdown"]
    inc = int(fb.get("include", 0) or 0)
    exc = int(fb.get("exclude", 0) or 0)
    unc = int(fb.get("unclear", 0) or 0)
    tot = inc + exc + unc
    print(f"🚀 FINAL DECISION     | ✅:{inc} ❌:{exc} ⚠️:{unc} | Σ:{tot}")
    print("=" * 50 + "\n")

    return {"status": "ok", "path": out_wide_csv}


def run(config: dict) -> dict:
    return step5_check_eligibility(config)


def run_step5(config: dict) -> dict:
    return step5_check_eligibility(config)


def main(config: dict) -> dict:
    return step5_check_eligibility(config)


if __name__ == "__main__":
    step5_check_eligibility({})
