#!/usr/bin/env python3
"""
step5_eligibility.py

Step 5 (Eligibility):
  - Merge Step 3 and Step 4.
  - Hard Filters (Year, Abstract Length).
  - AI Screening (Ollama).
  - Ensures CONSISTENT data (no empty cells).
"""

from __future__ import annotations
import os
import re
import json
import time
import requests
import yaml
import pandas as pd
from tqdm import tqdm
from typing import Optional, Dict, Any
import config as cfg

# --- CONFIGURATION ---
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:14b" 
TEMPERATURE = 0.0

def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def _load_yaml(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing criteria file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _normalize_doi(x: str) -> str:
    if not isinstance(x, str): return ""
    d = x.strip().lower()
    for p in ("https://doi.org/", "http://doi.org/", "doi:"):
        if d.startswith(p): d = d[len(p):]
    return d

def _build_criteria_prompt(criteria_dict: dict) -> str:
    prompt_lines = []
    for key in sorted(criteria_dict.keys()):
        val = criteria_dict[key]
        prompt_lines.append(f"{val['name']}:")
        prompt_lines.append(f"   - INCLUDE: {val['include']}")
        prompt_lines.append(f"   - EXCLUDE: {val['exclude']}")
        prompt_lines.append("")
    return "\n".join(prompt_lines)

def _get_year_from_text(text: str) -> Optional[int]:
    if not isinstance(text, str): return None
    m = re.search(r"\((\d{4})[a-z]?\)", text)
    if m: return int(m.group(1))
    m = re.search(r"\b(19|20)\d{2}\b", text)
    if m: return int(m.group(0))
    return None

def call_ollama(title: str, abstract: str, criteria_prompt: str, model: str) -> str:
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
    """
    user_message = f"TITLE: {title}\n\nABSTRACT: {abstract}"
    payload = {
        "model": model, "prompt": f"{system_prompt}\n\n{user_message}",
        "stream": False, "temperature": TEMPERATURE, "format": "json"
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=120)
        return r.json().get("response", "") if r.status_code == 200 else json.dumps({"error": f"HTTP {r.status_code}"})
    except Exception as e:
        return json.dumps({"error": str(e)})

def verify_quote(abstract: str, quote: str) -> Dict[str, Any]:
    if not quote or not abstract: return {"verified": False, "start": -1}
    abs_clean = " ".join(abstract.lower().split())
    qt_clean = " ".join(quote.lower().split())
    idx = abs_clean.find(qt_clean)
    return {"verified": True, "start": idx} if idx != -1 else {"verified": False, "start": -1}

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

    print(f"[Step 5] Loading criteria from {criteria_yml}")
    try:
        config_data = _load_yaml(criteria_yml)
        min_year = config_data.get("hard_filters", {}).get("min_year", 2005)
        criteria_text = _build_criteria_prompt(config_data.get("criteria", {}))
        crit_keys = sorted(config_data.get("criteria", {}).keys())
    except Exception as e:
        print(f"❌ Error loading criteria.yml: {e}")
        return {"status": "error"}

    if not os.path.exists(step3_csv) or not os.path.exists(step4_csv):
        print("❌ Missing Step 3 or Step 4 outputs.")
        return {"status": "error"}
        
    df3 = pd.read_csv(step3_csv)
    df4 = pd.read_csv(step4_csv)
    df3["doi_clean"] = df3["doi"].apply(_normalize_doi)
    df4["doi_clean"] = df4["doi"].apply(_normalize_doi)
    
    merged = pd.merge(df3, df4, on="doi_clean", suffixes=("_meta", "_abs"), how="left")
    merged["abstract"] = merged["abstract"].fillna("").astype(str)
    
    processed_dois = set()
    if os.path.exists(out_jsonl):
        with open(out_jsonl, "r") as f:
            for line in f:
                try:
                    j = json.loads(line)
                    if j.get("doi"): processed_dois.add(j["doi"])
                except: pass
    
    # Ensure Ollama is up
    try: requests.get(OLLAMA_URL.replace("/api/generate", ""), timeout=5)
    except: return {"status": "error", "message": "Ollama not running"}

    total_rows = len(merged)
    print(f"[Step 5] Processing {total_rows} papers...")
    pbar = tqdm(total=total_rows, desc="Eligibility Check", unit="paper")

    for idx, row in merged.iterrows():
        doi = row.get("doi_clean")
        if doi in processed_dois:
            pbar.update(1)
            continue
        
        title_cand = str(row.get("title_abs") if pd.notna(row.get("title_abs")) else row.get("title_meta"))
        rec = {
            "doi": doi, "title": title_cand, "abstract": row.get("abstract", ""),
            "year": _get_year_from_text(str(row.get("title_meta", ""))),
            "hard_filter_status": "pass", "final_decision": "pending"
        }
        
        # Hard Filter Logic
        skip = False
        skip_reason = ""
        
        if rec["year"] and rec["year"] < min_year:
            skip = True
            skip_reason = f"exclude_pre_{min_year}"
        elif len(rec["abstract"]) < 50 or rec["abstract"].lower() == "nan":
            skip = True
            skip_reason = "exclude_no_abstract"
            
        if skip:
            rec["hard_filter_status"] = skip_reason
            rec["final_decision"] = "unclear_no_abstract" if "abstract" in skip_reason else "exclude"
            # Fill criteria with placeholders to maintain data consistency
            for k in crit_keys:
                rec[f"{k}_decision"] = "unclear_skipped"
            
            with open(out_jsonl, "a") as f: f.write(json.dumps(rec) + "\n")
            pbar.update(1)
            continue

        # AI Call
        llm_resp = call_ollama(rec["title"], rec["abstract"], criteria_text, model)
        try:
            data = json.loads(llm_resp)
            decisions = []
            for k in crit_keys:
                item = data.get(k, {})
                decision = str(item.get("decision", "unclear")).lower()
                v = verify_quote(rec["abstract"], item.get("quote", ""))
                if decision == "yes" and not v["verified"]: decision = "unclear_bad_quote"
                
                rec[f"{k}_decision"] = decision
                rec[f"{k}_quote"] = item.get("quote", "")
                rec[f"{k}_verified"] = v["verified"]
                decisions.append(decision)
            
            if "no" in decisions: rec["final_decision"] = "exclude"
            elif any("unclear" in d for d in decisions): rec["final_decision"] = "unclear"
            else: rec["final_decision"] = "include"

        except Exception as e:
            rec["final_decision"] = "unclear_error"
            for k in crit_keys: rec[f"{k}_decision"] = "unclear_error"
        
        with open(out_jsonl, "a") as f: f.write(json.dumps(rec) + "\n")
        pbar.update(1)

    pbar.close()
    
    # --- STATISTICS ---
    print("[Step 5] Compiling stats...")
    final_rows = []
    if os.path.exists(out_jsonl):
        with open(out_jsonl, "r") as f:
            for line in f:
                try: final_rows.append(json.loads(line))
                except: pass
    
    df_final = pd.DataFrame(final_rows)
    # Fill any remaining NaNs (e.g. from older runs)
    crit_cols = [c for c in df_final.columns if "_decision" in c and c != "final_decision"]
    df_final[crit_cols] = df_final[crit_cols].fillna("unclear_skipped")

    stats = {
        "status": "completed", "rows": len(df_final), "timestamp_utc": _now_utc(),
        "criteria_breakdown": {}, "final_breakdown": {}
    }

    if not df_final.empty:
        for col in crit_cols:
            crit_name = col.replace("_decision", "")
            counts = df_final[col].astype(str).str.lower().value_counts()
            yes = sum(counts[k] for k in counts.keys() if "yes" in k or "include" in k)
            no = sum(counts[k] for k in counts.keys() if "no" in k or "exclude" in k)
            unclear = sum(counts[k] for k in counts.keys() if "unclear" in k or "pending" in k or "skipped" in k)
            stats["criteria_breakdown"][crit_name] = {"yes": int(yes), "no": int(no), "unclear": int(unclear)}

        final_counts = df_final["final_decision"].value_counts()
        unc_total = sum(final_counts[k] for k in final_counts.keys() if "unclear" in str(k))
        stats["final_breakdown"] = {
            "include": int(final_counts.get("include", 0)),
            "exclude": int(final_counts.get("exclude", 0)),
            "unclear": int(unc_total)
        }

    with open(out_meta, "w") as f: json.dump(stats, f, indent=2)
    
    cols = ["doi", "year", "final_decision", "hard_filter_status"]
    if not df_final.empty:
        c_cols = [c for c in df_final.columns if "_decision" in c or "_quote" in c or "_verified" in c]
        cols.extend(sorted(c_cols))
    out_df = df_final[[c for c in cols if c in df_final.columns]]
    out_df.to_csv(out_wide_csv, index=False)

    print("\n" + "="*50)
    print("📊 FINAL CONSISTENT SUMMARY")
    print("="*50)
    for k, v in stats["criteria_breakdown"].items():
        print(f"🔹 {k:<20} | ✅:{v['yes']:<3} ❌:{v['no']:<3} ⚠️:{v['unclear']:<3} | Σ:{v['yes']+v['no']+v['unclear']}")
    print("-" * 50)
    print(f"🚀 FINAL DECISION     | ✅:{stats['final_breakdown']['include']} ❌:{stats['final_breakdown']['exclude']} ⚠️:{stats['final_breakdown']['unclear']}")
    print("="*50 + "\n")

    return {"status": "ok", "path": out_wide_csv}

# --- ALIASES FOR run.py ---
def run(config): return step5_check_eligibility(config)
def run_step5(config): return step5_check_eligibility(config)
def main(config): return step5_check_eligibility(config)

if __name__ == "__main__":
    step5_check_eligibility({})