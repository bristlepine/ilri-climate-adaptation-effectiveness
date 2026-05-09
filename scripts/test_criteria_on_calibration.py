"""
test_criteria_on_calibration.py

Re-screens the R2a and R3a calibration CSVs using the current criteria.yml
and the same LLM as step12, then runs IRR analysis to check sensitivity/kappa.

Usage:
    python scripts/test_criteria_on_calibration.py

Outputs updated LLM column in scripts/results/ and re-runs step11.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import yaml

HERE      = Path(__file__).resolve().parent
RESULTS   = HERE / "results"
CRITERIA  = HERE / "criteria.yml"
STEP11    = HERE / "step11_irr_analysis.py"

MODEL     = "qwen2.5:14b"
OLLAMA    = "http://localhost:11434/api/generate"

# Which CSV files to test, and which column to write LLM decisions into
TARGETS = [
    {"file": "EPPI Review - R2a.csv", "llm_col": "LLM_r2a", "rec_col": "CJ Reconciled"},
    {"file": "EPPI Review - R3a.csv", "llm_col": "LLM",     "rec_col": "CJ Reconciled"},
]


def load_criteria() -> str:
    data = yaml.safe_load(CRITERIA.read_text())
    criteria = data.get("criteria", {})
    lines = []
    for key in sorted(criteria.keys()):
        c = criteria[key]
        name = c.get("name", key)
        inc  = c.get("include", "")
        exc  = c.get("exclude", "")
        r2i  = c.get("r2_include_further_guidelines", "")
        r2e  = c.get("r2_exclude_further_guidelines", "")
        lines.append(f"### {name}")
        lines.append(f"INCLUDE if: {inc}")
        lines.append(f"EXCLUDE if: {exc}")
        if r2i:
            lines.append(f"Further include guidance: {r2i.strip()}")
        if r2e:
            lines.append(f"Further exclude guidance: {r2e.strip()}")
        lines.append("")
    return "\n".join(lines)


SYSTEM_PROMPT = """You are a strict research assistant screening papers for a systematic review on climate adaptation effectiveness for smallholder producers.

Analyze the Title and Abstract against these eligibility criteria:

{criteria}

Respond with ONLY valid JSON:
{{"decision": "Include" or "Exclude", "reason": "brief reason"}}"""


def screen_one(title: str, abstract: str, criteria_text: str) -> str:
    prompt = f"TITLE: {title}\n\nABSTRACT: {abstract}"
    system = SYSTEM_PROMPT.format(criteria=criteria_text)
    payload = {
        "model": MODEL,
        "system": system,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 200},
    }
    try:
        r = requests.post(OLLAMA, json=payload, timeout=60)
        r.raise_for_status()
        text = r.json().get("response", "")
        # Extract JSON
        m = re.search(r'\{.*?\}', text, re.DOTALL)
        if m:
            data = json.loads(m.group())
            return data.get("decision", "Exclude")
    except Exception as e:
        print(f"    ERROR: {e}")
    return "Exclude"


def metrics(pred: pd.Series, truth: pd.Series) -> dict:
    p = pred.str.upper()
    t = truth.str.upper()
    tp = int(((p == "INCLUDE") & (t == "INCLUDE")).sum())
    tn = int(((p == "EXCLUDE") & (t == "EXCLUDE")).sum())
    fp = int(((p == "INCLUDE") & (t == "EXCLUDE")).sum())
    fn = int(((p == "EXCLUDE") & (t == "INCLUDE")).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    # kappa
    total = tp + tn + fp + fn
    p_o = (tp + tn) / total if total else 0
    p_inc_pred = (tp + fp) / total if total else 0
    p_inc_true = (tp + fn) / total if total else 0
    p_e = p_inc_pred * p_inc_true + (1 - p_inc_pred) * (1 - p_inc_true)
    kappa = (p_o - p_e) / (1 - p_e) if abs(1 - p_e) > 1e-9 else 1.0
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "sensitivity": round(sensitivity, 3),
        "precision":   round(precision, 3),
        "f1":          round(f1, 3),
        "kappa":       round(kappa, 3),
        "agreement":   round(100 * p_o, 1),
    }


def main():
    criteria_text = load_criteria()
    print(f"Loaded criteria from {CRITERIA}")
    print(f"Model: {MODEL}\n")

    all_pass = True
    for target in TARGETS:
        csv_path = RESULTS / target["file"]
        llm_col  = target["llm_col"]
        rec_col  = target["rec_col"]

        df = pd.read_csv(csv_path)
        print(f"=== {target['file']} (n={len(df)}) ===")

        decisions = []
        for i, (_, row) in enumerate(df.iterrows()):
            title    = str(row.get("Item", "") or "")
            abstract = ""  # calibration CSVs only have titles; use title as proxy
            decision = screen_one(title, abstract, criteria_text)
            decisions.append(decision)
            m = metrics(
                pd.Series(decisions + ["Exclude"] * (len(df) - len(decisions))),
                df[rec_col].iloc[:len(decisions) + (len(df) - len(decisions))]
            )
            print(f"  [{i+1}/{len(df)}] {decision:7s} | running: sens={m['sensitivity']:.3f} κ={m['kappa']:.3f}", end="\r")

        df[llm_col] = decisions
        df.to_csv(csv_path, index=False)

        m = metrics(pd.Series(decisions), df[rec_col])
        print(f"\n  Final: sensitivity={m['sensitivity']} precision={m['precision']} "
              f"F1={m['f1']} κ={m['kappa']} agreement={m['agreement']}%")

        fn_mask = (pd.Series(decisions).str.upper() == "EXCLUDE") & (df[rec_col].str.upper() == "INCLUDE")
        fn_titles = df.loc[fn_mask.values, "Item"].tolist()
        if fn_titles:
            print(f"  Still missing ({len(fn_titles)} false negatives):")
            for t in fn_titles:
                print(f"    - {str(t)[:100]}")

        sens_ok  = m["sensitivity"] >= 0.95
        kappa_ok = m["kappa"] >= 0.60
        if sens_ok and kappa_ok:
            print(f"  ✓ BOTH thresholds met")
        else:
            all_pass = False
            if not sens_ok:
                print(f"  ✗ Sensitivity {m['sensitivity']} < 0.95")
            if not kappa_ok:
                print(f"  ✗ Kappa {m['kappa']} < 0.60")
        print()

    # Re-run step11 to update figures
    print("Re-running step11 to update IRR figures...")
    import importlib.util
    spec = importlib.util.spec_from_file_location("step11", STEP11)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()

    print("\n=== SUMMARY ===")
    print("All thresholds met." if all_pass else "Some thresholds still not met — review false negatives above and update criteria.yml.")


if __name__ == "__main__":
    main()
