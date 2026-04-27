"""
recode_r1b.py — FT-R1b: LLM re-code of R1a papers using criteria v2

Runs the 5 R1a papers through the LLM using criteria_sysmap_v2.yml.
Saves output to coding_ft_r1b_LLM.csv in this folder.

Usage:
  conda run -n ilri01 python documentation/coding/systematic-map/rounds/FT-R1b/recode_r1b.py
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pandas as pd
import requests
import yaml

HERE      = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent.parent.parent.parent  # → project root
PDFS_DIR  = HERE.parent / "FT-R1a" / "FT-R1a pdfs"
CRITERIA  = (
    REPO_ROOT / "documentation" / "coding" / "systematic-map"
    / "llm-criteria" / "criteria_sysmap_v1.yml"
)
OUT_CSV   = HERE / "coding_ft_r1b_LLM.csv"

OLLAMA_URL  = "http://localhost:11434/api/generate"
MODEL       = "qwen2.5:14b"
TEMPERATURE = 0.0
MAX_CHARS   = 12_000

R1A_DOIS = [
    "10.1016/j.agee.2019.04.004",
    "10.1016/j.crm.2017.06.001",
    "10.1016/j.crm.2017.03.001",
    "10.1007/s10584-016-1792-0",
    "10.1080/17565529.2017.1411240",
]

# Map DOI → filename (same convention as step13)
def doi_to_filename(doi: str) -> str:
    return "doi_" + doi.replace("/", "_") + ".pdf"


def load_criteria(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)["fields"]


def read_pdf_text(pdf_path: Path, max_chars: int) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        text = "\n".join(
            page.extract_text() or "" for page in reader.pages
        )
        return text[:max_chars]
    except Exception as e:
        print(f"  [warn] Could not extract text from {pdf_path.name}: {e}")
        return ""


def build_prompt(doi: str, text: str, fields: dict) -> str:
    field_blocks = []
    for fname, fdef in fields.items():
        block = f'  "{fname}":\n'
        block += f'    # {fdef["name"]}\n'
        block += f'    # Extract: {fdef["extract"].strip()}\n'
        block += f'    # Valid values: {fdef["valid_values"]}\n'
        if "r1_further_guidance" in fdef:
            block += f'    # IMPORTANT: {fdef["r1_further_guidance"].strip()}\n'
        field_blocks.append(block)

    fields_str = "\n".join(field_blocks)

    return f"""You are a systematic map data extractor. Extract structured information from the academic paper below.

DOI: {doi}

Return ONLY a valid JSON object with exactly these fields:

{{
{fields_str}
}}

Rules:
- Use ONLY the valid values listed for each field (exact spelling, lowercase with underscores)
- For multi-value fields, separate values with semicolons
- For free-text fields, be concise and precise
- If a field cannot be determined from the text, enter "not_reported"
- Do NOT add fields not listed above
- Do NOT include any explanation outside the JSON object

PAPER TEXT:
{text}

JSON output:"""


def call_llm(prompt: str) -> dict:
    payload = {
        "model":  MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": TEMPERATURE},
        "keep_alive": "10m",
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=300)
    r.raise_for_status()
    raw = r.json().get("response", "")

    # Extract JSON from response
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in response:\n{raw[:300]}")
    return json.loads(match.group())


def main() -> None:
    print(f"FT-R1b: LLM re-code with updated criteria v1")
    print(f"Criteria : {CRITERIA}")
    print(f"PDFs     : {PDFS_DIR}")
    print(f"Output   : {OUT_CSV}")
    print()

    fields = load_criteria(CRITERIA)
    print(f"Fields   : {list(fields.keys())}\n")

    rows = []
    for doi in R1A_DOIS:
        fname = doi_to_filename(doi)
        pdf_path = PDFS_DIR / fname
        if not pdf_path.exists():
            print(f"  SKIP {doi} — PDF not found at {pdf_path}")
            continue

        print(f"  Coding {doi} ...", end=" ", flush=True)
        t0 = time.time()

        text = read_pdf_text(pdf_path, MAX_CHARS)
        if not text.strip():
            print("no text extracted — skipping")
            continue

        prompt = build_prompt(doi, text, fields)

        try:
            result = call_llm(prompt)
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        result["doi"]      = doi
        result["filename"] = fname
        result["coder_id"] = "LLM_v2"
        rows.append(result)
        print(f"done ({time.time()-t0:.1f}s)")

    if not rows:
        print("No rows produced — check Ollama is running.")
        return

    # Build DataFrame with consistent column order
    col_order = ["doi", "filename"] + list(fields.keys()) + ["coder_id"]
    df = pd.DataFrame(rows)
    for c in col_order:
        if c not in df.columns:
            df[c] = ""
    df = df[col_order]

    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(df)} rows → {OUT_CSV}")


if __name__ == "__main__":
    main()
