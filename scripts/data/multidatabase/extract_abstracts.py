"""
Extract abstracts from PDFs and patch them into RIS files.

For each org folder that has both a RIS file and PDFs:
1. Parse the RIS file
2. For each record, find the associated PDF (via L1/L2 field or title match)
3. Extract abstract from PDF using pdftotext
4. Patch the AB field into the RIS record if missing

Usage:
    python extract_abstracts.py
"""

import os
import re
import subprocess
from pathlib import Path

BASE = Path(__file__).parent

ORGS_WITH_PDFS = [
    "cgspace", "adb", "afdb", "gef", "gca", "fcdo", "ipam", "wasp", "unfccc"
]


def extract_text_from_pdf(pdf_path: Path) -> str:
    try:
        result = subprocess.run(
            ["pdftotext", str(pdf_path), "-"],
            capture_output=True, text=True, timeout=30
        )
        return result.stdout
    except Exception:
        return ""


def extract_abstract_from_text(text: str) -> str:
    """Try to find an abstract section in the PDF text."""
    # Try labeled abstract section
    patterns = [
        r"(?i)(?:^|\n)\s*abstract\s*[\n:]\s*(.*?)(?=\n\s*(?:introduction|keywords|background|1\.|contents|table of contents)|$)",
        r"(?i)(?:^|\n)\s*executive summary\s*[\n:]\s*(.*?)(?=\n\s*(?:introduction|1\.|contents|table of contents)|$)",
        r"(?i)(?:^|\n)\s*summary\s*[\n:]\s*(.*?)(?=\n\s*(?:introduction|1\.|contents|table of contents)|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            abstract = match.group(1).strip()
            abstract = re.sub(r'\s+', ' ', abstract)
            # Keep only first 1000 chars to avoid grabbing too much
            if len(abstract) > 100:
                return abstract[:1000].strip()

    # Fallback: use first substantive paragraph (skip title pages)
    lines = [l.strip() for l in text.split('\n') if len(l.strip()) > 80]
    if lines:
        candidate = ' '.join(lines[:3])
        candidate = re.sub(r'\s+', ' ', candidate)
        return candidate[:1000].strip()

    return ""


def parse_ris(ris_path: Path) -> list[dict]:
    """Parse RIS file into list of record dicts."""
    records = []
    current = {}
    with open(ris_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('ER  -'):
                if current:
                    records.append(current)
                current = {}
            elif len(line) >= 6 and line[2:6] == '  - ':
                tag = line[:2]
                value = line[6:]
                if tag in current:
                    if isinstance(current[tag], list):
                        current[tag].append(value)
                    else:
                        current[tag] = [current[tag], value]
                else:
                    current[tag] = value
    if current:
        records.append(current)
    return records


def write_ris(records: list[dict], ris_path: Path):
    """Write records back to RIS file."""
    with open(ris_path, 'w', encoding='utf-8') as f:
        for record in records:
            for tag, value in record.items():
                if isinstance(value, list):
                    for v in value:
                        f.write(f"{tag}  - {v}\n")
                else:
                    f.write(f"{tag}  - {value}\n")
            f.write("ER  - \n\n")


def find_pdf_for_record(record: dict, org_dir: Path) -> Path | None:
    """Find the PDF file for a record via L1/L2 field or title match."""
    # Try L1 field (relative path from Zotero)
    for field in ['L1', 'L2']:
        if field in record:
            rel_path = record[field]
            if isinstance(rel_path, list):
                rel_path = rel_path[0]
            # Try relative to org_dir
            candidate = org_dir / rel_path
            if candidate.exists():
                return candidate
            # Try just the filename
            filename = Path(rel_path).name
            candidate = org_dir / filename
            if candidate.exists():
                return candidate
            # Search recursively
            matches = list(org_dir.rglob(filename))
            if matches:
                return matches[0]

    # Try title-based match
    title = record.get('TI', '')
    if isinstance(title, list):
        title = title[0]
    if title:
        title_clean = re.sub(r'[^\w\s]', '', title.lower())[:40]
        for pdf in org_dir.rglob("*.pdf"):
            pdf_clean = re.sub(r'[^\w\s]', '', pdf.stem.lower())[:40]
            if title_clean[:20] in pdf_clean or pdf_clean[:20] in title_clean:
                return pdf

    return None


def process_org(org: str):
    org_dir = BASE / org
    ris_files = list(org_dir.glob("*.ris")) + list(org_dir.glob("*.RIS"))
    if not ris_files:
        print(f"  No RIS file found, skipping")
        return 0, 0

    ris_path = ris_files[0]
    records = parse_ris(ris_path)

    patched = 0
    skipped = 0

    for record in records:
        # Skip if abstract already exists
        if 'AB' in record:
            skipped += 1
            continue

        pdf_path = find_pdf_for_record(record, org_dir)
        if not pdf_path:
            print(f"  No PDF found for: {record.get('TI', 'Unknown')[:60]}")
            continue

        text = extract_text_from_pdf(pdf_path)
        abstract = extract_abstract_from_text(text)

        if abstract:
            record['AB'] = abstract
            patched += 1
            print(f"  Patched: {record.get('TI', 'Unknown')[:60]}")
        else:
            print(f"  No abstract extracted from: {pdf_path.name}")

    write_ris(records, ris_path)
    return patched, skipped


def main():
    total_patched = 0
    total_skipped = 0

    for org in ORGS_WITH_PDFS:
        org_dir = BASE / org
        if not org_dir.exists():
            print(f"\n[{org}] folder not found, skipping")
            continue

        print(f"\n[{org}]")
        patched, skipped = process_org(org)
        total_patched += patched
        total_skipped += skipped
        print(f"  => {patched} abstracts added, {skipped} already had abstracts")

    print(f"\nDone. Total patched: {total_patched}, already had abstracts: {total_skipped}")


if __name__ == "__main__":
    main()
