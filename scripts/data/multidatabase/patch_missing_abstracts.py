"""
Patch missing abstracts for: 3ie, Campbell, UNEP, UNFCCC, WoS, EconLit, ProQuest.
Fetches from web pages, PDFs, or DSpace APIs as appropriate.

Usage:
    python patch_missing_abstracts.py
"""

import re
import time
import subprocess
import tempfile
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE = Path(__file__).parent

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def parse_ris(ris_path: Path) -> list[dict]:
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
    with open(ris_path, 'w', encoding='utf-8') as f:
        for record in records:
            for tag, value in record.items():
                if isinstance(value, list):
                    for v in value:
                        f.write(f"{tag}  - {v}\n")
                else:
                    f.write(f"{tag}  - {value}\n")
            f.write("ER  - \n\n")


def extract_abstract_from_text(text: str) -> str:
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
            if len(abstract) > 100:
                return abstract[:1000].strip()
    lines = [l.strip() for l in text.split('\n') if len(l.strip()) > 80]
    if lines:
        candidate = ' '.join(lines[:3])
        return re.sub(r'\s+', ' ', candidate)[:1000].strip()
    return ""


def fetch_pdf_abstract(url: str, save_path: Path = None) -> str:
    try:
        r = requests.get(url, headers=HEADERS, timeout=30, stream=True)
        if r.status_code != 200:
            return ""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
            tmp_path = f.name
        if save_path:
            import shutil
            save_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(tmp_path, save_path)
            print(f"    Saved PDF: {save_path.name}")
        result = subprocess.run(["pdftotext", tmp_path, "-"], capture_output=True, text=True, timeout=30)
        Path(tmp_path).unlink(missing_ok=True)
        return extract_abstract_from_text(result.stdout)
    except Exception as e:
        print(f"    PDF error: {e}")
    return ""


def scrape_page_abstract(url: str) -> str:
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.text, 'html.parser')
        # Try meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name', '') or meta.get('property', '')
            if name in ('description', 'og:description', 'DC.description'):
                content = meta.get('content', '').strip()
                if len(content) > 100:
                    return content[:1000]
        # Try substantive paragraphs
        paras = [p.get_text().strip() for p in soup.find_all('p') if len(p.get_text().strip()) > 100]
        if paras:
            return re.sub(r'\s+', ' ', ' '.join(paras[:2]))[:1000]
    except Exception as e:
        print(f"    Scrape error: {e}")
    return ""


def fetch_dspace_abstract(url: str, pdf_dir: Path = None) -> str:
    """Works for UNEP wedocs (DSpace 8) and similar — get abstract + PDF from citation_pdf_url meta."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.text, 'html.parser')
        # Try citation_abstract or DC.description
        for name in ('citation_abstract', 'DC.description', 'description'):
            meta = soup.find('meta', {'name': name})
            if meta and meta.get('content', '').strip():
                return meta['content'].strip()[:1000]
        # Try to get PDF and extract
        pdf_meta = soup.find('meta', {'name': 'citation_pdf_url'})
        if pdf_meta and pdf_meta.get('content'):
            pdf_url = pdf_meta['content']
            safe_name = re.sub(r'[^\w\-]', '_', url.split('/')[-1]) + '.pdf'
            save_path = (pdf_dir / safe_name) if pdf_dir else None
            return fetch_pdf_abstract(pdf_url, save_path)
    except Exception as e:
        print(f"    DSpace error: {e}")
    return ""


def process_3ie():
    ris_path = BASE / "3ie" / "multidatabase3ie3ie_export_1.ris"
    records = parse_ris(ris_path)
    patched = 0
    for record in records:
        if 'AB' in record:
            continue
        url = record.get('UR', '')
        if isinstance(url, list): url = url[0]
        if not url:
            continue
        title = record.get('TI', '?')
        if isinstance(title, list): title = title[0]
        print(f"  {str(title)[:60]}")
        abstract = scrape_page_abstract(url)
        if abstract:
            record['AB'] = abstract
            patched += 1
            print(f"    OK ({len(abstract)} chars)")
        else:
            print(f"    Not found")
        time.sleep(1)
    write_ris(records, ris_path)
    return patched


def process_campbell():
    ris_path = BASE / "campbell" / "multidatabasecampbellcampbell_export_1.ris"
    records = parse_ris(ris_path)
    patched = 0
    for record in records:
        if 'AB' in record:
            continue
        url = record.get('UR', '')
        if isinstance(url, list): url = url[0]
        title = record.get('TI', '?')
        if isinstance(title, list): title = title[0]
        # Skip bad records (Google search URLs)
        if not url or 'google.com/search' in url:
            print(f"  Skipping bad record: {str(title)[:60]}")
            continue
        print(f"  {str(title)[:60]}")
        abstract = scrape_page_abstract(url)
        if abstract:
            record['AB'] = abstract
            patched += 1
            print(f"    OK ({len(abstract)} chars)")
        else:
            print(f"    Not found")
        time.sleep(1)
    write_ris(records, ris_path)
    return patched


def process_unep():
    ris_path = BASE / "unep" / "multidatabaseunepunep_export_1.ris"
    records = parse_ris(ris_path)
    patched = 0
    pdf_dir = BASE / "unep"
    for record in records:
        if 'AB' in record:
            continue
        url = record.get('UR', '')
        if isinstance(url, list): url = url[0]
        if not url:
            continue
        title = record.get('TI', '?')
        if isinstance(title, list): title = title[0]
        print(f"  {str(title)[:60]}")
        abstract = fetch_dspace_abstract(url, pdf_dir=pdf_dir)
        if abstract:
            record['AB'] = abstract
            patched += 1
            print(f"    OK ({len(abstract)} chars)")
        else:
            print(f"    Not found")
        time.sleep(1)
    write_ris(records, ris_path)
    return patched


def process_unfccc():
    ris_path = BASE / "unfccc" / "multidatabaseunfcccunfccc_export_1.ris"
    records = parse_ris(ris_path)
    patched = 0
    pdf_dir = BASE / "unfccc"
    for record in records:
        if 'AB' in record:
            continue
        # Try L1 PDF first, then UR
        url = record.get('L1', '')
        if isinstance(url, list): url = url[0]
        if not url or not url.endswith('.pdf'):
            url = record.get('UR', '')
            if isinstance(url, list): url = url[0]
        if not url:
            continue
        title = record.get('TI', '?')
        if isinstance(title, list): title = title[0]
        print(f"  {str(title)[:60]}")
        safe_name = re.sub(r'[^\w\-]', '_', str(title)[:50]) + '.pdf'
        save_path = pdf_dir / safe_name
        abstract = fetch_pdf_abstract(url, save_path=save_path if not save_path.exists() else None)
        if abstract:
            record['AB'] = abstract
            patched += 1
            print(f"    OK ({len(abstract)} chars)")
        else:
            print(f"    Not found")
        time.sleep(1)
    write_ris(records, ris_path)
    return patched


def process_wos():
    patched = 0
    for ris_path in sorted((BASE / "wos").glob("*.ris")):
        records = parse_ris(ris_path)
        changed = False
        for record in records:
            if 'AB' in record:
                continue
            # WoS records with no abstract and no URL — try DOI lookup via CrossRef
            doi = record.get('DO', '')
            if isinstance(doi, list): doi = doi[0]
            title = record.get('TI', '?')
            if isinstance(title, list): title = title[0]
            print(f"  {str(title)[:60]}")
            abstract = ""
            if doi:
                try:
                    r = requests.get(f"https://api.crossref.org/works/{doi}", timeout=15)
                    if r.status_code == 200:
                        data = r.json()
                        ab_list = data.get('message', {}).get('abstract', '')
                        if ab_list:
                            abstract = re.sub(r'<[^>]+>', '', ab_list).strip()[:1000]
                except Exception as e:
                    print(f"    CrossRef error: {e}")
            if abstract:
                record['AB'] = abstract
                patched += 1
                changed = True
                print(f"    OK ({len(abstract)} chars)")
            else:
                print(f"    Not found")
            time.sleep(0.5)
        if changed:
            write_ris(records, ris_path)
    return patched


def process_econlit():
    ris_path = BASE / "econlit" / "multidatabaseeconliteconlit_export.ris"
    records = parse_ris(ris_path)
    patched = 0
    for record in records:
        if 'AB' in record:
            continue
        doi = record.get('DO', '')
        if isinstance(doi, list): doi = doi[0]
        title = record.get('TI', '?')
        if isinstance(title, list): title = title[0]
        print(f"  {str(title)[:60]}")
        abstract = ""
        if doi:
            try:
                r = requests.get(f"https://api.crossref.org/works/{doi}", timeout=15)
                if r.status_code == 200:
                    ab = r.json().get('message', {}).get('abstract', '')
                    if ab:
                        abstract = re.sub(r'<[^>]+>', '', ab).strip()[:1000]
            except Exception as e:
                print(f"    CrossRef error: {e}")
        if abstract:
            record['AB'] = abstract
            patched += 1
            print(f"    OK ({len(abstract)} chars)")
        else:
            print(f"    Not found")
        time.sleep(0.5)
    write_ris(records, ris_path)
    return patched


def process_proq():
    ris_path = BASE / "proq" / "multidatabaseproqproq_export.RIS"
    records = parse_ris(ris_path)
    patched = 0
    for record in records:
        if 'AB' in record:
            continue
        doi = record.get('DO', '')
        if isinstance(doi, list): doi = doi[0]
        title = record.get('TI', '?')
        if isinstance(title, list): title = title[0]
        print(f"  {str(title)[:60]}")
        abstract = ""
        if doi:
            try:
                r = requests.get(f"https://api.crossref.org/works/{doi}", timeout=15)
                if r.status_code == 200:
                    ab = r.json().get('message', {}).get('abstract', '')
                    if ab:
                        abstract = re.sub(r'<[^>]+>', '', ab).strip()[:1000]
            except Exception as e:
                print(f"    CrossRef error: {e}")
        if abstract:
            record['AB'] = abstract
            patched += 1
            print(f"    OK ({len(abstract)} chars)")
        else:
            print(f"    Not found")
        time.sleep(0.5)
    write_ris(records, ris_path)
    return patched


def main():
    print("\n[3ie] Fetching missing abstracts...")
    n = process_3ie()
    print(f"  => {n} added")

    print("\n[Campbell] Fetching missing abstracts...")
    n = process_campbell()
    print(f"  => {n} added")

    print("\n[UNEP] Fetching missing abstracts...")
    n = process_unep()
    print(f"  => {n} added")

    print("\n[UNFCCC] Fetching missing abstracts...")
    n = process_unfccc()
    print(f"  => {n} added")

    print("\n[WoS] Fetching missing abstracts via CrossRef...")
    n = process_wos()
    print(f"  => {n} added")

    print("\n[EconLit] Fetching missing abstracts via CrossRef...")
    n = process_econlit()
    print(f"  => {n} added")

    print("\n[ProQuest] Fetching missing abstracts via CrossRef...")
    n = process_proq()
    print(f"  => {n} added")


if __name__ == "__main__":
    main()
