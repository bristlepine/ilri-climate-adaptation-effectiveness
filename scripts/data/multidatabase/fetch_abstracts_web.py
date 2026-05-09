"""
Fetch abstracts from the web for World Bank and IDB records that are missing them.

World Bank: uses the handle.net URL to fetch metadata via OAI-PMH API
IDB: downloads PDF from URL and extracts abstract via pdftotext

Usage:
    python fetch_abstracts_web.py
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
    "User-Agent": "Mozilla/5.0 (research project; abstract retrieval)"
}


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


def fetch_worldbank_abstract(url: str, pdf_dir: Path = None) -> str:
    """Fetch abstract from World Bank Open Knowledge via DSpace 7 REST API."""
    abstract = ""
    try:
        handle_match = re.search(r'(?:handle\.net|openknowledge\.worldbank\.org/handle)/(.+)', url)
        if not handle_match:
            return ""
        handle = handle_match.group(1).strip()

        # Use DSpace 7 search API to find item by handle
        encoded_handle = handle.replace("/", "%2F")
        search_url = f"https://openknowledge.worldbank.org/server/api/discover/search/objects?query={encoded_handle}&dsoType=item"
        r = requests.get(search_url, headers=HEADERS, timeout=20)
        data = r.json()
        objects = data.get('_embedded', {}).get('searchResult', {}).get('_embedded', {}).get('objects', [])

        item_uuid = None
        for obj in objects:
            embedded = obj.get('_embedded', {}).get('indexableObject', {})
            if embedded.get('handle') == handle:
                item_uuid = embedded.get('uuid')
                # Try to get abstract from metadata in search result
                meta = embedded.get('metadata', {})
                abs_list = meta.get('dc.description.abstract', [])
                if abs_list:
                    abstract = abs_list[0].get('value', '').strip()[:1000]
                break

        # If no abstract yet, fetch item directly
        if not abstract and item_uuid:
            r = requests.get(
                f"https://openknowledge.worldbank.org/server/api/core/items/{item_uuid}",
                headers=HEADERS, timeout=20
            )
            item_data = r.json()
            meta = item_data.get('metadata', {})
            abs_list = meta.get('dc.description.abstract', [])
            if abs_list:
                abstract = abs_list[0].get('value', '').strip()[:1000]

        # Try to download PDF via bundles API
        if pdf_dir and item_uuid:
            try:
                r = requests.get(
                    f"https://openknowledge.worldbank.org/server/api/core/items/{item_uuid}/bundles",
                    headers=HEADERS, timeout=20
                )
                bundles = r.json().get('_embedded', {}).get('bundles', [])
                for bundle in bundles:
                    if bundle.get('name') == 'ORIGINAL':
                        bs_url = bundle['_links']['bitstreams']['href']
                        r2 = requests.get(bs_url, headers=HEADERS, timeout=20)
                        bitstreams = r2.json().get('_embedded', {}).get('bitstreams', [])
                        for bs in bitstreams:
                            if bs.get('name', '').lower().endswith('.pdf'):
                                pdf_url = bs['_links']['content']['href']
                                safe_name = re.sub(r'[^\w\-]', '_', handle.replace('/', '_')) + '.pdf'
                                save_path = pdf_dir / safe_name
                                if not save_path.exists():
                                    fetched = fetch_pdf_and_extract(pdf_url, save_path)
                                    if fetched and not abstract:
                                        abstract = fetched
                                break
                        break
            except Exception as e:
                print(f"    Error fetching WB PDF: {e}")

    except Exception as e:
        print(f"    Error fetching World Bank abstract: {e}")
    return abstract


def fetch_pdf_and_extract(url: str, save_path: Path = None) -> str:
    """Download PDF, optionally save it, and extract abstract."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=30, stream=True)
        if r.status_code != 200:
            return ""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
            tmp_path = f.name

        # Save PDF if path provided
        if save_path:
            import shutil
            save_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(tmp_path, save_path)
            print(f"    Saved PDF: {save_path.name}")

        result = subprocess.run(
            ["pdftotext", tmp_path, "-"],
            capture_output=True, text=True, timeout=30
        )
        Path(tmp_path).unlink(missing_ok=True)
        return extract_abstract_from_text(result.stdout)

    except Exception as e:
        print(f"    Error fetching PDF: {e}")
    return ""


def fetch_idb_abstract(url: str, save_path: Path = None) -> str:
    """Download IDB PDF and extract abstract."""
    return fetch_pdf_and_extract(url, save_path)


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


def process_worldbank():
    ris_path = BASE / "worldbank" / "multidatabaseworldbankworldbank_export_1.ris"
    records = parse_ris(ris_path)
    patched = 0

    for record in records:
        if 'AB' in record:
            continue
        url = record.get('UR', '')
        if isinstance(url, list):
            url = url[0]
        if not url:
            continue

        title = record.get('TI', 'Unknown')
        if isinstance(title, list):
            title = title[0]
        print(f"  Fetching: {title[:60]}")

        abstract = fetch_worldbank_abstract(url, pdf_dir=BASE / "worldbank")
        if abstract:
            record['AB'] = abstract
            patched += 1
            print(f"    OK ({len(abstract)} chars)")
        else:
            print(f"    No abstract found")
        time.sleep(1)

    write_ris(records, ris_path)
    return patched


def process_idb():
    ris_path = BASE / "idb" / "multidatabaseidbidb_export_1.ris"
    records = parse_ris(ris_path)
    patched = 0

    for record in records:
        if 'AB' in record:
            continue
        # Try L1 first (direct PDF URL), then UR
        url = record.get('L1', record.get('UR', ''))
        if isinstance(url, list):
            url = url[0]
        if not url or not url.endswith('.pdf'):
            # Try UR
            url = record.get('UR', '')
            if isinstance(url, list):
                url = url[0]

        if not url:
            continue

        title = record.get('TI', 'Unknown')
        if isinstance(title, list):
            title = title[0]
        print(f"  Fetching: {title[:60]}")

        # Generate safe PDF filename from title
        safe_title = re.sub(r'[^\w\-]', '_', title[:50])
        pdf_save_path = BASE / "idb" / f"{safe_title}.pdf"
        abstract = fetch_idb_abstract(url, save_path=pdf_save_path) if url.endswith('.pdf') else ""

        # If no PDF URL, try scraping the page
        if not abstract and url:
            try:
                r = requests.get(url, headers=HEADERS, timeout=15)
                soup = BeautifulSoup(r.text, 'html.parser')
                meta = soup.find('meta', {'name': 'description'}) or \
                       soup.find('meta', {'property': 'og:description'})
                if meta and meta.get('content', '').strip():
                    abstract = meta['content'].strip()[:1000]
            except Exception:
                pass

        if abstract:
            record['AB'] = abstract
            patched += 1
            print(f"    OK ({len(abstract)} chars)")
        else:
            print(f"    No abstract found")
        time.sleep(1)

    write_ris(records, ris_path)
    return patched


def main():
    print("\n[World Bank] Fetching abstracts...")
    wb_patched = process_worldbank()
    print(f"  => {wb_patched} abstracts added")

    print("\n[IDB] Fetching abstracts...")
    idb_patched = process_idb()
    print(f"  => {idb_patched} abstracts added")

    print(f"\nDone. Total: {wb_patched + idb_patched} abstracts fetched")


if __name__ == "__main__":
    main()
