"""
zenodo_publish.py

Publish a deliverable to Zenodo via the REST API.

Supports:
  - Creating a new Zenodo deposition (new DOI)
  - Creating a new version of an existing record (e.g. D3 amendment → D5.7)
  - Uploading PDF/CSV/ZIP files
  - Setting metadata (title, authors, description, keywords, access)
  - Publishing and returning the DOI

Requires:
  - ZENODO_TOKEN in environment or .env file (deposit:write + deposit:actions scope)
  - Get token: https://zenodo.org/account/settings/applications

Usage:
  python deliverables/zenodo_publish.py --help
  python deliverables/zenodo_publish.py --file deliverables/D4_v01.pdf --title "Deliverable 4..."
  python deliverables/zenodo_publish.py --new-version RECORD_ID --file ...  # for D5.7 amendment

Never publishes without --confirm flag (dry-run by default).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

ZENODO_BASE    = "https://zenodo.org/api"
ZENODO_TOKEN   = os.getenv("ZENODO_TOKEN", "")

# Bristlepine standard authors (reuse across depositions)
DEFAULT_AUTHORS = [
    {"name": "Khan, Zarrar",        "affiliation": "Bristlepine Resilience Consultants"},
    {"name": "Denno Cissé, Jennifer","affiliation": "Bristlepine Resilience Consultants"},
    {"name": "Staub, Caroline G.",   "affiliation": "Bristlepine Resilience Consultants"},
]

DEFAULT_KEYWORDS = [
    "climate adaptation", "systematic map", "smallholder farmers",
    "adaptation outcomes", "adaptation processes", "agriculture", "LMICs",
    "monitoring and evaluation", "evidence synthesis",
]

DEFAULT_COMMUNITIES = [
    # {"identifier": "cgiar"},  # uncomment if submitting to CGIAR community
]


def _headers() -> dict:
    if not ZENODO_TOKEN:
        raise SystemExit(
            "ZENODO_TOKEN not set.\n"
            "Generate one at: https://zenodo.org/account/settings/applications\n"
            "Then add ZENODO_TOKEN=your_token to your .env file."
        )
    return {"Authorization": f"Bearer {ZENODO_TOKEN}"}


def _check(r: requests.Response, context: str) -> dict:
    if not r.ok:
        print(f"[zenodo] ERROR — {context}: HTTP {r.status_code}")
        try:
            print(json.dumps(r.json(), indent=2))
        except Exception:
            print(r.text[:500])
        raise SystemExit(1)
    return r.json()


def create_deposition() -> dict:
    """Create an empty new deposition. Returns deposition metadata."""
    r = requests.post(f"{ZENODO_BASE}/deposit/depositions", headers=_headers(), json={})
    return _check(r, "create deposition")


def get_new_version(record_id: str) -> dict:
    """
    Create a new version of an existing published record.
    record_id: the numeric Zenodo record ID (e.g. 18370029 for D3).
    Returns the new draft deposition metadata.
    """
    r = requests.post(
        f"{ZENODO_BASE}/deposit/depositions/{record_id}/actions/newversion",
        headers=_headers(),
    )
    data = _check(r, "create new version")
    # The new version draft URL is in links.latest_draft
    draft_url = data["links"]["latest_draft"]
    r2 = requests.get(draft_url, headers=_headers())
    return _check(r2, "get new version draft")


def upload_file(deposition_id: int, file_path: Path) -> dict:
    """Upload a file to the deposition bucket."""
    bucket_url = f"{ZENODO_BASE}/deposit/depositions/{deposition_id}/files"
    with open(file_path, "rb") as fh:
        r = requests.post(
            bucket_url,
            headers=_headers(),
            data={"name": file_path.name},
            files={"file": fh},
        )
    return _check(r, f"upload {file_path.name}")


def update_metadata(deposition_id: int, metadata: dict) -> dict:
    """Set/update metadata on a deposition."""
    r = requests.put(
        f"{ZENODO_BASE}/deposit/depositions/{deposition_id}",
        headers={**_headers(), "Content-Type": "application/json"},
        json={"metadata": metadata},
    )
    return _check(r, "update metadata")


def publish(deposition_id: int) -> dict:
    """Publish the deposition. Returns the published record with DOI."""
    r = requests.post(
        f"{ZENODO_BASE}/deposit/depositions/{deposition_id}/actions/publish",
        headers=_headers(),
    )
    return _check(r, "publish")


def build_metadata(
    title: str,
    description: str,
    *,
    upload_type: str = "publication",
    publication_type: str = "report",
    authors: Optional[list] = None,
    keywords: Optional[list] = None,
    access_right: str = "open",
    license: str = "cc-by-4.0",
    communities: Optional[list] = None,
    version: Optional[str] = None,
) -> dict:
    meta = {
        "title":            title,
        "description":      description,
        "upload_type":      upload_type,
        "publication_type": publication_type,
        "access_right":     access_right,
        "license":          license,
        "creators":         authors or DEFAULT_AUTHORS,
        "keywords":         keywords or DEFAULT_KEYWORDS,
        "communities":      communities or DEFAULT_COMMUNITIES,
    }
    if version:
        meta["version"] = version
    return meta


def run(
    file_path: Path,
    title: str,
    description: str,
    *,
    new_version_of: Optional[str] = None,
    version: Optional[str] = None,
    confirm: bool = False,
) -> Optional[str]:
    """
    Main entry point.
    Returns the DOI string if published, else None (dry-run).
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"[zenodo] File     : {file_path} ({file_path.stat().st_size / 1024:.1f} KB)")
    print(f"[zenodo] Title    : {title}")
    print(f"[zenodo] Version  : {version or '(not set)'}")
    print(f"[zenodo] New ver  : {new_version_of or '(new record)'}")

    if not confirm:
        print("\n[zenodo] DRY RUN — pass --confirm to actually publish.")
        return None

    # 1. Create or draft new version
    if new_version_of:
        print(f"[zenodo] Creating new version of record {new_version_of} ...")
        dep = get_new_version(new_version_of)
    else:
        print("[zenodo] Creating new deposition ...")
        dep = create_deposition()

    dep_id = dep["id"]
    print(f"[zenodo] Deposition ID: {dep_id}")

    # 2. Upload file
    print(f"[zenodo] Uploading {file_path.name} ...")
    upload_file(dep_id, file_path)

    # 3. Set metadata
    meta = build_metadata(title, description, version=version)
    update_metadata(dep_id, meta)
    print("[zenodo] Metadata set.")

    # 4. Publish
    print("[zenodo] Publishing ...")
    result = publish(dep_id)
    doi = result.get("doi") or result.get("metadata", {}).get("doi", "")
    doi_url = f"https://doi.org/{doi}" if doi else result.get("links", {}).get("doi", "")
    print(f"\n[zenodo] Published!")
    print(f"  DOI    : {doi}")
    print(f"  URL    : {doi_url}")
    print(f"  Record : {result.get('links', {}).get('record_html', '')}")
    return doi


def main():
    ap = argparse.ArgumentParser(description="Publish a file to Zenodo")
    ap.add_argument("--file",        required=True, help="Path to file to upload (PDF, ZIP, etc.)")
    ap.add_argument("--title",       required=True, help="Deposition title")
    ap.add_argument("--description", default="",    help="HTML description")
    ap.add_argument("--version",     default=None,  help="Version string (e.g. v01)")
    ap.add_argument("--new-version", default=None,  metavar="RECORD_ID",
                    help="Create new version of existing Zenodo record ID (e.g. 18370029 for D3)")
    ap.add_argument("--confirm",     action="store_true",
                    help="Actually publish (default is dry-run)")
    args = ap.parse_args()

    run(
        file_path      = Path(args.file).resolve(),
        title          = args.title,
        description    = args.description,
        new_version_of = args.new_version,
        version        = args.version,
        confirm        = args.confirm,
    )


if __name__ == "__main__":
    main()
