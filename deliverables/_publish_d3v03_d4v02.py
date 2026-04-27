"""
_publish_d3v03_d4v02.py

1. Copy D4 v01 → D4 v02 in Google Drive
2. Copy D3 v02 → D3 v03 in Google Drive
3. Export both as PDFs to deliverables/
4. Publish D4 v02 to Zenodo (new record)
5. Publish D3 v03 to Zenodo (new version of record 18370029)

Usage:
  conda run -n ilri01 python deliverables/_publish_d3v03_d4v02.py             # copy + export only
  conda run -n ilri01 python deliverables/_publish_d3v03_d4v02.py --confirm   # full publish
"""
from __future__ import annotations

import argparse
import io
import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

load_dotenv()

ROOT      = Path(__file__).resolve().parent.parent
CREDS_DIR = Path(__file__).resolve().parent / ".credentials"
SCOPES    = [
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/drive",
]

DELIVERABLES_FOLDER = "1alv6Ic7vtKZc_xo6AdlKP1exshMh-fR6"
D4_V01_ID           = "1i2dQUoXuNPwjvV_fT5CMiy0fvADy8oWCGK3sdHGApCs"
D3_V02_ID           = "1yLnB2b--XOtrMSQ1ekgBu6XKGcDWQ1Ggvn7mKQqjaqg"
D3_ZENODO_RECORD    = "18370029"

D4_V02_NAME = "Deliverable 4_Bristlepine_First Draft Systematic Map_v02"
D3_V03_NAME = "Deliverable 3_Bristlepine_Final Systematic Map Protocol_v03"

D4_PDF = ROOT / "deliverables" / "Deliverable 4_Bristlepine_First Draft Systematic Map_v02.pdf"
D3_PDF = ROOT / "deliverables" / "Deliverable 3_Bristlepine_Final Systematic Map Protocol_v03.pdf"

ZENODO_BASE  = "https://zenodo.org/api"
ZENODO_TOKEN = os.getenv("ZENODO_TOKEN", "")

AUTHORS = [
    {"name": "Denno Cissé, Jennifer", "affiliation": "Bristlepine Resilience Consultants"},
    {"name": "Khan, Zarrar",          "affiliation": "Bristlepine Resilience Consultants"},
    {"name": "Staub, Caroline G.",    "affiliation": "Bristlepine Resilience Consultants"},
]

KEYWORDS = [
    "climate adaptation", "systematic map", "smallholder farmers",
    "adaptation outcomes", "adaptation processes", "agriculture", "LMICs",
    "monitoring and evaluation", "evidence synthesis",
]


# ── Google helpers ─────────────────────────────────────────────────────────────

def get_creds() -> Credentials:
    creds = Credentials.from_authorized_user_file(str(CREDS_DIR / "token.json"), SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return creds


def copy_doc(drive, source_id: str, name: str, folder_id: str) -> str:
    result = drive.files().copy(
        fileId=source_id,
        body={"name": name, "parents": [folder_id]},
        supportsAllDrives=True,
    ).execute()
    return result["id"]


def export_pdf(drive, file_id: str, dest: Path) -> None:
    request = drive.files().export_media(fileId=file_id, mimeType="application/pdf")
    buf = io.BytesIO()
    dl = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = dl.next_chunk()
    dest.write_bytes(buf.getvalue())
    print(f"  Exported {dest.name} ({dest.stat().st_size / 1024:.0f} KB)")


def upload_pdf_to_drive(drive, local_path: Path, drive_name: str, folder_id: str) -> str:
    from googleapiclient.http import MediaFileUpload
    result = drive.files().create(
        body={"name": drive_name, "parents": [folder_id]},
        media_body=MediaFileUpload(str(local_path), mimetype="application/pdf", resumable=False),
        supportsAllDrives=True,
        fields="id",
    ).execute()
    fid = result["id"]
    print(f"  Uploaded {drive_name} → https://drive.google.com/file/d/{fid}/view")
    return fid


# ── Zenodo helpers ─────────────────────────────────────────────────────────────

def _zh() -> dict:
    if not ZENODO_TOKEN:
        raise SystemExit("ZENODO_TOKEN not set — add it to your .env file.")
    return {"Authorization": f"Bearer {ZENODO_TOKEN}"}


def _zcheck(r: requests.Response, ctx: str) -> dict:
    if not r.ok:
        print(f"[zenodo] ERROR {ctx}: HTTP {r.status_code}")
        try:
            print(json.dumps(r.json(), indent=2))
        except Exception:
            print(r.text[:500])
        raise SystemExit(1)
    return r.json()


def zenodo_publish_new(path: Path, title: str, version: str) -> str:
    dep = _zcheck(
        requests.post(f"{ZENODO_BASE}/deposit/depositions", headers=_zh(), json={}),
        "create deposition",
    )
    dep_id = dep["id"]
    _upload(dep_id, path)
    _set_meta(dep_id, title, version)
    result = _zcheck(
        requests.post(f"{ZENODO_BASE}/deposit/depositions/{dep_id}/actions/publish", headers=_zh()),
        "publish",
    )
    return result.get("doi") or result.get("metadata", {}).get("doi", "")


def zenodo_publish_new_version(path: Path, title: str, version: str, record_id: str) -> str:
    data = _zcheck(
        requests.post(
            f"{ZENODO_BASE}/deposit/depositions/{record_id}/actions/newversion",
            headers=_zh(),
        ),
        "new version",
    )
    draft = _zcheck(requests.get(data["links"]["latest_draft"], headers=_zh()), "get draft")
    dep_id = draft["id"]
    for f in draft.get("files", []):
        requests.delete(
            f"{ZENODO_BASE}/deposit/depositions/{dep_id}/files/{f['id']}",
            headers=_zh(),
        )
    _upload(dep_id, path)
    _set_meta(dep_id, title, version)
    result = _zcheck(
        requests.post(f"{ZENODO_BASE}/deposit/depositions/{dep_id}/actions/publish", headers=_zh()),
        "publish",
    )
    return result.get("doi") or result.get("metadata", {}).get("doi", "")


def _upload(dep_id: int, path: Path) -> None:
    with open(path, "rb") as fh:
        r = requests.post(
            f"{ZENODO_BASE}/deposit/depositions/{dep_id}/files",
            headers=_zh(),
            data={"name": path.name},
            files={"file": fh},
        )
    _zcheck(r, f"upload {path.name}")


def _set_meta(dep_id: int, title: str, version: str) -> None:
    r = requests.put(
        f"{ZENODO_BASE}/deposit/depositions/{dep_id}",
        headers={**_zh(), "Content-Type": "application/json"},
        json={"metadata": {
            "title":            title,
            "upload_type":      "publication",
            "publication_type": "report",
            "access_right":     "open",
            "license":          "cc-by-4.0",
            "creators":         AUTHORS,
            "keywords":         KEYWORDS,
            "version":          version,
        }},
    )
    _zcheck(r, "set metadata")


# ── Main ───────────────────────────────────────────────────────────────────────

def bump_version(docs, doc_id: str, old: str, new: str) -> None:
    docs.documents().batchUpdate(
        documentId=doc_id,
        body={"requests": [{"replaceAllText": {
            "containsText": {"text": old, "matchCase": True},
            "replaceText": new,
        }}]},
    ).execute()
    print(f"  Replaced '{old}' → '{new}'")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", action="store_true",
                    help="Actually publish to Zenodo (default: copy + export only)")
    args = ap.parse_args()

    creds = get_creds()
    drive = build("drive", "v3", credentials=creds)
    docs  = build("docs",  "v1", credentials=creds)

    # Step 1: copy docs in Drive
    print("Step 1: Copying docs ...")
    print(f"  D4 v01 → {D4_V02_NAME}")
    d4_v02_id = copy_doc(drive, D4_V01_ID, D4_V02_NAME, DELIVERABLES_FOLDER)
    print(f"  https://docs.google.com/document/d/{d4_v02_id}/edit")
    time.sleep(3)

    print(f"  D3 v02 → {D3_V03_NAME}")
    d3_v03_id = copy_doc(drive, D3_V02_ID, D3_V03_NAME, DELIVERABLES_FOLDER)
    print(f"  https://docs.google.com/document/d/{d3_v03_id}/edit")
    time.sleep(3)

    # Step 2: bump version numbers inside the docs
    print("\nStep 2: Updating version numbers in docs ...")
    bump_version(docs, d4_v02_id, "April 2026, v01", "April 2026, v02")
    bump_version(docs, d3_v03_id, ", v02",           ", v03")
    time.sleep(2)

    # Step 3: export PDFs and upload to Drive
    print("\nStep 3: Exporting PDFs and uploading to Drive ...")
    export_pdf(drive, d4_v02_id, D4_PDF)
    upload_pdf_to_drive(drive, D4_PDF, D4_V02_NAME + ".pdf", DELIVERABLES_FOLDER)

    export_pdf(drive, d3_v03_id, D3_PDF)
    upload_pdf_to_drive(drive, D3_PDF, D3_V03_NAME + ".pdf", DELIVERABLES_FOLDER)

    if not args.confirm:
        print("\nDRY RUN — PDFs exported, Zenodo skipped. Pass --confirm to publish.")
        print(f"  D4 v02 GDoc : https://docs.google.com/document/d/{d4_v02_id}/edit")
        print(f"  D3 v03 GDoc : https://docs.google.com/document/d/{d3_v03_id}/edit")
        return

    # Step 4: publish to Zenodo
    print("\nStep 4: Publishing to Zenodo ...")

    print("  D4 v02 → new Zenodo record ...")
    d4_doi = zenodo_publish_new(
        D4_PDF,
        "Deliverable 4: First Draft Systematic Map (Preliminary) — "
        "Measuring what matters: tracking climate adaptation processes and outcomes "
        "for smallholder producers in the agriculture sector",
        "v02",
    )
    print(f"  D4 v02 DOI: {d4_doi}")
    time.sleep(1)

    print(f"  D3 v03 → new version of record {D3_ZENODO_RECORD} ...")
    d3_doi = zenodo_publish_new_version(
        D3_PDF,
        "Deliverable 3: Final Systematic Map Protocol (v03, amended) — "
        "Measuring what matters: tracking climate adaptation processes and outcomes "
        "for smallholder producers in the agriculture sector",
        "v03",
        D3_ZENODO_RECORD,
    )
    print(f"  D3 v03 DOI: {d3_doi}")

    print("\n" + "─" * 60)
    print("Done! Update these in frontend/src/lib/deliverables.ts and deliverables_tracker.md:")
    print(f"  D4 v02 GDoc : https://docs.google.com/document/d/{d4_v02_id}/edit")
    print(f"  D3 v03 GDoc : https://docs.google.com/document/d/{d3_v03_id}/edit")
    print(f"  D4 v02 DOI  : https://doi.org/{d4_doi}")
    print(f"  D3 v03 DOI  : https://doi.org/{d3_doi}")


if __name__ == "__main__":
    main()
