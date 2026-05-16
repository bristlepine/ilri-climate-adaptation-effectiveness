#!/usr/bin/env python3
"""
push_codebook_update.py

Upload updated codebook PDFs to their existing Google Drive round folders.

Usage:
    # Step 1 — regenerate PDFs from markdown (inspect them before pushing):
    python documentation/coding/systematic-map/generate_codebook_pdfs.py

    # Step 2 — push to Drive once you're happy with the PDFs:
    python documentation/coding/systematic-map/push_codebook_update.py

For each round, the script finds the existing round folder on Drive (child of
DRIVE_PARENT_ID) and either updates the existing codebook file in-place or
uploads a fresh copy if none exists yet.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT       = Path(__file__).resolve().parent.parent.parent.parent
ROUNDS_DIR      = REPO_ROOT / "documentation" / "coding" / "systematic-map" / "rounds"
CREDS_DIR       = REPO_ROOT / "deliverables" / ".credentials"
DRIVE_PARENT_ID = "13p22XfvB6sNtTtnMS-dkI1t-joMn-6Bo"

ROUNDS = ["FT-R1a", "FT-R2a"]


# ── Drive helpers ──────────────────────────────────────────────────────────────

def gdrive_auth():
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request

    token_file = CREDS_DIR / "token.json"
    scopes = ["https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_authorized_user_file(str(token_file), scopes)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return creds


def find_folder(service, name: str, parent_id: str) -> str | None:
    """Return the Drive file ID of a folder with the given name under parent_id, or None."""
    resp = service.files().list(
        q=f"name='{name}' and '{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false",
        fields="files(id, name)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()
    files = resp.get("files", [])
    return files[0]["id"] if files else None


def find_file(service, name: str, parent_id: str) -> str | None:
    """Return the Drive file ID of a file with the given name under parent_id, or None."""
    resp = service.files().list(
        q=f"name='{name}' and '{parent_id}' in parents and trashed=false",
        fields="files(id, name)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()
    files = resp.get("files", [])
    return files[0]["id"] if files else None


def update_file(service, file_id: str, local_path: Path) -> None:
    """Replace the content of an existing Drive file."""
    from googleapiclient.http import MediaFileUpload
    media = MediaFileUpload(str(local_path), mimetype="application/pdf", resumable=True)
    service.files().update(
        fileId=file_id,
        media_body=media,
        supportsAllDrives=True,
    ).execute()


def upload_file(service, local_path: Path, drive_name: str, parent_id: str) -> str:
    """Upload a new file to Drive."""
    from googleapiclient.http import MediaFileUpload
    meta = {"name": drive_name, "parents": [parent_id]}
    media = MediaFileUpload(str(local_path), mimetype="application/pdf", resumable=True)
    f = service.files().create(
        body=meta, media_body=media, fields="id",
        supportsAllDrives=True,
    ).execute()
    return f["id"]


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    from googleapiclient.discovery import build

    print("Connecting to Google Drive...")
    creds   = gdrive_auth()
    service = build("drive", "v3", credentials=creds)

    for round_name in ROUNDS:
        pdf = ROUNDS_DIR / round_name / f"CODEBOOK_{round_name}.pdf"

        if not pdf.exists():
            print(f"  {round_name}: PDF not found at {pdf} — run generate_codebook_pdfs.py first")
            continue

        size_kb = pdf.stat().st_size // 1024
        print(f"\n  {round_name}: {pdf.name}  ({size_kb} KB)")

        # Find round folder on Drive
        folder_id = find_folder(service, round_name, DRIVE_PARENT_ID)
        if not folder_id:
            print(f"    WARNING: Drive folder '{round_name}' not found under parent — skipping")
            continue

        # Check for existing codebook file
        existing_id = find_file(service, pdf.name, folder_id)
        if existing_id:
            print(f"    Updating existing file (id={existing_id})...", end=" ", flush=True)
            update_file(service, existing_id, pdf)
            print("done")
        else:
            print(f"    Uploading new file...", end=" ", flush=True)
            new_id = upload_file(service, pdf, pdf.name, folder_id)
            print(f"done (id={new_id})")

    print("\nAll done.")


if __name__ == "__main__":
    main()
