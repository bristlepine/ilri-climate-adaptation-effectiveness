"""
setup_round.py — Create a new human coding round folder and upload to Google Drive.

Samples N papers from the coded pool (PDFs only), writes a blank coding template,
and uploads everything to the shared Google Drive coding folder.

Usage:
  # Create FT-R2a (first production batch, 100 papers, seed 42)
  conda run -n ilri01 python documentation/coding/systematic-map/rounds/setup_round.py \\
    --round FT-R2a --sample 100 --seed 42

  # Create FT-R2b (second batch, different seed so no overlap)
  conda run -n ilri01 python documentation/coding/systematic-map/rounds/setup_round.py \\
    --round FT-R2b --sample 100 --seed 43

Arguments:
  --round     Round name, e.g. FT-R2a  (also used as Drive subfolder name)
  --sample    Number of papers to sample (default 100)
  --seed      Random seed (default 42; increment per batch to avoid overlap)
  --dry-run   Print what would happen without uploading to Drive
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

HERE      = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent.parent.parent

CODED_CSV    = REPO_ROOT / "scripts" / "outputs" / "step15" / "step15_coded.csv"
MANIFEST_CSV = REPO_ROOT / "scripts" / "outputs" / "step13" / "step13_manifest.csv"
CODEBOOK_PDF = (REPO_ROOT / "documentation" / "coding" / "systematic-map"
                / "Systematic Map Codebook — D5.6 (for comments).pdf")
CREDS_DIR    = REPO_ROOT / "deliverables" / ".credentials"

# Google Drive parent folder (contains FT-R1a, FT-R2a, ...)
DRIVE_PARENT_ID = "13p22XfvB6sNtTtnMS-dkI1t-joMn-6Bo"

# Papers used in calibration — never re-sample these
CALIBRATION_DOIS = {
    "10.1016/j.agee.2019.04.004",
    "10.1016/j.crm.2017.06.001",
    "10.1016/j.crm.2017.03.001",
    "10.1007/s10584-016-1792-0",
    "10.1080/17565529.2017.1411240",
}

TEMPLATE_FIELDS = [
    "publication_year", "publication_type", "country_region", "geographic_scale",
    "producer_type", "marginalized_subpopulations", "adaptation_focus",
    "process_outcome_domains", "indicators_measured", "methodological_approach",
    "purpose_of_assessment", "data_sources", "temporal_coverage",
    "cost_data_reported", "strengths_and_limitations", "lessons_learned",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def load_pool() -> pd.DataFrame:
    """Return coded records that have a retrievable PDF, excluding calibration papers."""
    coded = pd.read_csv(CODED_CSV, usecols=["doi", "title", "year"])
    manifest = pd.read_csv(MANIFEST_CSV, usecols=["doi", "file_path"])
    manifest = manifest[manifest["file_path"].str.endswith(".pdf", na=False)]
    pool = coded.merge(manifest, on="doi", how="inner")
    pool = pool[~pool["doi"].isin(CALIBRATION_DOIS)].drop_duplicates("doi")
    return pool.reset_index(drop=True)


def load_already_sampled(exclude_round: str | None = None) -> set[str]:
    """Collect DOIs already assigned in previous rounds to avoid overlap."""
    sampled = set()
    for papers_csv in HERE.glob("*/papers_*.csv"):
        round_name = papers_csv.parent.name
        if exclude_round and round_name == exclude_round:
            continue
        df = pd.read_csv(papers_csv, usecols=["doi"])
        sampled.update(df["doi"].tolist())
    return sampled


def make_template(sample: pd.DataFrame) -> pd.DataFrame:
    """Blank coding template with doi/title/year pre-filled."""
    tpl = sample[["doi", "title", "year"]].copy()
    for field in TEMPLATE_FIELDS:
        tpl[field] = ""
    tpl["coder_id"] = ""
    tpl["notes"] = ""
    return tpl


def gdrive_auth():
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request

    token_file = CREDS_DIR / "token.json"
    scopes = ["https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_authorized_user_file(str(token_file), scopes)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return creds


def create_drive_folder(service, name: str, parent_id: str) -> str:
    meta = {"name": name, "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_id]}
    f = service.files().create(
        body=meta, fields="id",
        supportsAllDrives=True,
    ).execute()
    return f["id"]


def upload_file(service, local_path: Path, drive_name: str, parent_id: str) -> str:
    from googleapiclient.http import MediaFileUpload
    import mimetypes
    mime, _ = mimetypes.guess_type(str(local_path))
    mime = mime or "application/octet-stream"
    meta = {"name": drive_name, "parents": [parent_id]}
    media = MediaFileUpload(str(local_path), mimetype=mime, resumable=True)
    f = service.files().create(
        body=meta, media_body=media, fields="id",
        supportsAllDrives=True,
    ).execute()
    return f["id"]


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round",   required=True, help="Round name, e.g. FT-R2a")
    parser.add_argument("--sample",  type=int, default=100)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    round_name = args.round
    round_dir  = HERE / round_name
    round_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Setting up {round_name}")
    print(f"  Sample: {args.sample}   Seed: {args.seed}   Dry-run: {args.dry_run}")
    print(f"{'='*60}\n")

    # ── 1. Sample papers ──────────────────────────────────────────────────────
    pool = load_pool()
    already = load_already_sampled(exclude_round=round_name)
    available = pool[~pool["doi"].isin(already)]
    print(f"Pool: {len(pool)} coded PDFs  |  Already assigned: {len(already)}  |  Available: {len(available)}")

    if len(available) < args.sample:
        print(f"ERROR: only {len(available)} papers available, need {args.sample}")
        sys.exit(1)

    sample = available.sample(n=args.sample, random_state=args.seed).sort_values("doi")
    print(f"Sampled {len(sample)} papers (seed={args.seed})\n")

    # ── 2. Write local files ──────────────────────────────────────────────────
    papers_csv = round_dir / f"papers_{round_name.lower()}.csv"
    template_csv = round_dir / f"coding_{round_name.lower()}_template.csv"

    sample[["doi", "title", "year", "file_path"]].to_csv(papers_csv, index=False)
    make_template(sample).to_csv(template_csv, index=False)
    print(f"Written: {papers_csv.name}")
    print(f"Written: {template_csv.name}")

    if args.dry_run:
        print("\n[dry-run] Skipping Drive upload.")
        print("Sample head:")
        print(sample[["doi", "title"]].head(5).to_string(index=False))
        return

    # ── 3. Upload to Google Drive ─────────────────────────────────────────────
    from googleapiclient.discovery import build

    print("\nConnecting to Google Drive...")
    creds   = gdrive_auth()
    service = build("drive", "v3", credentials=creds)

    # Create round folder + pdfs subfolder
    print(f"Creating Drive folder: {round_name}")
    round_folder_id = create_drive_folder(service, round_name, DRIVE_PARENT_ID)
    pdfs_folder_name = f"{round_name} pdfs"
    print(f"Creating Drive subfolder: {pdfs_folder_name}")
    pdfs_folder_id = create_drive_folder(service, pdfs_folder_name, round_folder_id)

    # Upload codebook
    print(f"Uploading codebook: {CODEBOOK_PDF.name}")
    upload_file(service, CODEBOOK_PDF, CODEBOOK_PDF.name, round_folder_id)

    # Upload template
    print(f"Uploading template: {template_csv.name}")
    upload_file(service, template_csv, template_csv.name, round_folder_id)

    # Upload PDFs
    print(f"\nUploading {len(sample)} PDFs to '{pdfs_folder_name}'...")
    errors = []
    for i, (_, row) in enumerate(sample.iterrows(), 1):
        pdf_path = Path(row["file_path"])
        if not pdf_path.exists():
            print(f"  [{i:3d}/{len(sample)}] MISSING: {pdf_path.name}")
            errors.append(row["doi"])
            continue
        print(f"  [{i:3d}/{len(sample)}] {pdf_path.name}", end=" ", flush=True)
        upload_file(service, pdf_path, pdf_path.name, pdfs_folder_id)
        print("✓")

    print(f"\n{'='*60}")
    print(f"  Done: {round_name}")
    print(f"  Drive folder created (ID: {round_folder_id})")
    print(f"  PDFs uploaded: {len(sample) - len(errors)}/{len(sample)}")
    if errors:
        print(f"  Missing PDFs: {errors}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
