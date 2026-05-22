#!/usr/bin/env python3
"""
step14b_batch_draw.py — Draw random batches of papers for human full-text coding.

Draw pool: ALL papers that passed abstract/title screening (step13_all_included.csv),
regardless of whether a PDF has been retrieved.  After each draw the script checks
every sampled paper's PDF and writes a _missing.csv for papers that need procuring
before the batch can be assigned to a coder.

Creates one folder per batch under scripts/outputs/step14b/{round}/ containing:
  papers_{round}_missing.csv    — papers needing PDF procurement (status, doi, title, year)
  coding_{round}_template.csv  — blank coding template pre-filled with doi/title/year
  instruction_{round}.pdf      — one-page coder briefing (Drive link, how to code)

Uploads to Google Drive ({round} subfolder under DRIVE_PARENT_ID):
  instruction PDF + coding template + PDFs subfolder with available full-text PDFs

Shared codebook (CODEBOOK_FT.pdf in the Drive parent folder) is uploaded
separately via push_codebook_update.py — not touched here.

Coder assignment (name, email) is recorded separately in assignments.csv after
the round is created and a coder is hired. The script creates blank rows for
each round so the lead can fill them in.

Assignments are tracked in:
  documentation/coding/systematic-map/rounds/assignments.csv  (local)
  DRIVE_PARENT_ID/assignments.csv                              (Drive, updated in-place)

Round naming: FT-R3, FT-R4, FT-R5 … (FT-R1 = calibration, FT-R2x = legacy biased draws)

Usage:
  # Create 6 batches (auto-names FT-R3 … FT-R8, seeds 42–47)
  conda run -n ilri01 python scripts/step14b_batch_draw.py --rounds 6

  # Create 1 more batch later (auto-detects FT-R9, seed 48)
  conda run -n ilri01 python scripts/step14b_batch_draw.py --rounds 1

  # Dry-run to preview without uploading
  conda run -n ilri01 python scripts/step14b_batch_draw.py --rounds 6 --dry-run

Arguments:
  --rounds     Number of batches to create (required)
  --sample     Papers per batch (default 20)
  --min-chars  Min extracted character count to consider a retrieved PDF legit (default 2000)
  --dry-run    Print what would happen; skip Drive upload
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import string
import sys
from datetime import date
from pathlib import Path

import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────────

REPO_ROOT      = Path(__file__).resolve().parent.parent
SCRIPTS_DIR    = REPO_ROOT / "scripts"
OUTPUTS_DIR    = SCRIPTS_DIR / "outputs"
ROUNDS_DOC_DIR = REPO_ROOT / "documentation" / "coding" / "systematic-map" / "rounds"
CREDS_DIR      = REPO_ROOT / "deliverables" / ".credentials"

MANIFEST_CSV          = OUTPUTS_DIR / "step13" / "step13_manifest.csv"
STEP13_ALL_INCLUDED_CSV = OUTPUTS_DIR / "step13" / "step13_all_included.csv"
STEP14_RESULTS        = OUTPUTS_DIR / "step14" / "step14_results.csv"
ASSIGNMENTS_CSV       = OUTPUTS_DIR / "step14b" / "assignments.csv"

DRIVE_PARENT_ID    = "13p22XfvB6sNtTtnMS-dkI1t-joMn-6Bo"
CODEBOOK_LOCAL      = ROUNDS_DOC_DIR / "CODEBOOK_FT.pdf"
CODEBOOK_DRIVE_NAME = "CODEBOOK_FT.pdf"

# Papers used in calibration rounds — never re-sample
CALIBRATION_DOIS = {
    "10.1016/j.agee.2019.04.004",
    "10.1016/j.crm.2017.06.001",
    "10.1016/j.crm.2017.03.001",
    "10.1007/s10584-016-1792-0",
    "10.1080/17565529.2017.1411240",
}

TEMPLATE_FIELDS = [
    "confirmed_include",
    "publication_year", "publication_type", "country_region", "geographic_scale",
    "producer_type", "marginalized_subpopulations", "adaptation_focus",
    "process_outcome_domains", "indicators_measured", "methodological_approach",
    "purpose_of_assessment", "data_sources", "temporal_coverage",
    "cost_data_reported", "strengths_and_limitations", "lessons_learned",
]

MIN_FILE_SIZE_KB = 50  # fallback quality threshold when step14 char count not available


# ── pool helpers ───────────────────────────────────────────────────────────────

def load_pool(min_chars: int) -> pd.DataFrame:
    """
    Build the eligible draw pool from ALL papers that passed abstract/title screening.

    Uses step13_all_included.csv as the base (unbiased — includes papers with and
    without retrieved PDFs).  Merges manifest data so pdf_status can be assessed
    per-paper after sampling without filtering anyone out of the draw.
    """
    pool = pd.read_csv(
        STEP13_ALL_INCLUDED_CSV,
        usecols=["dedupe_key", "doi", "title", "year", "coverDate"],
        low_memory=False,
    )
    pool = pool.drop_duplicates("dedupe_key").reset_index(drop=True)

    # Year: prefer 'year' column, fall back to coverDate (format "YYYY-MM-DD")
    pool["year"] = pd.to_numeric(pool["year"], errors="coerce")
    pool["_cover_year"] = (
        pd.to_datetime(pool["coverDate"], errors="coerce").dt.year
    )
    pool["year"] = pool["year"].fillna(pool["_cover_year"])
    pool = pool.drop(columns=["coverDate", "_cover_year"])

    # Merge manifest for PDF availability (left join — papers not in manifest get NaN)
    manifest = pd.read_csv(
        MANIFEST_CSV,
        usecols=["dedupe_key", "file_path", "file_size_kb", "status"],
        low_memory=False,
    )
    manifest = manifest.drop_duplicates("dedupe_key")
    pool = pool.merge(manifest, on="dedupe_key", how="left")
    pool["file_size_kb"] = pd.to_numeric(pool["file_size_kb"], errors="coerce").fillna(0)
    pool["status"] = pool["status"].fillna("not_in_manifest")

    # Enrich with step14 char counts + year fallback (LLM decision not used for pool filtering)
    if STEP14_RESULTS.exists():
        s14 = pd.read_csv(
            STEP14_RESULTS,
            usecols=["dedupe_key", "s14_fulltext_chars", "coverDate"],
            low_memory=False,
        )
        s14["s14_fulltext_chars"] = pd.to_numeric(s14["s14_fulltext_chars"], errors="coerce")
        s14["s14_year"] = pd.to_datetime(s14["coverDate"], errors="coerce").dt.year
        s14 = s14.drop(columns=["coverDate"]).drop_duplicates("dedupe_key")
        pool = pool.merge(s14, on="dedupe_key", how="left")
        missing_yr = pool["year"].isna()
        pool.loc[missing_yr, "year"] = pool.loc[missing_yr, "s14_year"]
        pool = pool.drop(columns=["s14_year"])
    else:
        pool["s14_fulltext_chars"] = float("nan")

    pool = pool[~pool["doi"].isin(CALIBRATION_DOIS)]
    pool = pool.drop_duplicates("dedupe_key").reset_index(drop=True)

    # Fill missing year from DOI pattern as last resort (e.g. 10.1016/j.foo.2019.03.001)
    _doi_year = re.compile(r"\b(19|20)\d{2}\b")
    def _year_from_doi(row):
        if pd.notna(row["year"]):
            return row["year"]
        m = _doi_year.search(str(row.get("doi", "") or ""))
        return int(m.group()) if m else float("nan")
    pool["year"] = pool.apply(_year_from_doi, axis=1)

    has_pdf = (pool["status"] == "retrieved") & pool["file_path"].str.endswith(".pdf", na=False)
    print(
        f"  [pool] Total eligible: {len(pool):,}  |  "
        f"Have PDF: {has_pdf.sum():,}  |  "
        f"Missing PDF: {(~has_pdf).sum():,}"
    )

    return pool


def load_already_sampled(exclude_round: str | None = None) -> set[str]:
    """
    Return dedupe_keys already assigned in any prior batch (new + legacy locations).
    """
    sampled: set[str] = set()

    # New rounds: read DOIs from coding templates (papers_*.csv no longer generated)
    for template_csv in (OUTPUTS_DIR / "step14b").glob("*/coding_*_template.csv"):
        if exclude_round and template_csv.parent.name == exclude_round:
            continue
        df = pd.read_csv(template_csv, usecols=["doi"])
        sampled.update(df["doi"].dropna().tolist())

    # Legacy rounds (FT-R2x): papers_*.csv still present, read dedupe_key + doi
    for papers_csv in (OUTPUTS_DIR / "step14b").glob("*/papers_*.csv"):
        if exclude_round and papers_csv.parent.name == exclude_round:
            continue
        if "_missing" in papers_csv.name:
            continue
        df = pd.read_csv(papers_csv)
        if "dedupe_key" in df.columns:
            sampled.update(df["dedupe_key"].dropna().tolist())
        if "doi" in df.columns:
            sampled.update(df["doi"].dropna().tolist())

    # Legacy location: documentation/.../rounds/*/papers_*.csv
    for papers_csv in ROUNDS_DOC_DIR.glob("*/papers_*.csv"):
        round_name = papers_csv.parent.name
        if exclude_round and round_name == exclude_round:
            continue
        if "_missing" in papers_csv.name:
            continue
        df = pd.read_csv(papers_csv)
        if "dedupe_key" in df.columns:
            sampled.update(df["dedupe_key"].dropna().tolist())
        if "doi" in df.columns:
            sampled.update(df["doi"].dropna().tolist())

    return sampled


def make_template(sample: pd.DataFrame) -> pd.DataFrame:
    tpl = sample[["doi", "title", "year"]].copy()
    tpl["year"] = pd.to_numeric(tpl["year"], errors="coerce").apply(
        lambda x: str(int(x)) if pd.notna(x) else ""
    )
    for field in TEMPLATE_FIELDS:
        tpl[field] = ""
    tpl["coder_id"] = ""
    tpl["notes"]    = ""
    return tpl


# ── PDF quality check ──────────────────────────────────────────────────────────

def _check_pdf_quality(row: pd.Series, min_chars: int) -> str:
    """
    Returns 'ok', 'missing', or 'suspect'.

    ok      — file present and passes content quality threshold
    missing — no file_path recorded or file not found on disk
    suspect — file present but too small / too little text (likely error page or cover only)
    """
    fp = row.get("file_path")
    if not fp or (isinstance(fp, float) and pd.isna(fp)):
        return "missing"
    if not Path(str(fp)).exists():
        return "missing"
    # HTML files are not usable as coders need PDFs
    if Path(str(fp)).suffix.lower() in (".html", ".htm"):
        return "missing"
    # File exists — quality check
    chars = row.get("s14_fulltext_chars", float("nan"))
    if pd.notna(chars):
        return "ok" if float(chars) >= min_chars else "suspect"
    # No char count available yet — fall back to file size
    size_kb = row.get("file_size_kb", 0)
    return "ok" if float(size_kb) >= MIN_FILE_SIZE_KB else "suspect"


# ── round naming ───────────────────────────────────────────────────────────────

# FT-R1 = calibration  |  FT-R2x = legacy biased draws  |  FT-R3+ = unbiased
ROUND_NUMERIC_START = 3


def next_round_names(n: int) -> tuple[list[str], int]:
    """
    Auto-detect next available FT-R{N} round names (e.g. FT-R3, FT-R4, …).

    Scans existing batch folders for the highest FT-R{N} already used, then
    returns n names starting from the next number (minimum ROUND_NUMERIC_START).

    Returns (round_names, base_seed).
    """
    pattern = re.compile(r"^FT-R(\d+)$")
    used_nums: set[int] = set()

    for location in [OUTPUTS_DIR / "step14b", ROUNDS_DOC_DIR]:
        if location.exists():
            for child in location.iterdir():
                m = pattern.match(child.name)
                if m:
                    used_nums.add(int(m.group(1)))

    start = max(max(used_nums) + 1, ROUND_NUMERIC_START) if used_nums else ROUND_NUMERIC_START
    base_seed = 42 + (start - ROUND_NUMERIC_START)  # seed 42 = FT-R3

    names = [f"FT-R{start + i}" for i in range(n)]
    return names, base_seed


# ── assignments CSV ────────────────────────────────────────────────────────────

def _read_assignments() -> list[dict]:
    if not ASSIGNMENTS_CSV.exists():
        return []
    with ASSIGNMENTS_CSV.open() as f:
        return list(csv.DictReader(f))


def _write_assignments(rows: list[dict]) -> None:
    fieldnames = ["round", "coder_name", "coder_email", "assigned_date",
                  "n_papers", "drive_folder_id"]
    ASSIGNMENTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with ASSIGNMENTS_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def cleanup_stale_assignments() -> None:
    """
    Remove rows from assignments.csv whose output folder no longer exists.
    Called at the start of each run so deleted batches are re-drawn cleanly.
    """
    rows = _read_assignments()
    if not rows:
        return
    kept, dropped = [], []
    for row in rows:
        folder = OUTPUTS_DIR / "step14b" / row["round"]
        if folder.exists():
            kept.append(row)
        else:
            dropped.append(row["round"])
    if dropped:
        _write_assignments(kept)
        print(f"Removed stale assignment rows: {', '.join(dropped)}")


def record_assignment(round_name: str, n_papers: int, drive_folder_id: str) -> None:
    """
    Add or update the row for this round in assignments.csv.

    Preserves any manually entered coder_name, coder_email, and assigned_date
    already in the file — only n_papers and drive_folder_id are overwritten.
    """
    rows = _read_assignments()
    existing = next((r for r in rows if r.get("round") == round_name), {})
    rows = [r for r in rows if r.get("round") != round_name]
    rows.append({
        "round":           round_name,
        "coder_name":      existing.get("coder_name", ""),
        "coder_email":     existing.get("coder_email", ""),
        "assigned_date":   existing.get("assigned_date") or date.today().isoformat(),
        "n_papers":        n_papers,
        "drive_folder_id": drive_folder_id,
    })
    _write_assignments(rows)
    print(f"Updated local assignments: {ASSIGNMENTS_CSV.relative_to(REPO_ROOT)}")


# ── instruction PDF ────────────────────────────────────────────────────────────

def make_instruction_pdf(
    out_path: Path,
    round_name: str,
    n_papers: int,
    drive_folder_id: str,
    codebook_url: str = "",
) -> None:
    """Generate a one-page coder instruction sheet using reportlab."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    )

    drive_url = f"https://drive.google.com/drive/folders/{drive_folder_id}"

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=2.2 * cm, rightMargin=2.2 * cm,
        topMargin=2.2 * cm, bottomMargin=2.2 * cm,
    )

    styles = getSampleStyleSheet()
    green  = colors.HexColor("#2e6b3e")

    title_style = ParagraphStyle(
        "Title", parent=styles["Heading1"],
        fontSize=16, textColor=green, spaceAfter=4,
    )
    h2_style = ParagraphStyle(
        "H2", parent=styles["Heading2"],
        fontSize=11, textColor=green, spaceBefore=12, spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=10, leading=14,
    )
    small_style = ParagraphStyle(
        "Small", parent=styles["Normal"],
        fontSize=9, leading=13, textColor=colors.HexColor("#555555"),
    )
    link_style = ParagraphStyle(
        "Link", parent=styles["Normal"],
        fontSize=9, leading=13, textColor=colors.HexColor("#1a6496"),
    )

    story = []

    # Header
    story.append(Paragraph(f"Full-Text Coding Round: {round_name}", title_style))
    story.append(Paragraph(
        "ILRI Climate Adaptation Effectiveness — Systematic Map",
        small_style,
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=green, spaceAfter=10))

    # Round summary box
    summary_data = [
        ["Round",    round_name],
        ["Papers",   str(n_papers)],
        ["Created",  date.today().strftime("%d %b %Y")],
    ]
    tbl = Table(summary_data, colWidths=[3 * cm, 12 * cm])
    tbl.setStyle(TableStyle([
        ("FONTSIZE",      (0, 0), (-1, -1), 10),
        ("FONTNAME",      (0, 0), (0, -1), "Helvetica-Bold"),
        ("TEXTCOLOR",     (0, 0), (0, -1), green),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#f5f9f6"), colors.white]),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#ccddcc")),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.4 * cm))

    # Where to find files
    story.append(Paragraph("Where to find your files", h2_style))
    story.append(Paragraph(
        "Your papers and coding template are in the Google Drive folder below.",
        body_style,
    ))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(f'<a href="{drive_url}">open your Drive folder</a>', link_style))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(
        "Inside your folder you will find:", body_style,
    ))
    items = [
        f"<b>coding_{round_name.lower()}_template.csv</b> — your blank coding sheet",
        f"<b>{round_name} pdfs/</b> — the full-text PDFs for each paper",
    ]
    for item in items:
        story.append(Paragraph(f"&nbsp;&nbsp;• {item}", body_style))
    story.append(Spacer(1, 0.2 * cm))
    codebook_line = (
        f'<b>Codebook</b> — <a href="{codebook_url}">open codebook</a>'
        if codebook_url else
        '<b>Codebook</b> — shared separately by the review team'
    )
    story.append(Paragraph(codebook_line, body_style))
    story.append(Spacer(1, 0.3 * cm))

    # How to code
    story.append(Paragraph("How to code", h2_style))
    codebook_ref = (
        f'<a href="{codebook_url}">the codebook</a>' if codebook_url else "<b>CODEBOOK_FT.pdf</b>"
    )
    steps = [
        f"Open {codebook_ref} and read it before starting.",
        "Open the coding template CSV in Excel or Google Sheets.",
        "For each paper: read the PDF, decide <b>confirmed_include</b> (yes / no / unclear), "
        "then fill in all 16 fields if included.",
        "Use the 'When in doubt' scenario tables in the codebook for borderline cases.",
        "Leave a note in the <b>notes</b> column for any paper you are uncertain about.",
        "When finished, save your file as described below and upload it to the Drive folder.",
    ]
    for i, step in enumerate(steps, 1):
        story.append(Paragraph(f"<b>{i}.</b>&nbsp;&nbsp;{step}", body_style))
        story.append(Spacer(1, 0.15 * cm))

    # File naming
    story.append(Paragraph("File naming", h2_style))
    story.append(Paragraph(
        f"Save your completed sheet as: <b>coding_{round_name.lower()}_INITIALS.csv</b><br/>"
        f"For example: <b>coding_{round_name.lower()}_AS.csv</b>",
        body_style,
    ))
    story.append(Spacer(1, 0.3 * cm))

    # Footer
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#ccddcc")))
    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph(
        "Questions? Contact the review team. Generated automatically by step14b_batch_draw.py",
        small_style,
    ))

    doc.build(story)


# ── Google Drive helpers ───────────────────────────────────────────────────────

def gdrive_auth():
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request

    token_file = CREDS_DIR / "token.json"
    scopes = ["https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_authorized_user_file(str(token_file), scopes)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return creds


def ensure_codebook_in_parent(service) -> str:
    """
    Upload or replace CODEBOOK_FT.pdf in the parent folder, set anyone-with-link
    read access, and return the direct view URL.
    """
    if not CODEBOOK_LOCAL.exists():
        print(f"  WARNING: local codebook not found at {CODEBOOK_LOCAL} — skipping")
        return ""
    file_id = update_or_upload_file(service, CODEBOOK_LOCAL, CODEBOOK_DRIVE_NAME, DRIVE_PARENT_ID)
    print(f"  Codebook synced to parent folder  (id={file_id})")
    try:
        service.permissions().create(
            fileId=file_id,
            body={"type": "anyone", "role": "reader"},
            fields="id",
            supportsAllDrives=True,
        ).execute()
        print(f"  Codebook sharing: anyone with link can view")
    except Exception as e:
        print(f"  WARNING: could not set codebook sharing — {e}")
    return f"https://drive.google.com/file/d/{file_id}/view"


def find_folder(service, name: str, parent_id: str) -> str | None:
    resp = service.files().list(
        q=(f"name='{name}' and '{parent_id}' in parents "
           f"and mimeType='application/vnd.google-apps.folder' and trashed=false"),
        fields="files(id)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()
    files = resp.get("files", [])
    return files[0]["id"] if files else None


def find_file(service, name: str, parent_id: str) -> str | None:
    resp = service.files().list(
        q=f"name='{name}' and '{parent_id}' in parents and trashed=false",
        fields="files(id)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()
    files = resp.get("files", [])
    return files[0]["id"] if files else None


def create_drive_folder(service, name: str, parent_id: str) -> str:
    meta = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }
    f = service.files().create(body=meta, fields="id", supportsAllDrives=True).execute()
    return f["id"]


def get_or_create_folder(service, name: str, parent_id: str) -> tuple[str, bool]:
    """Return (folder_id, created). Reuses existing folder if found."""
    existing = find_folder(service, name, parent_id)
    if existing:
        return existing, False
    return create_drive_folder(service, name, parent_id), True


def upload_file(service, local_path: Path, drive_name: str, parent_id: str) -> str:
    import mimetypes
    from googleapiclient.http import MediaFileUpload

    mime, _ = mimetypes.guess_type(str(local_path))
    mime = mime or "application/octet-stream"
    meta = {"name": drive_name, "parents": [parent_id]}
    media = MediaFileUpload(str(local_path), mimetype=mime, resumable=True)
    f = service.files().create(
        body=meta, media_body=media, fields="id", supportsAllDrives=True,
    ).execute()
    return f["id"]


def update_or_upload_file(
    service, local_path: Path, drive_name: str, parent_id: str
) -> str:
    """Replace existing file in-place if found, otherwise upload fresh copy."""
    import mimetypes
    from googleapiclient.http import MediaFileUpload

    existing_id = find_file(service, drive_name, parent_id)
    if existing_id:
        mime, _ = mimetypes.guess_type(str(local_path))
        mime = mime or "application/octet-stream"
        media = MediaFileUpload(str(local_path), mimetype=mime, resumable=True)
        service.files().update(
            fileId=existing_id, media_body=media, supportsAllDrives=True,
        ).execute()
        return existing_id
    return upload_file(service, local_path, drive_name, parent_id)


def download_file(service, file_id: str, dest_path: Path) -> None:
    """Download a Drive file to a local path."""
    from googleapiclient.http import MediaIoBaseDownload
    import io

    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    dest_path.write_bytes(fh.getvalue())


# ── per-batch logic ────────────────────────────────────────────────────────────

def draw_batch(
    round_name: str,
    pool: pd.DataFrame,
    already_sampled: set[str],
    sample_n: int,
    seed: int,
    min_chars: int,
    dry_run: bool,
    service,  # Drive service or None in dry-run
) -> set[str]:
    """
    Draw one batch, write local files, upload to Drive.

    Returns the set of dedupe_keys assigned in this batch so the caller can
    exclude them from subsequent draws in the same run.
    """
    out_dir = OUTPUTS_DIR / "step14b" / round_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─'*60}")
    print(f"  Batch: {round_name}   Seed: {seed}")
    print(f"{'─'*60}")

    mask      = pool["dedupe_key"].isin(already_sampled) | pool["doi"].isin(already_sampled)
    available = pool[~mask].copy()
    print(f"  Available: {len(available):,}  (pool {len(pool):,} – assigned {len(already_sampled):,})")

    if len(available) < sample_n:
        print(f"  ERROR: only {len(available)} papers available, need {sample_n} — skipping")
        return set()

    sample = available.sample(n=sample_n, random_state=seed).sort_values("dedupe_key").copy()

    # ── PDF quality check ────────────────────────────────────────────────────
    sample["pdf_status"] = sample.apply(
        lambda r: _check_pdf_quality(r, min_chars), axis=1
    )
    status_counts = sample["pdf_status"].value_counts().to_dict()
    print(f"  PDF status: { {k: status_counts.get(k, 0) for k in ['ok', 'missing', 'suspect']} }")

    # ── Write local files ────────────────────────────────────────────────────
    template_csv = out_dir / f"coding_{round_name.lower()}_template.csv"

    make_template(sample).to_csv(template_csv, index=False)
    print(f"  Written: {template_csv.relative_to(REPO_ROOT)}")

    # ── Missing/suspect papers list (for procurement) ────────────────────────
    missing_csv = out_dir / f"papers_{round_name.lower()}_missing.csv"
    needs_pdf = sample[sample["pdf_status"] != "ok"]
    if not needs_pdf.empty:
        miss_out = needs_pdf[["doi", "title", "year"]].copy()
        miss_out["year"] = pd.to_numeric(miss_out["year"], errors="coerce").apply(
            lambda x: str(int(x)) if pd.notna(x) else ""
        )
        miss_out.insert(0, "status", "")
        miss_out.to_csv(missing_csv, index=False)
        print(f"  Written: {missing_csv.relative_to(REPO_ROOT)}  ({len(miss_out)} papers need PDF)")
    else:
        print(f"  All {sample_n} PDFs present and passing quality check")

    # ── Copy available PDFs into local batch pdfs folder ─────────────────────
    pdfs_local_dir = out_dir / f"{round_name} pdfs"
    pdfs_local_dir.mkdir(exist_ok=True)
    n_copied = 0
    for _, row in sample.iterrows():
        if row.get("pdf_status") != "ok":
            continue
        fp = row.get("file_path")
        if not fp or (isinstance(fp, float) and pd.isna(fp)):
            continue
        src = Path(str(fp))
        if src.exists() and src.suffix.lower() == ".pdf":
            shutil.copy2(src, pdfs_local_dir / src.name)
            n_copied += 1
    if n_copied:
        print(f"  Copied {n_copied} PDFs → {pdfs_local_dir.relative_to(REPO_ROOT)}")

    if dry_run:
        instruction_pdf = out_dir / f"instruction_{round_name.lower()}.pdf"
        make_instruction_pdf(
            out_path=instruction_pdf,
            round_name=round_name,
            n_papers=len(sample),
            drive_folder_id="[TO BE ASSIGNED]",
            codebook_url="",
        )
        print(f"  Written: {instruction_pdf.relative_to(REPO_ROOT)}")
        print(f"  [dry-run] skipping Drive upload")
        record_assignment(round_name, len(sample), drive_folder_id="DRY_RUN")
        return set(sample["dedupe_key"].tolist())

    # ── Drive upload ─────────────────────────────────────────────────────────
    round_folder_id, created = get_or_create_folder(service, round_name, DRIVE_PARENT_ID)
    action = "Created" if created else "Using"
    print(f"  {action} Drive folder: {round_name}  (id={round_folder_id})")

    codebook_url = ensure_codebook_in_parent(service)

    instruction_pdf = out_dir / f"instruction_{round_name.lower()}.pdf"
    make_instruction_pdf(
        out_path=instruction_pdf,
        round_name=round_name,
        n_papers=len(sample),
        drive_folder_id=round_folder_id,
        codebook_url=codebook_url,
    )
    print(f"  Written: {instruction_pdf.relative_to(REPO_ROOT)}")

    instr_id = update_or_upload_file(service, instruction_pdf, instruction_pdf.name, round_folder_id)
    tmpl_id  = update_or_upload_file(service, template_csv,    template_csv.name,    round_folder_id)
    print(f"  Uploaded: {instruction_pdf.name}  (id={instr_id})")
    print(f"  Uploaded: {template_csv.name}  (id={tmpl_id})")
    if missing_csv.exists():
        miss_id = update_or_upload_file(service, missing_csv, missing_csv.name, round_folder_id)
        print(f"  Uploaded: {missing_csv.name}  (id={miss_id})")

    # PDFs subfolder — upload only available PDFs; missing ones are tracked in _missing.csv
    pdfs_folder_name = f"{round_name} pdfs"
    pdfs_folder_id, _ = get_or_create_folder(service, pdfs_folder_name, round_folder_id)

    n_ok = status_counts.get("ok", 0)
    print(f"  Uploading {n_ok} available PDFs to '{pdfs_folder_name}'...")
    errors: list[str] = []
    for i, (_, row) in enumerate(sample.iterrows(), 1):
        doi_or_key = row.get("doi") or row["dedupe_key"]
        label = f"[{i:3d}/{len(sample)}]"
        fp = row.get("file_path")
        if not fp or (isinstance(fp, float) and pd.isna(fp)):
            print(f"    {label} NO PDF (not retrieved): {doi_or_key}")
            errors.append(doi_or_key)
            continue
        pdf_path = Path(str(fp))
        if not pdf_path.exists():
            print(f"    {label} MISSING (file not found): {pdf_path.name}")
            errors.append(doi_or_key)
            continue
        if pdf_path.suffix.lower() in (".html", ".htm"):
            print(f"    {label} HTML only (skipping, needs PDF): {pdf_path.name}")
            errors.append(doi_or_key)
            continue
        # Prefer the local batch copy if already copied there
        local_copy = pdfs_local_dir / pdf_path.name
        src = local_copy if local_copy.exists() else pdf_path
        print(f"    {label} {src.name}", end=" ", flush=True)
        update_or_upload_file(service, src, src.name, pdfs_folder_id)
        print("✓")

    record_assignment(round_name, len(sample), round_folder_id)

    n_uploaded = len(sample) - len(errors)
    print(f"  Done: {n_uploaded}/{len(sample)} PDFs uploaded  |  {len(errors)} need procuring")
    if errors:
        print(f"  See: papers_{round_name.lower()}_missing.csv for procurement list")
    print(f"  Drive: https://drive.google.com/drive/folders/{round_folder_id}")

    return set(sample["dedupe_key"].tolist())


# ── push dry-run batches to Drive ─────────────────────────────────────────────

def push_dry_run_batches(service) -> None:
    """
    Upload all dry-run batches (drive_folder_id == 'DRY_RUN') to Google Drive.

    For each pending batch:
      1. Re-generates the instruction PDF with the real Drive folder link.
      2. Uploads instruction PDF + coding template.
      3. Uploads available full-text PDFs (skips missing ones).
      4. Updates assignments.csv with the real folder ID.
    """
    rows = _read_assignments()
    pending = [r for r in rows if r.get("drive_folder_id") == "DRY_RUN"]

    if not pending:
        print("No dry-run batches found — nothing to push.")
        return

    print(f"Found {len(pending)} dry-run batch(es) to push: "
          f"{', '.join(r['round'] for r in pending)}")

    codebook_url = ensure_codebook_in_parent(service)

    for row in pending:
        round_name = row["round"]
        out_dir    = OUTPUTS_DIR / "step14b" / round_name

        template_csv = out_dir / f"coding_{round_name.lower()}_template.csv"

        if not template_csv.exists():
            print(f"\n  {round_name}: coding template missing — skipping (re-run --rounds to recreate)")
            continue

        n = len(pd.read_csv(template_csv))

        print(f"\n{'─'*60}")
        print(f"  Pushing: {round_name}  ({n} papers)")
        print(f"{'─'*60}")

        round_folder_id, created = get_or_create_folder(service, round_name, DRIVE_PARENT_ID)
        action = "Created" if created else "Using"
        print(f"  {action} Drive folder: {round_name}  (id={round_folder_id})")

        instruction_pdf = out_dir / f"instruction_{round_name.lower()}.pdf"
        make_instruction_pdf(
            out_path=instruction_pdf,
            round_name=round_name,
            n_papers=n,
            drive_folder_id=round_folder_id,
            codebook_url=codebook_url,
        )
        print(f"  Regenerated: {instruction_pdf.name}")

        instr_id = update_or_upload_file(service, instruction_pdf, instruction_pdf.name, round_folder_id)
        tmpl_id  = update_or_upload_file(service, template_csv,    template_csv.name,    round_folder_id)
        print(f"  Uploaded: {instruction_pdf.name}  (id={instr_id})")
        print(f"  Uploaded: {template_csv.name}  (id={tmpl_id})")

        pdfs_folder_name = f"{round_name} pdfs"
        pdfs_folder_id, _ = get_or_create_folder(service, pdfs_folder_name, round_folder_id)

        pdfs_local_dir = out_dir / pdfs_folder_name
        pdf_files = sorted(pdfs_local_dir.glob("*.pdf")) if pdfs_local_dir.exists() else []
        print(f"  Uploading {len(pdf_files)} PDFs from local pdfs/ folder to '{pdfs_folder_name}'...")
        for i, pdf_file in enumerate(pdf_files, 1):
            label = f"[{i:3d}/{len(pdf_files)}]"
            print(f"    {label} {pdf_file.name}", end=" ", flush=True)
            update_or_upload_file(service, pdf_file, pdf_file.name, pdfs_folder_id)
            print("✓")

        missing_csv = out_dir / f"papers_{round_name.lower()}_missing.csv"
        if missing_csv.exists():
            miss_id = update_or_upload_file(service, missing_csv, missing_csv.name, round_folder_id)
            print(f"  Uploaded: {missing_csv.name}  (id={miss_id})")

        record_assignment(round_name, n, round_folder_id)
        print(f"  Done: {len(pdf_files)}/{n} PDFs uploaded")
        print(f"  Drive: https://drive.google.com/drive/folders/{round_folder_id}")

    print(f"\nUploading assignments.csv to Drive parent folder...")
    update_or_upload_file(service, ASSIGNMENTS_CSV, "assignments.csv", DRIVE_PARENT_ID)
    print(f"\n{'='*60}")
    print(f"  All done — {len(pending)} batch(es) pushed")
    print(f"{'='*60}\n")


# ── deduplicate PDFs on Drive ─────────────────────────────────────────────────

def dedupe_drive_pdfs(service, rounds: list[str] | None = None) -> None:
    """
    For each FT-R3+ round on Drive, find duplicate filenames in the pdfs subfolder
    and trash all but the first (oldest) copy.

    If rounds is given, only process those rounds; otherwise process all pushed rounds.
    """
    rows = _read_assignments()
    live = [r for r in rows if r.get("drive_folder_id") not in ("", "DRY_RUN")]

    pattern = re.compile(r"^FT-R(\d+)$")
    live = [r for r in live if pattern.match(r["round"]) and int(pattern.match(r["round"]).group(1)) >= ROUND_NUMERIC_START]

    if rounds:
        live = [r for r in live if r["round"] in rounds]

    if not live:
        print("No eligible pushed rounds found.")
        return

    print(f"Checking {len(live)} round(s) for duplicate PDFs: {', '.join(r['round'] for r in live)}")
    total_trashed = 0

    for row in live:
        round_name = row["round"]
        folder_id  = row["drive_folder_id"]

        pdfs_folder_name = f"{round_name} pdfs"
        pdfs_folder_id = find_folder(service, pdfs_folder_name, folder_id)
        if not pdfs_folder_id:
            print(f"\n  {round_name}: '{pdfs_folder_name}' not found on Drive — skipping")
            continue

        # List ALL files (paginate in case >100)
        all_files: list[dict] = []
        page_token = None
        while True:
            resp = service.files().list(
                q=(f"'{pdfs_folder_id}' in parents and trashed=false "
                   f"and mimeType!='application/vnd.google-apps.folder'"),
                fields="nextPageToken, files(id, name, createdTime)",
                pageSize=1000,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
                pageToken=page_token,
            ).execute()
            all_files.extend(resp.get("files", []))
            page_token = resp.get("nextPageToken")
            if not page_token:
                break

        # Group by name, sort each group by createdTime (oldest first = keep)
        from collections import defaultdict
        by_name: dict[str, list[dict]] = defaultdict(list)
        for f in all_files:
            by_name[f["name"]].append(f)

        dupes = {name: files for name, files in by_name.items() if len(files) > 1}
        if not dupes:
            print(f"\n  {round_name}: {len(all_files)} files, no duplicates")
            continue

        n_trashed = 0
        print(f"\n  {round_name}: {len(all_files)} files, {len(dupes)} duplicate name(s)")
        for name, files in sorted(dupes.items()):
            files_sorted = sorted(files, key=lambda f: f.get("createdTime", ""))
            keep = files_sorted[0]
            trash_list = files_sorted[1:]
            print(f"    '{name}': keeping oldest ({keep['id']}), trashing {len(trash_list)}")
            for f in trash_list:
                service.files().update(
                    fileId=f["id"],
                    body={"trashed": True},
                    supportsAllDrives=True,
                ).execute()
                n_trashed += 1

        print(f"  Trashed {n_trashed} duplicate(s) in {round_name}")
        total_trashed += n_trashed

    print(f"\nDone — {total_trashed} duplicate file(s) trashed across all rounds.")


# ── pull PDFs from Drive ───────────────────────────────────────────────────────

def pull_pdfs(service, round_name: str) -> None:
    """
    Download PDFs from the Drive round folder into the local pdfs/ subfolder.

    Use this after Jenn has uploaded procured PDFs to the Drive round folder.
    Already-present local files are skipped.
    """
    rows = _read_assignments()
    row = next((r for r in rows if r.get("round") == round_name), None)
    if not row:
        print(f"ERROR: {round_name} not found in assignments.csv")
        sys.exit(1)

    folder_id = row.get("drive_folder_id", "")
    if not folder_id or folder_id in ("DRY_RUN", ""):
        print(f"ERROR: {round_name} has no real Drive folder (id={folder_id!r})")
        sys.exit(1)

    pdfs_folder_name = f"{round_name} pdfs"
    local_pdfs_dir = OUTPUTS_DIR / "step14b" / round_name / pdfs_folder_name
    local_pdfs_dir.mkdir(parents=True, exist_ok=True)
    pdfs_folder_id = find_folder(service, pdfs_folder_name, folder_id)
    if not pdfs_folder_id:
        print(f"ERROR: '{pdfs_folder_name}' subfolder not found in Drive folder {folder_id}")
        sys.exit(1)

    resp = service.files().list(
        q=(f"'{pdfs_folder_id}' in parents and trashed=false "
           f"and mimeType!='application/vnd.google-apps.folder'"),
        fields="files(id, name)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()
    drive_files = resp.get("files", [])

    print(f"\n  Drive '{pdfs_folder_name}': {len(drive_files)} file(s)")
    n_downloaded = n_skipped = 0
    for f in drive_files:
        dest = local_pdfs_dir / f["name"]
        if dest.exists():
            print(f"  Already local: {f['name']}")
            n_skipped += 1
            continue
        print(f"  Downloading: {f['name']}...", end=" ", flush=True)
        download_file(service, f["id"], dest)
        print("✓")
        n_downloaded += 1

    print(f"\n  Done: {n_downloaded} downloaded, {n_skipped} already present")
    print(f"  Local: {local_pdfs_dir.relative_to(REPO_ROOT)}")

    # ── sync updated missing CSV from Drive ─────────────────────────────────
    # Prefer a Google Sheets version (Jenn edits there); fall back to plain CSV.
    missing_csv_name    = f"papers_{round_name.lower()}_missing.csv"
    missing_sheets_name = f"papers_{round_name.lower()}_missing"   # no extension
    local_missing = OUTPUTS_DIR / "step14b" / round_name / missing_csv_name

    # Record current local statuses before overwriting
    old_statuses: dict[str, str] = {}
    if local_missing.exists():
        try:
            old_df = pd.read_csv(local_missing, dtype=str).fillna("")
            for _, r in old_df.iterrows():
                d = str(r.get("doi", "")).strip()
                s = str(r.get("status", "")).strip()
                if d:
                    old_statuses[d] = s
        except Exception:
            pass

    # Try Sheets version first
    sheets_id = find_file(service, missing_sheets_name, folder_id)
    if sheets_id:
        meta = service.files().get(
            fileId=sheets_id, fields="mimeType", supportsAllDrives=True,
        ).execute()
        if meta.get("mimeType") != "application/vnd.google-apps.spreadsheet":
            sheets_id = None

    if sheets_id:
        content = service.files().export(
            fileId=sheets_id, mimeType="text/csv",
        ).execute()
        if isinstance(content, bytes):
            local_missing.write_bytes(content)
        else:
            local_missing.write_text(str(content), encoding="utf-8")
        print(f"\n  Synced missing CSV from Google Sheets: {missing_sheets_name}")
    else:
        plain_id = find_file(service, missing_csv_name, folder_id)
        if plain_id:
            download_file(service, plain_id, local_missing)
            print(f"\n  Synced missing CSV from Drive: {missing_csv_name}")
        else:
            print(f"\n  No missing CSV found on Drive — skipping.")
            return  # nothing to reconcile

    new_miss = pd.read_csv(local_missing, dtype=str).fillna("")
    changed = []
    for _, r in new_miss.iterrows():
        doi   = str(r.get("doi", "")).strip()
        new_st = str(r.get("status", "")).strip()
        old_st = old_statuses.get(doi, "")
        if new_st != old_st:
            changed.append((doi, old_st, new_st))
    if changed:
        print(f"  Status changes ({len(changed)}):")
        for doi, old_st, new_st in changed:
            print(f"    {doi[:60]}  {old_st!r:10s} → {new_st!r}")
    else:
        print("  No status changes detected in missing CSV.")

    # Update coding template with procurement_status column
    _update_template_procurement_status(service, round_name,
                                         OUTPUTS_DIR / "step14b" / round_name,
                                         new_miss, folder_id)


def _update_template_procurement_status(
    service, round_name: str, local_dir: Path,
    missing_df: pd.DataFrame, folder_id: str,
) -> None:
    """Add/update procurement_status column in coding template, then re-upload to Drive."""
    template_csv = local_dir / f"coding_{round_name.lower()}_template.csv"
    if not template_csv.exists():
        print(f"  WARNING: {template_csv.name} not found — skipping template update.")
        return

    template_df = pd.read_csv(template_csv, dtype=str).fillna("")

    def _is_skip_or_exclude(ps: str) -> bool:
        return ps.startswith("Skip -") or ps == "exclude"

    doi_to_status: dict[str, str] = {}
    for _, r in missing_df.iterrows():
        doi = str(r.get("doi", "")).strip()
        st  = str(r.get("status", "")).strip()
        if doi:
            doi_to_status[doi] = st

    # Build normalised stem → DOI map from local pdfs folder
    pdfs_dir = local_dir / f"{round_name} pdfs"
    local_pdf_stems: list[str] = []
    if pdfs_dir.exists():
        for p in pdfs_dir.iterdir():
            if p.suffix.lower() in (".pdf",):
                stem = p.stem
                if stem.lower().startswith("doi_"):
                    stem = stem[4:]
                local_pdf_stems.append(stem.replace("_", "").lower())

    def _doi_has_local_pdf(doi: str) -> bool:
        doi_norm = doi.replace("/", "").replace("_", "").lower()
        return doi_norm in local_pdf_stems

    def _compute_status(doi: str) -> str:
        doi = str(doi).strip()
        if doi not in doi_to_status:
            return ""           # had PDF from the start
        jenn = doi_to_status[doi].strip()
        if not jenn:
            return "pdf_procured" if _doi_has_local_pdf(doi) else "missing"
        jenn_lower = jenn.lower()
        if jenn_lower == "done":
            return "done"
        if jenn_lower == "exclude":
            return "exclude"
        return f"Skip - {jenn}"

    # Update procurement_status — never overwrite an existing Skip-*/exclude decision
    # (those are manual finalisations that must survive repeated pulls).
    for idx, row in template_df.iterrows():
        existing = str(row.get("procurement_status", "")).strip()
        if _is_skip_or_exclude(existing):
            continue                        # preserve manual decision
        template_df.at[idx, "procurement_status"] = _compute_status(str(row.get("doi", "")))

    # For Skip-* / exclude rows: pre-fill confirmed_include=no so coders know to
    # skip them. procurement_status carries the reason; notes gets a plain-English
    # copy for analysts reading the coding data without the pipeline columns.
    def _skip_note(status: str) -> str:
        if status == "exclude":
            return "Excluded during procurement"
        s = status.removeprefix("Skip - ").strip()
        s_lower = s.lower()
        if s_lower == "na":
            return "PDF could not be retrieved during procurement"
        if s_lower == "book":
            return "Full book — no individual chapter PDF available"
        if s_lower.startswith("exclude"):
            reason = s[len("exclude"):].lstrip(" -").strip()
            return f"Excluded during procurement: {reason}" if reason else "Excluded during procurement"
        return f"Skipped during procurement: {s}"

    skip_mask = (
        template_df["procurement_status"].str.startswith("Skip -", na=False)
        | (template_df["procurement_status"] == "exclude")
    )
    template_df.loc[skip_mask, "confirmed_include"] = "no"
    for idx in template_df[skip_mask].index:
        note = _skip_note(template_df.at[idx, "procurement_status"])
        existing = str(template_df.at[idx, "notes"]).strip()
        if note not in existing:
            template_df.at[idx, "notes"] = f"{existing}; {note}" if existing else note

    template_df.to_csv(template_csv, index=False)
    print(f"  Updated local template: {template_csv.relative_to(REPO_ROOT)}")

    # Show summary
    counts = template_df["procurement_status"].value_counts()
    for val, cnt in counts.items():
        print(f"    {val or '(had PDF from start)':30s} {cnt}")

    # Re-upload updated template to Drive
    tmpl_id = update_or_upload_file(service, template_csv, template_csv.name, folder_id)
    print(f"  Re-uploaded template to Drive  (id={tmpl_id})")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw random batches of papers for human full-text coding."
    )
    parser.add_argument("--rounds",            type=int,
                        help="Number of batches to create, e.g. --rounds 6")
    parser.add_argument("--push",              action="store_true",
                        help="Upload all existing dry-run batches to Drive")
    parser.add_argument("--sync-assignments",  action="store_true",
                        help="Push the local assignments.csv to Drive (no other changes)")
    parser.add_argument("--sample",            type=int, default=20,
                        help="Papers per batch (default 20)")
    parser.add_argument("--min-chars",         type=int, default=2000,
                        help="Min extracted char count to consider a retrieved PDF legit (default 2000)")
    parser.add_argument("--dry-run",           action="store_true")
    parser.add_argument("--fix-instructions",  action="store_true",
                        help="Re-upload codebook to Drive parent and regenerate instruction PDFs "
                             "for all already-pushed rounds")
    parser.add_argument("--pull",              metavar="ROUND",
                        help="Download PDFs from Drive for a round into its local pdfs/ folder "
                             "(e.g. --pull FT-R3). Use after Jenn has uploaded procured PDFs.")
    parser.add_argument("--dedupe",            action="store_true",
                        help="Trash duplicate PDFs in Drive pdfs/ folders for all FT-R3+ rounds. "
                             "Keeps the oldest copy of each filename.")
    args = parser.parse_args()

    if not any([args.rounds, args.push, args.sync_assignments,
                args.fix_instructions, args.pull, args.dedupe]):
        parser.error("provide --rounds N, --push, --sync-assignments, --fix-instructions, or --pull ROUND")

    cleanup_stale_assignments()

    if args.sync_assignments:
        if not ASSIGNMENTS_CSV.exists():
            print(f"ERROR: {ASSIGNMENTS_CSV} not found")
            sys.exit(1)
        from googleapiclient.discovery import build
        print("Connecting to Google Drive...")
        service = build("drive", "v3", credentials=gdrive_auth())
        print(f"Uploading {ASSIGNMENTS_CSV.relative_to(REPO_ROOT)} → Drive parent folder...")
        update_or_upload_file(service, ASSIGNMENTS_CSV, "assignments.csv", DRIVE_PARENT_ID)
        print("Done.")
        return

    if args.dedupe:
        from googleapiclient.discovery import build
        print("Connecting to Google Drive...")
        service = build("drive", "v3", credentials=gdrive_auth())
        dedupe_drive_pdfs(service)
        return

    if args.push:
        from googleapiclient.discovery import build
        print("Connecting to Google Drive...")
        service = build("drive", "v3", credentials=gdrive_auth())
        push_dry_run_batches(service)
        return

    if args.pull:
        from googleapiclient.discovery import build
        print("Connecting to Google Drive...")
        service = build("drive", "v3", credentials=gdrive_auth())
        pull_pdfs(service, args.pull)
        return

    if args.fix_instructions:
        from googleapiclient.discovery import build
        print("Connecting to Google Drive...")
        service = build("drive", "v3", credentials=gdrive_auth())
        rows = _read_assignments()
        live = [r for r in rows if r.get("drive_folder_id") not in ("", "DRY_RUN")]
        if not live:
            print("No pushed batches found.")
            return
        print(f"Found {len(live)} pushed batch(es): {', '.join(r['round'] for r in live)}")
        codebook_url = ensure_codebook_in_parent(service)
        for row in live:
            round_name     = row["round"]
            folder_id      = row["drive_folder_id"]
            out_dir        = OUTPUTS_DIR / "step14b" / round_name
            template_csv_fix = out_dir / f"coding_{round_name.lower()}_template.csv"
            if not template_csv_fix.exists():
                print(f"  {round_name}: coding template not found — skipping")
                continue
            n = len(pd.read_csv(template_csv_fix))
            instruction_pdf = out_dir / f"instruction_{round_name.lower()}.pdf"
            print(f"\n  {round_name}: regenerating instruction PDF ({n} papers)...")
            make_instruction_pdf(
                out_path=instruction_pdf,
                round_name=round_name,
                n_papers=n,
                drive_folder_id=folder_id,
                codebook_url=codebook_url,
            )
            instr_id = update_or_upload_file(service, instruction_pdf, instruction_pdf.name, folder_id)
            print(f"  Re-uploaded: {instruction_pdf.name}  (id={instr_id})")
        print("\nDone.")
        return

    rounds, base_seed = next_round_names(args.rounds)

    print(f"\n{'='*60}")
    print(f"  step14b_batch_draw")
    print(f"  Creating {args.rounds} batch(es): {', '.join(rounds)}")
    print(f"  Sample/batch: {args.sample}   Seeds: {base_seed}–{base_seed + args.rounds - 1}")
    print(f"  Min chars (PDF quality): {args.min_chars}   Dry-run: {args.dry_run}")
    print(f"{'='*60}")

    # ── Build pool once ───────────────────────────────────────────────────────
    pool = load_pool(args.min_chars)
    already: set[str] = load_already_sampled(exclude_round=None)
    for r in rounds:
        already -= _keys_for_round(r)  # don't double-count rounds being recreated
    n_already = int((pool["dedupe_key"].isin(already) | pool["doi"].isin(already)).sum())
    print(f"\nPool (all abstract-screened papers): {len(pool):,}")
    print(f"Already assigned elsewhere:          {n_already:,}")
    print(f"Available before first draw:         {len(pool) - n_already:,}")

    # ── Connect to Drive once (skip in dry-run) ───────────────────────────────
    service = None
    if not args.dry_run:
        from googleapiclient.discovery import build
        print("\nConnecting to Google Drive...")
        service = build("drive", "v3", credentials=gdrive_auth())

    # ── Draw each batch ───────────────────────────────────────────────────────
    for i, round_name in enumerate(rounds):
        drawn = draw_batch(
            round_name=round_name,
            pool=pool,
            already_sampled=already,
            sample_n=args.sample,
            seed=base_seed + i,
            min_chars=args.min_chars,
            dry_run=args.dry_run,
            service=service,
        )
        already |= drawn

    # ── Upload final assignments.csv once ────────────────────────────────────
    if not args.dry_run and service is not None:
        print(f"\nUploading assignments.csv to Drive parent folder...")
        update_or_upload_file(service, ASSIGNMENTS_CSV, "assignments.csv", DRIVE_PARENT_ID)

    print(f"\n{'='*60}")
    print(f"  All done — {args.rounds} batch(es) created")
    print(f"{'='*60}\n")


def _keys_for_round(round_name: str) -> set[str]:
    """Return DOIs/keys already on disk for a specific round (to allow re-running)."""
    keys: set[str] = set()
    # New rounds: coding template is the source of truth
    template_csv = OUTPUTS_DIR / "step14b" / round_name / f"coding_{round_name.lower()}_template.csv"
    if template_csv.exists():
        df = pd.read_csv(template_csv, usecols=["doi"])
        keys.update(df["doi"].dropna())
    # Legacy rounds: papers CSV (FT-R2x)
    for papers_csv in [
        OUTPUTS_DIR / "step14b" / round_name / f"papers_{round_name.lower()}.csv",
        ROUNDS_DOC_DIR / round_name / f"papers_{round_name.lower()}.csv",
    ]:
        if papers_csv.exists():
            df = pd.read_csv(papers_csv)
            if "dedupe_key" in df.columns:
                keys.update(df["dedupe_key"].dropna())
            if "doi" in df.columns:
                keys.update(df["doi"].dropna())
    return keys


if __name__ == "__main__":
    main()
