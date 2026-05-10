#!/usr/bin/env python3
"""
step14b_batch_draw.py — Draw random batches of papers for human full-text coding.

Creates one folder per batch under scripts/outputs/step14b/{round}/ containing:
  papers_{round}.csv            — sampled records (doi, title, year, file_path)
  coding_{round}_template.csv  — blank coding template pre-filled with doi/title/year
  instruction_{round}.pdf      — one-page coder briefing (Drive link, how to code)

Uploads to Google Drive ({round} subfolder under DRIVE_PARENT_ID):
  instruction PDF + coding template + PDFs subfolder with full-text PDFs

Shared codebook (CODEBOOK_FT.pdf in the Drive parent folder) is uploaded
separately via push_codebook_update.py — not touched here.

Coder assignment (name, email) is recorded separately in assignments.csv after
the round is created and a coder is hired. The script creates blank rows for
each round so the lead can fill them in.

Assignments are tracked in:
  documentation/coding/systematic-map/rounds/assignments.csv  (local)
  DRIVE_PARENT_ID/assignments.csv                              (Drive, updated in-place)

Usage:
  # Create 6 batches (auto-names FT-R2a … FT-R2f, seeds 42–47)
  conda run -n ilri01 python scripts/step14b_batch_draw.py --rounds 6

  # Create 1 more batch later (auto-detects FT-R2g, seed 48)
  conda run -n ilri01 python scripts/step14b_batch_draw.py --rounds 1

  # Dry-run to preview without uploading
  conda run -n ilri01 python scripts/step14b_batch_draw.py --rounds 6 --dry-run

Arguments:
  --rounds     Number of batches to create (required)
  --sample     Papers per batch (default 20)
  --min-chars  Minimum extracted character count to accept a paper (default 2000)
  --dry-run    Print what would happen; skip Drive upload
"""
from __future__ import annotations

import argparse
import csv
import re
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

MANIFEST_CSV    = OUTPUTS_DIR / "step13" / "step13_manifest.csv"
STEP14_RESULTS  = OUTPUTS_DIR / "step14" / "step14_results.csv"
ASSIGNMENTS_CSV = OUTPUTS_DIR / "step14b" / "assignments.csv"

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

MIN_FILE_SIZE_KB = 50  # fallback filter when step14 char count not available


# ── pool helpers ───────────────────────────────────────────────────────────────

def load_pool(min_chars: int) -> pd.DataFrame:
    """
    Build the eligible draw pool from step13 manifest.

    Keeps only records with a retrieved PDF. Applies character-count quality
    filter using step14 extracted text lengths where available, falling back
    to file size for records step14 hasn't processed yet.
    """
    manifest = pd.read_csv(
        MANIFEST_CSV,
        usecols=["dedupe_key", "doi", "title", "year", "pub", "file_path",
                 "file_size_kb", "status"],
        low_memory=False,
    )

    # Keep retrieved PDFs only
    manifest = manifest[
        (manifest["status"] == "retrieved") &
        manifest["file_path"].str.endswith(".pdf", na=False)
    ].copy()
    manifest["file_size_kb"] = pd.to_numeric(manifest["file_size_kb"], errors="coerce").fillna(0)

    # Enrich with step14 char counts and coverDate (year fallback) where available
    if STEP14_RESULTS.exists():
        s14 = pd.read_csv(
            STEP14_RESULTS,
            usecols=["dedupe_key", "s14_fulltext_chars", "coverDate"],
            low_memory=False,
        )
        s14["s14_fulltext_chars"] = pd.to_numeric(s14["s14_fulltext_chars"], errors="coerce")
        s14["s14_year"] = pd.to_numeric(s14["coverDate"], errors="coerce").apply(
            lambda x: int(x) if pd.notna(x) else float("nan")
        )
        s14 = s14.drop(columns=["coverDate"])
        manifest = manifest.merge(s14, on="dedupe_key", how="left")
        # Fill missing manifest year from step14 coverDate
        missing = manifest["year"].isna()
        manifest.loc[missing, "year"] = manifest.loc[missing, "s14_year"]
        manifest = manifest.drop(columns=["s14_year"])
    else:
        manifest["s14_fulltext_chars"] = float("nan")

    # Quality filter: prefer char count; fall back to file size
    has_chars  = manifest["s14_fulltext_chars"].notna()
    good_chars = has_chars & (manifest["s14_fulltext_chars"] >= min_chars)
    good_size  = ~has_chars & (manifest["file_size_kb"] >= MIN_FILE_SIZE_KB)
    manifest   = manifest[good_chars | good_size].copy()

    manifest = manifest[~manifest["doi"].isin(CALIBRATION_DOIS)]
    manifest = manifest.drop_duplicates("dedupe_key").reset_index(drop=True)

    # Fill missing years from DOI (e.g. 10.1016/j.cosust.2016.03.003 → 2016)
    _doi_year = re.compile(r"\b(19|20)\d{2}\b")
    def _year_from_doi(row):
        if pd.notna(row["year"]):
            return row["year"]
        m = _doi_year.search(str(row.get("doi", "") or ""))
        return int(m.group()) if m else float("nan")

    manifest["year"] = manifest.apply(_year_from_doi, axis=1)

    return manifest


def load_already_sampled(exclude_round: str | None = None) -> set[str]:
    """
    Return dedupe_keys already assigned in any prior batch (new + legacy locations).
    """
    sampled: set[str] = set()

    # New location: scripts/outputs/step14b/*/papers_*.csv
    for papers_csv in (OUTPUTS_DIR / "step14b").glob("*/papers_*.csv"):
        if exclude_round and papers_csv.parent.name == exclude_round:
            continue
        df = pd.read_csv(papers_csv, usecols=["dedupe_key"])
        sampled.update(df["dedupe_key"].dropna().tolist())

    # Legacy location: documentation/.../rounds/*/papers_*.csv
    for papers_csv in ROUNDS_DOC_DIR.glob("*/papers_*.csv"):
        round_name = papers_csv.parent.name
        if exclude_round and round_name == exclude_round:
            continue
        df = pd.read_csv(papers_csv)
        # Legacy files used DOI as key; add both doi and dedupe_key if present
        if "dedupe_key" in df.columns:
            sampled.update(df["dedupe_key"].dropna().tolist())
        if "doi" in df.columns:
            sampled.update(df["doi"].dropna().tolist())

    return sampled


def make_template(sample: pd.DataFrame) -> pd.DataFrame:
    tpl = sample[["doi", "title", "year"]].copy()
    # Convert float years (e.g. 2021.0) to clean integers; leave blank if missing
    tpl["year"] = pd.to_numeric(tpl["year"], errors="coerce").apply(
        lambda x: str(int(x)) if pd.notna(x) else ""
    )
    for field in TEMPLATE_FIELDS:
        tpl[field] = ""
    tpl["coder_id"] = ""
    tpl["notes"]    = ""
    return tpl


ROUND_PREFIX = "FT-R2"  # production batch prefix; R1 is reserved for calibration


def next_round_names(n: int) -> tuple[list[str], int]:
    """
    Auto-detect the next available round names and starting seed.

    Scans existing batch folders (new + legacy locations) to find the highest
    suffix letter already used under ROUND_PREFIX, then returns n names
    starting from the next letter and the corresponding base seed.

    Returns (round_names, base_seed).
    """
    import re

    used_letters: set[str] = set()
    pattern = re.compile(rf"^{re.escape(ROUND_PREFIX)}([a-z])$")

    for location in [OUTPUTS_DIR / "step14b", ROUNDS_DOC_DIR]:
        if location.exists():
            for child in location.iterdir():
                m = pattern.match(child.name)
                if m:
                    used_letters.add(m.group(1))

    # Also scan legacy papers CSVs in documentation rounds
    for papers_csv in ROUNDS_DOC_DIR.glob(f"{ROUND_PREFIX}*/papers_*.csv"):
        m = pattern.match(papers_csv.parent.name)
        if m:
            used_letters.add(m.group(1))

    start_idx = (string.ascii_lowercase.index(max(used_letters)) + 1) if used_letters else 0
    base_seed = 42 + start_idx  # seed 42 = first batch ('a'), 43 = 'b', etc.

    if start_idx + n > 26:
        raise ValueError(
            f"Cannot generate {n} more batches — only {26 - start_idx} letters remaining "
            f"(already used: {sorted(used_letters)})"
        )

    names = [f"{ROUND_PREFIX}{string.ascii_lowercase[start_idx + i]}" for i in range(n)]
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


# ── per-batch logic ────────────────────────────────────────────────────────────

def draw_batch(
    round_name: str,
    pool: pd.DataFrame,
    already_sampled: set[str],
    sample_n: int,
    seed: int,
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

    sample = available.sample(n=sample_n, random_state=seed).sort_values("dedupe_key")

    papers_csv   = out_dir / f"papers_{round_name.lower()}.csv"
    template_csv = out_dir / f"coding_{round_name.lower()}_template.csv"

    out = sample[["dedupe_key", "doi", "title", "year", "file_path"]].copy()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").apply(
        lambda x: str(int(x)) if pd.notna(x) else ""
    )
    out.to_csv(papers_csv, index=False)
    make_template(sample).to_csv(template_csv, index=False)
    print(f"  Written: {papers_csv.relative_to(REPO_ROOT)}")
    print(f"  Written: {template_csv.relative_to(REPO_ROOT)}")

    if dry_run:
        # Generate instruction PDF with placeholder link so it can be inspected
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

    # Drive: find or create round folder
    round_folder_id, created = get_or_create_folder(service, round_name, DRIVE_PARENT_ID)
    action = "Created" if created else "Using"
    print(f"  {action} Drive folder: {round_name}  (id={round_folder_id})")

    codebook_url = ensure_codebook_in_parent(service)

    # Instruction PDF with real Drive link
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

    # PDFs subfolder
    pdfs_folder_name = f"{round_name} pdfs"
    pdfs_folder_id, _ = get_or_create_folder(service, pdfs_folder_name, round_folder_id)

    print(f"  Uploading {len(sample)} PDFs to '{pdfs_folder_name}'...")
    errors: list[str] = []
    for i, (_, row) in enumerate(sample.iterrows(), 1):
        pdf_path = Path(row["file_path"])
        label = f"[{i:3d}/{len(sample)}]"
        if not pdf_path.exists():
            print(f"    {label} MISSING: {pdf_path.name}")
            errors.append(row.get("doi", row["dedupe_key"]))
            continue
        print(f"    {label} {pdf_path.name}", end=" ", flush=True)
        upload_file(service, pdf_path, pdf_path.name, pdfs_folder_id)
        print("✓")

    record_assignment(round_name, len(sample), round_folder_id)

    n_ok = len(sample) - len(errors)
    print(f"  Done: {n_ok}/{len(sample)} PDFs uploaded")
    if errors:
        print(f"  Missing: {errors[:5]}{'...' if len(errors) > 5 else ''}")
    print(f"  Drive: https://drive.google.com/drive/folders/{round_folder_id}")

    return set(sample["dedupe_key"].tolist())


# ── push dry-run batches to Drive ─────────────────────────────────────────────

def push_dry_run_batches(service) -> None:
    """
    Upload all dry-run batches (drive_folder_id == 'DRY_RUN') to Google Drive.

    For each pending batch:
      1. Re-generates the instruction PDF with the real Drive folder link.
      2. Uploads instruction PDF + coding template.
      3. Uploads full-text PDFs.
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

        papers_csv   = out_dir / f"papers_{round_name.lower()}.csv"
        template_csv = out_dir / f"coding_{round_name.lower()}_template.csv"

        if not papers_csv.exists() or not template_csv.exists():
            print(f"\n  {round_name}: local files missing — skipping (re-run --rounds to recreate)")
            continue

        papers = pd.read_csv(papers_csv)
        n      = len(papers)

        print(f"\n{'─'*60}")
        print(f"  Pushing: {round_name}  ({n} papers)")
        print(f"{'─'*60}")

        # Find or create Drive folder
        round_folder_id, created = get_or_create_folder(service, round_name, DRIVE_PARENT_ID)
        action = "Created" if created else "Using"
        print(f"  {action} Drive folder: {round_name}  (id={round_folder_id})")

        # Re-generate instruction PDF with real Drive link
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

        # PDFs subfolder
        pdfs_folder_name = f"{round_name} pdfs"
        pdfs_folder_id, _ = get_or_create_folder(service, pdfs_folder_name, round_folder_id)

        print(f"  Uploading {n} PDFs to '{pdfs_folder_name}'...")
        errors: list[str] = []
        for i, (_, pr) in enumerate(papers.iterrows(), 1):
            pdf_path = Path(pr["file_path"])
            label    = f"[{i:3d}/{n}]"
            if not pdf_path.exists():
                print(f"    {label} MISSING: {pdf_path.name}")
                errors.append(pr.get("doi", pr["dedupe_key"]))
                continue
            print(f"    {label} {pdf_path.name}", end=" ", flush=True)
            upload_file(service, pdf_path, pdf_path.name, pdfs_folder_id)
            print("✓")

        record_assignment(round_name, n, round_folder_id)
        n_ok = n - len(errors)
        print(f"  Done: {n_ok}/{n} PDFs uploaded")
        if errors:
            print(f"  Missing: {errors[:5]}{'...' if len(errors) > 5 else ''}")
        print(f"  Drive: https://drive.google.com/drive/folders/{round_folder_id}")

    print(f"\nUploading assignments.csv to Drive parent folder...")
    update_or_upload_file(service, ASSIGNMENTS_CSV, "assignments.csv", DRIVE_PARENT_ID)
    print(f"\n{'='*60}")
    print(f"  All done — {len(pending)} batch(es) pushed")
    print(f"{'='*60}\n")


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
                        help="Min extracted character count (default 2000)")
    parser.add_argument("--dry-run",           action="store_true")
    parser.add_argument("--fix-instructions",  action="store_true",
                        help="Re-upload codebook to Drive parent and regenerate instruction PDFs "
                             "for all already-pushed rounds")
    args = parser.parse_args()

    if not args.rounds and not args.push and not args.sync_assignments and not args.fix_instructions:
        parser.error("provide --rounds N, --push, --sync-assignments, or --fix-instructions")

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

    if args.push:
        from googleapiclient.discovery import build
        print("Connecting to Google Drive...")
        service = build("drive", "v3", credentials=gdrive_auth())
        push_dry_run_batches(service)
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
            papers_csv     = out_dir / f"papers_{round_name.lower()}.csv"
            if not papers_csv.exists():
                print(f"  {round_name}: local papers CSV not found — skipping")
                continue
            n = len(pd.read_csv(papers_csv))
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
    print(f"  Min chars: {args.min_chars}   Dry-run: {args.dry_run}")
    print(f"{'='*60}")

    # ── Build pool once ───────────────────────────────────────────────────────
    pool = load_pool(args.min_chars)
    already: set[str] = load_already_sampled(exclude_round=None)
    for r in rounds:
        already -= _keys_for_round(r)  # don't double-count rounds being recreated
    print(f"\nPool (quality-filtered PDFs): {len(pool):,}")
    print(f"Already assigned elsewhere:   {len(already):,}")
    print(f"Available before first draw:  {len(pool) - sum(pool['dedupe_key'].isin(already)):,}")

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
    """Return dedupe_keys already on disk for a specific round (to allow re-running)."""
    keys: set[str] = set()
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
