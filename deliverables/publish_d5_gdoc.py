"""
publish_d5_gdoc.py

1. Trashes the old D5 GDoc (preserves D4 watermark/formatting by starting fresh)
2. Copies D4 v02 → new D5 GDoc
3. Targeted text replacements (title, date, numbers, removes "Preliminary")
4. Uploads all figure PNGs to Google Drive, makes them publicly readable
5. Inserts each figure inline at the right location in the document
6. Updates deliverables_tracker.md with the new Doc ID

Run: python3 deliverables/publish_d5_gdoc.py
"""

from __future__ import annotations

import time
from pathlib import Path

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

CREDS_DIR  = Path(__file__).resolve().parent / ".credentials"
TOKEN_FILE = CREDS_DIR / "token.json"
SCOPES     = [
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/drive",
]

ROOT             = Path(__file__).resolve().parent.parent
STEP16           = ROOT / "scripts" / "outputs" / "step16"
HUMAN            = STEP16 / "interactive" / "human"

DELIVERABLES_FOLDER = "1alv6Ic7vtKZc_xo6AdlKP1exshMh-fR6"
D4_V02_DOC_ID       = "14Hejjfr63HdxH21bqBxLZe3AsXxmLBrXYqJJk7ZbBxU"
OLD_D5_DOC_ID       = "14UaK_9lXCuE_AsOYXNvfImYlEndLUL4AM0lVz4thEAY"   # broken — trash it
D5_BASE_NAME        = "Deliverable 5_Bristlepine_Final Systematic Map"

# Figure PNGs — in order of appearance in the document
# (label, local path, caption, width_pt, height_pt)
FIGURES = [
    (
        "roses",
        STEP16 / "roses_flow.png",
        "Figure 1. ROSES flow diagram — record flow across all 29 sources and four screening stages (n = 39,113 identified; 2,368 included after full-text screening).",
        460, 300,
    ),
    (
        "egm_human",
        HUMAN / "evidence_gap_map.png",
        "Figure 2. Evidence gap map — human-coded results (n = 86). Blue bubbles = process domains; green = outcome domains; grey = evidence gaps. Bubble area ∝ number of studies.",
        460, 320,
    ),
    (
        "geo_map",
        STEP16 / "geographic_map.png",
        "Figure 3. Geographic distribution of included studies. Countries by study count; multi-country studies counted in each country.",
        460, 260,
    ),
    (
        "geo_bar",
        STEP16 / "geographic_bar.png",
        "Figure 4. Top countries by study count.",
        460, 260,
    ),
    (
        "method",
        STEP16 / "methodology_bar.png",
        "Figure 5. Distribution of methodological approaches across included studies.",
        440, 240,
    ),
    (
        "temporal",
        STEP16 / "temporal_trends.png",
        "Figure 6. Temporal trends in publication volume (all databases, final).",
        440, 240,
    ),
    (
        "equity",
        STEP16 / "equity_bar.png",
        "Figure 7. Equity and inclusion dimensions across included studies. Red bar = studies with no marginalized group focus.",
        440, 240,
    ),
    (
        "saturation",
        STEP16 / "saturation.png",
        "Figure 8. Information saturation curve. Top: cumulative unique categories as % of final total. Bottom: new categories per batch. All dimensions plateau by 49 papers.",
        460, 300,
    ),
    (
        "llm_human",
        STEP16 / "llm_vs_human.png",
        "Figure 9. LLM vs human comparison across key categorical variables. Human (amber, n=86) vs LLM (teal, n=2,368).",
        460, 260,
    ),
]


# ── text replacements ──────────────────────────────────────────────────────────
# Applied in order; "old" → "new"
REPLACEMENTS = [
    # Title / deliverable number
    ("Deliverable 4: First Draft Systematic Map (Preliminary)",
     "Deliverable 5: Final Systematic Map"),
    ("Deliverable 4_Bristlepine_First Draft Systematic Map_v02",
     "Deliverable 5_Bristlepine_Final Systematic Map_v01"),
    ("First Draft Systematic Map (Preliminary)",
     "Final Systematic Map"),
    ("First Draft Systematic Map",
     "Final Systematic Map"),
    # Date / version
    ("April 2026, v02",  "June 2026, v01"),
    ("April 2026",       "June 2026"),
    # Remove preliminary flags
    ("(Preliminary)",    ""),
    ("(preliminary)",    ""),
    ("Preliminary ",     ""),
    ("preliminary ",     ""),
    # Deliverable cross-reference
    ("will be reported in Deliverable 5",
     "are reported in this document (Deliverable 5)"),
    ("the final systematic map (Deliverable 5)",
     "the deliverable you are now reading (Deliverable 5)"),
    # Coded record count
    ("coded dataset of 2,093 studies",    "coded dataset of 2,368 studies"),
    ("2,093 records were successfully",   "2,368 records were successfully"),
    # Full-text retrieval
    ("2,644 of 6,218",                    "2,644 of 6,218 (Scopus); 3,476 total across all databases"),
    ("2,644 full texts",                  "3,476 full texts"),
    ("43%) were retrieved",               "40%) were retrieved"),
    ("3,574 require manual",              "5,232 require manual"),
    # Net-new screening
    ("8,187 additional records",          "8,895 additional records (across all non-Scopus databases)"),
    ("2,340 were included",               "2,340 were included at title/abstract stage"),
    # Human coding — update section 3.4
    ("Data extraction proceeded iteratively in batches",
     "Data extraction proceeded iteratively in batches of 20 papers"),
    # Screening tool name (D4 used Llama3; D5 used qwen2.5)
    ("Ollama/Llama3",    "Ollama/qwen2.5:14b"),
    # Section 4 title
    ("4.  Preliminary Results",  "4.  Final Results"),
    ("4. Preliminary Results",   "4. Final Results"),
    # Figure captions — replace D4's preliminary captions with D5 final versions
    ("Figure 3. Geographic distribution of coded studies (all databases, preliminary).",
     "Figure 3. Geographic distribution of included studies (all databases, final)."),
    ("Figure 4. Regional distribution of coded studies by country  (all databases, preliminary).",
     "Figure 4. Top countries by study count (all databases, final)."),
    ("Figure 5. Distribution of methodological approaches across coded studies  (all databases, preliminary).",
     "Figure 5. Methodological approaches across included studies (all databases, final)."),
    ("Figure 6. Temporal trends in publication volume, 1990–2024 (all databases, preliminary).",
     "Figure 6. Temporal trends in publication volume, 1990–2025 (all databases, final)."),
    # Abstract screening count
    ("25,208 unique records remained. Title and abstract screening identified 8,558",
     "25,208 unique records remained. Title and abstract screening identified 8,558"),
    # D4 note at top
    ("Note: This document is a preliminary first draft of the systematic map, based primarily on Scopus database records and illustrativ",
     "Note: This document presents the final systematic map."),
]


# ── helpers ───────────────────────────────────────────────────────────────────

def get_creds() -> Credentials:
    creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return creds


def copy_doc(drive, source_id: str, name: str, folder: str) -> str:
    res = drive.files().copy(
        fileId=source_id,
        body={"name": name, "parents": [folder]},
        supportsAllDrives=True,
    ).execute()
    return res["id"]


def trash_doc(drive, doc_id: str) -> None:
    try:
        drive.files().update(fileId=doc_id, body={"trashed": True},
                             supportsAllDrives=True).execute()
        print(f"[d5] Trashed old GDoc: {doc_id}")
    except Exception as e:
        print(f"[d5] Could not trash {doc_id}: {e}")


def upload_figure(drive, png: Path, folder: str) -> str | None:
    """Upload PNG to Drive, make public, return public download URL."""
    if not png.exists():
        print(f"[d5] WARN: PNG not found — {png}")
        return None
    try:
        meta  = {"name": png.name, "parents": [folder]}
        media = MediaFileUpload(str(png), mimetype="image/png", resumable=False)
        file  = drive.files().create(body=meta, media_body=media,
                                     fields="id", supportsAllDrives=True).execute()
        fid   = file["id"]
        drive.permissions().create(
            fileId=fid,
            body={"type": "anyone", "role": "reader"},
        ).execute()
        url = f"https://drive.google.com/uc?export=download&id={fid}"
        print(f"[d5]   uploaded {png.name} → {fid}")
        return url
    except Exception as e:
        print(f"[d5] WARN: upload failed for {png.name}: {e}")
        return None


def get_body(docs, doc_id: str):
    return docs.documents().get(documentId=doc_id).execute()["body"]["content"]


def para_text(elem) -> str:
    para = elem.get("paragraph", {})
    return "".join(r.get("textRun", {}).get("content", "") for r in para.get("elements", []))


def find_para(content, search: str) -> tuple[int, int] | tuple[None, None]:
    """Return (startIndex, endIndex) of first paragraph containing search."""
    for elem in content:
        txt = para_text(elem)
        if search in txt:
            return elem.get("startIndex"), elem.get("endIndex")
    return None, None


# ── figure insertion ──────────────────────────────────────────────────────────

# Which text to search for to locate each figure's insertion point
# We insert the image BEFORE the caption paragraph.
FIGURE_ANCHORS = {
    "roses":      "Figure 1 presents the ROSES",
    "egm_human":  "Figure 2 presents the",
    "geo_map":    "Figure 3. Geographic distribution",
    "geo_bar":    "Figure 4.",
    "method":     "Figure 5.",
    "temporal":   "Figure 6. Temporal trends",
    "equity":     "Figure 7.",      # new — appended after Figure 6
    "saturation": "Figure 8.",      # new
    "llm_human":  "Figure 9.",      # new
}

# For "roses" and "egm_human", the anchor is a PROSE paragraph that describes
# the figure (insert AFTER this paragraph, before the next one).
# For the others, the anchor IS the caption (insert BEFORE the caption).
INSERT_AFTER = {"roses", "egm_human"}

# Figures 7, 8, 9 don't exist in D4 — we'll ADD them at end of §4.2 (EGM section)
NEW_FIGURES = {"equity", "saturation", "llm_human"}


def insert_figure_at(docs, doc_id: str, index: int, url: str,
                     caption: str, w_pt: float, h_pt: float) -> None:
    """Insert inline image + caption at given index."""
    requests = []

    # 1. Insert the caption text first (at index) — we'll insert image before it
    requests.append({
        "insertText": {
            "location": {"index": index},
            "text": caption + "\n",
        }
    })
    # 2. Style caption as italic, small
    requests.append({
        "updateTextStyle": {
            "range": {"startIndex": index, "endIndex": index + len(caption)},
            "textStyle": {
                "italic": True,
                "fontSize": {"magnitude": 9, "unit": "PT"},
            },
            "fields": "italic,fontSize",
        }
    })

    docs.documents().batchUpdate(documentId=doc_id, body={"requests": requests}).execute()
    time.sleep(1)

    # 3. Insert image BEFORE the caption (at same index)
    docs.documents().batchUpdate(
        documentId=doc_id,
        body={"requests": [{
            "insertInlineImage": {
                "location": {"index": index},
                "uri": url,
                "objectSize": {
                    "width":  {"magnitude": w_pt, "unit": "PT"},
                    "height": {"magnitude": h_pt, "unit": "PT"},
                },
            }
        }]}
    ).execute()
    time.sleep(1)


# ── main ──────────────────────────────────────────────────────────────────────

def run():
    creds = get_creds()
    drive = build("drive", "v3", credentials=creds)
    docs  = build("docs",  "v1", credentials=creds)

    # 1. Trash old broken D5 GDoc
    trash_doc(drive, OLD_D5_DOC_ID)

    # 2. Copy D4 v02 → fresh D5 GDoc (preserves all formatting, watermark, styles)
    print("[d5] Copying D4 v02 → D5 v01...")
    new_id = copy_doc(drive, D4_V02_DOC_ID,
                      "Deliverable 5_Bristlepine_Final Systematic Map_v01",
                      DELIVERABLES_FOLDER)
    print(f"[d5] Created: {new_id}")
    print(f"[d5] URL: https://docs.google.com/document/d/{new_id}/edit")
    time.sleep(3)

    # 3. Text replacements
    print("[d5] Applying text replacements...")
    valid = [(o, n) for o, n in REPLACEMENTS if o != n]
    reqs  = [
        {"replaceAllText": {
            "containsText": {"text": o, "matchCase": True},
            "replaceText": n,
        }}
        for o, n in valid
    ]
    res = docs.documents().batchUpdate(documentId=new_id, body={"requests": reqs}).execute()
    replaced = sum(
        r.get("replaceAllTextResponse", {}).get("occurrencesChanged", 0)
        for r in res.get("replies", [])
    )
    print(f"[d5] {replaced} text substitutions made.")
    time.sleep(2)

    # 4. Upload figures to Drive and make public
    print("[d5] Uploading figures to Drive...")
    figure_urls: dict[str, str | None] = {}
    for label, png_path, caption, w, h in FIGURES:
        url = upload_figure(drive, png_path, DELIVERABLES_FOLDER)
        figure_urls[label] = url
        time.sleep(0.3)

    # 5. Insert figures into document
    # Process in REVERSE order so that earlier insertions don't shift later indices.
    # But we need to re-read doc after each insertion anyway (insertions shift indices).

    print("[d5] Inserting figures...")

    for label, png_path, caption, w_pt, h_pt in reversed(FIGURES):
        url = figure_urls.get(label)
        if not url:
            print(f"[d5]   SKIP {label} — no URL")
            continue

        anchor = FIGURE_ANCHORS.get(label)
        if not anchor:
            continue

        # Re-read doc to get fresh indices
        content = get_body(docs, new_id)

        if label in NEW_FIGURES:
            # Figures 7, 8, 9: append after the end of section 4 text.
            # Find the last "Searchable Database" heading to insert before it.
            si, ei = find_para(content, "5.  Searchable Database")
            if si is None:
                si, ei = find_para(content, "Searchable Database")
            if si is None:
                print(f"[d5]   WARN: could not find anchor for new figure {label}")
                continue
            # Insert at si (before "5. Searchable Database" heading)
            insert_figure_at(docs, new_id, si, url, caption, w_pt, h_pt)
            print(f"[d5]   Inserted {label} before §5")

        elif label in INSERT_AFTER:
            # Insert AFTER the prose paragraph that mentions the figure
            si, ei = find_para(content, anchor)
            if si is None:
                print(f"[d5]   WARN: anchor not found for {label}: '{anchor}'")
                continue
            # ei is the end of the prose paragraph — insert at ei
            insert_figure_at(docs, new_id, ei, url, caption, w_pt, h_pt)
            print(f"[d5]   Inserted {label} after prose para (idx {ei})")

        else:
            # Insert BEFORE the caption paragraph
            si, ei = find_para(content, anchor)
            if si is None:
                print(f"[d5]   WARN: anchor not found for {label}: '{anchor}'")
                continue
            insert_figure_at(docs, new_id, si, url, caption, w_pt, h_pt)
            print(f"[d5]   Inserted {label} before caption (idx {si})")

        time.sleep(1)

    # 6. Update deliverables_tracker.md
    tracker = Path(__file__).resolve().parent.parent / "deliverables" / "deliverables_tracker.md"
    if tracker.exists():
        text = tracker.read_text()
        # Replace both old D5 entries with the new doc ID
        import re
        text = re.sub(
            r"(\*\*D5\*\*[^\n]+)\| \[v01 \(GDoc\)\]\([^)]+\)",
            f"\\1| [v01 (GDoc)](https://docs.google.com/document/d/{new_id}/edit)",
            text,
        )
        tracker.write_text(text)
        print("[d5] Updated deliverables_tracker.md")

    print(f"\n[d5] ✓ Done.")
    print(f"  URL: https://docs.google.com/document/d/{new_id}/edit")
    print(f"  ID:  {new_id}")
    print()
    print("Review the doc, then export as Word (.docx) and upload to Teams.")
    return new_id


if __name__ == "__main__":
    run()
