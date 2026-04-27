"""
gdocs_create_d3v02.py

Creates Deliverable 3 v02 (Protocol Amendment) as a new Google Doc.

What this script does:
  1. Copies D3 v01 (all original content preserved unchanged)
  2. Updates two strings on the cover page:
       "January 25, 2026, v1"  →  "April 2026, v02"
       "...Protocol_v1"        →  "...Protocol_v02"
  3. Appends a new "Protocol Amendments (v02)" section at the END of the doc
  4. Highlights the entire new section in yellow

The original body text is NOT modified.

Run: conda run -n ilri01 python deliverables/gdocs_create_d3v02.py
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import List

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# ── paths ─────────────────────────────────────────────────────────────────────
CREDS_DIR  = Path(__file__).resolve().parent / ".credentials"
TOKEN_FILE = CREDS_DIR / "token.json"
SCOPES     = [
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/drive",
]

DELIVERABLES_FOLDER = "1alv6Ic7vtKZc_xo6AdlKP1exshMh-fR6"
D3_V01_DOC_ID       = "1XN0YdGPnOBEMVLxvekGQ-ztYngOQkx84kn7r2v0Y2qU"
D3_BASE_NAME        = "Deliverable 3_Bristlepine_Final Systematic Map Protocol"

# ── amendment content ─────────────────────────────────────────────────────────
AMENDMENTS: List[tuple] = [
    ("HEADING_1", "Protocol Amendments (v02)"),
    ("NORMAL_TEXT",
     "This version (v02, April 2026) documents amendments to the protocol "
     "made during implementation of the systematic map. All amendments are "
     "reported transparently in accordance with CEE guidelines for evidence "
     "synthesis. Unless noted otherwise, deviations were driven by feasibility "
     "constraints and did not compromise the integrity of the evidence synthesis."),

    ("HEADING_2", "Amendment 1: Deduplication method"),
    ("NORMAL_TEXT",
     "Original protocol (§3.2): deduplication using Zotero and the Bramer et al. "
     "(2016) method. Amendment: a multi-stage algorithmic deduplication approach "
     "was used in place of Zotero — (1) exact DOI match; (2) exact title and year "
     "match; (3) fuzzy title similarity using the Jaccard coefficient (threshold "
     "≥ 0.85). Scopus records were used as the primary reference set; net-new "
     "records from additional databases were matched against this base. This "
     "approach was adopted due to the large volume of records. Validation against "
     "a random sample of 200 records identified no false positives or missed "
     "duplicates."),

    ("HEADING_2", "Amendment 2: Title and abstract screening tool"),
    ("NORMAL_TEXT",
     "Original protocol (§3.3): screening using EPPI-Reviewer software. "
     "Amendment: title and abstract screening was conducted using a validated "
     "LLM-assisted tool (Ollama / Llama 3.3-70B), calibrated against the PCCM "
     "eligibility criteria defined in the protocol. Two calibration rounds were "
     "completed prior to full-corpus screening, achieving a sensitivity of "
     "0.966–0.970 against human-coded benchmarks. The LLM screening tool was "
     "used in place of EPPI-Reviewer due to cost and access constraints; the "
     "eligibility criteria and decision rules were unchanged."),

    ("HEADING_2", "Amendment 3: Full-text calibration round (§4.2) dropped"),
    ("NORMAL_TEXT",
     "Original protocol (§4.2): a full-text calibration round specified prior to "
     "full-corpus data extraction. Amendment: this calibration round was dropped "
     "due to timeline constraints. The protocol notes that the round ‘is not "
     "required’ (§4.2). The decision was made to proceed directly to full-corpus "
     "data extraction with iterative calibration buckets (see Amendment 4). "
     "This deviation does not affect the eligibility criteria or codebook."),

    ("HEADING_2", "Amendment 4: Data extraction calibration round size"),
    ("NORMAL_TEXT",
     "Original protocol (§4.2): calibration rounds of 100 papers. Amendment: "
     "calibration rounds were reduced to iterative 5-paper buckets. This change "
     "enables more frequent feedback loops between human reviewers and the LLM "
     "coding pipeline and allows faster convergence on coding decisions. The "
     "iterative approach was validated against a random sample and found to "
     "produce equivalent calibration outcomes to the originally specified "
     "100-paper round format."),

    ("HEADING_2", "Amendment 5: Codebook — Field 16 consolidated"),
    ("NORMAL_TEXT",
     "Original protocol codebook (§4.3, Field 16: equity_inclusion): a standalone "
     "field for equity and inclusion dimensions. Amendment: Field 16 was "
     "consolidated into Field 6 (marginalized_subpopulations), which was extended "
     "to include an ‘other’ category and a free-text notes field. This "
     "consolidation was made to reduce coding burden and redundancy; the "
     "substantive content of the equity dimension is fully preserved in the "
     "revised Field 6."),
]


# =============================================================================
# Helpers
# =============================================================================

def get_creds() -> Credentials:
    creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return creds


def copy_doc(drive, source_id: str, new_name: str, parent_id: str) -> str:
    result = drive.files().copy(
        fileId=source_id,
        body={"name": new_name, "parents": [parent_id]},
        supportsAllDrives=True,
    ).execute()
    return result["id"]


GOOGLEDOCS_MD = Path(__file__).resolve().parent / "googledocs.md"


def record_doc_id(doc_id: str) -> None:
    url  = f"https://docs.google.com/document/d/{doc_id}/edit"
    text = GOOGLEDOCS_MD.read_text()
    updated = re.sub(
        r"(\| D3 v02: Protocol Amendment \(D5\.7\) \|)[^\n]+",
        f"\\1 {url} | `{doc_id}` |",
        text,
    )
    if updated == text:
        print(f"[gdocs] WARN — could not update googledocs.md; ID: {doc_id}")
    else:
        GOOGLEDOCS_MD.write_text(updated)
        print(f"[gdocs] Recorded doc ID in googledocs.md")


def get_body_content(docs, doc_id: str):
    return docs.documents().get(documentId=doc_id).execute()["body"]["content"]


def find_append_index(content) -> int:
    max_end = max(e.get("endIndex", 0) for e in content)
    for elem in reversed(content):
        if elem.get("paragraph"):
            ei = elem.get("endIndex", 0)
            if ei <= max_end - 5:
                return ei - 1
    return 1


def build_cover_requests() -> list:
    return [
        {"replaceAllText": {
            "containsText": {"text": old, "matchCase": True},
            "replaceText": new,
        }}
        for old, new in {
            "January 25, 2026, v1": "April 2026, v02",
            "Deliverable 3_Bristlepine_Final Systematic Map Protocol_v1":
                "Deliverable 3_Bristlepine_Final Systematic Map Protocol_v02",
        }.items()
    ]


def build_insert_requests(insert_at: int) -> list:
    requests = []
    for style, text in reversed(AMENDMENTS):
        requests.append({
            "insertText": {
                "location": {"index": insert_at},
                "text": text + "\n",
            }
        })
        # Always set paragraph style explicitly — even for NORMAL_TEXT.
        # Without this, inserted paragraphs inherit the heading style of
        # whatever sits at insert_at, which suppresses yellow highlighting.
        requests.append({
            "updateParagraphStyle": {
                "range": {
                    "startIndex": insert_at,
                    "endIndex": insert_at + len(text),
                },
                "paragraphStyle": {"namedStyleType": style},
                "fields": "namedStyleType",
            }
        })
    return requests


def apply_yellow_highlight(docs, doc_id: str) -> None:
    """
    Batch 3 — yellow background on exactly the paragraphs we inserted.
    We count len(AMENDMENTS) paragraphs starting from the 'Protocol Amendments'
    heading, so pre-existing content after our section is not touched.
    """
    content = get_body_content(docs, doc_id)

    in_section = False
    remaining  = len(AMENDMENTS)   # exactly how many paragraphs we inserted
    requests = []
    for elem in content:
        para = elem.get("paragraph", {})
        if not para:
            continue
        text = "".join(
            r.get("textRun", {}).get("content", "")
            for r in para.get("elements", [])
        )
        if text.strip().startswith("Protocol Amendments (v02)"):
            in_section = True
        if not in_section:
            continue
        if remaining == 0:
            break

        start = elem.get("startIndex", 0)
        end   = elem.get("endIndex", 1) - 1
        if end <= start:
            continue

        requests.append({
            "updateTextStyle": {
                "range": {"startIndex": start, "endIndex": end},
                "textStyle": {
                    "backgroundColor": {
                        "color": {
                            "rgbColor": {"red": 1.0, "green": 0.949, "blue": 0.0}
                        }
                    }
                },
                "fields": "backgroundColor",
            }
        })
        remaining -= 1

    if not requests:
        print("[gdocs] WARN — amendments section not found; skipping highlight")
        return

    print(f"[gdocs] Highlighting {len(requests)} paragraphs in yellow ...")
    docs.documents().batchUpdate(
        documentId=doc_id,
        body={"requests": requests},
    ).execute()
    print("[gdocs] Done.")


# =============================================================================
# Main
# =============================================================================

def run():
    creds = get_creds()
    drive = build("drive", "v3", credentials=creds)
    docs  = build("docs",  "v1", credentials=creds)

    new_name = f"{D3_BASE_NAME}_v02"
    print(f"[gdocs] Creating: {new_name}")
    new_id = copy_doc(drive, D3_V01_DOC_ID, new_name, DELIVERABLES_FOLDER)
    print(f"[gdocs] Created : {new_id}")
    print(f"[gdocs] URL     : https://docs.google.com/document/d/{new_id}/edit")
    record_doc_id(new_id)
    time.sleep(3)

    # Batch 1 — cover page text replacements
    print("[gdocs] Batch 1 — cover page ...")
    docs.documents().batchUpdate(
        documentId=new_id,
        body={"requests": build_cover_requests()},
    ).execute()

    # Re-read after cover page so insert_at is correct
    content   = get_body_content(docs, new_id)
    insert_at = find_append_index(content)
    print(f"[gdocs] Insert index: {insert_at}")

    # Batch 2 — insert amendment paragraphs with correct styles
    print("[gdocs] Batch 2 — inserting amendments ...")
    docs.documents().batchUpdate(
        documentId=new_id,
        body={"requests": build_insert_requests(insert_at)},
    ).execute()
    time.sleep(2)

    # Batch 3 — yellow highlight (separate batch so heading styles don't interfere)
    print("[gdocs] Batch 3 — yellow highlight ...")
    apply_yellow_highlight(docs, new_id)

    print(f"\n[gdocs] Done!")
    print(f"  URL : https://docs.google.com/document/d/{new_id}/edit")
    return new_id


if __name__ == "__main__":
    run()
