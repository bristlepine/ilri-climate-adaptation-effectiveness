"""
gdocs_create_d5.py

Creates D5 v01 from D4 v02:
1. Copies D4 v02 → D5 v01
2. Updates title: "First Draft Systematic Map" → "Final Systematic Map"
3. Removes "Preliminary" labels
4. Updates cover page date/version
5. Updates in-text statistics with D5 pipeline numbers
6. Updates pipeline table with final FT screening numbers

Pipeline stats for D5 (as of 2026-06-01):
  - Scopus abstract included:  6,218  (step12, unchanged)
  - All-DB FT retrieved:       3,476  (step13 total, Scopus 2,644 + others 832)
  - FT screened (combined):    3,464  (step14 rows_screened_with_fulltext)
  - FT included (LLM):         2,368  (step14 Include decisions)
  - Human coding batch FT-R2a: 86 confirmed includes (step15b)
  - Net-new multi-DB after dedup: 8,187 (unchanged from D4)

Run: conda run -n ilri01 python deliverables/gdocs_create_d5.py
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# ── credentials ───────────────────────────────────────────────────────────────
CREDS_DIR  = Path(__file__).resolve().parent / ".credentials"
TOKEN_FILE = CREDS_DIR / "token.json"
SCOPES     = [
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/drive",
]

DELIVERABLES_FOLDER = "1alv6Ic7vtKZc_xo6AdlKP1exshMh-fR6"
D4_V02_DOC_ID       = "14Hejjfr63HdxH21bqBxLZe3AsXxmLBrXYqJJk7ZbBxU"
D5_BASE_NAME        = "Deliverable 5_Bristlepine_Final Systematic Map"

# ── D5 pipeline stats ─────────────────────────────────────────────────────────
STATS = {
    "scopus_total":        17_021,
    "abstract_included":    6_218,
    "abstract_excluded":   10_803,
    "missing_abstract":     1_328,
    "ft_retrieved_scopus":  2_644,
    "ft_retrieved_total":   3_476,
    "ft_screened":          3_464,
    "ft_included":          2_368,
    "ft_excluded":          1_096,
    "coded_records":        2_368,
    "human_included":          86,
    "net_new_after_dedup":  8_187,
    "net_new_included":     2_340,  # abstract-screening included from non-Scopus
    "search_date":         "January 2026",
    "report_date":         "June 2026",
    "version":             "v01",
}

# ── pipeline table ────────────────────────────────────────────────────────────
TABLE_HEADERS = [
    "Database",
    "Records\nReturned",
    "After\nDeduplication",
    "Abstract Screening\nIncluded",
    "Full Texts\nRetrieved",
    "Full Texts\nScreened",
    "Included\nfor Coding",
]

TABLE_ROWS = [
    ["Scopus",                   "17,021", "17,021", "6,218", "2,644", "2,644", "—"],
    ["Web of Science",           "15,179",  "4,683", "1,140",   "552",   "552", "—"],
    ["CAB Abstracts",             "5,723",  "3,229", "1,133",   "260",   "260", "—"],
    ["Academic Search Premier",   "1,187",    "274",    "66",    "20",    "20", "—"],
    ["AGRIS",                         "3",      "1",     "1",     "0",     "—", "—"],
    ["Total",                    "39,113", "25,208", "8,558", "3,476", "3,464", "2,368"],
]

TABLE_FOOTNOTE = (
    "Full-text screening was conducted on all automatically retrieved records "
    "across all databases (n = 3,476 retrieved; 3,464 screened; 12 could not be "
    "processed due to format or access issues). Records requiring manual full-text "
    "access (n = 5,232) are not included in these totals. Coding column shows "
    "all LLM-screened included records combined across all databases."
)

# Brand colours
GREEN    = {"red": 33/255,  "green": 71/255,  "blue": 46/255}
CHARCOAL = {"red": 60/255,  "green": 60/255,  "blue": 60/255}
WHITE    = {"red": 1.0,     "green": 1.0,     "blue": 1.0}


# =============================================================================
# Helpers (same as gdocs_create_d4v02.py)
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


def next_version_name(drive, folder_id: str, base_name: str) -> str:
    results = drive.files().list(
        q=f"'{folder_id}' in parents and trashed=false and name contains '{base_name}'",
        fields="files(name)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()
    max_v = 0
    for f in results.get("files", []):
        m = re.search(r"_v(\d+)$", f["name"])
        if m:
            max_v = max(max_v, int(m.group(1)))
    v = max_v + 1
    return f"{base_name}_v{v:02d}"


def get_body(docs, doc_id: str):
    return docs.documents().get(documentId=doc_id).execute()["body"]["content"]


def para_text(elem) -> str:
    para = elem.get("paragraph", {})
    return "".join(r.get("textRun", {}).get("content", "") for r in para.get("elements", []))


def find_range_to_delete(content) -> tuple[int, int]:
    start = end = None
    for elem in content:
        txt = para_text(elem)
        if start is None and "Table 1." in txt:
            start = elem.get("startIndex", 0)
        if start is not None and end is None and "Figure 2 presents" in txt:
            end = elem.get("startIndex", 0)
            break
    return start, end


def find_table_elem(content):
    for elem in content:
        if "table" in elem:
            return elem
    return None


def populate_table(docs, doc_id: str, table_elem: dict) -> None:
    rows = table_elem["table"]["tableRows"]
    requests = []
    all_rows = [TABLE_HEADERS] + TABLE_ROWS
    for r_idx, row_data in enumerate(all_rows):
        row = rows[r_idx]
        for c_idx, cell_text in enumerate(row_data):
            cell = row["tableCells"][c_idx]
            para = cell["content"][0]
            insert_at = para.get("startIndex", cell.get("startIndex", 0))
            requests.append({
                "insertText": {
                    "location": {"index": insert_at},
                    "text": cell_text,
                }
            })
    docs.documents().batchUpdate(documentId=doc_id, body={"requests": requests}).execute()
    print("[gdocs] Table cells populated.")
    time.sleep(1)


def style_table(docs, doc_id: str, table_elem: dict) -> None:
    rows = table_elem["table"]["tableRows"]
    n_cols = len(TABLE_HEADERS)
    requests = []
    for c_idx in range(n_cols):
        cell = rows[0]["tableCells"][c_idx]
        requests.append({
            "updateTableCellStyle": {
                "tableRange": {
                    "tableCellLocation": {
                        "tableStartLocation": {"index": table_elem["startIndex"]},
                        "rowIndex": 0,
                        "columnIndex": c_idx,
                    },
                    "rowSpan": 1,
                    "columnSpan": 1,
                },
                "tableCellStyle": {"backgroundColor": {"color": {"rgbColor": GREEN}}},
                "fields": "backgroundColor",
            }
        })
        para = cell["content"][0]
        si = para.get("startIndex", 0)
        ei = cell.get("endIndex", si + len(TABLE_HEADERS[c_idx]) + 1)
        requests.append({
            "updateTextStyle": {
                "range": {"startIndex": si, "endIndex": ei - 1},
                "textStyle": {
                    "foregroundColor": {"color": {"rgbColor": WHITE}},
                    "bold": True,
                },
                "fields": "foregroundColor,bold",
            }
        })
    total_row_idx = len(rows) - 1
    for c_idx in range(n_cols):
        cell = rows[total_row_idx]["tableCells"][c_idx]
        requests.append({
            "updateTableCellStyle": {
                "tableRange": {
                    "tableCellLocation": {
                        "tableStartLocation": {"index": table_elem["startIndex"]},
                        "rowIndex": total_row_idx,
                        "columnIndex": c_idx,
                    },
                    "rowSpan": 1,
                    "columnSpan": 1,
                },
                "tableCellStyle": {"backgroundColor": {"color": {"rgbColor": CHARCOAL}}},
                "fields": "backgroundColor",
            }
        })
        para = cell["content"][0]
        si = para.get("startIndex", 0)
        ei = cell.get("endIndex", si + 10)
        requests.append({
            "updateTextStyle": {
                "range": {"startIndex": si, "endIndex": ei - 1},
                "textStyle": {
                    "foregroundColor": {"color": {"rgbColor": WHITE}},
                    "bold": True,
                },
                "fields": "foregroundColor,bold",
            }
        })
    if requests:
        docs.documents().batchUpdate(documentId=doc_id, body={"requests": requests}).execute()
        print("[gdocs] Table styled.")


# =============================================================================
# Main
# =============================================================================

def run():
    creds = get_creds()
    drive = build("drive", "v3", credentials=creds)
    docs  = build("docs",  "v1", credentials=creds)

    # 1. Copy D4 v02 → D5 v01
    new_name = next_version_name(drive, DELIVERABLES_FOLDER, D5_BASE_NAME)
    print(f"[gdocs] Creating: {new_name}")
    new_id = copy_doc(drive, D4_V02_DOC_ID, new_name, DELIVERABLES_FOLDER)
    print(f"[gdocs] Created : {new_id}")
    print(f"[gdocs] URL     : https://docs.google.com/document/d/{new_id}/edit")
    time.sleep(3)

    # 2. Text replacements
    print("[gdocs] Batch 1 — updating title, date, and stats ...")
    replacements = {
        # Title and deliverable number
        "Deliverable 4: First Draft Systematic Map":
            "Deliverable 5: Final Systematic Map",
        "Deliverable 4_Bristlepine_First Draft Systematic Map_v02":
            "Deliverable 5_Bristlepine_Final Systematic Map_v01",
        "First Draft Systematic Map":
            "Final Systematic Map",
        # Remove "Preliminary" qualifiers
        " (Preliminary)": "",
        "(Preliminary) ": "",
        "Preliminary — ": "",
        ", preliminary": "",
        " preliminary": "",
        # Date and version
        "April 2026, v02":  "June 2026, v01",
        "April 2026":       "June 2026",
        # Coded records
        f"coded dataset of {2_093:,} studies":  f"coded dataset of {STATS['coded_records']:,} studies",
        f"{2_093:,} records were successfully": f"{STATS['coded_records']:,} records were successfully",
        # FT retrieval (Scopus)
        f"{2_644:,} full texts":                f"{STATS['ft_retrieved_scopus']:,} full texts",
        f"{2_644:,} of {6_218:,}":              f"{STATS['ft_retrieved_scopus']:,} of {STATS['abstract_included']:,}",
        "43%) were retrieved":
            f"{round(STATS['ft_retrieved_scopus']/STATS['abstract_included']*100)}%) were retrieved",
        # Remaining manual (Scopus)
        f"{3_574:,} require manual":            f"{6_218 - 2_644:,} require manual",
        # Net-new multi-DB (abstract screening)
        f"{8_187:,} additional records":        f"{STATS['net_new_after_dedup']:,} additional records",
        f"{2_340:,} were included":             f"{STATS['net_new_included']:,} were included",
        # D5 self-references
        "will be reported in Deliverable 5":    "are reported in this deliverable (Deliverable 5)",
        # Human coding note (if referenced)
        "full-text calibration":                "human validation",
    }
    r1_reqs = [
        {"replaceAllText": {
            "containsText": {"text": old, "matchCase": True},
            "replaceText": new,
        }}
        for old, new in replacements.items()
        if old != new
    ]
    if r1_reqs:
        docs.documents().batchUpdate(documentId=new_id, body={"requests": r1_reqs}).execute()
    print(f"[gdocs] Applied {len(r1_reqs)} text replacements.")
    time.sleep(1)

    # 3. Replace pipeline table
    print("[gdocs] Batch 2 — replacing pipeline table ...")
    content = get_body(docs, new_id)
    tbl_start, tbl_end = find_range_to_delete(content)

    if tbl_start is None or tbl_end is None:
        print("[gdocs] WARN — could not find Table 1 block; skipping table replacement.")
        print("[gdocs]        Update the pipeline table manually in the Google Doc.")
    else:
        n_rows = 1 + len(TABLE_ROWS)
        n_cols = len(TABLE_HEADERS)
        batch2 = [
            {"deleteContentRange": {
                "range": {"startIndex": tbl_start, "endIndex": tbl_end}
            }},
            {"insertTable": {
                "location": {"index": tbl_start},
                "rows": n_rows,
                "columns": n_cols,
            }},
        ]
        docs.documents().batchUpdate(documentId=new_id, body={"requests": batch2}).execute()
        print("[gdocs] Table inserted.")
        time.sleep(2)

        content = get_body(docs, new_id)
        table_elem = find_table_elem(content)
        if table_elem:
            populate_table(docs, new_id, table_elem)
            time.sleep(1)
            content = get_body(docs, new_id)
            table_elem = find_table_elem(content)
            if table_elem:
                style_table(docs, new_id, table_elem)
                time.sleep(1)
                content = get_body(docs, new_id)
                table_elem = find_table_elem(content)
                if table_elem:
                    after_table = table_elem.get("endIndex", 0)
                    docs.documents().batchUpdate(
                        documentId=new_id,
                        body={"requests": [
                            {"insertText": {
                                "location": {"index": after_table},
                                "text": TABLE_FOOTNOTE + "\n",
                            }},
                            {"updateTextStyle": {
                                "range": {
                                    "startIndex": after_table,
                                    "endIndex": after_table + len(TABLE_FOOTNOTE),
                                },
                                "textStyle": {"italic": True, "fontSize": {"magnitude": 9, "unit": "PT"}},
                                "fields": "italic,fontSize",
                            }},
                        ]},
                    ).execute()
                    print("[gdocs] Footnote inserted.")

    # 4. Record in deliverables_tracker.md
    tracker = Path(__file__).resolve().parent.parent / "deliverables" / "deliverables_tracker.md"
    if tracker.exists():
        text = tracker.read_text()
        updated = re.sub(
            r"(\*\*D5\*\* \| \*\*Final Systematic Map\*\* \|[^|]+\|[^|]+\|) — (\|)",
            f"\\1 [v01 (GDoc)](https://docs.google.com/document/d/{new_id}/edit) \\2",
            text,
        )
        if updated != text:
            tracker.write_text(updated)
            print("[gdocs] Updated deliverables_tracker.md")

    print(f"\n[gdocs] Done!")
    print(f"  Name : {new_name}")
    print(f"  URL  : https://docs.google.com/document/d/{new_id}/edit")
    print()
    print("Next steps:")
    print("  1. Open the Google Doc and review all content")
    print("  2. Update figures (EGM, ROSES, heatmaps) by inserting images from step16/")
    print("  3. Review and update the narrative text for D5 completion")
    print("  4. Export as Word .docx and upload to Teams → Deliverable 3 — EGM Report (draft)")
    print("  5. Share the evidence_gap_map.html link in the Teams upload message")
    print("  6. Tag @Aditi and @Neal in Teams chat")
    return new_id


if __name__ == "__main__":
    run()
