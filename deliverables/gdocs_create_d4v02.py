"""
gdocs_create_d4v02.py

Creates D4 v02 from D4 v01:
1. Copies D4 v01 → D4 v02
2. Updates cover page date/version
3. Updates in-text statistics with latest pipeline numbers
4. Replaces plain-text Table 1 with a styled per-database pipeline table
   (theme header: green #21472E / white; total row: charcoal #3C3C3C / white)

Run: conda run -n ilri01 python deliverables/gdocs_create_d4v02.py
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
D4_V01_DOC_ID       = "1i2dQUoXuNPwjvV_fT5CMiy0fvADy8oWCGK3sdHGApCs"
D4_BASE_NAME        = "Deliverable 4_Bristlepine_First Draft Systematic Map"

# ── latest pipeline stats ─────────────────────────────────────────────────────
STATS = {
    "scopus_total":       17_021,
    "abstract_included":   6_218,
    "abstract_excluded":  10_803,
    "missing_abstract":    1_328,
    "ft_retrieved":        2_644,
    "ft_screened":         2_596,
    "ft_included":         2_093,
    "ft_excluded":           501,
    "coded_records":       2_093,
    "net_new_total":       8_187,
    "net_new_included":    2_340,
    "search_date":        "January 2026",
    "report_date":        "April 2026",
    "version":            "v02",
}

# ── per-database table data ───────────────────────────────────────────────────
# Columns: Database | Returned | After Dedup | Abstr. Screened Incl. | FT Retrieved | FT Screened | Coded
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
    ["Scopus",                   "17,021", "17,021", "6,218", "2,644", "2,596", "2,093"],
    ["Web of Science",           "15,179",  "4,683", "1,140",   "552",     "—",     "—"],
    ["CAB Abstracts",             "5,723",  "3,229", "1,133",   "260",     "—",     "—"],
    ["Academic Search Premier",   "1,187",    "274",    "66",    "20",     "—",     "—"],
    ["AGRIS",                         "3",      "1",     "1",     "0",     "—",     "—"],
    ["Total",                    "39,113", "25,208", "8,558", "3,476", "2,596†", "2,093†"],
]

TABLE_FOOTNOTE = (
    "† Scopus only. Full-text screening for net-new records (Web of Science, "
    "CAB Abstracts, Academic Search Premier, AGRIS) is in progress and will "
    "be reported in Deliverable 5."
)

# Brand colours (from globals.css)
GREEN    = {"red": 33/255,  "green": 71/255,  "blue": 46/255}   # #21472E
CHARCOAL = {"red": 60/255,  "green": 60/255,  "blue": 60/255}   # #3C3C3C
WHITE    = {"red": 1.0,     "green": 1.0,     "blue": 1.0}
SAND     = {"red": 245/255, "green": 242/255, "blue": 236/255}  # #F5F2EC


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
    """
    Find start/end indices of the old plain-text Table 1 block.
    Starts at "Table 1." and ends before "Figure 2 presents".
    Returns (start_index, end_index) to delete.
    """
    start = end = None
    for elem in content:
        txt = para_text(elem)
        if start is None and "Table 1." in txt:
            start = elem.get("startIndex", 0)
        if start is not None and end is None and "Figure 2 presents" in txt:
            end = elem.get("startIndex", 0)
            break
    return start, end


def find_para_index(content, search: str) -> int | None:
    """Return the startIndex of the paragraph containing search text."""
    for elem in content:
        if search in para_text(elem):
            return elem.get("startIndex")
    return None


# =============================================================================
# Table helpers
# =============================================================================

def rgb(c: dict) -> dict:
    return {"color": {"rgbColor": c}}


def cell_color_requests(table_elem, row: int, col: int, bg: dict) -> list:
    """Return requests to set background colour for one table cell."""
    try:
        cell = table_elem["table"]["tableRows"][row]["tableCells"][col]
        si = cell.get("startIndex", 0)
        ei = cell.get("endIndex", si + 1)
        return [{
            "updateTableCellStyle": {
                "tableStartLocation": {"index": table_elem["startIndex"]},
                "rowSpan": 1,
                "columnSpan": 1,
                "tableCellStyle": {
                    "backgroundColor": rgb(bg),
                },
                "fields": "backgroundColor",
                "tableRange": {
                    "tableCellLocation": {
                        "tableStartLocation": {"index": table_elem["startIndex"]},
                        "rowIndex": row,
                        "columnIndex": col,
                    },
                    "rowSpan": 1,
                    "columnSpan": 1,
                },
            }
        }]
    except (IndexError, KeyError):
        return []


def find_table_elem(content):
    """Return the first table element in the body content."""
    for elem in content:
        if "table" in elem:
            return elem
    return None


def populate_table(docs, doc_id: str, table_elem: dict) -> None:
    """Fill table cells with text and apply header/total row styling."""
    rows = table_elem["table"]["tableRows"]
    requests = []

    all_rows = [TABLE_HEADERS] + TABLE_ROWS

    for r_idx, row_data in enumerate(all_rows):
        row = rows[r_idx]
        for c_idx, cell_text in enumerate(row_data):
            cell = row["tableCells"][c_idx]
            # Each cell has one paragraph with one empty text run at startIndex+1
            para = cell["content"][0]
            insert_at = para.get("startIndex", cell.get("startIndex", 0))
            requests.append({
                "insertText": {
                    "location": {"index": insert_at},
                    "text": cell_text,
                }
            })

    docs.documents().batchUpdate(
        documentId=doc_id, body={"requests": requests}
    ).execute()
    print("[gdocs] Table cells populated.")
    time.sleep(1)


def style_table(docs, doc_id: str, table_elem: dict) -> None:
    """Apply background colours and text styles to header + total rows."""
    rows = table_elem["table"]["tableRows"]
    n_cols = len(TABLE_HEADERS)
    requests = []

    for c_idx in range(n_cols):
        # Header row: green background
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
        # Header text: white + bold
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

    # Total row (last row): charcoal background
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
        docs.documents().batchUpdate(
            documentId=doc_id, body={"requests": requests}
        ).execute()
        print("[gdocs] Table styled.")


# =============================================================================
# Main
# =============================================================================

def run():
    creds = get_creds()
    drive = build("drive", "v3", credentials=creds)
    docs  = build("docs",  "v1", credentials=creds)

    # 1. Copy D4 v01 → D4 v02
    new_name = next_version_name(drive, DELIVERABLES_FOLDER, D4_BASE_NAME)
    print(f"[gdocs] Creating: {new_name}")
    new_id = copy_doc(drive, D4_V01_DOC_ID, new_name, DELIVERABLES_FOLDER)
    print(f"[gdocs] Created : {new_id}")
    print(f"[gdocs] URL     : https://docs.google.com/document/d/{new_id}/edit")
    time.sleep(3)

    # 2. Update cover page and in-text stats (replaceAllText)
    print("[gdocs] Batch 1 — updating cover page and stats ...")
    replacements = {
        "April 2026, v01":  "April 2026, v02",
        "Deliverable 4_Bristlepine_First Draft Systematic Map_v01":
            "Deliverable 4_Bristlepine_First Draft Systematic Map_v02",
        # Stats that appear literally in the text
        "coded dataset of 1,899 studies":  f"coded dataset of {STATS['coded_records']:,} studies",
        "1,899 records were successfully": f"{STATS['coded_records']:,} records were successfully",
        "3,570 full texts":                f"{STATS['ft_retrieved']:,} full texts",
        "3,570 of 6,218":                  f"{STATS['ft_retrieved']:,} of {STATS['abstract_included']:,}",
        "57%) were retrieved":             f"{round(STATS['ft_retrieved']/STATS['abstract_included']*100)}%) were retrieved",
        "2,648 require manual":            f"{STATS['abstract_included'] - STATS['ft_retrieved']:,} require manual",
        "7,913 additional records":        f"{STATS['net_new_total']:,} additional records",
        "2,271 were included":             f"{STATS['net_new_included']:,} were included",
        # Screening calibration note update
        "0.966–0.970 against":             "0.966–0.970 against",  # no change needed
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

    # 3. Delete old plain-text Table 1 block and insert styled table
    print("[gdocs] Batch 2 — replacing plain-text table with styled table ...")
    content = get_body(docs, new_id)
    tbl_start, tbl_end = find_range_to_delete(content)

    if tbl_start is None or tbl_end is None:
        print("[gdocs] WARN — could not find Table 1 block; skipping table replacement.")
    else:
        n_rows = 1 + len(TABLE_ROWS)   # header + data rows
        n_cols = len(TABLE_HEADERS)

        # Delete old block and insert empty table in its place
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

        # 4. Populate table cells (Batch 3)
        content = get_body(docs, new_id)
        table_elem = find_table_elem(content)
        if table_elem:
            populate_table(docs, new_id, table_elem)
            time.sleep(1)

            # 5. Style header + total rows (Batch 4)
            # Re-read after population so indices are accurate
            content = get_body(docs, new_id)
            table_elem = find_table_elem(content)
            if table_elem:
                style_table(docs, new_id, table_elem)
                time.sleep(1)

                # 6. Insert footnote after table
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
                                "range": {"startIndex": after_table, "endIndex": after_table + len(TABLE_FOOTNOTE)},
                                "textStyle": {"italic": True, "fontSize": {"magnitude": 9, "unit": "PT"}},
                                "fields": "italic,fontSize",
                            }},
                        ]},
                    ).execute()
                    print("[gdocs] Footnote inserted.")

    # 7. Record in googledocs.md
    md = Path(__file__).resolve().parent / "googledocs.md"
    if md.exists():
        text = md.read_text()
        updated = re.sub(
            r"(\| D4: First Draft Systematic Map \|)[^\n]+",
            f"\\1 https://docs.google.com/document/d/{new_id}/edit | `{new_id}` |",
            text,
        )
        if updated != text:
            md.write_text(updated)
            print("[gdocs] Updated googledocs.md")

    print(f"\n[gdocs] Done!")
    print(f"  Name : {new_name}")
    print(f"  URL  : https://docs.google.com/document/d/{new_id}/edit")
    return new_id


if __name__ == "__main__":
    run()
