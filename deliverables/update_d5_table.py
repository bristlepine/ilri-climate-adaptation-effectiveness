"""
Adds EconLit, ProQuest, Google Scholar, and Grey literature rows to Table 1,
fixes WoS abstract count, and updates §3.1 database list.

Pass 1: replaceAllText fixes (WoS count + §3.1 sentence)
Pass 2: insertTableRow × 4 below AGRIS
Pass 3: re-read doc, fill cell text bottom-to-top
"""

import json
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

DOC_ID = "1Dhi15o2pq-Bq4kvIsBYVzdgPnzIHJcW-lNaTHxiDpDI"
TOKEN  = "deliverables/.credentials/token.json"

with open(TOKEN) as f:
    tok = json.load(f)

creds = Credentials(
    token=tok["token"],
    refresh_token=tok["refresh_token"],
    token_uri=tok["token_uri"],
    client_id=tok["client_id"],
    client_secret=tok["client_secret"],
    scopes=tok["scopes"],
)
svc = build("docs", "v1", credentials=creds)

# ─────────────────────────────────────────────
# PASS 1 — text fixes via replaceAllText
# ─────────────────────────────────────────────
def r(find, rep):
    return {"replaceAllText": {"containsText": {"text": find, "matchCase": True}, "replaceText": rep}}

pass1 = [
    # §3.1 — add missing databases to the sentence
    r(
        "Additional databases searched in parallel include Web of Science Core Collection, CAB Abstracts, AGRIS, and Academic Search Premier.",
        "Additional databases searched in parallel include Web of Science Core Collection, CAB Abstracts, Academic Search Premier, EconLit, ProQuest, Google Scholar, and AGRIS.",
    ),
    # WoS abstract included: step12b gives 1,137 (3 Needs_Manual excluded)
    r("15,179 | 4,683 | 1,140", "15,179 | 4,683 | 1,137"),  # table cell style if pipe-separated
    # Plain number replacement as fallback (table cells are plain text)
    r("1,140", "1,137"),
]

res1 = svc.documents().batchUpdate(documentId=DOC_ID, body={"requests": pass1}).execute()
for i, req in enumerate(pass1):
    find  = req["replaceAllText"]["containsText"]["text"][:55]
    count = res1["replies"][i].get("replaceAllText", {}).get("occurrencesChanged", 0)
    print(f"  Pass 1 [{count:2d} replaced]  {find}…")

# ─────────────────────────────────────────────
# PASS 2 — read doc, find AGRIS row, insert 4 rows
# ─────────────────────────────────────────────
def cell_text(cell):
    parts = []
    for p in cell.get("content", []):
        for el in p.get("paragraph", {}).get("elements", []):
            parts.append(el.get("textRun", {}).get("content", ""))
    return "".join(parts).strip()

def find_table_and_agris(doc):
    for elem in doc["body"]["content"]:
        if "table" not in elem:
            continue
        tbl = elem["table"]
        for ri, row in enumerate(tbl["tableRows"]):
            for cell in row["tableCells"]:
                if "AGRIS" in cell_text(cell):
                    return elem["startIndex"], tbl, ri
    return None, None, None

doc = svc.documents().get(documentId=DOC_ID).execute()
tbl_start, tbl, agris_row = find_table_and_agris(doc)
print(f"\n  Table starts at index {tbl_start}, AGRIS at row {agris_row}")

# Insert 4 rows below AGRIS. Each insertion below rowIndex=agris_row pushes
# the previous new row down, so final order (top→bottom) after all 4 insertions:
#   EconLit  (inserted 4th → lands at agris+1)
#   ProQuest (inserted 3rd → agris+2)
#   Google Scholar (inserted 2nd → agris+3)
#   Grey literature (inserted 1st → agris+4)
pass2 = [
    {"insertTableRow": {
        "tableCellLocation": {
            "tableStartLocation": {"index": tbl_start},
            "rowIndex": agris_row,
            "columnIndex": 0,
        },
        "insertBelow": True,
    }}
    for _ in range(4)
]

svc.documents().batchUpdate(documentId=DOC_ID, body={"requests": pass2}).execute()
print("  Pass 2: 4 rows inserted below AGRIS")

# ─────────────────────────────────────────────
# PASS 3 — re-read doc, fill cell text
# ─────────────────────────────────────────────
doc2    = svc.documents().get(documentId=DOC_ID).execute()
_, tbl2, agris_row2 = find_table_and_agris(doc2)

# New rows and their data (in final top-to-bottom order after AGRIS)
new_rows = [
    ("EconLit",                    "479", "263", "83", "18", "18", "—"),  # agris+1
    ("ProQuest",                   "368", "117", "40", "10", "10", "—"),  # agris+2
    ("Google Scholar",             "198", "100", "34",  "0",  "—", "—"),  # agris+3
    ("Grey literature (20 sources)","495", "485", "41",  "1",  "1", "—"),  # agris+4
]

inserts = []  # (index, text) pairs — will sort descending before sending

for offset, cols in enumerate(new_rows):
    row_idx  = agris_row2 + 1 + offset
    tbl_row  = tbl2["tableRows"][row_idx]
    n_cells  = len(tbl_row["tableCells"])
    for col_idx, text in enumerate(cols):
        if col_idx >= n_cells:
            break
        cell    = tbl_row["tableCells"][col_idx]
        content = cell.get("content", [])
        if not content:
            continue
        # Insert at start of first paragraph in cell
        insert_at = content[0]["startIndex"]
        inserts.append((insert_at, text))

# Sort descending so earlier insertions don't shift later indexes
inserts.sort(key=lambda x: x[0], reverse=True)

pass3 = [
    {"insertText": {"location": {"index": idx}, "text": text}}
    for idx, text in inserts
]

svc.documents().batchUpdate(documentId=DOC_ID, body={"requests": pass3}).execute()
print(f"  Pass 3: {len(pass3)} cell texts inserted")

print("\nDone. Review: https://docs.google.com/document/d/1Dhi15o2pq-Bq4kvIsBYVzdgPnzIHJcW-lNaTHxiDpDI/edit")
