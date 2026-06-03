"""
Fix all section 4 issues in D5 GDoc:
  - Table 1 Total Returned: 39,113 → 40,653
  - Add missing 4.2 Evidence gap map heading
  - Fix figure numbering (methodology/equity swapped; two Figure 6s; Figure 7→8 saturation)
  - Add Figure 9 (LLM vs human) text and caption
  - Fix Section 5 links (split EGM and database)
  - Delete duplicate §4.1 paragraph at end of doc

Pass 1: replaceAllText (text only, no formatting changes)
Pass 2: apply HEADING_2 style + delete orphan paragraphs
"""

import json
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

DOC_ID = "1Dhi15o2pq-Bq4kvIsBYVzdgPnzIHJcW-lNaTHxiDpDI"
TOKEN  = "deliverables/.credentials/token.json"

with open(TOKEN) as f:
    tok = json.load(f)
creds = Credentials(
    token=tok["token"], refresh_token=tok["refresh_token"],
    token_uri=tok["token_uri"], client_id=tok["client_id"],
    client_secret=tok["client_secret"], scopes=tok["scopes"],
)
svc = build("docs", "v1", credentials=creds)

def r(find, rep):
    return {"replaceAllText": {
        "containsText": {"text": find, "matchCase": True},
        "replaceText": rep,
    }}

# ─── PASS 1: replaceAllText ───────────────────────────────────────────────────
pass1 = [
    # 1. Fix Table 1 Total Returned cell
    r("39,113", "40,653"),

    # 2-6. Figure renumbering (order: highest number first to avoid conflicts)
    r("Figure 7. Information saturation curve",
      "Figure 8. Information saturation curve"),
    r("(Figure 7)",
      "(Figure 8)"),
    r("Figure 6. Publication trends across included studies",
      "Figure 7. Publication trends across included studies"),
    r("Figure 5. Equity and inclusion dimensions",
      "Figure 6. Equity and inclusion dimensions"),
    r("Figure 6. Methodological approaches across included studies",
      "Figure 5. Methodological approaches across included studies"),

    # 7. Add Figure 9 text after saturation caption (now "Figure 8." from step 2)
    r(
        "Figure 8. Information saturation curve. Top: cumulative unique categories as % of final total by papers coded. Bottom: new categories per batch. All dimensions plateau by 49 papers.",
        "Figure 8. Information saturation curve. Top: cumulative unique categories as % of final total by papers coded. Bottom: new categories per batch. All dimensions plateau by 49 papers.\n"
        "Figure 9 compares LLM-screened and human-coded results across geographic, methodological, and equity dimensions. The LLM corpus broadly reflects human-coded patterns but diverges on equity — 85% of LLM records report no marginalized group versus 41% in the human-coded set.\n"
        "Figure 9. LLM vs human-coded comparison — key dimensions. Amber = human (n = 86) · teal = LLM (n = 2,368).",
    ),

    # 8. Insert 4.2 heading text before EGM paragraph (style applied in Pass 2)
    r(
        "The evidence gap map (EGM) is the primary output of this deliverable.",
        "4.2  Evidence gap map\nThe evidence gap map (EGM) is the primary output of this deliverable.",
    ),

    # 9. Fix Section 5 link sentence (URL paragraph deleted in Pass 2)
    r(
        "The interactive evidence gap map and searchable database are available at the project website:",
        "An interactive evidence gap map is available at https://main.d3a9jahzuu7xai.amplifyapp.com/systematic-map. A searchable database of all included studies is available at https://main.d3a9jahzuu7xai.amplifyapp.com/systematic-map#searchable-database",
    ),

    # 10. Strip unique suffix from duplicate §4.1 paragraph (enables deletion in Pass 2)
    r(" Table 1 gives the full breakdown by database.", ""),
]

res1 = svc.documents().batchUpdate(documentId=DOC_ID, body={"requests": pass1}).execute()
for i, req in enumerate(pass1):
    find  = req["replaceAllText"]["containsText"]["text"][:60]
    count = res1["replies"][i].get("replaceAllText", {}).get("occurrencesChanged", 0)
    print(f"  P1 [{count:2d}]  {find}…")

# ─── PASS 2: structural changes ───────────────────────────────────────────────
doc = svc.documents().get(documentId=DOC_ID).execute()

def paragraphs(doc):
    out = []
    for elem in doc["body"]["content"]:
        if "paragraph" in elem:
            text = "".join(
                e.get("textRun", {}).get("content", "")
                for e in elem["paragraph"].get("elements", [])
            ).strip()
            out.append((elem["startIndex"], elem["endIndex"], text))
    return out

paras = paragraphs(doc)
requests2 = []

# a. Apply HEADING_2 to the new "4.2  Evidence gap map" paragraph
for s, e, t in paras:
    if t == "4.2  Evidence gap map":
        requests2.append({
            "updateParagraphStyle": {
                "range": {"startIndex": s, "endIndex": e},
                "paragraphStyle": {"namedStyleType": "HEADING_2"},
                "fields": "namedStyleType",
            }
        })
        print(f"\n  P2 heading style → index {s}: {t!r}")
        break

# b. Delete orphaned URL paragraph (now standalone after sentence update in Pass 1)
for s, e, t in paras:
    if t == "https://main.d3a9jahzuu7xai.amplifyapp.com/systematic-map":
        requests2.append(("delete", s, e, "orphaned URL paragraph"))
        break

# c. Delete duplicate §4.1 paragraph (now identical to original after step 10 above)
dup_candidates = [(s, e, t) for s, e, t in paras if t.startswith("A total of 40,653 records")]
if len(dup_candidates) >= 2:
    dup_candidates.sort(key=lambda x: x[0])
    s, e, t = dup_candidates[-1]   # last occurrence = duplicate at end of doc
    requests2.append(("delete", s, e, "duplicate §4.1 paragraph"))
    print(f"  P2 delete duplicate at index {s}")

# Sort delete operations highest→lowest index to avoid shift issues
style_ops  = [r for r in requests2 if isinstance(r, dict)]
delete_ops = sorted([r for r in requests2 if isinstance(r, tuple)],
                    key=lambda x: x[1], reverse=True)

for op in delete_ops:
    _, s, e, label = op
    print(f"  P2 delete {label} ({s}–{e})")

final = style_ops + [
    {"deleteContentRange": {"range": {"startIndex": s, "endIndex": e}}}
    for _, s, e, _ in delete_ops
]

if final:
    svc.documents().batchUpdate(documentId=DOC_ID, body={"requests": final}).execute()

print("\nDone — review: https://docs.google.com/document/d/1Dhi15o2pq-Bq4kvIsBYVzdgPnzIHJcW-lNaTHxiDpDI/edit")
