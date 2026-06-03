"""
Fix numbers and source counts in section 4.1 to match PRISMA data.
Uses replaceAllText only — no formatting touched.
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

def r(find, replacement):
    return {
        "replaceAllText": {
            "containsText": {"text": find, "matchCase": True},
            "replaceText": replacement,
        }
    }

requests = [
    # Total records + source counts (text + table Total row)
    r("39,113 records were identified across five bibliographic databases and 24 grey literature sources",
      "40,653 records were identified across eight bibliographic databases and 20 grey literature sources"),

    # After dedup (text + table Total row)
    r("25,208 unique records", "26,173 unique records"),
    r("25,208",               "26,173"),   # catches table Total cell if not part of longer phrase

    # Abstract included (text + table Total row)
    r("identified 8,558 records for full-text assessment",
      "identified 8,753 records for full-text assessment"),
    r("8,558", "8,753"),  # catches table Total cell

    # FT retrieved (text + table Total row)
    r("3,476 full texts retrieved (40%)", "3,505 full texts retrieved (40%)"),
    r("3,476", "3,505"),  # catches table Total cell

    # FT screened (text + table Total row)
    r("3,464 were screened and 2,368", "3,505 were screened and 2,368"),
    r("3,464", "3,505"),  # catches table Total cell

    # Figure 1 caption source count
    r("record flow across 29 sources", "record flow across 28 sources"),

    # Table footnote: not retrievable count
    r("n = 5,243", "n = 5,248"),
]

result = svc.documents().batchUpdate(documentId=DOC_ID, body={"requests": requests}).execute()

replies = result.get("replies", [])
for i, req in enumerate(requests):
    find  = req["replaceAllText"]["containsText"]["text"][:55]
    count = replies[i].get("replaceAllText", {}).get("occurrencesChanged", 0)
    print(f"  [{count:2d} replaced]  {find}…")

print("\nDone — review at: https://docs.google.com/document/d/1Dhi15o2pq-Bq4kvIsBYVzdgPnzIHJcW-lNaTHxiDpDI/edit")
print("\nNote: table still needs EconLit / ProQuest / Google Scholar rows added manually.")
