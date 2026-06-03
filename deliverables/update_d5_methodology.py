"""
Targeted text-only updates to D5 GDoc methodology section.
Uses replaceAllText — preserves all formatting/styles.
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

def replace(find, replacement):
    return {
        "replaceAllText": {
            "containsText": {"text": find, "matchCase": True},
            "replaceText": replacement,
        }
    }

requests = [
    # 1. Remove cross-reference to Deliverable 3; keep standalone opener
    replace(
        "This systematic map follows the protocol published as Deliverable 3 (Bristlepine Resilience Consultants, January 2026). A brief summary is provided here; readers are referred to that document for full methodological details.",
        "A brief summary of methods is provided here.",
    ),

    # 2. Remove §3.3 cross-reference + fix tense; append total record count
    replace(
        "(approximately 20 repositories per §3.3 of the protocol) are also being searched.",
        "(approximately 20 repositories) were also searched. Across all sources, 40,653 records were retrieved prior to deduplication.",
    ),
]

result = svc.documents().batchUpdate(documentId=DOC_ID, body={"requests": requests}).execute()

replies = result.get("replies", [])
for i, req in enumerate(requests):
    label = list(req["replaceAllText"]["containsText"].values())[0][:60]
    count = replies[i].get("replaceAllText", {}).get("occurrencesChanged", 0)
    print(f"  [{count} replaced] {label}…")

print("\nDone. Review at: https://docs.google.com/document/d/1Dhi15o2pq-Bq4kvIsBYVzdgPnzIHJcW-lNaTHxiDpDI/edit")
