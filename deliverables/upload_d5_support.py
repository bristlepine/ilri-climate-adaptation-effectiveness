"""
Creates d5_support/ in the Drive deliverables folder and uploads:
  - 9 PNGs (exact files used in D5, named fig1_… through fig9_…)
  - table1_search_screening.csv  (search and screening results)
"""

import json, csv, io
from pathlib import Path
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseUpload

TOKEN              = "/Users/zarrar/projects/life/.credentials/token_personal.json"
DELIVERABLES_FOLDER = "1alv6Ic7vtKZc_xo6AdlKP1exshMh-fR6"
ROOT               = Path(__file__).parent.parent          # repo root
STEP16             = ROOT / "scripts/outputs/step16"
HUMAN              = STEP16 / "interactive/human"

with open(TOKEN) as f:
    tok = json.load(f)
creds = Credentials(
    token=tok["token"], refresh_token=tok["refresh_token"],
    token_uri=tok["token_uri"], client_id=tok["client_id"],
    client_secret=tok["client_secret"], scopes=tok["scopes"],
)
svc = build("drive", "v3", credentials=creds)

# ── Create d5_support folder ─────────────────────────────────────────────────
folder_meta = {
    "name": "d5_support",
    "mimeType": "application/vnd.google-apps.folder",
    "parents": [DELIVERABLES_FOLDER],
}
folder = svc.files().create(body=folder_meta, fields="id,name", supportsAllDrives=True).execute()
FOLDER_ID = folder["id"]
print(f"Created folder: d5_support  ({FOLDER_ID})")

# ── PNGs — exact files used by publish_d5_from_d4.py ────────────────────────
pngs = [
    ("fig1_prisma_flow.png",        ROOT / "deliverables/prisma_flow_d5.png"),
    ("fig2_evidence_gap_map.png",   HUMAN / "evidence_gap_map.png"),
    ("fig3_geographic_map.png",     HUMAN / "geographic_map.png"),
    ("fig4_geographic_bar.png",     HUMAN / "geographic_bar.png"),
    ("fig5_methodology.png",        HUMAN / "methodology.png"),
    ("fig6_temporal_trends.png",    HUMAN / "temporal_trends.png"),
    ("fig7_equity.png",             HUMAN / "equity.png"),
    ("fig8_saturation.png",         STEP16 / "saturation.png"),
    ("fig9_llm_vs_human.png",       STEP16 / "llm_vs_human.png"),
]

for drive_name, local_path in pngs:
    if not local_path.exists():
        print(f"  MISSING: {local_path}")
        continue
    meta  = {"name": drive_name, "parents": [FOLDER_ID]}
    media = MediaFileUpload(str(local_path), mimetype="image/png", resumable=False)
    f = svc.files().create(body=meta, media_body=media, fields="id,name", supportsAllDrives=True).execute()
    size_kb = local_path.stat().st_size // 1024
    print(f"  Uploaded {f['name']}  ({size_kb} KB)")

# ── Table 1 CSV — search and screening results (corrected numbers) ───────────
table1_rows = [
    ["Database", "Returned", "After dedup", "Abstr. incl.", "FT retrieved", "FT screened", "Coded"],
    ["Scopus",                       "17,021", "17,021", "6,218", "2,644", "2,644", "—"],
    ["Web of Science",               "15,179",  "4,683", "1,137",   "552",   "552", "—"],
    ["CAB Abstracts",                 "5,723",  "3,229", "1,133",   "260",   "260", "—"],
    ["Academic Search Premier",       "1,187",    "274",    "66",    "20",    "20", "—"],
    ["EconLit",                         "479",    "263",    "83",    "18",    "18", "—"],
    ["ProQuest",                        "368",    "117",    "40",    "10",    "10", "—"],
    ["Google Scholar",                  "198",    "100",    "34",     "0",    "—",  "—"],
    ["AGRIS",                             "3",      "1",     "1",     "0",    "—",  "—"],
    ["Grey literature (20 sources)",    "495",    "485",    "41",     "1",    "1",  "—"],
    ["Total",                        "40,653", "26,173", "8,753", "3,505", "3,505", "2,368"],
]

buf = io.StringIO()
writer = csv.writer(buf)
writer.writerows(table1_rows)
csv_bytes = buf.getvalue().encode("utf-8")

meta  = {"name": "table1_search_screening.csv", "parents": [FOLDER_ID]}
media = MediaIoBaseUpload(io.BytesIO(csv_bytes), mimetype="text/csv", resumable=False)
f = svc.files().create(body=meta, media_body=media, fields="id,name").execute()
print(f"  Uploaded {f['name']}")

print(f"\nDone — d5_support folder: https://drive.google.com/drive/folders/{FOLDER_ID}")
