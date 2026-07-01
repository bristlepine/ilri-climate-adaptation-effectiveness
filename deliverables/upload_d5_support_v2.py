"""
Upload updated D5 support files to existing d5_support folder on Google Drive.

Updates:
  - fig1_prisma_flow.png  → new human-only PRISMA (prisma_human.png)
  - All other PNGs        → refreshed from step16/interactive/human/
  - evidence_gap_map_d5.html → new human EGM HTML
  - table1_search_screening.csv → LLM columns removed

Finds existing d5_support folder in the deliverables folder; creates if absent.
"""

import json, csv, io
from pathlib import Path
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseUpload

TOKEN               = "/Users/zarrar/projects/life/.credentials/token_personal.json"
DELIVERABLES_FOLDER = "1alv6Ic7vtKZc_xo6AdlKP1exshMh-fR6"
ROOT                = Path(__file__).parent.parent
STEP16              = ROOT / "scripts/outputs/step16"
HUMAN               = STEP16 / "interactive/human"
DELIVERABLES        = ROOT / "deliverables"

with open(TOKEN) as f:
    tok = json.load(f)
creds = Credentials(
    token=tok["token"], refresh_token=tok["refresh_token"],
    token_uri=tok["token_uri"], client_id=tok["client_id"],
    client_secret=tok["client_secret"], scopes=tok["scopes"],
)
svc = build("drive", "v3", credentials=creds)

# ── Find or create d5_support folder ─────────────────────────────────────────
q = (f"name='d5_support' and '{DELIVERABLES_FOLDER}' in parents "
     f"and mimeType='application/vnd.google-apps.folder' and trashed=false")
res = svc.files().list(q=q, fields="files(id,name)", supportsAllDrives=True,
                       includeItemsFromAllDrives=True).execute()
existing = res.get("files", [])

if existing:
    FOLDER_ID = existing[0]["id"]
    print(f"Found existing d5_support folder: {FOLDER_ID}")
else:
    folder = svc.files().create(
        body={"name": "d5_support", "mimeType": "application/vnd.google-apps.folder",
              "parents": [DELIVERABLES_FOLDER]},
        fields="id,name", supportsAllDrives=True
    ).execute()
    FOLDER_ID = folder["id"]
    print(f"Created new d5_support folder: {FOLDER_ID}")

# ── List existing files (for reference only, not deleting) ───────────────────
r = svc.files().list(
    q=f"'{FOLDER_ID}' in parents and trashed=false",
    fields="files(id,name)", supportsAllDrives=True,
    includeItemsFromAllDrives=True
).execute()
existing_files = {f["name"]: f["id"] for f in r.get("files", [])}
print(f"  Existing files in folder: {list(existing_files.keys()) or 'none'}")

# ── Delete all existing files (re-fetch fresh IDs right before each delete) ──
def _fresh_ids():
    res = svc.files().list(
        q=f"'{FOLDER_ID}' in parents and trashed=false",
        fields="files(id,name)", supportsAllDrives=True,
        includeItemsFromAllDrives=True
    ).execute()
    return {f["name"]: f["id"] for f in res.get("files", [])}

fresh = _fresh_ids()
for name, fid in fresh.items():
    try:
        svc.files().delete(fileId=fid, supportsAllDrives=True).execute()
        print(f"  Deleted: {name}")
    except Exception as e:
        print(f"  Could not delete {name}: {e}")

# ── Upload PNGs ───────────────────────────────────────────────────────────────
pngs = [
    ("fig1_prisma_flow.png",              HUMAN  / "prisma.png"),
    ("fig2_evidence_gap_map.png",         HUMAN  / "evidence_gap_map.png"),
    ("fig3_geographic_map.png",           HUMAN  / "geographic_map.png"),
    ("fig4_geographic_bar.png",           HUMAN  / "geographic_bar.png"),
    ("fig5_methodology.png",              HUMAN  / "methodology.png"),
    ("fig6_equity.png",                   HUMAN  / "equity.png"),
    ("fig7_temporal_trends.png",          HUMAN  / "temporal_trends.png"),
    ("fig8_saturation.png",               STEP16 / "saturation.png"),
    ("fig9_crosstab_domain_time.png",     HUMAN  / "crosstab_domain_time.png"),
    ("fig10_crosstab_producer_region.png",HUMAN  / "crosstab_producer_region.png"),
]

for drive_name, local_path in pngs:
    if not local_path.exists():
        print(f"  MISSING: {local_path}")
        continue
    meta  = {"name": drive_name, "parents": [FOLDER_ID]}
    media = MediaFileUpload(str(local_path), mimetype="image/png", resumable=False)
    f = svc.files().create(body=meta, media_body=media, fields="id,name",
                           supportsAllDrives=True).execute()
    print(f"  Uploaded {f['name']}  ({local_path.stat().st_size // 1024} KB)")

# ── Upload EGM HTML ───────────────────────────────────────────────────────────
html_path = DELIVERABLES / "evidence_gap_map_d5.html"
if html_path.exists():
    meta  = {"name": "evidence_gap_map_d5.html", "parents": [FOLDER_ID]}
    media = MediaFileUpload(str(html_path), mimetype="text/html", resumable=False)
    f = svc.files().create(body=meta, media_body=media, fields="id,name",
                           supportsAllDrives=True).execute()
    print(f"  Uploaded {f['name']}  ({html_path.stat().st_size // 1024} KB)")
else:
    print(f"  MISSING: {html_path}")

# ── Upload Table 1 CSV (no LLM columns) ──────────────────────────────────────
table1_rows = [
    ["Database",                     "Returned", "After dedup", "Abstr. incl."],
    ["Scopus",                        "17,021",     "17,021",      "6,218"],
    ["Web of Science",                "15,179",      "4,683",      "1,137"],
    ["CAB Abstracts",                  "5,723",      "3,229",      "1,133"],
    ["Academic Search Premier",        "1,187",        "274",         "66"],
    ["AGRIS",                              "3",            "1",          "1"],
    ["EconLit",                          "479",          "263",         "83"],
    ["ProQuest",                         "368",          "117",         "40"],
    ["Google Scholar",                   "198",          "100",         "34"],
    ["Grey literature (20 sources)",     "495",          "485",         "41"],
    ["Total",                         "40,653",       "26,173",      "8,753"],
]

buf = io.StringIO()
csv.writer(buf).writerows(table1_rows)
csv_bytes = buf.getvalue().encode("utf-8")
meta  = {"name": "table1_search_screening.csv", "parents": [FOLDER_ID]}
media = MediaIoBaseUpload(io.BytesIO(csv_bytes), mimetype="text/csv", resumable=False)
f = svc.files().create(body=meta, media_body=media, fields="id,name",
                       supportsAllDrives=True).execute()
print(f"  Uploaded {f['name']}")

print(f"\nDone — d5_support: https://drive.google.com/drive/folders/{FOLDER_ID}")
