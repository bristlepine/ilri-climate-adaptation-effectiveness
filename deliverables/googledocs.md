# Google Docs / Drive Integration

Tracks all Google Drive folder links, document IDs, and credentials setup for the ILRI Climate Adaptation Effectiveness systematic map project.

---

## Drive Folders

| Folder | Link | ID |
|---|---|---|
| Deliverables (root) | https://drive.google.com/drive/folders/1alv6Ic7vtKZc_xo6AdlKP1exshMh-fR6 | `1alv6Ic7vtKZc_xo6AdlKP1exshMh-fR6` |
| Bristlepine Shared Drive | — | `0AN5dgziI6r5oUk9PVA` |

---

## Key Documents

| Document | Link | Doc ID |
|---|---|---|
| D3: Final Systematic Map Protocol v1 (native GDoc) | https://docs.google.com/document/d/1XN0YdGPnOBEMVLxvekGQ-ztYngOQkx84kn7r2v0Y2qU/edit | `1XN0YdGPnOBEMVLxvekGQ-ztYngOQkx84kn7r2v0Y2qU` |
| D3: Final Systematic Map Protocol v1 (docx upload) | — | `1ova85zng4wluk0YTdGwQsd3Xgm2Vvw7k` |
| PROCEED SM template | — | `10_Fa2yOpQ6gEDgvbQqcReLHXJ-l3qV21` |
| D4: First Draft Systematic Map (v01) | https://docs.google.com/document/d/1i2dQUoXuNPwjvV_fT5CMiy0fvADy8oWCGK3sdHGApCs/edit | `1i2dQUoXuNPwjvV_fT5CMiy0fvADy8oWCGK3sdHGApCs` |
| D4: First Draft Systematic Map (v02) | https://docs.google.com/document/d/18CZrKhDPEWB8jXIVE0OiYgOw6Kt__56y757MK4iByUQ/edit | `18CZrKhDPEWB8jXIVE0OiYgOw6Kt__56y757MK4iByUQ` |
| D3 v02: Protocol Amendment (D5.7) | https://docs.google.com/document/d/1yLnB2b--XOtrMSQ1ekgBu6XKGcDWQ1Ggvn7mKQqjaqg/edit | `1yLnB2b--XOtrMSQ1ekgBu6XKGcDWQ1Ggvn7mKQqjaqg` |

---

## GitHub + Zenodo

| Resource | Link | ID / Token |
|---|---|---|
| GitHub repo | https://github.com/bristlepine/ilri-climate-adaptation-effectiveness | `bristlepine/ilri-climate-adaptation-effectiveness` |
| Zenodo D3 record | https://zenodo.org/records/18370029 | `18370029` |
| Zenodo D4 record | — (to mint) | — |
| Zenodo GitHub webhook | https://zenodo.org/account/settings/github | enable repo for auto-DOI |
| Frontend (Amplify) | https://main.d3a9jahzuu7xai.amplifyapp.com/ | — |

### Zenodo token setup
1. Go to https://zenodo.org/account/settings/applications → Personal access tokens → New token
2. Name: `claude-code-local`, scopes: **deposit:write** + **deposit:actions**
3. Add to `.env` file (gitignored):
   ```
   ZENODO_TOKEN=your_token_here
   ```

### Deliverable publishing workflow (D4.4 and beyond)
```
1. conda run -n ilri01 python scripts/gdocs_create_d4.py
      → creates D4_v01 Google Doc in Drive

2. Export PDF from Google Doc (File → Download → PDF)
      → save to deliverables/

3. conda run -n ilri01 python scripts/zenodo_publish.py \
      --file deliverables/D4_v01.pdf \
      --title "Deliverable 4: First Draft Systematic Map (Preliminary)" \
      --version v01 \
      --confirm
      → publishes to Zenodo, returns DOI

4. gh release create d4-v01 deliverables/D4_v01.pdf \
      --title "D4: First Draft Systematic Map (Preliminary)" \
      --notes "Preliminary systematic map based on Scopus. DOI: https://doi.org/..."
      → GitHub release
```

### D5.7 protocol amendment (new version of D3)
```
conda run -n ilri01 python scripts/zenodo_publish.py \
  --file deliverables/D5.7_Protocol_Amendment_v01.pdf \
  --title "Deliverable 5.7: Protocol Amendment v2 — ..." \
  --new-version 18370029 \
  --version v2 \
  --confirm
```

---

## Credentials Setup

### Step 1 — Google Cloud project
1. Go to https://console.cloud.google.com/
2. Create a new project (or use an existing one) — name it e.g. `bristlepine-docs`
3. Enable these two APIs:
   - **Google Docs API** — https://console.cloud.google.com/apis/library/docs.googleapis.com
   - **Google Drive API** — https://console.cloud.google.com/apis/library/drive.googleapis.com

### Step 2 — OAuth credentials
1. Go to APIs & Services → Credentials → Create Credentials → **OAuth client ID**
2. Application type: **Desktop app**
3. Name: `claude-code-local`
4. Download the JSON → save as `deliverables/.credentials/client_secret.json`
   (this file is gitignored — never commit it)

### Step 3 — Install Python packages
```bash
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
```

### Step 4 — First-time auth
Run the helper script once — it opens a browser for OAuth consent, then saves a token:
```bash
conda run -n ilri01 python deliverables/gdocs_auth.py
```
Token saved to `deliverables/.credentials/token.json` (also gitignored).

---

## Gitignore rules (already added)
```
deliverables/.credentials/
```

---

## Usage (once set up)

```bash
# Authenticate (one-time)
conda run -n ilri01 python deliverables/gdocs_auth.py

# Create D4 from D3 template
conda run -n ilri01 python deliverables/gdocs_create_d4.py

# Publish to Zenodo (dry-run — add --confirm to actually publish)
conda run -n ilri01 python deliverables/zenodo_publish.py \
  --file deliverables/D4_v01.pdf --title "..." --version v01
```

---

## File / Version Guidelines

**These rules apply to all scripts that write to Google Drive:**

1. **Never overwrite an existing file.** Always create a new version.
2. **Version naming:** append `_v01`, `_v02`, etc. before the file extension.
   - e.g. `Deliverable 4_First Draft Systematic Map_v01`
   - Next revision → `_v02`, and so on
3. **Before creating:** list the folder contents, find the highest existing version number, increment by one.
4. **Native Google Docs** (not uploaded .docx): same rule — copy the doc, rename the copy with the new version suffix.
5. **Never delete or move existing versions** — leave the full history in place.

---

## Notes
- OAuth token expires after ~1 hour of inactivity; re-running auth refreshes it automatically
- Service account alternative: possible for fully automated (no browser) auth, but requires G Suite admin access
- All doc IDs are the long string in the Google Doc URL: `docs.google.com/document/d/DOC_ID_HERE/edit`
