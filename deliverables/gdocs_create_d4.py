"""
gdocs_create_d4.py

Creates Deliverable 4 (First Draft Systematic Map, preliminary) as a new
Google Doc in the Deliverables shared drive folder.

Strategy:
  1. Copy D3 (preserves Bristlepine cover page formatting, fonts, styles)
  2. Update cover page metadata (title, date, deliverable label)
  3. Delete D3 body content after the keywords line
  4. Insert D4 body content with correct heading styles

Never overwrites existing files — always creates _v01, _v02, etc.

Run: conda run -n ilri01 python deliverables/gdocs_create_d4.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Dict, Any

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# ── paths ─────────────────────────────────────────────────────────────────────
CREDS_DIR   = Path(__file__).resolve().parent / ".credentials"
TOKEN_FILE  = CREDS_DIR / "token.json"
SCOPES      = [
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/drive",
]

DELIVERABLES_FOLDER = "1alv6Ic7vtKZc_xo6AdlKP1exshMh-fR6"
D3_DOC_ID           = "1XN0YdGPnOBEMVLxvekGQ-ztYngOQkx84kn7r2v0Y2qU"
D4_BASE_NAME        = "Deliverable 4_Bristlepine_First Draft Systematic Map"

# ── pipeline stats (update if re-running after new pipeline outputs) ──────────
STATS = {
    "scopus_total":       17_021,
    "abstract_included":   6_218,
    "abstract_excluded":  10_803,
    "missing_abstract":    1_328,
    "fulltext_retrieved":  3_570,
    "fulltext_manual":     2_648,
    "fulltext_pct":           57,
    "coded_records":       1_899,
    "net_new_included":    2_271,
    "net_new_screened":    7_913,
    "search_date":        "January 2026",
    "report_date":        "April 2026",
    "version":            "v01",
}

# ── D4 body content ───────────────────────────────────────────────────────────
# Each tuple: (style, text)
# style: "HEADING_1" | "HEADING_2" | "HEADING_3" | "NORMAL_TEXT"
D4_BODY: List[tuple] = [
    ("NORMAL_TEXT", "Note: This is a preliminary first draft of the systematic map, based on Scopus database records only. Multi-database integration (Web of Science, CAB Abstracts, AGRIS, Academic Search Premier) is ongoing. Final results will be updated in Deliverable 5 (1 May 2026)."),
    ("HEADING_1",   "Executive Summary"),
    ("NORMAL_TEXT", f"This document presents the first draft of the systematic map of methods used to track and measure climate adaptation processes and outcomes for smallholder producers in the agriculture sector in low- and middle-income countries (LMICs). The map is preliminary, based on Scopus database records, and clearly labelled as such. Final results integrating all planned databases and full human-LLM calibration rounds will be delivered in Deliverable 5 (1 May 2026)."),
    ("NORMAL_TEXT", f"A total of {STATS['scopus_total']:,} records were identified through Scopus. After applying the PCCM (Population–Concept–Context–Methodological Focus) eligibility criteria at the title and abstract stage, {STATS['abstract_included']:,} records ({round(STATS['abstract_included']/STATS['scopus_total']*100, 1)}%) were retained for full-text review. Full-text retrieval is underway: {STATS['fulltext_retrieved']:,} full texts ({STATS['fulltext_pct']}%) were retrieved automatically; {STATS['fulltext_manual']:,} require manual access via institutional subscription. Preliminary LLM-assisted data extraction has produced an initial coded dataset of {STATS['coded_records']:,} studies, which form the basis of the evidence gap map and searchable database presented here."),
    ("NORMAL_TEXT", f"A parallel multi-database search (Web of Science, CAB Abstracts, AGRIS) has identified {STATS['net_new_screened']:,} additional records, of which {STATS['net_new_included']:,} were included at the abstract screening stage. These records will be integrated into the final systematic map (Deliverable 5)."),
    ("HEADING_1",   "Background"),
    ("NORMAL_TEXT", "The background, theory of change, stakeholder engagement process, and full PCCM eligibility criteria underpinning this systematic map are documented in Deliverable 3 (Final Systematic Map Protocol, January 2026). This section provides a brief overview for context."),
    ("NORMAL_TEXT", "A growing number of climate adaptation interventions are being deployed across the agriculture sector to support smallholder producers in LMICs. Despite growing investment, a central challenge persists: how to measure the effectiveness of these interventions. Monitoring and evaluation (M&E) in this space is particularly complex, due to long timeframes, evolving climate baselines, terminological differences, and attribution challenges. This systematic map responds to these challenges by systematically identifying, characterising, and comparing the methods used to assess climate adaptation interventions in the agriculture sector, with a focus on smallholder producers in LMICs."),
    ("HEADING_1",   "Methods"),
    ("NORMAL_TEXT", "This systematic map follows the protocol published as Deliverable 3 (Bristlepine Resilience Consultants, January 2026). A brief summary is provided here; readers are referred to that document for full methodological details."),
    ("HEADING_2",   "Search Strategy"),
    ("NORMAL_TEXT", f"Searches were conducted in Scopus (Elsevier) in {STATS['search_date']}, using a structured Boolean search string developed around the PCCM framework (Population: smallholder producers; Concept: adaptation processes and outcomes; Context: climate hazards, LMICs, agriculture sector; Methodological Focus: measurement, indicators, monitoring and evaluation approaches). Searches were restricted to literature published from 2005 onward and to English, French, and Spanish language sources. A total of {STATS['scopus_total']:,} records were retrieved."),
    ("NORMAL_TEXT", "Additional databases searched in parallel include Web of Science Core Collection, CAB Abstracts, and AGRIS. Results from these databases are being integrated and will be fully reported in Deliverable 5. Academic Search Premier and grey literature sources (approximately 20 repositories per §3.3 of the protocol) are also being searched."),
    ("HEADING_2",   "Deduplication"),
    ("NORMAL_TEXT", "Records from all databases were deduplicated using a multi-stage matching algorithm: exact DOI match, then exact title and year match, then fuzzy title similarity (Jaccard ≥ 0.85). Scopus records were used as the primary reference set; net-new records from other databases were identified against this base."),
    ("HEADING_2",   "Screening"),
    ("NORMAL_TEXT", f"Title and abstract screening was conducted using a validated LLM-assisted screening tool (Ollama/Llama3), calibrated against the PCCM eligibility criteria. Two calibration rounds were completed prior to full-corpus screening, achieving a sensitivity of 0.966–0.970 against human-coded benchmarks. All {STATS['scopus_total']:,} Scopus records were screened at the title and abstract stage. Records flagged as missing abstracts ({STATS['missing_abstract']:,} records, {round(STATS['missing_abstract']/STATS['scopus_total']*100,1)}%) were flagged for manual review."),
    ("HEADING_2",   "Full-Text Retrieval"),
    ("NORMAL_TEXT", f"Full texts were retrieved for all {STATS['abstract_included']:,} included records using automated retrieval from Unpaywall, Elsevier full-text API, Semantic Scholar, OpenAlex, Frontiers direct PDF, and CORE.ac.uk. A total of {STATS['fulltext_retrieved']:,} full texts ({STATS['fulltext_pct']}%) were retrieved automatically. The remaining {STATS['fulltext_manual']:,} records require manual access via Cornell University institutional subscriptions; this retrieval pass is ongoing."),
    ("HEADING_2",   "Data Extraction"),
    ("NORMAL_TEXT", f"Preliminary data extraction was conducted using an LLM-assisted coding pipeline applied to the {STATS['fulltext_retrieved']:,} retrieved full texts, guided by the codebook described in the systematic map protocol (Deliverable 3). The codebook covers 16 fields, including: publication year, country/region, producer type, climate hazard type, adaptation intervention type, methodological approach, outcome type, and equity dimensions. A total of {STATS['coded_records']:,} records were successfully coded in this preliminary pass. Human-LLM calibration rounds (iterative 5-paper buckets per §4 of the protocol) are ongoing; calibrated results will be reported in Deliverable 5."),
    ("HEADING_1",   "Preliminary Results"),
    ("HEADING_2",   "Search and Screening Summary"),
    ("NORMAL_TEXT", f"Figure 1 presents the ROSES (Reporting Standards for Systematic Evidence Syntheses) flow diagram for the Scopus search and screening process."),
    ("NORMAL_TEXT", f"Table 1. Search and screening results summary (Scopus, preliminary)."),
    ("NORMAL_TEXT", f"Records identified (Scopus): {STATS['scopus_total']:,}"),
    ("NORMAL_TEXT", f"Records included after title/abstract screening: {STATS['abstract_included']:,} ({round(STATS['abstract_included']/STATS['scopus_total']*100,1)}%)"),
    ("NORMAL_TEXT", f"Records excluded: {STATS['abstract_excluded']:,} (primary exclusion criterion: Concept — {round(9186/STATS['abstract_excluded']*100,0):.0f}%)"),
    ("NORMAL_TEXT", f"Records with missing abstracts flagged for manual review: {STATS['missing_abstract']:,}"),
    ("NORMAL_TEXT", f"Full texts retrieved automatically: {STATS['fulltext_retrieved']:,} ({STATS['fulltext_pct']}% of included records)"),
    ("NORMAL_TEXT", f"Full texts requiring manual retrieval: {STATS['fulltext_manual']:,}"),
    ("NORMAL_TEXT", f"Records coded in preliminary data extraction: {STATS['coded_records']:,}"),
    ("HEADING_2",   "Evidence Gap Map"),
    ("NORMAL_TEXT", "Figure 2 presents the preliminary evidence gap map, showing the distribution of {coded_records:,} coded studies across key dimensions: geography, producer type, adaptation intervention domain, measurement methodology, and temporal coverage. The map is interactive and available at the searchable database link below.".format(**STATS)),
    ("NORMAL_TEXT", "Key preliminary observations (to be confirmed and expanded in Deliverable 5):"),
    ("NORMAL_TEXT", "Geographic distribution: Studies are concentrated in Sub-Saharan Africa and South Asia, with sparser coverage of East Asia, Latin America, and the Middle East and North Africa."),
    ("NORMAL_TEXT", "Producer types: Crop producers dominate the evidence base; pastoralists, fishers, and forestry producers are underrepresented."),
    ("NORMAL_TEXT", "Methodological focus: Survey-based and quasi-experimental approaches are most common. Participatory and mixed-methods approaches represent a smaller but growing share of the evidence."),
    ("NORMAL_TEXT", "Temporal trends: Publication volume has grown substantially since 2010, with notable acceleration post-2015 (Paris Agreement)."),
    ("HEADING_1",   "Searchable Database"),
    ("NORMAL_TEXT", "The preliminary searchable database is available at: https://main.d3a9jahzuu7xai.amplifyapp.com/systematic-map"),
    ("NORMAL_TEXT", "The database includes all {coded_records:,} records coded in the preliminary data extraction pass. Users can filter by country, producer type, adaptation domain, methodology, and publication year. A CSV download of the full dataset is available on the same page.".format(**STATS)),
    ("NORMAL_TEXT", "Note: the database will be updated continuously as multi-database integration, full-text screening, and human-LLM calibration rounds are completed. The final version will be published with Deliverable 5."),
    ("HEADING_1",   "Limitations"),
    ("NORMAL_TEXT", "This preliminary report has several important limitations that will be addressed in Deliverable 5:"),
    ("NORMAL_TEXT", "1. Scopus-only search. The evidence base currently reflects Scopus records only. Integration of Web of Science, CAB Abstracts, AGRIS, Academic Search Premier, and grey literature sources is ongoing. Preliminary abstract screening of these additional databases has identified {net_new_included:,} additional included records.".format(**STATS)),
    ("NORMAL_TEXT", "2. Incomplete full-text screening. Full-text screening has not yet been completed. The preliminary data extraction covers full texts retrieved automatically ({fulltext_pct}% of included records); the remaining {fulltext_manual:,} records requiring manual access are not yet coded.".format(**STATS)),
    ("NORMAL_TEXT", "3. Preliminary data extraction. LLM-assisted data extraction has not yet undergone the full human-LLM calibration protocol described in Deliverable 3. Calibration rounds (iterative 5-paper buckets) are in progress; results will be updated in Deliverable 5."),
    ("NORMAL_TEXT", "4. Grey literature. Manual searching of approximately 20 organisational repositories (CGIAR, World Bank, 3ie, GCF, FAO, IFAD, regional development banks) is pending."),
    ("HEADING_1",   "Next Steps"),
    ("NORMAL_TEXT", "The following steps are planned prior to the final systematic map (Deliverable 5, 1 May 2026):"),
    ("NORMAL_TEXT", "Multi-database integration — full-text retrieval and screening for the {net_new_included:,} additional included records from Web of Science, CAB Abstracts, and AGRIS; Academic Search Premier export pending.".format(**STATS)),
    ("NORMAL_TEXT", "Grey literature search — manual search of approximately 20 organisational repositories per the protocol (Deliverable 3, §3.3); due 25 April 2026."),
    ("NORMAL_TEXT", "Full-text retrieval — manual retrieval pass via Cornell University institutional subscriptions for the {fulltext_manual:,} records not retrieved automatically.".format(**STATS)),
    ("NORMAL_TEXT", "Human-LLM calibration — iterative calibration rounds (5-paper buckets) with two human reviewers (Jennifer Denno Cissé, Caroline G. Staub); target κ ≥ 0.60 and sensitivity ≥ 0.80 per closed field; saturation criterion for emergent fields."),
    ("NORMAL_TEXT", "Protocol amendment (Deliverable 5.7) — documenting all deviations from Deliverable 3, including: dropped full-text calibration round, field 16 consolidation, calibration round size reduction (100 → 5 papers), and net-new database additions."),
    ("NORMAL_TEXT", "Final systematic map (Deliverable 5) — updated ROSES flow diagram, final searchable extraction database, evidence gap map, and full systematic map report."),
    ("HEADING_1",   "References"),
    ("NORMAL_TEXT", "Bristlepine Resilience Consultants (2026). Deliverable 3: Final Systematic Map Protocol — Measuring what matters: tracking climate adaptation processes and outcomes for smallholder producers in the agriculture sector. January 2026, v1. Zenodo. https://zenodo.org/records/18370029"),
    ("NORMAL_TEXT", "Collaboration for Environmental Evidence (2018). Guidelines and Standards for Evidence Synthesis in Environmental Management. Version 5.1. Pullin AS, Frampton GK, Livoreil B, Petrokofsky G (eds)."),
    ("NORMAL_TEXT", "Haddaway, N.R., Macura, B., Whaley, P. and Pullin, A.S. (2018). ROSES Reporting standards for Systematic Evidence Syntheses: pro forma, flow diagram and descriptive summary of the plan and conduct of environmental systematic reviews and systematic maps. Environmental Evidence 7, 7."),
]


# =============================================================================
# Helpers
# =============================================================================

def get_creds() -> Credentials:
    creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return creds


def next_version_name(drive, folder_id: str, base_name: str) -> str:
    """Find highest existing _vNN suffix and return base_name + _v(N+1)."""
    results = drive.files().list(
        q=f"'{folder_id}' in parents and trashed=false and name contains '{base_name}'",
        fields="files(name)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()

    max_v = 0
    for f in results.get("files", []):
        name = f["name"]
        import re
        m = re.search(r"_v(\d+)$", name)
        if m:
            max_v = max(max_v, int(m.group(1)))

    v = max_v + 1
    return f"{base_name}_v{v:02d}"


def copy_doc(drive, source_id: str, new_name: str, parent_id: str) -> str:
    """Copy a Google Doc and return the new doc ID."""
    body = {"name": new_name, "parents": [parent_id]}
    result = drive.files().copy(
        fileId=source_id,
        body=body,
        supportsAllDrives=True,
    ).execute()
    return result["id"]


def get_body_content(docs, doc_id: str):
    """Return the body content array from a Google Doc."""
    doc = docs.documents().get(documentId=doc_id).execute()
    return doc.get("body", {}).get("content", [])


def find_keyword_end_index(content) -> int:
    """
    Find the end index of the keywords paragraph (last cover-page element).
    Everything after this will be deleted and replaced with D4 body.
    """
    for elem in reversed(content):
        para = elem.get("paragraph")
        if not para:
            continue
        text = "".join(
            r.get("textRun", {}).get("content", "")
            for r in para.get("elements", [])
        )
        if "keywords" in text.lower() or "Keywords" in text:
            end = elem.get("endIndex", 0)
            return end
    # fallback: after first 10 elements
    return content[10].get("endIndex", 20) if len(content) > 10 else 20


def build_requests(content, keyword_end_idx: int) -> list:
    """
    Build the batchUpdate request list:
    1. Delete body content after cover page
    2. Replace cover-page text fields
    3. Insert D4 body with heading styles
    """
    requests = []

    # ── 1. Delete body after cover page ──────────────────────────────────────
    # Find the end of the document body (last content element)
    last_idx = max(e.get("endIndex", 0) for e in content) - 1
    if last_idx > keyword_end_idx:
        requests.append({
            "deleteContentRange": {
                "range": {
                    "startIndex": keyword_end_idx,
                    "endIndex": last_idx,
                }
            }
        })

    # ── 2. Update cover page metadata ────────────────────────────────────────
    replacements = {
        "Deliverable 3: Systematic Map Protocol": "Deliverable 4: First Draft Systematic Map (Preliminary)",
        "January 25, 2026, v1": f"{STATS['report_date']}, {STATS['version']}",
        "Deliverable 3_Bristlepine_Final Systematic Map Protocol_v1": "Deliverable 4_Bristlepine_First Draft Systematic Map_v01",
    }
    for old, new in replacements.items():
        requests.append({
            "replaceAllText": {
                "containsText": {"text": old, "matchCase": True},
                "replaceText": new,
            }
        })

    # ── 3. Insert D4 body content ─────────────────────────────────────────────
    # Insert in reverse order so indices don't shift
    insert_at = keyword_end_idx
    for style, text in reversed(D4_BODY):
        requests.append({
            "insertText": {
                "location": {"index": insert_at},
                "text": "\n" + text,
            }
        })
        # Apply paragraph style if heading
        if style != "NORMAL_TEXT":
            requests.append({
                "updateParagraphStyle": {
                    "range": {
                        "startIndex": insert_at + 1,
                        "endIndex": insert_at + 1 + len(text),
                    },
                    "paragraphStyle": {"namedStyleType": style},
                    "fields": "namedStyleType",
                }
            })

    return requests


# =============================================================================
# Main
# =============================================================================

def run():
    creds = get_creds()
    drive = build("drive", "v3", credentials=creds)
    docs  = build("docs",  "v1", credentials=creds)

    # 1. Find next version name
    new_name = next_version_name(drive, DELIVERABLES_FOLDER, D4_BASE_NAME)
    print(f"[gdocs] Creating: {new_name}")

    # 2. Copy D3 → D4_v01
    print(f"[gdocs] Copying D3 ({D3_DOC_ID}) ...")
    new_id = copy_doc(drive, D3_DOC_ID, new_name, DELIVERABLES_FOLDER)
    print(f"[gdocs] Created: {new_id}")
    print(f"[gdocs] URL: https://docs.google.com/document/d/{new_id}/edit")

    # Give Drive a moment to settle
    time.sleep(3)

    # 3. Get body content to find cover-page end index
    print("[gdocs] Reading new doc structure ...")
    content = get_body_content(docs, new_id)
    keyword_end = find_keyword_end_index(content)
    print(f"[gdocs] Cover page ends at index {keyword_end}")

    # 4. Build and apply batchUpdate
    print("[gdocs] Applying D4 content ...")
    reqs = build_requests(content, keyword_end)
    print(f"[gdocs] Sending {len(reqs)} requests ...")
    docs.documents().batchUpdate(
        documentId=new_id,
        body={"requests": reqs},
    ).execute()

    print(f"\n[gdocs] Done!")
    print(f"  Name : {new_name}")
    print(f"  ID   : {new_id}")
    print(f"  URL  : https://docs.google.com/document/d/{new_id}/edit")

    # 5. Update googledocs.md with the new doc ID
    md_path = Path(__file__).resolve().parent / "googledocs.md"
    if md_path.exists():
        text = md_path.read_text()
        old = "| D4: First Draft Systematic Map (to create) | — | — |"
        new = f"| D4: First Draft Systematic Map | https://docs.google.com/document/d/{new_id}/edit | `{new_id}` |"
        if old in text:
            md_path.write_text(text.replace(old, new))
            print(f"[gdocs] Updated deliverables/googledocs.md")

    return new_id


if __name__ == "__main__":
    run()
