"""
publish_d5_surgical.py

Surgical edit of D4 → D5:
- Copies D4 v02 (preserves ALL formatting: watermark, fonts, named styles, header/footer)
- Rewrites section bodies from BOTTOM to TOP (so earlier deletions don't shift later indices)
- Uploads human PNGs to Drive, inserts inline at figure positions
- Does NOT wipe the body — named styles (HEADING_1/2/NORMAL_TEXT) stay intact

Structure produced:
  1. Key findings   ← was Executive Summary
  2. Background     ← kept from D4
  3. Methods        ← updated (PRISMA, table, calibration brief, saturation)
  4. Final results  ← was Preliminary Results
     4.1 Screening summary
     4.2 Evidence gap map
     4.3 Geographic distribution   (new)
     4.4 Equity & inclusion        (new)
     4.5 Publication trends        (new)
  5. Searchable database
  6. Conclusions    ← replaces §6 Limitations + §7 Next Steps + old §8
  7. Acknowledgements
  8. References

Run: python3 deliverables/publish_d5_surgical.py
"""

from __future__ import annotations
import re, time
from pathlib import Path
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

HERE   = Path(__file__).resolve().parent
ROOT   = HERE.parent
STEP16 = ROOT / "scripts" / "outputs" / "step16"
HUMAN  = STEP16 / "interactive" / "human"

TOKEN_FILE = HERE / ".credentials" / "token.json"
SCOPES     = ["https://www.googleapis.com/auth/documents",
              "https://www.googleapis.com/auth/drive"]
FOLDER     = "1alv6Ic7vtKZc_xo6AdlKP1exshMh-fR6"
D4_ID      = "14Hejjfr63HdxH21bqBxLZe3AsXxmLBrXYqJJk7ZbBxU"

# ── Figures (label, path, width_pt, height_pt) ────────────────────────────────
FIGS = {
    "EGM":      (HUMAN / "evidence_gap_map.png",  450, 285),
    "GEO_MAP":  (HUMAN / "geographic_map.png",    450, 235),
    "GEO_BAR":  (HUMAN / "geographic_bar.png",    400, 220),
    "EQUITY":   (HUMAN / "equity.png",            400, 210),
    "TEMPORAL": (HUMAN / "temporal_trends.png",   400, 200),
    "PRISMA":   (HERE  / "prisma_flow_d5.png",    450, 340),
    "SAT":      (STEP16 / "saturation.png",       450, 260),
}

GREEN    = {"red": 33/255,  "green": 71/255,  "blue": 46/255}
CHARCOAL = {"red": 60/255,  "green": 60/255,  "blue": 60/255}
WHITE    = {"red": 1.0,     "green": 1.0,     "blue": 1.0}

KEY_FINDINGS = [
    ("A recent but thin evidence base (96%)",
     "96% of studies published since 2015 (median: 2022). Long-term impact evidence is nearly absent."),
    ("Marginalized groups largely invisible (86%)",
     "86% of studies provide no equity disaggregation — by gender, age, ethnicity, or disability."),
    ("Non-crop producers severely underrepresented (<6%)",
     "Fisheries/aquaculture and agroforestry appear in fewer than 6% of studies combined."),
    ("Cost data almost entirely absent (~80%)",
     "~80% of studies report no cost or efficiency data, making value-for-money assessment near-impossible."),
    ("Three countries dominate the evidence base (28%)",
     "Ethiopia, Ghana, and Kenya account for 28% of all human-coded studies."),
    ("Process outcomes studied; impact outcomes are not",
     "Knowledge, adoption, and decision-making are well covered. Income, wellbeing, and resilience remain sparse."),
]

PIPELINE_ROWS = [
    ["Database", "Returned", "After dedup", "Abstr. incl.", "FT retrieved", "FT screened", "Coded"],
    ["Scopus",                   "17,021", "17,021", "6,218",   "2,644", "2,644", "—"],
    ["Web of Science",           "15,179",  "4,683", "1,140",     "552",   "552", "—"],
    ["CAB Abstracts",             "5,723",  "3,229", "1,133",     "260",   "260", "—"],
    ["Academic Search Premier",   "1,187",    "274",    "66",      "20",    "20", "—"],
    ["AGRIS",                         "3",      "1",     "1",       "0",    "—", "—"],
    ["Total",                    "39,113", "25,208", "8,558",   "3,476", "3,464", "2,368"],
]

# ── API helpers ────────────────────────────────────────────────────────────────

def get_creds():
    c = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
    if c.expired and c.refresh_token: c.refresh(Request())
    return c

def get_body(docs, did):
    return docs.documents().get(documentId=did).execute()["body"]["content"]

def para_text(e):
    return "".join(r.get("textRun", {}).get("content", "") for r in e.get("paragraph", {}).get("elements", []))

def para_style(e):
    return e.get("paragraph", {}).get("paragraphStyle", {}).get("namedStyleType", "")

def find_heading(content, search):
    """Return (startIndex, endIndex) of first HEADING_1/2 paragraph containing search."""
    for e in content:
        if para_style(e) in ("HEADING_1", "HEADING_2") and search in para_text(e):
            return e.get("startIndex"), e.get("endIndex")
    return None, None

def section_body_range(content, h1_text, next_h1_text=None):
    """Return (body_start, body_end) — content between h1_text heading and next_h1_text heading."""
    h_si, h_ei = find_heading(content, h1_text)
    if h_si is None:
        return None, None
    # Start of body = immediately after heading paragraph
    body_start = h_ei
    if next_h1_text:
        n_si, _ = find_heading(content, next_h1_text)
        body_end = n_si if n_si else body_start
    else:
        body_end = content[-1].get("endIndex", body_start) - 1
    return body_start, body_end

def delete_range(docs, did, si, ei):
    if ei <= si: return
    docs.documents().batchUpdate(documentId=did, body={"requests": [
        {"deleteContentRange": {"range": {"startIndex": si, "endIndex": ei}}}
    ]}).execute()
    time.sleep(0.5)

def insert_text(docs, did, idx, text):
    docs.documents().batchUpdate(documentId=did, body={"requests": [
        {"insertText": {"location": {"index": idx}, "text": text}}
    ]}).execute()
    time.sleep(0.3)

def apply_heading(docs, did, si, ei, level):
    named = f"HEADING_{level}"
    docs.documents().batchUpdate(documentId=did, body={"requests": [
        {"updateParagraphStyle": {
            "range": {"startIndex": si, "endIndex": ei},
            "paragraphStyle": {"namedStyleType": named},
            "fields": "namedStyleType",
        }}
    ]}).execute()
    time.sleep(0.2)

def upload_fig(drive, png: Path) -> str | None:
    if not png.exists():
        print(f"  WARN: {png.name} not found"); return None
    meta  = {"name": f"d5_{png.name}", "parents": [FOLDER]}
    media = MediaFileUpload(str(png), mimetype="image/png", resumable=False)
    fid   = drive.files().create(body=meta, media_body=media,
                                 fields="id", supportsAllDrives=True).execute()["id"]
    drive.permissions().create(fileId=fid, body={"type": "anyone", "role": "reader"},
                               supportsAllDrives=True).execute()
    return f"https://lh3.googleusercontent.com/d/{fid}"

def insert_image(docs, did, idx, url, w, h):
    docs.documents().batchUpdate(documentId=did, body={"requests": [{
        "insertInlineImage": {
            "location": {"index": idx},
            "uri": url,
            "objectSize": {
                "width":  {"magnitude": w, "unit": "PT"},
                "height": {"magnitude": h, "unit": "PT"},
            },
        }
    }]}).execute()
    time.sleep(1)

def tbl_cell_bg(ts, row, col, color):
    return {"updateTableCellStyle": {
        "tableRange": {
            "tableCellLocation": {
                "tableStartLocation": {"index": ts},
                "rowIndex": row, "columnIndex": col,
            },
            "rowSpan": 1, "columnSpan": 1,
        },
        "tableCellStyle": {"backgroundColor": {"color": {"rgbColor": color}}},
        "fields": "backgroundColor",
    }}


# ── Content blocks ─────────────────────────────────────────────────────────────

def _lines(*args):
    """Join a list of strings with newlines, ending with a single newline."""
    return "\n".join(args) + "\n"


SEC1_KEY_FINDINGS = _lines(
    "This document presents the final systematic map of methods used to track and measure "
    "climate adaptation processes and outcomes for smallholder producers in LMICs "
    "(Deliverable 5, June 2026). Human-coded results are based on 86 studies coded across "
    "five batches; information saturation was reached by batch three, confirming the sample "
    "is representative. Six findings stand out:\n",
    *[f"• {label}: {desc}" for label, desc in KEY_FINDINGS],
    "\nAn interactive evidence gap map and searchable database are available at: "
    "https://main.d3a9jahzuu7xai.amplifyapp.com/systematic-map",
)

SEC3_4_UPDATE = _lines(
    "Data extraction used a random sampling approach. Papers were drawn in batches of 20 "
    "(fixed integer seeds 42–46; each batch excludes prior-batch DOIs). Coders applied the "
    "five PCCM inclusion criteria to confirm inclusion, then completed 16 extraction fields "
    "(country, scale, producer type, adaptation focus, domain, methodology, equity, etc.).",
    "",
    "Information saturation was tracked at each batch across three dimensions: process/outcome "
    "domains (13 canonical values), methodological approaches (5), and producer types (5). "
    "All three dimensions reached saturation by batch FT-R2c (49 papers coded) — zero new "
    "canonical categories were added across the final two batches (37 additional papers). "
    "Five batches and 86 papers were coded in total.",
)

SEC4_1_TEXT = _lines(
    "A total of 39,113 records were identified across five bibliographic databases and 24 grey "
    "literature sources. After deduplication, 25,208 unique records entered title and abstract "
    "screening. The validated LLM screening tool identified 8,558 records for full-text "
    "assessment. Of 3,476 full texts retrieved (40%), 3,464 were screened and 2,368 were "
    "included (LLM exploratory reference). A random sample of 86 papers was human-coded as the "
    "primary authoritative output. Table 1 gives the full breakdown by database.\n",
    "Table 1. Search and screening results by database. FT = full text. All FT screening was "
    "conducted on automatically retrieved records across all databases together. Records "
    "requiring manual library access (n = 5,243) are not reflected in these totals.",
    "",
    "Figure 1 presents the PRISMA flow diagram showing the full record flow across all 29 "
    "sources and screening stages, with the human-coded track (primary) and LLM track "
    "(exploratory reference) shown separately.",
    "",
    "[FIG_PRISMA]",
    "",
    "Figure 1. PRISMA flow diagram — record flow across 29 sources. Human-coded track "
    "(amber, n = 86) = primary output. LLM track (n = 2,368) = exploratory reference.",
)

SEC4_2_EGM = _lines(
    "The evidence gap map (EGM) is the primary output of this deliverable. It plots the "
    "distribution of human-coded studies (n = 86) across process and outcome domains (y-axis) "
    "and producer types (x-axis). Bubble size is proportional to the number of studies in each "
    "cell. Grey circles indicate evidence gaps — no studies in that combination.\n",
    "[FIG_EGM]",
    "",
    "Figure 2. Evidence gap map — human-coded results (n = 86). Blue = process domains · "
    "green = outcome domains · grey = evidence gaps. Bubble area ∝ number of studies.\n",
    "Key observations:",
    "• Process domains dominate. Decision-making & planning, knowledge & awareness, and "
    "uptake & adoption are the most studied domains, especially for crop farmers.",
    "• Outcome domains are nearly absent. Income & assets, wellbeing, risk reduction, and "
    "resilience are sparsely covered — and almost exclusively for crop systems.",
    "• Non-crop producer types are systematically uncovered. Livestock, fisheries, agroforestry, "
    "and mixed-system producers appear in very few cells; most cells for these types are gaps.",
    "• Institutional governance and access to services are absent across all producer types.",
)

SEC4_3_GEO = _lines(
    "Studies are heavily concentrated in Sub-Saharan Africa and South Asia. Ethiopia, Ghana, "
    "and Kenya alone account for 28% of all human-coded studies. Large evidence gaps persist "
    "across Central America, Southeast Asia, the Sahel, and MENA.\n",
    "[FIG_GEO_MAP]",
    "",
    "Figure 3. Geographic distribution of included studies (human-coded, n = 86).\n",
    "[FIG_GEO_BAR]",
    "",
    "Figure 4. Top countries by study count (human-coded, n = 86).",
)

SEC4_4_EQUITY = _lines(
    "86% of included studies provide no equity disaggregation. Where equity dimensions are "
    "addressed, gender is the most common focus — but still features in fewer than 15% of "
    "studies. Youth, indigenous peoples, people with disabilities, and pastoralist groups are "
    "nearly absent. The evidence base does not systematically track who benefits from "
    "adaptation interventions.\n",
    "[FIG_EQUITY]",
    "",
    "Figure 5. Equity and inclusion dimensions across included studies (human-coded, n = 86). "
    "Red bar = studies with no marginalized group focus.",
)

SEC4_5_TEMPORAL = _lines(
    "96% of included studies were published since 2015, with a median year of 2022. Publication "
    "volume has grown sharply — but this means that long-term impact evidence is very thin: "
    "most studies are too recently published to measure multi-year adaptation outcomes.\n",
    "[FIG_TEMPORAL]",
    "",
    "Figure 6. Publication trends across included studies (human-coded, n = 86).\n",
    "Saturation analysis (Figure 7) shows that all three tracked dimensions — process/outcome "
    "domains, methodological approaches, and producer types — reached saturation by 49 papers. "
    "Zero new canonical categories were added in the final two batches, confirming the "
    "human-coded sample is representative of the broader evidence base.\n",
    "[FIG_SAT]",
    "",
    "Figure 7. Information saturation curve. Top: cumulative unique categories as % of final "
    "total by papers coded. Bottom: new categories per batch. All dimensions plateau by 49 papers.",
)

SEC5_DB = _lines(
    "The interactive evidence gap map and searchable database are available at:\n",
    "https://main.d3a9jahzuu7xai.amplifyapp.com/systematic-map\n",
    "The database includes all 2,368 LLM-screened records with key metadata (year, title, "
    "country, producer type, domain, methodology). A toggle switches between the human-coded "
    "results (n = 86, authoritative) and the LLM corpus (n = 2,368, exploratory). All data "
    "and code are available at github.com/bristlepine/ilri-climate-adaptation-effectiveness",
)

SEC6_CONCLUSIONS = _lines(
    "The evidence gap map reveals a systematic mismatch between what adaptation M&E currently "
    "measures and what is needed to understand adaptation effectiveness.\n",
    "• Process outcomes are well-documented, impact outcomes are not. The literature reliably "
    "captures whether farmers adopted a practice or gained knowledge — it almost never measures "
    "whether adaptation improved incomes, wellbeing, or resilience.",
    "• Non-crop producer types are almost entirely unmeasured. Livestock, fisheries, agroforestry, "
    "and mixed-system producers appear in very few cells of the evidence map. Most are pure gaps.",
    "• Geographic concentration limits generalisability. Three countries account for 28% of all "
    "studies. The evidence base cannot support regional conclusions across most LMICs.",
    "• The equity gap is the most critical for policy. 86% of studies do not track who benefits. "
    "This makes it impossible to know whether adaptation is reaching the most vulnerable groups.\n",
    "Addressing these gaps requires deliberate investment in long-term cohort studies, "
    "equity-disaggregated data collection, and research that explicitly targets non-crop "
    "livelihood systems. The next phase (Deliverable 6) will develop a systematic review and "
    "meta-analysis protocol targeting the domains where evidence is densest.",
)


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    cr    = get_creds()
    drive = build("drive", "v3", credentials=cr)
    docs  = build("docs",  "v1", credentials=cr)

    # 1. Upload figures
    print("[d5] Uploading figures...")
    fig_urls = {}
    for label, (png, w, h) in FIGS.items():
        url = upload_fig(drive, png)
        fig_urls[label] = (url, w, h)
        print(f"  {label}: {'ok' if url else 'MISS'}")
        time.sleep(0.3)

    # 2. Copy D4 → fresh D5
    print("[d5] Copying D4 v02 → D5...")
    did = drive.files().copy(
        fileId=D4_ID,
        body={"name": "Deliverable 5_Bristlepine_Final Systematic Map_v01", "parents": [FOLDER]},
        supportsAllDrives=True,
    ).execute()["id"]
    print(f"[d5] Created: {did}")
    print(f"[d5] URL: https://docs.google.com/document/d/{did}/edit")
    time.sleep(3)

    # 3. Simple text replacements
    print("[d5] Applying text replacements...")
    replacements = [
        ("Deliverable 4: First Draft Systematic Map (Preliminary)", "Deliverable 5: Final Systematic Map"),
        ("Deliverable 4: First Draft Systematic Map",               "Deliverable 5: Final Systematic Map"),
        ("First Draft Systematic Map (Preliminary)",                "Final Systematic Map"),
        ("First Draft Systematic Map",                              "Final Systematic Map"),
        ("1.  Executive Summary",    "1.  Key findings"),
        ("4.  Preliminary Results",  "4.  Final results"),
        ("4. Preliminary Results",   "4. Final results"),
        ("April 2026, v02",          "June 2026, v01"),
        ("April 2026",               "June 2026"),
        (" (Preliminary)", ""),
        ("(Preliminary) ", ""),
        ("Preliminary — ", ""),
        (", preliminary",  ""),
        (" preliminary ",  " "),
        ("Ollama/Llama3",  "Ollama/qwen2.5:14b"),
        ("will be reported in Deliverable 5", "are reported in this deliverable"),
    ]
    reqs = [{"replaceAllText": {
        "containsText": {"text": o, "matchCase": True},
        "replaceText": n,
    }} for o, n in replacements if o != n]
    res = docs.documents().batchUpdate(documentId=did, body={"requests": reqs}).execute()
    n_subs = sum(r.get("replaceAllTextResponse", {}).get("occurrencesChanged", 0)
                 for r in res.get("replies", []))
    print(f"  {n_subs} substitutions")
    time.sleep(2)

    # 4. BOTTOM-UP section rewrites
    # ─────────────────────────────────────────────────────────────────────────
    # Helper: read → find section body → delete body → insert new content
    def rewrite_section(heading_search, next_heading_search, new_content, label=""):
        content = get_body(docs, did)
        bs, be  = section_body_range(content, heading_search, next_heading_search)
        if bs is None:
            print(f"  WARN: '{heading_search}' not found"); return
        delete_range(docs, did, bs, be)
        time.sleep(0.5)
        insert_text(docs, did, bs, new_content)
        print(f"  Rewrote: {label or heading_search[:40]}")

    # §8 Conclusions — remove Limitations (§6), Next Steps (§7), and old Conclusions,
    #                  replace with single new Conclusions section.
    # Step a: delete §7 Next Steps body + heading
    print("[d5] Removing §6 Limitations, §7 Next Steps (merging into §6 Conclusions)...")
    content = get_body(docs, did)
    lim_si, _ = find_heading(content, "6.  Limitations")   # may be renamed
    if lim_si is None:
        lim_si, _ = find_heading(content, "Limitations")
    con_si, _ = find_heading(content, "8.  Conclusions")
    if lim_si and con_si and lim_si < con_si:
        # Delete from start of §6 heading through start of §8 heading
        delete_range(docs, did, lim_si, con_si)
        time.sleep(0.5)
        # Now §8 has shifted; renumber heading to §6
        content = get_body(docs, did)
        new_con_si, new_con_ei = find_heading(content, "Conclusions")
        if new_con_si:
            # Replace heading text
            docs.documents().batchUpdate(documentId=did, body={"requests": [
                {"deleteContentRange": {"range": {"startIndex": new_con_si, "endIndex": new_con_ei}}},
                {"insertText": {"location": {"index": new_con_si}, "text": "6.  Conclusions\n"}},
            ]}).execute()
            time.sleep(0.5)
            content = get_body(docs, did)
            new_con_si, new_con_ei = find_heading(content, "6.  Conclusions")
            if new_con_si:
                apply_heading(docs, did, new_con_si, new_con_ei, 1)
        print("  §6/7 removed, Conclusions renumbered to §6")
    else:
        print("  WARN: Could not find Limitations or Conclusions to merge")

    # §6 Conclusions body (now renumbered) — rewrite
    rewrite_section("6.  Conclusions", "Acknowledgements", SEC6_CONCLUSIONS, "§6 Conclusions")

    # §5 Searchable Database body — update
    rewrite_section("5.  Searchable Database", "6.  Conclusions", SEC5_DB, "§5 Searchable Database")

    # §4.2 EGM — rewrite body
    # First add new subsections 4.3–4.5 after §4.2, before §5
    # Strategy: rewrite entire §4 body at once (from §4.1 through end of §4.2)
    print("[d5] Rewriting §4 results...")
    content = get_body(docs, did)
    # Find §4 heading end and §5 heading start
    _, h4_ei  = find_heading(content, "4.  Final results")
    h5_si, _  = find_heading(content, "5.  Searchable Database")
    if h4_ei and h5_si:
        # Delete all of §4 body (everything between §4 heading end and §5 heading start)
        delete_range(docs, did, h4_ei, h5_si)
        time.sleep(0.5)

        # Rebuild §4 body: 4.1 + table marker + 4.2 EGM + 4.3 Geo + 4.4 Equity + 4.5 Temporal
        idx = h4_ei

        # §4.1 heading + content + table placeholder
        insert_text(docs, did, idx, "4.1  Search and screening summary\n")
        content = get_body(docs, did)
        h41_si, h41_ei = find_heading(content, "4.1  Search")
        if h41_si: apply_heading(docs, did, h41_si, h41_ei, 2)
        time.sleep(0.3)
        idx = get_body(docs, did)[-1].get("endIndex", 2) - 1
        insert_text(docs, did, idx, SEC4_1_TEXT)

        # Insert actual pipeline table (delete [TABLE] placeholder, insert table)
        content = get_body(docs, did)
        h5_si_new, _ = find_heading(content, "5.  Searchable Database")
        # Find the table marker position
        tbl_marker = "[TABLE]"
        tbl_si = tbl_ei = None
        for e in content:
            txt = para_text(e)
            if "Table 1. Search and screening" in txt:
                # Insert table right after this caption paragraph
                tbl_si = e.get("endIndex")
                break
        # (Table will be inserted in step 5 below)

        # §4.2 EGM
        content = get_body(docs, did)
        cur_end = content[-1].get("endIndex", 2) - 1
        # Find position just before §5
        content = get_body(docs, did)
        h5_si_now, _ = find_heading(content, "5.  Searchable Database")
        insert_pos = h5_si_now

        insert_text(docs, did, insert_pos, "4.2  Evidence gap map\n")
        content = get_body(docs, did)
        h5_si_now, _ = find_heading(content, "5.  Searchable Database")
        h42_si, h42_ei = find_heading(content, "4.2  Evidence gap map")
        if h42_si: apply_heading(docs, did, h42_si, h42_ei, 2)
        time.sleep(0.3)
        insert_text(docs, did, h5_si_now - 1, SEC4_2_EGM)

        # §4.3 Geo
        content = get_body(docs, did)
        h5_si_now, _ = find_heading(content, "5.  Searchable Database")
        insert_text(docs, did, h5_si_now, "4.3  Geographic distribution\n")
        content = get_body(docs, did)
        h5_si_now, _ = find_heading(content, "5.  Searchable Database")
        h43_si, h43_ei = find_heading(content, "4.3  Geographic")
        if h43_si: apply_heading(docs, did, h43_si, h43_ei, 2)
        insert_text(docs, did, h5_si_now, SEC4_3_GEO)

        # §4.4 Equity
        content = get_body(docs, did)
        h5_si_now, _ = find_heading(content, "5.  Searchable Database")
        insert_text(docs, did, h5_si_now, "4.4  Equity and inclusion\n")
        content = get_body(docs, did)
        h5_si_now, _ = find_heading(content, "5.  Searchable Database")
        h44_si, h44_ei = find_heading(content, "4.4  Equity")
        if h44_si: apply_heading(docs, did, h44_si, h44_ei, 2)
        insert_text(docs, did, h5_si_now, SEC4_4_EQUITY)

        # §4.5 Temporal + Saturation
        content = get_body(docs, did)
        h5_si_now, _ = find_heading(content, "5.  Searchable Database")
        insert_text(docs, did, h5_si_now, "4.5  Publication trends and saturation\n")
        content = get_body(docs, did)
        h5_si_now, _ = find_heading(content, "5.  Searchable Database")
        h45_si, h45_ei = find_heading(content, "4.5  Publication")
        if h45_si: apply_heading(docs, did, h45_si, h45_ei, 2)
        insert_text(docs, did, h5_si_now, SEC4_5_TEMPORAL)
        print("  §4 rebuilt with 4.1–4.5")
    else:
        print("  WARN: Could not find §4 or §5 boundaries")

    # §3.4 — update random selection section
    rewrite_section("3.4  Random Selection", "4.  Final results", SEC3_4_UPDATE, "§3.4 Human coding/saturation")

    # §1 Key Findings — replace Executive Summary body
    rewrite_section("1.  Key findings", "2.  Background", SEC1_KEY_FINDINGS, "§1 Key findings")

    # 5. Insert pipeline table
    print("[d5] Inserting pipeline table...")
    time.sleep(2)
    content = get_body(docs, did)
    # Find the caption paragraph after which to insert the table
    tbl_insert_idx = None
    for e in content:
        txt = para_text(e)
        if "Table 1. Search and screening results" in txt:
            tbl_insert_idx = e.get("endIndex")
            break
    if tbl_insert_idx:
        n_rows = len(PIPELINE_ROWS)
        n_cols = len(PIPELINE_ROWS[0])
        docs.documents().batchUpdate(documentId=did, body={"requests": [
            {"insertTable": {"location": {"index": tbl_insert_idx}, "rows": n_rows, "columns": n_cols}}
        ]}).execute()
        time.sleep(2)
        content = get_body(docs, did)
        tbl_elem = next((e for e in content if "table" in e), None)
        if tbl_elem:
            rows = tbl_elem["table"]["tableRows"]
            ins_reqs = []
            for ri, row_data in enumerate(PIPELINE_ROWS):
                for ci, val in enumerate(row_data):
                    cell = rows[ri]["tableCells"][ci]
                    para = cell["content"][0]
                    ins_reqs.append({"insertText": {
                        "location": {"index": para.get("startIndex", 0)}, "text": val,
                    }})
            docs.documents().batchUpdate(documentId=did, body={"requests": ins_reqs}).execute()
            time.sleep(1)
            content = get_body(docs, did)
            tbl_elem = next((e for e in content if "table" in e), None)
            if tbl_elem:
                ts = tbl_elem["startIndex"]
                cr_reqs = []
                for ci in range(n_cols):
                    cr_reqs.append(tbl_cell_bg(ts, 0, ci, GREEN))
                    cr_reqs.append(tbl_cell_bg(ts, n_rows - 1, ci, CHARCOAL))
                docs.documents().batchUpdate(documentId=did, body={"requests": cr_reqs}).execute()
                print("  Table inserted and styled")

    # 6. Insert inline figures at placeholder markers
    print("[d5] Inserting figures...")
    PLACEHOLDER_MAP = {
        "[FIG_PRISMA]":   "PRISMA",
        "[FIG_EGM]":      "EGM",
        "[FIG_GEO_MAP]":  "GEO_MAP",
        "[FIG_GEO_BAR]":  "GEO_BAR",
        "[FIG_EQUITY]":   "EQUITY",
        "[FIG_TEMPORAL]": "TEMPORAL",
        "[FIG_SAT]":      "SAT",
    }
    # Process in reverse document order to avoid index drift
    for attempt in range(2):   # two passes in case first pass misses any
        time.sleep(1)
        content = get_body(docs, did)
        found_any = False
        # Build list of (startIndex, endIndex, marker, label) — sorted bottom-up
        placeholders = []
        for e in content:
            txt = para_text(e).strip()
            for marker, label in PLACEHOLDER_MAP.items():
                if txt == marker:
                    placeholders.append((e.get("startIndex"), e.get("endIndex"), marker, label))
        placeholders.sort(key=lambda x: x[0], reverse=True)  # bottom-up

        for si, ei, marker, label in placeholders:
            url_data = fig_urls.get(label)
            if not url_data or not url_data[0]:
                continue
            url, w, h = url_data
            # Delete placeholder paragraph
            delete_range(docs, did, si, ei)
            time.sleep(0.3)
            # Insert image at same position
            insert_image(docs, did, si, url, w, h)
            print(f"  {label} inserted")
            found_any = True

        if not found_any:
            break

    # 7. Update deliverables tracker
    tracker = ROOT / "deliverables" / "deliverables_tracker.md"
    if tracker.exists():
        txt = tracker.read_text()
        txt = re.sub(
            r"(\| \*\*D5\*\* \|[^|]+\|[^|]+\|[^|]+\|) [^\|]+ (\|)",
            f"\\1 [v01 (GDoc)](https://docs.google.com/document/d/{did}/edit) \\2",
            txt,
        )
        tracker.write_text(txt)

    print(f"\n[d5] Done!")
    print(f"  URL: https://docs.google.com/document/d/{did}/edit")
    print(f"  ID:  {did}")
    return did


if __name__ == "__main__":
    run()
