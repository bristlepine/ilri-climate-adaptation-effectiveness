"""
publish_d5_gdoc_v2.py

Definitive D5 Google Doc builder.

1. Copies D4 v02 → fresh D5 GDoc (preserves watermark, fonts, header/footer)
2. Clears body content
3. Uploads human-coded PNGs to Google Drive (makes them public)
4. Rebuilds body with new structure:
     Key Findings → Evidence Gap Map → Geographic → Equity → Temporal →
     Methods (PRISMA, Databases table, Convergence) → Conclusions →
     Searchable Database → References
5. Inserts figures inline via Drive public URLs

Run: python3 deliverables/publish_d5_gdoc_v2.py
"""

from __future__ import annotations
import re
import time
from pathlib import Path
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

# ── paths ──────────────────────────────────────────────────────────────────────
HERE   = Path(__file__).resolve().parent
ROOT   = HERE.parent
STEP16 = ROOT / "scripts" / "outputs" / "step16"
HUMAN  = STEP16 / "interactive" / "human"

CREDS_DIR  = HERE / ".credentials"
TOKEN_FILE = CREDS_DIR / "token.json"
SCOPES     = ["https://www.googleapis.com/auth/documents",
              "https://www.googleapis.com/auth/drive"]

DELIVERABLES_FOLDER = "1alv6Ic7vtKZc_xo6AdlKP1exshMh-fR6"
D4_V02_ID           = "14Hejjfr63HdxH21bqBxLZe3AsXxmLBrXYqJJk7ZbBxU"

# ── figure definitions (label → local path, width_pt, height_pt) ──────────────
FIGS = {
    "EGM":        (HUMAN  / "evidence_gap_map.png",    450, 290),
    "GEO_MAP":    (HUMAN  / "geographic_map.png",       450, 240),
    "GEO_BAR":    (HUMAN  / "geographic_bar.png",       400, 220),
    "EQUITY":     (HUMAN  / "equity.png",               400, 210),
    "TEMPORAL":   (HUMAN  / "temporal_trends.png",      400, 205),
    "PRISMA":     (HERE   / "prisma_flow_d5.png",       450, 340),
    "SATURATION": (STEP16 / "saturation.png",           450, 260),
}

# ── document content ───────────────────────────────────────────────────────────
# Each block: (style, text)
# Styles: TITLE, HEADING_1, HEADING_2, BODY, BULLET, FIGURE_PLACEHOLDER
# FIGURE_PLACEHOLDER text = the figure label key above (e.g. "EGM")

KEY_FINDINGS = [
    ("A recent but thin evidence base", "96%",
     "96% of studies published since 2015 (median: 2022). Long-term impact evidence is nearly absent."),
    ("Marginalized groups largely invisible", "86%",
     "86% of studies provide no equity disaggregation — by gender, age, ethnicity, or disability."),
    ("Non-crop producers severely underrepresented", "<6%",
     "Fisheries/aquaculture and agroforestry appear in fewer than 6% of studies combined. Evidence skews heavily toward crop farming."),
    ("Cost data almost entirely absent", "~80%",
     "~80% of studies report no cost or efficiency data, making value-for-money assessment very difficult."),
    ("Three countries dominate the evidence base", "28%",
     "Ethiopia, Ghana, and Kenya account for 28% of all human-coded studies. Most of Africa, Asia, and Latin America remain very poorly covered."),
    ("Process outcomes studied; impact outcomes are not", "—",
     "Knowledge, adoption, and decision-making are well documented. Income, wellbeing, and resilience outcomes remain sparse across all producer types."),
]

PIPELINE_TABLE = [
    ["Database", "Records returned", "After deduplication", "Abstract screening included",
     "Full texts retrieved", "Full texts screened", "Included"],
    ["Scopus",                   "17,021", "17,021", "6,218",   "2,644", "2,644",  "—"],
    ["Web of Science",           "15,179",  "4,683", "1,140",     "552",   "552",  "—"],
    ["CAB Abstracts",             "5,723",  "3,229", "1,133",     "260",   "260",  "—"],
    ["Academic Search Premier",   "1,187",    "274",    "66",      "20",    "20",  "—"],
    ["AGRIS",                         "3",      "1",     "1",       "0",    "—",  "—"],
    ["Total",                    "39,113", "25,208", "8,558",   "3,476", "3,464", "2,368"],
]

DOCUMENT = [
    # ── Cover ──────────────────────────────────────────────────────────────────
    ("TITLE",    "Measuring what matters: tracking climate adaptation processes and outcomes "
                 "for smallholder producers in the agriculture sector"),
    ("BODY",     "Deliverable 5: Final Systematic Map  ·  June 2026, v01"),
    ("BODY",     "Jennifer Denno Cissé · Zarrar Khan · Caroline G. Staub\n"
                 "Bristlepine Resilience Consultants"),
    ("BODY",     "Keywords: Climate adaptation · Smallholder producers · Systematic map · "
                 "Evidence synthesis · LMIC"),

    # ── 1. Key Findings ────────────────────────────────────────────────────────
    ("HEADING_1", "1.  Key findings"),
    ("BODY",
     "Across 86 human-coded studies, the evidence base is recent, geographically concentrated, "
     "and skewed toward process outcomes and crop systems. Six headline findings stand out:"),

    ("HEADING_2", "Evidence base: six headlines"),
    # Key findings inserted as bullets below — handled specially in build()

    # ── 2. Evidence Gap Map ────────────────────────────────────────────────────
    ("HEADING_1", "2.  Evidence gap map"),
    ("BODY",
     "The evidence gap map (EGM) is the primary output of this deliverable. It shows the "
     "distribution of human-coded studies (n = 86) across process and outcome domains (y-axis) "
     "and producer types (x-axis). Bubble size is proportional to the number of studies in each "
     "cell. Grey circles indicate evidence gaps — domain-producer combinations with no studies."),
    ("FIGURE_PLACEHOLDER", "EGM"),
    ("BODY",
     "Figure 1. Evidence gap map — human-coded results (n = 86). "
     "Blue = process domains · green = outcome domains · grey = evidence gaps."),

    ("HEADING_2", "Key observations from the evidence gap map"),
    ("BULLET",
     "Process domains dominate. Decision-making & planning, knowledge & awareness, and "
     "uptake & adoption are the most studied domains, especially for crop farmers. These "
     "capture early-stage changes in farmer behaviour, not downstream welfare outcomes."),
    ("BULLET",
     "Outcome domains are nearly absent. Income & assets, wellbeing, risk reduction, and "
     "resilience & adaptive capacity are sparsely covered — and where they appear, almost "
     "exclusively for crop systems."),
    ("BULLET",
     "Non-crop producer types are systematically uncovered. Livestock, fisheries/aquaculture, "
     "agroforestry, and mixed-system producers appear in very few cells. Most cells for these "
     "producer types are grey (evidence gaps) across all domains."),
    ("BULLET",
     "Institutional governance and access to services are absent across all producer types. "
     "These are critical dimensions for systemic adaptation M&E and are not being measured."),

    # ── 3. Geographic Distribution ─────────────────────────────────────────────
    ("HEADING_1", "3.  Geographic distribution"),
    ("BODY",
     "Studies are heavily concentrated in Sub-Saharan Africa and South Asia. Ethiopia, Ghana, "
     "and Kenya alone account for 28% of all human-coded studies. Large evidence gaps persist "
     "across Central America, Southeast Asia, the Sahel, and the Middle East and North Africa."),
    ("FIGURE_PLACEHOLDER", "GEO_MAP"),
    ("BODY",  "Figure 2. Geographic distribution of included studies (human-coded, n = 86)."),
    ("FIGURE_PLACEHOLDER", "GEO_BAR"),
    ("BODY",  "Figure 3. Top countries by study count (human-coded, n = 86)."),

    # ── 4. Equity & Inclusion ──────────────────────────────────────────────────
    ("HEADING_1", "4.  Equity and inclusion"),
    ("BODY",
     "86% of included studies provide no equity disaggregation. Where equity dimensions are "
     "addressed, gender is the most common focus — but still features in fewer than 15% of "
     "studies. Youth, indigenous peoples, people with disabilities, and pastoralist groups are "
     "nearly absent. Adaptation outcomes are rarely equivalent across social groups; the "
     "evidence base does not track who benefits."),
    ("FIGURE_PLACEHOLDER", "EQUITY"),
    ("BODY",  "Figure 4. Equity and inclusion dimensions across included studies (human-coded, n = 86). "
              "Red bar = studies with no marginalized group focus."),

    # ── 5. Publication Trends ──────────────────────────────────────────────────
    ("HEADING_1", "5.  Publication trends"),
    ("BODY",
     "96% of included studies were published since 2015, with a median year of 2022. Publication "
     "volume has grown sharply — but this also means that long-term impact evidence is very thin: "
     "most studies are too recently published to measure multi-year adaptation outcomes."),
    ("FIGURE_PLACEHOLDER", "TEMPORAL"),
    ("BODY",  "Figure 5. Publication trends in included studies (human-coded, n = 86)."),

    # ── 6. Methods ─────────────────────────────────────────────────────────────
    ("HEADING_1", "6.  Methods"),
    ("BODY",
     "This systematic map follows the pre-registered protocol published as Deliverable 3 "
     "(Bristlepine Resilience Consultants, January 2026; Zenodo: 10.5281/zenodo.19811629). "
     "A summary of methods is provided here; full technical details are in the protocol and in "
     "the protocol amendment (D5.7)."),

    ("HEADING_2", "6.1  Record flow — PRISMA diagram"),
    ("BODY",
     "A total of 40,653 records were identified across 29 sources. After deduplication, "
     "26,182 unique records entered title and abstract screening. 8,748 full texts were sought; "
     "3,476 (40%) were retrieved automatically. Full-text screening yielded 2,368 LLM-included "
     "records (exploratory reference). From these, 86 studies were human-coded across five "
     "batches as the primary authoritative output."),
    ("FIGURE_PLACEHOLDER", "PRISMA"),
    ("BODY",  "Figure 6. PRISMA flow diagram. Human-coded track (amber, n = 86) = primary output. "
              "LLM track (n = 2,368) = exploratory reference."),

    ("HEADING_2", "6.2  Databases searched"),
    ("BODY",
     "Five bibliographic databases were searched in January 2026, plus 24 grey literature "
     "sources. Table 1 summarises records by database and screening stage. Full-text screening "
     "was conducted across all databases together (3,476 retrieved; 3,464 screened; 2,368 "
     "included). Coded column = LLM-screened included records combined across all databases."),
    ("TABLE", "PIPELINE"),

    ("HEADING_2", "6.3  Screening and calibration"),
    ("BODY",
     "Title and abstract screening used a validated LLM tool (Ollama/qwen2.5:14b, temperature "
     "0.0), calibrated through six rounds with two independent human reviewers (Caroline Staub, "
     "Jennifer Cisse). Final performance: sensitivity = 0.966–0.970; κ = 0.720–0.721, exceeding "
     "the pre-specified thresholds (sensitivity ≥ 0.95; κ ≥ 0.60)."),

    ("HEADING_2", "6.4  Human coding and information saturation"),
    ("BODY",
     "From the pool of 8,748 abstract-screened records, a random sample was drawn in batches "
     "of 20 papers (fixed integer seeds, 42–46; each batch excludes prior-batch DOIs). Coders "
     "applied the five PCCM inclusion criteria and completed 16 extraction fields. Information "
     "saturation was tracked at each batch across three dimensions: process/outcome domains "
     "(13 canonical values), methodological approaches (5), and producer types (5)."),
    ("BODY",
     "All three dimensions reached saturation by batch FT-R2c (49 papers): zero new canonical "
     "categories were added across the final two batches (37 additional papers). This confirms "
     "the 86-paper human sample is representative without needing to code the full corpus."),
    ("FIGURE_PLACEHOLDER", "SATURATION"),
    ("BODY",  "Figure 7. Information saturation curve. Top: cumulative unique categories as % "
              "of final total by papers coded. Bottom: new categories per batch. "
              "All dimensions plateau by batch FT-R2c (49 papers)."),

    # ── 7. Conclusions ─────────────────────────────────────────────────────────
    ("HEADING_1", "7.  Conclusions"),
    ("BODY",
     "The evidence gap map reveals a systematic mismatch between what adaptation M&E currently "
     "measures and what is needed to understand adaptation effectiveness."),
    ("BULLET",
     "Process outcomes are well-documented, impact outcomes are not. The literature reliably "
     "captures whether farmers adopted a practice or gained knowledge. It almost never measures "
     "whether adaptation actually improved incomes, wellbeing, or resilience."),
    ("BULLET",
     "Non-crop producer types are almost entirely unmeasured. Livestock, fisheries, "
     "agroforestry, and mixed-system producers — who together represent the majority of "
     "smallholder livelihoods in many regions — appear in very few cells of the evidence map. "
     "Most are pure gaps."),
    ("BULLET",
     "Geographic concentration limits generalisability. Three countries account for 28% "
     "of all studies. The evidence base cannot currently support regional conclusions across "
     "most of Sub-Saharan Africa, South and Southeast Asia, or Latin America."),
    ("BULLET",
     "The equity gap is the most critical for policy. 86% of studies do not track who "
     "benefits. This makes it impossible to know whether adaptation interventions are "
     "reaching the most vulnerable groups."),
    ("BODY",
     "Addressing these gaps requires deliberate investment in long-term cohort studies that "
     "track outcome-level change, equity-disaggregated data collection, and research programmes "
     "that explicitly target non-crop livelihood systems. The next phase of this work "
     "(Deliverable 6) will develop a systematic review and meta-analysis protocol targeting "
     "the domains and producer types where the evidence is densest and outcome measurement "
     "is most tractable."),

    # ── 8. Searchable database ─────────────────────────────────────────────────
    ("HEADING_1", "8.  Searchable database and project website"),
    ("BODY",
     "The interactive evidence gap map, searchable database, and full methodology are available "
     "at the project website:"),
    ("BODY",  "https://main.d3a9jahzuu7xai.amplifyapp.com/systematic-map"),
    ("BODY",
     "The website includes all 2,368 LLM-screened records with key metadata, filterable by "
     "country, producer type, domain, and methodology. A toggle switches between the "
     "human-coded results (n = 86, primary) and the LLM corpus (n = 2,368, exploratory). "
     "All data and code are available on GitHub: "
     "github.com/bristlepine/ilri-climate-adaptation-effectiveness"),

    # ── 9. References ──────────────────────────────────────────────────────────
    ("HEADING_1", "9.  References"),
    ("BODY",
     "Bristlepine Resilience Consultants (2026). Deliverable 3: Final Systematic Map Protocol. "
     "Zenodo. https://doi.org/10.5281/zenodo.19811629"),
    ("BODY",
     "Bristlepine Resilience Consultants (2026). Deliverable 4: First Draft Systematic Map "
     "(Preliminary). Zenodo. https://doi.org/10.5281/zenodo.19811622"),
    ("BODY",
     "Collaboration for Environmental Evidence (2018). Guidelines and Standards for Evidence "
     "Synthesis in Environmental Management. Version 5.0."),
    ("BODY",
     "Haddaway, N.R. et al. (2018). ROSES Reporting standards for Systematic Evidence "
     "Syntheses. Environmental Evidence, 7, 7."),
]


# ── Google API helpers ─────────────────────────────────────────────────────────

def creds() -> Credentials:
    c = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
    if c.expired and c.refresh_token:
        c.refresh(Request())
    return c


def copy_doc(drive, src_id: str, name: str, folder: str) -> str:
    r = drive.files().copy(
        fileId=src_id, body={"name": name, "parents": [folder]},
        supportsAllDrives=True).execute()
    return r["id"]


def upload_png(drive, png: Path, folder: str) -> str | None:
    """Upload PNG, make public, return direct URL."""
    if not png.exists():
        print(f"  WARN: {png.name} not found")
        return None
    meta  = {"name": f"d5_{png.name}", "parents": [folder]}
    media = MediaFileUpload(str(png), mimetype="image/png", resumable=False)
    fid   = drive.files().create(body=meta, media_body=media,
                                 fields="id", supportsAllDrives=True).execute()["id"]
    drive.permissions().create(fileId=fid,
                               body={"type": "anyone", "role": "reader"},
                               supportsAllDrives=True).execute()
    # Use the lh3 direct URL — works for insertInlineImage
    url = f"https://lh3.googleusercontent.com/d/{fid}"
    return url


def body_content(docs, doc_id: str):
    return docs.documents().get(documentId=doc_id).execute()["body"]["content"]


def para_text(elem: dict) -> str:
    return "".join(r.get("textRun", {}).get("content", "")
                   for r in elem.get("paragraph", {}).get("elements", []))


def find_para_index(content, search: str) -> int | None:
    for e in content:
        if search in para_text(e):
            return e.get("startIndex")
    return None


def find_para_range(content, search: str) -> tuple[int, int] | tuple[None, None]:
    for e in content:
        if search in para_text(e):
            return e.get("startIndex"), e.get("endIndex")
    return None, None


# ── Build document ─────────────────────────────────────────────────────────────

GREEN    = {"red": 33/255,  "green": 71/255,  "blue": 46/255}
CHARCOAL = {"red": 60/255,  "green": 60/255,  "blue": 60/255}
WHITE    = {"red": 1.0,     "green": 1.0,     "blue": 1.0}


def _tbl_cell_bg(table_start: int, row: int, col: int, color: dict) -> dict:
    return {
        "updateTableCellStyle": {
            "tableRange": {
                "tableCellLocation": {
                    "tableStartLocation": {"index": table_start},
                    "rowIndex": row, "columnIndex": col,
                },
                "rowSpan": 1, "columnSpan": 1,
            },
            "tableCellStyle": {"backgroundColor": {"color": {"rgbColor": color}}},
            "fields": "backgroundColor",
        }
    }


def run():
    c = creds()
    drive = build("drive", "v3", credentials=c)
    docs  = build("docs",  "v1", credentials=c)

    # 1. Upload figures to Drive
    print("[d5] Uploading figures to Drive...")
    fig_urls: dict[str, str | None] = {}
    for label, (png, w, h) in FIGS.items():
        url = upload_png(drive, png, DELIVERABLES_FOLDER)
        fig_urls[label] = url
        print(f"  {label}: {'ok' if url else 'MISSING'}")
        time.sleep(0.4)

    # 2. Copy D4 → fresh D5 GDoc
    print("[d5] Copying D4 v02 → D5 v01...")
    doc_id = copy_doc(drive, D4_V02_ID,
                      "Deliverable 5_Bristlepine_Final Systematic Map_v01",
                      DELIVERABLES_FOLDER)
    print(f"[d5] Created: {doc_id}")
    print(f"[d5] URL: https://docs.google.com/document/d/{doc_id}/edit")
    time.sleep(3)

    # 3. Delete all existing body content
    print("[d5] Clearing body...")
    content = body_content(docs, doc_id)
    start   = content[0].get("startIndex", 1)
    end     = content[-1].get("endIndex", 2) - 1
    if end > start:
        docs.documents().batchUpdate(
            documentId=doc_id,
            body={"requests": [
                {"deleteContentRange": {"range": {"startIndex": start, "endIndex": end}}}
            ]}
        ).execute()
        time.sleep(2)

    # 4. Insert text content (with figure placeholders)
    print("[d5] Inserting content...")

    def _cur_end() -> int:
        ct = body_content(docs, doc_id)
        return ct[-1].get("endIndex", 2) - 1

    def _append(text: str) -> None:
        idx = _cur_end()
        docs.documents().batchUpdate(
            documentId=doc_id,
            body={"requests": [{"insertText": {"location": {"index": idx}, "text": text}}]}
        ).execute()

    lines: list[tuple[str, str]] = []  # (style, text) for style pass

    for block in DOCUMENT:
        style, text = block[0], block[1]

        if style == "TABLE":
            # Insert pipeline table marker — actual table inserted later
            _append("[TABLE_PIPELINE]\n")
            lines.append(("NORMAL_TEXT", "[TABLE_PIPELINE]"))
            time.sleep(0.3)
            continue

        if style == "FIGURE_PLACEHOLDER":
            marker = f"[FIG_{text}]"
            _append(marker + "\n")
            lines.append(("NORMAL_TEXT", marker))
            time.sleep(0.3)
            continue

        if style == "HEADING_1" and "Key findings" in text:
            _append(text + "\n")
            lines.append(("HEADING_1", text))
            # Insert key findings as bullets
            _append("Across 86 human-coded studies, the evidence base is recent, "
                    "geographically concentrated, and skewed toward process outcomes "
                    "and crop systems. Six headline findings stand out:\n")
            lines.append(("NORMAL_TEXT",
                           "Across 86 human-coded studies, the evidence base is recent,"))
            for label, stat, desc in KEY_FINDINGS:
                bullet = f"{label} ({stat}): {desc}"
                _append("• " + bullet + "\n")
                lines.append(("NORMAL_TEXT", "• " + bullet))
            time.sleep(0.5)
            continue

        _append(text + "\n")
        lines.append((style, text[:80]))  # store first 80 chars for matching
        time.sleep(0.3)

    # 5. Apply paragraph styles
    print("[d5] Applying heading styles...")
    time.sleep(2)
    content = body_content(docs, doc_id)
    style_reqs = []
    heading_styles = {s: t for s, t in lines if s in ("TITLE", "HEADING_1", "HEADING_2")}

    for elem in content:
        txt = para_text(elem).strip()
        if not txt:
            continue
        for (style, match_text) in [(s, t) for s, t in lines
                                    if s in ("TITLE", "HEADING_1", "HEADING_2")]:
            if txt.startswith(match_text.strip()[:60]):
                si = elem.get("startIndex", 0)
                ei = elem.get("endIndex", si + len(txt) + 1)
                named = {"TITLE": "TITLE",
                         "HEADING_1": "HEADING_1",
                         "HEADING_2": "HEADING_2"}[style]
                style_reqs.append({
                    "updateParagraphStyle": {
                        "range": {"startIndex": si, "endIndex": ei},
                        "paragraphStyle": {"namedStyleType": named},
                        "fields": "namedStyleType",
                    }
                })
                break

    for i in range(0, len(style_reqs), 40):
        docs.documents().batchUpdate(
            documentId=doc_id,
            body={"requests": style_reqs[i:i+40]}
        ).execute()
        time.sleep(0.5)
    print(f"[d5]   {len(style_reqs)} styles applied")

    # 6. Insert pipeline table
    print("[d5] Inserting pipeline table...")
    time.sleep(1)
    content = body_content(docs, doc_id)
    tbl_si, tbl_ei = find_para_range(content, "[TABLE_PIPELINE]")
    if tbl_si is not None:
        n_rows = len(PIPELINE_TABLE)
        n_cols = len(PIPELINE_TABLE[0])
        reqs = [
            {"deleteContentRange": {"range": {"startIndex": tbl_si, "endIndex": tbl_ei}}},
            {"insertTable": {"location": {"index": tbl_si}, "rows": n_rows, "columns": n_cols}},
        ]
        docs.documents().batchUpdate(documentId=doc_id, body={"requests": reqs}).execute()
        time.sleep(2)

        content = body_content(docs, doc_id)
        tbl_elem = next((e for e in content if "table" in e), None)
        if tbl_elem:
            rows = tbl_elem["table"]["tableRows"]
            insert_reqs = []
            for r_idx, row_data in enumerate(PIPELINE_TABLE):
                for c_idx, val in enumerate(row_data):
                    cell = rows[r_idx]["tableCells"][c_idx]
                    para = cell["content"][0]
                    insert_reqs.append({"insertText": {
                        "location": {"index": para.get("startIndex", 0)},
                        "text": val,
                    }})
            docs.documents().batchUpdate(documentId=doc_id,
                                         body={"requests": insert_reqs}).execute()
            time.sleep(1)

            # Style header + total rows
            content = body_content(docs, doc_id)
            tbl_elem = next((e for e in content if "table" in e), None)
            if tbl_elem:
                ts = tbl_elem["startIndex"]
                rows = tbl_elem["table"]["tableRows"]
                color_reqs = []
                for c in range(n_cols):
                    color_reqs.append(_tbl_cell_bg(ts, 0, c, GREEN))
                    color_reqs.append(_tbl_cell_bg(ts, n_rows - 1, c, CHARCOAL))
                docs.documents().batchUpdate(documentId=doc_id,
                                             body={"requests": color_reqs}).execute()
                print("[d5]   Table styled")

    # 7. Insert figures
    print("[d5] Inserting figures...")
    time.sleep(2)

    for label, (png, w_pt, h_pt) in FIGS.items():
        url = fig_urls.get(label)
        if not url:
            print(f"  SKIP {label} — no URL")
            continue

        content = body_content(docs, doc_id)
        marker  = f"[FIG_{label}]"
        si, ei  = find_para_range(content, marker)
        if si is None:
            print(f"  WARN: placeholder not found for {label}")
            continue

        # Delete placeholder text, insert image at same location
        docs.documents().batchUpdate(
            documentId=doc_id,
            body={"requests": [
                {"deleteContentRange": {"range": {"startIndex": si, "endIndex": ei}}},
            ]}
        ).execute()
        time.sleep(0.5)

        docs.documents().batchUpdate(
            documentId=doc_id,
            body={"requests": [{
                "insertInlineImage": {
                    "location": {"index": si},
                    "uri": url,
                    "objectSize": {
                        "width":  {"magnitude": w_pt, "unit": "PT"},
                        "height": {"magnitude": h_pt, "unit": "PT"},
                    },
                }
            }]}
        ).execute()
        print(f"  {label} inserted")
        time.sleep(1)

    # 8. Update deliverables_tracker.md
    tracker = ROOT / "deliverables" / "deliverables_tracker.md"
    if tracker.exists():
        txt = tracker.read_text()
        txt = re.sub(
            r"(\*\*D5\*\*[^\n]+)\| \[v01 \(GDoc\)\]\([^)]+\)",
            f"\\1| [v01 (GDoc)](https://docs.google.com/document/d/{doc_id}/edit)",
            txt,
        )
        tracker.write_text(txt)

    print(f"\n[d5] Done.")
    print(f"  URL: https://docs.google.com/document/d/{doc_id}/edit")
    print(f"  ID:  {doc_id}")
    return doc_id


if __name__ == "__main__":
    run()
