"""
step13 comprehensive cleanup — run ONCE before step14.

Does four things in a single pass:
  1. Fix corrupt paths — 475 records marked retrieved but file_path=doi_10.pdf
     (HTTP 403 downloads that failed silently). → needs_manual
  2. Ingest manual retrievals — scans manual/ and retrieved/ (sid_ files only)
     and updates matching needs_manual records to retrieved.
  3. Validate all paths — final pass: any retrieved record whose file doesn't
     exist on disk gets flipped to needs_manual.
  4. Write updated manifest, summary.json (with iteration cycle), and
     regenerated step13_missing_papers_01.csv.

Run: python scripts/fix_step13_manifest.py
Safe to re-run — idempotent after first run.
"""

import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

OUT_ROOT    = Path("scripts/outputs/step13")
MANIFEST    = OUT_ROOT / "step13_manifest.csv"
SUMMARY     = OUT_ROOT / "step13_summary.json"
MISSING_CSV = OUT_ROOT / "step13_missing_papers_01.csv"
MANUAL_DIR  = OUT_ROOT / "manual"
RETRIEVED_DIR = OUT_ROOT / "retrieved"

# ── helpers ──────────────────────────────────────────────────────────────────

def doi_to_stem(doi: str) -> str:
    return "doi_" + re.sub(r"[^\w\-.]", "_", doi.strip())

def note_append(existing: str, addition: str) -> str:
    existing = existing.strip()
    return (existing + " | " + addition) if existing else addition

# ── load manifest ─────────────────────────────────────────────────────────────
rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
fieldnames = list(rows[0].keys())
total = len(rows)
print(f"[cleanup] Loaded {total:,} manifest rows")

# ── Phase 1: fix corrupt file_path records ────────────────────────────────────
# These are records where step13 wrote status=retrieved but the download silently
# failed (HTTP 403 / bot-blocked) and the file was never saved.
phase1_fixed = 0
for r in rows:
    if r["status"] == "retrieved":
        fp = r.get("file_path", "")
        if fp and not Path(fp).exists():
            r["status"]    = "needs_manual"
            r["file_path"] = ""
            r["note"]      = note_append(
                r.get("note", ""),
                "step13_fix: corrupt file_path — download failed (HTTP 403 / bot-blocked)"
            )
            phase1_fixed += 1

print(f"[phase1] Corrupt paths fixed: {phase1_fixed} (retrieved → needs_manual)")

# ── Phase 2: ingest Jenn's manual retrievals ─────────────────────────────────
# Build lookup: dedupe_key → row  (for needs_manual rows only, post-phase1)
by_doi_stem = {doi_to_stem(r["doi"]): r
               for r in rows
               if r["status"] == "needs_manual" and r.get("doi", "").strip()}

by_scopus_id = {r["scopus_id"]: r
                for r in rows
                if r["status"] == "needs_manual" and r.get("scopus_id", "").strip()}

phase2_doi  = 0
phase2_sid  = 0
phase2_skip = 0

def ingest_file(f: Path, source_folder: str):
    """Try to match a file to a needs_manual record; update in place."""
    global phase2_doi, phase2_sid, phase2_skip
    name = f.name
    stem = f.stem  # without extension

    # Try DOI stem match
    if stem in by_doi_stem:
        r = by_doi_stem.pop(stem)
        r["status"]    = "retrieved"
        r["file_path"] = str(f.resolve())
        r["source"]    = "manual"
        r["note"]      = note_append(r.get("note", ""), f"step13_ingest: manually retrieved from {source_folder}/")
        phase2_doi += 1
        return

    # Try scopus_id match for sid_ files
    m = re.match(r"sid_(\d+)", stem)
    if m:
        sid = m.group(1)
        if sid in by_scopus_id:
            r = by_scopus_id.pop(sid)
            r["status"]    = "retrieved"
            r["file_path"] = str(f.resolve())
            r["source"]    = "manual"
            r["note"]      = note_append(r.get("note", ""), f"step13_ingest: manually retrieved (sid match) from {source_folder}/")
            phase2_sid += 1
            return

    phase2_skip += 1

# Scan manual/ first (highest signal — Jenn explicitly put these here)
if MANUAL_DIR.exists():
    for f in sorted(MANUAL_DIR.iterdir()):
        if f.is_file():
            ingest_file(f, "manual")

# Scan retrieved/ — only sid_ files (DOI-named files there are duplicates of fulltext/)
if RETRIEVED_DIR.exists():
    for f in sorted(RETRIEVED_DIR.iterdir()):
        if f.is_file() and f.name.startswith("sid_"):
            ingest_file(f, "retrieved")

print(f"[phase2] Ingested by DOI match:     {phase2_doi}")
print(f"[phase2] Ingested by scopus_id:     {phase2_sid}")
print(f"[phase2] Files skipped (no match):  {phase2_skip}")
print(f"[phase2] Total new papers ingested: {phase2_doi + phase2_sid}")

# ── Phase 3: final path validation ────────────────────────────────────────────
# Belt-and-suspenders: any retrieved record whose path still doesn't exist → needs_manual
phase3_fixed = 0
for r in rows:
    if r["status"] == "retrieved":
        fp = r.get("file_path", "")
        if not fp or not Path(fp).exists():
            r["status"]    = "needs_manual"
            r["file_path"] = ""
            r["note"]      = note_append(r.get("note", ""), "step13_fix_p3: path missing after ingestion")
            phase3_fixed += 1

if phase3_fixed:
    print(f"[phase3] Additional path validation fixes: {phase3_fixed}")
else:
    print(f"[phase3] All retrieved paths verified OK")

# ── Phase 4: write manifest ───────────────────────────────────────────────────
with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)
print(f"[phase4] Manifest saved: {MANIFEST}")

# ── Phase 5: update summary.json ─────────────────────────────────────────────
statuses = Counter(r["status"] for r in rows)
summary  = json.loads(SUMMARY.read_text())

# Iteration cycle tracking
prev_cycle = summary.get("manual_ingestion_cycle", 0)
cycle = prev_cycle + 1

summary["status_counts"]          = dict(statuses)
summary["retrieval_rate_pct"]     = round(100 * statuses.get("retrieved", 0) / total, 1)
summary["manual_ingestion_cycle"] = cycle
summary["cleanup_log"] = summary.get("cleanup_log", []) + [{
    "cycle":            cycle,
    "timestamp_utc":    datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "phase1_corrupt_fixed": phase1_fixed,
    "phase2_doi_ingested":  phase2_doi,
    "phase2_sid_ingested":  phase2_sid,
    "phase3_path_errors":   phase3_fixed,
    "retrieved_after":  statuses.get("retrieved", 0),
    "needs_manual_after": statuses.get("needs_manual", 0),
    "retrieval_rate_pct": summary["retrieval_rate_pct"],
}]

SUMMARY.write_text(json.dumps(summary, indent=2))
print(f"[phase5] Summary updated — cycle {cycle}")
print(f"         retrieved:    {statuses.get('retrieved',0):,}")
print(f"         needs_manual: {statuses.get('needs_manual',0):,}")
print(f"         rate:         {summary['retrieval_rate_pct']}%")

# ── Phase 6: regenerate missing papers CSV ───────────────────────────────────
step12_path = Path(summary.get("input_csv", "scripts/outputs/step12/step12_results.csv"))
step12 = {r["dedupe_key"]: r for r in csv.DictReader(open(step12_path, encoding="utf-8"))}

def publisher_category(r, s12=None):
    """Identify publisher from DOI prefix (same logic as step13d_update_manifest.py)."""
    doi = r.get("doi", "").strip()
    if not doi:
        return "No DOI"
    if doi.startswith("10.1080"):
        return "Taylor & Francis"
    if doi.startswith("10.1111") or doi.startswith("10.1002"):
        return "Wiley"
    if doi.startswith("10.1177"):
        return "SAGE"
    if doi.startswith("10.3390"):
        return "MDPI"
    if doi.startswith("10.1007") or doi.startswith("10.1023"):
        return "Springer"
    if doi.startswith("10.1016"):
        return "Elsevier"
    if doi.startswith("10.1093"):
        return "Oxford"
    if doi.startswith("10.1017"):
        return "Cambridge"
    return "Other"

out_fields = [
    "dedupe_key", "doi", "scopus_id", "title", "year", "pub",
    "publisher_category", "status", "source", "note", "url",
    "abstract_available"
]

missing_rows = [r for r in rows if r["status"] == "needs_manual"]
out_rows = []
for r in missing_rows:
    s12 = step12.get(r["dedupe_key"], {})
    out_rows.append({
        "dedupe_key":         r["dedupe_key"],
        "doi":                r.get("doi", ""),
        "scopus_id":          r.get("scopus_id", "") or s12.get("scopus_id", ""),
        "title":              r.get("title", "") or s12.get("title", ""),
        "year":               r.get("year", "") or s12.get("year", ""),
        "pub":                r.get("pub", "") or s12.get("publicationName", ""),
        "publisher_category": publisher_category(r, s12),
        "status":             r["status"],
        "source":             r.get("source", ""),
        "note":               r.get("note", ""),
        "url":                r.get("url", "") or s12.get("prism_url", ""),
        "abstract_available": "yes" if s12.get("abstract", "").strip() else "no",
    })

def sort_key(r):
    order = {"Taylor & Francis": 0, "Wiley": 1, "MDPI": 2, "Springer": 3,
             "SAGE": 4, "Elsevier": 5, "Cambridge": 6, "Oxford": 7, "Other": 8, "No DOI": 9}
    return (order.get(r["publisher_category"], 99),
            -(int(r["year"]) if r["year"].isdigit() else 0))

out_rows.sort(key=sort_key)

with open(MISSING_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=out_fields)
    w.writeheader()
    w.writerows(out_rows)

pub_counts = Counter(r["publisher_category"] for r in out_rows)
print(f"\n[phase6] Missing papers ({len(out_rows):,}) by publisher:")
for pub, count in sorted(pub_counts.items(), key=lambda x: -x[1]):
    print(f"  {count:>5}  {pub}")
print(f"\n[phase6] Missing CSV written: {MISSING_CSV}")

print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Step 13 cleanup — cycle {cycle} — COMPLETE
  Total papers:     {total:,}
  Retrieved:        {statuses.get('retrieved',0):,}  ({summary['retrieval_rate_pct']}%)
  Needs manual:     {statuses.get('needs_manual',0):,}
  Phase 1 (corrupt fix):     {phase1_fixed:,} records corrected
  Phase 2 (manual ingest):   {phase2_doi + phase2_sid:,} new papers added
  Phase 3 (path validate):   {phase3_fixed:,} additional fixes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Next steps:
  1. python scripts/step14_screen_fulltext.py   (re-run, mostly cached)
  2. python scripts/step16_generate_roses.py    (update ROSES/frontend)
""")
