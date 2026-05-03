#!/usr/bin/env python3
"""
step2b_multidatabase_dedupe.py

Step 2b: Ingest exported records from additional databases, deduplicate against
the Scopus corpus, and output net-new records ready for abstract screening.

Supported import formats:
  - RIS  (.ris, .txt)   — WoS, CAB Abstracts, AGRIS, Academic Search Premier
  - CSV  (.csv)          — any database that exports tabular CSV

Deduplication order (each step catches what the previous misses):
  1. DOI match (normalised, lowercased) — most reliable
  2. Exact title + year match (lowercased, punctuation-stripped)
  3. Fuzzy title match within same year (Jaccard token overlap ≥ 0.85)

Outputs (under outputs/step2b/):
  - step2b_net_new.csv         — net-new records not in Scopus corpus
  - step2b_duplicates.csv      — records matched to Scopus (for audit)
  - step2b_summary.json        — counts by database and dedup method
  - step2b_combined_raw.csv    — all imported records before dedup

Net-new records are formatted to match the step2 schema so they can be
concatenated with the Scopus corpus and fed directly into step12 (screening).

Usage:
  Place exported files in scripts/data/multidatabase/
    wos/     — Web of Science RIS exports
    cab/     — CAB Abstracts RIS exports
    agris/   — AGRIS RIS exports
    asp/     — Academic Search Premier RIS exports

  python step2b_multidatabase_dedupe.py

  Or via run.py with run_step2b = 1
"""

from __future__ import annotations

import hashlib
import json
import pickle
import re
import string
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd


# =============================================================================
# Settings
# =============================================================================

FUZZY_THRESHOLD = 0.85   # Jaccard token overlap for fuzzy title match
MIN_TITLE_TOKENS = 4     # skip fuzzy match for very short titles (too ambiguous)


# =============================================================================
# Normalisation helpers
# =============================================================================

def normalize_doi(doi: Any) -> str:
    if not doi or (isinstance(doi, float)):
        return ""
    s = str(doi).strip()
    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s, flags=re.IGNORECASE)
    return s.strip().rstrip(" .;,)\t").lower()


def normalize_title(title: Any) -> str:
    if not title:
        return ""
    s = str(title).lower()
    s = re.sub(r"[^\w\s]", " ", s)   # strip punctuation
    return " ".join(s.split())        # collapse whitespace


def title_tokens(title: str) -> Set[str]:
    return set(normalize_title(title).split())


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def safe_year(val: Any) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    m = re.search(r"\b(19|20)\d{2}\b", s)
    return m.group(0) if m else ""


# =============================================================================
# RIS parser
# =============================================================================

# Maps RIS tags → normalised field names
_RIS_FIELD_MAP = {
    "TI": "title", "T1": "title", "ST": "title",
    "AU": "author", "A1": "author",
    "PY": "year",   "Y1": "year",
    "DO": "doi",    "DI": "doi",
    "AB": "abstract",
    "JO": "publicationName", "JF": "publicationName",
    "T2": "publicationName",
    "KW": "keywords", "DE": "keywords",
    "UR": "url",
    "N1": "notes",
    "TY": "ref_type",
    "ID": "ref_id",
}

def parse_ris_file(path: Path) -> List[Dict[str, str]]:
    """Parse a RIS file into a list of record dicts."""
    records = []
    current: Dict[str, Any] = {}
    last_tag: Optional[str] = None

    with open(path, encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            line = raw.rstrip("\n\r")
            if len(line) >= 2 and line[1:3] in ("  ", " -") or (len(line) >= 4 and line[2:4] == "  "):
                pass  # fall through
            # Standard RIS: "TY  - value" or "TY- value"
            m = re.match(r"^([A-Z][A-Z0-9])\s*-\s*(.*)", line)
            if m:
                tag, value = m.group(1).strip(), m.group(2).strip()
                last_tag = tag
                if tag == "ER":
                    if current:
                        records.append(current)
                    current = {}
                    last_tag = None
                    continue
                field = _RIS_FIELD_MAP.get(tag, tag.lower())
                if field in current:
                    # Multi-value: append with separator
                    current[field] = current[field] + "; " + value
                else:
                    current[field] = value
            elif line.strip() and last_tag:
                # Continuation line
                field = _RIS_FIELD_MAP.get(last_tag, last_tag.lower())
                if field in current:
                    current[field] = current[field] + " " + line.strip()

    if current:
        records.append(current)
    return records


# =============================================================================
# CSV parser (generic — tries common column name variants)
# =============================================================================

_CSV_COL_MAP = {
    "title":           ["title", "article title", "document title", "article name"],
    "author":          ["author", "authors", "au"],
    "year":            ["year", "publication year", "py", "pub year"],
    "doi":             ["doi", "digital object identifier", "di"],
    "abstract":        ["abstract", "ab", "author abstract"],
    "publicationName": ["source title", "journal", "publication", "source", "journal title"],
    "keywords":        ["keywords", "author keywords", "de", "id"],
}

def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_cols:
            return lower_cols[cand.lower()]
    return None

def parse_csv_file(path: Path) -> List[Dict[str, str]]:
    df = pd.read_csv(path, dtype=str, encoding_errors="ignore").fillna("")
    records = []
    col_map = {field: _find_col(df, cands) for field, cands in _CSV_COL_MAP.items()}
    for _, row in df.iterrows():
        rec = {}
        for field, col in col_map.items():
            if col:
                rec[field] = str(row[col]).strip()
        records.append(rec)
    return records


# =============================================================================
# Load all imports from data/multidatabase/
# =============================================================================

DATABASE_DIRS = {
    "wos":     "Web of Science",
    "cab":     "CAB Abstracts",
    "agris":   "AGRIS",
    "asp":     "Academic Search Premier",
    "econlit": "EconLit",
    "proq":    "ProQuest",
}


# =============================================================================
# File-level parse cache
# =============================================================================

def _file_hash(path: Path) -> str:
    """MD5 of file contents — used as cache key."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def _cache_dir(out_root: Path) -> Path:
    d = out_root / "step2b" / ".parse_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _load_cached(cache_dir: Path, file_hash: str) -> Optional[List[Dict]]:
    p = cache_dir / f"{file_hash}.pkl"
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None

def _save_cached(cache_dir: Path, file_hash: str, records: List[Dict]) -> None:
    p = cache_dir / f"{file_hash}.pkl"
    with open(p, "wb") as f:
        pickle.dump(records, f)


def load_all_imports(data_root: Path, out_root: Optional[Path] = None) -> Tuple[pd.DataFrame, List[Dict], List[Dict]]:
    """
    Returns: (standard_records_df, google_scholar_records, ddg_pdfs_manifest)

    - standard_records_df: deduplicated against Scopus
    - google_scholar_records: flagged for abstract retrieval (separate step4b)
    - ddg_pdfs_manifest: grey literature PDFs for manual screening
    """
    multi_root = data_root / "multidatabase"
    all_records: List[Dict] = []
    google_scholar_recs: List[Dict] = []
    ddg_pdfs: List[Dict] = []

    cdir = _cache_dir(out_root) if out_root else None

    # Load standard RIS databases (WoS, CAB, AGRIS, ASP, EconLit, ProQuest)
    for subdir, db_label in DATABASE_DIRS.items():
        db_dir = multi_root / subdir
        if not db_dir.exists():
            print(f"[step2b] {db_label}: folder not found ({db_dir}) — skipping")
            continue
        files = list(db_dir.glob("*.ris")) + list(db_dir.glob("*.txt")) + list(db_dir.glob("*.csv"))
        # Exclude search string txt files
        files = [f for f in files if not f.name.startswith("search_string")]
        if not files:
            print(f"[step2b] {db_label}: no files found in {db_dir} — skipping")
            continue
        n_before = len(all_records)
        n_cached = 0
        for f in files:
            fhash = _file_hash(f) if cdir else None
            recs = _load_cached(cdir, fhash) if (cdir and fhash) else None
            if recs is not None:
                n_cached += 1
            else:
                if f.suffix.lower() == ".csv":
                    recs = parse_csv_file(f)
                else:
                    recs = parse_ris_file(f)
                if cdir and fhash:
                    _save_cached(cdir, fhash, recs)
            for r in recs:
                r["source_db"] = db_label
                r["source_file"] = f.name
            all_records.extend(recs)
        n_added = len(all_records) - n_before
        cache_note = f" ({n_cached}/{len(files)} from cache)" if cdir else ""
        print(f"[step2b] {db_label}: {n_added:,} records from {len(files)} file(s){cache_note}")

    # Load Google Scholar CSV (if present) — flagged for abstract retrieval
    gsch_dir = multi_root / "gsch"
    if gsch_dir.exists():
        gsch_files = list(gsch_dir.glob("*.csv"))
        if gsch_files:
            for f in gsch_files:
                recs = parse_csv_file(f)
                for r in recs:
                    r["source_db"] = "Google Scholar"
                    r["source_file"] = f.name
                    r["needs_abstract_retrieval"] = "yes"  # flag for step4b
                google_scholar_recs.extend(recs)
            print(f"[step2b] Google Scholar: {len(google_scholar_recs):,} records (flagged for abstract retrieval)")

    # Load DDG grey literature PDFs — create manifest for manual screening
    ddg_dir = multi_root / "ddg"
    if ddg_dir.exists():
        pdf_files = list(ddg_dir.glob("*.pdf"))
        if pdf_files:
            for pdf_path in pdf_files:
                ddg_pdfs.append({
                    "pdf_filename": pdf_path.name,
                    "pdf_path": str(pdf_path),
                    "source_db": "Grey Literature (DDG)",
                    "status": "pending_manual_review",
                    "note": "Requires manual title/abstract extraction and full-text review",
                })
            print(f"[step2b] Grey literature (DDG): {len(ddg_pdfs):,} PDFs (flagged for manual review)")

    if not all_records:
        print("[step2b] No RIS/CSV records found. Place exports in scripts/data/multidatabase/<db>/")

    df = pd.DataFrame(all_records) if all_records else pd.DataFrame()
    if not df.empty:
        for col in ["title", "doi", "year", "abstract", "publicationName", "author", "keywords", "source_db", "source_file"]:
            if col not in df.columns:
                df[col] = ""
        df = df.fillna("")

    return df, google_scholar_recs, ddg_pdfs


# =============================================================================
# Build deduplication index from Scopus corpus
# =============================================================================

def build_scopus_index(scopus_csv: Path) -> Tuple[Set[str], Dict[str, str], Dict[Tuple[str, str], str]]:
    """
    Returns:
      doi_set          — normalised DOIs in Scopus
      title_year_map   — (norm_title, year) → record_key
      title_token_map  — list of (token_set, norm_title, year) for fuzzy matching
    """
    df = pd.read_csv(scopus_csv, dtype=str).fillna("")
    doi_set: Set[str] = set()
    title_year_map: Dict[Tuple[str, str], str] = {}
    title_token_list: List[Tuple[Set[str], str, str]] = []

    for _, row in df.iterrows():
        doi = normalize_doi(row.get("doi", ""))
        if doi:
            doi_set.add(doi)
        nt = normalize_title(row.get("title", ""))
        yr = safe_year(row.get("coverDate", "") or row.get("year", ""))
        rk = str(row.get("record_key", ""))
        if nt:
            title_year_map[(nt, yr)] = rk
            toks = title_tokens(nt)
            if len(toks) >= MIN_TITLE_TOKENS:
                title_token_list.append((toks, nt, yr))

    return doi_set, title_year_map, title_token_list


# =============================================================================
# Deduplicate
# =============================================================================

def deduplicate(
    imports: pd.DataFrame,
    doi_set: Set[str],
    title_year_map: Dict[Tuple[str, str], str],
    title_token_list: List[Tuple[Set[str], str, str]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (net_new_df, duplicates_df)."""
    net_new = []
    duplicates = []

    for _, row in imports.iterrows():
        doi = normalize_doi(row.get("doi", ""))
        nt  = normalize_title(row.get("title", ""))
        yr  = safe_year(row.get("year", ""))

        match_method = None

        # 1. DOI match
        if doi and doi in doi_set:
            match_method = "doi"

        # 2. Exact title + year
        if not match_method and nt and (nt, yr) in title_year_map:
            match_method = "title_year_exact"

        # 3. Fuzzy title match within year
        if not match_method and nt:
            toks = title_tokens(nt)
            if len(toks) >= MIN_TITLE_TOKENS:
                for s_toks, s_title, s_yr in title_token_list:
                    if s_yr and yr and s_yr != yr:
                        continue   # only compare within same year
                    j = jaccard(toks, s_toks)
                    if j >= FUZZY_THRESHOLD:
                        match_method = f"title_fuzzy(j={j:.2f})"
                        break

        rec = row.to_dict()
        rec["dedup_match_method"] = match_method or ""

        if match_method:
            duplicates.append(rec)
        else:
            net_new.append(rec)

    return pd.DataFrame(net_new), pd.DataFrame(duplicates)


# =============================================================================
# Format net-new records to match step2 schema
# =============================================================================

def format_for_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds columns expected by step12 (screening) so net-new records can be
    appended to the Scopus corpus and screened in the same pipeline run.
    """
    out = pd.DataFrame()
    out["record_key"]       = df.apply(
        lambda r: f"multdb:{r.get('source_db','?')[:3].lower()}:{normalize_doi(r.get('doi','')) or normalize_title(r.get('title',''))[:60]}",
        axis=1
    )
    out["title"]            = df.get("title", "")
    out["coverDate"]        = df.get("year", "")
    out["publicationName"]  = df.get("publicationName", "")
    out["doi"]              = df.get("doi", "").apply(normalize_doi)
    out["eid"]              = ""
    out["scopus_id"]        = ""
    out["abstract"]         = df.get("abstract", "")
    out["author"]           = df.get("author", "")
    out["keywords"]         = df.get("keywords", "")
    out["source_db"]        = df.get("source_db", "")
    out["source_file"]      = df.get("source_file", "")
    out["citation_raw"]     = ""
    out["citedby_count"]    = ""
    out["prism_url"]        = df.get("url", "")
    return out


# =============================================================================
# Main
# =============================================================================

def run(config: dict) -> dict:
    c = config or {}
    root      = Path(__file__).parent
    data_root = root / "data"
    out_root  = Path(str(c.get("out_dir", "")) or root / "outputs")
    base      = out_root / "step2b"
    base.mkdir(parents=True, exist_ok=True)

    scopus_csv = out_root / "step2" / "step2_total_records.csv"
    if not scopus_csv.exists():
        raise FileNotFoundError(f"Scopus corpus not found: {scopus_csv} — run step2 first.")

    print(f"[step2b] Loading Scopus corpus: {scopus_csv}")
    doi_set, title_year_map, title_token_list = build_scopus_index(scopus_csv)
    scopus_df = pd.read_csv(scopus_csv, dtype=str)
    print(f"[step2b] Scopus corpus: {len(scopus_df):,} records | {len(doi_set):,} with DOIs")

    print(f"[step2b] Loading multi-database imports...")
    imports, google_scholar_recs, ddg_pdfs = load_all_imports(data_root, out_root)
    if imports.empty and not google_scholar_recs and not ddg_pdfs:
        return {"error": "no imports found"}

    raw_csv = base / "step2b_combined_raw.csv"
    if not imports.empty:
        imports.to_csv(raw_csv, index=False)
        print(f"[step2b] Total RIS/CSV imported: {len(imports):,} records")
    else:
        print(f"[step2b] No RIS/CSV records imported")

    net_new = imports
    net_new_formatted = imports
    duplicates = pd.DataFrame()
    dedup_methods = {}

    if not imports.empty:
        print(f"[step2b] Deduplicating...")
        t0 = time.time()
        net_new, duplicates = deduplicate(imports, doi_set, title_year_map, title_token_list)
        elapsed = time.time() - t0

        # Format net-new for pipeline
        net_new_formatted = format_for_pipeline(net_new)
        dedup_methods = duplicates.get("dedup_match_method", pd.Series(dtype=str)).value_counts(dropna=False).to_dict() if not duplicates.empty else {}
    else:
        print(f"[step2b] No RIS/CSV records to deduplicate")

    # Write outputs
    net_new_csv  = base / "step2b_net_new.csv"
    dupes_csv    = base / "step2b_duplicates.csv"
    gsch_csv     = base / "step2b_google_scholar_pending.csv"
    ddg_csv      = base / "step2b_ddg_grey_literature.csv"
    summary_json = base / "step2b_summary.json"

    net_new_formatted.to_csv(net_new_csv, index=False)
    if not duplicates.empty:
        duplicates.to_csv(dupes_csv, index=False)
    if google_scholar_recs:
        df_gsch = pd.DataFrame(google_scholar_recs)
        df_gsch.to_csv(gsch_csv, index=False)
    if ddg_pdfs:
        df_ddg = pd.DataFrame(ddg_pdfs)
        df_ddg.to_csv(ddg_csv, index=False)

    # Count by database
    db_counts = imports.get("source_db", pd.Series(dtype=str)).value_counts(dropna=False).to_dict() if not imports.empty else {}
    net_new_by_db = net_new.get("source_db", pd.Series(dtype=str)).value_counts(dropna=False).to_dict() if not net_new.empty else {}

    summary = {
        "scopus_corpus_size":         int(len(scopus_df)),
        "total_ris_csv_imported":     int(len(imports)),
        "total_ris_csv_duplicates":   int(len(duplicates)),
        "total_ris_csv_net_new":      int(len(net_new)),
        "dedup_rate_pct":             round(100 * len(duplicates) / len(imports), 1) if len(imports) else 0,
        "google_scholar_records":     int(len(google_scholar_recs)),
        "ddg_grey_literature_pdfs":   int(len(ddg_pdfs)),
        "by_database_imported":       {str(k): int(v) for k, v in db_counts.items()},
        "by_database_net_new":        {str(k): int(v) for k, v in net_new_by_db.items()},
        "dedup_match_methods":        {str(k): int(v) for k, v in dedup_methods.items()},
        "net_new_csv":                str(net_new_csv),
        "duplicates_csv":             str(dupes_csv),
        "google_scholar_csv":         str(gsch_csv),
        "ddg_grey_literature_csv":    str(ddg_csv),
        "raw_csv":                    str(raw_csv),
        "timestamp_utc":              time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[step2b] ── Results ─────────────────────────────────")
    print(f"[step2b]   RIS/CSV Imported:       {len(imports):,}")
    print(f"[step2b]   Duplicates (Scopus):    {len(duplicates):,} ({summary['dedup_rate_pct']}%)")
    print(f"[step2b]   Net-new (RIS/CSV):      {len(net_new):,}  ← step12")
    print(f"[step2b]   Google Scholar:        {len(google_scholar_recs):,}  ← step4b (abstract retrieval)")
    print(f"[step2b]   Grey Lit (DDG PDFs):   {len(ddg_pdfs):,}  ← manual review")
    print(f"[step2b]   By database: {net_new_by_db}")
    print(f"[step2b] ─────────────────────────────────────────────")
    return summary


if __name__ == "__main__":
    import config as cfg
    run(cfg.config if hasattr(cfg, "config") else {})
