#!/usr/bin/env python3
"""
step8_clean_scopus.py

Step 8: Clean + dedupe Step 2 Scopus records and export a robust RIS
for Zotero and EPPI Reviewer.

Inputs:
  - outputs/step2/step2_total_records.csv   (default)

Outputs (default under outputs/step8/):
  - step8_scopus_cleaned.csv
  - step8_scopus_cleaned.ris
  - step8_scopus_cleaned.meta.json

Key features:
  - Safe cleaning: HTML unescape, whitespace normalization
  - DOI canonicalization (strip URL prefixes, trailing punctuation)
  - Year extraction
  - Robust dedupe: DOI first, then title+year fallback
  - Optional Crossref "repair" for missing authors/journal/year/etc (cached)
  - RIS export with repeated AU lines (EPPI/Zotero friendly)

Run:
  python step8_clean_scopus.py --out-dir outputs
  python step8_clean_scopus.py --no-crossref
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import time
import hashlib  # add near imports
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# ----------------------------
# Cleaning helpers
# ----------------------------

def safe_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    s = html.unescape(str(x))
    s = s.replace("\r", " ").replace("\n", " ")
    s = " ".join(s.split()).strip()
    return s


def normalize_doi(doi: str) -> str:
    """
    Canonical DOI core:
      - removes https://doi.org/ prefixes
      - strips spaces and trailing punctuation
      - lowercases
    """
    s = safe_str(doi)
    if not s:
        return ""
    s = s.strip()
    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s, flags=re.IGNORECASE).strip()
    s = s.rstrip(" .;,)\t")
    return s.lower()


def year_from_any(x) -> str:
    """
    Extract 4-digit year from coverDate or year-like strings.
    """
    s = safe_str(x)
    if not s:
        return ""
    m = re.search(r"\b(19|20)\d{2}\b", s)
    return m.group(0) if m else ""


def clean_for_key_title(s: str) -> str:
    """
    Aggressive normalization ONLY for dedupe keys (never for export fields).
    """
    s = safe_str(s).lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def dedupe_key(row: pd.Series) -> str:
    """
    Dedupe key priority:
      1) DOI (best)
      2) title+year fallback
      3) title only (last resort)
    """
    doi = normalize_doi(row.get("doi", ""))
    if doi:
        return f"doi:{doi}"

    t = clean_for_key_title(row.get("title", ""))
    y = year_from_any(row.get("coverDate", "")) or year_from_any(row.get("year", "")) or ""
    if t and y:
        return f"ty:{t}:{y}"
    if t:
        return f"t:{t}"

    # ultimate fallback: use eid/scopus_id if present
    eid = safe_str(row.get("eid", ""))
    sid = safe_str(row.get("scopus_id", ""))
    if eid:
        return f"eid:{eid}"
    if sid:
        return f"sid:{sid}"
    return ""


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Find a column by exact match (case-insensitive) then substring match.
    """
    cand_l = [c.lower() for c in candidates]
    for col in df.columns:
        if col.strip().lower() in cand_l:
            return col
    for col in df.columns:
        cl = col.lower()
        for c in cand_l:
            if len(c) >= 3 and c in cl:
                return col
    return None


# ----------------------------
# Crossref repair (optional)
# ----------------------------

def crossref_fetch(doi_core: str, timeout: float = 8.0) -> dict:
    """
    Fetch Crossref JSON message for a DOI core.
    """
    if not doi_core:
        return {}
    url = f"https://api.crossref.org/works/{doi_core}"
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "step8_clean_scopus/1.0 (mailto:unknown@example.com)"})
    if r.status_code != 200:
        return {}
    return (r.json() or {}).get("message", {}) or {}


def crossref_parse(message: dict) -> dict:
    """
    Extract a few useful fields:
      - authors list ["Family, Given", ...]
      - journal/container title
      - year
      - volume, issue, page
      - issn
      - abstract (often absent; Crossref sometimes has it)
    """
    out = {}

    # authors
    authors = []
    for a in (message.get("author") or []):
        family = safe_str(a.get("family"))
        given = safe_str(a.get("given"))
        if family and given:
            authors.append(f"{family}, {given}")
        elif family:
            authors.append(family)
    out["xref_authors"] = authors

    # journal/container
    container = message.get("container-title") or []
    out["xref_journal"] = safe_str(container[0]) if container else ""

    # year
    year = ""
    issued = ((message.get("issued") or {}).get("date-parts") or [[None]])[0]
    if issued and issued[0]:
        year = str(issued[0])
    out["xref_year"] = year

    # volume/issue/pages
    out["xref_volume"] = safe_str(message.get("volume"))
    out["xref_issue"] = safe_str(message.get("issue"))
    out["xref_page"] = safe_str(message.get("page"))

    # ISSN
    issn = message.get("ISSN") or []
    out["xref_issn"] = safe_str(issn[0]) if issn else ""

    # abstract (sometimes present, may contain JATS tags)
    abs_raw = safe_str(message.get("abstract"))
    if abs_raw:
        abs_raw = re.sub(r"<[^>]+>", " ", abs_raw)  # strip tags if present
        abs_raw = " ".join(abs_raw.split()).strip()
    out["xref_abstract"] = abs_raw

    return out


def crossref_cached_lookup(
    doi_core: str,
    cache_dir: str,
    *,
    sleep_s: float = 0.1,
    timeout: float = 8.0
) -> dict:
    """
    Cached Crossref lookup by DOI.
    """
    os.makedirs(cache_dir, exist_ok=True)
    h = hashlib.sha1(doi_core.encode("utf-8")).hexdigest()[:16]
    path = os.path.join(cache_dir, f"{h}.json")


    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            with open(path, "r", encoding="utf-8") as f:
                msg = json.load(f)
            return crossref_parse(msg)
        except Exception:
            pass

    try:
        msg = crossref_fetch(doi_core, timeout=timeout)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(msg, f, ensure_ascii=False, indent=2)
        if sleep_s > 0:
            time.sleep(sleep_s)
        return crossref_parse(msg)
    except Exception:
        return {}


# ----------------------------
# RIS export
# ----------------------------

def split_pages(page_range: str) -> Tuple[str, str]:
    pr = safe_str(page_range)
    if not pr:
        return "", ""
    if "-" in pr:
        a, b = pr.split("-", 1)
        return a.strip(), b.strip()
    return pr.strip(), ""


def ris_write_record(f, rec: dict) -> None:
    """
    Write one RIS record with CRLF line endings.
    """
    ty = rec.get("TY") or "JOUR"
    f.write(f"TY  - {ty}\r\n")

    for au in rec.get("AU", []):
        au = safe_str(au)
        if au:
            f.write(f"AU  - {au}\r\n")

    if rec.get("TI"):
        f.write(f"TI  - {safe_str(rec['TI'])}\r\n")
    if rec.get("T2"):
        f.write(f"T2  - {safe_str(rec['T2'])}\r\n")
    if rec.get("PY"):
        f.write(f"PY  - {safe_str(rec['PY'])}\r\n")

    if rec.get("VL"):
        f.write(f"VL  - {safe_str(rec['VL'])}\r\n")
    if rec.get("IS"):
        f.write(f"IS  - {safe_str(rec['IS'])}\r\n")
    if rec.get("SP"):
        f.write(f"SP  - {safe_str(rec['SP'])}\r\n")
    if rec.get("EP"):
        f.write(f"EP  - {safe_str(rec['EP'])}\r\n")
    if rec.get("SN"):
        f.write(f"SN  - {safe_str(rec['SN'])}\r\n")

    if rec.get("DO"):
        f.write(f"DO  - {safe_str(rec['DO'])}\r\n")
    if rec.get("UR"):
        f.write(f"UR  - {safe_str(rec['UR'])}\r\n")
    if rec.get("AB"):
        f.write(f"AB  - {safe_str(rec['AB'])}\r\n")

    f.write("ER  - \r\n\r\n")


def df_to_ris(df: pd.DataFrame, out_path: str) -> int:
    n = 0
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        for _, r in df.iterrows():
            title = safe_str(r.get("title"))
            journal = safe_str(r.get("publicationName")) or safe_str(r.get("xref_journal"))
            year = year_from_any(r.get("coverDate")) or safe_str(r.get("xref_year"))
            doi_core = normalize_doi(r.get("doi"))
            url = ""
            if doi_core:
                url = f"https://doi.org/{doi_core}"
            else:
                url = safe_str(r.get("prism_url"))

            # Authors: prefer authors_joined (Crossref), else author_names, else creator
            authors: List[str] = []

            authors_joined = safe_str(r.get("authors_joined"))
            if authors_joined:
                # Stored as "A; B; C"
                authors = [a.strip() for a in authors_joined.split(";") if a.strip()]
            else:
                author_names = safe_str(r.get("author_names"))
                if author_names:
                    # Some sources use ";" separators; others might be comma-separated names already.
                    if ";" in author_names:
                        authors = [a.strip() for a in author_names.split(";") if a.strip()]
                    else:
                        authors = [author_names]
                else:
                    creator = safe_str(r.get("creator"))
                    if creator:
                        authors = [creator]


            # volume/issue/pages/issn/abstract
            vol = safe_str(r.get("volume")) or safe_str(r.get("xref_volume"))
            iss = safe_str(r.get("issue")) or safe_str(r.get("xref_issue"))
            pages = safe_str(r.get("page_range")) or safe_str(r.get("xref_page"))
            sp, ep = split_pages(pages)
            issn = safe_str(r.get("issn")) or safe_str(r.get("xref_issn"))
            abstract = safe_str(r.get("abstract")) or safe_str(r.get("xref_abstract"))

            ty = "JOUR" if journal else "GEN"

            rec = {
                "TY": ty,
                "AU": authors,
                "TI": title,
                "T2": journal,
                "PY": year,
                "VL": vol,
                "IS": iss,
                "SP": sp,
                "EP": ep,
                "SN": issn,
                "DO": doi_core,
                "UR": url,
                "AB": abstract,
            }

            ris_write_record(f, rec)
            n += 1
    return n


# ----------------------------
# Main pipeline
# ----------------------------

def step8_clean_scopus(out_dir: str, step2_csv: str, *, use_crossref: bool, crossref_sleep_s: float, crossref_max: Optional[int]) -> dict:
    step8_dir = os.path.join(out_dir, "step8")
    os.makedirs(step8_dir, exist_ok=True)

    out_csv = os.path.join(step8_dir, "step8_scopus_cleaned.csv")
    out_ris = os.path.join(step8_dir, "step8_scopus_cleaned.ris")
    out_meta = os.path.join(step8_dir, "step8_scopus_cleaned.meta.json")
    cache_dir = os.path.join(step8_dir, "crossref_cache")

    if not os.path.exists(step2_csv) or os.path.getsize(step2_csv) == 0:
        raise SystemExit(f"Missing Step 2 CSV: {step2_csv}")

    df = pd.read_csv(step2_csv, engine="python", on_bad_lines="skip")
    if df.empty:
        raise SystemExit(f"Step 2 CSV is empty: {step2_csv}")

    # Standardize expected columns if present
    # (These are typical from your Step 2 script)
    title_col = find_col(df, ["title", "dc:title"])
    journal_col = find_col(df, ["publicationName", "journal", "source", "prism:publicationName"])
    cover_col = find_col(df, ["coverDate", "date", "year", "prism:coverDate"])
    doi_col = find_col(df, ["doi", "prism:doi", "DOI"])
    prism_url_col = find_col(df, ["prism_url", "prism:url", "url"])
    creator_col = find_col(df, ["creator", "dc:creator", "first_author"])
    author_names_col = find_col(df, ["author_names", "authors", "author"])

    # Create normalized export columns (do NOT destroy original columns)
    df["title"] = df[title_col].apply(safe_str) if title_col else ""
    df["publicationName"] = df[journal_col].apply(safe_str) if journal_col else ""
    df["coverDate"] = df[cover_col].apply(safe_str) if cover_col else ""
    df["doi"] = df[doi_col].apply(lambda x: normalize_doi(x)) if doi_col else ""
    df["prism_url"] = df[prism_url_col].apply(safe_str) if prism_url_col else ""
    df["creator"] = df[creator_col].apply(safe_str) if creator_col else ""
    # author_names is kept as a *string* if available
    df["author_names"] = df[author_names_col].apply(safe_str) if author_names_col else ""

    # Add derived year + dedupe key
    df["year"] = df["coverDate"].apply(year_from_any)
    df["dedupe_key"] = df.apply(dedupe_key, axis=1)

    before = len(df)

    # Drop rows that have no usable key and no title
    df = df[(df["dedupe_key"] != "") | (df["title"] != "")].copy()

    # Dedupe: keep first occurrence
    df = df.drop_duplicates(subset=["dedupe_key"], keep="first").copy()

    after = len(df)

    # Optional Crossref repair for missing metadata
    crossref_repaired = 0
    crossref_attempted = 0

    if use_crossref:
        # Only attempt when DOI exists and authors/journal/year are weak
        mask = (df["doi"] != "") & (
            (df["publicationName"] == "") |
            (df["year"] == "") |
            ((df["author_names"] == "") & (df["creator"] == ""))
        )
        idxs = df[mask].index.tolist()

        if crossref_max is not None:
            idxs = idxs[: max(0, int(crossref_max))]

        it = idxs
        if tqdm is not None:
            it = tqdm(idxs, desc="[step8] Crossref repair", unit="rec")

        for idx in it:
            doi_core = df.at[idx, "doi"]
            if not doi_core:
                continue

            crossref_attempted += 1
            meta = crossref_cached_lookup(doi_core, cache_dir, sleep_s=crossref_sleep_s)
            if not meta:
                continue

            # Store parsed fields (keep as new columns)
            # xref_authors is a list; store as object
            if meta.get("xref_authors"):
                authors_list = meta.get("xref_authors") or []
                df.at[idx, "authors_joined"] = "; ".join([safe_str(a) for a in authors_list if safe_str(a)])
            if meta.get("xref_journal"):
                df.at[idx, "xref_journal"] = meta.get("xref_journal")
            if meta.get("xref_year"):
                df.at[idx, "xref_year"] = meta.get("xref_year")
            if meta.get("xref_volume"):
                df.at[idx, "xref_volume"] = meta.get("xref_volume")
            if meta.get("xref_issue"):
                df.at[idx, "xref_issue"] = meta.get("xref_issue")
            if meta.get("xref_page"):
                df.at[idx, "xref_page"] = meta.get("xref_page")
            if meta.get("xref_issn"):
                df.at[idx, "xref_issn"] = meta.get("xref_issn")
            if meta.get("xref_abstract"):
                df.at[idx, "xref_abstract"] = meta.get("xref_abstract")

            crossref_repaired += 1

        # If Crossref provided year and coverDate/year was empty, we can backfill df["year"] for convenience
        if "xref_year" in df.columns:
            needs_year = df["year"] == ""
            df.loc[needs_year, "year"] = df.loc[needs_year, "xref_year"].apply(year_from_any)

    # Write cleaned CSV (keep useful columns first, then the rest)
    preferred = [
    "dedupe_key", "title", "publicationName", "year", "coverDate", "doi", "prism_url",
    "author_names", "creator", "authors_joined",
    "xref_authors", "xref_journal", "xref_year", ...
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df_out = df[cols].copy()
    df_out.to_csv(out_csv, index=False)

    # Write RIS
    ris_n = df_to_ris(df_out, out_ris)

    meta = {
        "input_step2_csv": step2_csv,
        "output_csv": out_csv,
        "output_ris": out_ris,
        "rows_before_clean": int(before),
        "rows_after_dedupe": int(after),
        "crossref_enabled": bool(use_crossref),
        "crossref_attempted": int(crossref_attempted),
        "crossref_repaired": int(crossref_repaired),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ris_records_written": int(ris_n),
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[step8] Wrote cleaned CSV: {out_csv}")
    print(f"[step8] Wrote RIS:        {out_ris} ({ris_n} records)")
    print(f"[step8] Dedupe:          {before} -> {after}")
    if use_crossref:
        print(f"[step8] Crossref:        attempted={crossref_attempted}, repaired={crossref_repaired}")

    return meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 8: clean + dedupe Step 2 Scopus records and export robust RIS.")
    p.add_argument("--out-dir", default="outputs", help="Pipeline outputs root (default: outputs)")
    p.add_argument("--step2-csv", default=None, help="Path to step2_total_records.csv (default: <out-dir>/step2/...)")
    p.add_argument("--no-crossref", action="store_true", help="Disable Crossref repair")
    p.add_argument("--crossref-sleep-s", type=float, default=0.1, help="Sleep between Crossref calls (default: 0.1s)")
    p.add_argument("--crossref-max", type=int, default=None, help="Max Crossref repairs to attempt (default: no limit)")
    return p.parse_args()


def run(config: dict) -> dict:
    """
    Runner entrypoint expected by scripts/run.py:
    run(config) -> dict
    """
    out_dir = config.get("out_dir", "outputs")

    # allow override, otherwise default to outputs/step2/step2_total_records.csv
    step2_csv = config.get("step2_csv") or os.path.join(out_dir, "step2", "step2_total_records.csv")

    use_crossref = bool(config.get("use_crossref", True))
    crossref_sleep_s = float(config.get("crossref_sleep_s", 0.1))
    crossref_max = config.get("crossref_max", None)
    crossref_max = int(crossref_max) if crossref_max not in (None, "") else None

    return step8_clean_scopus(
        out_dir=out_dir,
        step2_csv=step2_csv,
        use_crossref=use_crossref,
        crossref_sleep_s=crossref_sleep_s,
        crossref_max=crossref_max,
    )


def run_step8(config: dict) -> dict:
    # optional alias if your runner looks for run_step8 specifically
    return run(config)


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir
    step2_csv = args.step2_csv or os.path.join(out_dir, "step2", "step2_total_records.csv")
    meta = step8_clean_scopus(
        out_dir=out_dir,
        step2_csv=step2_csv,
        use_crossref=(not args.no_crossref),
        crossref_sleep_s=float(args.crossref_sleep_s),
        crossref_max=args.crossref_max,
    )
    return 0 if meta else 1


if __name__ == "__main__":
    raise SystemExit(main())
