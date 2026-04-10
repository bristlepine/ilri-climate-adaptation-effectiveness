#!/usr/bin/env python3
"""
get_missing.py
Retrieve missing full texts via Cornell institutional campus access.

Reads:   missing_papers.csv       3,574 papers — 2,616 have DOIs, 958 do not
         .env (optional)          SCOPUS_API_KEY + SCOPUS_INST_TOKEN for Elsevier lookup

Writes:  retrieved/               downloaded PDFs and HTML files
         retrieval_meta.json      run summary (counts + per-record results)

Strategy:
  Records WITH a DOI  → Unpaywall → Semantic Scholar → publisher URLs (campus IP)
  Records WITHOUT DOI → Elsevier API by Scopus ID → Semantic Scholar title search → CORE title search

Resume-safe: files already in retrieved/ are skipped on re-run.
Stop and restart any time — picks up where it left off.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Optional
from urllib.parse import quote, urlparse

import requests

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE         = Path(__file__).parent
IN_CSV       = HERE / "missing_papers.csv"
OUT_DIR      = HERE / "retrieved"
MANUAL_DIR   = HERE / "manually_retrieved"
META_FILE    = HERE / "retrieval_meta.json"
ENV_FILE     = HERE / ".env"

# ── Settings ──────────────────────────────────────────────────────────────────
SLEEP_S         = 1.2    # polite delay between requests — do not reduce
TIMEOUT         = 45
MAX_MB          = 30
UNPAYWALL_EMAIL = "zarrar@bristlep.com"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/pdf,*/*",
}


# ── Load credentials from .env ────────────────────────────────────────────────

def load_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


# ── Filename helpers ──────────────────────────────────────────────────────────

def norm_doi(doi: str) -> str:
    doi = doi.strip()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
    return doi.strip().lower()


def safe_stem(identifier: str) -> str:
    s = re.sub(r"[^\w\-.]", "_", identifier)
    s = re.sub(r"_+", "_", s).strip("_")
    if len(s) > 160:
        h = hashlib.sha1(identifier.encode()).hexdigest()[:8]
        s = s[:150] + "_" + h
    return s


def already_have(identifier: str) -> Optional[Path]:
    """Check retrieved/ and manually_retrieved/ for an existing file."""
    stem = safe_stem(identifier)
    for folder in (OUT_DIR, MANUAL_DIR):
        for ext in (".pdf", ".html", ".htm", ".xml"):
            p = folder / (stem + ext)
            if p.exists() and p.stat().st_size > 2_000:
                return p
    return None


# ── Download helper ───────────────────────────────────────────────────────────

def download(
    session: requests.Session,
    url: str,
    stem: str,
    *,
    extra_headers: Optional[dict] = None,
    raw_bytes: Optional[bytes] = None,
    ext_hint: str = "",
) -> Optional[Path]:
    try:
        if raw_bytes is not None:
            dest = OUT_DIR / (stem + (ext_hint or ".pdf"))
            dest.write_bytes(raw_bytes)
            return dest if dest.stat().st_size > 2_000 else None

        hdrs = dict(HEADERS)
        if extra_headers:
            hdrs.update(extra_headers)

        r = session.get(url, headers=hdrs, timeout=TIMEOUT,
                        stream=True, allow_redirects=True)
        if r.status_code != 200:
            return None

        ct = r.headers.get("Content-Type", "").lower()
        if "pdf" in ct:
            ext = ".pdf"
        elif "html" in ct or "xml" in ct:
            ext = ".html"
        else:
            p = urlparse(r.url).path.lower()
            ext = ".pdf" if p.endswith(".pdf") else (ext_hint or ".html")

        dest = OUT_DIR / (stem + ext)
        total = 0
        max_b = int(MAX_MB * 1024 * 1024)
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(chunk_size=16_384):
                total += len(chunk)
                if total > max_b:
                    dest.unlink(missing_ok=True)
                    return None
                fh.write(chunk)

        if total < 2_000:
            dest.unlink(missing_ok=True)
            return None

        return dest

    except Exception:
        return None


# ── Sources for DOI records ───────────────────────────────────────────────────

def unpaywall(doi: str, session: requests.Session) -> Optional[str]:
    try:
        r = session.get(
            f"https://api.unpaywall.org/v2/{doi}?email={UNPAYWALL_EMAIL}",
            timeout=15,
        )
        if r.status_code != 200:
            return None
        loc = r.json().get("best_oa_location") or {}
        return loc.get("url_for_pdf") or loc.get("url")
    except Exception:
        return None


def semantic_scholar_by_doi(doi: str, session: requests.Session) -> Optional[str]:
    try:
        r = session.get(
            f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
            "?fields=openAccessPdf",
            timeout=15,
        )
        if r.status_code != 200:
            return None
        return (r.json().get("openAccessPdf") or {}).get("url")
    except Exception:
        return None


def publisher_urls(doi: str) -> list[str]:
    """Candidate PDF URLs by DOI prefix — work best on campus IP."""
    prefix = doi.split("/")[0] if "/" in doi else ""
    urls: list[str] = []

    if prefix in ("10.1007", "10.1038", "10.1186", "10.1057"):   # Springer / Nature / BMC
        urls += [f"https://link.springer.com/content/pdf/{doi}.pdf",
                 f"https://link.springer.com/article/{doi}"]

    if prefix in ("10.1080", "10.4324"):                          # Taylor & Francis / Routledge
        urls += [f"https://www.tandfonline.com/doi/pdf/{doi}?download=true",
                 f"https://www.tandfonline.com/doi/full/{doi}"]

    if prefix in ("10.1002", "10.1111"):                          # Wiley / Blackwell
        urls += [f"https://onlinelibrary.wiley.com/doi/pdfdirect/{doi}",
                 f"https://onlinelibrary.wiley.com/doi/pdf/{doi}"]

    if prefix == "10.1108":                                        # Emerald
        urls += [f"https://www.emerald.com/insight/content/doi/{doi}/full/pdf",
                 f"https://www.emerald.com/insight/content/doi/{doi}/full/html"]

    if prefix == "10.1177":                                        # SAGE
        urls += [f"https://journals.sagepub.com/doi/pdf/{doi}",
                 f"https://journals.sagepub.com/doi/full/{doi}"]

    if prefix == "10.3390":                                        # MDPI (open access)
        urls += [f"https://www.mdpi.com/{doi}/pdf",
                 f"https://res.mdpi.com/{doi}/pdf"]

    if prefix == "10.2166":                                        # IWA Publishing
        urls += [f"https://iwaponline.com/{doi}/pdf"]

    if prefix == "10.1175":                                        # AMS
        urls += [f"https://journals.ametsoc.org/downloadpdf/doi/{quote(doi, safe='')}"]

    if prefix == "10.1155":                                        # Hindawi (open access)
        urls += [f"https://downloads.hindawi.com/journals/{doi}/pdf"]

    if prefix == "10.1504":                                        # Inderscience
        urls += [f"https://www.inderscienceonline.com/doi/pdf/{doi}"]

    return urls


# ── Sources for no-DOI records ────────────────────────────────────────────────

def elsevier_by_scopus_id(
    scopus_id: str,
    session: requests.Session,
    *,
    api_key: str,
    inst_token: str,
) -> Optional[tuple[bytes, str]]:
    """Fetch full text from Elsevier API using Scopus ID. Returns (bytes, ext) or None."""
    if not api_key or not scopus_id:
        return None
    hdrs = {
        "X-ELS-APIKey":    api_key,
        "X-ELS-Insttoken": inst_token,
        "Accept":          "application/pdf",
    }
    url = f"https://api.elsevier.com/content/article/scopus_id/{scopus_id}"
    try:
        r = session.get(url, headers=hdrs, timeout=TIMEOUT, stream=True)
        if r.status_code == 200:
            ct = r.headers.get("Content-Type", "").lower()
            ext = ".pdf" if "pdf" in ct else ".xml"
            data = b"".join(r.iter_content(65_536))
            if len(data) > 2_000:
                return data, ext
        # Fallback: request XML
        hdrs["Accept"] = "text/xml"
        r2 = session.get(url, headers=hdrs, timeout=TIMEOUT, stream=True)
        if r2.status_code == 200:
            data = b"".join(r2.iter_content(65_536))
            if len(data) > 2_000:
                return data, ".xml"
    except Exception:
        pass
    return None


def semantic_scholar_by_title(title: str, session: requests.Session) -> Optional[str]:
    if not title:
        return None
    try:
        r = session.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": title[:120], "fields": "openAccessPdf", "limit": 1},
            timeout=15,
        )
        if r.status_code != 200:
            return None
        data = r.json().get("data") or []
        if not data:
            return None
        return (data[0].get("openAccessPdf") or {}).get("url")
    except Exception:
        return None


def core_by_title(title: str, session: requests.Session) -> Optional[str]:
    if not title:
        return None
    try:
        r = session.get(
            "https://api.core.ac.uk/v3/search/works",
            params={"q": f'title:"{title[:100]}"', "limit": 1},
            timeout=15,
        )
        if r.status_code != 200:
            return None
        results = r.json().get("results") or []
        if not results:
            return None
        hit = results[0]
        return hit.get("downloadUrl") or (hit.get("sourceFulltextUrls") or [None])[0]
    except Exception:
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    MANUAL_DIR.mkdir(exist_ok=True)

    env        = load_env(ENV_FILE)
    api_key    = env.get("SCOPUS_API_KEY", "")
    inst_token = env.get("SCOPUS_INST_TOKEN", "")

    if api_key:
        print("Elsevier credentials loaded — Scopus ID lookup enabled for no-DOI records")
    else:
        print("No .env found — Elsevier Scopus ID lookup disabled (DOI records unaffected)")

    with open(IN_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    manual_done = [r for r in rows if (r.get("manually_retrieved") or "").strip()]
    rows        = [r for r in rows if not (r.get("manually_retrieved") or "").strip()]

    has_doi = [r for r in rows if r.get("doi", "").strip()]
    no_doi  = [r for r in rows if not r.get("doi", "").strip()]
    print(f"\nLoaded {len(rows) + len(manual_done)} records total")
    print(f"  {len(manual_done)} already marked manually_retrieved in CSV — skipping")
    print(f"  {len(has_doi)} with DOI, {len(no_doi)} without DOI — attempting these")
    print(f"Output: {OUT_DIR}\n")

    session = requests.Session()
    session.headers.update(HEADERS)

    results: list[dict] = []
    n_retrieved = n_skipped = n_failed = 0
    all_rows = has_doi + no_doi
    total    = len(all_rows)

    for i, row in enumerate(all_rows, 1):
        doi       = norm_doi(row.get("doi", "") or "")
        title     = (row.get("title") or "").strip()
        scopus_id = (row.get("scopus_id") or "").strip()
        key       = row.get("dedupe_key") or doi or scopus_id or f"row{i}"

        print(f"[{i}/{total}] {(title or doi or scopus_id)[:70]}")

        file_id  = f"doi_{doi}" if doi else f"sid_{scopus_id}"
        existing = already_have(file_id)
        if existing:
            print(f"  → already have {existing.name}")
            results.append({"key": key, "status": "already_retrieved", "file": existing.name})
            n_skipped += 1
            continue

        stem  = safe_stem(file_id)
        found = False

        if doi:
            # ── DOI path ──────────────────────────────────────────────────────

            oa_url = unpaywall(doi, session)
            if oa_url:
                path = download(session, oa_url, stem)
                if path:
                    print(f"  ✓ unpaywall → {path.name}")
                    results.append({"key": key, "doi": doi, "status": "retrieved",
                                    "source": "unpaywall", "file": path.name})
                    n_retrieved += 1
                    found = True

            if not found:
                ss_url = semantic_scholar_by_doi(doi, session)
                if ss_url:
                    path = download(session, ss_url, stem)
                    if path:
                        print(f"  ✓ semantic_scholar → {path.name}")
                        results.append({"key": key, "doi": doi, "status": "retrieved",
                                        "source": "semantic_scholar", "file": path.name})
                        n_retrieved += 1
                        found = True

            if not found:
                for pub_url in publisher_urls(doi):
                    path = download(session, pub_url, stem, ext_hint=".pdf")
                    if path:
                        print(f"  ✓ campus → {path.name}")
                        results.append({"key": key, "doi": doi, "status": "retrieved",
                                        "source": "campus_publisher", "file": path.name})
                        n_retrieved += 1
                        found = True
                        break
                    time.sleep(SLEEP_S * 0.4)

        else:
            # ── No-DOI path ───────────────────────────────────────────────────

            if api_key and scopus_id:
                result = elsevier_by_scopus_id(
                    scopus_id, session, api_key=api_key, inst_token=inst_token
                )
                if result:
                    raw, ext = result
                    path = download(session, "", stem, raw_bytes=raw, ext_hint=ext)
                    if path:
                        print(f"  ✓ elsevier_scopus_id → {path.name}")
                        results.append({"key": key, "scopus_id": scopus_id,
                                        "status": "retrieved",
                                        "source": "elsevier_scopus_id", "file": path.name})
                        n_retrieved += 1
                        found = True

            if not found:
                ss_url = semantic_scholar_by_title(title, session)
                if ss_url:
                    path = download(session, ss_url, stem)
                    if path:
                        print(f"  ✓ semantic_scholar_title → {path.name}")
                        results.append({"key": key, "scopus_id": scopus_id,
                                        "status": "retrieved",
                                        "source": "semantic_scholar_title", "file": path.name})
                        n_retrieved += 1
                        found = True

            if not found:
                core_url = core_by_title(title, session)
                if core_url:
                    path = download(session, core_url, stem)
                    if path:
                        print(f"  ✓ core → {path.name}")
                        results.append({"key": key, "scopus_id": scopus_id,
                                        "status": "retrieved",
                                        "source": "core", "file": path.name})
                        n_retrieved += 1
                        found = True

        if not found:
            print("  ✗ not retrieved")
            results.append({"key": key, "doi": doi, "scopus_id": scopus_id,
                            "status": "failed", "file": None})
            n_failed += 1

        time.sleep(SLEEP_S)

    # ── Write meta ────────────────────────────────────────────────────────────
    n_auto   = len([f for f in OUT_DIR.glob("*") if f.is_file()])
    n_manual = len([f for f in MANUAL_DIR.glob("*") if f.is_file() and f.name != ".gitkeep"])
    meta = {
        "total_input":              total + len(manual_done),
        "manually_marked_in_csv":   len(manual_done),
        "with_doi":                 len(has_doi),
        "without_doi":              len(no_doi),
        "retrieved_this_run":       n_retrieved,
        "already_had":              n_skipped,
        "failed":                   n_failed,
        "files_in_retrieved":       n_auto,
        "files_in_manually_retrieved": n_manual,
        "results":                  results,
    }
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    pct = round(n_retrieved / total * 100, 1) if total else 0
    print(f"\n{'─'*60}")
    print(f"Retrieved  : {n_retrieved} ({pct}%)")
    print(f"Already had: {n_skipped}")
    print(f"Manual (CSV): {len(manual_done)}")
    print(f"Failed     : {n_failed}")
    print(f"Auto files : {n_auto}  (in retrieved/)")
    print(f"Manual files: {n_manual}  (in manually_retrieved/)")
    print(f"\nZip and send to Zarrar:")
    print(f"  zip -r retrieved.zip retrieved/ manually_retrieved/")


if __name__ == "__main__":
    main()
