#!/usr/bin/env python3
"""
step13_retrieve_fulltext.py

Step 13: Full-text retrieval for all "Include" records from step12.

For each included record, tries these sources in order:
  1. Unpaywall       — best open-access URL (PDF or HTML), free, no key needed
  2. Elsevier        — full-text PDF via Scopus API key + institutional token
  3. Semantic Scholar — openAccessPdf link
  4. OpenAlex        — open_access.oa_url
  5. Frontiers       — direct PDF for 10.3389 DOIs (frontiersin.org/articles/{doi}/pdf)
  6. CORE.ac.uk      — institutional repository aggregator, strong for dev/ag literature

Downloads available PDFs/HTML to outputs/step13/fulltext/.
Records that cannot be retrieved automatically are flagged as "needs_manual".

Inputs:
  - outputs/step12/step12_results.csv

Outputs (under outputs/step13/):
  - fulltext/           downloaded files (PDF or HTML)
  - step13_manifest.csv full record of retrieval status per paper
  - step13_summary.json counts by source/status
  - step13_manual.csv   records needing manual retrieval (for Cornell ILL/VPN)

Resume-safe: skips records already in manifest cache.

Run:
  python step13_retrieve_fulltext.py
  (or via run.py with run_step13 = 1)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, urlparse

import pandas as pd
import requests
from dotenv import load_dotenv


# =============================================================================
# SETTINGS
# =============================================================================

SLEEP_S = 0.15           # polite delay between API calls
MAX_FILE_MB = 25         # skip downloads larger than this
DOWNLOAD_TIMEOUT = 60    # seconds per download
API_TIMEOUT = 15         # seconds for metadata API calls

PRINT_EVERY = 50


# =============================================================================
# Helpers
# =============================================================================

def safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        try:
            import math
            if math.isnan(x):
                return ""
        except Exception:
            pass
    s = str(x).replace("\r", " ").replace("\n", " ")
    return " ".join(s.split()).strip()


def normalize_doi(doi: Any) -> str:
    s = safe_str(doi)
    if not s:
        return ""
    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s, flags=re.IGNORECASE).strip()
    return s.rstrip(" .;,)\t").lower()


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def safe_filename(key: str, ext: str = ".pdf") -> str:
    """Turn a dedupe_key or DOI into a safe filename."""
    s = re.sub(r"[^\w\-.]", "_", key)
    s = re.sub(r"_+", "_", s).strip("_")
    if len(s) > 180:
        h = hashlib.sha1(key.encode()).hexdigest()[:8]
        s = s[:170] + "_" + h
    return s + ext


def _file_ext_from_content_type(ct: str) -> str:
    ct = (ct or "").lower().split(";")[0].strip()
    if "pdf" in ct:
        return ".pdf"
    if "html" in ct:
        return ".html"
    if "xml" in ct:
        return ".xml"
    return ".bin"


def _file_ext_from_url(url: str) -> str:
    path = urlparse(url).path.lower()
    for ext in (".pdf", ".html", ".htm", ".xml"):
        if path.endswith(ext):
            return ext if ext != ".htm" else ".html"
    return ""


# =============================================================================
# IO paths
# =============================================================================

def step12_csv_path(out_root: Path) -> Path:
    p = out_root / "step12" / "step12_results.csv"
    if p.exists() and p.stat().st_size > 0:
        return p
    raise SystemExit(f"Step 12 results not found at {p}. Run step12 first.")


def step13_dirs(out_root: Path) -> Tuple[Path, Path, Path, Path]:
    base = out_root / "step13"
    fulltext = base / "fulltext"
    fulltext.mkdir(parents=True, exist_ok=True)
    (base / "manual").mkdir(parents=True, exist_ok=True)
    manifest_csv = base / "step13_manifest.csv"
    summary_json = base / "step13_summary.json"
    manual_csv   = base / "step13_manual.csv"
    return base, fulltext, manifest_csv, summary_json


# =============================================================================
# Manifest cache (JSON lines keyed by dedupe_key)
# =============================================================================

def _manifest_cache_path(base: Path) -> Path:
    return base / "step13_manifest_cache.jsonl"


def load_manifest_cache(base: Path) -> Dict[str, dict]:
    path = _manifest_cache_path(base)
    cache: Dict[str, dict] = {}
    if not path.exists():
        return cache
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
                k = safe_str(j.get("dedupe_key"))
                if k:
                    cache[k] = j
            except Exception:
                continue
    return cache


def append_manifest(base: Path, rec: dict) -> None:
    path = _manifest_cache_path(base)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# =============================================================================
# Download helper
# =============================================================================

def _download_file(
    session: requests.Session,
    url: str,
    dest: Path,
    *,
    headers: Optional[Dict] = None,
    max_mb: float = MAX_FILE_MB,
    timeout: int = DOWNLOAD_TIMEOUT,
) -> Tuple[bool, str]:
    """
    Download url to dest. Returns (ok, reason).
    Checks Content-Length before downloading if available.
    """
    try:
        r = session.get(url, headers=headers or {}, timeout=timeout, stream=True)
        if r.status_code != 200:
            return False, f"HTTP {r.status_code}"

        # Check size from headers
        cl = r.headers.get("Content-Length", "")
        if cl and cl.isdigit():
            mb = int(cl) / (1024 * 1024)
            if mb > max_mb:
                return False, f"Too large ({mb:.1f} MB > {max_mb} MB)"

        # Determine extension
        ct = r.headers.get("Content-Type", "")
        ext_from_ct = _file_ext_from_content_type(ct)
        ext_from_url = _file_ext_from_url(url)
        ext = ext_from_ct or ext_from_url or ".pdf"

        # Rename dest if extension differs
        if dest.suffix != ext:
            dest = dest.with_suffix(ext)

        # Stream download with size guard
        total = 0
        max_bytes = int(max_mb * 1024 * 1024)
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    total += len(chunk)
                    if total > max_bytes:
                        fh.close()
                        dest.unlink(missing_ok=True)
                        return False, f"Exceeded {max_mb} MB during download"
                    fh.write(chunk)

        if total < 1000:
            dest.unlink(missing_ok=True)
            return False, f"File too small ({total} bytes), likely error page"

        return True, str(dest)

    except Exception as e:
        dest.unlink(missing_ok=True)
        return False, f"{type(e).__name__}: {e}"


# =============================================================================
# Source 1: Unpaywall
# =============================================================================

def try_unpaywall(
    session: requests.Session,
    doi: str,
    *,
    email: str,
    base_url: str = "https://api.unpaywall.org/v2/",
) -> Tuple[Optional[str], str]:
    """Returns (oa_url, note) — oa_url is None if nothing found."""
    if not doi:
        return None, "no_doi"
    url = f"{base_url.rstrip('/')}/{quote(doi, safe='')}?email={quote(email)}"
    try:
        r = session.get(url, timeout=API_TIMEOUT)
        if r.status_code == 404:
            return None, "unpaywall_not_found"
        if r.status_code != 200:
            return None, f"unpaywall_http_{r.status_code}"
        data = r.json()
        # best_oa_location first
        best = data.get("best_oa_location") or {}
        oa_url = safe_str(best.get("url_for_pdf") or best.get("url"))
        if oa_url:
            return oa_url, "unpaywall"
        # fallback: any oa location with a PDF
        for loc in (data.get("oa_locations") or []):
            u = safe_str(loc.get("url_for_pdf") or loc.get("url"))
            if u:
                return u, "unpaywall_fallback"
        return None, "unpaywall_no_oa"
    except Exception as e:
        return None, f"unpaywall_error:{type(e).__name__}"


# =============================================================================
# Source 2: Elsevier full-text API
# =============================================================================

def try_elsevier(
    session: requests.Session,
    doi: str,
    scopus_id: str,
    *,
    api_key: str,
    inst_token: str,
    article_url: str = "https://api.elsevier.com/content/article/doi/",
) -> Tuple[Optional[str], str, Optional[bytes]]:
    """
    Returns (url_or_none, note, pdf_bytes_or_none).
    Elsevier returns the PDF directly so we return the bytes rather than a URL.
    """
    if not api_key:
        return None, "elsevier_no_key", None

    headers = {"X-ELS-APIKey": api_key, "Accept": "application/pdf"}
    if inst_token:
        headers["X-ELS-Insttoken"] = inst_token

    # Try by DOI first
    if doi:
        url = f"{article_url.rstrip('/')}/{quote(doi, safe='')}"
        try:
            r = session.get(url, headers=headers, timeout=DOWNLOAD_TIMEOUT, stream=True)
            if r.status_code == 200:
                ct = r.headers.get("Content-Type", "")
                if "pdf" in ct.lower():
                    data = b"".join(r.iter_content(65536))
                    if len(data) > 1000:
                        return url, "elsevier_doi", data
            # Try XML if PDF not available
            headers2 = dict(headers)
            headers2["Accept"] = "text/xml"
            r2 = session.get(url, headers=headers2, timeout=DOWNLOAD_TIMEOUT, stream=True)
            if r2.status_code == 200:
                data = b"".join(r2.iter_content(65536))
                if len(data) > 1000:
                    return url, "elsevier_doi_xml", data
        except Exception as e:
            pass  # fall through to scopus_id attempt

    # Try by Scopus ID
    if scopus_id:
        sid_url = f"https://api.elsevier.com/content/article/scopus_id/{scopus_id}"
        try:
            r = session.get(sid_url, headers=headers, timeout=DOWNLOAD_TIMEOUT, stream=True)
            if r.status_code == 200:
                data = b"".join(r.iter_content(65536))
                if len(data) > 1000:
                    return sid_url, "elsevier_scopus_id", data
        except Exception as e:
            pass

    return None, "elsevier_unavailable", None


# =============================================================================
# Source 3: Semantic Scholar
# =============================================================================

def try_semantic_scholar(
    session: requests.Session,
    doi: str,
    title: str,
    *,
    base_url: str = "https://api.semanticscholar.org/graph/v1/paper/",
) -> Tuple[Optional[str], str]:
    if not doi and not title:
        return None, "s2_no_identifier"

    identifier = f"DOI:{doi}" if doi else f"search/{quote(title[:100])}"
    url = f"{base_url.rstrip('/')}/{identifier}?fields=openAccessPdf,externalIds"
    try:
        r = session.get(url, timeout=API_TIMEOUT)
        if r.status_code == 404:
            return None, "s2_not_found"
        if r.status_code == 429:
            return None, "s2_rate_limited"
        if r.status_code != 200:
            return None, f"s2_http_{r.status_code}"
        data = r.json()
        pdf_info = data.get("openAccessPdf") or {}
        pdf_url = safe_str(pdf_info.get("url"))
        if pdf_url:
            return pdf_url, "semantic_scholar"
        return None, "s2_no_oa_pdf"
    except Exception as e:
        return None, f"s2_error:{type(e).__name__}"


# =============================================================================
# Source 4: OpenAlex
# =============================================================================

def try_openalex(
    session: requests.Session,
    doi: str,
    *,
    email: str,
    base_url: str = "https://api.openalex.org/works/",
) -> Tuple[Optional[str], str]:
    if not doi:
        return None, "openalex_no_doi"
    url = f"{base_url.rstrip('/')}/doi:{quote(doi, safe='')}"
    params = {"mailto": email} if email else {}
    try:
        r = session.get(url, params=params, timeout=API_TIMEOUT)
        if r.status_code == 404:
            return None, "openalex_not_found"
        if r.status_code != 200:
            return None, f"openalex_http_{r.status_code}"
        data = r.json()
        oa = data.get("open_access") or {}
        oa_url = safe_str(oa.get("oa_url"))
        if oa_url:
            return oa_url, "openalex"
        # check locations
        for loc in (data.get("locations") or []):
            u = safe_str(loc.get("pdf_url") or loc.get("landing_page_url"))
            if u and "pdf" in u.lower():
                return u, "openalex_location"
        return None, "openalex_no_oa"
    except Exception as e:
        return None, f"openalex_error:{type(e).__name__}"


def try_frontiers(
    session: requests.Session,
    doi: str,
) -> Tuple[Optional[str], str]:
    """
    Frontiers journals (10.3389) serve PDFs directly at a predictable URL.
    No API needed — fully open access, no Cloudflare block on their PDF endpoint.
    """
    if not doi or not doi.startswith("10.3389"):
        return None, "frontiers_not_applicable"
    url = f"https://www.frontiersin.org/articles/{doi}/pdf"
    return url, "frontiers"


def try_core(
    session: requests.Session,
    doi: str,
    title: str,
    *,
    base_url: str = "https://api.core.ac.uk/v3/",
) -> Tuple[Optional[str], str]:
    """
    CORE.ac.uk — aggregates open-access full texts from institutional
    repositories worldwide. Free, no API key required for basic use.
    Particularly good for development/agriculture/CGIAR literature.
    Tries DOI lookup first, then title search fallback.
    """
    # 1. DOI lookup
    if doi:
        try:
            url = f"{base_url.rstrip('/')}/works/doi:{quote(doi, safe='')}"
            r = session.get(url, timeout=API_TIMEOUT)
            if r.status_code == 200:
                data = r.json()
                pdf_url = safe_str(data.get("downloadUrl") or data.get("pdfUrl") or "")
                if pdf_url:
                    return pdf_url, "core_doi"
        except Exception:
            pass

    # 2. Title search fallback
    if title:
        try:
            url = f"{base_url.rstrip('/')}/search/works"
            params = {"q": f'title:"{title[:120]}"', "limit": 1}
            r = session.get(url, params=params, timeout=API_TIMEOUT)
            if r.status_code == 200:
                results = r.json().get("results") or []
                for hit in results:
                    pdf_url = safe_str(hit.get("downloadUrl") or hit.get("pdfUrl") or "")
                    if pdf_url:
                        return pdf_url, "core_title"
        except Exception:
            pass

    return None, "core_not_found"


# =============================================================================
# Main retrieval loop
# =============================================================================

def retrieve_fulltexts(
    df: pd.DataFrame,
    *,
    out_root: Path,
    api_key: str,
    inst_token: str,
    email: str,
    endpoints: dict,
    run_limit: Optional[int] = None,
) -> List[dict]:
    base, fulltext_dir, _, _ = step13_dirs(out_root)
    manual_dir = base / "manual"
    cache = load_manifest_cache(base)
    already_done = set(cache.keys())
    print(f"[step13] Cache: {len(already_done):,} records already retrieved")

    # ---- Register manually downloaded files ----
    # Drop PDFs/HTMLs into outputs/step13/manual/ named by DOI:
    #   doi_10.1016_j.agsy.2021.103149.pdf
    # Step13 will match them to records and register as source=manual.
    manual_files = list(manual_dir.glob("*"))
    if manual_files:
        print(f"[step13] Manual folder: {len(manual_files)} file(s) found — registering...")
        # Build DOI → dedupe_key lookup from input df
        doi_to_dk = {}
        for _, row in df.iterrows():
            d = normalize_doi(row.get("doi", ""))
            dk = safe_str(row.get("dedupe_key", ""))
            if d and dk:
                doi_to_dk[d] = dk

        n_manual_registered = 0
        for mf in manual_files:
            if mf.suffix.lower() not in (".pdf", ".html", ".htm", ".txt"):
                continue
            # Extract DOI from filename: doi_10.1016_j.agsy.2021.103149.pdf
            stem = mf.stem  # e.g. doi_10.1016_j.agsy.2021.103149
            if stem.startswith("doi_"):
                doi_from_name = stem[4:].replace("_", "/", 1).replace("_", ".")
                # re-normalise: first _ is the 10.XXXX separator
                # pattern: doi_10.NNNN_rest → 10.NNNN/rest
                parts = stem[4:].split("_", 1)
                doi_from_name = parts[0] + "/" + parts[1].replace("_", ".") if len(parts) == 2 else stem[4:]
            else:
                doi_from_name = stem.replace("_", "/", 1)

            doi_from_name = normalize_doi(doi_from_name)
            dk = doi_to_dk.get(doi_from_name, "")

            if not dk or dk in already_done:
                continue

            rec = {
                "dedupe_key": dk,
                "doi": doi_from_name,
                "scopus_id": "",
                "title": "",
                "year": "",
                "pub": "",
                "missing_abstract": False,
                "status": "retrieved",
                "source": "manual",
                "file_path": str(mf),
                "file_size_kb": round(mf.stat().st_size / 1024, 1),
                "url": "",
                "note": f"manually downloaded: {mf.name}",
                "timestamp_utc": "",
            }
            append_manifest(base, rec)
            already_done.add(dk)
            n_manual_registered += 1
            print(f"[step13]   Registered manual: {mf.name} → {doi_from_name}")

        print(f"[step13] Manual: {n_manual_registered} new file(s) registered (source=manual)")
    else:
        print(f"[step13] Manual folder: empty (drop PDFs into outputs/step13/manual/ to register)")

    # Filter to Include only
    inc_mask = df["screen_decision"].str.strip().str.lower() == "include"
    inc_df = df[inc_mask].reset_index(drop=True)
    print(f"[step13] Included records: {len(inc_df):,} / {len(df):,} total")

    n_total = len(inc_df)
    n_run = min(int(run_limit), n_total) if run_limit else n_total

    results: List[dict] = list(cache.values())

    t0 = time.time()
    n_got = n_skip = n_fail = n_manual = 0

    with requests.Session() as session:
        session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/pdf,*/*;q=0.9",
            "Accept-Language": "en-US,en;q=0.9",
        })

        for i in range(n_run):
            row = inc_df.iloc[i]
            dk   = safe_str(row.get("dedupe_key", ""))
            doi  = normalize_doi(row.get("doi", ""))
            sid  = safe_str(row.get("scopus_id", ""))
            title = safe_str(row.get("title", ""))
            year  = safe_str(row.get("year", ""))
            pub   = safe_str(row.get("publicationName", ""))

            # Use DOI for filename, fall back to dedupe_key
            fname_base = f"doi_{doi.replace('/', '_')}" if doi else f"key_{dk[:80]}"

            if (i + 1) % PRINT_EVERY == 0 or i == 0 or (i + 1) == n_run:
                elapsed = time.time() - t0
                print(
                    f"[step13] {i+1:,}/{n_run:,} | got={n_got:,} skip={n_skip:,} "
                    f"fail={n_fail:,} manual={n_manual:,} | {elapsed:,.0f}s"
                )

            # Already retrieved
            if dk and dk in already_done:
                n_skip += 1
                continue

            screen_reasons = safe_str(row.get("screen_reasons", ""))
            is_missing_abstract = screen_reasons.startswith("Missing abstract")

            rec: Dict[str, Any] = {
                "dedupe_key": dk,
                "doi": doi,
                "scopus_id": sid,
                "title": title,
                "year": year,
                "pub": pub,
                "missing_abstract": is_missing_abstract,
                "timestamp_utc": _now_utc(),
                "status": "pending",
                "source": "",
                "file_path": "",
                "file_size_kb": 0,
                "url": "",
                "note": "",
            }

            retrieved = False

            # ---- Source 1: Unpaywall ----
            if doi and not retrieved:
                oa_url, note = try_unpaywall(
                    session, doi,
                    email=email,
                    base_url=endpoints.get("unpaywall_url", "https://api.unpaywall.org/v2/"),
                )
                time.sleep(SLEEP_S)
                if oa_url:
                    dest = fulltext_dir / safe_filename(fname_base)
                    ok, detail = _download_file(session, oa_url, dest)
                    if ok:
                        actual_path = Path(detail)
                        rec.update({
                            "status": "retrieved",
                            "source": note,
                            "file_path": str(actual_path),
                            "file_size_kb": round(actual_path.stat().st_size / 1024, 1),
                            "url": oa_url,
                        })
                        retrieved = True
                        n_got += 1
                    else:
                        rec["note"] = f"unpaywall_dl_fail: {detail}"

            # ---- Source 2: Elsevier ----
            if not retrieved:
                el_url, el_note, el_bytes = try_elsevier(
                    session, doi, sid,
                    api_key=api_key,
                    inst_token=inst_token,
                    article_url=endpoints.get("elsevier_article_url", "https://api.elsevier.com/content/article/doi/"),
                )
                time.sleep(SLEEP_S)
                if el_bytes:
                    ext = ".xml" if "xml" in el_note else ".pdf"
                    dest = fulltext_dir / safe_filename(fname_base, ext)
                    try:
                        dest.write_bytes(el_bytes)
                        rec.update({
                            "status": "retrieved",
                            "source": el_note,
                            "file_path": str(dest),
                            "file_size_kb": round(len(el_bytes) / 1024, 1),
                            "url": el_url or "",
                        })
                        retrieved = True
                        n_got += 1
                    except Exception as e:
                        rec["note"] = f"elsevier_write_fail: {e}"

            # ---- Source 3: Semantic Scholar ----
            if not retrieved:
                s2_url, s2_note = try_semantic_scholar(
                    session, doi, title,
                    base_url=endpoints.get("semantic_scholar_url", "https://api.semanticscholar.org/graph/v1/paper/"),
                )
                time.sleep(SLEEP_S)
                if s2_url:
                    dest = fulltext_dir / safe_filename(fname_base)
                    ok, detail = _download_file(session, s2_url, dest)
                    if ok:
                        actual_path = Path(detail)
                        rec.update({
                            "status": "retrieved",
                            "source": s2_note,
                            "file_path": str(actual_path),
                            "file_size_kb": round(actual_path.stat().st_size / 1024, 1),
                            "url": s2_url,
                        })
                        retrieved = True
                        n_got += 1
                    else:
                        rec["note"] = (rec["note"] + f" | s2_dl_fail: {detail}").strip(" |")

            # ---- Source 4: OpenAlex ----
            if not retrieved:
                oa_url2, oa_note = try_openalex(
                    session, doi,
                    email=email,
                    base_url=endpoints.get("openalex_works_url", "https://api.openalex.org/works/"),
                )
                time.sleep(SLEEP_S)
                if oa_url2:
                    dest = fulltext_dir / safe_filename(fname_base)
                    ok, detail = _download_file(session, oa_url2, dest)
                    if ok:
                        actual_path = Path(detail)
                        rec.update({
                            "status": "retrieved",
                            "source": oa_note,
                            "file_path": str(actual_path),
                            "file_size_kb": round(actual_path.stat().st_size / 1024, 1),
                            "url": oa_url2,
                        })
                        retrieved = True
                        n_got += 1
                    else:
                        rec["note"] = (rec["note"] + f" | openalex_dl_fail: {detail}").strip(" |")

            # ---- Source 5: Frontiers direct PDF ----
            if not retrieved:
                front_url, front_note = try_frontiers(session, doi)
                if front_url:
                    dest = fulltext_dir / safe_filename(fname_base)
                    ok, detail = _download_file(session, front_url, dest)
                    if ok:
                        actual_path = Path(detail)
                        rec.update({
                            "status": "retrieved",
                            "source": front_note,
                            "file_path": str(actual_path),
                            "file_size_kb": round(actual_path.stat().st_size / 1024, 1),
                            "url": front_url,
                        })
                        retrieved = True
                        n_got += 1
                    else:
                        rec["note"] = (rec["note"] + f" | frontiers_dl_fail: {detail}").strip(" |")

            # ---- Source 6: CORE.ac.uk ----
            if not retrieved:
                core_url, core_note = try_core(
                    session, doi, title,
                    base_url=endpoints.get("core_url", "https://api.core.ac.uk/v3/"),
                )
                time.sleep(SLEEP_S)
                if core_url:
                    dest = fulltext_dir / safe_filename(fname_base)
                    ok, detail = _download_file(session, core_url, dest)
                    if ok:
                        actual_path = Path(detail)
                        rec.update({
                            "status": "retrieved",
                            "source": core_note,
                            "file_path": str(actual_path),
                            "file_size_kb": round(actual_path.stat().st_size / 1024, 1),
                            "url": core_url,
                        })
                        retrieved = True
                        n_got += 1
                    else:
                        rec["note"] = (rec["note"] + f" | core_dl_fail: {detail}").strip(" |")

            # ---- Not retrieved ----
            if not retrieved:
                rec["status"] = "needs_manual"
                n_manual += 1

            results.append(rec)
            append_manifest(base, rec)
            if dk:
                already_done.add(dk)

    elapsed_total = time.time() - t0
    print(
        f"[step13] Done | retrieved={n_got:,} skipped={n_skip:,} "
        f"needs_manual={n_manual:,} failed={n_fail:,} | {elapsed_total:,.1f}s"
    )
    return results


# =============================================================================
# Write outputs
# =============================================================================

def write_outputs(
    results: List[dict],
    *,
    out_root: Path,
    input_csv: Path,
    elapsed_seconds: Optional[float],
) -> dict:
    base, _, manifest_csv, summary_json = step13_dirs(out_root)
    manual_csv = base / "step13_manual.csv"

    df_out = pd.DataFrame(results)
    if df_out.empty:
        print("[step13] No results to write.")
        return {}

    # Ensure consistent column order
    cols = ["dedupe_key", "doi", "scopus_id", "title", "year", "pub",
            "missing_abstract", "status", "source", "file_path", "file_size_kb", "url", "note", "timestamp_utc"]
    for c in cols:
        if c not in df_out.columns:
            df_out[c] = ""
    df_out[cols].to_csv(manifest_csv, index=False)

    # Manual retrieval list
    manual_df = df_out[df_out["status"] == "needs_manual"][
        ["dedupe_key", "doi", "scopus_id", "title", "year", "pub", "missing_abstract", "note"]
    ]
    manual_df.to_csv(manual_csv, index=False)

    # Missing-abstract subset CSV
    missing_abs_csv = base / "step13_missing_abstract_results.csv"
    missing_df = df_out[df_out["missing_abstract"] == True]
    missing_df[cols].to_csv(missing_abs_csv, index=False)

    # Summary
    status_counts = df_out["status"].value_counts(dropna=False).to_dict()
    source_counts = df_out[df_out["status"] == "retrieved"]["source"].value_counts(dropna=False).to_dict()

    # Missing-abstract breakdown
    ma = df_out[df_out["missing_abstract"] == True]
    ma_status = ma["status"].value_counts(dropna=False).to_dict() if not ma.empty else {}

    summary = {
        "input_csv": str(input_csv),
        "manifest_csv": str(manifest_csv),
        "manual_csv": str(manual_csv),
        "missing_abstract_csv": str(missing_abs_csv),
        "fulltext_dir": str(base / "fulltext"),
        "total_processed": int(len(df_out)),
        "status_counts": {k: int(v) for k, v in status_counts.items()},
        "source_breakdown": {k: int(v) for k, v in source_counts.items()},
        "missing_abstract_count": int(len(ma)),
        "missing_abstract_status": {k: int(v) for k, v in ma_status.items()},
        "elapsed_seconds": float(elapsed_seconds) if elapsed_seconds else None,
        "elapsed_hms": time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds)) if elapsed_seconds else None,
        "timestamp_utc": _now_utc(),
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[step13] Wrote: {manifest_csv}")
    print(f"[step13] Wrote: {manual_csv}  ({len(manual_df):,} records need manual retrieval)")
    print(f"[step13] Wrote: {missing_abs_csv}  ({len(missing_df):,} missing-abstract records)")
    print(f"[step13] Status: {status_counts}")
    print(f"[step13] By source: {source_counts}")
    print(f"[step13] Missing-abstract retrieval: {ma_status}")
    return summary


# =============================================================================
# Entrypoint
# =============================================================================

def run(config: dict) -> dict:
    load_dotenv()
    t_start = time.time()
    c = config or {}

    out_root   = Path(safe_str(c.get("out_dir", "")) or "outputs")
    api_key    = safe_str(c.get("scopus_api_key", "")) or os.getenv("SCOPUS_API_KEY", "")
    inst_token = safe_str(c.get("scopus_inst_token", "")) or os.getenv("SCOPUS_INST_TOKEN", "")
    email      = safe_str(c.get("contact_email", "")) or os.getenv("CONTACT_EMAIL", "")
    run_limit  = c.get("step13_run_limit") or None

    endpoints = c.get("endpoints") or {}

    if not email:
        print("[step13] WARNING: CONTACT_EMAIL not set — Unpaywall requests may be rate-limited")

    input_csv = step12_csv_path(out_root)
    print(f"[step13] Input  : {input_csv}")
    print(f"[step13] Output : {out_root / 'step13'}")
    print(f"[step13] Email  : {email or '(not set)'}")
    print(f"[step13] Elsevier key: {'set' if api_key else 'NOT SET'}")
    print(f"[step13] Inst token : {'set' if inst_token else 'NOT SET'}")

    df = pd.read_csv(input_csv, engine="python", on_bad_lines="skip")
    if df.empty:
        raise SystemExit(f"Step 12 CSV is empty: {input_csv}")

    results = retrieve_fulltexts(
        df,
        out_root=out_root,
        api_key=api_key,
        inst_token=inst_token,
        email=email,
        endpoints=endpoints,
        run_limit=run_limit,
    )

    return write_outputs(
        results,
        out_root=out_root,
        input_csv=input_csv,
        elapsed_seconds=time.time() - t_start,
    )


def main() -> int:
    try:
        import config as _cfg
        cfg_dict = {
            "out_dir":           str(getattr(_cfg, "out_dir", "outputs")),
            "endpoints":         getattr(_cfg, "endpoints", {}),
            "step13_run_limit":  getattr(_cfg, "step13_run_limit", None),
        }
    except ImportError:
        cfg_dict = {}
    return 0 if run(cfg_dict) else 1


if __name__ == "__main__":
    raise SystemExit(main())
