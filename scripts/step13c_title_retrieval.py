"""
step13c_title_retrieval.py
---------------------------
Three-pronged retrieval pass targeting remaining missing records:

  Pool A — No-DOI records (~959)
       → Search CrossRef by title to find a DOI
       → Then query OpenAlex/Unpaywall for OA PDF URL and download
       → Fallback: CORE API title search

  Pool B — Remaining HTTP 403 records (non-MDPI) (~265)
       → Try CORE API (institutional repositories) via direct DOI lookup
         Uses quoted DOI query: doi:"10.1080/..." to handle slashes correctly

  Pool C — MDPI 403 records (~77)
       → Follow DOI redirect to MDPI landing page, append /pdf
         MDPI is fully OA — these are just Cloudflare-blocked, not paywalled

Run step13d after this to refresh the summary JSON and export the missing papers list.

Usage:
    python scripts/step13c_title_retrieval.py             # run all pools
    python scripts/step13c_title_retrieval.py --pool a    # Pool A only (title search)
    python scripts/step13c_title_retrieval.py --pool b    # Pool B only (CORE for non-MDPI 403s)
    python scripts/step13c_title_retrieval.py --pool c    # Pool C only (MDPI direct PDF)
    python scripts/step13c_title_retrieval.py --pool bc   # Pools B and C together
"""

import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

here = Path(__file__).parent
sys.path.insert(0, str(here))
from step13_retrieve_fulltext import step13_dirs, _HTML_BAD_PATTERNS

def _is_html_fake(text: str) -> bool:
    return any(re.search(p, text) for p in _HTML_BAD_PATTERNS)

load_dotenv()

try:
    import config as _cfg
    out_dir = Path(getattr(_cfg, "out_dir", "outputs"))
    EMAIL   = getattr(_cfg, "contact_email", "") or os.getenv("CONTACT_EMAIL", "")
except ImportError:
    out_dir = here / "outputs"
    EMAIL   = os.getenv("CONTACT_EMAIL", "")

CORE_API_KEY = os.getenv("CORE_API_KEY", "")

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/pdf,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
}

TIMEOUT = 30
PAUSE   = 1.2


# =============================================================================
# Helpers
# =============================================================================

def doi_to_dest_stem(doi: str, fulltext_dir: Path) -> Path:
    safe = re.sub(r"[/\\]", "_", doi.strip())
    safe = re.sub(r"[^\w\-.]", "_", safe)
    return fulltext_dir / f"doi_{safe}"


def title_to_dest_stem(title: str, fulltext_dir: Path) -> Path:
    safe = re.sub(r"[^\w\s\-]", "", title.lower())
    safe = re.sub(r"\s+", "_", safe.strip())[:80]
    return fulltext_dir / f"title_{safe}"


def already_present(dest_stem: Path) -> Path | None:
    """Return existing file at dest_stem.{pdf,html,htm} if any."""
    for ext in [".pdf", ".html", ".htm"]:
        p = Path(str(dest_stem) + ext)
        if p.exists():
            return p
    return None


def download(url: str, dest_stem: Path) -> Path | None:
    try:
        r = requests.get(url, headers=BROWSER_HEADERS, timeout=TIMEOUT, stream=True)
        if r.status_code != 200:
            return None
        content = b"".join(r.iter_content(chunk_size=8192))
        if len(content) < 500:
            return None
        ctype = r.headers.get("content-type", "").lower()
        is_html = "html" in ctype or url.rstrip("/").endswith(".html")
        ext = ".html" if is_html else ".pdf"
        if is_html:
            text = content.decode("utf-8", errors="ignore").lower()
            if _is_html_fake(text):
                return None
        # String concat avoids Path.with_suffix() truncating DOIs that contain dots
        dest = Path(str(dest_stem) + ext)
        dest.write_bytes(content)
        return dest
    except Exception:
        return None


# =============================================================================
# Source 1: CrossRef title search → DOI
# =============================================================================

def crossref_doi_from_title(title: str, year: str = "") -> str | None:
    """Search CrossRef by title, return best-matching DOI."""
    try:
        params = {
            "query.title": title,
            "rows": 3,
            "select": "DOI,title,published",
        }
        if EMAIL:
            params["mailto"] = EMAIL
        r = requests.get("https://api.crossref.org/works", params=params, timeout=15)
        if r.status_code != 200:
            return None
        items = r.json().get("message", {}).get("items", [])
        for item in items:
            candidate = " ".join(item.get("title", [""])).lower()
            query_words = set(title.lower().split())
            candidate_words = set(candidate.split())
            overlap = len(query_words & candidate_words) / max(len(query_words), 1)
            if overlap >= 0.7:
                if year:
                    pub = item.get("published", {}).get("date-parts", [[None]])[0][0]
                    if pub and str(pub) != str(year):
                        continue
                return item.get("DOI")
    except Exception:
        pass
    return None


# =============================================================================
# Source 2: OpenAlex OA URL from DOI
# =============================================================================

def openalex_oa_url(doi: str) -> str | None:
    try:
        params = {"select": "best_oa_location"}
        if EMAIL:
            params["mailto"] = EMAIL
        r = requests.get(
            f"https://api.openalex.org/works/https://doi.org/{doi}",
            params=params, timeout=15,
        )
        if r.status_code != 200:
            return None
        loc = r.json().get("best_oa_location") or {}
        return loc.get("pdf_url") or loc.get("landing_page_url")
    except Exception:
        return None


# =============================================================================
# Source 3: CORE API (institutional repos, preprints)
# =============================================================================

def _core_headers() -> dict:
    return {"Authorization": f"Bearer {CORE_API_KEY}"} if CORE_API_KEY else {}


def core_url_from_doi(doi: str) -> str | None:
    """
    Query CORE API for a download URL by DOI.
    Uses two strategies:
      1. Direct endpoint:  GET /v3/works/doi:{encoded_doi}
      2. Quoted search:    q=doi:"10.xxx/yyy"  (quotes prevent slash from breaking parser)
    """
    headers = _core_headers()

    # Strategy 1: direct DOI endpoint (slash encoded as %2F)
    encoded = doi.strip().replace("/", "%2F")
    try:
        r = requests.get(
            f"https://api.core.ac.uk/v3/works/doi:{encoded}",
            headers=headers,
            timeout=10,
        )
        if r.status_code == 200:
            work = r.json()
            url = work.get("downloadUrl") or (work.get("sourceFulltextUrls") or [None])[0]
            if url:
                return url
    except Exception:
        pass

    # Strategy 2: search with quoted DOI
    try:
        r = requests.get(
            "https://api.core.ac.uk/v3/works",
            params={"q": f'doi:"{doi.strip()}"', "limit": 1},
            headers=headers,
            timeout=10,
        )
        if r.status_code == 200:
            results = r.json().get("results", [])
            if results:
                work = results[0]
                url = work.get("downloadUrl") or (work.get("sourceFulltextUrls") or [None])[0]
                if url:
                    return url
    except Exception:
        pass

    return None


def core_url_from_title(title: str) -> str | None:
    """Query CORE API for a download URL by title."""
    try:
        r = requests.get(
            "https://api.core.ac.uk/v3/works",
            params={"q": f'title:"{title}"', "limit": 3},
            headers=_core_headers(),
            timeout=10,
        )
        if r.status_code != 200:
            return None
        results = r.json().get("results", [])
        query_words = set(title.lower().split())
        for work in results:
            candidate = (work.get("title") or "").lower()
            candidate_words = set(candidate.split())
            overlap = len(query_words & candidate_words) / max(len(query_words), 1)
            if overlap >= 0.7:
                url = work.get("downloadUrl") or (work.get("sourceFulltextUrls") or [None])[0]
                if url:
                    return url
    except Exception:
        pass
    return None


# =============================================================================
# Source 4: MDPI direct PDF (Pool C)
# =============================================================================

def mdpi_direct_pdf_url(doi: str) -> str | None:
    """
    Follow the DOI redirect to the MDPI landing page, then construct the /pdf URL.
    MDPI papers are fully open access — the 403 comes from Cloudflare bot detection,
    not a paywall. Direct /pdf URLs bypass this in many cases.
    """
    try:
        r = requests.get(
            f"https://doi.org/{doi}",
            headers=BROWSER_HEADERS,
            allow_redirects=True,
            timeout=TIMEOUT,
        )
        if r.ok and "mdpi.com" in r.url:
            base = r.url.split("?")[0].rstrip("/")
            return base if base.endswith("/pdf") else base + "/pdf"
    except Exception:
        pass
    return None


# =============================================================================
# Main
# =============================================================================

def main(pool: str = "all"):
    base, fulltext_dir, manifest_csv, _ = step13_dirs(out_dir)
    fulltext_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_csv.exists():
        print("[step13c] No manifest found — run step13 first.")
        return

    df = pd.read_csv(manifest_csv, dtype=str).fillna("")
    missing = df[df["status"] == "needs_manual"].copy()

    # Pool A: no DOI at all
    pool_a = missing[missing["doi"] == ""].copy()

    # Pool B: 403s, non-MDPI
    pool_b = missing[
        (missing["doi"] != "") &
        missing["note"].str.contains("403", na=False) &
        ~missing["doi"].str.startswith("10.3390")
    ].copy()

    # Pool C: MDPI 403s (10.3390/*)
    pool_c = missing[
        missing["doi"].str.startswith("10.3390") &
        missing["note"].str.contains("403", na=False)
    ].copy()

    # Apply pool filter
    run_a = pool in ("a", "all") or "a" in pool
    run_b = pool in ("b", "all", "bc") or "b" in pool
    run_c = pool in ("c", "all", "bc") or "c" in pool

    if not run_a: pool_a = pool_a.iloc[0:0]
    if not run_b: pool_b = pool_b.iloc[0:0]
    if not run_c: pool_c = pool_c.iloc[0:0]

    total = len(pool_a) + len(pool_b) + len(pool_c)

    print(f"[step13c] Pool A — no-DOI (CrossRef title + CORE title) : {len(pool_a):,}{'  (skipped)' if not run_a else ''}")
    print(f"[step13c] Pool B — non-MDPI 403s (CORE direct + search) : {len(pool_b):,}{'  (skipped)' if not run_b else ''}")
    print(f"[step13c] Pool C — MDPI 403s (DOI redirect → /pdf)      : {len(pool_c):,}{'  (skipped)' if not run_c else ''}")
    print(f"[step13c] Total to attempt                               : {total:,}")
    if not CORE_API_KEY:
        print("[step13c] NOTE: No CORE_API_KEY — CORE requests will be rate-limited")
    print()

    counters = {"success": 0, "no_url": 0, "still_failed": 0}

    try:
        from tqdm import tqdm
        pbar = tqdm(total=total, unit="rec", dynamic_ncols=True)
    except ImportError:
        pbar = None

    def _log(msg: str):
        if pbar:
            pbar.write(f"  {msg}")
        else:
            print(f"  {msg}")

    def _save():
        """Persist manifest to disk — called after every successful retrieval."""
        df.to_csv(manifest_csv, index=False)

    def _tick(result: str):
        counters[result] += 1
        if pbar:
            pbar.set_postfix(ok=counters["success"], no_url=counters["no_url"], fail=counters["still_failed"])
            pbar.update(1)

    # ------------------------------------------------------------------
    # Pool A: no-DOI records — CrossRef title search → DOI → OA URL
    # ------------------------------------------------------------------
    for idx, row in pool_a.iterrows():
        title = row.get("title", "").strip()
        year  = row.get("year", "").strip()

        desc = title[:60] or "(no title)"
        if pbar:
            pbar.set_description(f"[A] {desc:<60}")
            pbar.write(f"  [A] {desc}", end=" ... ")
        else:
            print(f"  [A] {desc}", end=" ... ", flush=True)

        if not title:
            _log("no title — skipping")
            _tick("no_url")
            continue

        dest_stem = title_to_dest_stem(title, fulltext_dir)
        existing = already_present(dest_stem)
        if existing:
            _log(f"already present → {existing.name}")
            df.at[idx, "status"] = "retrieved"
            df.at[idx, "file_path"] = str(existing)
            df.at[idx, "note"] = row["note"] + " | step13c: already present"
            _tick("success")
            continue

        # CrossRef → DOI
        doi = crossref_doi_from_title(title, year)
        time.sleep(PAUSE)

        if doi:
            # DOI found — try OpenAlex then CORE
            oa_url = openalex_oa_url(doi)
            time.sleep(PAUSE)
            if not oa_url:
                oa_url = core_url_from_doi(doi)
                time.sleep(PAUSE)

            if oa_url:
                doi_stem = doi_to_dest_stem(doi, fulltext_dir)
                saved = download(oa_url, doi_stem)
                time.sleep(PAUSE)
                if saved:
                    _log(f"CrossRef DOI={doi} → OK → {saved.name}")
                    df.at[idx, "status"] = "retrieved"
                    df.at[idx, "doi"] = doi
                    df.at[idx, "file_path"] = str(saved)
                    df.at[idx, "source"] = "step13c_crossref"
                    df.at[idx, "url"] = oa_url
                    df.at[idx, "note"] = row["note"] + f" | step13c: DOI={doi} retrieved"
                    _save()
                    _tick("success")
                    continue
                else:
                    _log(f"CrossRef DOI={doi} found, OA URL found, download failed")
                    _tick("still_failed")
                    continue
            else:
                _log(f"CrossRef DOI={doi} found but no OA URL")
                # fall through to CORE title search

        # CORE title search (no DOI found, or no OA URL from DOI)
        core_url = core_url_from_title(title)
        time.sleep(PAUSE)
        if core_url:
            saved = download(core_url, dest_stem)
            time.sleep(PAUSE)
            if saved:
                _log(f"CORE title → OK → {saved.name}")
                df.at[idx, "status"] = "retrieved"
                df.at[idx, "file_path"] = str(saved)
                df.at[idx, "source"] = "step13c_core_title"
                df.at[idx, "url"] = core_url
                df.at[idx, "note"] = row["note"] + " | step13c: retrieved via CORE title"
                _save()
                _tick("success")
                continue

        _log("no match")
        _tick("no_url")

    # ------------------------------------------------------------------
    # Pool B: non-MDPI 403s — CORE direct DOI lookup
    # ------------------------------------------------------------------
    for idx, row in pool_b.iterrows():
        doi   = row["doi"].strip()
        title = row.get("title", "").strip()

        desc = doi[:60]
        if pbar:
            pbar.set_description(f"[B] {desc:<60}")
            pbar.write(f"  [B] {desc}", end=" ... ")
        else:
            print(f"  [B] {desc}", end=" ... ", flush=True)

        dest_stem = doi_to_dest_stem(doi, fulltext_dir)
        existing = already_present(dest_stem)
        if existing:
            _log(f"already present → {existing.name}")
            df.at[idx, "status"] = "retrieved"
            df.at[idx, "file_path"] = str(existing)
            df.at[idx, "note"] = row["note"] + " | step13c: already present"
            _tick("success")
            continue

        url = core_url_from_doi(doi)
        time.sleep(PAUSE)

        if not url and title:
            url = core_url_from_title(title)
            time.sleep(PAUSE)

        if not url:
            _log("no CORE match")
            _tick("no_url")
            continue

        saved = download(url, dest_stem)
        time.sleep(PAUSE)

        if saved:
            _log(f"CORE → OK → {saved.name}")
            df.at[idx, "status"] = "retrieved"
            df.at[idx, "file_path"] = str(saved)
            df.at[idx, "source"] = "step13c_core"
            df.at[idx, "url"] = url
            df.at[idx, "note"] = row["note"] + " | step13c: retrieved via CORE"
            _save()
            _tick("success")
        else:
            _log("CORE URL found but download failed")
            _tick("still_failed")

    # ------------------------------------------------------------------
    # Pool C: MDPI 403s — DOI redirect → landing page → /pdf
    # ------------------------------------------------------------------
    for idx, row in pool_c.iterrows():
        doi   = row["doi"].strip()
        title = row.get("title", "").strip()

        desc = doi[:60]
        if pbar:
            pbar.set_description(f"[C] {desc:<60}")
            pbar.write(f"  [C] {desc}", end=" ... ")
        else:
            print(f"  [C] {desc}", end=" ... ", flush=True)

        dest_stem = doi_to_dest_stem(doi, fulltext_dir)
        existing = already_present(dest_stem)
        if existing:
            _log(f"already present → {existing.name}")
            df.at[idx, "status"] = "retrieved"
            df.at[idx, "file_path"] = str(existing)
            df.at[idx, "note"] = row["note"] + " | step13c: already present"
            _tick("success")
            continue

        # Try MDPI direct PDF
        pdf_url = mdpi_direct_pdf_url(doi)
        time.sleep(PAUSE)

        if pdf_url:
            saved = download(pdf_url, dest_stem)
            time.sleep(PAUSE)
            if saved:
                _log(f"MDPI direct /pdf → OK → {saved.name}")
                df.at[idx, "status"] = "retrieved"
                df.at[idx, "file_path"] = str(saved)
                df.at[idx, "source"] = "step13c_mdpi"
                df.at[idx, "url"] = pdf_url
                df.at[idx, "note"] = row["note"] + " | step13c: MDPI direct PDF"
                _save()
                _tick("success")
                continue
            else:
                _log("MDPI /pdf URL resolved but download failed")
                _tick("still_failed")
                continue

        # MDPI fallback: try CORE
        url = core_url_from_doi(doi)
        time.sleep(PAUSE)
        if url:
            saved = download(url, dest_stem)
            time.sleep(PAUSE)
            if saved:
                _log(f"CORE fallback → OK → {saved.name}")
                df.at[idx, "status"] = "retrieved"
                df.at[idx, "file_path"] = str(saved)
                df.at[idx, "source"] = "step13c_mdpi_core"
                df.at[idx, "url"] = url
                df.at[idx, "note"] = row["note"] + " | step13c: MDPI via CORE"
                _save()
                _tick("success")
                continue

        _log("no match")
        _tick("no_url")

    if pbar:
        pbar.close()

    df.to_csv(manifest_csv, index=False)

    print()
    print("=" * 56)
    print(f"  Total attempted  : {total:,}")
    print(f"  Newly retrieved  : {counters['success']:,}")
    print(f"  No URL found     : {counters['no_url']:,}")
    print(f"  Still failing    : {counters['still_failed']:,}")
    print("=" * 56)
    print(f"\n[step13c] Manifest updated: {manifest_csv}")
    print(f"[step13c] Run step13d next to refresh the summary JSON and export the missing papers list.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pool",
        default="all",
        help="Which pools to run: a, b, c, bc, or all (default: all)",
    )
    args = ap.parse_args()
    main(pool=args.pool)
