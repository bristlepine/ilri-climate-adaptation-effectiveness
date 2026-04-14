"""
step13d_title_retrieval.py
---------------------------
Two-pronged retrieval pass targeting the remaining ~2,242 missing records:

  1. No-DOI records (~959)
       → Search CrossRef by title to find a DOI
       → Then query OpenAlex/Unpaywall for OA PDF URL and download

  2. Remaining HTTP 403 records (~343)
       → Try CORE API (institutional repositories) — different source
         from OpenAlex, often has accepted manuscripts and preprints

Run step13b after this to refresh the summary JSON.

Usage:
    python scripts/step13d_title_retrieval.py           # run both pools
    python scripts/step13d_title_retrieval.py --pool b  # Pool B only (CORE for 403s — faster)
    python scripts/step13d_title_retrieval.py --pool a  # Pool A only (title search)
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

CORE_API_KEY = os.getenv("CORE_API_KEY", "")  # optional — free key at core.ac.uk/services/api

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
        # Use string concatenation to avoid Path.with_suffix() truncating DOIs with dots
        ext = ".html" if is_html else ".pdf"
        if is_html:
            text = content.decode("utf-8", errors="ignore").lower()
            if _is_html_fake(text):
                return None
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
            # Basic title similarity check
            candidate = " ".join(item.get("title", [""])).lower()
            query_words = set(title.lower().split())
            candidate_words = set(candidate.split())
            overlap = len(query_words & candidate_words) / max(len(query_words), 1)
            if overlap >= 0.7:
                # Year check if available
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

def core_url_from_doi(doi: str) -> str | None:
    """Query CORE API for a download URL by DOI."""
    try:
        headers = {"Authorization": f"Bearer {CORE_API_KEY}"} if CORE_API_KEY else {}
        r = requests.get(
            f"https://api.core.ac.uk/v3/works",
            params={"q": f"doi:{doi}", "limit": 1},
            headers=headers,
            timeout=10,
        )
        if r.status_code != 200:
            return None
        results = r.json().get("results", [])
        if not results:
            return None
        work = results[0]
        return work.get("downloadUrl") or work.get("sourceFulltextUrls", [None])[0]
    except Exception:
        return None


def core_url_from_title(title: str) -> str | None:
    """Query CORE API for a download URL by title."""
    try:
        headers = {"Authorization": f"Bearer {CORE_API_KEY}"} if CORE_API_KEY else {}
        r = requests.get(
            f"https://api.core.ac.uk/v3/works",
            params={"q": f'title:"{title}"', "limit": 3},
            headers=headers,
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
# Main
# =============================================================================

def main(pool: str = "both"):
    base, fulltext_dir, manifest_csv, _ = step13_dirs(out_dir)
    fulltext_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_csv.exists():
        print("[step13d] No manifest found — run step13 first.")
        return

    df = pd.read_csv(manifest_csv, dtype=str).fillna("")
    missing = df[df["status"] == "needs_manual"].copy()

    # Split into two target pools
    no_doi  = missing[missing["doi"] == ""].copy()
    has_403 = missing[
        (missing["doi"] != "") &
        missing["note"].str.contains("403", na=False) &
        ~missing["note"].str.contains("step13c", na=False)  # not already retried successfully
    ].copy()

    if pool == "b":
        no_doi = no_doi.iloc[0:0]  # empty
    elif pool == "a":
        has_403 = has_403.iloc[0:0]  # empty

    print(f"[step13d] Pool A — no DOI (title search) : {len(no_doi):,}{'  (skipped)' if pool == 'b' else ''}")
    print(f"[step13d] Pool B — remaining 403s (CORE)  : {len(has_403):,}{'  (skipped)' if pool == 'a' else ''}")
    print(f"[step13d] Total to attempt                : {len(no_doi) + len(has_403):,}")
    if not CORE_API_KEY:
        print("[step13d] NOTE: No CORE_API_KEY set — CORE requests will be rate-limited (unauthenticated)")
    print()

    counters = {"success": 0, "no_url": 0, "still_failed": 0}
    total = len(no_doi) + len(has_403)

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

    def _update(result: str, doi_or_title: str):
        counters[result] += 1
        if pbar:
            pbar.set_postfix(ok=counters["success"], failed=counters["still_failed"], no_url=counters["no_url"])
            pbar.update(1)

    # ------------------------------------------------------------------
    # Pool A: no-DOI records — CrossRef title search → DOI → OA URL
    # ------------------------------------------------------------------
    for idx, row in no_doi.iterrows():
        title = row.get("title", "").strip()
        year  = row.get("year", "").strip()

        if pbar:
            pbar.set_description(f"[A] {title[:50]:<50}")
            pbar.write(f"  [A] {title[:70]}", end=" ... ")
        else:
            print(f"  [A] {title[:70]}", end=" ... ", flush=True)

        if not title:
            _log(f"[A] no title — skipping")
            _update("no_url", "")
            continue

        dest_stem = title_to_dest_stem(title, fulltext_dir)
        for ext in [".pdf", ".html", ".htm"]:
            if dest_stem.with_suffix(ext).exists():
                _log(f"[A] {title[:60]} → already present")
                df.at[idx, "status"] = "retrieved"
                df.at[idx, "file_path"] = str(dest_stem.with_suffix(ext))
                df.at[idx, "note"] = row["note"] + " | step13d: already present"
                _update("success", title)
                break
        else:
            # Step 1: CrossRef title → DOI
            doi = crossref_doi_from_title(title, year)
            time.sleep(PAUSE)

            if not doi:
                # Try CORE title search directly
                url = core_url_from_title(title)
                time.sleep(PAUSE)
                if url:
                    saved = download(url, dest_stem)
                    time.sleep(PAUSE)
                    if saved:
                        _log(f"[A] {title[:55]} → OK (CORE title) → {saved.name}")
                        df.at[idx, "status"] = "retrieved"
                        df.at[idx, "file_path"] = str(saved)
                        df.at[idx, "source"] = "step13d_core_title"
                        df.at[idx, "url"] = url
                        df.at[idx, "note"] = row["note"] + " | step13d: retrieved via CORE title"
                        _update("success", title)
                        continue
                _log(f"[A] {title[:55]} → no DOI / no CORE match")
                _update("no_url", title)
                continue

            # Step 2: DOI → OpenAlex OA URL
            oa_url = openalex_oa_url(doi)
            time.sleep(PAUSE)

            if not oa_url:
                # Try CORE with the found DOI
                oa_url = core_url_from_doi(doi)
                time.sleep(PAUSE)

            if not oa_url:
                _log(f"[A] {title[:55]} → DOI found ({doi}) but no OA URL")
                _update("no_url", title)
                continue

            # Step 3: download
            doi_stem = doi_to_dest_stem(doi, fulltext_dir)
            saved = download(oa_url, doi_stem)
            time.sleep(PAUSE)

            if saved:
                _log(f"[A] {title[:55]} → OK → {saved.name}")
                df.at[idx, "status"] = "retrieved"
                df.at[idx, "doi"] = doi
                df.at[idx, "file_path"] = str(saved)
                df.at[idx, "source"] = "step13d_crossref"
                df.at[idx, "url"] = oa_url
                df.at[idx, "note"] = row["note"] + f" | step13d: DOI={doi} retrieved"
                _update("success", title)
            else:
                _log(f"[A] {title[:55]} → DOI={doi} found but download failed")
                _update("still_failed", title)

    # ------------------------------------------------------------------
    # Pool B: remaining 403s — try CORE API
    # ------------------------------------------------------------------
    for idx, row in has_403.iterrows():
        doi   = row["doi"].strip()
        title = row.get("title", "").strip()

        if pbar:
            pbar.set_description(f"[B] {doi[:50]:<50}")
            pbar.write(f"  [B] {doi[:70]}", end=" ... ")
        else:
            print(f"  [B] {doi[:70]}", end=" ... ", flush=True)

        dest_stem = doi_to_dest_stem(doi, fulltext_dir)
        for ext in [".pdf", ".html", ".htm"]:
            if dest_stem.with_suffix(ext).exists():
                _log(f"[B] {doi} → already present")
                df.at[idx, "status"] = "retrieved"
                df.at[idx, "file_path"] = str(dest_stem.with_suffix(ext))
                df.at[idx, "note"] = row["note"] + " | step13d: already present"
                _update("success", doi)
                break
        else:
            # Try CORE by DOI first
            url = core_url_from_doi(doi)
            time.sleep(PAUSE)

            # Fallback: CORE by title
            if not url and title:
                url = core_url_from_title(title)
                time.sleep(PAUSE)

            if not url:
                _log(f"[B] {doi} → no CORE match")
                _update("no_url", doi)
                continue

            saved = download(url, dest_stem)
            time.sleep(PAUSE)

            if saved:
                _log(f"[B] {doi} → OK (CORE) → {saved.name}")
                df.at[idx, "status"] = "retrieved"
                df.at[idx, "file_path"] = str(saved)
                df.at[idx, "source"] = "step13d_core"
                df.at[idx, "url"] = url
                df.at[idx, "note"] = row["note"] + " | step13d: retrieved via CORE"
                _update("success", doi)
            else:
                _log(f"[B] {doi} → CORE URL found but download failed")
                _update("still_failed", doi)

    if pbar:
        pbar.close()

    df.to_csv(manifest_csv, index=False)

    print()
    print("=" * 50)
    print(f"  Total attempted  : {total:,}")
    print(f"  Newly retrieved  : {counters['success']:,}")
    print(f"  No URL found     : {counters['no_url']:,}")
    print(f"  Still failing    : {counters['still_failed']:,}")
    print("=" * 50)
    print(f"\n[step13d] Manifest updated: {manifest_csv}")
    print(f"[step13d] Run step13b next to refresh the summary JSON.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", choices=["a", "b", "both"], default="both",
                    help="Which pool to run: a=no-DOI title search, b=CORE for 403s, both=default")
    args = ap.parse_args()
    main(pool=args.pool)
