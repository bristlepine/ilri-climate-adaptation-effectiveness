"""
step13c_retry.py
-----------------
Targeted retry for all records that failed during the original step13 run.
Handles each failure type with the appropriate strategy:

  HTTP 403 (bot-blocked OA content)
      → Re-fetch OA URL from OpenAlex, download with browser-like headers.
        For MDPI (10.3390), also tries direct DOI-redirect → /pdf.

  DNS / connection error (network issue at run time)
      → Simple retry with browser headers — these likely succeed now.

  HTTP 404 (broken link)
      → Re-query OpenAlex for a fresh OA URL and try that instead.

  No DOI / not attempted
      → Skipped (can't retrieve without a DOI).

  Other / unknown
      → Re-query OpenAlex and attempt download.

Run step13b after this to refresh the summary JSON.

Usage:
    python scripts/step13c_retry.py
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
    import re as _re
    return any(_re.search(p, text) for p in _HTML_BAD_PATTERNS)

load_dotenv()

try:
    import config as _cfg
    out_dir = Path(getattr(_cfg, "out_dir", "outputs"))
    EMAIL = getattr(_cfg, "contact_email", "") or os.getenv("CONTACT_EMAIL", "")
except ImportError:
    out_dir = here / "outputs"
    EMAIL = os.getenv("CONTACT_EMAIL", "")

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/pdf,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Referer": "https://www.google.com/",
}

TIMEOUT = 30
PAUSE = 1.2  # seconds between requests


# =============================================================================
# Failure classification
# =============================================================================

def classify_failure(note: str) -> str:
    n = note.lower()
    if not note.strip():
        return "no_doi"
    if "403" in n:
        return "http_403"
    if "nameresolution" in n or "failed to resolve" in n or "connection error" in n:
        return "dns_error"
    if "404" in n:
        return "http_404"
    return "other"


# =============================================================================
# URL resolution helpers
# =============================================================================

def get_oa_url_openalex(doi: str) -> str | None:
    """Query OpenAlex for the best OA PDF URL."""
    try:
        params = {"select": "open_access,best_oa_location"}
        if EMAIL:
            params["mailto"] = EMAIL
        r = requests.get(
            f"https://api.openalex.org/works/https://doi.org/{doi}",
            params=params,
            timeout=15,
        )
        if r.status_code != 200:
            return None
        loc = r.json().get("best_oa_location") or {}
        return loc.get("pdf_url") or loc.get("landing_page_url")
    except Exception:
        return None


def try_mdpi_direct(doi: str) -> str | None:
    """For MDPI DOIs, follow the DOI redirect and construct the /pdf URL."""
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


def extract_url_from_note(note: str) -> str | None:
    """Pull an https URL out of a failure note (e.g. DNS error notes contain the attempted URL)."""
    m = re.search(r"https?://\S+", note)
    return m.group(0).rstrip(").,") if m else None


def resolve_url(doi: str, note: str, failure_type: str) -> str | None:
    """Return the best URL to attempt for this record."""
    # For MDPI 403s, try the direct redirect first
    if failure_type == "http_403" and doi.startswith("10.3390"):
        url = try_mdpi_direct(doi)
        time.sleep(PAUSE)
        if url:
            return url

    # For DNS errors, the original URL is in the note — just retry it
    if failure_type == "dns_error":
        url = extract_url_from_note(note)
        if url:
            return url

    # For everything else: ask OpenAlex for a fresh OA URL
    url = get_oa_url_openalex(doi)
    time.sleep(PAUSE)
    return url


# =============================================================================
# Download
# =============================================================================

def download(url: str, dest_stem: Path) -> Path | None:
    """
    Download url with browser headers. Returns the saved Path on success, else None.
    Chooses .pdf or .html extension based on content-type.
    """
    try:
        r = requests.get(url, headers=BROWSER_HEADERS, timeout=TIMEOUT, stream=True)
        if r.status_code != 200:
            return None
        content = b"".join(r.iter_content(chunk_size=8192))
        if len(content) < 500:
            return None
        ctype = r.headers.get("content-type", "").lower()
        is_html = "html" in ctype or url.rstrip("/").endswith(".html")
        if is_html:
            text = content.decode("utf-8", errors="ignore").lower()
            if _is_html_fake(text):
                return None
            dest = dest_stem.with_suffix(".html")
        else:
            dest = dest_stem.with_suffix(".pdf")
        dest.write_bytes(content)
        return dest
    except Exception:
        return None


def doi_to_dest_stem(doi: str, fulltext_dir: Path) -> Path:
    safe = re.sub(r"[/\\]", "_", doi.strip())
    safe = re.sub(r"[^\w\-.]", "_", safe)
    return fulltext_dir / f"doi_{safe}"


# =============================================================================
# Main
# =============================================================================

def main():
    base, fulltext_dir, manifest_csv, _ = step13_dirs(out_dir)
    fulltext_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_csv.exists():
        print("[step13c] No manifest found — run step13 first.")
        return

    df = pd.read_csv(manifest_csv, dtype=str).fillna("")

    # Classify all needs_manual records
    missing = df[df["status"] == "needs_manual"].copy()
    missing["_failure"] = missing["note"].apply(classify_failure)

    # Exclude no_doi — nothing we can do without a DOI
    retryable = missing[missing["_failure"] != "no_doi"].copy()
    skipped_no_doi = int((missing["_failure"] == "no_doi").sum())

    print(f"[step13c] needs_manual total : {len(missing):,}")
    print(f"[step13c] Skipped (no DOI)   : {skipped_no_doi:,}")
    print(f"[step13c] Retrying           : {len(retryable):,}")
    print()

    # Tally by failure type
    for ftype, grp in retryable.groupby("_failure"):
        print(f"  {ftype:<20}: {len(grp):,}")
    print()

    counters = {"success": 0, "no_url": 0, "still_failed": 0}

    try:
        from tqdm import tqdm
        pbar = tqdm(total=len(retryable), unit="rec", dynamic_ncols=True)
    except ImportError:
        pbar = None

    for i, (idx, row) in enumerate(retryable.iterrows(), 1):
        doi = row["doi"].strip()
        note = row["note"]
        ftype = row["_failure"]

        if pbar:
            pbar.set_description(f"{doi[:45]:<45} ({ftype})")
        else:
            print(f"  [{i}/{len(retryable)}] {doi[:55]:<55} ({ftype})", end=" ... ", flush=True)

        def _log(msg: str):
            if pbar:
                pbar.write(f"  {doi[:55]} → {msg}")
            else:
                print(msg)

        # Skip if file already in fulltext/
        dest_stem = doi_to_dest_stem(doi, fulltext_dir)
        for ext in [".pdf", ".html", ".htm"]:
            if dest_stem.with_suffix(ext).exists():
                _log("already present")
                df.at[idx, "status"] = "retrieved"
                df.at[idx, "file_path"] = str(dest_stem.with_suffix(ext))
                df.at[idx, "note"] = note + " | step13c: already present"
                counters["success"] += 1
                break
        else:
            # Resolve URL
            url = resolve_url(doi, note, ftype)
            if not url:
                _log("no URL found")
                counters["no_url"] += 1
                if pbar:
                    pbar.update(1)
                continue

            # Download
            saved = download(url, dest_stem)
            time.sleep(PAUSE)

            if saved:
                _log(f"OK → {saved.name}")
                df.at[idx, "status"] = "retrieved"
                df.at[idx, "file_path"] = str(saved)
                df.at[idx, "source"] = "step13c_retry"
                df.at[idx, "url"] = url
                df.at[idx, "note"] = note + f" | step13c: retrieved ({ftype})"
                counters["success"] += 1
            else:
                _log("still failed")
                counters["still_failed"] += 1

        if pbar:
            pbar.set_postfix(ok=counters["success"], failed=counters["still_failed"], no_url=counters["no_url"])
            pbar.update(1)

    if pbar:
        pbar.close()

    df.to_csv(manifest_csv, index=False)

    print()
    print("=" * 50)
    print(f"  Retried          : {len(retryable):,}")
    print(f"  Newly retrieved  : {counters['success']:,}")
    print(f"  No URL found     : {counters['no_url']:,}")
    print(f"  Still failing    : {counters['still_failed']:,}")
    print(f"  Skipped (no DOI) : {skipped_no_doi:,}")
    print("=" * 50)
    print(f"\n[step13c] Manifest updated: {manifest_csv}")
    print(f"[step13c] Run step13b next to refresh the summary JSON.")


if __name__ == "__main__":
    main()
