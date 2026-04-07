"""
check_institutional_token.py

Tests whether the Elsevier institutional token improves:
  (A) Abstract retrieval — records that previously returned HTTP 403
  (B) Full-text retrieval — records that failed in step13

Each record is tried twice: without token, then with token.
Prints a side-by-side comparison and summary for both tests.

Usage:
    python scripts/check_institutional_token.py
"""

import os
import time
import textwrap
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY     = os.getenv("SCOPUS_API_KEY", "").strip()
INST_TOKEN  = os.getenv("SCOPUS_INST_TOKEN", "").strip()

MISSING_ABSTRACT_CSV = Path("scripts/outputs/step9a/step9a_missing.csv")
FAILED_FULLTEXT_CSV  = Path("scripts/outputs/step13/step13_manifest.csv")

ABSTRACT_URL = "https://api.elsevier.com/content/abstract/scopus_id/{sid}"
ARTICLE_URL  = "https://api.elsevier.com/content/article/doi/{doi}"
CORE_DOI_URL = "https://api.core.ac.uk/v3/works/doi/{doi}"

N_SAMPLES = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _headers(use_token: bool) -> dict:
    h = {"X-ELS-APIKey": API_KEY, "Accept": "application/json"}
    if use_token and INST_TOKEN:
        h["X-ELS-Insttoken"] = INST_TOKEN
    return h


def fetch_abstract(scopus_id: str, use_token: bool) -> dict:
    url = ABSTRACT_URL.format(sid=scopus_id)
    result = {"status": None, "abstract": None}
    try:
        r = requests.get(url, headers=_headers(use_token), timeout=15)
        result["status"] = r.status_code
        if r.status_code == 200:
            ab = (r.json()
                   .get("abstracts-retrieval-response", {})
                   .get("coredata", {})
                   .get("dc:description", ""))
            result["abstract"] = ab.strip() if ab else None
    except Exception as e:
        result["status"] = f"ERROR: {e}"
    time.sleep(0.3)
    return result


def fetch_fulltext(doi: str, scopus_id: str, use_token: bool) -> dict:
    """Try Elsevier full-text API by DOI, fall back to scopus_id."""
    result = {"status": None, "got_text": False, "size_kb": None}

    urls = []
    if doi:
        urls.append(f"https://api.elsevier.com/content/article/doi/{doi}")
    if scopus_id:
        urls.append(f"https://api.elsevier.com/content/article/scopus_id/{scopus_id}")

    h = _headers(use_token)
    h["Accept"] = "application/pdf"

    for url in urls:
        try:
            r = requests.get(url, headers=h, timeout=20)
            result["status"] = r.status_code
            if r.status_code == 200 and len(r.content) > 1000:
                result["got_text"] = True
                result["size_kb"] = round(len(r.content) / 1024, 1)
                return result
        except Exception as e:
            result["status"] = f"ERROR: {e}"
        time.sleep(0.3)

    return result


def print_header(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


# ---------------------------------------------------------------------------
# Test A: Abstract retrieval
# ---------------------------------------------------------------------------

def test_abstracts():
    print_header("TEST A: Abstract retrieval (previously HTTP 403)")

    if not MISSING_ABSTRACT_CSV.exists():
        print(f"  Missing CSV not found: {MISSING_ABSTRACT_CSV}")
        print("  Run step9 and step9a first.")
        return 0, 0

    df = pd.read_csv(MISSING_ABSTRACT_CSV, dtype=str, keep_default_na=False)
    elsevier = df[df["last_source_attempted"].str.contains("elsevier", case=False, na=False)]
    sample = elsevier[elsevier["scopus_id"].str.strip() != ""].head(N_SAMPLES)

    recovered = 0
    for _, row in sample.iterrows():
        sid   = row["scopus_id"].strip()
        title = row.get("title", "(no title)")[:60]
        print(f"\n  Record : {title}...")
        print(f"  Scopus ID: {sid}")
        print(f"  {'-' * 60}")

        without = fetch_abstract(sid, use_token=False)
        with_   = fetch_abstract(sid, use_token=True)

        print(f"  WITHOUT token → HTTP {without['status']}  |  abstract: {'✓' if without['abstract'] else '✗ none'}")
        print(f"  WITH    token → HTTP {with_['status']}  |  abstract: {'✓ GOT IT' if with_['abstract'] else '✗ none'}")

        if with_["abstract"] and not without["abstract"]:
            recovered += 1
            preview = textwrap.fill(with_["abstract"][:300], width=66, initial_indent="    > ")
            print(f"\n  Token unlocked abstract preview:")
            print(preview + ("..." if len(with_["abstract"]) > 300 else ""))
        elif not with_["abstract"]:
            print("  Token did not help for this record.")

        time.sleep(0.3)

    n = len(sample)
    print(f"\n  Result: {recovered}/{n} abstracts unlocked by token")
    if recovered > 0:
        est = int((recovered / n) * 1314)
        print(f"  Estimated abstracts recoverable from 1,314 missing: ~{est:,}")
    return recovered, n


# ---------------------------------------------------------------------------
# Test B: Full-text retrieval
# ---------------------------------------------------------------------------

def test_fulltext():
    print_header("TEST B: Full-text retrieval (previously failed in step13)")

    if not FAILED_FULLTEXT_CSV.exists():
        print(f"  Manifest CSV not found: {FAILED_FULLTEXT_CSV}")
        print("  Run step13 first.")
        return 0, 0

    df = pd.read_csv(FAILED_FULLTEXT_CSV, dtype=str, keep_default_na=False)
    failed = df[df["status"] == "needs_manual"]
    # Focus on Elsevier-published records (10.1016 = ScienceDirect)
    # Non-Elsevier publishers (Springer, Wiley etc.) will always 404 regardless of token
    has_doi = failed[failed["doi"].str.startswith("10.1016", na=False)].head(N_SAMPLES)

    recovered = 0
    for _, row in has_doi.iterrows():
        doi   = row.get("doi", "").strip()
        sid   = row.get("scopus_id", "").strip()
        title = row.get("title", "(no title)")[:60]
        print(f"\n  Record : {title}...")
        print(f"  DOI: {doi or '(none)'}  |  Scopus ID: {sid or '(none)'}")
        print(f"  {'-' * 60}")

        without = fetch_fulltext(doi, sid, use_token=False)
        with_   = fetch_fulltext(doi, sid, use_token=True)

        print(f"  WITHOUT token → HTTP {without['status']}  |  full text: {'✓ ' + str(without['size_kb']) + ' KB' if without['got_text'] else '✗ none'}")
        print(f"  WITH    token → HTTP {with_['status']}  |  full text: {'✓ ' + str(with_['size_kb']) + ' KB  ← TOKEN WORKS' if with_['got_text'] else '✗ none'}")

        if with_["got_text"] and not without["got_text"]:
            recovered += 1
        elif not with_["got_text"]:
            print("  Token did not help for this record.")

        time.sleep(0.3)

    n = len(has_doi)
    print(f"\n  Result: {recovered}/{n} full texts unlocked by token")
    if recovered > 0:
        est = int((recovered / n) * 5277)
        print(f"  Estimated full texts recoverable from 5,277 missing: ~{est:,}")
    return recovered, n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def test_core():
    """Test CORE.ac.uk on non-Elsevier records that step13 couldn't retrieve."""
    print_header("TEST C: CORE.ac.uk (non-Elsevier open-access repositories)")

    if not FAILED_FULLTEXT_CSV.exists():
        print(f"  Manifest CSV not found: {FAILED_FULLTEXT_CSV}")
        return 0, 0

    df = pd.read_csv(FAILED_FULLTEXT_CSV, dtype=str, keep_default_na=False)
    failed = df[df["status"] == "needs_manual"]
    # Use non-Elsevier records — these are what CORE is good for
    non_elsevier = failed[
        failed["doi"].str.strip().ne("") &
        ~failed["doi"].str.startswith("10.1016", na=True)
    ].head(N_SAMPLES)

    found = 0
    for _, row in non_elsevier.iterrows():
        doi   = row.get("doi", "").strip()
        title = row.get("title", "(no title)")[:60]
        print(f"\n  Record : {title}...")
        print(f"  DOI: {doi}")
        print(f"  {'-' * 60}")

        url = CORE_DOI_URL.format(doi=doi)
        try:
            r = requests.get(url, timeout=15)
            pdf_url = ""
            if r.status_code == 200:
                data = r.json()
                pdf_url = (data.get("downloadUrl") or data.get("pdfUrl") or "").strip()
            status = r.status_code
        except Exception as e:
            status = f"ERROR: {e}"
            pdf_url = ""

        if pdf_url:
            found += 1
            print(f"  CORE → HTTP {status}  |  ✓ PDF found: {pdf_url[:80]}")
        else:
            print(f"  CORE → HTTP {status}  |  ✗ not in CORE")

        time.sleep(0.3)

    n = len(non_elsevier)
    print(f"\n  Result: {found}/{n} full texts found in CORE")
    if found > 0:
        est = int((found / n) * 4500)  # ~4500 non-Elsevier failed records
        print(f"  Estimated additional full texts from CORE: ~{est:,}")
    return found, n


BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/pdf,*/*;q=0.9",
    "Accept-Language": "en-US,en;q=0.9",
}


def test_browser_headers():
    """Test whether browser-like headers fix 403 downloads on MDPI/Frontiers."""
    print_header("TEST D: Browser headers fix for MDPI/Frontiers 403s")

    if not FAILED_FULLTEXT_CSV.exists():
        print(f"  Manifest CSV not found: {FAILED_FULLTEXT_CSV}")
        return 0, 0

    df = pd.read_csv(FAILED_FULLTEXT_CSV, dtype=str, keep_default_na=False)
    failed = df[df["status"] == "needs_manual"]
    # Focus on MDPI (10.3390) and Frontiers (10.3389) — fully open access, should work
    open_access = failed[
        failed["doi"].str.startswith(("10.3390", "10.3389"), na=False)
    ].head(N_SAMPLES)

    fixed = 0
    for _, row in open_access.iterrows():
        doi   = row.get("doi", "").strip()
        title = row.get("title", "(no title)")[:60]
        note  = row.get("note", "")
        print(f"\n  Record : {title}...")
        print(f"  DOI: {doi}")
        print(f"  Previous failure: {note[:80]}")
        print(f"  {'-' * 60}")

        # Get PDF URL via Unpaywall
        try:
            r = requests.get(
                f"https://api.unpaywall.org/v2/{doi}",
                params={"email": "zarrar85@gmail.com"},
                timeout=15
            )
            pdf_url = ""
            if r.status_code == 200:
                data = r.json()
                pdf_url = data.get("best_oa_location", {}) or {}
                pdf_url = pdf_url.get("url_for_pdf") or pdf_url.get("url") or ""
                if not pdf_url:
                    for loc in (data.get("oa_locations") or []):
                        pdf_url = loc.get("url_for_pdf") or loc.get("url") or ""
                        if pdf_url:
                            break
        except Exception as e:
            pdf_url = ""
            print(f"  Unpaywall error: {e}")

        if not pdf_url:
            print(f"  Unpaywall: no URL found — skipping download test")
            continue

        # Try download WITHOUT browser headers
        try:
            r1 = requests.get(pdf_url, timeout=15, stream=True)
            status_without = r1.status_code
            size_without = len(r1.content) if r1.status_code == 200 else 0
        except Exception as e:
            status_without = f"ERROR"
            size_without = 0

        # Try download WITH browser headers
        try:
            r2 = requests.get(pdf_url, headers=BROWSER_HEADERS, timeout=15, stream=True)
            status_with = r2.status_code
            size_with = len(r2.content) if r2.status_code == 200 else 0
        except Exception as e:
            status_with = f"ERROR"
            size_with = 0

        print(f"  PDF URL: {pdf_url[:80]}")
        print(f"  WITHOUT browser headers → HTTP {status_without}  |  {'✓ ' + str(round(size_without/1024,1)) + ' KB' if size_without > 1000 else '✗ blocked'}")
        print(f"  WITH    browser headers → HTTP {status_with}  |  {'✓ ' + str(round(size_with/1024,1)) + ' KB  ← FIXED' if size_with > 1000 else '✗ still blocked'}")

        if size_with > 1000 and size_without <= 1000:
            fixed += 1

        time.sleep(0.5)

    n = len(open_access)
    print(f"\n  Result: {fixed}/{n} downloads fixed by browser headers")
    if fixed > 0:
        est = int((fixed / n) * 606)  # ~606 MDPI+Frontiers papers
        print(f"  Estimated additional full texts from header fix: ~{est:,}")
    return fixed, n


def main():
    print(f"\nAPI key   : {'set' if API_KEY else 'NOT SET'}")
    print(f"Inst token: {'set ✓' if INST_TOKEN else 'NOT SET'}")

    if not API_KEY:
        print("\nERROR: SCOPUS_API_KEY not set in .env")
        return

    # ab_recovered, ab_n = test_abstracts()   # confirmed working — 5/5
    # ft_recovered, ft_n = test_fulltext()    # confirmed working — 5/5 for 10.1016
    # core_found, core_n = test_core()        # 0/5 on old papers — may help on newer
    fixed, fix_n = test_browser_headers()

    print_header("SUMMARY")
    # print(f"  Abstracts (Elsevier token) : {ab_recovered}/{ab_n} unlocked")
    # print(f"  Full texts (Elsevier token): {ft_recovered}/{ft_n} unlocked")
    # print(f"  Full texts (CORE.ac.uk)    : {core_found}/{core_n} found")
    print(f"  MDPI/Frontiers header fix  : {fixed}/{fix_n} downloads unblocked")
    print()


if __name__ == "__main__":
    main()
