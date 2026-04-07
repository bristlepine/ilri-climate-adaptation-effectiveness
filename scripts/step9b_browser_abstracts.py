"""
step9b_browser_abstracts.py

Step 9b: Browser-based abstract scraping for publishers that block
regular HTTP requests (Cloudflare, bot detection, JS-rendered content).

Targets records still missing abstracts after step9 and step9a.
Uses Playwright (headless Chromium) to render pages as a real browser would.

Supported publishers (auto-detected from DOI prefix):
  - Taylor & Francis  (10.1080)
  - Springer          (10.1007)
  - Wiley             (10.1002, 10.1111)
  - AIP Publishing    (10.1063)
  - World Scientific  (10.1142)
  - Generic fallback  (any DOI with a landing page)

Outputs:
  - Updates scripts/outputs/step9/step9_scopus_enriched.csv in place
  - Writes scripts/outputs/step9b/step9b_summary.json
  - Writes scripts/outputs/step9b/step9b_recovered.csv

Usage:
    python scripts/step9b_browser_abstracts.py

Requirements:
    pip install playwright
    playwright install chromium
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# Settings
# =============================================================================

HERE      = Path(__file__).resolve().parent
OUT_ROOT  = HERE / "outputs"
STEP9_CSV = OUT_ROOT / "step9" / "step9_scopus_enriched.csv"
OUT_DIR   = OUT_ROOT / "step9b"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PAGE_TIMEOUT  = 15_000   # ms — max wait for page load
SLEEP_MS      = 1_500    # ms — polite delay between pages
MAX_RECORDS   = 10       # set to None to run all

# CSS selectors to try for abstract text, by publisher pattern
ABSTRACT_SELECTORS = [
    # Taylor & Francis
    ".abstractSection p",
    ".abstract-group p",
    # Springer / Nature
    "#Abs1-content p",
    ".c-article-section__content p",
    # Wiley
    ".article-section__content p",
    # AIP / generic
    ".abstract p",
    "[class*='abstract'] p",
    "section.abstract p",
    # Meta tag fallback (many publishers)
]

META_SELECTORS = [
    'meta[name="description"]',
    'meta[name="DC.Description"]',
    'meta[name="citation_abstract"]',
    'meta[property="og:description"]',
]

# DOI prefix → landing page URL template
def doi_to_url(doi: str) -> Optional[str]:
    doi = doi.strip()
    if not doi:
        return None
    prefix = doi.split("/")[0] if "/" in doi else ""
    # Publisher-specific URLs
    if prefix == "10.1080":
        return f"https://www.tandfonline.com/doi/abs/{doi}"
    if prefix == "10.1007":
        return f"https://link.springer.com/article/{doi}"
    if prefix in ("10.1002", "10.1111"):
        return f"https://onlinelibrary.wiley.com/doi/{doi}"
    if prefix == "10.1063":
        return f"https://pubs.aip.org/aip/apl/article-abstract/{doi}"
    if prefix == "10.1142":
        return f"https://www.worldscientific.com/doi/abs/{doi}"
    # Generic DOI resolver fallback
    return f"https://doi.org/{doi}"


# =============================================================================
# Browser scraping
# =============================================================================

async def scrape_abstract(page, url: str) -> Optional[str]:
    """Navigate to URL and extract abstract text."""
    try:
        await page.goto(url, timeout=PAGE_TIMEOUT, wait_until="domcontentloaded")
        await page.wait_for_timeout(1500)  # let JS render

        # Try CSS selectors
        for sel in ABSTRACT_SELECTORS:
            try:
                els = await page.query_selector_all(sel)
                if els:
                    texts = [await el.inner_text() for el in els]
                    combined = " ".join(t.strip() for t in texts if t.strip())
                    if len(combined) > 80:
                        return combined
            except Exception:
                continue

        # Try meta tags
        for sel in META_SELECTORS:
            try:
                el = await page.query_selector(sel)
                if el:
                    content = await el.get_attribute("content")
                    if content and len(content.strip()) > 80:
                        return content.strip()
            except Exception:
                continue

    except Exception as e:
        pass

    return None


async def run_browser_scraping(missing_df: pd.DataFrame) -> list[dict]:
    from playwright.async_api import async_playwright

    results = []
    n = len(missing_df)
    recovered = 0

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            locale="en-US",
            viewport={"width": 1280, "height": 800},
        )
        page = await context.new_page()

        for i, (_, row) in enumerate(missing_df.iterrows()):
            doi   = str(row.get("doi", "") or "").strip()
            sid   = str(row.get("scopus_id", "") or "").strip()
            title = str(row.get("title", "") or "")[:60]

            url = doi_to_url(doi)
            if not url:
                print(f"  [{i+1}/{n}] No URL for: {title}... — skipping")
                results.append({"doi": doi, "scopus_id": sid, "abstract": None, "source": "no_url"})
                continue

            print(f"  [{i+1}/{n}] {title}...")
            abstract = await scrape_abstract(page, url)

            if abstract:
                recovered += 1
                print(f"    ✓ Got abstract ({len(abstract)} chars)")
            else:
                print(f"    ✗ No abstract found")

            results.append({
                "doi": doi,
                "scopus_id": sid,
                "title": str(row.get("title", "")),
                "abstract": abstract,
                "url": url,
                "source": "browser" if abstract else "browser_failed",
            })

            await page.wait_for_timeout(SLEEP_MS)

        await browser.close()

    print(f"\n  Recovered {recovered}/{n} abstracts via browser")
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    if not STEP9_CSV.exists():
        print(f"ERROR: {STEP9_CSV} not found. Run step9 first.")
        return

    df = pd.read_csv(STEP9_CSV, dtype=str, keep_default_na=False)
    missing = df[df["abstract"].str.strip() == ""].copy()
    print(f"\nStep 9b: Browser-based abstract scraping")
    print(f"  Total records  : {len(df):,}")
    print(f"  Missing abstracts: {len(missing):,}")

    if missing.empty:
        print("  Nothing to do — all abstracts present.")
        return

    if MAX_RECORDS:
        missing = missing.head(MAX_RECORDS)
        print(f"  Limited to first {MAX_RECORDS} records (MAX_RECORDS setting)")

    print(f"  Running browser scraper on {len(missing):,} records...\n")
    t0 = time.time()

    results = asyncio.run(run_browser_scraping(missing))

    # Update main CSV
    recovered_df = pd.DataFrame([r for r in results if r.get("abstract")])
    n_recovered = len(recovered_df)

    if n_recovered > 0:
        # Update abstract column in main df
        doi_to_abstract = {r["doi"]: r["abstract"] for r in results if r.get("abstract")}
        sid_to_abstract = {r["scopus_id"]: r["abstract"] for r in results if r.get("abstract") and r.get("scopus_id")}

        def fill_abstract(row):
            if row["abstract"].strip():
                return row["abstract"]
            ab = doi_to_abstract.get(row.get("doi", ""), "")
            if not ab:
                ab = sid_to_abstract.get(row.get("scopus_id", ""), "")
            return ab

        df["abstract"] = df.apply(fill_abstract, axis=1)
        df.to_csv(STEP9_CSV, index=False)
        print(f"\n  Updated: {STEP9_CSV}")

    # Write outputs
    elapsed = time.time() - t0
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_missing": len(missing),
        "recovered": n_recovered,
        "still_missing": len(missing) - n_recovered,
        "elapsed_hms": time.strftime("%H:%M:%S", time.gmtime(elapsed)),
    }

    (OUT_DIR / "step9b_summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(results).to_csv(OUT_DIR / "step9b_results.csv", index=False)

    print(f"\n  Summary: {n_recovered}/{len(missing)} recovered in {summary['elapsed_hms']}")
    print(f"  Wrote: {OUT_DIR / 'step9b_summary.json'}")
    print(f"  Wrote: {OUT_DIR / 'step9b_results.csv'}")


if __name__ == "__main__":
    main()
