"""
Test alternative access routes for open-access publishers blocked by Cloudflare.
Tests MDPI (10.3390), Frontiers (10.3389), and Europe PMC as a mirror.
"""
import requests

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/pdf,*/*",
}

def test(label, url):
    try:
        r = requests.get(url, headers=headers, timeout=12)
        ct = r.headers.get("Content-Type", "")[:50]
        size = len(r.content)
        status = "✓ accessible" if r.status_code == 200 and size > 1000 else "✗ blocked/empty"
        print(f"  {label}: HTTP {r.status_code} | {size:,} bytes | {status}")
        if r.status_code == 200 and "json" in ct:
            data = r.json()
            for key in ["downloadUrl", "pdfUrl", "openAccessPdf"]:
                if data.get(key):
                    print(f"    → {key}: {data[key]}")
    except Exception as e:
        print(f"  {label}: ERROR {e}")

print("\n--- MDPI paper (10.3390) ---")
mdpi_doi = "10.3390/cli4040063"
test("MDPI PDF",       "https://www.mdpi.com/2225-1154/4/4/63/pdf")
test("Europe PMC",     f"https://europepmc.org/api/search?query=DOI:{mdpi_doi}&format=json&resulttype=core")
test("CORE",           f"https://api.core.ac.uk/v3/works/doi:{mdpi_doi}")
test("PubMed Central", f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={mdpi_doi}")

print("\n--- Frontiers paper (10.3389) ---")
front_doi = "10.3389/fsufs.2021.622026"  # sample Frontiers paper from failed list
import pandas as pd
df = pd.read_csv("scripts/outputs/step13/step13_manifest.csv", dtype=str, keep_default_na=False)
front_sample = df[(df["status"]=="needs_manual") & df["doi"].str.startswith("10.3389", na=False)].head(1)
if not front_sample.empty:
    front_doi = front_sample.iloc[0]["doi"]
    print(f"  Using: {front_doi}")

test("Frontiers PDF",  f"https://www.frontiersin.org/articles/{front_doi}/pdf")
test("Europe PMC",     f"https://europepmc.org/api/search?query=DOI:{front_doi}&format=json&resulttype=core")
test("CORE",           f"https://api.core.ac.uk/v3/works/doi:{front_doi}")
test("Unpaywall",      f"https://api.unpaywall.org/v2/{front_doi}?email=zarrar85@gmail.com")

