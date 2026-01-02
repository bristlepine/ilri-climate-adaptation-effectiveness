#!/usr/bin/env python3
"""
Fetch abstract for ONE DOI using Elsevier Article Retrieval API (META_ABS view).
Reads SCOPUS_API_KEY and optional SCOPUS_INST_TOKEN from .env.

Run:
  python scripts/fetch_elsevier_abstract_one.py
"""

import os
import json
import re
import requests
from urllib.parse import quote
from dotenv import load_dotenv

# ---- HARD-CODED DOI (your case) ----
DOI = "10.1016/j.crm.2017.06.001"

ELSEVIER_ARTICLE_URL = "https://api.elsevier.com/content/article/"

def find_in_obj(o, wanted=("dc:description", "abstract", "description")):
    if isinstance(o, dict):
        for k, v in o.items():
            if k in wanted and isinstance(v, str) and v.strip():
                return v.strip()
            got = find_in_obj(v, wanted)
            if got:
                return got
    elif isinstance(o, list):
        for it in o:
            got = find_in_obj(it, wanted)
            if got:
                return got
    return None

def main():
    load_dotenv()  # loads .env from repo root if present

    api_key = os.getenv("SCOPUS_API_KEY") or os.getenv("ELSEVIER_API_KEY")
    inst_token = os.getenv("SCOPUS_INST_TOKEN")

    if not api_key:
        raise SystemExit("Missing SCOPUS_API_KEY (or ELSEVIER_API_KEY) in .env")

    url = ELSEVIER_ARTICLE_URL + "doi/" + quote(DOI, safe="")
    headers = {
        "Accept": "application/json",
        "X-ELS-APIKey": api_key,
        "User-Agent": "ilri-climate-adaptation-effectiveness (one-doi abstract fetch)",
    }
    if inst_token:
        headers["X-ELS-Insttoken"] = inst_token

    params = {
        "view": "META_ABS",
        "httpAccept": "application/json",
    }

    r = requests.get(url, headers=headers, params=params, timeout=60)
    print("Request URL:", r.url)
    print("HTTP:", r.status_code)

    if r.status_code != 200:
        print("Body (first 600 chars):")
        print(r.text[:600])
        raise SystemExit(1)

    data = r.json()

    # Try to find title + abstract anywhere in the JSON
    title = find_in_obj(data, wanted=("dc:title", "title"))
    abstract = find_in_obj(data, wanted=("dc:description", "abstract", "description"))

    print("\nTITLE:\n", title or "(none)")
    print("\nABSTRACT:\n", abstract or "(none found)")

    # Optional: dump the whole response for debugging
    # with open("elsevier_response.json", "w", encoding="utf-8") as f:
    #     json.dump(data, f, indent=2)

if __name__ == "__main__":
    main()
