# Full-Text Retrieval — Cornell Campus Run

Run this from Cornell campus or Cornell VPN. Downloads missing full texts into `retrieved/`.

---

## How to run

- Connect to Cornell VPN if off campus (Cisco AnyConnect → `vpn.cornell.edu`). On-campus physical connection gets slightly more papers — some publishers (Wiley) block VPN traffic. VPN is fine for a first pass.
- Open Terminal
- Go to this folder:
  ```
  cd /path/to/missing_papers
  ```
- Install the one dependency if you haven't already:
  ```
  pip install requests
  ```
- Run:
  ```
  python3 get_missing.py
  ```
- You'll see progress printed for each paper — `✓ retrieved` or `✗ not retrieved`
- Safe to stop and restart at any time — it skips anything already downloaded

---

## The .env file

Zarrar will include a `.env` file in this folder when he sends it to you. It contains two API keys that unlock an extra retrieval source (Elsevier by Scopus ID) for the 958 papers that have no DOI. The script reads it automatically — you don't need to do anything with it.

If the `.env` is missing the script still runs and handles all DOI-based records normally.

---

## What it does

- **2,616 papers have a DOI** — tried via Unpaywall, Semantic Scholar, then publisher-direct URLs (Springer, Taylor & Francis, Wiley, Emerald, SAGE, MDPI, and others). Campus IP unlocks the paywalled ones.
- **958 papers have no DOI** — tried via Elsevier API (Scopus ID), then Semantic Scholar title search, then CORE. Most are older regional journals; some will not be retrievable.

---

## When done

- All files are in `retrieved/` inside this folder
- `retrieval_meta.json` has a summary of what worked and what didn't
- Zip `retrieved/` and send to Zarrar:
  ```
  zip -r retrieved.zip retrieved/
  ```

---

## Notes

- Runs at ~1 paper per 1.2 seconds — expect 1.5–2 hours total
- Files over 30 MB are skipped
- Already-downloaded files are skipped automatically on re-run

---

## Current numbers and expected outcome

- 6,218 papers included total; 2,644 (42.5%) already retrieved automatically
- This script targets the remaining 3,574 — 2,616 have DOIs, 958 do not
- Expected recovery: ~1,500–1,800 DOI papers via campus access + up to ~200 no-DOI papers via Elsevier/title search
- Projected total after this run: ~4,200–4,500 retrieved (68–72%)
- Some no-DOI papers are older regional journals with no online version — these are expected failures and are not discarded
