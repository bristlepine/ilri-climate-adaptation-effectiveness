# Database Search Summary

Last updated: 2026-05-09

## Academic Databases

| Database | Records | Abstracts | Full Texts (PDFs) | Notes |
|---|---|---|---|---|
| Web of Science (wos) | 15,179 | 15,170 | No | 9 records missing abstracts |
| CABI (cab) | 5,723 | 5,723 | No | |
| EconLit (econlit) | 479 | 478 | No | 1 record missing abstract |
| ProQuest (proq) | 368 | 367 | No | 1 record missing abstract |
| ASP (asp) | 1,187 | 1,187 | No | |
| AGRIS (agris) | 3 | 3 | No | |
| Google Scholar (gsch) | — | — | No | CSV export, no RIS — abstracts not captured |
| DuckDuckGo (ddg) | — | — | 3 PDFs | No RIS — PDFs downloaded manually |
| **Academic subtotal** | **22,939** | **22,928** | | |

## Organizational Website Searches

### Development Agencies

| Organization | Records | Abstracts | Full Texts (PDFs) | Notes |
|---|---|---|---|---|
| World Bank | 28 | 28 | 21 PDFs | Abstracts fetched via DSpace API; PDFs saved where available |
| GCF | 157 | 157 | No | HTML snapshots only, no PDFs |
| GEF | 7 | 7 | 7 PDFs | |
| ADB | 16 | 16 | 16 PDFs | |
| AfDB | 5 | 5 | 7 PDFs | |
| IDB | 126 | 111 | 100 PDFs | 15 short factsheets without extractable abstracts |
| FCDO | 2 | 2 | 2 PDFs | RIS created manually |
| USAID | 0 | — | — | DEC taken offline in 2025 |

### UN Agencies

| Organization | Records | Abstracts | Full Texts (PDFs) | Notes |
|---|---|---|---|---|
| FAO | 2 | 2 | No | |
| IFAD | 16 | 16 | No | |
| UNDP | 9 | 9 | No | |
| UNEP | 2 | 2 | 1 PDF | |
| UNFCCC | 30 | 28 | 29 PDFs | 2 records missing abstracts — 1 is a presentation (no abstract section), 1 URL is 404 |

### International Research Centers (IRCs)

| Organization | Records | Abstracts | Full Texts (PDFs) | Notes |
|---|---|---|---|---|
| CGSpace (CGIAR: CIMMYT, CIAT, ICARDA, IFPRI, ILRI, WorldFish) | 65 | 65 | 69 PDFs | RIS created manually |
| IPAM | 7 | 7 | 7 PDFs | RIS created manually |
| Adaptation Research Alliance (ARA) | 5 | 5 | No | |
| GCA | 3 | 3 | 3 PDFs | RIS created manually |

### M&E Networks

| Organization | Records | Abstracts | Full Texts (PDFs) | Notes |
|---|---|---|---|---|
| WASP | 6 | 6 | 6 PDFs | PDFs restored from git history into flat wasp/ folder |
| 3ie | 5 | 5 | No | |
| Campbell Collaboration | 5 | 3 | No | 1 JS-rendered page (no extractable text), 1 bad record (Google search URL) |
| J-PAL | 0 | — | — | No relevant results |
| CLEAR | 0 | — | — | Not yet searched |
| IPA | 0 | — | — | Not yet searched |

### NGOs

| Organization | Records | Abstracts | Full Texts (PDFs) | Notes |
|---|---|---|---|---|
| CARE | 0 | — | — | Not yet searched |
| Oxfam | 0 | — | — | Not yet searched |
| Mercy Corps | 0 | — | — | Not yet searched |
| Practical Action | 0 | — | — | Not yet searched |
| World Vision | 0 | — | — | Not yet searched |

## Totals

| Category | Records | With Abstracts | With PDFs |
|---|---|---|---|
| Academic databases | 22,939 | 22,928 | 0 |
| Org website searches | 490 | 471 | 241 |
| **Grand total** | **23,429** | **23,399** | **241** |

## Records Missing Abstracts

| Source | Missing | Notes |
|---|---|---|
| IDB | 15 | Short factsheets — no abstract section in PDF |
| Campbell | 2 | 1 JS-rendered page, 1 bad record |
| UNFCCC | 2 | 1 presentation with no text, 1 URL is 404 |
| WoS | 9 | Minor |
| EconLit | 1 | Minor |
| ProQuest | 1 | Minor |

## Notes
- Abstract extraction scripts: `extract_abstracts.py` (local PDFs), `fetch_abstracts_web.py` (World Bank DSpace API), `patch_missing_abstracts.py` (web scraping for 3ie, Campbell, UNEP)
- UNFCCC PDFs are in `unfccc/files/` subdirectories (Zotero structure)
- NGOs and some M&E networks (CLEAR, IPA) not yet searched
- Deduplication not yet run across all sources
