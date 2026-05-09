# Database Search Summary

Last updated: 2026-05-09 (IDB updated)

## Academic Databases

| Database | Records | Abstracts | Full Texts (PDFs) | Notes |
|---|---|---|---|---|
| Web of Science (wos) | 15,179 | 15,170 | No | 9 records missing abstracts |
| CABI (cab) | 5,723 | 5,723 | No | |
| EconLit (econlit) | 479 | 478 | No | 1 record missing abstract |
| ProQuest (proq) | 368 | 367 | No | 1 record missing abstract |
| ASP (asp) | 1,187 | 1,187 | No | |
| AGRIS (agris) | 3 | 3 | No | |
| Google Scholar (gsch) | — | No | No | CSV export, no RIS — abstracts not captured |
| DuckDuckGo (ddg) | — | — | 3 PDFs | No RIS — PDFs downloaded manually |
| **Academic subtotal** | **22,939** | **22,928** | | |

## Organizational Website Searches

### Development Agencies

| Organization | Records | Abstracts | Full Texts (PDFs) | Notes |
|---|---|---|---|---|
| World Bank | 28 | 28 | 10 PDFs | Abstracts fetched via DSpace API; 1 bad record removed; PDFs saved where available |
| GCF | 157 | 157 | No | HTML snapshots only, no PDFs |
| GEF | 7 | 7 | 7 PDFs | |
| ADB | 16 | 16 | 16 PDFs | |
| AfDB | 5 | 5 | 7 PDFs | |
| IDB | 126 | 111 | 100 PDFs | Expanded from 70 to 126 records; abstracts extracted from PDFs; 15 short factsheets without extractable abstracts |
| FCDO | 2 | 2 | 2 PDFs | RIS created manually |
| USAID | 0 | — | — | DEC taken offline in 2025 |

### UN Agencies

| Organization | Records | Abstracts | Full Texts (PDFs) | Notes |
|---|---|---|---|---|
| FAO | 2 | 2 | No | |
| IFAD | 16 | 16 | No | |
| UNDP | 9 | 9 | No | |
| UNEP | 2 | 1 | No | 1 record missing abstract |
| UNFCCC | 30 | 28 | 29 PDFs | 2 records missing abstracts — PDF not found or no abstract section |

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
| WASP | 6 | 6 | 6 PDFs | |
| 3ie | 5 | 2 | No | 3 records missing abstracts |
| Campbell Collaboration | 5 | 0 | No | Abstracts not captured |
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
| Org website searches | 435 | 327 | 146 |
| **Grand total** | **23,374** | **23,255** | **146** |

## Records Missing Abstracts (Priority to Fix)

| Source | Missing Abstracts | Action |
|---|---|---|
| IDB | 70 | Need to fetch from web — Cloudflare blocks automation, requires manual browser download |
| Campbell Collaboration | 5 | Need to fetch PDFs |
| 3ie | 3 | Need to fetch PDFs |
| UNEP | 1 | Need to fetch PDF |
| UNFCCC | 2 | PDF not found or no abstract section |
| WoS | 9 | Minor — already near complete |
| EconLit | 1 | Minor |
| ProQuest | 1 | Minor |

## Notes
- Abstract extraction script: `extract_abstracts.py` — patches AB fields from PDFs into RIS files
- PDFs attached in Zotero are in `files/` subfolders where present
- NGOs and some M&E networks (CLEAR, IPA) not yet searched — folders created and ready
- Deduplication not yet run across all sources
