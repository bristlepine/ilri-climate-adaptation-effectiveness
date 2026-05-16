# Database Search Summary

Last updated: 2026-05-09

All sources listed below are required by the Systematic Map Protocol (D3, Section 3). Records without recoverable abstracts have been removed from RIS files. Folders for sources not named in the protocol have been deleted.

After deduplication (step2b) and multi-pass abstract recovery (CrossRef, OpenAlex, user-retrieved RIS/PDFs), all 9,152 net-new records entering screening have 100% abstract coverage.

## Protocol Compliance Table

### Bibliographic Databases (Protocol §3.1)

| Source | Folder | Records | Abstracts | PDFs | Notes |
|---|---|---|---|---|---|
| Scopus | — | — | — | — | Processed separately; forms primary Scopus corpus |
| Web of Science | `wos` | 15,170 | 15,170 | — | 9 unrecoverable records removed |
| CAB Abstracts | `cab` | 5,723 | 5,723 | — | |
| AGRIS | `agris` | 3 | 3 | — | |
| Academic Search Premier | `asp` | 1,187 | 1,187 | — | |
| EconLit | `econlit` | 478 | 478 | — | 1 unrecoverable record removed |
| ProQuest | `proq` | 367 | 367 | — | 1 unrecoverable record removed |

### Web Search Engines (Protocol §3.2)

| Source | Folder | Records | Abstracts | PDFs | Notes |
|---|---|---|---|---|---|
| Google Scholar | `gsch` | 193 | 193 | — | RIS created from CSV export; 5 unrecoverable records removed |
| DuckDuckGo | `ddg` | 3 | 3 | 3 | RIS created from PDFs |

### Organizational Websites — UN Agencies (Protocol §3.3)

| Source | Folder | Records | Abstracts | PDFs | Notes |
|---|---|---|---|---|---|
| FAO | `fao` | 2 | 2 | — | |
| IFAD | `ifad` | 16 | 16 | — | |
| UNDP | `undp` | 9 | 9 | — | |
| UNEP | `unep` | 2 | 2 | 1 | |
| UNFCCC | `unfccc` | 28 | 28 | 29 | 2 unrecoverable records removed (1 dead URL, 1 presentation with no text) |

### Organizational Websites — Development Agencies (Protocol §3.3)

| Source | Folder | Records | Abstracts | PDFs | Notes |
|---|---|---|---|---|---|
| World Bank | `worldbank` | 28 | 28 | 21 | Abstracts via DSpace 7 REST API |
| GEF | `gef` | 7 | 7 | 7 | |
| GCF | `gcf` | 157 | 157 | — | HTML snapshots; no PDFs |
| IDB | `idb` | 123 | 123 | 100 | 15 short factsheets removed (no extractable abstract section); 3 further unrecoverable records removed |
| ADB | `adb` | 16 | 16 | 16 | |
| AfDB | `afdb` | 5 | 5 | 7 | |
| FCDO | `fcdo` | 2 | 2 | 2 | RIS created manually |
| USAID | — | 0 | — | — | DEC taken offline in 2025; search attempted, not searchable |

### Organizational Websites — International Research Centers (Protocol §3.3)

| Source | Folder | Records | Abstracts | PDFs | Notes |
|---|---|---|---|---|---|
| CGSpace (CGIAR) | `cgspace` | 65 | 65 | 69 | CIMMYT, CIAT, ICARDA, IFPRI, ILRI, WorldFish; RIS created manually |
| IPAM | `ipam` | 7 | 7 | 7 | RIS created manually |
| Adaptation Research Alliance | `ara` | 5 | 5 | — | |
| GCA | `gca` | 3 | 3 | 3 | RIS created manually |
| WASP | `wasp` | 6 | 6 | 6 | |

### Organizational Websites — Evaluation and M&E Networks (Protocol §3.3)

| Source | Folder | Records | Abstracts | PDFs | Notes |
|---|---|---|---|---|---|
| 3ie | `3ie` | 5 | 5 | — | |
| Campbell Collaboration | `campbell` | 3 | 3 | — | 2 unrecoverable records removed (1 JS-rendered, 1 bad Google search URL) |
| J-PAL | `jpal` | 0 | — | — | Searched; no relevant results |

## Totals

| Category | Records | With Abstracts | With PDFs |
|---|---|---|---|
| Bibliographic databases (excl. Scopus) | 22,928 | 22,928 | — |
| Web search engines | 196 | 196 | 3 |
| Organizational websites | 489 | 489 | 241 |
| **Grand total (excl. Scopus)** | **23,613** | **23,613** | **244** |

All records entering screening have abstracts (100% coverage). Unrecoverable records removed at source before deduplication.

## Notes
- Abstract extraction scripts: `extract_abstracts.py` (local PDFs), `fetch_abstracts_web.py` (World Bank DSpace API), `patch_missing_abstracts.py` (3ie, Campbell, UNEP, UNFCCC, WoS, EconLit, ProQ)
- Abstract recovery for net-new records: CrossRef API, OpenAlex API, user-retrieved RIS/PDFs
- After deduplication (step2b): 9,152 net-new records, all with abstracts
