#!/usr/bin/env python3
"""
scopus.py

Step 1 (summary-only):
  - Build subgroup queries + element aggregates + TOTAL combined query from scripts/search_strings.yml
  - Fetch ONLY record counts (opensearch:totalResults) for each query (no record retrieval)
  - Cache per-query counts so reruns can SKIP unchanged queries
  - Write:
      outputs/step1/step1_summary.csv
      outputs/step1/step1_hits_plot.png (+ outputs/step1/step1_hits_plot.csv)
      outputs/step1/step1_queries.json
      outputs/step1/step1_total_query.txt
      outputs/step1/step1_counts_cache.json

Step 2 (retrieve TOTAL__ALL records; STREAMING, no cursor):
  - Scopus deep paging has a practical start-based limit, so we avoid it by slicing the TOTAL query into
    PUBYEAR ranges where each slice has <= DEEP_PAGING_LIMIT results.
  - Also includes a final "PUBYEAR_MISSING_OR_OUTSIDE" slice for records with missing/odd years.
  - Streams each slice to CSV as it arrives (constant memory).
  - Optionally writes per-slice state JSON for resumability.
  - Produces a final combined CSV:
      outputs/step2/step2_total_records.csv
    by concatenating all slice CSVs.

Step 3 (benchmark DOI match against Step 2 output):
  - Parse DOIs from scripts/Benchmark List - List.csv (Study column)
  - Match against DOIs retrieved in outputs/step2/step2_total_records.csv
  - Write:
      outputs/step3/step3_benchmark_scopus_match.csv

Run:
  python scripts/scopus.py
"""

import os
import re
import json
import time
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================================================
# Initial Setup (edit these)
# ============================================================
RUN_STEP1 = 1  # Step 1: counts only (uses API)
RUN_STEP2 = 1  # Step 2: retrieve TOTAL__ALL (uses API)
RUN_STEP3 = 1  # Step 3: benchmark DOI match (no API)

# Step 1 controls
SKIP_EXISTING_STEP1 = 1   # if cached counts exist and query unchanged, don't call API
USE_POST_FOR_SEARCH = 1   # avoids 413 for long queries
COUNT_PER_PAGE = 1        # count-only requests
SLEEP_S = 0.10
VIEW_STEP1 = "STANDARD"   # counts only


# Step 2 controls
MAX_RESULTS_TOTAL = None      # None = retrieve all; or set e.g. 500 to test
COUNT_PER_PAGE_RETRIEVE = 25  # commonly 25 for this endpoint
SLEEP_S_RETRIEVE = 0.15

SKIP_EXISTING_STEP2 = 1       # Step 2: if outputs already exist, skip (delete outputs/step2 to rerun)

DEEP_PAGING_LIMIT = 5000
PUBYEAR_MIN = 1990
PUBYEAR_MAX = 2025
VIEW_STEP2 = "STANDARD"   # retrieval (try to include abstracts/keywords)

# Benchmark controls
HERE = os.path.dirname(os.path.abspath(__file__))
SEARCH_STRINGS_YML = os.path.join(HERE, "search_strings.yml")
BENCHMARK_CSV = os.path.join(HERE, "Benchmark List - List.csv")

# Out DIRs
OUT_DIR = os.path.join(HERE, "outputs")
STEP1_DIR = os.path.join(OUT_DIR, "step1")
STEP2_DIR = os.path.join(OUT_DIR, "step2")
STEP3_DIR = os.path.join(OUT_DIR, "step3")

# Step 1 outputs
STEP1_QUERIES_JSON = "step1_queries.json"
STEP1_SUMMARY_CSV = "step1_summary.csv"
STEP1_PLOT_PNG = "step1_hits_plot.png"
STEP1_TOTAL_QUERY_TXT = "step1_total_query.txt"
STEP1_CACHE_JSON = "step1_counts_cache.json"

# Step 2 outputs
STEP2_TOTAL_CSV = "step2_total_records.csv"
STEP2_TOTAL_META_JSON = "step2_total_records.meta.json"

# Step 3 outputs
STEP3_MATCH_CSV = "step3_benchmark_scopus_match.csv"
STEP3_SUMMARY_JSON = "step3_benchmark_scopus_match.summary.json"

# ============================================================
# Scopus
# ============================================================
SCOPUS_SEARCH_URL = "https://api.elsevier.com/content/search/scopus"
DOI_REGEX = re.compile(r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", re.IGNORECASE)


@dataclass
class ScopusAuth:
    api_key: str
    inst_token: Optional[str] = None


def _headers(auth: ScopusAuth) -> Dict[str, str]:
    h = {"X-ELS-APIKey": auth.api_key, "Accept": "application/json"}
    if auth.inst_token:
        h["X-ELS-Insttoken"] = auth.inst_token
    return h


def _read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _rate_headers(r: requests.Response) -> dict:
    keys = ["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"]
    return {k: r.headers.get(k) for k in keys if r.headers.get(k) is not None}


def _request_with_retries(
    session: requests.Session,
    method: str,
    url: str,
    headers: dict,
    params: dict,
    tries: int = 6,
) -> Tuple[dict, dict]:
    backoff = 1.0
    last_rate = {}

    for _ in range(tries):
        if method.upper() == "POST":
            r = session.post(url, headers=headers, data=params, timeout=60)
        else:
            r = session.get(url, headers=headers, params=params, timeout=60)

        last_rate = _rate_headers(r)

        if r.status_code == 200:
            return r.json(), last_rate

        if r.status_code == 429:
            reset = r.headers.get("X-RateLimit-Reset")
            if reset is not None:
                try:
                    wait_s = max(1.0, float(reset))
                except Exception:
                    wait_s = backoff
            else:
                wait_s = backoff
            time.sleep(wait_s)
            backoff = min(backoff * 2, 60.0)
            continue

        if r.status_code in (500, 502, 503, 504):
            time.sleep(backoff)
            backoff = min(backoff * 2, 60.0)
            continue

        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:800]}")

    raise RuntimeError(f"Failed after retries. Last rate headers: {last_rate}")


def scopus_count_only(auth: ScopusAuth, query: str, count_per_page: int = 1) -> Tuple[int, dict]:
    """Return (totalResults, meta) without retrieving records."""
    headers = _headers(auth)
    meta = {"query": query, "count_per_page": count_per_page, "view": VIEW_STEP1, "rate_headers_last": {}}

    with requests.Session() as session:
        params = {"query": query, "start": "0", "count": str(count_per_page), "view": VIEW_STEP1}
        method = "POST" if USE_POST_FOR_SEARCH else "GET"
        data, rate = _request_with_retries(session, method, SCOPUS_SEARCH_URL, headers, params)
        meta["rate_headers_last"] = rate

    sr = (data or {}).get("search-results", {}) or {}
    total = int(sr.get("opensearch:totalResults", "0"))
    return total, meta


def _extract_entry_row(e: dict) -> dict:
    """
    Scopus Search API entry fields.

    NOTE:
      - If you run Step 2 with view=COMPLETE, Scopus may include:
          * dc:description (often the abstract)
          * authkeywords
      - Some records will still have missing abstracts/keywords.
    """
    dcid = e.get("dc:identifier")
    scopus_id = (
        dcid.split(":", 1)[1].strip()
        if isinstance(dcid, str) and dcid.startswith("SCOPUS_ID:")
        else None
    )

    doi = e.get("prism:doi")
    doi = doi.strip() if isinstance(doi, str) else None

    # Abstract (often available when view=COMPLETE)
    abstract = e.get("dc:description")
    abstract = abstract.strip() if isinstance(abstract, str) else None

    # Author keywords can come as list or string depending on response
    authkw = e.get("authkeywords")
    if isinstance(authkw, list):
        authkeywords = "; ".join([str(x).strip() for x in authkw if str(x).strip()]) or None
    elif isinstance(authkw, str):
        authkeywords = authkw.strip() or None
    else:
        authkeywords = None

    # Some extra fields that are often handy for screening/triage
    pubyear = e.get("prism:coverDate")
    # (keep coverDate as-is; you can parse year later if needed)

    return {
        # Core identifiers
        "eid": e.get("eid"),
        "scopus_id": scopus_id,
        "doi": doi or None,

        # Bibliographic
        "title": e.get("dc:title"),
        "abstract": abstract,
        "authkeywords": authkeywords,
        "coverDate": e.get("prism:coverDate"),
        "publicationName": e.get("prism:publicationName"),
        "aggregationType": e.get("prism:aggregationType"),   # e.g., Journal, Book, Conference Proceeding
        "subtypeDescription": e.get("subtypeDescription"),   # e.g., Article, Review (if present)

        # Impact / links
        "citedby_count": e.get("citedby-count"),
        "openaccessFlag": e.get("openaccessFlag"),
        "prism_url": e.get("prism:url"),
    }

def _build_queries(cfg: dict) -> Tuple[Dict[str, str], Dict[str, str], str]:
    """
    Returns:
      subgroup_queries: { "P__smallholder_small_scale": "TITLE-ABS-KEY(...)" , ...}
      element_queries:  { "P__ALL": "TITLE-ABS-KEY((...) OR (...))", ... }
      combined_query:   "TITLE-ABS-KEY((Praw) AND (Craw) AND ...)"
    """
    field = cfg.get("field", "TITLE-ABS-KEY").strip()
    elements = cfg.get("elements", {}) or {}

    subgroup_queries: Dict[str, str] = {}
    element_queries: Dict[str, str] = {}
    element_raw_or: Dict[str, str] = {}

    for element_key, subgroups in elements.items():
        raw_exprs: List[str] = []
        for sub_key, expr in (subgroups or {}).items():
            expr = str(expr).strip()
            qname = f"{element_key}__{sub_key}"
            subgroup_queries[qname] = f"{field}({expr})"
            raw_exprs.append(f"({expr})")

        if raw_exprs:
            or_raw = " OR ".join(raw_exprs)
            element_raw_or[element_key] = f"({or_raw})"
            element_queries[f"{element_key}__ALL"] = f"{field}({element_raw_or[element_key]})"

    combined_raw = " AND ".join([f"({v})" for v in element_raw_or.values()])
    combined_query = f"{field}({combined_raw})" if combined_raw else ""

    return subgroup_queries, element_queries, combined_query


def _plot_counts(summary: pd.DataFrame, out_png: str, title: str) -> None:
    """
    Colors bars by top-level protocol bucket and annotates counts above bars.
    Expected top-level keys (before '__'): P, C_concept, C_context_climate, C_context_agriculture, M, TOTAL
    """
    import matplotlib.patches as mpatches

    plot_df = summary.copy()
    plot_df["total_results_num"] = pd.to_numeric(plot_df["total_results"], errors="coerce")
    plot_df = plot_df.dropna(subset=["total_results_num"]).copy()

    def group_of(qname: str) -> str:
        if qname == "TOTAL__ALL":
            return "TOTAL"
        return qname.split("__", 1)[0]

    plot_df["group"] = plot_df["query_name"].astype(str).apply(group_of)

    group_order = ["P", "C_concept", "C_context_climate", "C_context_agriculture", "M", "TOTAL"]
    plot_df["group_order"] = plot_df["group"].apply(lambda g: group_order.index(g) if g in group_order else 999)
    plot_df = plot_df.sort_values(["group_order", "query_name"]).reset_index(drop=True)

    color_map = {
        "P": "#1f77b4",
        "C_concept": "#ff7f0e",
        "C_context_climate": "#2ca02c",
        "C_context_agriculture": "#d62728",
        "M": "#9467bd",
        "TOTAL": "#111111",
    }
    legend_labels = {
        "P": "Population (P): Smallholder producers and marginalized subpopulations",
        "C_concept": "Concept (C): Adaptation processes and outcomes",
        "C_context_climate": "Context (C): Climate stressors and hazards",
        "C_context_agriculture": "Context (C): Agricultural systems",
        "M": "Methodological focus (M): Measurement and evaluation",
        "TOTAL": "TOTAL: Combined query",
    }

    labels = plot_df["query_name"].tolist()
    totals = plot_df["total_results_num"].astype(float).tolist()
    groups = plot_df["group"].tolist()
    colors = [color_map.get(g, "#7f7f7f") for g in groups]
    x = list(range(len(labels)))

    plt.figure(figsize=(max(14, 0.55 * len(labels)), 7))
    bars = plt.bar(x, totals, color=colors)

    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Scopus totalResults")
    plt.title(title)

    present = [g for g in group_order if g in set(groups)]
    handles = [mpatches.Patch(color=color_map[g], label=legend_labels.get(g, g)) for g in present]
    if handles:
        plt.legend(handles=handles, loc="upper left", frameon=True)

    y_max = max(totals) if totals else 0.0
    pad = max(1.0, y_max * 0.012)
    rotate_numbers = len(bars) > 18

    for rect, val in zip(bars, totals):
        plt.text(
            rect.get_x() + rect.get_width() / 2.0,
            rect.get_height() + pad,
            f"{int(round(val)):,}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90 if rotate_numbers else 0,
        )

    plt.ylim(0, y_max + 18 * pad)
    plt.tight_layout(rect=[0, 0, 1, 0.90])

    plt.savefig(out_png, dpi=200)
    plt.close()


def _queries_signature(queries: List[Tuple[str, str]]) -> str:
    payload = [{"name": n, "query": q} for n, q in queries]
    s = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(s).hexdigest()


def _query_hash(q: str) -> str:
    return hashlib.sha256(q.strip().encode("utf-8")).hexdigest()


def _load_json(path: str, default):
    try:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default


def _normalize_doi(x: str) -> str:
    d = str(x).strip().lower()
    for p in ("https://doi.org/", "http://doi.org/", "doi:"):
        if d.startswith(p):
            d = d[len(p):]
    return d


def _doi_from_text(s: str) -> Optional[str]:
    if not isinstance(s, str):
        return None
    m = DOI_REGEX.search(s)
    return _normalize_doi(m.group(1)) if m else None


def _with_pubyear_range(base_query: str, y0: int, y1: int) -> str:
    # inclusive range
    return f"({base_query}) AND PUBYEAR > {y0 - 1} AND PUBYEAR < {y1 + 1}"


def _with_pubyear_missing_or_outside(base_query: str, y0: int, y1: int) -> str:
    # Captures records not in the inclusive year window (often missing/odd PUBYEAR)
    return f"({base_query}) AND NOT (PUBYEAR > {y0 - 1} AND PUBYEAR < {y1 + 1})"


def _plan_year_slices(auth: ScopusAuth, base_query: str, y0: int, y1: int) -> List[Tuple[int, int, int]]:
    """
    Recursively split [y0,y1] until each slice count <= DEEP_PAGING_LIMIT.
    Returns list of (start_year, end_year, total_results_for_slice).
    """
    total, _ = scopus_count_only(auth, _with_pubyear_range(base_query, y0, y1), count_per_page=1)
    if total <= DEEP_PAGING_LIMIT or y0 == y1:
        return [(y0, y1, total)]

    mid = (y0 + y1) // 2
    left = _plan_year_slices(auth, base_query, y0, mid)
    right = _plan_year_slices(auth, base_query, mid + 1, y1)
    return left + right


def _safe_slice_tag(y0: int, y1: int) -> str:
    return f"PUBYEAR_{y0}_{y1}"


def _concat_csvs(csv_paths: List[str], out_csv: str) -> int:
    """Concatenate multiple CSVs with identical headers into out_csv. Returns row count (excluding header)."""
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # Remove output if exists
    if os.path.exists(out_csv):
        os.remove(out_csv)

    wrote_header = False
    total_rows = 0

    with open(out_csv, "w", encoding="utf-8", newline="") as out_f:
        for p in csv_paths:
            if not os.path.exists(p) or os.path.getsize(p) == 0:
                continue
            with open(p, "r", encoding="utf-8", errors="ignore") as in_f:
                for i, line in enumerate(in_f):
                    if i == 0:
                        if wrote_header:
                            continue
                        wrote_header = True
                        out_f.write(line)
                        continue
                    out_f.write(line)
                    total_rows += 1

    return total_rows


def scopus_retrieve_stream_to_csv_start(
    auth: ScopusAuth,
    query: str,
    out_csv: str,
    count_per_page: int,
    sleep_s: float,
    max_results: Optional[int],
) -> Tuple[int, int, dict]:
    """
    Stream retrieval using start-based paging (NO cursor, NO resume):
      - overwrites out_csv for a clean slice download
      - streams page-by-page (constant memory)

    Returns: (total_reported, retrieved_rows, meta)
    """
    headers = _headers(auth)
    method = "POST" if USE_POST_FOR_SEARCH else "GET"

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # Always overwrite slice output for a clean run of that slice
    if os.path.exists(out_csv):
        os.remove(out_csv)

    retrieved = 0
    total_reported = None

    # ---- coverage counters (per-slice) ----
    n_with_abstract = 0
    n_with_authkeywords = 0
    n_with_doi = 0
    n_with_title = 0
    n_with_coverdate = 0
    n_with_pubname = 0

    meta = {
        "query": query,
        "view": VIEW_STEP2,
        "count_per_page": count_per_page,
        "sleep_s": sleep_s,
        "max_results": max_results,
        "rate_headers_last": {},
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    pbar = None

    with requests.Session() as session:
        page_n = 0

        while True:
            if max_results is not None and retrieved >= max_results:
                break

            params = {
                "query": query,
                "start": str(retrieved),
                "count": str(count_per_page),
                "view": VIEW_STEP2,
            }

            data, rate = _request_with_retries(session, method, SCOPUS_SEARCH_URL, headers, params)
            meta["rate_headers_last"] = rate

            sr = (data or {}).get("search-results", {}) or {}

            if total_reported is None:
                total_reported = int(sr.get("opensearch:totalResults", "0"))
                meta["total_reported_by_scopus"] = total_reported

                pbar_total = total_reported if max_results is None else min(total_reported, max_results)
                pbar = tqdm(total=pbar_total, desc="[step2] Retrieving", unit="rec")

                print(f"[step2] Total reported by Scopus: {total_reported:,}")
                if rate:
                    print(f"[step2] Rate headers (initial): {rate}")

            entries = sr.get("entry", []) or []
            if not entries:
                break

            rows = [_extract_entry_row(e) for e in entries]

            # ---- update counters from rows (no extra API calls) ----
            for r in rows:
                if r.get("abstract"):
                    n_with_abstract += 1
                if r.get("authkeywords"):
                    n_with_authkeywords += 1
                if r.get("doi"):
                    n_with_doi += 1
                if r.get("title"):
                    n_with_title += 1
                if r.get("coverDate"):
                    n_with_coverdate += 1
                if r.get("publicationName"):
                    n_with_pubname += 1

            df = pd.DataFrame(rows)

            header_needed = not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0
            df.to_csv(out_csv, mode="a", header=header_needed, index=False)

            got = len(df)
            retrieved += got
            page_n += 1

            if pbar:
                pbar.update(got)

            if page_n % 10 == 0:
                print(f"[step2] progress: retrieved_total={retrieved:,} page={page_n}")
                if rate:
                    print(f"[step2] rate: {rate}")

            if total_reported is not None and retrieved >= total_reported:
                break

            if sleep_s > 0:
                time.sleep(sleep_s)

    if pbar:
        pbar.close()

    meta["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta["retrieved_rows"] = int(retrieved)
    meta["total_reported_by_scopus"] = int(total_reported or 0)

    # ---- attach coverage stats ----
    meta["coverage"] = {
        "with_title": int(n_with_title),
        "with_coverDate": int(n_with_coverdate),
        "with_publicationName": int(n_with_pubname),
        "with_doi": int(n_with_doi),
        "with_abstract": int(n_with_abstract),
        "with_authkeywords": int(n_with_authkeywords),
    }

    return int(total_reported or 0), int(retrieved), meta

# ============================================================
# Step 1: counts only (with per-query caching + SKIP logs)
# ============================================================
def step1_counts_only(auth: ScopusAuth) -> None:
    os.makedirs(STEP1_DIR, exist_ok=True)

    cfg = _read_yaml(SEARCH_STRINGS_YML)
    subgroup_q, element_q, combined_q = _build_queries(cfg)

    queries: List[Tuple[str, str]] = []
    queries += sorted(subgroup_q.items(), key=lambda x: x[0])
    queries += sorted(element_q.items(), key=lambda x: x[0])
    if combined_q:
        queries.append(("TOTAL__ALL", combined_q))

    queries_json_path = os.path.join(STEP1_DIR, STEP1_QUERIES_JSON)
    summary_csv_path = os.path.join(STEP1_DIR, STEP1_SUMMARY_CSV)
    plot_png_path = os.path.join(STEP1_DIR, STEP1_PLOT_PNG)
    plot_csv_path = os.path.splitext(plot_png_path)[0] + ".csv"
    total_query_txt_path = os.path.join(STEP1_DIR, STEP1_TOTAL_QUERY_TXT)
    cache_json_path = os.path.join(STEP1_DIR, STEP1_CACHE_JSON)

    if combined_q:
        with open(total_query_txt_path, "w", encoding="utf-8") as f:
            f.write(combined_q.strip() + "\n")

    sig = _queries_signature(queries)
    payload = [{"_signature": sig, "_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}]
    payload += [{"name": n, "query": q} for n, q in queries]
    with open(queries_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    cache = _load_json(cache_json_path, default={})
    if not isinstance(cache, dict):
        cache = {}

    rows = []
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    for qname, q in queries:
        qh = _query_hash(q)
        cached = cache.get(qname)

        if SKIP_EXISTING_STEP1 and isinstance(cached, dict) and cached.get("query_hash") == qh:
            print(f"[step1] SKIP  {qname} (cached)")
            rate = cached.get("rate_headers_last") or {}
            rows.append(
                {
                    "query_name": qname,
                    "total_results": cached.get("total_results"),
                    "timestamp_utc": cached.get("timestamp_utc", ts),
                    "error": cached.get("error"),
                    "rate_limit_limit": rate.get("X-RateLimit-Limit"),
                    "rate_limit_remaining": rate.get("X-RateLimit-Remaining"),
                    "rate_limit_reset": rate.get("X-RateLimit-Reset"),
                }
            )
            continue

        print(f"[step1] COUNT  {qname}")
        try:
            total, meta = scopus_count_only(auth, q, count_per_page=COUNT_PER_PAGE)
            err = None
        except Exception as ex:
            total = None
            meta = {"rate_headers_last": {}}
            err = str(ex)
            print(f"[step1] ERROR {qname}: {ex}")

        cache[qname] = {
            "query_hash": qh,
            "query": q,
            "total_results": total,
            "timestamp_utc": ts,
            "error": err,
            "rate_headers_last": meta.get("rate_headers_last") or {},
        }

        rate = meta.get("rate_headers_last") or {}
        rows.append(
            {
                "query_name": qname,
                "total_results": total,
                "timestamp_utc": ts,
                "error": err,
                "rate_limit_limit": rate.get("X-RateLimit-Limit"),
                "rate_limit_remaining": rate.get("X-RateLimit-Remaining"),
                "rate_limit_reset": rate.get("X-RateLimit-Reset"),
            }
        )

        if SLEEP_S > 0:
            time.sleep(SLEEP_S)

    with open(cache_json_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

    summary = pd.DataFrame(rows)
    summary.to_csv(summary_csv_path, index=False)
    summary.to_csv(plot_csv_path, index=False)

    print("\n[step1] Summary table:")
    cols = ["query_name", "total_results", "error"]
    print(summary[cols].to_string(index=False))

    _plot_counts(summary, plot_png_path, "Scopus record counts by query")


# ============================================================
# Step 2: retrieve TOTAL__ALL records (PUBYEAR slicing, NO cursor)
# ============================================================
def step2_retrieve_total(auth: ScopusAuth) -> None:
    os.makedirs(STEP2_DIR, exist_ok=True)

    # If you want a clean rerun, delete outputs/step2 yourself.
    # With SKIP_EXISTING_STEP2=1, we won't do any work if the final combined CSV already exists.
    final_csv = os.path.join(STEP2_DIR, STEP2_TOTAL_CSV)
    if SKIP_EXISTING_STEP2 and os.path.exists(final_csv) and os.path.getsize(final_csv) > 0:
        print(f"[step2] SKIP (exists): {final_csv}")
        return

    cfg = _read_yaml(SEARCH_STRINGS_YML)
    _, _, base_q = _build_queries(cfg)
    if not base_q:
        raise SystemExit("[step2] TOTAL__ALL query is empty; check search_strings.yml")

    base_total, _ = scopus_count_only(auth, base_q, count_per_page=1)
    print(f"[step2] TOTAL__ALL reported by Scopus: {base_total:,}")

    print("[step2] Planning PUBYEAR slices (to avoid deep paging limits)...")
    slices = _plan_year_slices(auth, base_q, PUBYEAR_MIN, PUBYEAR_MAX)

    # Missing/outside PUBYEAR slice (captures records with no PUBYEAR or outside the range)
    missing_q = _with_pubyear_missing_or_outside(base_q, PUBYEAR_MIN, PUBYEAR_MAX)
    missing_total, _ = scopus_count_only(auth, missing_q, count_per_page=1)

    planned_total = sum(t for _, _, t in slices) + int(missing_total)
    print("[step2] Slice plan:")
    for y0, y1, t in slices:
        print(f"  - {y0}-{y1}: {t:,}")
    print(f"  - PUBYEAR_MISSING_OR_OUTSIDE: {missing_total:,}")
    print(f"[step2] Planned total across slices: {planned_total:,}")

    if planned_total != base_total:
        print(f"[step2] WARNING: planned_total ({planned_total:,}) != base_total ({base_total:,}).")

    slice_csvs: List[str] = []
    slice_metas: List[dict] = []

    # ---- Year slices ----
    for y0, y1, slice_total in slices:
        tag = _safe_slice_tag(y0, y1)
        q_slice = _with_pubyear_range(base_q, y0, y1)

        out_csv = os.path.join(STEP2_DIR, f"step2_total_records__{tag}.csv")
        out_meta = os.path.join(STEP2_DIR, f"step2_total_records__{tag}.meta.json")

        if SKIP_EXISTING_STEP2 and os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
            print(f"[step2] SKIP {tag} (exists): {out_csv}")
            slice_csvs.append(out_csv)
            # try to load existing meta so coverage rollup stays correct
            if os.path.exists(out_meta) and os.path.getsize(out_meta) > 0:
                try:
                    with open(out_meta, "r", encoding="utf-8") as f:
                        slice_metas.append(json.load(f) or {})
                except Exception:
                    pass
            continue


        print(f"\n[step2] RETRIEVE {tag} (expected {slice_total:,})")
        total_reported, retrieved, meta = scopus_retrieve_stream_to_csv_start(
            auth,
            q_slice,
            out_csv=out_csv,
            count_per_page=COUNT_PER_PAGE_RETRIEVE,
            sleep_s=SLEEP_S_RETRIEVE,
            max_results=MAX_RESULTS_TOTAL,
        )

        meta.update(
            {
                "slice_tag": tag,
                "slice_pubyear_start": y0,
                "slice_pubyear_end": y1,
                "expected_total_from_planner": int(slice_total),
                "total_reported_by_scopus": int(total_reported),
                "retrieved_rows": int(retrieved),
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )
        with open(out_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(f"[step2] Done {tag}: reported={total_reported:,} retrieved={retrieved:,}")
        slice_csvs.append(out_csv)
        slice_metas.append(meta)

    # ---- Missing/outside slice ----
    if missing_total > 0:
        tag = "PUBYEAR_MISSING_OR_OUTSIDE"
        q_slice = missing_q

        out_csv = os.path.join(STEP2_DIR, f"step2_total_records__{tag}.csv")
        out_meta = os.path.join(STEP2_DIR, f"step2_total_records__{tag}.meta.json")

        if SKIP_EXISTING_STEP2 and os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
            print(f"[step2] SKIP {tag} (exists): {out_csv}")
            slice_csvs.append(out_csv)
        else:
            print(f"\n[step2] RETRIEVE {tag} (expected {missing_total:,})")
            total_reported, retrieved, meta = scopus_retrieve_stream_to_csv_start(
                auth,
                q_slice,
                out_csv=out_csv,
                count_per_page=COUNT_PER_PAGE_RETRIEVE,
                sleep_s=SLEEP_S_RETRIEVE,
                max_results=MAX_RESULTS_TOTAL,
            )

            meta.update(
                {
                    "slice_tag": tag,
                    "slice_pubyear_start": None,
                    "slice_pubyear_end": None,
                    "expected_total_from_planner": int(missing_total),
                    "total_reported_by_scopus": int(total_reported),
                    "retrieved_rows": int(retrieved),
                    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            )
            with open(out_meta, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            print(f"[step2] Done {tag}: reported={total_reported:,} retrieved={retrieved:,}")
            slice_csvs.append(out_csv)
            slice_metas.append(meta)

    # ---- Combine slices into one final CSV ----
    print(f"\n[step2] Combining slice CSVs -> {final_csv}")
    combined_rows = _concat_csvs(slice_csvs, final_csv)

    # ---- Roll up coverage across slices ----
    cov_total = {
        "with_title": 0,
        "with_coverDate": 0,
        "with_publicationName": 0,
        "with_doi": 0,
        "with_abstract": 0,
        "with_authkeywords": 0,
    }
    for m in slice_metas:
        cov = (m or {}).get("coverage") or {}
        for k in cov_total.keys():
            cov_total[k] += int(cov.get(k) or 0)

    # Print a quick summary to terminal
    denom = combined_rows if combined_rows else 0
    if denom:
        print("\n[step2] Coverage summary (combined):")
        print(f"  - with_abstract:     {cov_total['with_abstract']:,}  ({cov_total['with_abstract']/denom:.1%})")
        print(f"  - with_authkeywords: {cov_total['with_authkeywords']:,}  ({cov_total['with_authkeywords']/denom:.1%})")
        print(f"  - with_doi:          {cov_total['with_doi']:,}  ({cov_total['with_doi']/denom:.1%})")

    meta_out = {
        "base_total_reported_by_scopus": int(base_total),
        "planned_total_across_slices": int(planned_total),
        "combined_rows_written": int(combined_rows),
        "pubyear_min": int(PUBYEAR_MIN),
        "pubyear_max": int(PUBYEAR_MAX),
        "deep_paging_limit": int(DEEP_PAGING_LIMIT),
        "count_per_page": int(COUNT_PER_PAGE_RETRIEVE),
        "sleep_s": float(SLEEP_S_RETRIEVE),
        "max_results_total": MAX_RESULTS_TOTAL,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "slice_metas": slice_metas,
        "coverage_combined": cov_total,
    }
    final_meta = os.path.join(STEP2_DIR, STEP2_TOTAL_META_JSON)
    with open(final_meta, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)

    print(f"[step2] Final combined rows: {combined_rows:,}")
    if combined_rows != base_total:
        print(f"[step2] WARNING: combined_rows ({combined_rows:,}) != base_total ({base_total:,}).")


# ============================================================
# Step 3: benchmark DOI match
# ============================================================
def step3_benchmark_match() -> None:
    os.makedirs(STEP3_DIR, exist_ok=True)

    step2_csv = os.path.join(STEP2_DIR, STEP2_TOTAL_CSV)
    if not os.path.exists(step2_csv) or os.path.getsize(step2_csv) == 0:
        raise SystemExit(f"[step3] Missing Step2 CSV: {step2_csv}")

    bench = pd.read_csv(BENCHMARK_CSV)

    # Normalize benchmark columns defensively
    colmap = {c.strip(): c for c in bench.columns}
    study_col = colmap.get("Study", "Study")
    type_col = colmap.get("Type", "Type")
    via_col = colmap.get("Identified Via", "Identified Via")
    num_col = colmap.get("#", "#")

    # Parse DOI from Study text
    bench["benchmark_doi"] = bench[study_col].apply(_doi_from_text)

    # Load scopus DOIs
    scopus_df = pd.read_csv(step2_csv)
    scopus_dois = set()
    if "doi" in scopus_df.columns:
        scopus_dois = set(
            _normalize_doi(d)
            for d in scopus_df["doi"].dropna().astype(str).tolist()
            if str(d).strip()
        )

    # Match
    def hit_flag(d):
        if not d:
            return "No"
        return "Yes" if d in scopus_dois else "No"

    bench["Scopus_hit"] = bench["benchmark_doi"].apply(hit_flag)

    # Output CSV
    out = pd.DataFrame(
        {
            "#": bench[num_col] if num_col in bench.columns else None,
            "title": bench[study_col],
            "type": bench[type_col] if type_col in bench.columns else None,
            "identified_via": bench[via_col] if via_col in bench.columns else None,
            "doi": bench["benchmark_doi"],
            "Scopus_hit": bench["Scopus_hit"],
        }
    )

    out_path = os.path.join(STEP3_DIR, STEP3_MATCH_CSV)
    out.to_csv(out_path, index=False)

    # Summary stats
    n_total = int(len(out))
    n_with_doi = int(out["doi"].notna().sum())
    n_hits = int((out["Scopus_hit"] == "Yes").sum())
    n_missing_doi = n_total - n_with_doi
    hit_rate_overall = (n_hits / n_total) if n_total else 0.0
    hit_rate_among_doi = (n_hits / n_with_doi) if n_with_doi else 0.0

    summary = {
        "benchmark_csv": BENCHMARK_CSV,
        "step2_scopus_csv": step2_csv,
        "output_match_csv": out_path,
        "benchmark_rows_total": n_total,
        "benchmark_rows_with_doi": n_with_doi,
        "benchmark_rows_missing_doi": n_missing_doi,
        "matched_rows_by_doi": n_hits,
        "hit_rate_overall": hit_rate_overall,
        "hit_rate_among_rows_with_doi": hit_rate_among_doi,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # Write summary JSON
    summary_path = os.path.join(STEP3_DIR, STEP3_SUMMARY_JSON)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Terminal summary
    print(f"[step3] Benchmark rows: {n_total:,}")
    print(f"[step3] With DOI parsed: {n_with_doi:,} (missing DOI: {n_missing_doi:,})")
    print(f"[step3] Scopus hits (by DOI): {n_hits:,}")
    print(f"[step3] Hit rate overall: {hit_rate_overall:.1%}")
    print(f"[step3] Hit rate among DOI rows: {hit_rate_among_doi:.1%}")
    print(f"[step3] Output: {out_path}")
    print(f"[step3] Summary: {summary_path}")

# ============================================================
# Main runner
# ============================================================
def main() -> None:
    load_dotenv()

    api_key = os.getenv("SCOPUS_API_KEY")
    inst_token = os.getenv("SCOPUS_INST_TOKEN")

    if not api_key:
        raise SystemExit("SCOPUS_API_KEY missing (set it in .env at repo root).")

    os.makedirs(OUT_DIR, exist_ok=True)
    auth = ScopusAuth(api_key=api_key, inst_token=inst_token)

    if RUN_STEP1:
        step1_counts_only(auth)

    if RUN_STEP2:
        step2_retrieve_total(auth)

    if RUN_STEP3:
        step3_benchmark_match()

    print("\nDone.")
    print(f"Outputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
