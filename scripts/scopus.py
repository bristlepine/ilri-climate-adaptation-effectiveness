#!/usr/bin/env python3
"""
scopus.py

Minimal, practical pipeline:

Step 1 (counts only):
  - Build subgroup queries + element aggregates + TOTAL combined query from scripts/search_strings.yml
  - Fetch ONLY record counts (opensearch:totalResults) for each query (no record retrieval)
  - Cache per-query counts so reruns skip unchanged queries
  - Writes to outputs/step1/

Step 2 (retrieve TOTAL__ALL records; streaming):
  - Avoid deep paging limits by slicing TOTAL query into PUBYEAR ranges (<= DEEP_PAGING_LIMIT)
  - Stream each slice to CSV, then concatenate -> outputs/step2/step2_total_records.csv

Step 3 (benchmark DOI match):
  - Parse DOIs from scripts/Benchmark List - List.csv (Study column)
  - Match against step2_total_records.csv
  - Writes outputs/step3/step3_benchmark_scopus_match.csv and *_filtered.csv

Step 4 (abstracts):
  - Always uses cache and ALWAYS skips cached keys (ok/no_abstract/error).
    If you want to retry, delete outputs/step4/step4_abstracts_cache.json (and/or outputs/step4/ folder).
  - For Elsevier/ScienceDirect DOIs, uses Elsevier Article Retrieval API (reliable; no ScienceDirect scraping).
  - Fallbacks: Semantic Scholar -> OpenAlex -> Crossref -> Unpaywall links -> generic landing-page scrape.
  - Writes outputs/step4/step4_abstracts.csv and outputs/step4/step4_abstracts_cache.json

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
from urllib.parse import quote
from html.parser import HTMLParser
import html

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================================================
# Which step(s) to run
# ============================================================
RUN_STEP1 = 1
RUN_STEP2 = 1
RUN_STEP3 = 1
RUN_STEP4 = 1


# ============================================================
# General controls
# ============================================================
USE_POST_FOR_SEARCH = 1   # avoids 413 for long queries (Scopus search endpoint)
VIEW = "STANDARD"

# Step 1
STEP1_SLEEP_S = 0.10

# Step 2
COUNT_PER_PAGE_RETRIEVE = 25
STEP2_SLEEP_S = 0.15
DEEP_PAGING_LIMIT = 5000
PUBYEAR_MIN = 1990
PUBYEAR_MAX = 2025
MAX_RESULTS_TOTAL = None  # set e.g. 500 to test, or None for all

# Step 4
STEP4_ONLY_BENCHMARKS = 1
STEP4_MAX_RECORDS = None
STEP4_SLEEP_S = 0.6


# ============================================================
# Endpoints
# ============================================================
SCOPUS_SEARCH_URL = "https://api.elsevier.com/content/search/scopus"

# Elsevier Article Retrieval API (this is the key fix for ScienceDirect/Elsevier abstracts)
ELSEVIER_ARTICLE_URL = "https://api.elsevier.com/content/article/doi/"

SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/"
OPENALEX_WORKS_URL = "https://api.openalex.org/works/"
CROSSREF_WORKS_URL = "https://api.crossref.org/works/"
UNPAYWALL_URL = "https://api.unpaywall.org/v2/"


# ============================================================
# Paths
# ============================================================
HERE = os.path.dirname(os.path.abspath(__file__))
SEARCH_STRINGS_YML = os.path.join(HERE, "search_strings.yml")
BENCHMARK_CSV = os.path.join(HERE, "Benchmark List - List.csv")

OUT_DIR = os.path.join(HERE, "outputs")
STEP1_DIR = os.path.join(OUT_DIR, "step1")
STEP2_DIR = os.path.join(OUT_DIR, "step2")
STEP3_DIR = os.path.join(OUT_DIR, "step3")
STEP4_DIR = os.path.join(OUT_DIR, "step4")

# Step 1 outputs
STEP1_QUERIES_JSON = os.path.join(STEP1_DIR, "step1_queries.json")
STEP1_SUMMARY_CSV = os.path.join(STEP1_DIR, "step1_summary.csv")
STEP1_PLOT_PNG = os.path.join(STEP1_DIR, "step1_hits_plot.png")
STEP1_PLOT_CSV = os.path.join(STEP1_DIR, "step1_hits_plot.csv")
STEP1_TOTAL_QUERY_TXT = os.path.join(STEP1_DIR, "step1_total_query.txt")
STEP1_CACHE_JSON = os.path.join(STEP1_DIR, "step1_counts_cache.json")

# Step 2 outputs
STEP2_TOTAL_CSV = os.path.join(STEP2_DIR, "step2_total_records.csv")
STEP2_TOTAL_META_JSON = os.path.join(STEP2_DIR, "step2_total_records.meta.json")

# Step 3 outputs
STEP3_MATCH_CSV = os.path.join(STEP3_DIR, "step3_benchmark_scopus_match.csv")
STEP3_MATCH_FILTERED_CSV = os.path.join(STEP3_DIR, "step3_benchmark_scopus_match_filtered.csv")
STEP3_SUMMARY_JSON = os.path.join(STEP3_DIR, "step3_benchmark_scopus_match.summary.json")

# Step 4 outputs
STEP4_OUT_CSV = os.path.join(STEP4_DIR, "step4_abstracts.csv")
STEP4_CACHE_JSON = os.path.join(STEP4_DIR, "step4_abstracts_cache.json")
STEP4_METHOD_SUMMARY_JSON = os.path.join(STEP4_DIR, "step4_method_summary.json")


# ============================================================
# Helpers
# ============================================================
DOI_REGEX = re.compile(r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", re.IGNORECASE)

@dataclass
class ScopusAuth:
    api_key: str
    inst_token: Optional[str] = None


def _load_json(path: str, default):
    try:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default


def _save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


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


def _headers(auth: ScopusAuth) -> Dict[str, str]:
    h = {"X-ELS-APIKey": auth.api_key, "Accept": "application/json"}
    if auth.inst_token:
        h["X-ELS-Insttoken"] = auth.inst_token
    return h


def _rate_headers(r: requests.Response) -> dict:
    keys = ["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset", "Retry-After"]
    return {k: r.headers.get(k) for k in keys if r.headers.get(k) is not None}


def _request_with_retries(
    session: requests.Session,
    method: str,
    url: str,
    headers: dict,
    params: Optional[dict] = None,
    data: Optional[dict] = None,
    tries: int = 6,
) -> Tuple[dict, dict]:
    backoff = 1.0
    last_rate = {}

    for _ in range(tries):
        if method.upper() == "POST":
            r = session.post(url, headers=headers, data=data or params, timeout=60)
        else:
            r = session.get(url, headers=headers, params=params, timeout=60)

        last_rate = _rate_headers(r)

        if r.status_code == 200:
            return r.json(), last_rate

        # throttle
        if r.status_code == 429:
            ra = r.headers.get("Retry-After")
            if ra:
                try:
                    wait_s = max(1.0, float(ra))
                except Exception:
                    wait_s = backoff
            else:
                wait_s = backoff
            time.sleep(wait_s)
            backoff = min(backoff * 2, 60.0)
            continue

        # transient
        if r.status_code in (500, 502, 503, 504):
            time.sleep(backoff)
            backoff = min(backoff * 2, 60.0)
            continue

        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:800]}")

    raise RuntimeError(f"Failed after retries. Last rate headers: {last_rate}")


def _read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _query_hash(q: str) -> str:
    return hashlib.sha256(q.strip().encode("utf-8")).hexdigest()


def _queries_signature(queries: List[Tuple[str, str]]) -> str:
    payload = [{"name": n, "query": q} for n, q in queries]
    s = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(s).hexdigest()


def _strip_tags(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s or "")
    s = html.unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _find_first_str(o, wanted_keys: Tuple[str, ...]) -> Optional[str]:
    """Recursively find the first non-empty string value for any of wanted_keys."""
    if isinstance(o, dict):
        for k, v in o.items():
            if k in wanted_keys and isinstance(v, str) and v.strip():
                return v.strip()
        for v in o.values():
            got = _find_first_str(v, wanted_keys)
            if got:
                return got
    elif isinstance(o, list):
        for it in o:
            got = _find_first_str(it, wanted_keys)
            if got:
                return got
    return None


# ============================================================
# Step 1: counts only
# ============================================================
def scopus_count_only(auth: ScopusAuth, query: str, count_per_page: int = 1) -> Tuple[int, dict]:
    headers = _headers(auth)
    meta = {"query": query, "count_per_page": count_per_page, "view": VIEW, "rate_headers_last": {}}

    with requests.Session() as session:
        params = {"query": query, "start": "0", "count": str(count_per_page), "view": VIEW}
        method = "POST" if USE_POST_FOR_SEARCH else "GET"
        data, rate = _request_with_retries(session, method, SCOPUS_SEARCH_URL, headers, params=params, data=params)
        meta["rate_headers_last"] = rate

    sr = (data or {}).get("search-results", {}) or {}
    total = int(sr.get("opensearch:totalResults", "0"))
    return total, meta


def _build_queries(cfg: dict) -> Tuple[Dict[str, str], Dict[str, str], str]:
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
        "P": "Population (P)",
        "C_concept": "Concept (C)",
        "C_context_climate": "Context (C) climate",
        "C_context_agriculture": "Context (C) agriculture",
        "M": "Methodological focus (M)",
        "TOTAL": "TOTAL",
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

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


def step1_counts_only(auth: ScopusAuth) -> None:
    os.makedirs(STEP1_DIR, exist_ok=True)

    cfg = _read_yaml(SEARCH_STRINGS_YML)
    subgroup_q, element_q, combined_q = _build_queries(cfg)

    queries: List[Tuple[str, str]] = []
    queries += sorted(subgroup_q.items(), key=lambda x: x[0])
    queries += sorted(element_q.items(), key=lambda x: x[0])
    if combined_q:
        queries.append(("TOTAL__ALL", combined_q))

    if combined_q:
        with open(STEP1_TOTAL_QUERY_TXT, "w", encoding="utf-8") as f:
            f.write(combined_q.strip() + "\n")

    sig = _queries_signature(queries)
    payload = [{"_signature": sig, "_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}]
    payload += [{"name": n, "query": q} for n, q in queries]
    _save_json(STEP1_QUERIES_JSON, payload)

    cache = _load_json(STEP1_CACHE_JSON, default={})
    if not isinstance(cache, dict):
        cache = {}

    rows = []
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    for qname, q in queries:
        qh = _query_hash(q)
        cached = cache.get(qname)

        if isinstance(cached, dict) and cached.get("query_hash") == qh:
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
            total, meta = scopus_count_only(auth, q, count_per_page=1)
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

        if STEP1_SLEEP_S > 0:
            time.sleep(STEP1_SLEEP_S)

    _save_json(STEP1_CACHE_JSON, cache)

    summary = pd.DataFrame(rows)
    summary.to_csv(STEP1_SUMMARY_CSV, index=False)
    summary.to_csv(STEP1_PLOT_CSV, index=False)

    print("\n[step1] Summary table:")
    print(summary[["query_name", "total_results", "error"]].to_string(index=False))

    _plot_counts(summary, STEP1_PLOT_PNG, "Scopus record counts by query")


# ============================================================
# Step 2: retrieve TOTAL__ALL records
# ============================================================
def _extract_entry_row(e: dict) -> dict:
    dcid = e.get("dc:identifier")
    scopus_id = dcid.split(":", 1)[1].strip() if isinstance(dcid, str) and dcid.startswith("SCOPUS_ID:") else None

    doi = e.get("prism:doi")
    doi = doi.strip() if isinstance(doi, str) else None

    return {
        "title": e.get("dc:title"),
        "coverDate": e.get("prism:coverDate"),
        "publicationName": e.get("prism:publicationName"),
        "doi": doi or None,
        "eid": e.get("eid"),
        "scopus_id": scopus_id,
        "citedby_count": e.get("citedby-count"),
        "prism_url": e.get("prism:url"),
    }


def _with_pubyear_range(base_query: str, y0: int, y1: int) -> str:
    return f"({base_query}) AND PUBYEAR > {y0 - 1} AND PUBYEAR < {y1 + 1}"


def _with_pubyear_missing_or_outside(base_query: str, y0: int, y1: int) -> str:
    return f"({base_query}) AND NOT (PUBYEAR > {y0 - 1} AND PUBYEAR < {y1 + 1})"


def _plan_year_slices(auth: ScopusAuth, base_query: str, y0: int, y1: int) -> List[Tuple[int, int, int]]:
    total, _ = scopus_count_only(auth, _with_pubyear_range(base_query, y0, y1), count_per_page=1)
    if total <= DEEP_PAGING_LIMIT or y0 == y1:
        return [(y0, y1, total)]
    mid = (y0 + y1) // 2
    return _plan_year_slices(auth, base_query, y0, mid) + _plan_year_slices(auth, base_query, mid + 1, y1)


def _safe_slice_tag(y0: int, y1: int) -> str:
    return f"PUBYEAR_{y0}_{y1}"


def _concat_csvs(csv_paths: List[str], out_csv: str) -> int:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
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
    headers = _headers(auth)
    method = "POST" if USE_POST_FOR_SEARCH else "GET"

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if os.path.exists(out_csv):
        os.remove(out_csv)

    retrieved = 0
    total_reported = None

    meta = {
        "query": query,
        "view": VIEW,
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

            params = {"query": query, "start": str(retrieved), "count": str(count_per_page), "view": VIEW}
            data, rate = _request_with_retries(session, method, SCOPUS_SEARCH_URL, headers, params=params, data=params)
            meta["rate_headers_last"] = rate

            sr = (data or {}).get("search-results", {}) or {}
            if total_reported is None:
                total_reported = int(sr.get("opensearch:totalResults", "0"))
                meta["total_reported_by_scopus"] = total_reported
                pbar_total = total_reported if max_results is None else min(total_reported, max_results)
                pbar = tqdm(total=pbar_total, desc="[step2] Retrieving", unit="rec")
                print(f"[step2] Total reported by Scopus: {total_reported:,}")

            entries = sr.get("entry", []) or []
            if not entries:
                break

            rows = [_extract_entry_row(e) for e in entries]
            df = pd.DataFrame(rows)

            header_needed = (not os.path.exists(out_csv)) or (os.path.getsize(out_csv) == 0)
            df.to_csv(out_csv, mode="a", header=header_needed, index=False)

            got = len(df)
            retrieved += got
            page_n += 1
            if pbar:
                pbar.update(got)

            if total_reported is not None and retrieved >= total_reported:
                break

            if sleep_s > 0:
                time.sleep(sleep_s)

    if pbar:
        pbar.close()

    meta["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta["retrieved_rows"] = int(retrieved)
    meta["total_reported_by_scopus"] = int(total_reported or 0)
    return int(total_reported or 0), int(retrieved), meta


def step2_retrieve_total(auth: ScopusAuth) -> None:
    os.makedirs(STEP2_DIR, exist_ok=True)

    if os.path.exists(STEP2_TOTAL_CSV) and os.path.getsize(STEP2_TOTAL_CSV) > 0:
        print(f"[step2] SKIP (exists): {STEP2_TOTAL_CSV}")
        return

    cfg = _read_yaml(SEARCH_STRINGS_YML)
    _, _, base_q = _build_queries(cfg)
    if not base_q:
        raise SystemExit("[step2] TOTAL__ALL query is empty; check search_strings.yml")

    base_total, _ = scopus_count_only(auth, base_q, count_per_page=1)
    print(f"[step2] TOTAL__ALL reported by Scopus: {base_total:,}")

    print("[step2] Planning PUBYEAR slices...")
    slices = _plan_year_slices(auth, base_q, PUBYEAR_MIN, PUBYEAR_MAX)

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

    for y0, y1, slice_total in slices:
        tag = _safe_slice_tag(y0, y1)
        q_slice = _with_pubyear_range(base_q, y0, y1)

        out_csv = os.path.join(STEP2_DIR, f"step2_total_records__{tag}.csv")
        out_meta = os.path.join(STEP2_DIR, f"step2_total_records__{tag}.meta.json")

        if os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
            print(f"[step2] SKIP {tag} (exists): {out_csv}")
            slice_csvs.append(out_csv)
            continue

        print(f"\n[step2] RETRIEVE {tag} (expected {slice_total:,})")
        total_reported, retrieved, meta = scopus_retrieve_stream_to_csv_start(
            auth,
            q_slice,
            out_csv=out_csv,
            count_per_page=COUNT_PER_PAGE_RETRIEVE,
            sleep_s=STEP2_SLEEP_S,
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
        _save_json(out_meta, meta)

        print(f"[step2] Done {tag}: reported={total_reported:,} retrieved={retrieved:,}")
        slice_csvs.append(out_csv)
        slice_metas.append(meta)

    if missing_total > 0:
        tag = "PUBYEAR_MISSING_OR_OUTSIDE"
        out_csv = os.path.join(STEP2_DIR, f"step2_total_records__{tag}.csv")
        out_meta = os.path.join(STEP2_DIR, f"step2_total_records__{tag}.meta.json")

        if os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
            print(f"[step2] SKIP {tag} (exists): {out_csv}")
            slice_csvs.append(out_csv)
        else:
            print(f"\n[step2] RETRIEVE {tag} (expected {missing_total:,})")
            total_reported, retrieved, meta = scopus_retrieve_stream_to_csv_start(
                auth,
                missing_q,
                out_csv=out_csv,
                count_per_page=COUNT_PER_PAGE_RETRIEVE,
                sleep_s=STEP2_SLEEP_S,
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
            _save_json(out_meta, meta)
            print(f"[step2] Done {tag}: reported={total_reported:,} retrieved={retrieved:,}")
            slice_csvs.append(out_csv)
            slice_metas.append(meta)

    print(f"\n[step2] Combining slice CSVs -> {STEP2_TOTAL_CSV}")
    combined_rows = _concat_csvs(slice_csvs, STEP2_TOTAL_CSV)

    meta_out = {
        "base_total_reported_by_scopus": int(base_total),
        "planned_total_across_slices": int(planned_total),
        "combined_rows_written": int(combined_rows),
        "pubyear_min": int(PUBYEAR_MIN),
        "pubyear_max": int(PUBYEAR_MAX),
        "deep_paging_limit": int(DEEP_PAGING_LIMIT),
        "count_per_page": int(COUNT_PER_PAGE_RETRIEVE),
        "sleep_s": float(STEP2_SLEEP_S),
        "max_results_total": MAX_RESULTS_TOTAL,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "slice_metas": slice_metas,
    }
    _save_json(STEP2_TOTAL_META_JSON, meta_out)

    print(f"[step2] Final combined rows: {combined_rows:,}")
    if combined_rows != base_total:
        print(f"[step2] WARNING: combined_rows ({combined_rows:,}) != base_total ({base_total:,}).")


# ============================================================
# Step 3: benchmark DOI match
# ============================================================
def step3_benchmark_match() -> None:
    os.makedirs(STEP3_DIR, exist_ok=True)

    if not os.path.exists(STEP2_TOTAL_CSV) or os.path.getsize(STEP2_TOTAL_CSV) == 0:
        raise SystemExit(f"[step3] Missing Step2 CSV: {STEP2_TOTAL_CSV}")

    # --- Load benchmark list ---
    bench = pd.read_csv(BENCHMARK_CSV)

    colmap = {c.strip(): c for c in bench.columns}
    study_col = colmap.get("Study", "Study")
    type_col = colmap.get("Type", "Type")
    via_col = colmap.get("Identified Via", "Identified Via")
    num_col = colmap.get("#", "#")

    bench["benchmark_doi"] = bench[study_col].apply(_doi_from_text)

    # --- Load Scopus results (Step 2) and build comparison sets + totals ---
    scopus_df = pd.read_csv(STEP2_TOTAL_CSV)

    scopus_rows_total = int(len(scopus_df))
    scopus_rows_with_doi = int(scopus_df["doi"].notna().sum()) if "doi" in scopus_df.columns else 0

    scopus_dois = set()
    if "doi" in scopus_df.columns:
        scopus_dois = set(
            _normalize_doi(d)
            for d in scopus_df["doi"].dropna().astype(str).tolist()
            if str(d).strip()
        )
    scopus_unique_dois = int(len(scopus_dois))

    def hit_flag(d):
        if not d:
            return "No"
        return "Yes" if d in scopus_dois else "No"

    bench["Scopus_hit"] = bench["benchmark_doi"].apply(hit_flag)

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

    # Write outputs (overwrite to keep summary consistent with current inputs)
    out.to_csv(STEP3_MATCH_CSV, index=False)
    matches_only = out[out["Scopus_hit"] == "Yes"].copy()
    matches_only.to_csv(STEP3_MATCH_FILTERED_CSV, index=False)

    # --- Existing summary stats ---
    n_total = int(len(out))
    n_with_doi = int(out["doi"].notna().sum())
    n_hits = int((out["Scopus_hit"] == "Yes").sum())
    n_missing_doi = n_total - n_with_doi
    hit_rate_overall = (n_hits / n_total) if n_total else 0.0
    hit_rate_among_doi = (n_hits / n_with_doi) if n_with_doi else 0.0

    # --- NEW: counts by identified_via ---
    via_norm = (
        out["identified_via"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace({"": "Unknown"})
    )

    benchmark_counts_by_identified_via = {
        str(k): int(v) for k, v in via_norm.value_counts(dropna=False).items()
    }

    hits_counts_by_identified_via = {
        str(k): int(v)
        for k, v in via_norm[out["Scopus_hit"] == "Yes"].value_counts(dropna=False).items()
    }

    summary = {
        "benchmark_csv": BENCHMARK_CSV,
        "step2_scopus_csv": STEP2_TOTAL_CSV,
        "output_match_csv": STEP3_MATCH_CSV,
        "output_matches_only_csv": STEP3_MATCH_FILTERED_CSV,

        # benchmark totals
        "benchmark_rows_total": n_total,
        "benchmark_rows_with_doi": n_with_doi,
        "benchmark_rows_missing_doi": n_missing_doi,
        "matched_rows_by_doi": n_hits,
        "hit_rate_overall": hit_rate_overall,
        "hit_rate_among_rows_with_doi": hit_rate_among_doi,

        # NEW: scopus comparison totals
        "scopus_items_total_rows_compared_against": scopus_rows_total,
        "scopus_items_with_doi": scopus_rows_with_doi,
        "scopus_unique_dois_compared_against": scopus_unique_dois,

        # NEW: benchmark breakdowns
        "benchmark_counts_by_identified_via": benchmark_counts_by_identified_via,
        "matched_counts_by_identified_via": hits_counts_by_identified_via,

        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _save_json(STEP3_SUMMARY_JSON, summary)

    print(f"[step3] Benchmark rows: {n_total:,}")
    print(f"[step3] With DOI parsed: {n_with_doi:,} (missing DOI: {n_missing_doi:,})")
    print(f"[step3] Scopus hits (by DOI): {n_hits:,}")
    print(f"[step3] Hit rate overall: {hit_rate_overall:.1%}")
    print(f"[step3] Hit rate among DOI rows: {hit_rate_among_doi:.1%}")

    # NEW prints
    print(f"[step3] Scopus items compared against (rows): {scopus_rows_total:,}")
    print(f"[step3] Scopus items with DOI:               {scopus_rows_with_doi:,}")
    print(f"[step3] Scopus unique DOIs compared:        {scopus_unique_dois:,}")

    print("[step3] Benchmark counts by identified_via:")
    for k, v in benchmark_counts_by_identified_via.items():
        print(f"  - {k}: {v:,}")

    print("[step3] Matches (hits) by identified_via:")
    for k, v in hits_counts_by_identified_via.items():
        print(f"  - {k}: {v:,}")

    print(f"[step3] Output (full): {STEP3_MATCH_CSV}")
    print(f"[step3] Output (matches only): {STEP3_MATCH_FILTERED_CSV}")
    print(f"[step3] Summary: {STEP3_SUMMARY_JSON}")

# ============================================================
# Step 4: abstracts (clean + cache-only)
# ============================================================
class _MetaAndJsonLdParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.metas = {}
        self._in_jsonld = False
        self._jsonld_buf = []
        self.jsonld_blobs = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "meta":
            d = {k.lower(): (v or "") for k, v in attrs}
            key = (d.get("name") or d.get("property") or d.get("itemprop") or "").strip()
            content = (d.get("content") or "").strip()
            if key and content and key not in self.metas:
                self.metas[key] = content

        if tag.lower() == "script":
            d = {k.lower(): (v or "") for k, v in attrs}
            if d.get("type", "").lower() == "application/ld+json":
                self._in_jsonld = True
                self._jsonld_buf = []

    def handle_endtag(self, tag):
        if tag.lower() == "script" and self._in_jsonld:
            self._in_jsonld = False
            blob = "".join(self._jsonld_buf).strip()
            if blob:
                self.jsonld_blobs.append(blob)

    def handle_data(self, data):
        if self._in_jsonld:
            self._jsonld_buf.append(data)


def elsevier_article_fetch(auth: ScopusAuth, doi_norm: str) -> dict:
    """
    Elsevier Article Retrieval API:
      GET https://api.elsevier.com/content/article/doi/<doi>?view=META_ABS&httpAccept=application/json
    """
    url = ELSEVIER_ARTICLE_URL + quote(doi_norm, safe="")
    headers = _headers(auth)
    params = {"view": "META_ABS", "httpAccept": "application/json"}

    with requests.Session() as session:
        data, _rate = _request_with_retries(session, "GET", url, headers, params=params)

    # Title is usually in coredata dc:title
    title = _find_first_str(data, ("dc:title", "title"))
    # Abstract is often in coredata dc:description (Elsevier) or elsewhere
    abstract = _find_first_str(data, ("dc:description", "ce:abstract", "abstract", "description"))
    if abstract:
        abstract = _strip_tags(abstract)

    return {
        "title": title,
        "abstract": abstract if abstract and abstract.strip() else None,
        "url": url,
        "source": "elsevier_api",
        "status": 200,
    }


def _http_get_json(
    url: str,
    headers: dict,
    params: Optional[dict] = None,
    tries: int = 6,
) -> Tuple[Optional[dict], Optional[str], int]:
    backoff = 1.0
    last_status = 0
    last_err = None

    with requests.Session() as session:
        for _ in range(tries):
            try:
                r = session.get(url, headers=headers, params=params, timeout=60)
                last_status = r.status_code

                if r.status_code == 200:
                    return r.json(), None, r.status_code

                if r.status_code == 429:
                    ra = r.headers.get("Retry-After")
                    if ra:
                        try:
                            wait_s = max(1.0, float(ra))
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

                last_err = f"HTTP {r.status_code}: {r.text[:300]}"
                return None, last_err, r.status_code

            except Exception as ex:
                last_err = str(ex)
                time.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    return None, f"Failed after retries: {last_err}", last_status


def semantic_scholar_fetch(doi_norm: str, contact_email: Optional[str]) -> dict:
    doi_id = "DOI:" + doi_norm
    url = SEMANTIC_SCHOLAR_URL + quote(doi_id, safe=":/")
    fields = "title,abstract,year,venue,url,externalIds,openAccessPdf"
    headers = {
        "Accept": "application/json",
        "User-Agent": f"ilri-climate-adaptation-effectiveness/step4 ({contact_email or 'no-email'})",
    }
    data, err, status = _http_get_json(url, headers=headers, params={"fields": fields})
    if err:
        return {"error": err, "status": status, "source": "semantic_scholar"}

    abstract = (data or {}).get("abstract")
    title = (data or {}).get("title")
    return {
        "title": title,
        "abstract": abstract.strip() if isinstance(abstract, str) and abstract.strip() else None,
        "url": (data or {}).get("url"),
        "source": "semantic_scholar",
        "status": 200,
    }


def _openalex_abstract_from_inverted_index(inv: dict) -> Optional[str]:
    if not isinstance(inv, dict) or not inv:
        return None
    pos_to_word = {}
    for word, positions in inv.items():
        if not isinstance(positions, list):
            continue
        for p in positions:
            if isinstance(p, int):
                pos_to_word[p] = word
    if not pos_to_word:
        return None
    return " ".join(pos_to_word[k] for k in sorted(pos_to_word.keys())).strip() or None


def openalex_fetch(doi_norm: str, contact_email: Optional[str]) -> dict:
    doi_url = "https://doi.org/" + doi_norm
    url = OPENALEX_WORKS_URL + quote(doi_url, safe=":/")

    params = {}
    if contact_email:
        params["mailto"] = contact_email

    headers = {
        "Accept": "application/json",
        "User-Agent": f"ilri-climate-adaptation-effectiveness/step4 ({contact_email or 'no-email'})",
    }
    data, err, status = _http_get_json(url, headers=headers, params=params)
    if err:
        return {"error": err, "status": status, "source": "openalex"}

    abstract = _openalex_abstract_from_inverted_index((data or {}).get("abstract_inverted_index"))
    title = (data or {}).get("title")

    primary_loc = (data or {}).get("primary_location") or {}
    landing = (primary_loc.get("landing_page_url") if isinstance(primary_loc, dict) else None)
    pdf = (primary_loc.get("pdf_url") if isinstance(primary_loc, dict) else None)

    return {
        "title": title,
        "abstract": abstract,
        "url": (data or {}).get("id"),
        "landing_url": landing,
        "pdf_url": pdf,
        "source": "openalex",
        "status": 200,
    }


def crossref_fetch(doi_norm: str, contact_email: Optional[str]) -> dict:
    url = CROSSREF_WORKS_URL + quote(doi_norm)
    params = {}
    if contact_email:
        params["mailto"] = contact_email

    headers = {
        "Accept": "application/json",
        "User-Agent": f"ilri-climate-adaptation-effectiveness/step4 ({contact_email or 'no-email'})",
    }
    data, err, status = _http_get_json(url, headers=headers, params=params)
    if err:
        return {"error": err, "status": status, "source": "crossref"}

    msg = (data or {}).get("message") or {}
    title = None
    t = msg.get("title")
    if isinstance(t, list) and t:
        title = t[0]

    abstract = msg.get("abstract")
    if isinstance(abstract, str) and abstract.strip():
        abstract = _strip_tags(abstract)
    else:
        abstract = None

    return {
        "title": title,
        "abstract": abstract,
        "url": msg.get("URL"),
        "source": "crossref",
        "status": 200,
    }


def unpaywall_fetch(doi_norm: str, email: str) -> dict:
    url = UNPAYWALL_URL + quote(doi_norm)
    headers = {
        "Accept": "application/json",
        "User-Agent": f"ilri-climate-adaptation-effectiveness/step4 ({email})",
    }
    params = {"email": email}

    data, err, status = _http_get_json(url, headers=headers, params=params)
    if err:
        return {"error": err, "status": status, "source": "unpaywall"}

    best = (data or {}).get("best_oa_location") or {}
    if not isinstance(best, dict):
        best = {}

    return {
        "is_oa": (data or {}).get("is_oa"),
        "oa_pdf_url": best.get("url_for_pdf"),
        "oa_landing_url": best.get("url"),
        "source": "unpaywall",
        "status": 200,
    }


def fetch_abstract_from_landing_url(url: str, contact_email: Optional[str], max_bytes: int = 2_000_000) -> dict:
    """
    Generic HTML landing-page scrape (best-effort).
    NOTE: ScienceDirect often blocks (403). This is for non-Elsevier publishers.
    """
    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "User-Agent": f"ilri-climate-adaptation-effectiveness/step4 ({contact_email or 'no-email'})",
    }

    with requests.Session() as session:
        r = session.get(url, headers=headers, timeout=60, allow_redirects=True, stream=True)
        ct = (r.headers.get("Content-Type") or "").lower()
        final_url = getattr(r, "url", url)
        status = int(getattr(r, "status_code", 0) or 0)

        if status != 200:
            return {"error": f"landing_http_{status}", "final_url": final_url, "content_type": ct, "status": status}

        if "html" not in ct:
            return {"error": f"landing_not_html: {ct}", "final_url": final_url, "content_type": ct, "status": status}

        raw = b""
        for chunk in r.iter_content(chunk_size=65536):
            if not chunk:
                break
            raw += chunk
            if len(raw) >= max_bytes:
                break

        try:
            text = raw.decode(r.encoding or "utf-8", errors="replace")
        except Exception:
            text = raw.decode("utf-8", errors="replace")

    p = _MetaAndJsonLdParser()
    p.feed(text)
    metas = {k.lower(): v for k, v in (p.metas or {}).items()}

    title = None
    for tk in ("citation_title", "dc.title", "og:title", "twitter:title"):
        if tk in metas and metas[tk].strip():
            title = re.sub(r"\s+", " ", html.unescape(metas[tk])).strip()
            break

    abstract = None
    method = None

    for k in ("citation_abstract", "dcterms.abstract", "dc.description", "dc:description", "description", "og:description"):
        if k in metas and metas[k].strip():
            cand = re.sub(r"\s+", " ", html.unescape(metas[k])).strip()
            if k in ("description", "og:description") and len(cand) < 80:
                continue
            abstract = cand
            method = "landing_meta"
            break

    if not abstract:
        for blob in p.jsonld_blobs:
            try:
                obj = json.loads(blob)
            except Exception:
                continue
            d = _find_first_str(obj, ("abstract", "description"))
            if d and len(d.strip()) >= 80:
                abstract = re.sub(r"\s+", " ", d).strip()
                method = "landing_jsonld"
                break

    return {
        "title": title,
        "abstract": abstract,
        "source": "landing_html",
        "method": method,
        "final_url": final_url,
        "status": status,
        "content_type": ct,
        "bytes_read": len(raw),
    }


def _choose_key(scopus_id: Optional[str], doi_norm: Optional[str]) -> Optional[str]:
    if scopus_id and str(scopus_id).strip():
        return str(scopus_id).strip()
    if doi_norm and str(doi_norm).strip():
        return str(doi_norm).strip()
    return None


def step4_fetch_abstracts(auth: ScopusAuth, contact_email: Optional[str] = None) -> None:
    os.makedirs(STEP4_DIR, exist_ok=True)

    if os.path.exists(STEP4_OUT_CSV) and os.path.getsize(STEP4_OUT_CSV) > 0:
        print(f"[step4] SKIP (exists): {STEP4_OUT_CSV}")
        return

    if not os.path.exists(STEP2_TOTAL_CSV) or os.path.getsize(STEP2_TOTAL_CSV) == 0:
        raise SystemExit(f"[step4] Missing Step2 CSV: {STEP2_TOTAL_CSV}")

    unpaywall_email = os.getenv("UNPAYWALL_EMAIL") or contact_email

    df = pd.read_csv(STEP2_TOTAL_CSV)
    if "doi" not in df.columns:
        raise SystemExit("[step4] Step2 CSV missing 'doi' column.")

    df["_doi_norm"] = df["doi"].apply(lambda x: _normalize_doi(x) if pd.notna(x) and str(x).strip() else None)
    df["_key"] = df.apply(
        lambda r: _choose_key(
            str(r.get("scopus_id")).strip() if pd.notna(r.get("scopus_id")) else None,
            r.get("_doi_norm"),
        ),
        axis=1,
    )
    df = df[df["_key"].notna()].copy()

    if STEP4_ONLY_BENCHMARKS:
        if not os.path.exists(STEP3_MATCH_FILTERED_CSV) or os.path.getsize(STEP3_MATCH_FILTERED_CSV) == 0:
            raise SystemExit(f"[step4] STEP4_ONLY_BENCHMARKS=1 but missing: {STEP3_MATCH_FILTERED_CSV}")
        b = pd.read_csv(STEP3_MATCH_FILTERED_CSV)
        if "doi" not in b.columns:
            raise SystemExit(f"[step4] Filtered benchmark file missing 'doi': {STEP3_MATCH_FILTERED_CSV}")
        b["_doi_norm"] = b["doi"].apply(lambda x: _normalize_doi(x) if pd.notna(x) and str(x).strip() else None)
        wanted = set(b["_doi_norm"].dropna().astype(str).tolist())
        df = df[df["_doi_norm"].isin(wanted)].copy()
        print(f"[step4] Restricting to benchmark matches: {len(df):,} records")

    df = df.drop_duplicates(subset=["_key"]).reset_index(drop=True)

    if STEP4_MAX_RECORDS is not None:
        df = df.head(int(STEP4_MAX_RECORDS)).copy()
        print(f"[step4] Capped to first {len(df):,} records")

    cache = _load_json(STEP4_CACHE_JSON, default={})
    if not isinstance(cache, dict):
        cache = {}

    total_n = len(df)
    print(f"[step4] Total to process: {total_n:,} (cache keys: {len(cache):,})")

    # We'll stream-write output CSV in chunks, but ALWAYS reuse cached rows if present.
    if os.path.exists(STEP4_OUT_CSV):
        os.remove(STEP4_OUT_CSV)

    chunk: List[dict] = []
    wrote_header = False

    def flush_chunk():
        nonlocal chunk, wrote_header
        if not chunk:
            return
        out_df = pd.DataFrame(chunk)
        out_df.to_csv(STEP4_OUT_CSV, mode="a", header=(not wrote_header), index=False)
        wrote_header = True
        chunk = []

    pbar = tqdm(total=total_n, desc="[step4] Abstracts", unit="rec")

    # summary counts
    fetched_new = 0
    reused_cached = 0

    for idx, r in df.iterrows():
        scopus_id = str(r.get("scopus_id")).strip() if pd.notna(r.get("scopus_id")) else None
        doi_norm = r.get("_doi_norm")
        key = str(r.get("_key")).strip()

        # ALWAYS skip network if key exists in cache (even if it was no_abstract/error).
        if key in cache and isinstance(cache[key], dict):
            chunk.append(cache[key])
            reused_cached += 1
            pbar.update(1)
            if len(chunk) >= 100:
                flush_chunk()
            continue

        fetched_new += 1

        # Build record skeleton
        rec = {
            "scopus_id": scopus_id,
            "doi": doi_norm,
            "title": r.get("title"),
            "coverDate": r.get("coverDate"),
            "publicationName": r.get("publicationName"),
            "abstract": None,
            "abstract_source": "none",
            "url": None,
            "landing_url": None,
            "pdf_url": None,
            "is_oa": None,
            "error": None,
            "fetch_status": "no_abstract",
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        if not doi_norm:
            rec["fetch_status"] = "error"
            rec["error"] = "no_doi"
            cache[key] = rec
            chunk.append(rec)
            pbar.update(1)
            if len(chunk) >= 100:
                flush_chunk()
            continue

        # ---- Fetch order (fastest/most reliable first) ----
        # 1) Elsevier Article Retrieval API (fixes ScienceDirect 403)
        try:
            e = elsevier_article_fetch(auth, doi_norm)
            if e.get("abstract"):
                rec["title"] = e.get("title") or rec["title"]
                rec["abstract"] = e.get("abstract")
                rec["abstract_source"] = "elsevier_api"
                rec["url"] = e.get("url")
                rec["fetch_status"] = "ok"
        except Exception as ex:
            # swallow and continue to fallbacks
            rec["error"] = f"elsevier_api: {ex}"

        # 2) Semantic Scholar
        if not rec["abstract"]:
            try:
                s2 = semantic_scholar_fetch(doi_norm, contact_email)
                if s2.get("abstract"):
                    rec["title"] = s2.get("title") or rec["title"]
                    rec["abstract"] = s2.get("abstract")
                    rec["abstract_source"] = "semantic_scholar"
                    rec["url"] = s2.get("url")
                    rec["fetch_status"] = "ok"
            except Exception as ex:
                rec["error"] = rec["error"] or f"semantic_scholar: {ex}"

        # 3) OpenAlex
        if not rec["abstract"]:
            try:
                oa = openalex_fetch(doi_norm, contact_email)
                if oa.get("abstract"):
                    rec["title"] = oa.get("title") or rec["title"]
                    rec["abstract"] = oa.get("abstract")
                    rec["abstract_source"] = "openalex"
                    rec["url"] = oa.get("url")
                    rec["landing_url"] = oa.get("landing_url")
                    rec["pdf_url"] = oa.get("pdf_url")
                    rec["fetch_status"] = "ok"
                else:
                    rec["landing_url"] = rec["landing_url"] or oa.get("landing_url")
                    rec["pdf_url"] = rec["pdf_url"] or oa.get("pdf_url")
            except Exception as ex:
                rec["error"] = rec["error"] or f"openalex: {ex}"

        # 4) Crossref
        if not rec["abstract"]:
            try:
                cr = crossref_fetch(doi_norm, contact_email)
                if cr.get("abstract"):
                    rec["title"] = cr.get("title") or rec["title"]
                    rec["abstract"] = cr.get("abstract")
                    rec["abstract_source"] = "crossref"
                    rec["url"] = cr.get("url")
                    rec["fetch_status"] = "ok"
            except Exception as ex:
                rec["error"] = rec["error"] or f"crossref: {ex}"

        # 5) Unpaywall links + landing scrape
        if not rec["abstract"] and unpaywall_email:
            try:
                upw = unpaywall_fetch(doi_norm, unpaywall_email)
                rec["is_oa"] = upw.get("is_oa")
                rec["pdf_url"] = rec["pdf_url"] or upw.get("oa_pdf_url")
                rec["landing_url"] = rec["landing_url"] or upw.get("oa_landing_url")

                landing_try = rec["landing_url"] or ("https://doi.org/" + doi_norm)
                landing = fetch_abstract_from_landing_url(landing_try, contact_email=contact_email)

                if landing.get("abstract"):
                    rec["title"] = landing.get("title") or rec["title"]
                    rec["abstract"] = landing.get("abstract")
                    rec["abstract_source"] = landing.get("method") or "landing_html"
                    rec["landing_url"] = landing_try
                    rec["fetch_status"] = "ok"
            except Exception as ex:
                rec["error"] = rec["error"] or f"unpaywall/landing: {ex}"

        # finalize
        if rec["fetch_status"] != "ok" and rec.get("error") is None:
            rec["error"] = None  # keep clean: truly just "no_abstract"

        cache[key] = rec
        chunk.append(rec)

        # persist cache periodically
        if fetched_new % 25 == 0:
            _save_json(STEP4_CACHE_JSON, cache)

        if STEP4_SLEEP_S > 0:
            time.sleep(STEP4_SLEEP_S)

        pbar.update(1)

        if len(chunk) >= 100:
            flush_chunk()

    pbar.close()
    flush_chunk()
    _save_json(STEP4_CACHE_JSON, cache)

    # Summary
    outdf = pd.read_csv(STEP4_OUT_CSV) if (os.path.exists(STEP4_OUT_CSV) and os.path.getsize(STEP4_OUT_CSV) > 0) else pd.DataFrame()
    total = int(len(outdf))
    ok = int((outdf.get("fetch_status") == "ok").sum()) if "fetch_status" in outdf.columns else 0
    noabs = int((outdf.get("fetch_status") == "no_abstract").sum()) if "fetch_status" in outdf.columns else 0
    err = int((outdf.get("fetch_status") == "error").sum()) if "fetch_status" in outdf.columns else 0

    print("\n[step4] Retrieval summary:")
    print(f"[step4] Total rows written: {total:,} | ok: {ok:,} | no_abstract: {noabs:,} | error: {err:,}")
    print(f"[step4] Reused cached (no network): {reused_cached:,}")
    print(f"[step4] New fetched (network):      {fetched_new:,}")

    if "abstract_source" in outdf.columns and ok > 0:
        vc_ok = outdf.loc[outdf["fetch_status"] == "ok", "abstract_source"].value_counts(dropna=False)
        print("\n[step4] OK abstracts by source:")
        print(vc_ok.to_string())
        _save_json(
            STEP4_METHOD_SUMMARY_JSON,
            {
                "total_rows": total,
                "ok": ok,
                "no_abstract": noabs,
                "error": err,
                "ok_by_abstract_source": {str(k): int(v) for k, v in vc_ok.items()},
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )

    print(f"\n[step4] Output: {STEP4_OUT_CSV}")
    print(f"[step4] Cache:  {STEP4_CACHE_JSON}")


# ============================================================
# Main
# ============================================================
def main() -> None:
    load_dotenv()

    api_key = os.getenv("SCOPUS_API_KEY")
    inst_token = os.getenv("SCOPUS_INST_TOKEN")
    contact_email = os.getenv("CONTACT_EMAIL")

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
    if RUN_STEP4:
        step4_fetch_abstracts(auth, contact_email=contact_email)

    print("\nDone.")
    print(f"Outputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
