#!/usr/bin/env python3
"""
scopus.py

Step 1 (summary-only):
  - Build subgroup queries + element aggregates + TOTAL combined query from scripts/search_strings.yml
  - Fetch ONLY record counts (opensearch:totalResults) for each query (no record retrieval)
  - Cache per-query counts so reruns can SKIP unchanged queries
  - Write:
      - step1_summary.csv
      - step1_hits_plot.png
      - step1_hits_plot.csv   (same table as the figure)
      - step1_queries.json
      - step1_total_query.txt
      - step1_counts_cache.json

Run:
  python scripts/scopus.py
"""

import os
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


# ============================================================
# Initial Setup (edit these)
# ============================================================
RUN_STEP1 = 1            # Step 1: Scopus counts only (uses API)
SKIP_EXISTING_STEP1 = 1  # If cached counts exist and query unchanged, don't call API

USE_POST_FOR_SEARCH = 1  # avoids 413 for long queries
COUNT_PER_PAGE = 1       # we only need totalResults, so keep this tiny
SLEEP_S = 0.10           # small delay to be gentle
VIEW = "STANDARD"

HERE = os.path.dirname(os.path.abspath(__file__))
SEARCH_STRINGS_YML = os.path.join(HERE, "search_strings.yml")

OUT_DIR = os.path.join(HERE, "outputs")
STEP1_DIR = os.path.join(OUT_DIR, "step1")

# Step 1 output names (ALL prepended with step1_)
STEP1_QUERIES_JSON = "step1_queries.json"
STEP1_SUMMARY_CSV = "step1_summary.csv"
STEP1_PLOT_PNG = "step1_hits_plot.png"
STEP1_TOTAL_QUERY_TXT = "step1_total_query.txt"
STEP1_CACHE_JSON = "step1_counts_cache.json"  # per-query cache so reruns can skip query-by-query


# ============================================================
# Scopus
# ============================================================
SCOPUS_SEARCH_URL = "https://api.elsevier.com/content/search/scopus"


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


def scopus_count_only(
    auth: ScopusAuth,
    query: str,
    count_per_page: int = 1,
) -> Tuple[int, dict]:
    """
    Summary-only: return (totalResults, meta) without retrieving records.
    """
    headers = _headers(auth)
    meta = {"query": query, "count_per_page": count_per_page, "view": VIEW, "rate_headers_last": {}}

    with requests.Session() as session:
        params = {"query": query, "start": "0", "count": str(count_per_page), "view": VIEW}
        method = "POST" if USE_POST_FOR_SEARCH else "GET"
        data, rate = _request_with_retries(session, method, SCOPUS_SEARCH_URL, headers, params)
        meta["rate_headers_last"] = rate

    sr = (data or {}).get("search-results", {}) or {}
    total = int(sr.get("opensearch:totalResults", "0"))
    return total, meta


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

    # Store raw OR expressions per element (no field wrapper) for TOTAL__ALL
    element_raw_or: Dict[str, str] = {}

    for element_key, subgroups in elements.items():
        raw_exprs: List[str] = []

        for sub_key, expr in (subgroups or {}).items():
            expr = str(expr).strip()
            qname = f"{element_key}__{sub_key}"

            # Subgroup query: field-wrapped
            subgroup_queries[qname] = f"{field}({expr})"
            raw_exprs.append(f"({expr})")

        if raw_exprs:
            or_raw = " OR ".join(raw_exprs)
            element_raw_or[element_key] = f"({or_raw})"

            # Element aggregate: field-wrapped once
            element_queries[f"{element_key}__ALL"] = f"{field}({element_raw_or[element_key]})"

    # TOTAL combined: field-wrapped once
    combined_raw = " AND ".join([f"({v})" for v in element_raw_or.values()])
    combined_query = f"{field}({combined_raw})" if combined_raw else ""

    return subgroup_queries, element_queries, combined_query


def _plot_counts(summary: pd.DataFrame, out_png: str, title: str) -> None:
    """
    Colors bars by top-level protocol bucket and annotates counts above bars.
    Expected top-level keys (before '__'): P, C_concept, C_context_climate, C_context_agriculture, M, TOTAL
    """
    import matplotlib.patches as mpatches

    # Clean numeric totals (ignore failed rows)
    plot_df = summary.copy()
    plot_df["total_results_num"] = pd.to_numeric(plot_df["total_results"], errors="coerce")
    plot_df = plot_df.dropna(subset=["total_results_num"]).copy()

    def group_of(qname: str) -> str:
        if qname == "TOTAL__ALL":
            return "TOTAL"
        return qname.split("__", 1)[0]

    plot_df["group"] = plot_df["query_name"].astype(str).apply(group_of)

    # Protocol order (matches your 5 categories + total)
    group_order = ["P", "C_concept", "C_context_climate", "C_context_agriculture", "M", "TOTAL"]
    plot_df["group_order"] = plot_df["group"].apply(lambda g: group_order.index(g) if g in group_order else 999)
    plot_df = plot_df.sort_values(["group_order", "query_name"]).reset_index(drop=True)

    # Colors + legend labels
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

    # Slightly taller + wider for readability
    plt.figure(figsize=(max(14, 0.55 * len(labels)), 7))
    bars = plt.bar(x, totals, color=colors)

    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Scopus totalResults")
    plt.title(title)

    # Legend (only show groups that appear)
    present = []
    seen = set()
    for g in group_order:
        if g in set(groups) and g not in seen:
            present.append(g)
            seen.add(g)

    handles = [mpatches.Patch(color=color_map[g], label=legend_labels.get(g, g)) for g in present]
    if handles:
        plt.legend(handles=handles, loc="upper left", frameon=True)

    # Add counts above each bar
    y_max = max(totals) if totals else 0.0
    pad = max(1.0, y_max * 0.012)  # ~1.2% headroom (min 1)
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

    # More space above bars/labels
    plt.ylim(0, y_max + 18 * pad)

    # Extra top margin so the top labels don't collide with the border/title
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


# ============================================================
# Step 1: counts only (with per-query caching + SKIP logs)
# ============================================================
def step1_counts_only(auth: ScopusAuth) -> None:
    os.makedirs(STEP1_DIR, exist_ok=True)

    cfg = _read_yaml(SEARCH_STRINGS_YML)
    subgroup_q, element_q, combined_q = _build_queries(cfg)

    # Order: subgroup queries, element aggregates, total
    queries: List[Tuple[str, str]] = []
    queries += sorted(subgroup_q.items(), key=lambda x: x[0])
    queries += sorted(element_q.items(), key=lambda x: x[0])
    if combined_q:
        queries.append(("TOTAL__ALL", combined_q))

    queries_json_path = os.path.join(STEP1_DIR, STEP1_QUERIES_JSON)
    summary_csv_path = os.path.join(STEP1_DIR, STEP1_SUMMARY_CSV)
    plot_png_path = os.path.join(STEP1_DIR, STEP1_PLOT_PNG)
    plot_csv_path = os.path.splitext(plot_png_path)[0] + ".csv"  # same basename as plot
    total_query_txt_path = os.path.join(STEP1_DIR, STEP1_TOTAL_QUERY_TXT)
    cache_json_path = os.path.join(STEP1_DIR, STEP1_CACHE_JSON)

    # Save TOTAL query alone (easy copy/paste)
    if combined_q:
        with open(total_query_txt_path, "w", encoding="utf-8") as f:
            f.write(combined_q.strip() + "\n")

    # Write queries.json (always; cheap + helps debugging)
    sig = _queries_signature(queries)
    payload = [{"_signature": sig, "_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}]
    payload += [{"name": n, "query": q} for n, q in queries]
    with open(queries_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Load per-query cache
    cache = _load_json(cache_json_path, default={})
    if not isinstance(cache, dict):
        cache = {}

    rows = []
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    for qname, q in queries:
        qh = _query_hash(q)
        cached = cache.get(qname)

        # Skip unchanged query if cache exists
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

        # Update cache entry
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

    # Save cache
    with open(cache_json_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

    # Outputs
    summary = pd.DataFrame(rows)
    summary.to_csv(summary_csv_path, index=False)
    summary.to_csv(plot_csv_path, index=False)

    # Print the same results as a table
    print("\n[step1] Summary table:")
    cols = ["query_name", "total_results", "error"]
    print(summary[cols].to_string(index=False))

    # Plot
    _plot_counts(summary, plot_png_path, "Scopus record counts by query")


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

    print("\nDone.")
    print(f"Outputs: {STEP1_DIR}")


if __name__ == "__main__":
    main()
