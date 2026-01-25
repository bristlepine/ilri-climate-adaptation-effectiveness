#!/usr/bin/env python3
"""
step1_counts.py

Step 1 (counts only):
- Build subgroup queries + element aggregates + TOTAL combined query from search_strings.yml
- Fetch ONLY record counts (opensearch:totalResults) for each query (no record retrieval)
- Cache per-query counts so reruns skip unchanged queries
- Writes to <out_dir>/step1/
"""

import os
import time
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import yaml
import matplotlib.pyplot as plt


USE_POST_FOR_SEARCH = 1
VIEW = "STANDARD"
STEP1_SLEEP_S = 0.10

DEFAULT_SCOPUS_SEARCH_URL = "https://api.elsevier.com/content/search/scopus"


@dataclass
class ScopusAuth:
    api_key: str
    inst_token: Optional[str] = None


def _load_json(path: str, default):
    try:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            import json

            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default


def _save_json(path: str, obj) -> None:
    import json

    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _query_hash(q: str) -> str:
    return hashlib.sha256(q.strip().encode("utf-8")).hexdigest()


def _queries_signature(queries: List[Tuple[str, str]]) -> str:
    import json

    payload = [{"name": n, "query": q} for n, q in queries]
    s = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(s).hexdigest()


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

        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:800]}")

    raise RuntimeError(f"Failed after retries. Last rate headers: {last_rate}")


def scopus_count_only(scopus_search_url: str, auth: ScopusAuth, query: str, count_per_page: int = 1) -> Tuple[int, dict]:
    headers = _headers(auth)
    meta = {"query": query, "count_per_page": count_per_page, "view": VIEW, "rate_headers_last": {}}

    with requests.Session() as session:
        params = {"query": query, "start": "0", "count": str(count_per_page), "view": VIEW}
        method = "POST" if USE_POST_FOR_SEARCH else "GET"
        data, rate = _request_with_retries(session, method, scopus_search_url, headers, params=params, data=params)
        meta["rate_headers_last"] = rate

    sr = (data or {}).get("search-results", {}) or {}
    total = int(sr.get("opensearch:totalResults", "0"))
    return total, meta


def _build_limits_expr(cfg: dict) -> str:
    lim = cfg.get("limits")
    if not lim:
        return ""

    if isinstance(lim, str):
        s = lim.strip()
        return f"({s})" if s else ""

    parts: List[str] = []
    if isinstance(lim, dict):
        for v in lim.values():
            s = str(v).strip()
            if s:
                parts.append(f"({s})")
    elif isinstance(lim, list):
        for v in lim:
            s = str(v).strip()
            if s:
                parts.append(f"({s})")

    return " AND ".join(parts)


def _build_queries(cfg: dict) -> Tuple[Dict[str, str], Dict[str, str], str]:
    field = cfg.get("field", "TITLE-ABS-KEY").strip()
    elements = cfg.get("elements", {}) or {}
    limits_expr = _build_limits_expr(cfg)

    subgroup_queries: Dict[str, str] = {}
    element_queries: Dict[str, str] = {}
    element_raw_or: Dict[str, str] = {}

    for element_key, subgroups in elements.items():
        raw_exprs: List[str] = []
        for sub_key, expr in (subgroups or {}).items():
            expr = str(expr).strip()
            qname = f"{element_key}__{sub_key}"
            q = f"{field}({expr})"
            if limits_expr:
                q = f"{q} AND {limits_expr}"
            subgroup_queries[qname] = q
            raw_exprs.append(f"({expr})")

        if raw_exprs:
            or_raw = " OR ".join(raw_exprs)
            element_raw_or[element_key] = f"({or_raw})"
            q_all = f"{field}({element_raw_or[element_key]})"
            if limits_expr:
                q_all = f"{q_all} AND {limits_expr}"
            element_queries[f"{element_key}__ALL"] = q_all

    combined_raw = " AND ".join([f"({v})" for v in element_raw_or.values()])
    combined_query = f"{field}({combined_raw})" if combined_raw else ""
    if combined_query and limits_expr:
        combined_query = f"{combined_query} AND {limits_expr}"
    return subgroup_queries, element_queries, combined_query


def _normalize_group(g: str) -> str:
    gl = (g or "").lower()
    if gl in ("c_context_limcs", "c_context_lmics"):
        return "C_context_LMICs"
    return g


def _plot_counts(summary: pd.DataFrame, out_png: str, title: str, timestamp_utc: str) -> None:
    import matplotlib.patches as mpatches

    plot_df = summary.copy()
    plot_df["total_results_num"] = pd.to_numeric(plot_df["total_results"], errors="coerce")
    plot_df = plot_df.dropna(subset=["total_results_num"]).copy()

    def group_of(qname: str) -> str:
        if qname == "TOTAL__ALL":
            return "TOTAL"
        return _normalize_group(qname.split("__", 1)[0])

    plot_df["group"] = plot_df["query_name"].astype(str).apply(group_of)
    group_order = ["P", "C_concept", "C_context_climate", "C_context_agriculture", "C_context_LMICs", "M", "TOTAL"]
    plot_df["group_order"] = plot_df["group"].apply(lambda g: group_order.index(g) if g in group_order else 999)
    plot_df = plot_df.sort_values(["group_order", "query_name"]).reset_index(drop=True)

    color_map = {
        "P": "#1f77b4",
        "C_concept": "#ff7f0e",
        "C_context_climate": "#2ca02c",
        "C_context_agriculture": "#d62728",
        "C_context_LMICs": "#8c564b",
        "M": "#9467bd",
        "TOTAL": "#111111",
    }
    legend_labels = {
        "P": "Population (P)",
        "C_concept": "Concept (C)",
        "C_context_climate": "Context (C) climate",
        "C_context_agriculture": "Context (C) agriculture",
        "C_context_LMICs": "Context (C) LMICs",
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
    plt.figtext(
        0.99,
        0.01,
        f"Generated: {timestamp_utc.replace('T', ' ').replace('Z', ' UTC')}",
        ha="right",
        va="bottom",
        fontsize=8,
        color="gray",
    )
    plt.savefig(out_png, dpi=200)
    plt.close()


def run(config: dict) -> dict:
    api_key = os.getenv("SCOPUS_API_KEY")
    inst_token = os.getenv("SCOPUS_INST_TOKEN")

    if not api_key:
        raise SystemExit("SCOPUS_API_KEY missing (set it in .env at repo root).")

    auth = ScopusAuth(api_key=api_key, inst_token=inst_token)

    out_dir = config["out_dir"]
    step_dir = os.path.join(out_dir, "step1")
    os.makedirs(step_dir, exist_ok=True)

    search_yml = config["search_strings_yml"]

    endpoints = config.get("endpoints") or {}
    scopus_search_url = (
        endpoints.get("scopus_search_url")
        or endpoints.get("scopus_search")
        or DEFAULT_SCOPUS_SEARCH_URL
    )

    step1_queries_json = os.path.join(step_dir, "step1_queries.json")
    step1_summary_csv = os.path.join(step_dir, "step1_summary.csv")
    step1_summary_json = os.path.join(step_dir, "step1_summary.json")
    step1_plot_png = os.path.join(step_dir, "step1_hits_plot.png")
    step1_plot_csv = os.path.join(step_dir, "step1_hits_plot.csv")
    step1_total_query_txt = os.path.join(step_dir, "step1_total_query.txt")
    step1_cache_json = os.path.join(step_dir, "step1_counts_cache.json")

    cfg_yml = _read_yaml(search_yml)
    subgroup_q, element_q, combined_q = _build_queries(cfg_yml)

    queries: List[Tuple[str, str]] = []
    queries += sorted(subgroup_q.items(), key=lambda x: x[0])
    queries += sorted(element_q.items(), key=lambda x: x[0])
    if combined_q:
        queries.append(("TOTAL__ALL", combined_q))

    if combined_q:
        with open(step1_total_query_txt, "w", encoding="utf-8") as f:
            f.write(combined_q.strip() + "\n")

    sig = _queries_signature(queries)
    payload = [{"_signature": sig, "_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}]
    payload += [{"name": n, "query": q} for n, q in queries]
    _save_json(step1_queries_json, payload)

    cache = _load_json(step1_cache_json, default={})
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
            total, meta = scopus_count_only(scopus_search_url, auth, q, count_per_page=1)
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

    _save_json(step1_cache_json, cache)

    summary = pd.DataFrame(rows)

    def _group_of(qname: str) -> str:
        if qname == "TOTAL__ALL":
            return "TOTAL"
        return _normalize_group(qname.split("__", 1)[0])

    summary["_group"] = summary["query_name"].astype(str).apply(_group_of)

    total_all = (
        summary.loc[summary["query_name"] == "TOTAL__ALL", "total_results"]
        .dropna()
        .astype(int)
        .tolist()
    )
    total_all = total_all[0] if total_all else None

    by_element = (
        summary[summary["_group"] != "TOTAL"]
        .groupby("_group")["total_results"]
        .max()
        .dropna()
        .astype(int)
        .to_dict()
    )

    by_subgroup = (
        summary[
            (~summary["query_name"].str.endswith("__ALL")) &
            (summary["query_name"] != "TOTAL__ALL")
        ]
        .set_index("query_name")["total_results"]
        .dropna()
        .astype(int)
        .to_dict()
    )

    step1_summary_obj = {
        "timestamp_utc": ts,
        "query_signature": sig,
        "total_all": total_all,
        "by_element": by_element,
        "by_subgroup": by_subgroup,
    }

    _save_json(step1_summary_json, step1_summary_obj)
    summary.to_csv(step1_summary_csv, index=False)
    summary.to_csv(step1_plot_csv, index=False)

    print("\n[step1] Summary table:")
    print(summary[["query_name", "total_results", "error"]].to_string(index=False))

    _plot_counts(
        summary,
        step1_plot_png,
        "Scopus record counts by query",
        ts,
    )

    return {
        "step_dir": step_dir,
        "summary_csv": step1_summary_csv,
        "plot_png": step1_plot_png,
        "total_query_txt": step1_total_query_txt,
        "cache_json": step1_cache_json,
        "queries_json": step1_queries_json,
        "summary_json": step1_summary_json,
    }
