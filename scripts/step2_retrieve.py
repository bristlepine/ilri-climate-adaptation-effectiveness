#!/usr/bin/env python3
"""
step2_retrieve.py

Step 2 (retrieve TOTAL__ALL records; streaming):
  - Avoid deep paging limits by slicing TOTAL query into PUBYEAR ranges (<= DEEP_PAGING_LIMIT)
  - Stream each slice to CSV, then concatenate -> outputs/step2/step2_total_records.csv

Designed to work with scripts/run.py "new runner" style:
  - provides run_step1b_retrieve_total(config)
"""

from __future__ import annotations

import os
import json
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

import config as cfg

# -----------------------------
# Defaults (can be overridden via config dict)
# -----------------------------
DEFAULT_USE_POST_FOR_SEARCH = 1   # avoids 413 for long queries
DEFAULT_VIEW = "STANDARD"

DEFAULT_COUNT_PER_PAGE_RETRIEVE = 25
DEFAULT_STEP2_SLEEP_S = 0.15
DEFAULT_DEEP_PAGING_LIMIT = 5000
DEFAULT_PUBYEAR_MIN = 1990
DEFAULT_PUBYEAR_MAX = 2025
DEFAULT_MAX_RESULTS_TOTAL = None  # set e.g. 500 to test, or None for all


# -----------------------------
# Small JSON helpers
# -----------------------------
def _save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# -----------------------------
# Auth helpers (works even if utils.py changes)
# -----------------------------
def _headers(api_key: str, inst_token: Optional[str] = None) -> Dict[str, str]:
    h = {"X-ELS-APIKey": api_key, "Accept": "application/json"}
    if inst_token:
        h["X-ELS-Insttoken"] = inst_token
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


# -----------------------------
# Query builder (same as scopus.py)
# -----------------------------
def _build_queries(search_cfg: dict) -> Tuple[Dict[str, str], Dict[str, str], str]:
    field = (search_cfg.get("field", "TITLE-ABS-KEY") or "TITLE-ABS-KEY").strip()
    elements = search_cfg.get("elements", {}) or {}

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


# -----------------------------
# Step 2 logic (copied with minimal change)
# -----------------------------
def scopus_count_only(
    api_key: str,
    inst_token: Optional[str],
    scopus_search_url: str,
    query: str,
    use_post: int,
    view: str,
    count_per_page: int = 1,
) -> Tuple[int, dict]:
    headers = _headers(api_key, inst_token)
    meta = {"query": query, "count_per_page": count_per_page, "view": view, "rate_headers_last": {}}

    with requests.Session() as session:
        params = {"query": query, "start": "0", "count": str(count_per_page), "view": view}
        method = "POST" if use_post else "GET"
        data, rate = _request_with_retries(session, method, scopus_search_url, headers, params=params, data=params)
        meta["rate_headers_last"] = rate

    sr = (data or {}).get("search-results", {}) or {}
    total = int(sr.get("opensearch:totalResults", "0"))
    return total, meta


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


def _plan_year_slices(
    api_key: str,
    inst_token: Optional[str],
    scopus_search_url: str,
    base_query: str,
    y0: int,
    y1: int,
    *,
    deep_paging_limit: int,
    use_post: int,
    view: str,
) -> List[Tuple[int, int, int]]:
    total, _ = scopus_count_only(
        api_key, inst_token, scopus_search_url,
        _with_pubyear_range(base_query, y0, y1),
        use_post=use_post,
        view=view,
        count_per_page=1,
    )
    if total <= deep_paging_limit or y0 == y1:
        return [(y0, y1, total)]
    mid = (y0 + y1) // 2
    return (
        _plan_year_slices(api_key, inst_token, scopus_search_url, base_query, y0, mid,
                          deep_paging_limit=deep_paging_limit, use_post=use_post, view=view)
        + _plan_year_slices(api_key, inst_token, scopus_search_url, base_query, mid + 1, y1,
                            deep_paging_limit=deep_paging_limit, use_post=use_post, view=view)
    )


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
    api_key: str,
    inst_token: Optional[str],
    scopus_search_url: str,
    *,
    query: str,
    out_csv: str,
    count_per_page: int,
    sleep_s: float,
    max_results: Optional[int],
    use_post: int,
    view: str,
) -> Tuple[int, int, dict]:
    headers = _headers(api_key, inst_token)
    method = "POST" if use_post else "GET"

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if os.path.exists(out_csv):
        os.remove(out_csv)

    retrieved = 0
    total_reported = None

    meta = {
        "query": query,
        "view": view,
        "count_per_page": count_per_page,
        "sleep_s": sleep_s,
        "max_results": max_results,
        "rate_headers_last": {},
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    pbar = None
    with requests.Session() as session:
        while True:
            if max_results is not None and retrieved >= max_results:
                break

            params = {"query": query, "start": str(retrieved), "count": str(count_per_page), "view": view}
            data, rate = _request_with_retries(session, method, scopus_search_url, headers, params=params, data=params)
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


def step2_retrieve_total(config: dict) -> dict:
    """
    Pure function for runner: reads env/config, writes outputs, returns a small summary dict.
    """

    load_dotenv()

    api_key = (
        config.get("scopus_api_key")
        or os.getenv("SCOPUS_API_KEY")
    )
    inst_token = (
        config.get("scopus_inst_token")
        or os.getenv("SCOPUS_INST_TOKEN")
    )

    if not api_key:
        raise SystemExit("SCOPUS_API_KEY missing (set it in .env at repo root).")

    scopus_search_url = (
        config.get("scopus_search_url")
        or getattr(cfg, "scopus_search_url", None)
    )
    if not scopus_search_url:
        raise SystemExit("SCOPUS_SEARCH_URL missing in config.py (or config dict).")

    search_strings_yml = config.get("search_strings_yml") or getattr(cfg, "SEARCH_STRINGS_YML", None)
    if not search_strings_yml:
        raise SystemExit("SEARCH_STRINGS_YML missing in config.py (or config dict).")

    out_dir = config.get("out_dir") or getattr(cfg, "out_dir", None) or getattr(cfg, "OUT_DIR", None)
    if not out_dir:
        raise SystemExit("out_dir/OUT_DIR missing in config.py (or config dict).")

    step2_dir = os.path.join(out_dir, "step2")
    step2_total_csv = os.path.join(step2_dir, "step2_total_records.csv")
    step2_total_meta_json = os.path.join(step2_dir, "step2_total_records.meta.json")

    # knobs (override-friendly)
    use_post = int(config.get("use_post_for_search", getattr(cfg, "USE_POST_FOR_SEARCH", DEFAULT_USE_POST_FOR_SEARCH)))
    view = str(config.get("view", getattr(cfg, "VIEW", DEFAULT_VIEW)))

    count_per_page = int(config.get("count_per_page_retrieve", getattr(cfg, "COUNT_PER_PAGE_RETRIEVE", DEFAULT_COUNT_PER_PAGE_RETRIEVE)))
    sleep_s = float(config.get("step2_sleep_s", getattr(cfg, "STEP2_SLEEP_S", DEFAULT_STEP2_SLEEP_S)))
    deep_paging_limit = int(config.get("deep_paging_limit", getattr(cfg, "DEEP_PAGING_LIMIT", DEFAULT_DEEP_PAGING_LIMIT)))
    pubyear_min = int(config.get("pubyear_min", getattr(cfg, "PUBYEAR_MIN", DEFAULT_PUBYEAR_MIN)))
    pubyear_max = int(config.get("pubyear_max", getattr(cfg, "PUBYEAR_MAX", DEFAULT_PUBYEAR_MAX)))
    max_results_total = config.get("max_results_total", getattr(cfg, "MAX_RESULTS_TOTAL", DEFAULT_MAX_RESULTS_TOTAL))

    os.makedirs(step2_dir, exist_ok=True)

    if os.path.exists(step2_total_csv) and os.path.getsize(step2_total_csv) > 0:
        print(f"[step2] SKIP (exists): {step2_total_csv}")
        return {"status": "skipped", "reason": "already_exists", "step2_total_csv": step2_total_csv}

    search_cfg = _read_yaml(search_strings_yml)
    _, _, base_q = _build_queries(search_cfg)
    if not base_q:
        raise SystemExit("[step2] TOTAL__ALL query is empty; check search_strings.yml")

    base_total, _ = scopus_count_only(
        api_key, inst_token, scopus_search_url, base_q,
        use_post=use_post, view=view, count_per_page=1
    )
    print(f"[step2] TOTAL__ALL reported by Scopus: {base_total:,}")

    print("[step2] Planning PUBYEAR slices...")
    slices = _plan_year_slices(
        api_key, inst_token, scopus_search_url, base_q, pubyear_min, pubyear_max,
        deep_paging_limit=deep_paging_limit, use_post=use_post, view=view
    )

    missing_q = _with_pubyear_missing_or_outside(base_q, pubyear_min, pubyear_max)
    missing_total, _ = scopus_count_only(
        api_key, inst_token, scopus_search_url, missing_q,
        use_post=use_post, view=view, count_per_page=1
    )

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

        out_csv = os.path.join(step2_dir, f"step2_total_records__{tag}.csv")
        out_meta = os.path.join(step2_dir, f"step2_total_records__{tag}.meta.json")

        if os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
            print(f"[step2] SKIP {tag} (exists): {out_csv}")
            slice_csvs.append(out_csv)
            continue

        print(f"\n[step2] RETRIEVE {tag} (expected {slice_total:,})")
        total_reported, retrieved, meta = scopus_retrieve_stream_to_csv_start(
            api_key, inst_token, scopus_search_url,
            query=q_slice,
            out_csv=out_csv,
            count_per_page=count_per_page,
            sleep_s=sleep_s,
            max_results=max_results_total,
            use_post=use_post,
            view=view,
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
        out_csv = os.path.join(step2_dir, f"step2_total_records__{tag}.csv")
        out_meta = os.path.join(step2_dir, f"step2_total_records__{tag}.meta.json")

        if os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
            print(f"[step2] SKIP {tag} (exists): {out_csv}")
            slice_csvs.append(out_csv)
        else:
            print(f"\n[step2] RETRIEVE {tag} (expected {missing_total:,})")
            total_reported, retrieved, meta = scopus_retrieve_stream_to_csv_start(
                api_key, inst_token, scopus_search_url,
                query=missing_q,
                out_csv=out_csv,
                count_per_page=count_per_page,
                sleep_s=sleep_s,
                max_results=max_results_total,
                use_post=use_post,
                view=view,
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

    print(f"\n[step2] Combining slice CSVs -> {step2_total_csv}")
    combined_rows = _concat_csvs(slice_csvs, step2_total_csv)

    meta_out = {
        "base_total_reported_by_scopus": int(base_total),
        "planned_total_across_slices": int(planned_total),
        "combined_rows_written": int(combined_rows),
        "pubyear_min": int(pubyear_min),
        "pubyear_max": int(pubyear_max),
        "deep_paging_limit": int(deep_paging_limit),
        "count_per_page": int(count_per_page),
        "sleep_s": float(sleep_s),
        "max_results_total": max_results_total,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "slice_metas": slice_metas,
    }
    _save_json(step2_total_meta_json, meta_out)

    print(f"[step2] Final combined rows: {combined_rows:,}")
    if combined_rows != base_total:
        print(f"[step2] WARNING: combined_rows ({combined_rows:,}) != base_total ({base_total:,}).")

    return {
        "status": "ok",
        "step2_total_csv": step2_total_csv,
        "step2_total_meta_json": step2_total_meta_json,
        "base_total_reported_by_scopus": int(base_total),
        "combined_rows_written": int(combined_rows),
    }


# -----------------------------
# Runner entrypoint (this is what run.py should call)
# -----------------------------
def run_step1b_retrieve_total(config: dict) -> dict:
    return step2_retrieve_total(config)


if __name__ == "__main__":
    # optional direct execution (mostly for debugging)
    step2_retrieve_total({})


# -----------------------------
# Aliases expected by scripts/run.py
# -----------------------------
def run(config: dict) -> dict:
    return step2_retrieve_total(config)

def run_step2(config: dict) -> dict:
    return step2_retrieve_total(config)

def main(config: dict | None = None) -> dict:
    return step2_retrieve_total(config or {})
