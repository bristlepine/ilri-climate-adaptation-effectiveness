#!/usr/bin/env python3
"""
run.py

Orchestrates Scopus pipeline steps using flags from config.py,
with consistent step console logging.
"""

import logging
import sys
import time
from typing import Any, Callable, Optional

from dotenv import load_dotenv

import config as cfg

import step1_counts as step1
import step2_retrieve as step2
import step3_benchmark as step3
import step4_abstracts as step4
import step5_eligibility as step5
import step6_visualize as step6
import step7_scopus_check as step7  # <--- Added Step 7 Import


# ----------------------------
# Logging setup
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)

ICONS = {
    "start": "🎬",
    "finish": "🎯",
    "run": "▶️",
    "ok": "✅",
    "skip": "⏭️",
    "fail": "❌",
}

STEP_DIV = "-" * 60


def build_config_dict() -> dict:
    """Expose config.py values to step modules as one dict."""
    return {
        "out_dir": cfg.out_dir,
        "search_strings_yml": cfg.search_strings_yml,
        "benchmark_csv": cfg.benchmark_csv,
        "endpoints": cfg.endpoints,

        "run_step1": cfg.run_step1,
        "run_step2": cfg.run_step2,
        "run_step3": cfg.run_step3,
        "run_step4": cfg.run_step4,
        "run_step5": cfg.run_step5,
        "run_step6": cfg.run_step6,
        "run_step7": cfg.run_step7, # <--- Added Step 7 Flag
    }


def log_step_header(step_num: int, title: str, module: str, skipped: bool) -> None:
    if skipped:
        logger.info("%s Step %d: %s [%s] (skipped)\n%s", ICONS["skip"], step_num, title, module, STEP_DIV)
    else:
        logger.info("%s Step %d: %s [%s]\n%s", ICONS["run"], step_num, title, module, STEP_DIV)


def log_step_result(step_num: int, module: str, ok: bool, elapsed_s: Optional[float] = None, extra: str | None = None) -> None:
    icon = ICONS["ok"] if ok else ICONS["fail"]
    t = f" — {elapsed_s:.1f}s" if elapsed_s is not None else ""
    tail = f" — {extra}" if extra else ""
    logger.info("%s Step %d complete [%s]%s%s", icon, step_num, module, t, tail)


def resolve_callable(module: Any, candidates: list[str]) -> Callable[[dict], Any]:
    """Find a callable in a module by name (lets you keep step module APIs simple)."""
    for name in candidates:
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    raise AttributeError(f"{module.__name__} missing callable. Expected one of: {', '.join(candidates)}")


def run_step(enabled: bool, step_num: int, title: str, module_label: str, fn: Callable[[dict], Any], config: dict) -> dict:
    if not enabled:
        log_step_header(step_num, title, module_label, skipped=True)
        return {"status": "skipped"}

    log_step_header(step_num, title, module_label, skipped=False)
    t0 = time.time()
    try:
        result = fn(config)
        log_step_result(step_num, module_label, ok=True, elapsed_s=time.time() - t0)
        return {"status": "ok", "result": result}
    except Exception as e:
        log_step_result(step_num, module_label, ok=False, elapsed_s=time.time() - t0, extra=f"{type(e).__name__}: {e}")
        raise


def main() -> None:
    load_dotenv()
    config = build_config_dict()

    logger.info("%s Scopus pipeline", ICONS["start"])

    run_step(
        config.get("run_step1", 1),
        1,
        "Counts only",
        "step1_counts",
        resolve_callable(step1, ["run", "main", "run_step1", "step1_counts_only"]),
        config,
    )

    run_step(
        config.get("run_step2", 1),
        2,
        "Retrieve TOTAL__ALL records",
        "step2_retrieve",
        resolve_callable(step2, ["run", "main", "run_step2", "run_step1b_retrieve_total", "step2_retrieve_total"]),
        config,
    )

    run_step(
        config.get("run_step3", 1),
        3,
        "Benchmark DOI match",
        "step3_benchmark",
        resolve_callable(step3, ["run", "main", "run_step3", "step3_benchmark_match"]),
        config,
    )

    run_step(
        config.get("run_step4", 1),
        4,
        "Fetch abstracts",
        "step4_abstracts",
        resolve_callable(step4, ["run", "main", "run_step4", "step4_fetch_abstracts"]),
        config,
    )

    run_step(
        config.get("run_step5", 1),
        5,
        "Eligibility check",
        "step5_eligibility",
        resolve_callable(step5, ["run", "main", "run_step5"]),
        config,
    )

    run_step(
        config.get("run_step6", 1),
        6,
        "Visualize Heatmap",
        "step6_visualize",
        resolve_callable(step6, ["run", "main", "run_step6"]),
        config,
    )

    # --- Added Step 7 ---
    run_step(
        config.get("run_step7", 1),
        7,
        "Check Benchmark vs Scopus",
        "step7_scopus_check",
        resolve_callable(step7, ["run", "main", "run_step7"]),
        config,
    )

    logger.info("%s Pipeline completed successfully", ICONS["finish"])


if __name__ == "__main__":
    main()