#!/usr/bin/env python3
"""
step11_analysis.py

Step 11: Inter-rater reliability analysis for calibration rounds.

For each CSV in scripts/results/:
  - Auto-detects reviewer columns and a reconciled column (name contains 'reconcil')
  - Computes Cohen's kappa (pairwise between all raters)
  - Computes per-reviewer confusion breakdown vs reconciled decision

Outputs (under outputs/step11/):
  - <stem>_analysis.png   — 3-panel summary figure
  - <stem>_summary.json  — numeric summary
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# =============================================================================
# CONFIG
# =============================================================================

RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs" / "step11"

INCLUDE_LABEL = "INCLUDE"
EXCLUDE_LABEL = "EXCLUDE"
SKIP_COLS = {"id", "item", "ref", "title", "year", "doi"}  # lowercase

# Colours
C_INCLUDE    = "#27ae60"   # green  — INCLUDE bar / both-include
C_EXCLUDE    = "#c0392b"   # red    — EXCLUDE bar
C_BOTH_EXC   = "#95a5a6"   # gray   — both exclude (agreement)
C_OVER       = "#e67e22"   # orange — rater includes, reconciled excludes
C_MISS       = "#8e44ad"   # purple — rater excludes, reconciled includes


# =============================================================================
# HELPERS
# =============================================================================

def short_name(col: str) -> str:
    """First word, or 'Reconciled' for the reconciled column."""
    if "reconcil" in col.lower():
        return "Reconciled"
    return col.split()[0]


def _upper(series: pd.Series) -> pd.Series:
    """Uppercase string normalization that works on any dtype (handles NaN columns)."""
    return series.apply(lambda x: str(x).upper() if pd.notna(x) else x)


def encode(series: pd.Series) -> np.ndarray:
    """INCLUDE -> 1, EXCLUDE -> 0, else NaN (case-insensitive)."""
    return _upper(series).map({INCLUDE_LABEL: 1, EXCLUDE_LABEL: 0}).to_numpy(dtype=float)


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa for two binary arrays (0/1 only, no NaN)."""
    p_o = np.mean(a == b)
    p_e = np.mean(a) * np.mean(b) + (1 - np.mean(a)) * (1 - np.mean(b))
    if abs(1.0 - p_e) < 1e-9:
        return 1.0
    return float((p_o - p_e) / (1 - p_e))


def detect_columns(df: pd.DataFrame) -> Tuple[List[str], Optional[str]]:
    """Return (reviewer_cols, reconciled_col). Skips id/item/unnamed/blank columns."""
    reconciled_col: Optional[str] = None
    reviewer_cols: List[str] = []
    for c in df.columns:
        c_str = str(c).strip()
        # Skip unnamed columns (empty header or pandas Unnamed: N default)
        if not c_str or c_str.lower().startswith("unnamed:"):
            continue
        # Skip columns that are entirely empty
        if df[c].isna().all():
            continue
        if any(s in c_str.lower() for s in SKIP_COLS):
            continue
        if "reconcil" in c_str.lower():
            reconciled_col = c
        else:
            reviewer_cols.append(c)
    return reviewer_cols, reconciled_col


def pairwise_kappa(df: pd.DataFrame, cols: List[str]) -> Dict[Tuple[str, str], Optional[float]]:
    result: Dict[Tuple[str, str], Optional[float]] = {}
    for a, b in combinations(cols, 2):
        mask = df[a].notna() & df[b].notna()
        ea = encode(df.loc[mask, a])
        eb = encode(df.loc[mask, b])
        valid = ~(np.isnan(ea) | np.isnan(eb))
        if valid.sum() < 2:
            result[(a, b)] = None
        else:
            result[(a, b)] = round(cohen_kappa(ea[valid], eb[valid]), 3)
    return result


def confusion_vs_reconciled(
    df: pd.DataFrame,
    reviewer_cols: List[str],
    reconciled_col: str,
) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for r in reviewer_cols:
        mask = df[r].notna() & df[reconciled_col].notna()
        rv = df.loc[mask, r]
        rc = df.loc[mask, reconciled_col]
        tp = int(((_upper(rv) == INCLUDE_LABEL) & (_upper(rc) == INCLUDE_LABEL)).sum())
        tn = int((((_upper(rv) == EXCLUDE_LABEL) & (_upper(rc) == EXCLUDE_LABEL)).sum()))
        fp = int(((_upper(rv) == INCLUDE_LABEL) & (_upper(rc) == EXCLUDE_LABEL)).sum())
        fn = int(((_upper(rv) == EXCLUDE_LABEL) & (_upper(rc) == INCLUDE_LABEL)).sum())
        total = tp + tn + fp + fn
        ea, eb = encode(rv), encode(rc)
        valid = ~(np.isnan(ea) | np.isnan(eb))
        k = cohen_kappa(ea[valid], eb[valid]) if valid.sum() >= 2 else None
        out[r] = {
            "both_include":   tp,
            "both_exclude":   tn,
            "over_include":   fp,
            "under_include":  fn,
            "pct_agreement":  round(100 * (tp + tn) / total, 1) if total else 0.0,
            "kappa_vs_reconciled": round(k, 3) if k is not None else None,
        }
    return out


def compute_stats(
    df: pd.DataFrame,
    reviewer_cols: List[str],
    reconciled_col: Optional[str],
) -> Dict:
    all_raters = reviewer_cols + ([reconciled_col] if reconciled_col else [])

    rater_stats: Dict[str, Dict] = {}
    for r in all_raters:
        n_inc = int((_upper(df[r]) == INCLUDE_LABEL).sum())
        n_exc = int((_upper(df[r]) == EXCLUDE_LABEL).sum())
        total = n_inc + n_exc
        rater_stats[r] = {
            "n_include":   n_inc,
            "n_exclude":   n_exc,
            "pct_include": round(100 * n_inc / total, 1) if total else 0.0,
        }

    kappa_pairs = pairwise_kappa(df, all_raters)
    conf = confusion_vs_reconciled(df, reviewer_cols, reconciled_col) if reconciled_col else {}

    return {
        "n_items": len(df),
        "rater_stats": rater_stats,
        "pairwise_kappa": {f"{a} vs {b}": v for (a, b), v in kappa_pairs.items()},
        "confusion_vs_reconciled": conf,
    }


# =============================================================================
# FIGURE
# =============================================================================

def build_figure(
    df: pd.DataFrame,
    reviewer_cols: List[str],
    reconciled_col: Optional[str],
    stats: Dict,
    title: str,
    out_path: Path,
) -> None:
    all_raters = reviewer_cols + ([reconciled_col] if reconciled_col else [])
    n_raters   = len(all_raters)
    has_rec    = reconciled_col is not None

    sns.set_theme(style="whitegrid", font_scale=0.95)
    n_panels = 3 if has_rec else 2
    w_ratios = [3.5, max(n_raters * 1.1, 2.5), 3.2] if has_rec else [3.5, max(n_raters * 1.1, 2.5)]
    fig, axes = plt.subplots(
        1, n_panels, figsize=(sum(w_ratios) * 1.6, 5.5),
        gridspec_kw={"width_ratios": w_ratios},
    )
    axes = list(axes)
    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.02)

    n_total = stats["n_items"]
    shorts  = [short_name(r) for r in all_raters]

    # ------------------------------------------------------------------
    # Panel A – decision distribution (stacked horizontal bar)
    # ------------------------------------------------------------------
    ax = axes[0]
    n_inc = [stats["rater_stats"][r]["n_include"] for r in all_raters]
    n_exc = [stats["rater_stats"][r]["n_exclude"] for r in all_raters]
    pct   = [stats["rater_stats"][r]["pct_include"] for r in all_raters]
    y     = np.arange(n_raters)

    ax.barh(y, n_inc, color=C_INCLUDE, label="Include", height=0.6)
    ax.barh(y, n_exc, left=n_inc, color=C_EXCLUDE, label="Exclude", height=0.6)

    for i, (ni, p) in enumerate(zip(n_inc, pct)):
        ax.text(ni / 2, i, f"{p:.0f}%",
                ha="center", va="center", color="white", fontsize=8, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(shorts)
    if has_rec:
        ax.get_yticklabels()[-1].set_fontstyle("italic")
    ax.set_xlabel(f"Papers (n={n_total})")
    ax.set_title("Decision distribution")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, n_total * 1.05)
    ax.invert_yaxis()

    # ------------------------------------------------------------------
    # Panel B – Cohen's kappa heatmap (lower triangle + diagonal)
    # ------------------------------------------------------------------
    ax = axes[1]
    kappa_pairs = {(a, b): v
                   for key, v in stats["pairwise_kappa"].items()
                   for a, b in [key.split(" vs ", 1)]}

    km = np.full((n_raters, n_raters), np.nan)
    np.fill_diagonal(km, 1.0)
    for i, a in enumerate(all_raters):
        for j, b in enumerate(all_raters):
            if i == j:
                continue
            v = kappa_pairs.get((a, b)) or kappa_pairs.get((b, a))
            if v is not None:
                km[i, j] = v

    # Mask upper triangle
    mask = np.triu(np.ones_like(km, dtype=bool), k=1)

    # Build annotation strings
    annot = np.full_like(km, "", dtype=object)
    for i in range(n_raters):
        for j in range(n_raters):
            if mask[i, j]:
                continue
            v = km[i, j]
            annot[i, j] = "—" if np.isnan(v) else ("1.0" if v == 1.0 else f"{v:.2f}")

    sns.heatmap(
        km, mask=mask, annot=annot, fmt="",
        cmap="RdYlGn", vmin=0, vmax=1,
        xticklabels=shorts, yticklabels=shorts,
        ax=ax, linewidths=0.5, square=True,
        cbar_kws={"shrink": 0.75, "label": "κ"},
        annot_kws={"size": 10, "weight": "bold"},
    )
    ax.set_title("Pairwise Cohen's κ")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)

    # ------------------------------------------------------------------
    # Panel C – confusion breakdown vs reconciled
    # ------------------------------------------------------------------
    if has_rec:
        ax = axes[2]
        conf  = stats["confusion_vs_reconciled"]
        rnames = [short_name(r) for r in reviewer_cols]
        y = np.arange(len(reviewer_cols))

        tp_v = [conf[r]["both_include"]  for r in reviewer_cols]
        tn_v = [conf[r]["both_exclude"]  for r in reviewer_cols]
        fp_v = [conf[r]["over_include"]  for r in reviewer_cols]
        fn_v = [conf[r]["under_include"] for r in reviewer_cols]

        left1 = tp_v
        left2 = [a + b for a, b in zip(tp_v, tn_v)]
        left3 = [a + b for a, b in zip(left2, fp_v)]

        ax.barh(y, tp_v,  color=C_INCLUDE,  label="Both Include",   height=0.6)
        ax.barh(y, tn_v,  left=left1, color=C_BOTH_EXC, label="Both Exclude",   height=0.6)
        ax.barh(y, fp_v,  left=left2, color=C_OVER,     label="Over-include",   height=0.6)
        ax.barh(y, fn_v,  left=left3, color=C_MISS,     label="Under-include",  height=0.6)

        # Annotate % agreement + kappa
        for i, r in enumerate(reviewer_cols):
            pct_a = conf[r]["pct_agreement"]
            k_val = conf[r]["kappa_vs_reconciled"]
            k_str = f"  κ={k_val:.2f}" if k_val is not None else ""
            ax.text(n_total * 1.01, i, f"{pct_a:.0f}%{k_str}",
                    va="center", ha="left", fontsize=8)

        ax.set_yticks(y)
        ax.set_yticklabels(rnames)
        rec_short = short_name(reconciled_col)
        ax.set_title(f"Agreement vs {rec_short}")
        ax.set_xlabel("Papers")
        ax.set_xlim(0, n_total * 1.3)
        ax.legend(loc="lower right", fontsize=7.5, ncol=2)
        ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[step11] Saved: {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def process_file(csv_path: Path, out_dir: Path) -> None:
    print(f"[step11] Processing: {csv_path.name}")
    df = pd.read_csv(csv_path)

    reviewer_cols, reconciled_col = detect_columns(df)
    if not reviewer_cols:
        print(f"[step11] No reviewer columns found in {csv_path.name}, skipping.")
        return

    print(f"[step11]   Reviewers : {reviewer_cols}")
    print(f"[step11]   Reconciled: {reconciled_col}")

    stats = compute_stats(df, reviewer_cols, reconciled_col)

    # Print quick console summary
    print(f"[step11]   n={stats['n_items']}  |  % include per rater:")
    for r, s in stats["rater_stats"].items():
        print(f"           {short_name(r):15s}  include={s['n_include']:3d}  exclude={s['n_exclude']:3d}  ({s['pct_include']:.0f}%)")
    print(f"[step11]   Pairwise kappa:")
    for pair, k in stats["pairwise_kappa"].items():
        print(f"           {pair}: {k}")
    if reconciled_col:
        print(f"[step11]   Agreement vs reconciled:")
        for r, c in stats["confusion_vs_reconciled"].items():
            print(f"           {short_name(r):15s}  {c['pct_agreement']:.0f}%  κ={c['kappa_vs_reconciled']}")

    stem = csv_path.stem.replace(" ", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = out_dir / f"{stem}_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[step11] Saved: {json_path}")

    # Save figure
    fig_path = out_dir / f"{stem}_analysis.png"
    build_figure(
        df, reviewer_cols, reconciled_col, stats,
        title=csv_path.stem,
        out_path=fig_path,
    )


def run(config: dict) -> dict:
    """Pipeline entry point — called by run.py."""
    from pathlib import Path as _Path
    _here = _Path(__file__).resolve().parent
    _fallback = str(_here / "outputs")
    out_root = _Path(str((config or {}).get("out_dir", "") or _fallback))
    out_dir  = out_root / "step11"

    csv_files = sorted(RESULTS_DIR.glob("*.csv"))
    if not csv_files:
        print(f"[step11] No CSV files found in {RESULTS_DIR}")
        return {"status": "no_files"}

    results = {}
    for csv_path in csv_files:
        process_file(csv_path, out_dir)
        results[csv_path.name] = "ok"

    print("[step11] Done.")
    return {"status": "ok", "files": results}


def main() -> int:
    csv_files = sorted(RESULTS_DIR.glob("*.csv"))
    if not csv_files:
        print(f"[step11] No CSV files found in {RESULTS_DIR}")
        return 1

    out_dir = OUTPUTS_DIR
    for csv_path in csv_files:
        process_file(csv_path, out_dir)

    print("[step11] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
