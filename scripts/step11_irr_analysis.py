#!/usr/bin/env python3
"""
step11_irr_analysis.py

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
        sensitivity = round(tp / (tp + fn), 3) if (tp + fn) > 0 else None
        specificity = round(tn / (tn + fp), 3) if (tn + fp) > 0 else None
        precision   = round(tp / (tp + fp), 3) if (tp + fp) > 0 else None
        f1          = round(2 * precision * sensitivity / (precision + sensitivity), 3) \
                      if (precision is not None and sensitivity is not None
                          and (precision + sensitivity) > 0) else None
        out[r] = {
            "both_include":        tp,
            "both_exclude":        tn,
            "over_include":        fp,
            "under_include":       fn,
            "pct_agreement":       round(100 * (tp + tn) / total, 1) if total else 0.0,
            "kappa_vs_reconciled": round(k, 3) if k is not None else None,
            "sensitivity":         sensitivity,
            "specificity":         specificity,
            "precision":           precision,
            "f1":                  f1,
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

    if has_rec:
        fig, axes_grid = plt.subplots(
            2, 2, figsize=(14, 10),
            gridspec_kw={"width_ratios": [1, 1], "height_ratios": [1, 1]},
        )
        axes = [axes_grid[0, 0], axes_grid[0, 1], axes_grid[1, 0], axes_grid[1, 1]]
    else:
        fig, axes_grid = plt.subplots(1, 2, figsize=(10, 5))
        axes = list(axes_grid)

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)

    n_total = stats["n_items"]
    shorts  = [short_name(r) for r in all_raters]

    # ------------------------------------------------------------------
    # Panel A (top-left) – decision distribution (stacked horizontal bar)
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
    # Panel B (top-right) – Cohen's kappa heatmap (lower triangle + diagonal)
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
    # Panel C (bottom-left) – confusion breakdown vs reconciled
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

        # Annotate % agreement only (metrics now in Panel D)
        for i, r in enumerate(reviewer_cols):
            pct_a = conf[r]["pct_agreement"]
            k_val = conf[r]["kappa_vs_reconciled"]
            k_str = f" κ={k_val:.2f}" if k_val is not None else ""
            ax.text(n_total * 1.01, i, f"{pct_a:.0f}%{k_str}",
                    va="center", ha="left", fontsize=8)

        ax.set_yticks(y)
        ax.set_yticklabels(rnames)
        rec_short = short_name(reconciled_col)
        ax.set_title(f"Agreement vs {rec_short}")
        ax.set_xlabel("Papers")
        ax.set_xlim(0, n_total * 1.45)
        ax.legend(loc="lower right", fontsize=7.5, ncol=2)
        ax.invert_yaxis()

        # ------------------------------------------------------------------
        # Panel D (bottom-right) – metrics table (Sensitivity, Specificity, Precision, F1, κ)
        # ------------------------------------------------------------------
        ax = axes[3]
        ax.axis("off")

        # Metric | (benchmark display string, threshold for colouring or None = no formal threshold)
        BENCHMARKS = {
            "Sensitivity": ("≥ 0.95 [a]",  0.95),
            "Specificity": ("No threshold [b]", None),
            "Precision":   ("0.632 [c]",    0.632),
            "F1":          ("0.708 [c]",    0.708),
            "κ":           ("≥ 0.60 [d]",   0.60),
        }

        metric_keys = {
            "Sensitivity": "sensitivity",
            "Specificity": "specificity",
            "Precision":   "precision",
            "F1":          "f1",
            "κ":           "kappa_vs_reconciled",
        }

        col_labels = ["Metric", "Benchmark"] + [short_name(r) for r in reviewer_cols]
        table_data = []
        cell_colors = []

        for metric_name, (bench_str, bench_val) in BENCHMARKS.items():
            key = metric_keys[metric_name]
            row_vals = [metric_name, bench_str]
            row_colors = [["#f0f0f0", "#f0f0f0"]]
            for r in reviewer_cols:
                val = conf[r].get(key)
                val_str = f"{val:.3f}" if val is not None else "—"
                if val is not None and bench_val is not None:
                    color = "#d5f5d5" if val >= bench_val else "#fde0dc"
                else:
                    color = "#ffffff"
                row_vals.append(val_str)
                row_colors[0].append(color)
            table_data.append(row_vals)
            cell_colors.append(row_colors[0])

        tbl = ax.table(
            cellText=table_data,
            colLabels=col_labels,
            cellLoc="center",
            loc="center",
            cellColours=cell_colors,
            bbox=[0.0, 0.25, 1.0, 0.65],  # [left, bottom, width, height] in axes coords
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        tbl.scale(1, 1.6)

        # Bold header
        for j in range(len(col_labels)):
            tbl[0, j].set_text_props(fontweight="bold")
            tbl[0, j].set_facecolor("#d0d8e8")

        ax.set_title("Performance metrics vs reconciled", fontsize=9, fontweight="bold")

        footnotes = (
            "[a] O'Mara-Eves et al. 2015 / Cochrane Handbook — minimum sensitivity for T/A screening\n"
            "[b] No formal minimum specificity exists in SR methodology. Sensitivity-specificity tradeoff\n"
            "    is intentional: maximising sensitivity (recall) reduces specificity by design.\n"
            "    Ref: Zhan et al. 2025 GPT-4 tool reported 0.836; Scherbakov et al. 2025 do not report specificity.\n"
            "[c] Scherbakov et al. 2025 — mean across 172 AI screening studies (precision 0.632; "
            "F1 computed from reported sensitivity & precision)\n"
            "[d] Landis & Koch 1977 — minimum κ for substantial agreement\n"
            "Green = meets/exceeds benchmark   White = no formal threshold   Red = below benchmark"
        )
        ax.text(0.01, 0.22, footnotes, transform=ax.transAxes,
                fontsize=6.5, va="top", ha="left",
                color="#333333", linespacing=1.5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#f9f9f9", edgecolor="#cccccc"))

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
            sens_str = f"{c['sensitivity']:.3f}" if c['sensitivity'] is not None else "N/A"
            spec_str = f"{c['specificity']:.3f}" if c['specificity'] is not None else "N/A"
            prec_str = f"{c['precision']:.3f}"   if c['precision']   is not None else "N/A"
            f1_str   = f"{c['f1']:.3f}"          if c['f1']          is not None else "N/A"
            print(f"           {short_name(r):15s}  {c['pct_agreement']:.0f}%  "
                  f"κ={c['kappa_vs_reconciled']}  Sensitivity={sens_str}  "
                  f"Specificity={spec_str}  Precision={prec_str}  F1={f1_str}")

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


# =============================================================================
# EVOLUTION FIGURE — metrics across calibration rounds
# =============================================================================

# Which LLM column to use per file (last/best column for each round)
# Maps (filename, llm_column) -> round_label for the evolution figure.
# Use a list of tuples to allow the same file to appear more than once
# (R2a and R2b share the same EPPI CSV but different LLM columns).
ROUND_LLM_COL = [
    ("EPPI Review - R1.csv",   "LLM",     "R1"),
    ("EPPI Review - R1a.csv",  "LLMr1a",  "R1a"),
    ("EPPI Review - R1b.csv",  "LLMr1b",  "R1b"),
    ("EPPI Review - R2a.csv",  "LLM_r2a", "R2a"),
    ("EPPI Review - R2b.csv",  "LLM_r2b", "R2b"),
    ("EPPI Review - R3a.csv",  "LLM",     "R3a"),
]


def _round_metrics(df: pd.DataFrame, llm_col: str, rec_col: str) -> Optional[Dict]:
    mask = df[llm_col].notna() & df[rec_col].notna()
    pred  = _upper(df.loc[mask, llm_col])
    truth = _upper(df.loc[mask, rec_col])
    tp = int(((pred == INCLUDE_LABEL) & (truth == INCLUDE_LABEL)).sum())
    tn = int(((pred == EXCLUDE_LABEL) & (truth == EXCLUDE_LABEL)).sum())
    fp = int(((pred == INCLUDE_LABEL) & (truth == EXCLUDE_LABEL)).sum())
    fn = int(((pred == EXCLUDE_LABEL) & (truth == INCLUDE_LABEL)).sum())
    total = tp + tn + fp + fn
    if total == 0:
        return None
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1   = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0
    ea   = encode(df.loc[mask, llm_col])
    eb   = encode(df.loc[mask, rec_col])
    valid = ~(np.isnan(ea) | np.isnan(eb))
    k = cohen_kappa(ea[valid], eb[valid]) if valid.sum() >= 2 else None
    return {"sensitivity": sens, "specificity": spec, "precision": prec,
            "f1": f1, "kappa": k}


def build_evolution_figure(out_path: Path) -> None:
    rounds, metrics_by_round = [], {}

    for fname, llm_col, label in ROUND_LLM_COL:
        csv_path = RESULTS_DIR / fname
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        rec_cols = [c for c in df.columns if "reconcil" in c.lower()]
        if not rec_cols or llm_col not in df.columns:
            continue
        m = _round_metrics(df, llm_col, rec_cols[0])
        if m:
            rounds.append(label)
            for k, v in m.items():
                metrics_by_round.setdefault(k, []).append(v)

    if len(rounds) < 2:
        print("[step11] Not enough rounds for evolution figure — skipping.")
        return

    x = np.arange(len(rounds))

    METRICS = [
        ("sensitivity", "Sensitivity",  "#27ae60", "≥ 0.95",  0.95),
        ("specificity", "Specificity",  "#2980b9", None,       None),
        ("precision",   "Precision",    "#8e44ad", "≥ 0.632", 0.632),
        ("f1",          "F1",           "#e67e22", "≥ 0.708", 0.708),
        ("kappa",       "κ (Kappa)",    "#c0392b", "≥ 0.60",  0.60),
    ]

    sns.set_theme(style="whitegrid", font_scale=0.95)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("LLM Screening Performance: Evolution Across Calibration Rounds",
                 fontsize=12, fontweight="bold")

    # One subplot per metric
    for idx, (key, label, color, bench_label, bench_val) in enumerate(METRICS):
        ax = axes[idx // 3][idx % 3]
        vals = metrics_by_round.get(key, [])
        if not vals:
            ax.axis("off")
            continue

        ax.plot(x, vals, color=color, marker="o", linewidth=2, markersize=7, zorder=3)

        # Annotate values
        for xi, v in zip(x, vals):
            ax.annotate(f"{v:.3f}", (xi, v),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8, fontweight="bold", color=color)

        # Benchmark line
        if bench_val is not None:
            ax.axhline(bench_val, color="gray", linestyle="--", linewidth=1.2, zorder=2)
            ax.text(len(rounds) - 0.05, bench_val + 0.01, bench_label,
                    ha="right", va="bottom", fontsize=7.5, color="gray", style="italic")

            # Shade the "pass" zone above the benchmark
            ax.axhspan(bench_val, 1.02, alpha=0.06, color="#27ae60", zorder=1)

        ax.set_title(label, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(rounds)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_xlabel("Calibration round")
        ax.grid(axis="y", alpha=0.4)

    # Last panel — all-rounds summary table
    ax = axes[1][2]
    ax.axis("off")

    # Build: rows = metrics, cols = Metric | Benchmark | R1 | R1a | ... | Rna
    col_labels_tbl = ["Metric", "Benchmark"] + rounds
    n_cols = len(col_labels_tbl)
    table_data = []
    cell_colors_tbl = []

    for key, label, _, bench_label, bench_val in METRICS:
        bench_str = bench_label if bench_label else "No threshold"
        row = [label, bench_str]
        row_colors = ["#f0f0f0", "#f0f0f0"]
        vals = metrics_by_round.get(key, [])
        for v in vals:
            val_str = f"{v:.3f}" if v is not None else "—"
            if bench_val is not None and v is not None:
                color = "#d5f5d5" if v >= bench_val else "#fde0dc"
            else:
                color = "#ffffff"
            row.append(val_str)
            row_colors.append(color)
        table_data.append(row)
        cell_colors_tbl.append(row_colors)

    # Use abbreviated metric names to avoid overflow
    METRIC_SHORT = {
        "Sensitivity": "Sensitivity",
        "Specificity": "Specificity",
        "Precision":   "Precision",
        "F1":          "F1",
        "κ (Kappa)":   "Kappa (κ)",
    }
    for row in table_data:
        row[0] = METRIC_SHORT.get(row[0], row[0])

    tbl = ax.table(
        cellText=table_data,
        colLabels=col_labels_tbl,
        cellLoc="center",
        loc="center",
        bbox=[0.0, 0.15, 1.0, 0.80],
        cellColours=cell_colors_tbl,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.6)

    # Set column widths: wider for Metric and Benchmark, narrower for round cols
    col_widths = [0.22, 0.20] + [0.12] * len(rounds)
    for j, w in enumerate(col_widths):
        for i in range(len(table_data) + 1):
            tbl[i, j].set_width(w)

    for j in range(n_cols):
        tbl[0, j].set_text_props(fontweight="bold")
        tbl[0, j].set_facecolor("#d0d8e8")

    ax.set_title("All-rounds performance summary", fontweight="bold", fontsize=9)

    ax.text(0.01, 0.12,
            "Green = meets benchmark   Red = below benchmark   No threshold = no formal minimum\n"
            "Sensitivity benchmark: O'Mara-Eves et al. 2015 / Cochrane Handbook (>=0.95)\n"
            "Kappa benchmark: Landis & Koch 1977 (>=0.60 substantial agreement)\n"
            "Precision & F1 benchmarks: Scherbakov et al. 2025 (mean across 172 AI screening studies)\n"
            "Specificity: no formal minimum — sensitivity-specificity tradeoff is intentional by design",
            transform=ax.transAxes, fontsize=6.5, va="top", color="#555555",
            linespacing=1.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[step11] Saved evolution figure: {out_path}")

    # Write CSV with all numbers
    rows = []
    for key, label, _, bench_label, bench_val in METRICS:
        vals = metrics_by_round.get(key, [])
        row = {"metric": label, "benchmark": bench_label if bench_label else "No threshold"}
        for round_label, val in zip(rounds, vals):
            row[round_label] = round(val, 4) if val is not None else None
        rows.append(row)
    csv_path = out_path.with_suffix(".csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"[step11] Saved evolution CSV: {csv_path}")


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

    build_evolution_figure(out_dir / "step11_evolution.png")
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

    build_evolution_figure(out_dir / "step11_evolution.png")
    print("[step11] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
