"""Generate two separate PRISMA flow diagrams — human-only and LLM-only."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

OUT_HUMAN = Path(__file__).parent.parent / "scripts/outputs/step16/prisma_human.png"
OUT_LLM   = Path(__file__).parent.parent / "scripts/outputs/step16/prisma_llm.png"

# ── Colours ────────────────────────────────────────────────────────────────────
GREEN   = "#21472E"
L_GREEN = "#eef4f0"
MID_GRN = "#c8dece"
RED_BG  = "#fef2f2"
RED_BDR = "#fca5a5"
RED_TXT = "#b91c1c"
AMBER   = "#b85c00"
AMB_BG  = "#fff8f0"
AMB_BDR = "#f6ad55"
BLUE    = "#1e40af"
BLUE_BG = "#f0f4ff"
BLUE_BD = "#93c5fd"
GREY    = "#9ca3af"
BLACK   = "#1a1a1a"

# ── Numbers ────────────────────────────────────────────────────────────────────
N_IDENTIFIED  = "40,653"
N_SOURCES     = "28"
N_DUPES       = "14,471"
N_DEDUP       = "26,173"
N_EXCL_TA     = "17,420"
N_FT_SOUGHT   = "8,753"
N_NOT_RETR    = "5,248"
N_FT_RETR     = "3,505"
N_HUM_DRAWN   = "100"
N_HUM_EXCL    = "14"
N_HUM_INCL    = "86"
N_LLM_EXCL    = "1,096"
N_LLM_INCL    = "2,368"

FW, FH = 12, 14
DPI = 200


def make_figure():
    fig, ax = plt.subplots(figsize=(FW, FH))
    ax.set_xlim(0, FW)
    ax.set_ylim(0, FH)
    ax.axis("off")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    return fig, ax


def rect(ax, x, y, w, h, bg="white", ec=GREEN, lw=1.5):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                       linewidth=lw, edgecolor=ec, facecolor=bg, zorder=2)
    ax.add_patch(p)


def label(ax, x, y, w, h, lines, fsizes, colors, bolds=None):
    n = len(lines)
    row_h = h / (n + 0.4)
    for i, (text, fs, color) in enumerate(zip(lines, fsizes, colors)):
        ty = y + h - (i + 0.7) * row_h
        weight = "bold" if (bolds and bolds[i]) else "normal"
        ax.text(x + w / 2, ty, text, ha="center", va="center",
                fontsize=fs, color=color, fontweight=weight,
                fontfamily="DejaVu Sans", zorder=3)


def down_arrow(ax, x, y_from, y_to, color=GREY):
    ax.annotate("", xy=(x, y_to + 0.06), xytext=(x, y_from - 0.06),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4,
                                mutation_scale=14), zorder=4)


def right_arrow(ax, x_from, x_to, y):
    ax.annotate("", xy=(x_to + 0.06, y), xytext=(x_from - 0.06, y),
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.2,
                                mutation_scale=12), zorder=4)


def phase_band(ax, y_bot, y_top, text):
    p = FancyBboxPatch((0.12, y_bot), 0.55, y_top - y_bot,
                       boxstyle="round,pad=0.06", linewidth=1,
                       edgecolor=MID_GRN, facecolor=L_GREEN, zorder=1)
    ax.add_patch(p)
    ax.text(0.39, (y_bot + y_top) / 2, text, ha="center", va="center",
            fontsize=9, color=GREEN, fontweight="bold", rotation=90,
            fontfamily="DejaVu Sans")


def draw_pipeline(ax, CX, CW, EX, EW, BH=0.90):
    """Draw the shared pipeline: Identification → Screening → Eligibility boxes."""
    CM = CX + CW / 2

    # Box 1 — Records identified
    rect(ax, CX, 11.8, CW, 1.05)
    label(ax, CX, 11.8, CW, 1.05,
          ["Records identified", f"n = {N_IDENTIFIED}  ({N_SOURCES} sources)"],
          [11.5, 14], [BLACK, GREEN], [False, True])

    # Exclusion: duplicates
    EY1 = 11.8 + 1.05 / 2
    rect(ax, EX, EY1 - 0.37, EW, 0.74, bg=RED_BG, ec=RED_BDR)
    label(ax, EX, EY1 - 0.37, EW, 0.74,
          ["Duplicates removed", f"n = {N_DUPES}  (36%)"],
          [10.5, 12], [RED_TXT, RED_TXT], [False, True])
    right_arrow(ax, CX + CW, EX, EY1)

    # Box 2 — After dedup
    rect(ax, CX, 9.85, CW, BH)
    label(ax, CX, 9.85, CW, BH,
          ["Records after deduplication", f"n = {N_DEDUP}"],
          [11.5, 14], [BLACK, GREEN], [False, True])
    down_arrow(ax, CM, 11.8, 9.85 + BH)

    # Box 3 — T&A screening
    rect(ax, CX, 7.90, CW, BH)
    label(ax, CX, 7.90, CW, BH,
          ["Records screened (title & abstract)", f"n = {N_DEDUP}"],
          [11.5, 14], [BLACK, GREEN], [False, True])
    down_arrow(ax, CM, 9.85, 7.90 + BH)

    # Exclusion: T&A
    EY3 = 7.90 + BH / 2
    rect(ax, EX, EY3 - 0.50, EW, 1.00, bg=RED_BG, ec=RED_BDR)
    label(ax, EX, EY3 - 0.50, EW, 1.00,
          ["Records excluded", f"n = {N_EXCL_TA}  (67%)",
           "concept / methodology / context /", "population / geography"],
          [10.5, 12, 9.5, 9.5], [RED_TXT, RED_TXT, RED_TXT, RED_TXT],
          [False, True, False, False])
    right_arrow(ax, CX + CW, EX, EY3)

    # Box 4 — Full texts sought
    rect(ax, CX, 5.90, CW, BH)
    label(ax, CX, 5.90, CW, BH,
          ["Full texts sought for retrieval", f"n = {N_FT_SOUGHT}  (33% of screened)"],
          [11.5, 14], [BLACK, GREEN], [False, True])
    down_arrow(ax, CM, 7.90, 5.90 + BH)

    # Exclusion: not retrieved
    EY4 = 5.90 + BH / 2
    rect(ax, EX, EY4 - 0.37, EW, 0.74, bg=RED_BG, ec=RED_BDR)
    label(ax, EX, EY4 - 0.37, EW, 0.74,
          ["Not retrieved (paywall / no DOI)", f"n = {N_NOT_RETR}  (60%)"],
          [10.5, 12], [RED_TXT, RED_TXT], [False, True])
    right_arrow(ax, CX + CW, EX, EY4)

    # Box 5 — Full texts retrieved
    rect(ax, CX, 3.95, CW, BH)
    label(ax, CX, 3.95, CW, BH,
          ["Full texts retrieved", f"n = {N_FT_RETR}  (40% of sought)"],
          [11.5, 14], [BLACK, GREEN], [False, True])
    down_arrow(ax, CM, 5.90, 3.95 + BH)

    return CM


# ══════════════════════════════════════════════════════════════════════════════
# HUMAN-ONLY DIAGRAM
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = make_figure()

CX, CW = 1.0, 5.4
EX, EW = 7.1, 4.5
CM = CX + CW / 2

# Phase bands
phase_band(ax, 10.9, 13.0, "IDENTIFICATION")
phase_band(ax, 6.9,  10.9, "SCREENING")
phase_band(ax, 3.0,   6.9, "ELIGIBILITY")
phase_band(ax, 0.2,   3.0, "INCLUDED")

# Shared pipeline (boxes 1–5)
draw_pipeline(ax, CX, CW, EX, EW)

# Box 6 — Random sample drawn
rect(ax, CX, 2.05, CW, 0.90)
label(ax, CX, 2.05, CW, 0.90,
      ["Random sample drawn for coding", f"n = {N_HUM_DRAWN} papers"],
      [11.5, 14], [BLACK, AMBER], [False, True])
down_arrow(ax, CM, 3.95, 2.05 + 0.90)

# Exclusion: human excluded
EY6 = 2.05 + 0.90 / 2
rect(ax, EX, EY6 - 0.37, EW, 0.74, bg=RED_BG, ec=RED_BDR)
label(ax, EX, EY6 - 0.37, EW, 0.74,
      ["Excluded (inclusion criteria)", f"n = {N_HUM_EXCL}  (14%)"],
      [10.5, 12], [RED_TXT, RED_TXT], [False, True])
right_arrow(ax, CX + CW, EX, EY6)

# Box 7 — Human included (PRIMARY)
rect(ax, CX, 0.35, CW, 1.35, bg=AMB_BG, ec=AMBER, lw=2.2)
label(ax, CX, 0.35, CW, 1.35,
      ["STUDIES INCLUDED", f"n = {N_HUM_INCL}  (86% of coded)", "PRIMARY OUTPUT  ★"],
      [12, 15, 11], [AMBER, AMBER, GREEN], [True, True, True])
down_arrow(ax, CM, 2.05, 0.35 + 1.35, color=AMBER)

plt.tight_layout(pad=0.2)
plt.savefig(str(OUT_HUMAN), dpi=DPI, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {OUT_HUMAN}  ({OUT_HUMAN.stat().st_size/1024:.0f} KB)")


# ══════════════════════════════════════════════════════════════════════════════
# LLM-ONLY DIAGRAM
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = make_figure()

CX, CW = 1.0, 5.4
EX, EW = 7.1, 4.5
CM = CX + CW / 2

# Phase bands
phase_band(ax, 10.9, 13.0, "IDENTIFICATION")
phase_band(ax, 6.9,  10.9, "SCREENING")
phase_band(ax, 3.0,   6.9, "ELIGIBILITY")
phase_band(ax, 0.2,   3.0, "INCLUDED")

# Shared pipeline (boxes 1–5)
draw_pipeline(ax, CX, CW, EX, EW)

# Box 6 — LLM screening
rect(ax, CX, 2.05, CW, 0.90, bg=BLUE_BG, ec=BLUE_BD)
label(ax, CX, 2.05, CW, 0.90,
      ["LLM screening (auto-retrieved only)", f"n = {N_FT_RETR}  screened"],
      [11.5, 14], [BLACK, BLUE], [False, True])
down_arrow(ax, CM, 3.95, 2.05 + 0.90)

# Exclusion: LLM excluded
EY6 = 2.05 + 0.90 / 2
rect(ax, EX, EY6 - 0.37, EW, 0.74, bg=RED_BG, ec=RED_BDR)
label(ax, EX, EY6 - 0.37, EW, 0.74,
      ["Excluded by LLM", f"n = {N_LLM_EXCL}  (31%)"],
      [10.5, 12], [RED_TXT, RED_TXT], [False, True])
right_arrow(ax, CX + CW, EX, EY6)

# Box 7 — LLM included
rect(ax, CX, 0.35, CW, 1.35, bg=BLUE_BG, ec=BLUE_BD, lw=2.2)
label(ax, CX, 0.35, CW, 1.35,
      ["STUDIES INCLUDED", f"n = {N_LLM_INCL}  (68% of screened)", "LLM EXPLORATORY REFERENCE"],
      [12, 15, 11], [BLUE, BLUE, GREY], [True, True, True])
down_arrow(ax, CM, 2.05, 0.35 + 1.35, color=BLUE_BD)

plt.tight_layout(pad=0.2)
plt.savefig(str(OUT_LLM), dpi=DPI, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {OUT_LLM}  ({OUT_LLM.stat().st_size/1024:.0f} KB)")
