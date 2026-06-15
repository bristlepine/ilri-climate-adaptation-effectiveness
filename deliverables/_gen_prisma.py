"""Generate two separate PRISMA flow diagrams:
  - prisma_human_d5.png  (human coding path — use in report)
  - prisma_llm_d5.png    (LLM path — supplementary reference)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

STEP16 = Path(__file__).parent.parent / "scripts" / "outputs" / "step16"
OUT_HUMAN  = STEP16 / "interactive" / "human" / "prisma.png"
OUT_LLM    = STEP16 / "interactive" / "llm"   / "prisma.png"
# Legacy path in deliverables/ — kept for backwards compat only
OUT_LEGACY = Path(__file__).parent / "prisma_flow_d5.png"

GREEN   = "#21472E"
L_GREEN = "#eef4f0"
MID_GRN = "#c8dece"
RED_BG  = "#fef2f2"
RED_BDR = "#fca5a5"
RED_TXT = "#b91c1c"
AMBER   = "#b85c00"
AMB_BG  = "#fff8f0"
AMB_BDR = "#f6ad55"
GREY    = "#9ca3af"
BLACK   = "#1a1a1a"
BLUE    = "#1e40af"
BLUE_BG = "#f8fafc"
BLUE_BDR= "#94a3b8"

DPI = 200


def make_figure(fh=10):
    FW, FH = 13, fh
    fig, ax = plt.subplots(figsize=(FW, FH))
    ax.set_xlim(0, FW)
    ax.set_ylim(0, FH)
    ax.axis("off")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    return fig, ax, FW, FH


def rect(ax, x, y, w, h, bg="white", ec=GREEN, lw=1.5):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                       linewidth=lw, edgecolor=ec, facecolor=bg, zorder=2)
    ax.add_patch(p)


def label(ax, x, y, w, h, lines, fsizes, colors, bold_mask=None):
    n = len(lines)
    row_h = h / (n + 0.4)
    for i, (text, fs, color) in enumerate(zip(lines, fsizes, colors)):
        ty = y + h - (i + 0.7) * row_h
        weight = "bold" if (bold_mask and bold_mask[i]) else "normal"
        ax.text(x + w/2, ty, text, ha="center", va="center",
                fontsize=fs, color=color, fontweight=weight,
                fontfamily="DejaVu Sans", zorder=3)


def down_arrow(ax, x, y_from, y_to):
    GAP = 0.08
    ax.annotate("", xy=(x, y_to + GAP), xytext=(x, y_from - GAP),
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.4,
                                mutation_scale=14), zorder=4)


def right_arrow(ax, x_from, x_to, y):
    GAP = 0.06
    ax.annotate("", xy=(x_to + GAP, y), xytext=(x_from - GAP, y),
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.2,
                                mutation_scale=12), zorder=4)


def phase_band(ax, y_bot, y_top, text):
    p = FancyBboxPatch((0.15, y_bot), 0.6, y_top - y_bot,
                       boxstyle="round,pad=0.06", linewidth=1,
                       edgecolor=MID_GRN, facecolor=L_GREEN, zorder=1)
    ax.add_patch(p)
    ax.text(0.45, (y_bot + y_top) / 2, text, ha="center", va="center",
            fontsize=9, color=GREEN, fontweight="bold", rotation=90,
            fontfamily="DejaVu Sans")


# ── shared layout ─────────────────────────────────────────────────────────────
CX, CW = 1.2, 8.0
CM     = CX + CW / 2
EX, EW = CX + CW + 0.35, 2.8
BH_STD  = 0.90
BH_TALL = 1.10
BH_INC  = 1.20

Y_BOX1 = 7.80
Y_BOX2 = 6.10
Y_BOX3 = 4.40


def draw_shared_boxes(ax):
    """Boxes 1–3 identical in both diagrams."""
    # Box 1
    rect(ax, CX, Y_BOX1, CW, BH_TALL)
    label(ax, CX, Y_BOX1, CW, BH_TALL,
          ["Records identified", "n = 40,653  (29 sources)"],
          [11.5, 13.5], [BLACK, GREEN], [False, True])

    # Box 2
    rect(ax, CX, Y_BOX2, CW, BH_STD)
    label(ax, CX, Y_BOX2, CW, BH_STD,
          ["Records after deduplication", "n = 26,182"],
          [11.5, 13.5], [BLACK, GREEN], [False, True])
    down_arrow(ax, CM, Y_BOX1, Y_BOX2 + BH_STD)

    EY = Y_BOX2 + BH_STD / 2
    rect(ax, EX, EY - 0.38, EW, 0.76, bg=RED_BG, ec=RED_BDR)
    label(ax, EX, EY - 0.38, EW, 0.76,
          ["Duplicates removed", "n = 14,471"],
          [10.5, 12], [RED_TXT, RED_TXT], [False, True])
    right_arrow(ax, CX + CW, EX, EY)

    # Box 3
    rect(ax, CX, Y_BOX3, CW, BH_STD)
    label(ax, CX, Y_BOX3, CW, BH_STD,
          ["Records screened (title & abstract)", "n = 26,182"],
          [11.5, 13.5], [BLACK, GREEN], [False, True])
    down_arrow(ax, CM, Y_BOX2, Y_BOX3 + BH_STD)

    EY = Y_BOX3 + BH_STD / 2
    rect(ax, EX, EY - 0.52, EW, 1.04, bg=RED_BG, ec=RED_BDR)
    label(ax, EX, EY - 0.52, EW, 1.04,
          ["Records excluded", "n = 17,429  (67%)",
           "concept, methodology,", "context, population, geography"],
          [10.5, 12, 9.5, 9.5], [RED_TXT, RED_TXT, RED_TXT, RED_TXT],
          [False, True, False, False])
    right_arrow(ax, CX + CW, EX, EY)


# ══════════════════════════════════════════════════════════════════════════════
#  HUMAN PRISMA
# ══════════════════════════════════════════════════════════════════════════════
fig, ax, FW, FH = make_figure(fh=10)

# Title
ax.text(FW/2, 9.75, "Figure 1. PRISMA Flow Diagram — Human Coding (Primary)",
        ha="center", va="center", fontsize=11.5, color=BLACK,
        fontweight="bold", fontfamily="DejaVu Sans", zorder=3)

phase_band(ax, 7.40, 9.20, "IDENTIFICATION")
phase_band(ax, 4.00, 7.40, "SCREENING")
phase_band(ax, 0.40, 4.00, "INCLUDED")

draw_shared_boxes(ax)

# Arrow from Box 3 down to Human coding
Y_HCODE = 2.35
BH_HCODE = 1.10
down_arrow(ax, CM, Y_BOX3, Y_HCODE + BH_HCODE)

# Human coding box
rect(ax, CX, Y_HCODE, CW, BH_HCODE, bg=AMB_BG, ec=AMB_BDR)
label(ax, CX, Y_HCODE, CW, BH_HCODE,
      ["HUMAN CODING  (primary)", "random sample: n = 180"],
      [11, 12.5], [AMBER, "#92400e"], [True, True])

# Exclusion box — split into not retrievable vs full-text excluded
EY_H = Y_HCODE + BH_HCODE / 2
EH_H = 1.20
rect(ax, EX, EY_H - EH_H / 2, EW, EH_H, bg=RED_BG, ec=RED_BDR)
label(ax, EX, EY_H - EH_H / 2, EW, EH_H,
      ["Not retrievable", "n = 8  (4%)",
       "Full-text excluded", "n = 21  (12%)"],
      [10.5, 11.5, 10.5, 11.5],
      [RED_TXT, RED_TXT, RED_TXT, RED_TXT],
      [False, True, False, True])
right_arrow(ax, CX + CW, EX, EY_H)

# Arrow to Human included
Y_HINC = 0.55
down_arrow(ax, CM, Y_HCODE, Y_HINC + BH_INC)

# Human included
rect(ax, CX, Y_HINC, CW, BH_INC, bg=AMB_BG, ec=AMBER, lw=2.0)
label(ax, CX, Y_HINC, CW, BH_INC,
      ["HUMAN INCLUDED", "n = 151  (84%)", "PRIMARY OUTPUT"],
      [11, 14, 11], [AMBER, AMBER, GREEN], [True, True, True])

plt.tight_layout(pad=0.2)
plt.savefig(str(OUT_HUMAN), dpi=DPI, bbox_inches="tight", facecolor="white")
import shutil; shutil.copy(str(OUT_HUMAN), str(OUT_LEGACY))
plt.close()
print(f"Saved: {OUT_HUMAN}  ({OUT_HUMAN.stat().st_size//1024} KB)")


# ══════════════════════════════════════════════════════════════════════════════
#  LLM PRISMA
# ══════════════════════════════════════════════════════════════════════════════
fig, ax, FW, FH = make_figure()

phase_band(ax, 7.40, 9.20, "IDENTIFICATION")
phase_band(ax, 4.00, 7.40, "SCREENING")
phase_band(ax, 3.00, 4.00, "ELIGIBILITY")
phase_band(ax, 0.40, 3.00, "INCLUDED")

draw_shared_boxes(ax)

# Box 4: Full texts sought
Y_BOX4 = 3.10
down_arrow(ax, CM, Y_BOX3, Y_BOX4 + BH_STD)
rect(ax, CX, Y_BOX4, CW, BH_STD)
label(ax, CX, Y_BOX4, CW, BH_STD,
      ["Full texts sought", "n = 8,748  (33% of screened)"],
      [11.5, 13.5], [BLACK, GREEN], [False, True])

EY = Y_BOX4 + BH_STD / 2
rect(ax, EX, EY - 0.42, EW, 0.84, bg=RED_BG, ec=RED_BDR)
label(ax, EX, EY - 0.42, EW, 0.84,
      ["Not retrieved", "n = 5,243  (60%)"],
      [10.5, 12], [RED_TXT, RED_TXT], [False, True])
right_arrow(ax, CX + CW, EX, EY)

# LLM screening box
Y_LTRACK = 1.90
BH_LTRACK = 1.25
down_arrow(ax, CM, Y_BOX4, Y_LTRACK + BH_LTRACK)
rect(ax, CX, Y_LTRACK, CW, BH_LTRACK, bg=BLUE_BG, ec=BLUE_BDR)
label(ax, CX, Y_LTRACK, CW, BH_LTRACK,
      ["LLM SCREENING", "retrieved: n = 3,505  (40%)", "excluded: n = 1,096  (31%)"],
      [11, 12.5, 11], ["#475569", BLUE, "#475569"], [True, True, False])

# LLM included
Y_LINC = 0.65
down_arrow(ax, CM, Y_LTRACK, Y_LINC + BH_INC)
rect(ax, CX, Y_LINC, CW, BH_INC, bg=BLUE_BG, ec=BLUE_BDR, lw=2.0)
label(ax, CX, Y_LINC, CW, BH_INC,
      ["LLM INCLUDED", "n = 2,368  (68%)", "exploratory reference only"],
      [11, 14, 11], [BLUE, BLUE, "#64748b"], [True, True, False])

plt.tight_layout(pad=0.2)
plt.savefig(str(OUT_LLM), dpi=DPI, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {OUT_LLM}  ({OUT_LLM.stat().st_size//1024} KB)")
