"""Generate PRISMA flow diagram PNG — clean layout, readable text."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

OUT = Path(__file__).parent / "prisma_flow_d5.png"

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

FW, FH = 13, 12         # figure width, height (inches)
DPI     = 180

fig, ax = plt.subplots(figsize=(FW, FH))
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.axis("off")
ax.set_facecolor("white")
fig.patch.set_facecolor("white")

# ── helpers ────────────────────────────────────────────────────────────────────

def rect(x, y, w, h, bg="white", ec=GREEN, lw=1.5):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                       linewidth=lw, edgecolor=ec, facecolor=bg, zorder=2)
    ax.add_patch(p)

def label(x, y, w, h, lines, fsizes, colors, bold_mask=None):
    """Vertically-centred multi-line text inside a box."""
    n = len(lines)
    row_h = h / (n + 0.4)
    for i, (text, fs, color) in enumerate(zip(lines, fsizes, colors)):
        ty = y + h - (i + 0.7) * row_h
        weight = "bold" if (bold_mask and bold_mask[i]) else "normal"
        ax.text(x + w/2, ty, text, ha="center", va="center",
                fontsize=fs, color=color, fontweight=weight,
                fontfamily="DejaVu Sans", zorder=3)

def down_arrow(x, y_from, y_to):
    """Downward arrow with a small gap at source and destination."""
    GAP = 0.08
    ax.annotate("", xy=(x, y_to + GAP), xytext=(x, y_from - GAP),
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.4,
                                mutation_scale=14),
                zorder=4)

def right_arrow(x_from, x_to, y):
    GAP = 0.06
    ax.annotate("", xy=(x_to + GAP, y), xytext=(x_from - GAP, y),
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.2,
                                mutation_scale=12),
                zorder=4)

def phase_band(y_bot, y_top, text):
    p = FancyBboxPatch((0.15, y_bot), 0.7, y_top - y_bot,
                       boxstyle="round,pad=0.06", linewidth=1,
                       edgecolor=MID_GRN, facecolor=L_GREEN, zorder=1)
    ax.add_patch(p)
    ax.text(0.50, (y_bot + y_top) / 2, text, ha="center", va="center",
            fontsize=8, color=GREEN, fontweight="bold", rotation=90,
            fontfamily="DejaVu Sans")

# ── layout constants ───────────────────────────────────────────────────────────
# Main column
CX, CW  = 1.1,  5.2      # x, width of centre boxes
CM      = CX + CW / 2    # centre x

# Exclusion column
EX, EW  = 7.0,  5.4      # x, width of exclusion boxes
EM      = EX + EW / 2

# Row y-positions (all in inches, y=0 at bottom)
# Phase bands
P_IDENT_BOT, P_IDENT_TOP = 9.4, 12.0
P_SCREE_BOT, P_SCREE_TOP = 7.0,  9.4
P_ELIGB_BOT, P_ELIGB_TOP = 4.8,  7.0
P_INCLD_BOT, P_INCLD_TOP = 0.3,  4.8

# Box y positions (y_bot of each box)
Y_BOX1  = 10.6    # Records identified (top of diagram)
BH_STD  = 0.90
BH_TALL = 1.10

Y_BOX2  = 8.65    # After deduplication
Y_BOX3  = 6.10    # T&A screened
Y_BOX4  = 4.10    # Full texts sought
Y_FORK  = 3.55    # Fork point (just above LLM/Human boxes)

# INCLUDED section: two tracks side by side
TW, TG  = 2.2, 0.3
LX      = CX + (CW - 2*TW - TG) / 2
HX      = LX + TW + TG
LC      = LX + TW / 2
HC      = HX + TW / 2

Y_TRACK = 2.05    # top of LLM/Human screening boxes
BH_TRK  = 1.25
Y_INC   = 0.50    # top of LLM/Human included boxes
BH_INC  = 1.10

# ── Phase bands ───────────────────────────────────────────────────────────────
phase_band(P_IDENT_BOT, P_IDENT_TOP, "IDENTIFICATION")
phase_band(P_SCREE_BOT, P_SCREE_TOP, "SCREENING")
phase_band(P_ELIGB_BOT, P_ELIGB_TOP, "ELIGIBILITY")
phase_band(P_INCLD_BOT, P_INCLD_TOP, "INCLUDED")

# ── Box 1: Records identified ──────────────────────────────────────────────────
rect(CX, Y_BOX1, CW, BH_TALL)
label(CX, Y_BOX1, CW, BH_TALL,
      ["Records identified", "n = 40,653  (29 sources)"],
      [10, 12], [BLACK, GREEN], [False, True])

# ── Box 2: After dedup ────────────────────────────────────────────────────────
rect(CX, Y_BOX2, CW, BH_STD)
label(CX, Y_BOX2, CW, BH_STD,
      ["Records after deduplication", "n = 26,182"],
      [10, 12], [BLACK, GREEN], [False, True])

# arrow 1 → 2
down_arrow(CM, Y_BOX1, Y_BOX2 + BH_STD)

# exclusion: duplicates
EY_DED = Y_BOX2 + BH_STD / 2
rect(EX, EY_DED - 0.38, EW, 0.76, bg=RED_BG, ec=RED_BDR)
label(EX, EY_DED - 0.38, EW, 0.76,
      ["Duplicates removed", "n = 14,471"],
      [9.5, 10.5], [RED_TXT, RED_TXT], [False, True])
right_arrow(CX + CW, EX, EY_DED)

# ── Box 3: T&A screening ──────────────────────────────────────────────────────
rect(CX, Y_BOX3, CW, BH_STD)
label(CX, Y_BOX3, CW, BH_STD,
      ["Records screened (title & abstract)", "n = 26,182"],
      [10, 12], [BLACK, GREEN], [False, True])

# arrow 2 → 3
down_arrow(CM, Y_BOX2, Y_BOX3 + BH_STD)

# exclusion: T&A
EY_SCR = Y_BOX3 + BH_STD / 2
rect(EX, EY_SCR - 0.52, EW, 1.04, bg=RED_BG, ec=RED_BDR)
label(EX, EY_SCR - 0.52, EW, 1.04,
      ["Records excluded", "n = 17,429  (67%)",
       "concept, methodology, context,", "population, geography"],
      [9.5, 10.5, 8.5, 8.5], [RED_TXT, RED_TXT, RED_TXT, RED_TXT],
      [False, True, False, False])
right_arrow(CX + CW, EX, EY_SCR)

# ── Box 4: Full texts sought ──────────────────────────────────────────────────
rect(CX, Y_BOX4, CW, BH_STD)
label(CX, Y_BOX4, CW, BH_STD,
      ["Full texts sought", "n = 8,748  (33% of screened)"],
      [10, 12], [BLACK, GREEN], [False, True])

# arrow 3 → 4
down_arrow(CM, Y_BOX3, Y_BOX4 + BH_STD)

# exclusion: not retrieved
EY_FT = Y_BOX4 + BH_STD / 2
rect(EX, EY_FT - 0.42, EW, 0.84, bg=RED_BG, ec=RED_BDR)
label(EX, EY_FT - 0.42, EW, 0.84,
      ["Not retrieved (paywall / no DOI)", "n = 5,243  (60%)"],
      [9.5, 10.5], [RED_TXT, RED_TXT], [False, True])
right_arrow(CX + CW, EX, EY_FT)

# ── Fork: arrow from Box 4 down to fork point ─────────────────────────────────
down_arrow(CM, Y_BOX4, Y_FORK)

# Horizontal fork line
ax.plot([LC, HC], [Y_FORK, Y_FORK], color=GREY, lw=1.4, zorder=3)

# Sub-arrows from fork to track boxes
down_arrow(LC, Y_FORK, Y_TRACK + BH_TRK)
down_arrow(HC, Y_FORK, Y_TRACK + BH_TRK)

# ── LLM track (left) ──────────────────────────────────────────────────────────
rect(LX, Y_TRACK, TW, BH_TRK, bg="#f8fafc", ec="#94a3b8")
label(LX, Y_TRACK, TW, BH_TRK,
      ["LLM SCREENING", "retrieved: n = 3,505  (40%)", "excluded: n = 1,096  (31%)"],
      [9, 10, 9], ["#475569", "#1e40af", "#475569"], [True, True, False])

# ── Human track (right) ───────────────────────────────────────────────────────
rect(HX, Y_TRACK, TW, BH_TRK, bg=AMB_BG, ec=AMB_BDR)
label(HX, Y_TRACK, TW, BH_TRK,
      ["HUMAN CODING  (primary)", "sampled: n = 100", "excluded: n = 14  (14%)"],
      [9, 10, 9], [AMBER, "#92400e", AMBER], [True, True, False])

# ── Arrows to included boxes ──────────────────────────────────────────────────
down_arrow(LC, Y_TRACK, Y_INC + BH_INC)
down_arrow(HC, Y_TRACK, Y_INC + BH_INC)

# ── LLM included ──────────────────────────────────────────────────────────────
rect(LX, Y_INC, TW, BH_INC, bg="#f8fafc", ec="#94a3b8")
label(LX, Y_INC, TW, BH_INC,
      ["LLM INCLUDED", "n = 2,368  (68%)", "exploratory reference"],
      [9, 12, 8.5], ["#475569", "#1e40af", "#64748b"], [True, True, False])

# ── Human included ────────────────────────────────────────────────────────────
rect(HX, Y_INC, TW, BH_INC, bg=AMB_BG, ec=AMBER, lw=2.0)
label(HX, Y_INC, TW, BH_INC,
      ["HUMAN INCLUDED", "n = 86  (86%)", "PRIMARY OUTPUT"],
      [9, 12, 9], [AMBER, AMBER, GREEN], [True, True, True])

plt.tight_layout(pad=0.2)
plt.savefig(str(OUT), dpi=DPI, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")
