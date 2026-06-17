"""Generate a PRISMA 2020-compliant flow diagram (human primary track only).

Covers PRISMA 2020 + ROSES for Systematic Maps mandatory items:
  • All sources named individually with record counts (Item 14a/b)
  • Total records identified, duplicates removed (14c)
  • Records screened at T/A with exclusion reasons (14d) — LLM-assisted
  • Full texts sought, not retrieved, assessed, excluded with reasons (14e-h)
  • Studies included (14i)
  • Human vs LLM roles clearly labelled throughout

Outputs:
  scripts/outputs/step16/interactive/human/prisma.png
  deliverables/prisma_flow_d5.png  (legacy copy)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
import shutil

OUT = (Path(__file__).parent.parent /
       "scripts/outputs/step16/interactive/human/prisma.png")
LEG = Path(__file__).parent / "prisma_flow_d5.png"

# ── Colours ───────────────────────────────────────────────────────────────────
G    = "#21472E";  LG = "#eef4f0";  MG = "#c8dece"
RBG  = "#fef2f2"; RBD = "#fca5a5"; RT = "#b91c1c"
AMB  = "#b85c00"; ABG = "#fff8f0"; ABD = "#f6ad55"
BLU  = "#1e40af"; BBG = "#eff6ff"; BBD = "#93c5fd"
GRY  = "#9ca3af"; DRK = "#374151"; BLK = "#1a1a1a"

FW, FH, DPI = 15, 22, 200

fig, ax = plt.subplots(figsize=(FW, FH))
ax.set_xlim(0, FW); ax.set_ylim(0, FH)
ax.axis("off"); fig.patch.set_facecolor("white")


def bx(x, y, w, h, bg="white", ec=G, lw=1.5):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.12",
        linewidth=lw, edgecolor=ec, facecolor=bg, zorder=2))


def tx(x, y, s, fs=10, c=BLK, bold=False, ha="center", va="center"):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs, color=c,
            fontweight="bold" if bold else "normal",
            fontfamily="DejaVu Sans", zorder=3)


def dn(x, y0, y1, col=GRY):
    """Downward arrow from y0 to y1."""
    ax.annotate("", xy=(x, y1+0.07), xytext=(x, y0-0.07),
                arrowprops=dict(arrowstyle="-|>", color=col,
                                lw=1.5, mutation_scale=15), zorder=4)


def rt(x0, x1, y, col=GRY):
    """Rightward arrow from x0 to x1."""
    ax.annotate("", xy=(x1-0.07, y), xytext=(x0+0.07, y),
                arrowprops=dict(arrowstyle="-|>", color=col,
                                lw=1.3, mutation_scale=13), zorder=4)


def band(yb, yt, lbl):
    ax.add_patch(FancyBboxPatch(
        (0.15, yb), 0.52, yt-yb, boxstyle="round,pad=0.06",
        linewidth=1, edgecolor=MG, facecolor=LG, zorder=1))
    lb = ax.text(0.41, (yb+yt)/2, lbl, ha="center", va="center",
                 fontsize=9, color=G, fontweight="bold",
                 fontfamily="DejaVu Sans", zorder=2)
    lb.set_rotation(90)


# ── Layout constants ──────────────────────────────────────────────────────────
# Main column (dedup, screening, eligibility boxes)
MX, MW = 0.85, 10.8
MC     = MX + MW / 2        # ≈ 6.25

# Exclusion column
EX, EW = MX + MW + 0.3, 2.9    # right edge ≈ 14.95

# Source identification boxes (side by side, together span main column width)
AX, AW = 0.85, 5.25             # Box A right edge = 6.10
BX, BW = 6.35, 5.25             # Box B right edge = 11.60
SOURCE_CENTRE = (AX + BX + BW) / 2   # ≈ 6.23


# ── Phase bands ───────────────────────────────────────────────────────────────
#   IDENTIFICATION  :  18.3 – 21.8
#   SCREENING       :  12.2 – 18.3
#   ELIGIBILITY     :   7.0 – 12.2
#   INCLUDED        :   0.3 –  7.0
band(18.3, 21.8, "IDENTIFICATION")
band(12.2, 18.3, "SCREENING")
band( 7.0, 12.2, "ELIGIBILITY")
band( 0.3,  7.0, "INCLUDED")


# ═══════════════════════════════════════════════════════
# IDENTIFICATION — two source boxes side by side
# ═══════════════════════════════════════════════════════

AH = BH = 2.65
AY = BY = 18.8      # bottom of source boxes; top = 21.45

bx(AX, AY, AW, AH, ec=G)
tx(AX + AW/2, AY+AH-0.32, "Bibliographic databases  (7)",
   fs=10.5, bold=True, c=G)
db = [
    ("Scopus",                   "17,083"),
    ("Web of Science",           "15,170"),
    ("CAB Abstracts",             "5,723"),
    ("Academic Search Premier",   "1,187"),
    ("EconLit",                     "478"),
    ("ProQuest",                    "367"),
    ("AGRIS",                         "3"),
]
rh = (AH - 0.65) / len(db)
for i, (nm, n) in enumerate(db):
    ry = AY + AH - 0.65 - (i + 0.5) * rh
    tx(AX + 0.22, ry, nm, fs=8.5, c=DRK, ha="left")
    tx(AX + AW - 0.18, ry, n, fs=8.5, c=G, bold=True, ha="right")

bx(BX, BY, BW, BH, ec=G)
tx(BX + BW/2, BY+BH-0.32, "Grey literature  (21 sources)",
   fs=10.5, bold=True, c=G)
gl = [
    ("Google Scholar",                                           "193"),
    ("DuckDuckGo",                                                 "3"),
    ("UN agencies  (FAO · IFAD · UNDP · UNEP · UNFCCC)",         "57"),
    ("Development banks  (WB · GCF · GEF · IDB · ADB · AfDB · FCDO)", "338"),
    ("Research centres  (CGSpace · IPAM · ARA · GCA · WASP)",    "86"),
    ("M&E networks  (3ie · Campbell Collaboration)",               "8"),
]
rh2 = (BH - 0.65) / len(gl)
for i, (nm, n) in enumerate(gl):
    ry = BY + BH - 0.65 - (i + 0.5) * rh2
    tx(BX + 0.22, ry, nm, fs=8.2, c=DRK, ha="left")
    tx(BX + BW - 0.18, ry, n, fs=8.5, c=G, bold=True, ha="right")

# Total identified summary bar
tx(SOURCE_CENTRE, 18.60,
   "Total records identified:   n = 40,653   (28 sources)",
   fs=11.5, bold=True, c=G)

dn(AX + AW/2, AY, 18.63)
dn(BX + BW/2, BY, 18.63)


# ═══════════════════════════════════════════════════════
# DEDUPLICATION
# ═══════════════════════════════════════════════════════
DD_Y, DD_H = 16.8, 1.00
bx(MX, DD_Y, MW, DD_H, ec=G)
tx(MC, DD_Y+DD_H-0.33, "Records after deduplication", fs=11.5, c=DRK)
tx(MC, DD_Y+0.28, "n = 26,173", fs=16, bold=True, c=G)
dn(MC, 18.56, DD_Y+DD_H)

bx(EX, DD_Y, EW, DD_H, bg=RBG, ec=RBD)
tx(EX+EW/2, DD_Y+DD_H-0.33, "Duplicates removed", fs=10, c=RT)
tx(EX+EW/2, DD_Y+0.28, "n = 14,480", fs=13, bold=True, c=RT)
rt(MX+MW, EX, DD_Y+DD_H/2, col=RBD)


# ═══════════════════════════════════════════════════════
# T/A SCREENING — LLM-assisted
# ═══════════════════════════════════════════════════════
SC_Y, SC_H = 14.55, 1.40
bx(MX, SC_Y, MW, SC_H, bg=BBG, ec=BBD, lw=2.0)
tx(MC, SC_Y+SC_H-0.35,
   "Records screened at title & abstract   [LLM-assisted, human-validated]",
   fs=11, c=BLU)
tx(MC, SC_Y+0.60, "n = 26,173", fs=16, bold=True, c=BLU)
tx(MC, SC_Y+0.24,
   "qwen2.5:14b  ·  temperature 0.0  ·  calibration κ = 0.72  ·  sensitivity = 0.97",
   fs=8.5, c="#475569")
dn(MC, DD_Y, SC_Y+SC_H)

bx(EX, SC_Y, EW, SC_H, bg=RBG, ec=RBD)
tx(EX+EW/2, SC_Y+SC_H-0.35, "Records excluded", fs=10, c=RT)
tx(EX+EW/2, SC_Y+0.78, "n = 17,425  (66%)", fs=13, bold=True, c=RT)
tx(EX+EW/2, SC_Y+0.46, "Concept · methodology ·", fs=9, c=RT)
tx(EX+EW/2, SC_Y+0.22, "context · population", fs=9, c=RT)
rt(MX+MW, EX, SC_Y+SC_H/2, col=RBD)

# Passed T/A
PT_Y, PT_H = 12.65, 0.90
bx(MX, PT_Y, MW, PT_H, ec=G)
tx(MC, PT_Y+PT_H-0.30, "Records passing title & abstract screening", fs=11.5, c=DRK)
tx(MC, PT_Y+0.24, "n = 8,748", fs=16, bold=True, c=G)
dn(MC, SC_Y, PT_Y+PT_H)


# ═══════════════════════════════════════════════════════
# ELIGIBILITY — human full-text screening
# ═══════════════════════════════════════════════════════

# Full texts sought (random sample)
FS_Y, FS_H = 10.85, 1.00
bx(MX, FS_Y, MW, FS_H, bg=ABG, ec=ABD)
tx(MC, FS_Y+FS_H-0.32,
   "Full texts sought   (random sample from 8,748)", fs=11, c=AMB)
tx(MC, FS_Y+0.27, "n = 180", fs=16, bold=True, c=AMB)
dn(MC, PT_Y, FS_Y+FS_H)

bx(EX, FS_Y, EW, FS_H, bg=RBG, ec=RBD)
tx(EX+EW/2, FS_Y+FS_H-0.32, "Not retrieved", fs=10, c=RT)
tx(EX+EW/2, FS_Y+0.27, "n = 8  (4%)", fs=13, bold=True, c=RT)
rt(MX+MW, EX, FS_Y+FS_H/2, col=RBD)

# Full texts assessed for eligibility
FA_Y, FA_H = 8.65, 1.00
bx(MX, FA_Y, MW, FA_H, bg=ABG, ec=ABD)
tx(MC, FA_Y+FA_H-0.32, "Full texts assessed for eligibility", fs=11, c=AMB)
tx(MC, FA_Y+0.27, "n = 172", fs=16, bold=True, c=AMB)
dn(MC, FS_Y, FA_Y+FA_H)

bx(EX, FA_Y, EW, FA_H, bg=RBG, ec=RBD)
tx(EX+EW/2, FA_Y+FA_H-0.32, "Full texts excluded", fs=10, c=RT)
tx(EX+EW/2, FA_Y+0.56, "n = 21  (12%)", fs=13, bold=True, c=RT)
tx(EX+EW/2, FA_Y+0.26, "PCCM criteria not met", fs=9, c=RT)
rt(MX+MW, EX, FA_Y+FA_H/2, col=RBD)


# ═══════════════════════════════════════════════════════
# INCLUDED
# ═══════════════════════════════════════════════════════
IY, IH = 5.55, 1.25
bx(MX, IY, MW, IH, bg=ABG, ec=AMB, lw=2.3)
tx(MC, IY+IH-0.35, "Studies included in systematic map",
   fs=12.5, bold=True, c=AMB)
tx(MC, IY+0.32, "n = 151", fs=22, bold=True, c=AMB)
dn(MC, FA_Y, IY+IH)


# ═══════════════════════════════════════════════════════
# Footer notes  (within INCLUDED band)
# ═══════════════════════════════════════════════════════
tx(MC, 4.78,
   "Title & abstract screening performed by LLM (Ollama; qwen2.5:14b; temperature 0.0), "
   "calibrated against reconciled human decisions (κ = 0.72, sensitivity = 0.97).",
   fs=8.0, c=DRK)
tx(MC, 4.40,
   "Full-text screening performed by human reviewers on a random sample of n = 180 "
   "drawn from n = 8,748 records passing title & abstract screening.",
   fs=8.0, c=DRK)
tx(MC, 4.05,
   "USAID DEC taken offline 2025; J-PAL searched with no relevant results; "
   "Lens.org listed in protocol but superseded by Scopus + WoS coverage.",
   fs=7.5, c=GRY)
tx(MC, 3.72,
   "PRISMA 2020 flow diagram.   CEE Guidelines for Systematic Evidence Syntheses v6.0.",
   fs=7.5, c=GRY)


# ── Save ──────────────────────────────────────────────────────────────────────
plt.tight_layout(pad=0.15)
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(str(OUT), dpi=DPI, bbox_inches="tight", facecolor="white")
shutil.copy(str(OUT), str(LEG))
plt.close()
print(f"Saved : {OUT}  ({OUT.stat().st_size//1024} KB)")
print(f"Copied: {LEG}")
