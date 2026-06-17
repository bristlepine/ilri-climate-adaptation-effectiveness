"""PRISMA 2020 + ROSES Item 14 flow diagram — human primary track.

ALL positions are calculated deterministically below. No magic numbers.

Canvas  : FW=15, FH=15 inches at DPI=200
Gap     : 0.5 between every box (deterministic)
Margins : 0.4 top, 0.25 bottom, 0.4 left (after phase band), 0.25 right

Phase bands (y ranges, verified against all boxes):
  IDENTIFICATION :  11.5 – 15.0
  SCREENING      :   7.0 – 11.5
  ELIGIBILITY    :   4.0 –  7.0
  INCLUDED       :   0.2 –  4.0

Box centres (x): main column MC=6.0, excl column EC=12.65
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
GRY  = "#9ca3af"; DRK = "#374151"

DPI = 200

# ── Canvas ────────────────────────────────────────────────────────────────────
FW, FH = 15, 15
fig, ax = plt.subplots(figsize=(FW, FH))
ax.set_xlim(0, FW); ax.set_ylim(0, FH)
ax.axis("off"); fig.patch.set_facecolor("white")

# ── Phase band ────────────────────────────────────────────────────────────────
BAND_X, BAND_W = 0.15, 0.52        # right edge = 0.67

# ── Column geometry ───────────────────────────────────────────────────────────
# Gap between phase band right edge and main box left edge = 0.53
MX  = BAND_X + BAND_W + 0.53       # = 1.20
MW  = 9.40                          # main box width
MR  = MX + MW                       # = 10.60   (right edge)
MC  = MX + MW / 2                   # = 5.90    (centre)

# Exclusion boxes: 0.35 gap after main right edge
EX  = MR + 0.35                     # = 10.95
EW  = 3.75                          # right edge = 14.70 (< FW=15 ✓)
EC  = EX + EW / 2                   # = 12.825

# Source boxes (identification): together span exactly MX to MR
AX, AW = MX, (MW - 0.20) / 2       # = 1.20, 4.60   right = 5.80
BX, BW = MX + AW + 0.20, AW        # = 6.00, 4.60   right = 10.60

# ── Vertical constants ────────────────────────────────────────────────────────
GAP  = 0.52     # gap between consecutive boxes
GAPL = 0.48     # smaller gap where space is tight (source→note, note→dedup)

# ── All box y positions (bottom edge) ────────────────────────────────────────
# Work TOP-DOWN: start at FH - top_margin and subtract box heights + gaps

_y = FH - 0.40               # running cursor (top margin = 0.40)

# Source boxes
SRC_H = 2.35
SRC_Y = _y - SRC_H           # = 15.25   top = 17.60
_y    = SRC_Y                 # bottom of source boxes

# Total-identified note (text only, no box)
NOTE_Y = _y - GAPL            # = 14.73
_y     = NOTE_Y

# Dedup box
DD_H  = 0.92
DD_Y  = _y - GAPL - DD_H     # = 13.29   top = 14.21
_y    = DD_Y

# T/A screening box (blue, taller)
SC_H  = 1.32
SC_Y  = _y - GAP - SC_H      # = 11.45   top = 12.77
_y    = SC_Y

# Passed T/A box
PT_H  = 0.85
PT_Y  = _y - GAP - PT_H      # = 10.08   top = 10.93
_y    = PT_Y

# FT sought box
FS_H  = 0.88
FS_Y  = _y - GAP - FS_H      # = 8.68    top = 9.56
_y    = FS_Y

# FT assessed box
FA_H  = 0.88
FA_Y  = _y - GAP - FA_H      # = 7.28    top = 8.16
_y    = FA_Y

# Studies included — floats directly below FA_Y with standard gap
IH    = 1.15
IY    = _y - GAP - IH         # = FA_Y - 0.52 - 1.15  (tight, standard gap)

# Footer notes (below included box, inside INCLUDED band)
F1_Y  = IY - 0.33
F2_Y  = F1_Y - 0.33
F3_Y  = F2_Y - 0.30
F4_Y  = F3_Y - 0.28

# ── Verify all boxes fall inside their phase bands ────────────────────────────
# (assertions so this explodes loudly if layout drifts)
assert 11.5 <= SRC_Y            <= 15.0,  f"SRC_Y={SRC_Y}"       # IDENTIFICATION
assert  7.0 <= DD_Y  and DD_Y + DD_H <= 11.5, f"DD_Y={DD_Y}"     # SCREENING
assert  7.0 <= SC_Y  and SC_Y + SC_H <= 11.5, f"SC_Y={SC_Y}"     # SCREENING
assert  7.0 <= PT_Y  and PT_Y + PT_H <= 11.5, f"PT_Y={PT_Y}"     # SCREENING
assert  4.0 <= FS_Y  and FS_Y + FS_H <=  7.0, f"FS_Y={FS_Y}"     # ELIGIBILITY
assert  4.0 <= FA_Y  and FA_Y + FA_H <=  7.0, f"FA_Y={FA_Y}"     # ELIGIBILITY
assert  0.2 <= IY    and IY   + IH   <=  4.0, f"IY={IY}"         # INCLUDED


# ── Drawing helpers ───────────────────────────────────────────────────────────
def bx(x, y, w, h, bg="white", ec=G, lw=1.5):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.12",
        linewidth=lw, edgecolor=ec, facecolor=bg, zorder=2))


def tx(x, y, s, fs=11, c=DRK, bold=False, ha="center", va="center"):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs, color=c,
            fontweight="bold" if bold else "normal",
            fontfamily="DejaVu Sans", zorder=3)


def dn(x, y0, y1, col=GRY):
    ax.annotate("", xy=(x, y1+0.07), xytext=(x, y0-0.07),
                arrowprops=dict(arrowstyle="-|>", color=col,
                                lw=1.6, mutation_scale=15), zorder=4)


def rt(x0, x1, y, col=GRY):
    ax.annotate("", xy=(x1-0.07, y), xytext=(x0+0.07, y),
                arrowprops=dict(arrowstyle="-|>", color=col,
                                lw=1.4, mutation_scale=13), zorder=4)


def band(yb, yt, lbl):
    ax.add_patch(FancyBboxPatch(
        (BAND_X, yb), BAND_W, yt-yb, boxstyle="round,pad=0.06",
        linewidth=1, edgecolor=MG, facecolor=LG, zorder=1))
    lb = ax.text(BAND_X + BAND_W/2, (yb+yt)/2, lbl,
                 ha="center", va="center", fontsize=9.5, color=G,
                 fontweight="bold", fontfamily="DejaVu Sans", zorder=2)
    lb.set_rotation(90)


# ── Phase bands ───────────────────────────────────────────────────────────────
band(11.5, 15.0, "IDENTIFICATION")
band( 7.0, 11.5, "SCREENING")
band( 4.0,  7.0, "ELIGIBILITY")
band( 0.2,  4.0, "INCLUDED")


# ═══════════════════════════════════════
# IDENTIFICATION
# ═══════════════════════════════════════

bx(AX, SRC_Y, AW, SRC_H, ec=G)
tx(AX + AW/2, SRC_Y+SRC_H-0.34, "Bibliographic databases  (7)",
   fs=11, bold=True, c=G)
db = [
    ("Scopus",                   "17,083"),
    ("Web of Science",           "15,170"),
    ("CAB Abstracts",             "5,723"),
    ("Academic Search Premier",   "1,187"),
    ("EconLit",                     "478"),
    ("ProQuest",                    "367"),
    ("AGRIS",                         "3"),
]
rh = (SRC_H - 0.70) / len(db)
for i, (nm, n) in enumerate(db):
    ry = SRC_Y + SRC_H - 0.70 - (i + 0.5) * rh
    tx(AX + 0.22, ry, nm, fs=9, c=DRK, ha="left")
    tx(AX + AW - 0.18, ry, n,  fs=9, c=G, bold=True, ha="right")

bx(BX, SRC_Y, BW, SRC_H, ec=G)
tx(BX + BW/2, SRC_Y+SRC_H-0.34, "Grey literature  (21 sources)",
   fs=11, bold=True, c=G)
gl = [
    ("Google Scholar",                                           "193"),
    ("DuckDuckGo",                                                 "3"),
    ("UN agencies  (FAO · IFAD · UNDP · UNEP · UNFCCC)",     "57"),
    ("Dev. banks  (WB · GCF · GEF · IDB · ADB · AfDB · FCDO)", "338"),
    ("Research centres  (CGSpace · IPAM · ARA · GCA · WASP)", "86"),
    ("M&E networks  (3ie · Campbell Collaboration)",               "8"),
]
rh2 = (SRC_H - 0.70) / len(gl)
for i, (nm, n) in enumerate(gl):
    ry = SRC_Y + SRC_H - 0.70 - (i + 0.5) * rh2
    tx(BX + 0.22, ry, nm, fs=8.8, c=DRK, ha="left")
    tx(BX + BW - 0.18, ry, n,  fs=9,   c=G, bold=True, ha="right")

# Total-identified summary line
tx(MC, NOTE_Y, "Total records identified:   n = 40,653   (28 sources)",
   fs=12.5, bold=True, c=G)
dn(AX + AW/2, SRC_Y, NOTE_Y + 0.05)
dn(BX + BW/2, SRC_Y, NOTE_Y + 0.05)


# ═══════════════════════════════════════
# DEDUPLICATION
# ═══════════════════════════════════════
bx(MX, DD_Y, MW, DD_H, ec=G)
tx(MC, DD_Y + DD_H - 0.32, "Records after deduplication", fs=12)
tx(MC, DD_Y + 0.26, "n = 26,173", fs=17, bold=True, c=G)
dn(MC, NOTE_Y - 0.05, DD_Y + DD_H)

bx(EX, DD_Y, EW, DD_H, bg=RBG, ec=RBD)
tx(EC, DD_Y + DD_H - 0.32, "Duplicates removed", fs=10.5, c=RT)
tx(EC, DD_Y + 0.26, "n = 14,480", fs=14, bold=True, c=RT)
rt(MR, EX, DD_Y + DD_H/2, col=RBD)


# ═══════════════════════════════════════
# T/A SCREENING  (LLM-assisted)
# ═══════════════════════════════════════
bx(MX, SC_Y, MW, SC_H, bg=BBG, ec=BBD, lw=2.1)
tx(MC, SC_Y + SC_H - 0.36,
   "Records screened at title & abstract   [LLM-assisted, human-validated]",
   fs=11.5, c=BLU)
tx(MC, SC_Y + 0.57, "n = 26,173", fs=17, bold=True, c=BLU)
tx(MC, SC_Y + 0.22,
   "qwen2.5:14b  ·  temperature 0.0  ·  calibration κ = 0.72  ·  sensitivity = 0.97",
   fs=9, c="#475569")
dn(MC, DD_Y, SC_Y + SC_H)

bx(EX, SC_Y, EW, SC_H, bg=RBG, ec=RBD)
tx(EC, SC_Y + SC_H - 0.36, "Records excluded", fs=10.5, c=RT)
tx(EC, SC_Y + 0.75, "n = 17,425  (66%)", fs=13.5, bold=True, c=RT)
tx(EC, SC_Y + 0.44, "Concept · methodology ·", fs=9.5, c=RT)
tx(EC, SC_Y + 0.20, "context · population", fs=9.5, c=RT)
rt(MR, EX, SC_Y + SC_H/2, col=RBD)

# Passed T/A
bx(MX, PT_Y, MW, PT_H, ec=G)
tx(MC, PT_Y + PT_H - 0.30, "Records passing title & abstract screening", fs=12)
tx(MC, PT_Y + 0.22, "n = 8,748", fs=17, bold=True, c=G)
dn(MC, SC_Y, PT_Y + PT_H)


# ═══════════════════════════════════════
# ELIGIBILITY
# ═══════════════════════════════════════

# Full texts sought
bx(MX, FS_Y, MW, FS_H, bg=ABG, ec=ABD)
tx(MC, FS_Y + FS_H - 0.30,
   "Full texts sought   (random sample from 8,748)", fs=11, c=AMB)
tx(MC, FS_Y + 0.24, "n = 180", fs=17, bold=True, c=AMB)
dn(MC, PT_Y, FS_Y + FS_H)

bx(EX, FS_Y, EW, FS_H, bg=RBG, ec=RBD)
tx(EC, FS_Y + FS_H - 0.30, "Not retrieved", fs=10.5, c=RT)
tx(EC, FS_Y + 0.24, "n = 8  (4%)", fs=14, bold=True, c=RT)
rt(MR, EX, FS_Y + FS_H/2, col=RBD)

# Full texts assessed
bx(MX, FA_Y, MW, FA_H, bg=ABG, ec=ABD)
tx(MC, FA_Y + FA_H - 0.30, "Full texts assessed for eligibility", fs=11, c=AMB)
tx(MC, FA_Y + 0.24, "n = 172", fs=17, bold=True, c=AMB)
dn(MC, FS_Y, FA_Y + FA_H)

bx(EX, FA_Y, EW, FA_H, bg=RBG, ec=RBD)
tx(EC, FA_Y + FA_H*0.68, "n = 21  excluded  (12%)", fs=13, bold=True, c=RT)
tx(EC, FA_Y + FA_H*0.26, "PCCM criteria not met", fs=9.5, c=RT)
rt(MR, EX, FA_Y + FA_H/2, col=RBD)


# ═══════════════════════════════════════
# INCLUDED
# ═══════════════════════════════════════
bx(MX, IY, MW, IH, bg=ABG, ec=AMB, lw=2.4)
tx(MC, IY + IH - 0.36, "Studies included in systematic map",
   fs=13, bold=True, c=AMB)
tx(MC, IY + 0.32, "n = 151", fs=24, bold=True, c=AMB)
dn(MC, FA_Y, IY + IH)

# Footer
tx(MC, F1_Y,
   "Title & abstract screening performed by LLM (Ollama; qwen2.5:14b; temperature 0.0), "
   "calibrated against reconciled human decisions (κ = 0.72, sensitivity = 0.97).",
   fs=8.5, c=DRK)
tx(MC, F2_Y,
   "Full-text screening performed by human reviewers on a random sample of n = 180 "
   "drawn from n = 8,748 records passing title & abstract screening.",
   fs=8.5, c=DRK)
tx(MC, F3_Y,
   "USAID DEC taken offline 2025; J-PAL searched with no results; "
   "Lens.org in protocol but superseded by Scopus + WoS coverage.",
   fs=8.0, c=GRY)
tx(MC, F4_Y,
   "PRISMA 2020.   CEE Guidelines for Systematic Evidence Syntheses v6.0.",
   fs=8.0, c=GRY)


# ── Save ──────────────────────────────────────────────────────────────────────
plt.tight_layout(pad=0.10)
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(str(OUT), dpi=DPI, bbox_inches="tight", facecolor="white")
shutil.copy(str(OUT), str(LEG))
plt.close()
print(f"Saved : {OUT}  ({OUT.stat().st_size//1024} KB)")
print(f"Copied: {LEG}")
