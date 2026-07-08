"""PRISMA 2020 + ROSES Item 14 flow diagram — human primary track.

Every number below is LOADED from the real pipeline output files, not typed in.
If the underlying data changes, rerun this script and the diagram updates —
it cannot silently drift out of sync with Table 5 again.

Sources:
  scripts/outputs/step2b/step2b_summary.json   — identification (per-source, raw)
  scripts/outputs/step12/step12_results.meta.json   — Scopus T&A screening
  scripts/outputs/step12b/step12b_results.meta.json — non-Scopus T&A screening (post-dedup pool)
  scripts/outputs/step13/step13_summary.json        — post-cross-dedup T&A pass, per database
  scripts/outputs/step13/step13_merge_summary.json  — Scopus included count
  scripts/outputs/step15/step15b_summary.json       — full-text sample / final included

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

import json
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
import shutil

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "scripts" / "outputs"
OUT = OUT_DIR / "step16" / "interactive" / "human" / "prisma.png"
LEG = Path(__file__).parent / "prisma_flow_d5.png"


def _load(rel):
    with open(OUT_DIR / rel) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD REAL NUMBERS FROM PIPELINE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════

step2b = _load("step2b/step2b_summary.json")
step12 = _load("step12/step12_results.meta.json")
step12b = _load("step12b/step12b_results.meta.json")
step13_summary = _load("step13/step13_summary.json")
step13_merge = _load("step13/step13_merge_summary.json")
step15b = _load("step15/step15b_summary.json")

# ── IDENTIFICATION ──────────────────────────────────────────────────────────
SCOPUS_RETURNED = step2b["scopus_corpus_size"]
BY_DB = step2b["by_database_imported"]

NAMED_DBS = ["Web of Science", "CAB Abstracts", "Academic Search Premier",
             "EconLit", "ProQuest", "Google Scholar", "AGRIS"]
db = [("Scopus", SCOPUS_RETURNED)] + [(nm, BY_DB[nm]) for nm in NAMED_DBS]

# Grouping of grey-lit sources into display categories — this categorisation
# (which source goes under which label) is a content choice; the NUMBERS are
# always summed live from step2b_summary.json, never typed in.
GREY_LIT_GROUPS = {
    "UN agencies  (FAO · IFAD · UNDP · UNEP · UNFCCC)":
        ["FAO", "IFAD", "UNDP", "UNEP", "UNFCCC"],
    "Dev. banks  (WB · GCF · GEF · IDB · ADB · AfDB · FCDO)":
        ["World Bank", "GEF", "GCF", "IDB", "ADB", "AfDB", "FCDO"],
    "Research centres  (CGSpace · IPAM · ARA · GCA · WASP)":
        ["CGSpace (CGIAR)", "IPAM", "Adaptation Research Alliance", "GCA", "WASP"],
    "M&E networks  (3ie · Campbell Collaboration)":
        ["3ie", "Campbell Collaboration"],
    "DuckDuckGo": ["DuckDuckGo"],
}
grey_lit_keys = set(BY_DB) - set(NAMED_DBS)
grouped_keys = {k for grp in GREY_LIT_GROUPS.values() for k in grp}
assert grey_lit_keys == grouped_keys, (
    f"Grey-lit source list in step2b_summary.json no longer matches the "
    f"GREY_LIT_GROUPS mapping in this script — new/removed source(s): "
    f"{grey_lit_keys ^ grouped_keys}. Update GREY_LIT_GROUPS above."
)
gl = [(label, sum(BY_DB[k] for k in keys)) for label, keys in GREY_LIT_GROUPS.items()]
GREY_LIT_TOTAL = sum(n for _, n in gl)

N_DATABASES = len(db)              # 8
N_GREY_SOURCES = len(grey_lit_keys)  # 20
N_SOURCES = N_DATABASES + N_GREY_SOURCES

IDENTIFIED_TOTAL = SCOPUS_RETURNED + sum(BY_DB.values())
assert IDENTIFIED_TOTAL == sum(n for _, n in db) + GREY_LIT_TOTAL

# ── DEDUPLICATION ────────────────────────────────────────────────────────────
# rows_total in each screening meta file = the exact record count that fed
# that screening pass, i.e. the real post-dedup pool (Scopus is never
# deduplicated against itself; non-Scopus net-new count already reflects
# step2b's dedup + the later manual unrecoverable-abstract cleanup).
AFTER_DEDUP = step12["rows_total"] + step12b["rows_total"]
DUPLICATES_REMOVED = IDENTIFIED_TOTAL - AFTER_DEDUP

# ── T&A SCREENING ────────────────────────────────────────────────────────────
# step13_summary's total_processed is POST cross-database dedup — the number
# that actually proceeds to full-text retrieval (5 fewer than the raw
# per-database Include sum, because a handful of records screened Include
# under two different source databases turned out to be duplicates of each
# other only once merged).
PASSED_TA = step13_summary["total_processed"]
EXCLUDED_TA = AFTER_DEDUP - PASSED_TA
EXCLUDED_TA_PCT = round(100 * EXCLUDED_TA / AFTER_DEDUP)

# ── ELIGIBILITY / INCLUDED ───────────────────────────────────────────────────
FT_SOUGHT = step15b["total_coder_rows"]
BATCHES = step15b["batches"]
EXCLUDED_FT = step15b["excluded_before_filter"]
EXCLUDED_FT_PCT = round(100 * EXCLUDED_FT / FT_SOUGHT)
INCLUDED = step15b["final_human_records"]
assert FT_SOUGHT - EXCLUDED_FT == INCLUDED


def fmt(n):
    return f"{n:,}"


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

# Source boxes (identification): together span exactly MX to MR, with a wider
# gap between them for visual separation
ID_GAP = 0.40
AX, AW = MX, (MW - ID_GAP) / 2     # = 1.20, 4.50   right = 5.70
BX, BW = MX + AW + ID_GAP, AW      # = 6.10, 4.50   right = 10.60

# ── Vertical constants ────────────────────────────────────────────────────────
GAP  = 0.52     # gap between consecutive boxes
GAPL = 0.40     # smaller gap where space is tight (source→note, note→dedup)

# ── All box y positions (bottom edge) ────────────────────────────────────────
# Work TOP-DOWN: start at FH - top_margin and subtract box heights + gaps

_y = FH - 0.40               # running cursor (top margin = 0.40)

# Source boxes — extra height gives the row lists room to breathe
SRC_H = 2.55
SRC_Y = _y - SRC_H
_y    = SRC_Y                 # bottom of source boxes

# Total-identified note (text only, no box)
NOTE_Y = _y - GAPL
_y     = NOTE_Y

# Dedup box
DD_H  = 0.92
DD_Y  = _y - GAPL - DD_H
_y    = DD_Y

# T/A screening box (blue, taller)
SC_H  = 1.32
SC_Y  = _y - GAP - SC_H
_y    = SC_Y

# Passed T/A box
PT_H  = 0.85
PT_Y  = _y - GAP - PT_H
_y    = PT_Y

# FT sought & assessed box (single stage — see note below)
FS_H  = 0.88
FS_Y  = _y - GAP - FS_H
_y    = FS_Y

# Kept purely to preserve the original vertical spacing of the INCLUDED box
# (a second box used to sit here — "full texts assessed" — but its 8
# not-retrieved / 21 excluded-on-PCCM split isn't tracked anywhere as an
# authoritative field, only the combined total is (step15b_summary.json's
# excluded_before_filter). Rather than fabricate the split, both exclusion
# reasons are now shown as one honest combined box at FS_Y.)
_FA_H = 0.88
_FA_Y = _y - GAP - _FA_H

# Studies included
IH    = 1.15
IY    = _FA_Y - GAP - IH

# Footer notes (below included box) — wrapped to the main column width so
# they stay under the box instead of sprawling across the phase band
FOOTER_TOP = IY - 0.45

# ── Verify all boxes fall inside their phase bands ────────────────────────────
# (assertions so this explodes loudly if layout drifts)
assert 11.5 <= SRC_Y            <= 15.0,  f"SRC_Y={SRC_Y}"       # IDENTIFICATION
assert  7.0 <= DD_Y  and DD_Y + DD_H <= 11.5, f"DD_Y={DD_Y}"     # SCREENING
assert  7.0 <= SC_Y  and SC_Y + SC_H <= 11.5, f"SC_Y={SC_Y}"     # SCREENING
assert  7.0 <= PT_Y  and PT_Y + PT_H <= 11.5, f"PT_Y={PT_Y}"     # SCREENING
assert  4.0 <= FS_Y  and FS_Y + FS_H <=  7.0, f"FS_Y={FS_Y}"     # ELIGIBILITY
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
tx(AX + AW/2, SRC_Y+SRC_H-0.34, f"Bibliographic databases  ({N_DATABASES})",
   fs=11, bold=True, c=G)
rh = (SRC_H - 0.70) / len(db)
for i, (nm, n) in enumerate(db):
    ry = SRC_Y + SRC_H - 0.70 - (i + 0.5) * rh
    tx(AX + 0.22, ry, nm, fs=9, c=DRK, ha="left")
    tx(AX + AW - 0.18, ry, fmt(n), fs=9, c=G, bold=True, ha="right")

bx(BX, SRC_Y, BW, SRC_H, ec=G)
tx(BX + BW/2, SRC_Y+SRC_H-0.34, f"Grey literature  ({N_GREY_SOURCES} sources)",
   fs=11, bold=True, c=G)
rh2 = (SRC_H - 0.70) / len(gl)
for i, (nm, n) in enumerate(gl):
    ry = SRC_Y + SRC_H - 0.70 - (i + 0.5) * rh2
    tx(BX + 0.22, ry, nm, fs=6.7, c=DRK, ha="left")
    tx(BX + BW - 0.18, ry, fmt(n), fs=9,   c=G, bold=True, ha="right")

# Total-identified summary line
tx(MC, NOTE_Y, f"Total records identified:   n = {fmt(IDENTIFIED_TOTAL)}   ({N_SOURCES} sources)",
   fs=12.5, bold=True, c=G)
dn(AX + AW/2, SRC_Y, NOTE_Y + 0.05)
dn(BX + BW/2, SRC_Y, NOTE_Y + 0.05)


# ═══════════════════════════════════════
# DEDUPLICATION
# ═══════════════════════════════════════
bx(MX, DD_Y, MW, DD_H, ec=G)
tx(MC, DD_Y + DD_H - 0.32, "Records after deduplication", fs=12)
tx(MC, DD_Y + 0.26, f"n = {fmt(AFTER_DEDUP)}", fs=17, bold=True, c=G)
dn(MC, NOTE_Y - 0.05, DD_Y + DD_H)

bx(EX, DD_Y, EW, DD_H, bg=RBG, ec=RBD)
tx(EC, DD_Y + DD_H - 0.32, "Duplicates removed", fs=10.5, c=RT)
tx(EC, DD_Y + 0.26, f"n = {fmt(DUPLICATES_REMOVED)}", fs=14, bold=True, c=RT)
rt(MR, EX, DD_Y + DD_H/2, col=RBD)


# ═══════════════════════════════════════
# T/A SCREENING  (LLM-assisted)
# ═══════════════════════════════════════
bx(MX, SC_Y, MW, SC_H, bg=BBG, ec=BBD, lw=2.1)
tx(MC, SC_Y + SC_H - 0.36,
   "Records screened at title & abstract   [LLM-assisted, human-validated]",
   fs=11.5, c=BLU)
tx(MC, SC_Y + 0.57, f"n = {fmt(AFTER_DEDUP)}", fs=17, bold=True, c=BLU)
tx(MC, SC_Y + 0.22,
   "qwen2.5:14b  ·  temperature 0.0  ·  calibration κ = 0.72  ·  sensitivity = 0.97",
   fs=9, c="#475569")
dn(MC, DD_Y, SC_Y + SC_H)

bx(EX, SC_Y, EW, SC_H, bg=RBG, ec=RBD)
tx(EC, SC_Y + SC_H - 0.36, "Records excluded", fs=10.5, c=RT)
tx(EC, SC_Y + 0.75, f"n = {fmt(EXCLUDED_TA)}  ({EXCLUDED_TA_PCT}%)", fs=13.5, bold=True, c=RT)
tx(EC, SC_Y + 0.44, "Concept · methodology ·", fs=9.5, c=RT)
tx(EC, SC_Y + 0.20, "context · population", fs=9.5, c=RT)
rt(MR, EX, SC_Y + SC_H/2, col=RBD)

# Passed T/A
bx(MX, PT_Y, MW, PT_H, ec=G)
tx(MC, PT_Y + PT_H - 0.30, "Records passing title & abstract screening", fs=12)
tx(MC, PT_Y + 0.22, f"n = {fmt(PASSED_TA)}", fs=17, bold=True, c=G)
dn(MC, SC_Y, PT_Y + PT_H)


# ═══════════════════════════════════════
# ELIGIBILITY
# ═══════════════════════════════════════

# Full texts sought & assessed for eligibility (combined — see note above IY)
bx(MX, FS_Y, MW, FS_H, bg=ABG, ec=ABD)
tx(MC, FS_Y + FS_H - 0.30,
   f"Full texts sought & assessed for eligibility   (random sample from {fmt(PASSED_TA)})",
   fs=10.5, c=AMB)
tx(MC, FS_Y + 0.24, f"n = {fmt(FT_SOUGHT)}", fs=17, bold=True, c=AMB)
dn(MC, PT_Y, FS_Y + FS_H)

bx(EX, FS_Y, EW, FS_H, bg=RBG, ec=RBD)
tx(EC, FS_Y + FS_H - 0.30, "Excluded", fs=10.5, c=RT)
tx(EC, FS_Y + 0.24, f"n = {fmt(EXCLUDED_FT)}  ({EXCLUDED_FT_PCT}%)", fs=13, bold=True, c=RT)
rt(MR, EX, FS_Y + FS_H/2, col=RBD)


# ═══════════════════════════════════════
# INCLUDED
# ═══════════════════════════════════════
bx(MX, IY, MW, IH, bg=ABG, ec=AMB, lw=2.4)
tx(MC, IY + IH - 0.36, "Studies included in systematic map",
   fs=13, bold=True, c=AMB)
tx(MC, IY + 0.32, f"n = {fmt(INCLUDED)}", fs=24, bold=True, c=AMB)
dn(MC, FS_Y, IY + IH, col=ABD)

# Footer — wrapped to the main column width (MW) so it stays contained under
# the box instead of spilling past it into the phase band.
def footer_block(y_top, text, fs=8.0, c=DRK, width_chars=98, line_h=0.20):
    lines = textwrap.wrap(text, width=width_chars)
    for i, line in enumerate(lines):
        tx(MC, y_top - i * line_h, line, fs=fs, c=c)
    return y_top - (len(lines) - 1) * line_h


_fy = FOOTER_TOP
_fy = footer_block(_fy,
    "Title & abstract screening performed by LLM (Ollama; qwen2.5:14b; temperature 0.0), "
    "calibrated against reconciled human decisions (κ = 0.72, sensitivity = 0.97).",
    fs=8.3, c=DRK, width_chars=95) - 0.30
_fy = footer_block(_fy,
    f"Full-text screening performed by human reviewers on a random sample of n = {fmt(FT_SOUGHT)} "
    f"({BATCHES} batches of 20) drawn from n = {fmt(PASSED_TA)} records passing title & abstract "
    f"screening. “Excluded” combines not-retrieved and did-not-meet-inclusion-criteria records "
    f"(not separately tracked in the pipeline).",
    fs=8.0, c=DRK, width_chars=100) - 0.30
_fy = footer_block(_fy,
    "USAID DEC taken offline 2025; J-PAL searched with no results; "
    "Lens.org in protocol but superseded by Scopus + WoS coverage.",
    fs=7.8, c=GRY, width_chars=105) - 0.26
_fy = footer_block(_fy,
    "PRISMA 2020.   CEE Guidelines for Systematic Evidence Syntheses v6.0.",
    fs=7.8, c=GRY, width_chars=105)


# ── Save ──────────────────────────────────────────────────────────────────────
plt.tight_layout(pad=0.10)
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(str(OUT), dpi=DPI, bbox_inches="tight", facecolor="white")
shutil.copy(str(OUT), str(LEG))
plt.close()
print(f"Saved : {OUT}  ({OUT.stat().st_size//1024} KB)")
print(f"Copied: {LEG}")
print()
print(f"Identified {fmt(IDENTIFIED_TOTAL)} ({N_SOURCES} sources) -> "
      f"dedup {fmt(AFTER_DEDUP)} -> passed T&A {fmt(PASSED_TA)} -> "
      f"FT sought {fmt(FT_SOUGHT)} -> included {fmt(INCLUDED)}")
