#!/usr/bin/env python3
"""
r6_build_pdf.py

Assembles rework/outputs/diffs/impact_memo.pdf: the impact_memo.md content
as rendered text pages, followed by every diff visual from r5_diff_figures.py
(PRISMA side-by-side, field-value shift bars, per-figure side-by-sides).

Built with matplotlib's PdfPages rather than pandoc/weasyprint — neither
had a working PDF backend available in this environment (no LaTeX engine
for pandoc; weasyprint's system libs weren't installed) — so this avoids
a new dependency / any system-level install.

A minimal markdown->page renderer: headers, bullets, tables and plain
paragraphs. Good enough for a report this is actually going to be a page
of plots, not for general-purpose markdown.

Run (after r1-r5):
  conda run -n ilri01 python rework/scripts/r6_build_pdf.py
"""
from __future__ import annotations

import re
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

REWORK_ROOT = Path(__file__).parent.parent
OUT_DIR = REWORK_ROOT / "outputs" / "diffs"
MEMO_MD = OUT_DIR / "impact_memo.md"
SBS_DIR = OUT_DIR / "side_by_side"
PDF_OUT = OUT_DIR / "impact_memo.pdf"

PAGE_W, PAGE_H = 8.5, 11.0
TEXT_COLOR = "#1f2937"
HEAD_COLOR = "#21472E"
RULE_COLOR = "#d1d5db"


def _new_text_page():
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return fig, ax


def _strip_md_inline(s: str) -> str:
    """Drop inline markdown emphasis markers (no bold-run tracking, just declutter)."""
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"`([^`]+)`", r"\1", s)
    return s


def render_text_pages(pdf: PdfPages, md_text: str) -> None:
    lines = md_text.splitlines()
    fig, ax = _new_text_page()
    y = 0.95
    margin = 0.07
    line_h = 0.021

    def new_page():
        nonlocal fig, ax, y
        pdf.savefig(fig)
        plt.close(fig)
        fig, ax = _new_text_page()
        y = 0.95

    def ensure_space(n_lines: float = 1):
        nonlocal y
        if y - n_lines * line_h < 0.06:
            new_page()

    in_table = False
    table_rows: list[list[str]] = []

    def flush_table():
        nonlocal in_table, table_rows, y
        if not table_rows:
            return
        rows = [r for r in table_rows if not re.match(r"^\s*-{2,}\s*$", r[0] or "")]
        rows = [r for r in rows if not all(set(c.strip()) <= {"-", ":"} for c in r)]
        n_cols = max(len(r) for r in rows)
        col_w = (1 - 2 * margin) / n_cols
        ensure_space(len(rows) + 1)
        for ri, row in enumerate(rows):
            x = margin
            bold = ri == 0
            for ci in range(n_cols):
                cell = _strip_md_inline(row[ci].strip()) if ci < len(row) else ""
                ax.text(x, y, cell, fontsize=7.6, color=TEXT_COLOR,
                        fontweight="bold" if bold else "normal",
                        va="top", ha="left", wrap=False)
                x += col_w
            y -= line_h * 1.15
            if ri == 0:
                ax.plot([margin, 1 - margin], [y + line_h * 0.3] * 2,
                        color=RULE_COLOR, lw=0.6)
                y -= line_h * 0.3
        y -= line_h * 0.6
        table_rows = []
        in_table = False

    for raw in lines:
        line = raw.rstrip()

        if line.strip().startswith("|"):
            in_table = True
            cells = [c for c in line.strip().strip("|").split("|")]
            table_rows.append(cells)
            continue
        elif in_table:
            flush_table()

        if not line.strip():
            y -= line_h * 0.6
            continue

        if line.startswith("# "):
            ensure_space(2.5)
            ax.text(margin, y, _strip_md_inline(line[2:]), fontsize=17, fontweight="bold",
                    color=HEAD_COLOR, va="top", ha="left")
            y -= line_h * 2.2
            ax.plot([margin, 1 - margin], [y + line_h * 0.6] * 2,
                    color=HEAD_COLOR, lw=1.2)
            y -= line_h * 0.6
            continue
        if line.startswith("## "):
            ensure_space(1.8)
            ax.text(margin, y, _strip_md_inline(line[3:]), fontsize=12.5, fontweight="bold",
                    color=HEAD_COLOR, va="top", ha="left")
            y -= line_h * 1.8
            continue

        prefix = ""
        text = _strip_md_inline(line)
        if line.strip().startswith("- "):
            prefix = "•  "
            text = _strip_md_inline(line.strip()[2:])

        wrapped = textwrap.wrap(prefix + text, width=108) or [""]
        ensure_space(len(wrapped))
        for i, wline in enumerate(wrapped):
            indent = margin + (0.02 if prefix and i > 0 else 0)
            ax.text(indent, y, wline, fontsize=8.3, color=TEXT_COLOR,
                    va="top", ha="left")
            y -= line_h

    if in_table:
        flush_table()

    pdf.savefig(fig)
    plt.close(fig)


def render_image_page(pdf: PdfPages, img_path: Path, title: str = "") -> None:
    img = Image.open(img_path)
    iw, ih = img.size
    aspect = ih / iw

    margin = 0.4
    avail_w = PAGE_W - 2 * margin
    avail_h = PAGE_H - 2 * margin - (0.4 if title else 0)

    if avail_w * aspect <= avail_h:
        fig_w, fig_h = PAGE_W, avail_w * aspect + 2 * margin + (0.4 if title else 0)
        fig_h = min(fig_h, PAGE_H)
    else:
        fig_w, fig_h = PAGE_W, PAGE_H

    fig = plt.figure(figsize=(fig_w, fig_h))
    if title:
        fig.text(0.5, 0.975, title, ha="center", va="top", fontsize=11,
                  fontweight="bold", color=HEAD_COLOR)
    ax = fig.add_axes([0.03, 0.02, 0.94, 0.92 if title else 0.96])
    ax.imshow(img)
    ax.axis("off")
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    md_text = MEMO_MD.read_text()

    with PdfPages(PDF_OUT) as pdf:
        render_text_pages(pdf, md_text)

        render_image_page(
            pdf, OUT_DIR / "prisma_side_by_side.png",
            "PRISMA flow — before (n=151) vs after (rework, n=115, prelim)",
        )
        render_image_page(
            pdf, OUT_DIR / "field_shifts.png",
        )

        figure_order = [
            "evidence_gap_map", "geographic_bar", "geographic_map",
            "methodology", "methodology_detail", "equity",
            "temporal_trends", "domain_heatmap", "domain_type",
            "producer_type", "domain_venn",
        ]
        for name in figure_order:
            path = SBS_DIR / f"{name}.png"
            if path.exists():
                render_image_page(pdf, path)

    print(f"[r6] Wrote -> {PDF_OUT}")


if __name__ == "__main__":
    main()
