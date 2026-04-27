"""
export_prisma_png.py

Renders the PRISMA flow diagram to a high-resolution PNG using Playwright.
Output: frontend/public/map/prisma_flow.png  (also copied to deliverables/)

Usage:
  conda run -n ilri01 python scripts/export_prisma_png.py
"""
from pathlib import Path

# ── SVG layout constants (mirror PrismaFlow.tsx) ──────────────────────────────
GREEN     = "#21472E"
LIGHT_GRN = "#eef4f0"
MID_GRN   = "#c8dece"
BOX_BG    = "#ffffff"
EXC_BG    = "#fef2f2"
EXC_BDR   = "#fca5a5"
EXC_TEXT  = "#b91c1c"
GREY_ARR  = "#9ca3af"

W, H   = 800, 550
PX     = 92
CX     = PX + 8
CW     = 320
CM     = CX + CW // 2
EX     = CX + CW + 28
EW     = W - EX - 6
BH     = 52
BH1    = 80

Y1 = 14
Y2 = Y1 + BH1 + 22
P1 = Y2 + BH + 18
Y3 = P1 + 14
P2 = Y3 + BH + 18
Y4 = P2 + 14
Y5 = Y4 + BH + 22
P3 = Y5 + BH + 18
Y6 = P3 + 16


def rect(x, y, w, h, rx=4, fill=BOX_BG, stroke=GREEN, sw=1.5):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def text(x, y, content, size=11, weight="normal", fill="#555", anchor="middle", family="Arial, sans-serif", spacing=""):
    ls = f' letter-spacing="{spacing}"' if spacing else ""
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}" fill="{fill}" font-family="{family}"{ls}>{content}</text>'


def center_box(y, h, label, n, sub=None, highlight=False):
    bg    = "#f0f7f2" if highlight else BOX_BG
    sw    = 2 if highlight else 1.5
    n_col = GREEN if highlight else "#111"
    out   = [rect(CX, y, CW, h, fill=bg, stroke=GREEN, sw=sw)]
    if sub:
        out.append(text(CM, y + h * 0.28, label, size=10))
        out.append(text(CM, y + h * 0.52, n, size=14, weight="700", fill=n_col))
        out.append(text(CM, y + h * 0.78, sub, size=8.5, fill="#888"))
    else:
        out.append(text(CM, y + h * 0.35, label, size=10))
        out.append(text(CM, y + h * 0.65, n, size=14, weight="700", fill=n_col))
    return "\n".join(out)


def exclusion_box(y, label, n):
    return "\n".join([
        rect(EX, y, EW, BH, fill=EXC_BG, stroke=EXC_BDR, sw=1),
        text(EX + EW // 2, y + BH * 0.36, label, size=10, fill=EXC_TEXT),
        text(EX + EW // 2, y + BH * 0.68, n, size=13, weight="700", fill=EXC_TEXT),
    ])


def phase_label(label, y1, y2):
    mid = (y1 + y2) // 2
    return f"""
<rect x="2" y="{y1+2}" width="{PX-6}" height="{y2-y1-4}" rx="4" fill="{MID_GRN}"/>
<text x="{(PX-6)//2+2}" y="{mid}" text-anchor="middle" dominant-baseline="middle"
  transform="rotate(-90, {(PX-6)//2+2}, {mid})"
  font-size="9.5" font-weight="700" font-family="Arial, sans-serif"
  fill="{GREEN}" letter-spacing="1.5">{label}</text>"""


def down_arrow(y):
    return f'<line x1="{CM}" y1="{y}" x2="{CM}" y2="{y+16}" stroke="{GREEN}" stroke-width="1.5" marker-end="url(#arr-down)"/>'


def right_arrow(source_y):
    mid_y = source_y + BH // 2
    return f'<line x1="{CX+CW}" y1="{mid_y}" x2="{EX-1}" y2="{mid_y}" stroke="{GREY_ARR}" stroke-width="1.5" stroke-dasharray="5 3" marker-end="url(#arr-right)"/>'


def build_svg():
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">']

    # Defs / markers
    parts.append(f"""<defs>
  <marker id="arr-down" markerWidth="9" markerHeight="7" refX="9" refY="3.5" orient="auto">
    <polygon points="0 0, 9 3.5, 0 7" fill="{GREEN}"/>
  </marker>
  <marker id="arr-right" markerWidth="9" markerHeight="7" refX="9" refY="3.5" orient="auto">
    <polygon points="0 0, 9 3.5, 0 7" fill="{GREY_ARR}"/>
  </marker>
</defs>""")

    # Phase backgrounds
    parts.append(rect(0, 0,  W, P1,      rx=0, fill=LIGHT_GRN, stroke="none", sw=0))
    parts.append(rect(0, P1, W, P2 - P1, rx=0, fill="#fafafa",  stroke="none", sw=0))
    parts.append(rect(0, P2, W, P3 - P2, rx=0, fill=LIGHT_GRN, stroke="none", sw=0))
    parts.append(rect(0, P3, W, H - P3,  rx=0, fill="#fafafa",  stroke="none", sw=0))

    # Phase labels
    parts.append(phase_label("IDENTIFICATION", 0,  P1))
    parts.append(phase_label("SCREENING",      P1, P2))
    parts.append(phase_label("ELIGIBILITY",    P2, P3))
    parts.append(phase_label("INCLUDED",       P3, H))

    # Box 1
    parts.append(center_box(Y1, BH1,
        "Records identified via database searching", "n = 39,113",
        sub="Scopus 17,021 · WoS 15,179 · CAB 5,723 · ASP 1,187 · AGRIS 3"))
    parts.append(down_arrow(Y1 + BH1))

    # Box 2 + exclusion
    parts.append(center_box(Y2, BH, "Records after deduplication", "n = 25,208"))
    parts.append(exclusion_box(Y2, "Duplicates removed", "n = 13,905"))
    parts.append(right_arrow(Y2))
    parts.append(down_arrow(Y2 + BH))

    # Box 3 + exclusion
    parts.append(center_box(Y3, BH, "Records screened (title &amp; abstract)", "n = 25,208"))
    parts.append(exclusion_box(Y3, "Records excluded", "n = 16,653"))
    parts.append(right_arrow(Y3))
    parts.append(down_arrow(Y3 + BH))

    # Box 4 + exclusion
    parts.append(center_box(Y4, BH, "Full texts sought for retrieval", "n = 8,555"))
    parts.append(exclusion_box(Y4, "Not retrieved", "n = 5,079"))
    parts.append(right_arrow(Y4))
    parts.append(down_arrow(Y4 + BH))

    # Box 5 + exclusion
    parts.append(center_box(Y5, BH, "Full texts assessed for eligibility", "n = 3,476"))
    parts.append(exclusion_box(Y5, "Full texts excluded", "n = 726"))
    parts.append(right_arrow(Y5))
    parts.append(down_arrow(Y5 + BH))

    # Box 6 (highlight)
    parts.append(center_box(Y6, BH + 8, "Studies included in data extraction", "n = 2,750", highlight=True))

    parts.append("</svg>")
    return "\n".join(parts)


def main():
    from playwright.sync_api import sync_playwright

    svg = build_svg()
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>body{{margin:0;padding:0;background:#fff;}}</style>
</head><body>{svg}</body></html>"""

    out_map  = Path(__file__).parent.parent / "frontend" / "public" / "map" / "prisma_flow.png"
    out_del  = Path(__file__).parent.parent / "deliverables" / "prisma_flow.png"

    print("Rendering PRISMA PNG via Playwright...")
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page    = browser.new_page(viewport={"width": W * 2, "height": H * 2})
        page.set_content(html)
        page.wait_for_load_state("networkidle")
        svg_el = page.query_selector("svg")
        svg_el.screenshot(path=str(out_map), scale="device")
        browser.close()

    import shutil
    shutil.copy(out_map, out_del)

    print(f"Saved: {out_map}")
    print(f"Saved: {out_del}")


if __name__ == "__main__":
    main()
