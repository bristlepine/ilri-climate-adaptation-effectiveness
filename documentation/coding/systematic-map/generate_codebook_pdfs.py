#!/usr/bin/env python3
"""
Generate PDFs from codebook markdown files.

Usage:
    python generate_codebook_pdfs.py

Requirements:
    - pandoc  (brew install pandoc)
    - Google Chrome
"""

import subprocess
import sys
import re
from pathlib import Path

REPO_ROOT  = Path(__file__).parent.parent.parent.parent
ROUNDS_DIR = REPO_ROOT / "documentation" / "coding" / "systematic-map" / "rounds"
ROUNDS     = ["FT-R1a", "FT-R2a"]
CHROME     = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
* { margin: 0; padding: 0; box-sizing: border-box; }

@page { size: A4; margin: 0.75in; }

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
    font-size: 10.5pt;
    line-height: 1.5;
    color: #1a1a1a;
}

h1 {
    font-size: 24pt;
    font-weight: 700;
    color: #21472E;
    margin: 0 0 6pt 0;
    padding-bottom: 5pt;
    border-bottom: 3px solid #21472E;
    page-break-after: avoid;
}

h2 {
    font-size: 14pt;
    font-weight: 700;
    color: #21472E;
    margin: 14pt 0 4pt 0;
    page-break-after: avoid;
}

h3 {
    font-size: 11.5pt;
    font-weight: 700;
    color: #21472E;
    margin: 10pt 0 3pt 0;
    page-break-after: avoid;
}

h4 {
    font-size: 10.5pt;
    font-weight: 700;
    color: #1a1a1a;
    margin: 6pt 0 2pt 0;
    page-break-after: avoid;
}

p { margin: 3pt 0; }

em { font-style: italic; }
strong, b { font-weight: 700; color: #21472E; }

ul, ol { margin: 3pt 0 3pt 22pt; }
li { margin: 1.5pt 0; line-height: 1.4; }
ul li { list-style-type: disc; }
ol li { list-style-type: decimal; }

code, tt {
    font-family: 'Courier New', monospace;
    font-size: 9pt;
    background: #f0f0f0;
    padding: 1px 4px;
    border-radius: 2px;
}

a { color: #0066cc; text-decoration: none; }

/* Hide all --- dividers — heading spacing handles separation */
hr { display: none !important; }

/* Page breaks */
.page-break {
    display: block !important;
    break-before: page !important;
    page-break-before: always !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    border: none !important;
    visibility: hidden;
}

/* Tables */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 5pt 0;
    font-size: 9.5pt;
    page-break-inside: avoid;
    border: 2px solid #21472E;
}

th {
    background-color: #21472E;
    color: white;
    padding: 5px 8px;
    text-align: left;
    font-weight: 700;
    font-size: 9.5pt;
    line-height: 1.2;
    vertical-align: middle;
    border: 1px solid #1a3a24;
}

td {
    padding: 4px 8px;
    border: 1px solid #ccc;
    vertical-align: top;
    line-height: 1.3;
}

tbody tr:nth-child(even) { background-color: #f6f8f6; }
tbody tr:nth-child(odd)  { background-color: white; }

@media print {
    h1, h2, h3, h4 { page-break-after: avoid; }
    table { page-break-inside: avoid; }
    p, li { orphans: 3; widows: 3; }
}
"""


def make_html(md_file: Path, title: str) -> str:
    result = subprocess.run(
        ["pandoc", "-f", "markdown", "-t", "html", str(md_file)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"pandoc error: {result.stderr}")
        sys.exit(1)

    body = re.sub(r' style="[^"]*"', '', result.stdout)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{title}</title>
  <style>{CSS}</style>
</head>
<body>
{body}
</body>
</html>"""


def to_pdf(html: str, out: Path) -> None:
    tmp = out.parent / ".codebook_tmp.html"
    tmp.write_text(html, encoding="utf-8")
    try:
        script = f"""
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto("file://{tmp}", wait_until="networkidle")
    page.pdf(
        path="{out}",
        format="A4",
        margin={{"top": "0.75in", "bottom": "0.75in", "left": "0.75in", "right": "0.75in"}},
        display_header_footer=False,
        print_background=True,
    )
    browser.close()
"""
        subprocess.run(
            ["conda", "run", "-n", "ilri01", "python", "-c", script],
            check=True, capture_output=True,
        )
    finally:
        tmp.unlink(missing_ok=True)


def main():
    print("Generating codebook PDFs...\n")

    for round_name in ROUNDS:
        md  = ROUNDS_DIR / round_name / f"CODEBOOK_{round_name}.md"
        pdf = ROUNDS_DIR / round_name / f"CODEBOOK_{round_name}.pdf"

        if not md.exists():
            print(f"  Skipping {round_name} — {md.name} not found")
            continue

        print(f"  {round_name}: converting...", end=" ", flush=True)
        html = make_html(md, f"Systematic Map Codebook — {round_name}")
        to_pdf(html, pdf)
        print(f"✓  ({pdf.stat().st_size // 1024} KB)  →  {pdf}")

    print("\nDone.")


if __name__ == "__main__":
    main()
