"""
generate_egm_html.py

Generates a standalone HTML file for the Evidence Gap Map from the current
Plotly JSON (scripts/outputs/step16/interactive/evidence_gap_map.json).

The .html files in the interactive/ folder are stale wrappers from April 2026.
This script creates a fresh, standalone HTML that can be uploaded to Teams or
shared as a link.

Output: deliverables/evidence_gap_map_d5.html

Run: conda run -n ilri01 python deliverables/generate_egm_html.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
JSON_PATH = ROOT / "scripts" / "outputs" / "step16" / "interactive" / "human" / "evidence_gap_map.json"
OUT_HTML  = ROOT / "deliverables" / "evidence_gap_map_d5.html"

PLOTLY_CDN = "https://cdn.plot.ly/plotly-3.5.0.min.js"

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Evidence Gap Map — Bristlepine / ILRI D5</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body {{
      font-family: Lato, Arial, sans-serif;
      margin: 0;
      padding: 20px 40px;
      background: #fafafa;
      color: #424242;
    }}
    h1 {{
      font-size: 1.2rem;
      font-weight: 700;
      margin-bottom: 4px;
      color: #21472E;
    }}
    .meta {{
      font-size: 0.85rem;
      color: #757575;
      margin-bottom: 20px;
    }}
    #egm-container {{
      background: white;
      border: 1px solid #e0e0e0;
      border-radius: 4px;
      padding: 12px;
    }}
  </style>
</head>
<body>
  <h1>Evidence Gap Map</h1>
  <div class="meta">
    Measuring what matters: tracking climate adaptation processes and outcomes for smallholder producers
    in the agriculture sector &mdash; Deliverable 5 (Final Systematic Map) &mdash; Bristlepine Resilience Consultants / ILRI &mdash; June 2026
  </div>
  <div id="egm-container">
    <div id="egm-plot" style="height: 700px; width: 100%;"></div>
  </div>
  <script src="{cdn}"></script>
  <script>
    var fig = {fig_json};
    Plotly.newPlot('egm-plot', fig.data, fig.layout, {{responsive: true, displaylogo: false}});
  </script>
</body>
</html>
"""


def main():
    if not JSON_PATH.exists():
        print(f"ERROR: JSON not found: {JSON_PATH}")
        print("Run step16_map_visualise.py first.")
        return

    with open(JSON_PATH, encoding="utf-8") as f:
        fig_json = f.read()

    # Parse to verify it's valid JSON
    fig_data = json.loads(fig_json)
    n_traces = len(fig_data.get("data", []))
    print(f"[egm_html] Loaded figure: {n_traces} traces")

    html = HTML_TEMPLATE.format(
        cdn=PLOTLY_CDN,
        fig_json=fig_json,
    )

    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"[egm_html] Written: {OUT_HTML}")
    print(f"[egm_html] File size: {OUT_HTML.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
