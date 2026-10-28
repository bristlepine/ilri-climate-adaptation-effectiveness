#!/usr/bin/env python3
"""
r2b_export_pngs.py

Export PNGs for the rework's human/compare interactive figures, the same
way scripts/step16b_export_pngs.py does for the original pipeline. That
script hardcodes its paths at module level (no config dict), so rather than
edit it we load it and repoint its path constants at rework/outputs/step16
before calling its functions — zero edits to the shared script.

Prerequisite: run r2_run_figures.py first (produces
rework/outputs/step16/interactive/{human,compare}/*.json).

Run:
  conda run -n ilri01 python rework/scripts/r2b_export_pngs.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REWORK_ROOT = Path(__file__).parent.parent
REPO_ROOT = REWORK_ROOT.parent
STEP16B_PATH = REPO_ROOT / "scripts" / "step16b_export_pngs.py"
OUT_ROOT = REWORK_ROOT / "outputs" / "step16"

# step16b's DIMS hardcodes a fixed PNG height per chart, overriding the
# height each chart's own JSON layout already computes correctly (e.g.
# geographic_bar's JSON sets height = max(400, n_countries*22+120), but the
# PNG export forces 480px regardless — with ~39 countries that crams the
# bars so tightly Plotly silently drops every other y-axis label). Rather
# than edit the shared script's DIMS dict, read each listed chart's own
# JSON height here and use that instead, so the name never gets lost again
# regardless of how many bars there are.
DYNAMIC_HEIGHT_CHARTS = ["geographic_bar"]


def main() -> None:
    spec = importlib.util.spec_from_file_location("step16b_export_pngs", STEP16B_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    module.OUT_ROOT = OUT_ROOT
    module.INTER = OUT_ROOT / "interactive"
    module.HUMAN_J = module.INTER / "human"
    module.COMPARE_J = module.INTER / "compare"
    module.LLM_PNG = module.INTER / "llm"
    module.HUMAN_PNG = module.INTER / "human"
    module.COMP_PNG = module.INTER / "compare"
    module.LLM_PNG.mkdir(exist_ok=True, parents=True)

    for name in DYNAMIC_HEIGHT_CHARTS:
        json_path = module.HUMAN_J / f"{name}.json"
        if json_path.exists():
            height = json.loads(json_path.read_text())["layout"]["height"]
            w, _ = module.DIMS.get(name, (900, height))
            module.DIMS[name] = (w, height)
            print(f"[r2b] {name}: using its own JSON height ({height}px) "
                  f"instead of the fixed default")

    print(f"[r2b] Exporting PNGs under {OUT_ROOT}")
    module.main()


if __name__ == "__main__":
    main()
