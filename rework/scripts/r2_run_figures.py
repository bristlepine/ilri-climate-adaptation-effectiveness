#!/usr/bin/env python3
"""
r2_run_figures.py

Re-run scripts/step16_map_visualise.py's existing run() against the rework
data (rework/outputs/, built by r1_build_database.py), to regenerate every
human-track figure (EGM, geographic, methodology, equity, temporal,
LLM-vs-human) with the 115-record rework database instead of the original
151-record one.

No edits to scripts/step16_map_visualise.py itself — it already takes
out_dir via a config dict. The one behaviour we must not inherit is
_sync_frontend(), which unconditionally copies step16 outputs into the live
Next.js frontend/public/map/data/ directory: that would overwrite the real
site's data with prelim rework figures. We monkeypatch it to a no-op rather
than edit the shared script, so this stays a zero-touch wrapper.

Prerequisite: rework/outputs/step15/ must already contain step15_human.csv
and step15b_summary.json (run r1_build_database.py first), plus the symlinks
to the unaffected upstream meta files (step12/step12b/step13/step14/step2b,
and step15_coded.csv/.meta.json — the LLM exploratory track, unchanged by
the rework) set up alongside it.

Run:
  conda run -n ilri01 python rework/scripts/r2_run_figures.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REWORK_ROOT = Path(__file__).parent.parent
REPO_ROOT = REWORK_ROOT.parent
STEP16_PATH = REPO_ROOT / "scripts" / "step16_map_visualise.py"
OUT_DIR = REWORK_ROOT / "outputs"


def _load_step16_module():
    spec = importlib.util.spec_from_file_location("step16_map_visualise", STEP16_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    human_csv = OUT_DIR / "step15" / "step15_human.csv"
    if not human_csv.exists():
        raise SystemExit(
            f"[r2] {human_csv} not found — run r1_build_database.py first."
        )

    module = _load_step16_module()
    module._sync_frontend = lambda *args, **kwargs: print(
        "[r2] Skipping frontend sync (rework is a prelim side pipeline, "
        "not the live site)."
    )

    print(f"[r2] Running step16_map_visualise.run() with out_dir={OUT_DIR}")
    meta = module.run({"out_dir": str(OUT_DIR)})
    print(f"[r2] Done: {meta}")


if __name__ == "__main__":
    main()
