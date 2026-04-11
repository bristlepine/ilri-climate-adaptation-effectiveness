#!/usr/bin/env python3
"""
stepX_sync_frontend.py

Syncs pipeline outputs to the Next.js frontend public directory so they are
served by the website. Run this after any step that updates figures or data.

Files synced:
  outputs/step16/*.png        → frontend/public/map/
  outputs/step15/step15_coded.csv → frontend/public/map/step15_coded.csv

Usage:
  python stepX_sync_frontend.py

Or via run.py with run_stepX = 1
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUTPUTS = ROOT / "scripts" / "outputs"
PUBLIC_MAP = ROOT / "frontend" / "public" / "map"


def sync_frontend() -> None:
    PUBLIC_MAP.mkdir(parents=True, exist_ok=True)

    # --- Step 16 figures ---
    step16_dir = OUTPUTS / "step16"
    figures_copied = 0
    for png in step16_dir.glob("*.png"):
        dest = PUBLIC_MAP / png.name
        shutil.copy2(png, dest)
        print(f"  [map] {png.name} → public/map/")
        figures_copied += 1

    if figures_copied == 0:
        print("  [map] No step16 figures found — run step16 first.")

    # --- Step 15 coded CSV ---
    coded_csv = OUTPUTS / "step15" / "step15_coded.csv"
    if coded_csv.exists():
        dest = PUBLIC_MAP / "step15_coded.csv"
        shutil.copy2(coded_csv, dest)
        print(f"  [map] step15_coded.csv → public/map/")
    else:
        print("  [map] step15_coded.csv not found — run step15 first.")

    print(f"\nSync complete. {figures_copied} figure(s) copied to frontend/public/map/")


if __name__ == "__main__":
    sync_frontend()
