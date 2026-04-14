"""
step13a_rescan_fulltexts.py
----------------------------
Re-checks all HTML files in the step13 fulltext directory against the
current fake-page / paywall pattern list in step13_retrieve_fulltext.py.

Run this after:
  - Copying in new files retrieved externally (e.g. from a campus library run)
  - Updating the bad-pattern list in step13_retrieve_fulltext.py

Deletes confirmed fakes, marks them needs_manual in the manifest.
Does not download anything.

Usage:
    python scripts/step13a_rescan_fulltexts.py
"""

import sys
from pathlib import Path

# Allow running from repo root or from scripts/
here = Path(__file__).parent
sys.path.insert(0, str(here))

from step13_retrieve_fulltext import rescan_html_files

try:
    import config as _cfg
    out_dir = Path(getattr(_cfg, "out_dir", "outputs"))
except ImportError:
    out_dir = here / "outputs"

if __name__ == "__main__":
    print(f"[step13a] Scanning HTML files in: {out_dir / 'step13' / 'fulltext'}")
    stats = rescan_html_files(out_dir)
    if stats:
        print(f"[step13a] Done — {stats['rescanned']} checked, "
              f"{stats['newly_flagged']} fakes removed, "
              f"{stats['kept']} kept.")
    else:
        print("[step13a] Nothing to rescan — run step13 first to populate the fulltext folder.")
