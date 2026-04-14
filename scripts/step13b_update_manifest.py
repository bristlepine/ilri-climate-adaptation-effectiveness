"""
step13b_update_manifest.py
---------------------------
Updates the step13 manifest to reflect files that were manually copied
into the fulltext/ directory after the initial pipeline run (e.g. from
a campus library retrieval session).

For every record currently marked needs_manual or failed, checks whether
a matching file now exists in fulltext/. If so, marks it retrieved.

Also regenerates step13_manual.csv (the missing_papers list) and
step13_summary.json with updated counts.

Usage:
    python scripts/step13b_update_manifest.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

here = Path(__file__).parent
sys.path.insert(0, str(here))

from step13_retrieve_fulltext import step13_dirs

try:
    import config as _cfg
    out_dir = Path(getattr(_cfg, "out_dir", "outputs"))
except ImportError:
    out_dir = here / "outputs"

def doi_to_filename_base(doi: str) -> str:
    """Convert a DOI to the base filename stem step13 uses (doi_<sanitised>)."""
    import re
    safe = re.sub(r'[/\\]', '_', doi.strip())
    safe = re.sub(r'[^\w\-.]', '_', safe)
    return f"doi_{safe}"


def main():
    base, fulltext_dir, manifest_csv, summary_json = step13_dirs(out_dir)

    if not manifest_csv.exists():
        print("[step13b] No manifest found — run step13 first.")
        return

    df = pd.read_csv(manifest_csv, dtype=str).fillna("")

    # Build set of filenames currently in fulltext/
    existing = {f.name for f in fulltext_dir.iterdir()} if fulltext_dir.exists() else set()
    print(f"[step13b] Files in fulltext/: {len(existing):,}")

    newly_found = 0
    for idx, row in df.iterrows():
        if row["status"] == "retrieved":
            continue

        # First try: match via existing file_path field
        fp = row.get("file_path", "")
        fname = Path(fp).name if fp else ""
        if fname and fname in existing:
            df.at[idx, "status"] = "retrieved"
            df.at[idx, "note"] = (row.get("note", "") + " | manually_added").strip(" |")
            newly_found += 1
            continue

        # Second try: derive filename from DOI
        doi = row.get("doi", "")
        if doi:
            base_name = doi_to_filename_base(doi)
            for ext in [".pdf", ".html", ".htm"]:
                candidate = base_name + ext
                if candidate in existing:
                    df.at[idx, "status"] = "retrieved"
                    df.at[idx, "file_path"] = str(fulltext_dir / candidate)
                    df.at[idx, "note"] = (row.get("note", "") + " | manually_added").strip(" |")
                    newly_found += 1
                    break

    df.to_csv(manifest_csv, index=False)
    print(f"[step13b] Newly marked retrieved: {newly_found:,}")

    # Regenerate manual CSV
    manual_csv = base / "step13_manual.csv"
    manual_cols = ["dedupe_key", "doi", "scopus_id", "title", "year", "pub", "missing_abstract", "note"]
    manual_cols = [c for c in manual_cols if c in df.columns]
    manual_df = df[df["status"] == "needs_manual"][manual_cols]
    manual_df.to_csv(manual_csv, index=False)

    # Status summary
    status_counts = df["status"].value_counts(dropna=False).to_dict()
    total = len(df)
    retrieved = int(status_counts.get("retrieved", 0))
    needs_manual = int(status_counts.get("needs_manual", 0))
    failed = int(status_counts.get("failed", 0))

    # --- Missing papers breakdown ---
    missing = df[df["status"] == "needs_manual"].copy()

    # By publisher (DOI prefix)
    publisher_map = [
        ("MDPI",              "10.3390"),
        ("Taylor & Francis",  "10.1080"),
        ("Wiley",             "10.1111"),
        ("Wiley",             "10.1002"),
        ("Springer",          "10.1007"),
        ("Springer",          "10.1023"),
        ("Elsevier",          "10.1016"),
        ("Cambridge",         "10.1017"),
        ("Oxford",            "10.1093"),
        ("SAGE",              "10.1177"),
        ("Frontiers",         "10.3389"),
    ]
    pub_counts: dict = {}
    for label, prefix in publisher_map:
        n = int(missing["doi"].str.startswith(prefix).sum())
        if n:
            pub_counts[label] = pub_counts.get(label, 0) + n
    pub_counts["No DOI"] = int((missing["doi"] == "").sum())
    pub_counts["Other"] = needs_manual - sum(pub_counts.values())

    # By failure reason (parsed from note field)
    import re as _re
    def classify_note(note: str) -> str:
        n = note.lower()
        if "403" in n:
            return "HTTP 403 (bot-blocked, OA available)"
        if "404" in n:
            return "HTTP 404 (broken link)"
        if "nameresolution" in n or "failed to resolve" in n:
            return "DNS / connection error (network issue at run time)"
        if "timeout" in n:
            return "Timeout"
        if "no oa" in n or "no open" in n:
            return "No OA version found (subscription only)"
        if note.strip() == "":
            return "No DOI / not attempted"
        return "Other"

    reason_counts: dict = {}
    for note in missing["note"]:
        r = classify_note(str(note))
        reason_counts[r] = reason_counts.get(r, 0) + 1

    print()
    print("=" * 56)
    print(f"  Total records          : {total:,}")
    print(f"  Retrieved              : {retrieved:,}  ({100*retrieved/total:.1f}%)")
    print(f"  Still needs manual     : {needs_manual:,}  ({100*needs_manual/total:.1f}%)")
    print()
    print("  Missing — by publisher:")
    for pub, n in sorted(pub_counts.items(), key=lambda x: -x[1]):
        print(f"    {pub:<22}: {n:,}")
    print()
    print("  Missing — by failure reason:")
    for reason, n in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"    {reason}: {n:,}")
    print("=" * 56)
    print(f"\n[step13b] Updated manifest : {manifest_csv}")
    print(f"[step13b] Updated manual CSV: {manual_csv}  ({needs_manual:,} records still outstanding)")

    # Update summary JSON
    if summary_json.exists():
        with open(summary_json) as f:
            summary = json.load(f)
    else:
        summary = {}

    summary["status_counts"] = {k: int(v) for k, v in status_counts.items()}
    summary["step13b_newly_found"] = newly_found
    summary["missing_by_publisher"] = {k: int(v) for k, v in pub_counts.items()}
    summary["missing_by_failure_reason"] = {k: int(v) for k, v in reason_counts.items()}
    summary["step13b_timestamp_utc"] = __import__("datetime").datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[step13b] Updated summary   : {summary_json}")

if __name__ == "__main__":
    main()
