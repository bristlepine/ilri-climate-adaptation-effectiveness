"""
step13d_update_manifest.py
---------------------------
Final step in the step13 sequence. Run this after every retrieval pass
(step13, step13a, step13b, step13c) to:

  1. Reconcile the manifest with whatever is now in the fulltext/ folder
     (marks newly present files as retrieved).

  2. Regenerate step13_manual.csv — all records still needing manual retrieval.

  3. Export a versioned campus-retrieval list:
       step13_missing_papers_01.csv, _02.csv, ... (auto-incremented each run)
     Contains only records with a DOI and a 403 failure — these are publisher-
     blocked papers accessible via a campus library proxy. Share this list
     with a collaborator to download manually.

  4. Update step13_summary.json with current counts and breakdowns.

Usage:
    python scripts/step13d_update_manifest.py
"""

import glob
import json
import re
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
    safe = re.sub(r'[/\\]', '_', doi.strip())
    safe = re.sub(r'[^\w\-.]', '_', safe)
    return f"doi_{safe}"


def publisher_label(doi: str) -> str:
    if doi.startswith("10.1080"):                        return "Taylor & Francis"
    if doi.startswith("10.1111") or doi.startswith("10.1002"): return "Wiley"
    if doi.startswith("10.1177"):                        return "SAGE"
    if doi.startswith("10.3390"):                        return "MDPI"
    if doi.startswith("10.1007") or doi.startswith("10.1023"): return "Springer"
    if doi.startswith("10.1016"):                        return "Elsevier"
    if doi.startswith("10.1093"):                        return "Oxford"
    if doi.startswith("10.1017"):                        return "Cambridge"
    return "Other"


def next_version_path(base_dir: Path, stem: str, new_dois: set) -> tuple[Path, bool]:
    """
    Return (path, is_new_version).
    Only increments the version number if the set of DOIs differs from the last version.
    If nothing changed, returns the existing path and is_new_version=False.
    """
    pattern = str(base_dir / f"{stem}_*.csv")
    existing = sorted(glob.glob(pattern))
    if existing:
        last_path = Path(existing[-1])
        try:
            last_dois = set(pd.read_csv(last_path, dtype=str, usecols=["doi"]).fillna("")["doi"])
            if last_dois == new_dois:
                return last_path, False  # identical — reuse, don't bump
        except Exception:
            pass
        m = re.search(r'_(\d+)\.csv$', existing[-1])
        n = int(m.group(1)) + 1 if m else 1
    else:
        n = 1
    return base_dir / f"{stem}_{n:02d}.csv", True


def main():
    base, fulltext_dir, manifest_csv, summary_json = step13_dirs(out_dir)

    if not manifest_csv.exists():
        print("[step13d] No manifest found — run step13 first.")
        return

    df = pd.read_csv(manifest_csv, dtype=str).fillna("")

    # -------------------------------------------------------------------------
    # 1. Reconcile fulltext/ folder with manifest
    # -------------------------------------------------------------------------
    existing = {f.name for f in fulltext_dir.iterdir()} if fulltext_dir.exists() else set()
    print(f"[step13d] Files in fulltext/ : {len(existing):,}")

    newly_found = 0
    for idx, row in df.iterrows():
        if row["status"] == "retrieved":
            continue

        # Try matching via existing file_path field
        fp = row.get("file_path", "")
        fname = Path(fp).name if fp else ""
        if fname and fname in existing:
            df.at[idx, "status"] = "retrieved"
            df.at[idx, "note"] = (row.get("note", "") + " | manually_added").strip(" |")
            newly_found += 1
            continue

        # Derive filename from DOI
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
    print(f"[step13d] Newly marked retrieved : {newly_found:,}")

    # -------------------------------------------------------------------------
    # 2. Regenerate step13_manual.csv (all needs_manual records)
    # -------------------------------------------------------------------------
    manual_csv = base / "step13_manual.csv"
    manual_cols = ["dedupe_key", "doi", "scopus_id", "title", "year", "pub", "missing_abstract", "note"]
    manual_cols = [c for c in manual_cols if c in df.columns]
    manual_df = df[df["status"] == "needs_manual"][manual_cols]
    manual_df.to_csv(manual_csv, index=False)

    # -------------------------------------------------------------------------
    # 3. Versioned campus-retrieval list (403-blocked papers with DOIs)
    # -------------------------------------------------------------------------
    missing = df[df["status"] == "needs_manual"].copy()
    blocked = missing[
        (missing["doi"] != "") &
        missing["note"].str.contains("403", na=False)
    ].copy()
    blocked["publisher"] = blocked["doi"].apply(publisher_label)
    campus_cols = [c for c in ["doi", "title", "year", "publisher", "pub", "note"] if c in blocked.columns]
    new_dois = set(blocked["doi"].fillna(""))
    campus_csv, is_new = next_version_path(base, "step13_missing_papers", new_dois)
    if is_new:
        blocked[campus_cols].to_csv(campus_csv, index=False)
        print(f"[step13d] Campus retrieval list  : {campus_csv.name}  ({len(blocked):,} records)  [new version]")
    else:
        print(f"[step13d] Campus retrieval list  : {campus_csv.name}  ({len(blocked):,} records)  [unchanged — no new version created]")

    # -------------------------------------------------------------------------
    # 4. Status summary + update summary JSON
    # -------------------------------------------------------------------------
    status_counts = df["status"].value_counts(dropna=False).to_dict()
    total = len(df)
    retrieved  = int(status_counts.get("retrieved", 0))
    needs_manual = int(status_counts.get("needs_manual", 0))

    # Missing — by publisher (DOI prefix)
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
    pub_counts["Other"]  = needs_manual - sum(pub_counts.values())

    # Missing — by failure reason
    def classify_note(note: str) -> str:
        n = note.lower()
        if "403" in n:              return "HTTP 403 (bot-blocked, OA available)"
        if "404" in n:              return "HTTP 404 (broken link)"
        if "nameresolution" in n or "failed to resolve" in n: return "DNS / connection error"
        if "timeout" in n:          return "Timeout"
        if "no oa" in n or "no open" in n: return "No OA version found (subscription only)"
        if not note.strip():        return "No DOI / not attempted"
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
    print(f"\n[step13d] Manifest updated       : {manifest_csv}")
    print(f"[step13d] Manual CSV updated     : {manual_csv}  ({needs_manual:,} records)")
    print(f"[step13d] Campus retrieval list  : {campus_csv.name}  ({len(blocked):,} records)")

    # Update summary JSON
    if summary_json.exists():
        with open(summary_json) as f:
            summary = json.load(f)
    else:
        summary = {}

    summary["status_counts"] = {k: int(v) for k, v in status_counts.items()}
    summary["retrieval_rate_pct"] = round(100 * retrieved / total, 1) if total else 0
    summary["step13d_newly_found"] = newly_found
    summary["missing_by_publisher"] = {k: int(v) for k, v in pub_counts.items()}
    summary["missing_by_failure_reason"] = {k: int(v) for k, v in reason_counts.items()}
    summary["campus_retrieval_csv"] = str(campus_csv)
    summary["step13d_timestamp_utc"] = __import__("datetime").datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[step13d] Summary updated        : {summary_json}")


if __name__ == "__main__":
    main()
