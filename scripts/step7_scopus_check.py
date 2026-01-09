"""
step7_scopus_check.py

Analyzes retrieval performance by comparing Step 5 results against Step 2 Scopus records.

Outputs:
1. 'step7_full_benchmark_status.csv': The original Step 5 file with an appended 'in_scopus_retrieval' column.
2. 'step7_missed_eligible_studies.csv': A list of ONLY the eligible (Passed) studies that Scopus missed.
3. 'step7_retrieval_analysis_2x2.png': Faceted bar charts with proper spacing.
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt

def _safe_str(val):
    """Return a clean string or empty string if NaN/None."""
    if val is None:
        return ""
    if isinstance(val, float) and pd.isna(val):
        return ""
    return str(val).strip()

def build_citation(row):
    title = _safe_str(row.get("title"))
    year = _safe_str(row.get("year"))
    journal = _safe_str(row.get("publicationName"))
    doi = _safe_str(row.get("doi"))

    parts = []

    if year:
        parts.append(f"({year[:4]})")

    if title:
        parts.append(f"{title.rstrip('.')}.")
    else:
        parts.append("[Title unavailable].")

    if journal:
        parts.append(f"{journal}.")
    else:
        parts.append("[Journal unavailable – not indexed in Scopus].")

    if doi:
        if not doi.lower().startswith("http"):
            doi = f"https://doi.org/{doi}"
        parts.append(doi)
    else:
        parts.append("[DOI unavailable].")

    return " ".join(parts)

def clean_text(text):
    """Normalizes text: lowercase, remove non-alphanumeric."""
    if not isinstance(text, str):
        return ""
    return re.sub(r'[^a-z0-9]', '', text.lower())

def find_col(df, keywords):
    """Finds a column matching keywords (exact or partial)."""
    # 1. Exact match
    for col in df.columns:
        if col.strip().lower() in [k.lower() for k in keywords]:
            return col
    # 2. Partial match (safe)
    for col in df.columns:
        for k in keywords:
            if len(k) > 2 and k.lower() in col.lower():
                return col
    return None

def run(config: dict):
    print("--- Running Step 7: Benchmark (Step 5) vs. Scopus (Step 2) ---")
    
    # --- 1. Define Paths & Directories ---
    out_root = config.get("out_dir", "outputs")
    step7_dir = os.path.join(out_root, "step7")
    os.makedirs(step7_dir, exist_ok=True)

    scopus_file = os.path.join(out_root, "step2", "step2_total_records.csv")
    step5_file = os.path.join(out_root, "step5", "step5_eligibility_wide.csv")
    orig_bm_file = config.get("benchmark_csv")

    # --- 2. Load Files ---
    if not os.path.exists(scopus_file):
        print(f"❌ Error: Step 2 file not found: {scopus_file}")
        return
    if not os.path.exists(step5_file):
        print(f"❌ Error: Step 5 result file not found: {step5_file}")
        return

    try:
        scopus_df = pd.read_csv(scopus_file)
        step5_df = pd.read_csv(step5_file)
        orig_bm_df = pd.read_csv(orig_bm_file)
        print(f"Loaded:")
        print(f"  - Step 5 (Targets): {len(step5_df)} records")
        print(f"  - Step 2 (Scopus):  {len(scopus_df)} records")
    except Exception as e:
        print(f"Error loading CSVs: {e}")
        return

    # --- 3. Prepare Step 5 Data ---
    s5_title_col = find_col(step5_df, ['title', 'study'])
    s5_doi_col = find_col(step5_df, ['doi'])
    s5_status_col = find_col(step5_df, ['final_decision', 'decision'])
    
    if not s5_title_col: 
        print("CRITICAL: No Title column in Step 5 file.")
        return

    # Create Clean Match Keys (Internal use only)
    step5_df['match_title'] = step5_df[s5_title_col].apply(clean_text)
    step5_df['match_doi'] = step5_df[s5_doi_col].apply(clean_text) if s5_doi_col else ""

    # --- 4. Restore "Source Type" (Peer/Grey) ---
    orig_title_col = find_col(orig_bm_df, ['Study', 'Title'])
    orig_type_col = find_col(orig_bm_df, ['Type', 'Source Type'])
    
    if orig_title_col and orig_type_col:
        orig_bm_df['match_title'] = orig_bm_df[orig_title_col].apply(clean_text)
        type_lookup = orig_bm_df.drop_duplicates('match_title').set_index('match_title')[orig_type_col].to_dict()
        step5_df['restored_type'] = step5_df['match_title'].map(type_lookup).fillna('Unknown')
    else:
        step5_df['restored_type'] = 'Unknown Source'

    # Map Source Type (Default Peer, Keyword Grey)
    def map_source_type(val):
        val = str(val).lower()
        grey_keywords = ['grey', 'gray', 'report', 'thesis', 'dissertation', 'working paper', 'brief', 'website', 'blog']
        if any(k in val for k in grey_keywords):
            return 'Grey Literature'
        return 'Peer Reviewed'
    
    step5_df['source_type'] = step5_df['restored_type'].apply(map_source_type)

    # --- 5. Prepare Scopus Data ---
    sc_title_col = find_col(scopus_df, ['Title', 'Article Title'])
    sc_doi_col = find_col(scopus_df, ['DOI', 'doi'])
    
    scopus_df['clean_title'] = scopus_df[sc_title_col].apply(clean_text)
    scopus_titles = set(scopus_df['clean_title'].dropna().unique())
    if "" in scopus_titles: scopus_titles.remove("")

    scopus_dois = set()
    if sc_doi_col:
        scopus_df['clean_doi'] = scopus_df[sc_doi_col].apply(clean_text)
        scopus_dois = set(scopus_df['clean_doi'].dropna().unique())
        if "" in scopus_dois: scopus_dois.remove("")

    # --- 5b. Attach Scopus metadata for citation ---
    sc_meta_cols = {
        "publicationName": sc_title_col.replace("Title", "publicationName") if sc_title_col else None,
        "coverDate": "coverDate",
        "doi": sc_doi_col
    }

    scopus_meta = scopus_df[[
        sc_title_col,
        "publicationName",
        "coverDate",
        sc_doi_col
    ]].dropna(subset=[sc_title_col]).copy()

    scopus_meta["match_title"] = scopus_meta[sc_title_col].apply(clean_text)

    step5_df = step5_df.merge(
        scopus_meta,
        how="left",
        on="match_title",
        suffixes=("", "_scopus")
    )

    # Extract year
    # Ensure year is numeric where possible
    step5_df["year"] = pd.to_numeric(step5_df.get("year"), errors="coerce")

    # Only fill missing years from Scopus coverDate
    if "coverDate" in step5_df.columns:
        step5_df.loc[
            step5_df["year"].isna() & step5_df["coverDate"].notna(),
            "year"
        ] = (
            step5_df.loc[
                step5_df["year"].isna(),
                "coverDate"
            ]
            .astype(str)
            .str[:4]
            .astype(float, errors="ignore")
        )

    # --- 6. Match Logic ---
    def check_inclusion(row):
        if row.get('match_doi') and row['match_doi'] in scopus_dois:
            return True
        if row.get('match_title') and row['match_title'] in scopus_titles:
            return True
        return False

    # This is the boolean status column
    step5_df['in_scopus_retrieval'] = step5_df.apply(check_inclusion, axis=1)

    # --- 7. Simplify Status for Logic/Plotting ---
    def simple_status(val):
        val = str(val).lower()
        if 'include' in val or 'pass' in val: return 'Passed'
        if 'unclear' in val: return 'Unclear'
        return 'Failed'

    step5_df['simple_status'] = step5_df[s5_status_col].apply(simple_status)

    # --- Build citation column ---
    step5_df["citation"] = step5_df.apply(build_citation, axis=1)


    # --- 8. Export Data ---
    
    # A. Full File with Appended Status
    # We drop the internal matching columns to keep it clean
    cols_to_drop = [
        'match_title',
        'match_doi',
        'restored_type',
        'simple_status',
        'clean_title'
    ]
    final_df = step5_df.drop(columns=[c for c in cols_to_drop if c in step5_df.columns])
    
    out_csv_full = os.path.join(step7_dir, "step7_full_benchmark_status.csv")
    final_df.to_csv(out_csv_full, index=False)
    print(f"✅ Saved full status file: {out_csv_full}")

    # B. Missed Eligible Studies (Passed but NOT in Scopus)
    missed_df = final_df[
        (step5_df['simple_status'] == 'Passed') & 
        (final_df['in_scopus_retrieval'] == False)
    ]
    
    out_csv_missed = os.path.join(step7_dir, "step7_missed_eligible_studies.csv")
    missed_df.to_csv(out_csv_missed, index=False)
    print(f"✅ Saved list of {len(missed_df)} missed eligible studies: {out_csv_missed}")

    # C. All Passed Benchmark Studies (FULL LIST)
    passed_df = final_df[step5_df['simple_status'] == 'Passed']

    out_csv_passed = os.path.join(step7_dir, "step7_passed_benchmark_studies.csv")
    passed_df.to_csv(out_csv_passed, index=False)

    print(f"✅ Saved full list of passed benchmark studies: {out_csv_passed}")


    # --- 9. Visualization (2x2 Grid) ---
    plot_configs = [
        {"title": "All Studies", "filter": lambda df: df},
        {"title": "Passed Studies", "filter": lambda df: df[df['simple_status'] == 'Passed']},
        {"title": "Unclear Studies", "filter": lambda df: df[df['simple_status'] == 'Unclear']},
        {"title": "Failed Studies", "filter": lambda df: df[df['simple_status'] == 'Failed']}
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()
    
    # Muted Palette
    palette = {True: "#93c47d", False: "#e06666"} 
    cats = ['Peer Reviewed', 'Grey Literature']

    for i, cfg in enumerate(plot_configs):
        ax = axes[i]
        subset = cfg["filter"](step5_df)
        
        counts = subset.groupby(['source_type', 'in_scopus_retrieval']).size().unstack(fill_value=0)
        counts = counts.reindex(cats).fillna(0)
        
        if True not in counts.columns: counts[True] = 0
        if False not in counts.columns: counts[False] = 0

        # Stacked Bars
        p1 = ax.bar(counts.index, counts[True], label='In Scopus search results', color=palette[True], alpha=0.9)
        p2 = ax.bar(counts.index, counts[False], bottom=counts[True], label='Not in Scopus search results', color=palette[False], alpha=0.9)

        # Dynamic Y-Axis (15% headroom)
        max_height = (counts[True] + counts[False]).max()
        if max_height > 0:
            ax.set_ylim(0, max_height * 1.15)

        ax.set_title(f"{cfg['title']} (n={len(subset)})", fontsize=14, fontweight='bold')
        if i % 2 == 0: ax.set_ylabel("Count")
        ax.grid(axis='y', linestyle='--', alpha=0.3)

        # Annotations (Numbers)
        for bar_group in [p1, p2]:
            for bar in bar_group:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_y() + height/2., 
                            f'{int(height)}', ha='center', va='center', color='white', fontweight='bold')
        
        # Annotations (Percent)
        totals = counts[True] + counts[False]
        for idx, cat in enumerate(counts.index):
            total = totals.loc[cat]
            found = counts.loc[cat, True]
            if total > 0:
                pct = (found / total) * 100
                ax.text(idx, total + (total * 0.02), f"{pct:.1f}%", 
                        ha='center', va='bottom', color='black', fontweight='bold', fontsize=10)

    # Layout
    plt.suptitle("Scopus Retrieval Analysis", fontsize=18, y=0.98, fontweight='bold')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.94), ncol=2, fontsize=12, frameon=False)
    plt.subplots_adjust(top=0.88, hspace=0.3)
    
    out_path = os.path.join(step7_dir, "step7_retrieval_analysis_2x2.png")
    plt.savefig(out_path, bbox_inches='tight')
    print(f"📊 Chart saved to: {out_path}")

    # Summary
    print("\n--- Summary Statistics ---")
    summary = step5_df.groupby(['simple_status', 'source_type']).agg(
        Total=('in_scopus_retrieval', 'size'),
        Found=('in_scopus_retrieval', 'sum'),
        Missing=('in_scopus_retrieval', lambda x: (~x).sum())
    )
    summary['% Found'] = (summary['Found'] / summary['Total'] * 100).round(1)
    print(summary)

if __name__ == "__main__":
    run({"benchmark_csv": "Benchmark List - List.csv", "out_dir": "outputs"})