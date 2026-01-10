"""
step7_scopus_check.py

Analyzes retrieval performance by comparing Step 5 results against Step 2 Scopus records.
- Matches records based on DOI or Clean Title.
- FAIL-SAFE: If Author/Journal metadata is missing (whether missed by Scopus OR just missing columns),
  it fetches metadata from Crossref to ensure proper citations.
- FIX: Handles missing 'author_names' column gracefully.
- OUTPUT: CSVs, Charts, AND Word Documents (.docx)

Outputs:
1. 'step7_full_benchmark_status.csv'
2. 'step7_missed_eligible_studies.csv' + .docx
3. 'step7_passed_benchmark_studies.csv' + .docx
4. 'step7_retrieval_analysis_2x2.png'
"""

import os
import re
import time
import requests
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import html  # Fixes &amp; issues

# Try importing python-docx for Word export
try:
    from docx import Document
    from docx.shared import Pt
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False
    print("⚠️ 'python-docx' not installed. Word files will not be generated. Run: pip install python-docx")

def _safe_str(val):
    """Return a clean string or empty string if NaN/None."""
    if val is None:
        return ""
    if isinstance(val, float) and pd.isna(val):
        return ""
    # Fix HTML entities (e.g., &amp; -> &) and strip whitespace
    return html.unescape(str(val)).strip()

def clean_text(text):
    """Normalizes text: lowercase, remove non-alphanumeric."""
    if not isinstance(text, str):
        return ""
    return re.sub(r'[^a-z0-9]', '', text.lower())

def find_col(df, keywords):
    """Finds a column matching keywords."""
    # 1. Exact match (case insensitive)
    for col in df.columns:
        if col.strip().lower() in [k.lower() for k in keywords]:
            return col
    # 2. Partial match
    for col in df.columns:
        for k in keywords:
            if len(k) > 2 and k.lower() in col.lower():
                return col
    return None

def fetch_crossref_metadata(doi):
    """
    Fetches Author, Year, and Journal from Crossref for a given DOI.
    """
    if not doi or not isinstance(doi, str):
        return {}
    
    # FIX: Remove http prefix AND trailing periods/spaces which break the API
    clean_doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "").strip().rstrip('.')
    
    url = f"https://api.crossref.org/works/{clean_doi}"
    
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            data = r.json().get('message', {})
            
            # 1. Authors
            authors_list = data.get('author', [])
            author_parts = []
            for a in authors_list:
                family = a.get('family')
                given = a.get('given')
                if family:
                    name = family
                    if given: name += f", {given[0]}."
                    author_parts.append(name)
            
            if len(author_parts) > 3:
                author_str = ", ".join(author_parts[:3]) + " et al."
            else:
                author_str = ", ".join(author_parts)

            # 2. Journal
            container = data.get('container-title', [])
            journal = container[0] if container else ""

            # 3. Year
            issued = data.get('issued', {}).get('date-parts', [[None]])[0][0]
            
            return {
                "fetched_authors": author_str,
                "fetched_journal": journal,
                "fetched_year": str(issued) if issued else ""
            }
    except Exception:
        pass
    return {}

def build_citation(row):
    """
    Builds a Harvard-style citation using the best available data.
    Priority: Fetched (Crossref) > Scopus Merge > Original CSV
    """
    # --- 1. Resolve Author ---
    author = _safe_str(row.get("fetched_authors"))
    if not author: author = _safe_str(row.get("author_names")) # From Scopus
    if not author: author = _safe_str(row.get("authors"))      # From Original
    if not author: author = "Unknown Author"

    # --- 2. Resolve Year ---
    year_raw = _safe_str(row.get("fetched_year"))
    if not year_raw: year_raw = _safe_str(row.get("year_scopus"))
    if not year_raw: year_raw = _safe_str(row.get("year"))
    
    y_match = re.search(r'\d{4}', year_raw)
    year_str = y_match.group(0) if y_match else "n.d."

    # --- 3. Resolve Title ---
    title = _safe_str(row.get("title"))
    if not title: title = _safe_str(row.get("title_scopus"))

    # --- 4. Resolve Journal ---
    journal = _safe_str(row.get("fetched_journal"))
    if not journal: journal = _safe_str(row.get("publicationName")) # From Scopus
    
    # --- 5. Resolve DOI ---
    doi = _safe_str(row.get("doi"))
    if not doi: doi = _safe_str(row.get("doi_scopus"))

    # --- Build String ---
    parts = []
    parts.append(author)
    parts.append(f"({year_str})")
    
    if title:
        parts.append(f"'{title.rstrip('. ')}'.")
    
    if journal:
        parts.append(f"{journal}.")
    else:
        st = _safe_str(row.get("source_type"))
        if "Grey" in st:
            parts.append("[Grey Literature].")
        else:
            parts.append("[Journal unavailable].")

    if doi:
        if not doi.lower().startswith("http"):
            doi = f"https://doi.org/{doi}"
        parts.append(doi)

    return " ".join(parts)

def export_to_word(df, title, filename):
    """Exports a dataframe's 'citation' column to a Word document."""
    if not HAS_DOCX:
        return

    doc = Document()
    doc.add_heading(title, level=1)
    
    if 'citation' not in df.columns:
        doc.add_paragraph("No citation column found.")
        doc.save(filename)
        return

    # Add each citation as a list item
    for idx, row in df.iterrows():
        citation = row['citation']
        if pd.notna(citation) and citation:
            p = doc.add_paragraph(citation, style='List Number')
            # Optional: Add spacing between items
            p.paragraph_format.space_after = Pt(6)

    doc.save(filename)
    print(f"📄 Saved Word document: {filename}")

def run(config: dict):
    print("--- Running Step 7: Benchmark (Step 5) vs. Scopus (Step 2) ---")
    
    # Paths
    out_root = config.get("out_dir", "outputs")
    step7_dir = os.path.join(out_root, "step7")
    os.makedirs(step7_dir, exist_ok=True)

    scopus_file = os.path.join(out_root, "step2", "step2_total_records.csv")
    step5_file = os.path.join(out_root, "step5", "step5_eligibility_wide.csv")
    orig_bm_file = config.get("benchmark_csv")

    if not os.path.exists(scopus_file) or not os.path.exists(step5_file):
        print("❌ Missing input files.")
        return

    # Load Data
    scopus_df = pd.read_csv(scopus_file)
    step5_df = pd.read_csv(step5_file)
    orig_bm_df = pd.read_csv(orig_bm_file)

    # --- Prepare Step 5 ---
    s5_title_col = find_col(step5_df, ['title', 'study'])
    s5_doi_col = find_col(step5_df, ['doi'])
    s5_status_col = find_col(step5_df, ['final_decision', 'decision'])
    
    step5_df['match_title'] = step5_df[s5_title_col].apply(clean_text)
    step5_df['match_doi'] = step5_df[s5_doi_col].apply(clean_text) if s5_doi_col else ""

    # Source Type logic
    orig_title_col = find_col(orig_bm_df, ['Study', 'Title'])
    orig_type_col = find_col(orig_bm_df, ['Type', 'Source Type'])
    if orig_title_col and orig_type_col:
        orig_bm_df['match_title'] = orig_bm_df[orig_title_col].apply(clean_text)
        type_lookup = orig_bm_df.drop_duplicates('match_title').set_index('match_title')[orig_type_col].to_dict()
        step5_df['restored_type'] = step5_df['match_title'].map(type_lookup).fillna('Unknown')
    else:
        step5_df['restored_type'] = 'Unknown Source'

    def map_source_type(val):
        val = str(val).lower()
        if any(k in val for k in ['grey', 'report', 'thesis', 'working paper', 'brief']): return 'Grey Literature'
        return 'Peer Reviewed'
    step5_df['source_type'] = step5_df['restored_type'].apply(map_source_type)

    # --- Prepare Scopus ---
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

    # --- Merge Metadata from Scopus ---
    sc_author_col = find_col(scopus_df, ['Authors', 'Author', 'author_names', 'Creator'])
    sc_journal_col = find_col(scopus_df, ['Source title', 'publicationName', 'Journal'])
    sc_date_col = find_col(scopus_df, ['Year', 'coverDate', 'Date'])

    needed_cols = [sc_title_col]
    if sc_author_col: needed_cols.append(sc_author_col)
    if sc_journal_col: needed_cols.append(sc_journal_col)
    if sc_date_col: needed_cols.append(sc_date_col)
    if sc_doi_col: needed_cols.append(sc_doi_col)

    scopus_meta = scopus_df[needed_cols].dropna(subset=[sc_title_col]).copy()
    
    rename_map = {sc_title_col: 'title_scopus'}
    if sc_author_col: rename_map[sc_author_col] = 'author_names'
    if sc_journal_col: rename_map[sc_journal_col] = 'publicationName'
    if sc_date_col: rename_map[sc_date_col] = 'year_scopus'
    if sc_doi_col: rename_map[sc_doi_col] = 'doi_scopus'

    scopus_meta = scopus_meta.rename(columns=rename_map)
    scopus_meta["match_title"] = scopus_meta['title_scopus'].apply(clean_text)

    step5_df = step5_df.merge(scopus_meta, how="left", on="match_title", suffixes=("", "_scopus_merge"))

    # --- Match Status ---
    def check_inclusion(row):
        if row.get('match_doi') and row['match_doi'] in scopus_dois: return True
        if row.get('match_title') and row['match_title'] in scopus_titles: return True
        return False

    step5_df['in_scopus_retrieval'] = step5_df.apply(check_inclusion, axis=1)

    def simple_status(val):
        val = str(val).lower()
        if 'include' in val or 'pass' in val: return 'Passed'
        if 'unclear' in val: return 'Unclear'
        return 'Failed'
    step5_df['simple_status'] = step5_df[s5_status_col].apply(simple_status)

    # --- 7. REPAIR CITATIONS (Aggressive Mode + Safety Fix) ---
    print("\n--- repairing missing metadata for Eligible Studies (Fetching from Crossref) ---")
    
    # SAFETY: Ensure author_names column exists, even if Scopus didn't provide it
    if 'author_names' not in step5_df.columns:
        step5_df['author_names'] = ""
    
    # Use empty string for safety if column exists but has NaNs
    has_scopus_author = step5_df['author_names'].notna() & (step5_df['author_names'] != "")
    
    # We repair if:
    # 1. Passed AND Has DOI
    # 2. AND (Not in Scopus OR Scopus didn't give us an author name)
    mask_needs_repair = (
        (step5_df['simple_status'] == 'Passed') & 
        (step5_df[s5_doi_col].notna()) &
        ( (step5_df['in_scopus_retrieval'] == False) | (~has_scopus_author) )
    )
    
    to_repair_indices = step5_df[mask_needs_repair].index
    print(f"Fetching metadata for {len(to_repair_indices)} studies (including those with missing authors)...")

    for idx in tqdm(to_repair_indices):
        doi = step5_df.at[idx, s5_doi_col]
        meta = fetch_crossref_metadata(doi)
        if meta:
            step5_df.at[idx, 'fetched_authors'] = meta.get('fetched_authors')
            step5_df.at[idx, 'fetched_journal'] = meta.get('fetched_journal')
            step5_df.at[idx, 'fetched_year'] = meta.get('fetched_year')
        time.sleep(0.1) 

    # --- Build Citations ---
    step5_df["citation"] = step5_df.apply(build_citation, axis=1)

    # --- Export CSVs ---
    cols_to_drop = ['match_title', 'match_doi', 'restored_type', 'simple_status', 'clean_title']
    final_df = step5_df.drop(columns=[c for c in cols_to_drop if c in step5_df.columns])
    
    final_df.to_csv(os.path.join(step7_dir, "step7_full_benchmark_status.csv"), index=False)

    missed_df = final_df[
        (step5_df['simple_status'] == 'Passed') & 
        (final_df['in_scopus_retrieval'] == False)
    ]
    missed_df.to_csv(os.path.join(step7_dir, "step7_missed_eligible_studies.csv"), index=False)
    
    passed_df = final_df[step5_df['simple_status'] == 'Passed']
    passed_df.to_csv(os.path.join(step7_dir, "step7_passed_benchmark_studies.csv"), index=False)

    print(f"✅ Saved CSV outputs to {step7_dir}")

    # --- Export Word Docs ---
    print("\n--- Generating Word Documents ---")
    export_to_word(missed_df, "Benchmark List: Not Retrieved in Scopus Search Results", 
                   os.path.join(step7_dir, "step7_missed_eligible_studies.docx"))
    
    export_to_word(passed_df, "Full Benchmark List: Passed Eligibility", 
                   os.path.join(step7_dir, "step7_passed_benchmark_studies.docx"))

    # --- Visualization ---
    plot_configs = [
        {"title": "All Studies", "filter": lambda df: df},
        {"title": "Passed Studies", "filter": lambda df: df[df['simple_status'] == 'Passed']},
        {"title": "Unclear Studies", "filter": lambda df: df[df['simple_status'] == 'Unclear']},
        {"title": "Failed Studies", "filter": lambda df: df[df['simple_status'] == 'Failed']}
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()
    palette = {True: "#93c47d", False: "#e06666"} 
    cats = ['Peer Reviewed', 'Grey Literature']

    for i, cfg in enumerate(plot_configs):
        ax = axes[i]
        subset = cfg["filter"](step5_df)
        counts = subset.groupby(['source_type', 'in_scopus_retrieval']).size().unstack(fill_value=0)
        counts = counts.reindex(cats).fillna(0)
        if True not in counts.columns: counts[True] = 0
        if False not in counts.columns: counts[False] = 0

        p1 = ax.bar(counts.index, counts[True], label='In Scopus', color=palette[True], alpha=0.9)
        p2 = ax.bar(counts.index, counts[False], bottom=counts[True], label='Missed', color=palette[False], alpha=0.9)

        max_height = (counts[True] + counts[False]).max()
        if max_height > 0: ax.set_ylim(0, max_height * 1.15)

        ax.set_title(f"{cfg['title']} (n={len(subset)})", fontweight='bold')
        
        # Labels inside bars
        for bar_group in [p1, p2]:
            for bar in bar_group:
                h = bar.get_height()
                if h > 0: ax.text(bar.get_x() + bar.get_width()/2., bar.get_y() + h/2., f'{int(h)}', ha='center', va='center', color='white', fontweight='bold')
        
        # Percentages ABOVE bars
        totals = counts[True] + counts[False]
        for idx, cat in enumerate(counts.index):
            total = totals.loc[cat]
            found = counts.loc[cat, True]
            if total > 0:
                pct = (found / total) * 100
                ax.text(idx, total + (max_height * 0.02), f"{pct:.1f}%", 
                        ha='center', va='bottom', color='black', fontweight='bold', fontsize=10)

    plt.suptitle("Scopus Retrieval Analysis", fontsize=16, fontweight='bold')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.94), ncol=2, frameon=False)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(os.path.join(step7_dir, "step7_retrieval_analysis_2x2.png"))
    print("📊 Chart saved.")

if __name__ == "__main__":
    run({"benchmark_csv": "Benchmark List - List.csv", "out_dir": "outputs"})