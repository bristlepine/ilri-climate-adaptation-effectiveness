#!/usr/bin/env python3
"""
step6_visualize.py

Generates STATIC summary charts and a formatted Excel report.
Final Updates:
- BAR CHART: Displays "Pass Rate %" above each bar instead of Total Count.
- EXCEL: Strict sorting (Pass -> Unclear No Abstract -> Unclear -> Fail).
- DATA: Pulls abstracts from step4_abstracts.csv.
- LAYOUT: Fixed absolute positioning for Titles/Legends.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from openpyxl.styles import PatternFill, Alignment
import config as cfg

# --- CONFIG ---
PALETTE = {
    "pass": "#d4edda",    
    "fail": "#f8d7da",    
    "unclear": "#fff3cd", 
    "excel_pass": "C6EFCE", 
    "excel_fail": "FFC7CE", 
    "excel_unclear": "FFEB9C" 
}

def _clean_doi(val):
    """Standardizes DOI for matching."""
    val = str(val).lower().strip()
    for p in ["https://doi.org/", "http://doi.org/", "doi:"]:
        if val.startswith(p):
            val = val[len(p):]
    return val

def _get_sort_rank(val):
    """
    Custom Sort Order:
    1. Pass
    2. Unclear (No Abstract)
    3. Unclear (General)
    4. Fail
    """
    val = str(val).lower().strip()
    if any(x in val for x in ["yes", "include", "pass"]): return 1
    if "no_abstract" in val: return 2
    if any(x in val for x in ["unclear", "pending"]): return 3
    if any(x in val for x in ["no", "exclude", "fail"]): return 4
    return 5

def _normalize_decision(val):
    val = str(val).lower().strip()
    if any(x in val for x in ["yes", "include", "pass"]): return 2
    if any(x in val for x in ["no", "exclude", "fail"]): return 0
    return 1 

def _get_smart_key(row):
    doi = _clean_doi(row.get("doi", ""))
    if doi and doi not in ["nan", "none", ""]: return doi
    title = str(row.get("title", "")).strip().lower()
    return "".join(e for e in title if e.isalnum())

def _find_col(df, candidates):
    for c in candidates:
        if c in df.columns: return c
    return None

def run_step6(config: dict) -> dict:
    out_dir = config.get("out_dir") or getattr(cfg, "out_dir", "outputs")
    viz_dir = os.path.join(out_dir, "step6")
    os.makedirs(viz_dir, exist_ok=True)
    
    step5_csv = os.path.join(out_dir, "step5", "step5_eligibility_wide.csv")
    step3_csv = os.path.join(out_dir, "step3", "step3_benchmark_list.csv")
    step4_csv = os.path.join(out_dir, "step4", "step4_abstracts.csv") 
    out_excel = os.path.join(viz_dir, "eligibility_report.xlsx")

    if not os.path.exists(step5_csv):
        print("❌ Missing Step 5 output.")
        return {"status": "error"}

    print(f"[Step 6] Generating visuals in: {viz_dir}")

    # 1. Load Data
    df_res = pd.read_csv(step5_csv)
    df_meta = pd.read_csv(step3_csv)
    
    # 2. LOAD ABSTRACTS FROM STEP 4
    abstract_map = {}
    if os.path.exists(step4_csv):
        print("   Found Step 4 Abstracts. Loading...")
        df_abs = pd.read_csv(
            step4_csv,
            engine="python",
            on_bad_lines="skip"
        )
        abs_doi_col = _find_col(df_abs, ["doi", "DOI"])
        abs_txt_col = _find_col(df_abs, ["abstract", "Abstract", "text", "content"])
        
        if abs_doi_col and abs_txt_col:
            for _, row in df_abs.iterrows():
                d = _clean_doi(row[abs_doi_col])
                t = str(row[abs_txt_col])
                if d and t and t.lower() != "nan":
                    abstract_map[d] = t
    else:
        print("   ⚠️ Step 4 file not found. Abstracts may be missing.")

    # 3. Prepare Metadata
    t_col = _find_col(df_meta, ["title", "Title", "Study", "Article Title"])
    if t_col: df_meta.rename(columns={t_col: "title"}, inplace=True)
    else: df_meta["title"] = df_meta["doi"]
    
    # Create Keys
    df_res["_join_key"] = df_res.apply(_get_smart_key, axis=1)
    df_meta["_join_key"] = df_meta.apply(_get_smart_key, axis=1)
    
    # 4. Merge Results + Title
    merged = pd.merge(df_res, df_meta[["_join_key", "title"]], on="_join_key", how="left")
    merged = merged.drop_duplicates(subset=["_join_key"]) 
    
    # Clean Artifacts
    if "title_x" in merged.columns: merged.rename(columns={"title_x": "title"}, inplace=True)
    if "title_y" in merged.columns: merged["title"] = merged["title"].fillna(merged["title_y"])
    merged["title"] = merged["title"].fillna(merged["doi"])
    merged.drop(columns=["_join_key", "title_y"], inplace=True, errors="ignore")

    # 5. INJECT ABSTRACTS
    def get_abstract(row):
        d = _clean_doi(row.get("doi", ""))
        return abstract_map.get(d, "") 
    
    merged["abstract"] = merged.apply(get_abstract, axis=1)
    merged["abstract"] = merged["abstract"].replace("", "No Abstract Available")

    # 6. Data Cleaning
    crit_cols = sorted([c for c in merged.columns if "_decision" in c and "final" not in c.lower()])
    cols_to_fix = crit_cols + ["final_decision"]
    
    for c in cols_to_fix:
        merged[c] = merged[c].astype(str)
        merged[c] = merged[c].replace([r'^\s*$', 'nan', 'None', 'NaN'], "unclear", regex=True)
        merged[c] = merged[c].fillna("unclear")

    # --- LAYOUT HELPER ---
    def setup_fixed_header_layout(fig, height_inches, title_text, legend_elements, custom_top_margin=3.0):
        TITLE_Y_OFFSET = 0.5     
        LEGEND_Y_OFFSET = 1.0    
        
        top_frac = 1.0 - (custom_top_margin / height_inches)
        bottom_frac = 1.0 / height_inches if height_inches > 10 else 0.1
        
        plt.subplots_adjust(top=top_frac, bottom=bottom_frac, left=0.15, right=0.95)
        
        # Fixed Title
        title_y = 1.0 - (TITLE_Y_OFFSET / height_inches)
        fig.suptitle(title_text, fontsize=18, weight='bold', y=title_y)
        
        # Fixed Legend
        legend_y = 1.0 - (LEGEND_Y_OFFSET / height_inches)
        fig.legend(handles=legend_elements, loc='upper center', 
                   bbox_to_anchor=(0.5, legend_y), ncol=3, frameon=False, fontsize=12)

    # --- VISUAL 1: ANNOTATED BAR CHART ---
    summary_data = []
    total_papers = len(merged) 
    for col in cols_to_fix:
        counts = merged[col].value_counts()
        yes = counts[counts.index.str.contains("yes|include|pass", case=False, regex=True)].sum()
        no = counts[counts.index.str.contains("no|exclude|fail", case=False, regex=True)].sum()
        unc = total_papers - (yes + no) 
        
        label = col.replace("_decision", "").replace("final_decision", "FINAL").upper()
        summary_data.append({"Criterion": label, "Pass": yes, "Fail": no, "Unclear": unc})
    
    df_sum = pd.DataFrame(summary_data).set_index("Criterion")[["Pass", "Fail", "Unclear"]]
    
    fig_height = 10
    fig, ax = plt.subplots(figsize=(12, fig_height))
    
    colors = [PALETTE["pass"], PALETTE["fail"], PALETTE["unclear"]]
    df_sum.plot(kind="bar", stacked=True, color=colors, edgecolor="black", width=0.7, ax=ax, legend=False)
    
    # Inner Bar Labels (Counts)
    for c in ax.containers:
        labels = [int(v) if v > 0 else "" for v in c.datavalues]
        ax.bar_label(c, labels=labels, label_type='center', fontsize=10, color='black', weight='bold')

    # Top Bar Labels (Pass Rate %)
    totals = df_sum.sum(axis=1)
    # We assume 'Pass' is the first column in df_sum
    pass_counts = df_sum["Pass"].values
    
    for i, total in enumerate(totals):
        p_count = pass_counts[i]
        p_rate = (p_count / total) * 100 if total > 0 else 0
        # Label: "XX.X%"
        ax.text(i, total + 2, f"{p_rate:.1f}%", ha='center', weight='bold', fontsize=11, color='black')

    legend_elements = [
        Patch(facecolor=PALETTE["pass"], edgecolor='gray', label='Pass'),
        Patch(facecolor=PALETTE["fail"], edgecolor='gray', label='Fail'),
        Patch(facecolor=PALETTE["unclear"], edgecolor='gray', label='Unclear')
    ]
    
    setup_fixed_header_layout(
        fig,
        fig_height,
        f"Screening Results Breakdown (N={total_papers})",
        legend_elements,
        custom_top_margin=2.0
    )

    plt.xticks(rotation=45, ha="right")

    ax.set_xlabel("Eligibility Criteria", labelpad=12)
    ax.set_ylabel("Number of Papers")

    plt.subplots_adjust(bottom=0.25)

    plt.savefig(os.path.join(viz_dir, "step6_1_criteria_summary.png"), dpi=300)
    plt.close()


    # --- HEATMAP GENERATOR ---
    def generate_long_heatmap(sub_df, title_text, filename):
        if sub_df.empty: return
        
        temp_scores = sub_df[crit_cols].applymap(_normalize_decision)
        sub_df = sub_df.copy()
        sub_df["_sort_score"] = temp_scores.sum(axis=1)
        sub_df = sub_df.sort_values(by=["_sort_score", "title"], ascending=[False, True])
        
        matrix = sub_df[crit_cols].applymap(_normalize_decision)
        matrix.columns = [c.replace("_decision", "").upper() for c in crit_cols]
        matrix.index = sub_df["title"].astype(str).apply(lambda x: x[:80] + "..." if len(x)>80 else x)
        
        n_rows = len(sub_df)
        fig_height = max(8, 4.0 + (n_rows * 0.35))
        
        fig = plt.figure(figsize=(12, fig_height))
        
        cmap = ListedColormap([PALETTE["fail"], PALETTE["unclear"], PALETTE["pass"]])
        ax = sns.heatmap(matrix, cmap=cmap, cbar=False, linewidths=0.5, linecolor='white', square=False, vmin=0, vmax=2)
        
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top') 
        plt.xticks(rotation=45, ha='left')
        
        legend_elements = [
            Patch(facecolor=PALETTE["pass"], edgecolor='gray', label='Pass'),
            Patch(facecolor=PALETTE["unclear"], edgecolor='gray', label='Unclear'),
            Patch(facecolor=PALETTE["fail"], edgecolor='gray', label='Fail')
        ]
        
        setup_fixed_header_layout(fig, fig_height, f"{title_text} ({len(sub_df)} Papers)", legend_elements, custom_top_margin=3.5)
        
        plt.savefig(os.path.join(viz_dir, filename), dpi=300)
        plt.close()
        print(f"   ✅ Saved Heatmap: {filename}")

    # Heatmaps
    df_inc = merged[merged["final_decision"].str.contains("include|yes|pass", case=False)].copy()
    generate_long_heatmap(df_inc, "INCLUDED Papers", "step6_2_heatmap_included.png")

    df_exc = merged[merged["final_decision"].str.contains("exclude|no|fail", case=False)].copy()
    generate_long_heatmap(df_exc, "EXCLUDED Papers", "step6_3_heatmap_excluded.png")
    
    df_unc = merged[~merged.index.isin(df_inc.index) & ~merged.index.isin(df_exc.index)].copy()
    generate_long_heatmap(df_unc, "UNCLEAR Papers", "step6_4_heatmap_unclear.png")

    # --- EXCEL REPORT ---
    print(f"   📊 Generating Styled Excel...")
    
    final_df = merged.drop(columns=["_sort_score"], errors="ignore").copy()
    
    # Sort
    final_df["_rank"] = final_df["final_decision"].apply(_get_sort_rank)
    final_df = final_df.sort_values(by=["_rank", "title"])
    final_df.drop(columns=["_rank"], inplace=True)

    # Reorder
    cols = list(final_df.columns)
    if "abstract" in cols: 
        cols.remove("abstract")
        cols.append("abstract")
    final_df = final_df[cols]

    with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
        final_df.to_excel(writer, sheet_name="All Papers", index=False)
        
        worksheet = writer.sheets["All Papers"]
        
        fill_pass = PatternFill(start_color=PALETTE["excel_pass"], end_color=PALETTE["excel_pass"], fill_type="solid")
        fill_fail = PatternFill(start_color=PALETTE["excel_fail"], end_color=PALETTE["excel_fail"], fill_type="solid")
        fill_unclear = PatternFill(start_color=PALETTE["excel_unclear"], end_color=PALETTE["excel_unclear"], fill_type="solid")
        
        align_single = Alignment(wrap_text=False, vertical='top') 
        
        for row in worksheet.iter_rows(min_row=2, max_row=len(final_df)+1, min_col=1, max_col=len(cols)):
            for cell in row:
                cell.alignment = align_single
                
                val = str(cell.value).lower().strip()
                if any(x in val for x in ["yes", "include", "pass"]):
                    cell.fill = fill_pass
                elif any(x in val for x in ["no", "exclude", "fail"]):
                    cell.fill = fill_fail
                elif any(x in val for x in ["unclear", "pending", "no_abstract"]):
                    cell.fill = fill_unclear
    
    print(f"   ✅ Saved Excel Report: {out_excel}")
    return {"status": "ok", "path": viz_dir}

if __name__ == "__main__":
    run_step6({})