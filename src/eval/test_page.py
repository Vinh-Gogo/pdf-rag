"""
Quick Summary Script
Xem nhanh kết quả từ comprehensive_results.csv
"""

import pandas as pd
import sys

def quick_summary():
    """Print quick summary of analysis results"""
    
    try:
        df = pd.read_csv('src/analysis/comprehensive_results.csv')
    except FileNotFoundError:
        print("❌ File not found! Run comprehensive_comparison.py first.")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("📊 QUICK SUMMARY - COMPREHENSIVE FILE COMPARISON")
    print("="*80)
    
    # Basic stats
    print(f"\n📁 Total Files: {len(df)}")
    print(f"✅ Perfect Match (0 diff): {len(df[df['total_diff'] == 0])} ({len(df[df['total_diff'] == 0])/len(df)*100:.1f}%)")
    print(f"⚠️  With Differences: {len(df[df['total_diff'] > 0])} ({len(df[df['total_diff'] > 0])/len(df)*100:.1f}%)")
    
    # Word differences
    print(f"\n📝 Word Differences:")
    print(f"   Average: {df['total_diff'].mean():.1f} words")
    print(f"   Median:  {df['total_diff'].median():.1f} words")
    print(f"   Max:     {df['total_diff'].max():.0f} words (Page {df.loc[df['total_diff'].idxmax(), 'page']})")
    
    # Similarity stats
    print(f"\n🎯 Cosine Similarity:")
    for model in ['DangVanTuan', 'Qwen', 'Halong']:
        col = f'sim_{model}'
        if col in df.columns:
            mean = df[col].mean()
            min_val = df[col].min()
            min_page = df.loc[df[col].idxmin(), 'page']
            print(f"   {model:<15} Mean: {mean:.4f}  Min: {min_val:.4f} (Page {min_page})")
    
    # Top 20 issues - sorted by Word Diff
    print(f"\n⚠️  Top 20 Problematic Pages (sorted by Word Diff):")
    print(f"{'Page':<8} {'Word Diff':<12} {'DangVanTuan':<13} {'Qwen':<13} {'Halong':<13}")
    print("-" * 65)
    
    # Sort by total_diff (word differences)
    df_temp = df.copy()
    top_20 = df_temp.nlargest(50, 'total_diff')
    
    for _, row in top_20.iterrows():
        sim_dang = row.get('sim_DangVanTuan', 0)
        sim_qwen = row.get('sim_Qwen', 0)
        sim_halong = row.get('sim_Halong', 0)
        print(f"{row['page']:<8.0f} {row['total_diff']:<12.0f} {sim_dang:<13.4f} {sim_qwen:<13.4f} {sim_halong:<13.4f}")
    
    # Top 5 perfect - sorted by lowest word diff and highest similarity
    print(f"\n✅ Top 5 Perfect Pages (sorted by lowest Word Diff):")
    print(f"{'Page':<8} {'Word Diff':<12} {'DangVanTuan':<13} {'Qwen':<13} {'Halong':<13}")
    print("-" * 65)
    
    # Sort by total_diff ascending (lowest differences first)
    top_5_perfect = df_temp.nsmallest(5, 'total_diff')
    
    for _, row in top_5_perfect.iterrows():
        sim_dang = row.get('sim_DangVanTuan', 0)
        sim_qwen = row.get('sim_Qwen', 0)
        sim_halong = row.get('sim_Halong', 0)
        print(f"{row['page']:<8.0f} {row['total_diff']:<12.0f} {sim_dang:<13.4f} {sim_qwen:<13.4f} {sim_halong:<13.4f}")
    
    print("\n" + "="*80)
    print("📈 View full visualization: src/analysis/comprehensive_analysis.png")
    print("📊 Full data: src/analysis/comprehensive_results.csv")
    print("="*80 + "\n")


if __name__ == "__main__":
    quick_summary()
