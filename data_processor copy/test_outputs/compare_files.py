#!/usr/bin/env python3
"""
Compare two final_ratios CSV files to identify differences.
"""

import pandas as pd
import numpy as np

def compare_files():
    """Compare final_ratios.csv and final_ratios_20250712.csv"""
    
    print("=" * 80)
    print("Comparing final_ratios.csv and final_ratios_20250712.csv")
    print("=" * 80)
    
    # Load both files
    print("Loading files...")
    df1 = pd.read_csv('final_ratios.csv')
    df2 = pd.read_csv('final_ratios_20250712.csv')
    
    print(f"\nFile 1 (final_ratios.csv):")
    print(f"  Shape: {df1.shape}")
    print(f"  Columns: {list(df1.columns)}")
    print(f"  Memory usage: {df1.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"\nFile 2 (final_ratios_20250712.csv):")
    print(f"  Shape: {df2.shape}")
    print(f"  Columns: {list(df2.columns)}")
    print(f"  Memory usage: {df2.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Compare shapes
    print(f"\nShape comparison:")
    print(f"  Rows difference: {df1.shape[0] - df2.shape[0]}")
    print(f"  Columns difference: {df1.shape[1] - df2.shape[1]}")
    
    # Compare columns
    print(f"\nColumn comparison:")
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    
    if cols1 == cols2:
        print("  ✓ All columns are identical")
    else:
        print("  ✗ Columns differ:")
        only_in_1 = cols1 - cols2
        only_in_2 = cols2 - cols1
        if only_in_1:
            print(f"    Only in file 1: {list(only_in_1)}")
        if only_in_2:
            print(f"    Only in file 2: {list(only_in_2)}")
    
    # Compare data types
    print(f"\nData type comparison:")
    common_cols = list(cols1.intersection(cols2))
    dtype_diffs = []
    for col in common_cols:
        if df1[col].dtype != df2[col].dtype:
            dtype_diffs.append((col, df1[col].dtype, df2[col].dtype))
    
    if dtype_diffs:
        print("  ✗ Data types differ:")
        for col, dtype1, dtype2 in dtype_diffs:
            print(f"    {col}: {dtype1} vs {dtype2}")
    else:
        print("  ✓ All data types are identical")
    
    # Compare date ranges
    print(f"\nDate range comparison:")
    if 'date' in common_cols:
        df1['date'] = pd.to_datetime(df1['date'])
        df2['date'] = pd.to_datetime(df2['date'])
        
        print(f"  File 1 date range: {df1['date'].min()} to {df1['date'].max()}")
        print(f"  File 2 date range: {df2['date'].min()} to {df2['date'].max()}")
        
        date_diff = abs((df1['date'].max() - df1['date'].min()) - (df2['date'].max() - df2['date'].min()))
        print(f"  Date range difference: {date_diff.days} days")
    
    # Compare unique values
    print(f"\nUnique values comparison:")
    if 'gvkey' in common_cols:
        gvkey1 = set(df1['gvkey'].unique())
        gvkey2 = set(df2['gvkey'].unique())
        print(f"  File 1 unique gvkeys: {len(gvkey1)}")
        print(f"  File 2 unique gvkeys: {len(gvkey2)}")
        print(f"  gvkey difference: {len(gvkey1.symmetric_difference(gvkey2))}")
    
    if 'tic' in common_cols:
        tic1 = set(df1['tic'].unique())
        tic2 = set(df2['tic'].unique())
        print(f"  File 1 unique tickers: {len(tic1)}")
        print(f"  File 2 unique tickers: {len(tic2)}")
        print(f"  ticker difference: {len(tic1.symmetric_difference(tic2))}")
    
    # Compare numeric columns
    print(f"\nNumeric values comparison:")
    numeric_cols = []
    for col in common_cols:
        if pd.api.types.is_numeric_dtype(df1[col]) and pd.api.types.is_numeric_dtype(df2[col]):
            numeric_cols.append(col)
    
    if numeric_cols:
        print(f"  Comparing {len(numeric_cols)} numeric columns...")
        for col in numeric_cols[:5]:  # Show first 5 columns
            mean1 = df1[col].mean()
            mean2 = df2[col].mean()
            std1 = df1[col].std()
            std2 = df2[col].std()
            
            print(f"    {col}:")
            print(f"      Mean: {mean1:.6f} vs {mean2:.6f} (diff: {abs(mean1-mean2):.6f})")
            print(f"      Std:  {std1:.6f} vs {std2:.6f} (diff: {abs(std1-std2):.6f})")
    
    # Sample data comparison
    print(f"\nSample data comparison (first 3 rows):")
    print("File 1 (final_ratios.csv):")
    print(df1.head(3).to_string())
    print("\nFile 2 (final_ratios_20250712.csv):")
    print(df2.head(3).to_string())
    
    print("\n" + "=" * 80)
    print("Comparison completed!")
    print("=" * 80)

if __name__ == "__main__":
    compare_files() 