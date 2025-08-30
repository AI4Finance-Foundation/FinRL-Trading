#!/usr/bin/env python3
"""
Detailed comparison of two final_ratios CSV files.
"""

import pandas as pd
import numpy as np

def detailed_compare():
    """Detailed comparison of final_ratios.csv and final_ratios_20250712.csv"""
    
    print("=" * 80)
    print("DETAILED COMPARISON: final_ratios.csv vs final_ratios_20250712.csv")
    print("=" * 80)
    
    # Load both files
    print("Loading files...")
    df1 = pd.read_csv('final_ratios.csv')
    df2 = pd.read_csv('final_ratios_20250712.csv')
    
    print(f"\n1. BASIC FILE INFORMATION:")
    print(f"   File 1 (final_ratios.csv):")
    print(f"     Shape: {df1.shape}")
    print(f"     Columns: {list(df1.columns)}")
    print(f"     Memory usage: {df1.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"\n   File 2 (final_ratios_20250712.csv):")
    print(f"     Shape: {df2.shape}")
    print(f"     Columns: {list(df2.columns)}")
    print(f"     Memory usage: {df2.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Compare shapes
    print(f"\n2. SHAPE COMPARISON:")
    print(f"   Rows: {df1.shape[0]} vs {df2.shape[0]} (diff: {df1.shape[0] - df2.shape[0]})")
    print(f"   Columns: {df1.shape[1]} vs {df2.shape[1]} (diff: {df1.shape[1] - df2.shape[1]})")
    
    # Compare columns
    print(f"\n3. COLUMN COMPARISON:")
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    
    if cols1 == cols2:
        print("   ✓ All columns are identical")
    else:
        print("   ✗ Columns differ:")
        only_in_1 = cols1 - cols2
        only_in_2 = cols2 - cols1
        if only_in_1:
            print(f"     Only in file 1: {list(only_in_1)}")
        if only_in_2:
            print(f"     Only in file 2: {list(only_in_2)}")
    
    # Remove index column if present
    if 'Unnamed: 0' in df2.columns:
        df2 = df2.drop('Unnamed: 0', axis=1)
        print("   Note: Removed 'Unnamed: 0' index column from file 2 for comparison")
    
    # Compare data types
    print(f"\n4. DATA TYPE COMPARISON:")
    common_cols = list(set(df1.columns).intersection(set(df2.columns)))
    dtype_diffs = []
    for col in common_cols:
        if df1[col].dtype != df2[col].dtype:
            dtype_diffs.append((col, df1[col].dtype, df2[col].dtype))
    
    if dtype_diffs:
        print("   ✗ Data types differ:")
        for col, dtype1, dtype2 in dtype_diffs:
            print(f"     {col}: {dtype1} vs {dtype2}")
    else:
        print("   ✓ All data types are identical")
    
    # Compare date ranges
    print(f"\n5. DATE RANGE COMPARISON:")
    if 'date' in common_cols:
        df1['date'] = pd.to_datetime(df1['date'])
        df2['date'] = pd.to_datetime(df2['date'])
        
        print(f"   File 1 date range: {df1['date'].min()} to {df1['date'].max()}")
        print(f"   File 2 date range: {df2['date'].min()} to {df2['date'].max()}")
        
        date_diff = abs((df1['date'].max() - df1['date'].min()) - (df2['date'].max() - df2['date'].min()))
        print(f"   Date range difference: {date_diff.days} days")
    
    # Compare unique values
    print(f"\n6. UNIQUE VALUES COMPARISON:")
    if 'gvkey' in common_cols:
        gvkey1 = set(df1['gvkey'].unique())
        gvkey2 = set(df2['gvkey'].unique())
        print(f"   File 1 unique gvkeys: {len(gvkey1)}")
        print(f"   File 2 unique gvkeys: {len(gvkey2)}")
        print(f"   gvkey difference: {len(gvkey1.symmetric_difference(gvkey2))}")
        
        if len(gvkey1.symmetric_difference(gvkey2)) > 0:
            only_in_1 = gvkey1 - gvkey2
            only_in_2 = gvkey2 - gvkey1
            if only_in_1:
                print(f"     Only in file 1: {list(only_in_1)[:5]}...")  # Show first 5
            if only_in_2:
                print(f"     Only in file 2: {list(only_in_2)[:5]}...")  # Show first 5
    
    if 'tic' in common_cols:
        tic1 = set(df1['tic'].unique())
        tic2 = set(df2['tic'].unique())
        print(f"   File 1 unique tickers: {len(tic1)}")
        print(f"   File 2 unique tickers: {len(tic2)}")
        print(f"   ticker difference: {len(tic1.symmetric_difference(tic2))}")
    
    if 'gsector' in common_cols:
        sector1 = set(df1['gsector'].unique())
        sector2 = set(df2['gsector'].unique())
        print(f"   File 1 unique sectors: {len(sector1)} - {sorted(list(sector1))}")
        print(f"   File 2 unique sectors: {len(sector2)} - {sorted(list(sector2))}")
    
    # Compare numeric columns
    print(f"\n7. NUMERIC VALUES COMPARISON:")
    numeric_cols = []
    for col in common_cols:
        if pd.api.types.is_numeric_dtype(df1[col]) and pd.api.types.is_numeric_dtype(df2[col]):
            numeric_cols.append(col)
    
    if numeric_cols:
        print(f"   Comparing {len(numeric_cols)} numeric columns...")
        print(f"   Columns: {numeric_cols}")
        
        for col in numeric_cols:
            mean1 = df1[col].mean()
            mean2 = df2[col].mean()
            std1 = df1[col].std()
            std2 = df2[col].std()
            min1 = df1[col].min()
            min2 = df2[col].min()
            max1 = df1[col].max()
            max2 = df2[col].max()
            
            print(f"\n     {col}:")
            print(f"       Mean: {mean1:.6f} vs {mean2:.6f} (diff: {abs(mean1-mean2):.6f})")
            print(f"       Std:  {std1:.6f} vs {std2:.6f} (diff: {abs(std1-std2):.6f})")
            print(f"       Min:  {min1:.6f} vs {min2:.6f} (diff: {abs(min1-min2):.6f})")
            print(f"       Max:  {max1:.6f} vs {max2:.6f} (diff: {abs(max1-max2):.6f})")
            
            # Check if values are identical
            if np.allclose(df1[col], df2[col], equal_nan=True):
                print(f"       ✓ Values are identical")
            else:
                print(f"       ✗ Values differ")
                # Count differences
                diff_count = np.sum(~np.isclose(df1[col], df2[col], equal_nan=True))
                print(f"       Different values: {diff_count} out of {len(df1)} ({diff_count/len(df1)*100:.2f}%)")
    
    # Sample data comparison
    print(f"\n8. SAMPLE DATA COMPARISON:")
    print("   File 1 (final_ratios.csv) - First 3 rows:")
    print(df1.head(3).to_string())
    print("\n   File 2 (final_ratios_20250712.csv) - First 3 rows:")
    print(df2.head(3).to_string())
    
    # Check for exact equality
    print(f"\n9. EXACT EQUALITY CHECK:")
    if df1.shape == df2.shape:
        # Reset index for comparison
        df1_reset = df1.reset_index(drop=True)
        df2_reset = df2.reset_index(drop=True)
        
        if df1_reset.equals(df2_reset):
            print("   ✓ Files are exactly identical (after removing index column)")
        else:
            print("   ✗ Files are not identical")
            # Find first difference
            for i in range(len(df1_reset)):
                if not df1_reset.iloc[i].equals(df2_reset.iloc[i]):
                    print(f"   First difference at row {i}:")
                    print(f"     File 1: {df1_reset.iloc[i].to_dict()}")
                    print(f"     File 2: {df2_reset.iloc[i].to_dict()}")
                    break
    else:
        print("   ✗ Files have different shapes, cannot be identical")
    
    print("\n" + "=" * 80)
    print("DETAILED COMPARISON COMPLETED!")
    print("=" * 80)

if __name__ == "__main__":
    detailed_compare() 