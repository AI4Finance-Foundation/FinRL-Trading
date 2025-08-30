#!/usr/bin/env python3
"""
check_duplicate_gvkeys.py

This script checks for duplicate gvkeys in the mapping table and shows details.
"""

import pandas as pd

def main():
    # Load mapping table
    df = pd.read_csv('tic_gvkey_mapping_2025.csv')
    
    print("=" * 60)
    print("DUPLICATE GVKEY ANALYSIS")
    print("=" * 60)
    
    # Find duplicate gvkeys
    duplicate_gvkeys = df[df['gvkey'].duplicated(keep=False)].sort_values('gvkey')
    
    print(f"Total records: {len(df)}")
    print(f"Unique gvkeys: {df['gvkey'].nunique()}")
    print(f"Duplicate gvkeys found: {len(duplicate_gvkeys)}")
    print()
    
    if len(duplicate_gvkeys) > 0:
        print("Duplicate gvkey details:")
        print("-" * 60)
        
        for gvkey in duplicate_gvkeys['gvkey'].unique():
            records = duplicate_gvkeys[duplicate_gvkeys['gvkey'] == gvkey]
            print(f"GVKEY: {gvkey}")
            for _, record in records.iterrows():
                print(f"  TIC: {record['tic']}, Date: {record['latest_date']}")
            print()
    
    print("=" * 60)
    print("ANALYSIS COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    main() 