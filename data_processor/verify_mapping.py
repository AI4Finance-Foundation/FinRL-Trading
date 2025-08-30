#!/usr/bin/env python3
"""
verify_mapping.py

This script verifies the quality of the generated tic-gvkey mapping table.

Usage:
    python verify_mapping.py --mapping_file "tic_gvkey_mapping_2025.csv"
"""

import argparse
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_mapping(mapping_file: str) -> pd.DataFrame:
    """Load the mapping table."""
    logger.info(f"Loading mapping table from: {mapping_file}")
    df = pd.read_csv(mapping_file)
    logger.info(f"Loaded {len(df)} records")
    return df


def verify_mapping_quality(df: pd.DataFrame) -> None:
    """Verify the quality of the mapping table."""
    logger.info("=" * 50)
    logger.info("MAPPING TABLE QUALITY VERIFICATION")
    logger.info("=" * 50)
    
    # Basic statistics
    logger.info(f"Total records: {len(df)}")
    logger.info(f"Unique tickers: {df['tic'].nunique()}")
    logger.info(f"Unique gvkeys: {df['gvkey'].nunique()}")
    
    # Check for missing values
    missing_tic = df['tic'].isna().sum()
    missing_gvkey = df['gvkey'].isna().sum()
    missing_date = df['latest_date'].isna().sum()
    
    logger.info(f"Missing tic values: {missing_tic}")
    logger.info(f"Missing gvkey values: {missing_gvkey}")
    logger.info(f"Missing date values: {missing_date}")
    
    # Check for duplicates
    duplicate_tic = df['tic'].duplicated().sum()
    duplicate_gvkey = df['gvkey'].duplicated().sum()
    
    logger.info(f"Duplicate tickers: {duplicate_tic}")
    logger.info(f"Duplicate gvkeys: {duplicate_gvkey}")
    
    # Check date range
    df['latest_date'] = pd.to_datetime(df['latest_date'])
    logger.info(f"Date range: {df['latest_date'].min()} to {df['latest_date'].max()}")
    
    # Check for multiple gvkeys per ticker
    multiple_gvkeys = df[df['count_gvkeys'] > 1]
    logger.info(f"Tickers with multiple gvkeys: {len(multiple_gvkeys)}")
    
    if len(multiple_gvkeys) > 0:
        logger.info("Sample tickers with multiple gvkeys:")
        for _, row in multiple_gvkeys.head(5).iterrows():
            logger.info(f"  {row['tic']}: {row['count_gvkeys']} gvkeys, selected: {row['gvkey']}")
    
    # Check for duplicate gvkeys (different tickers with same gvkey)
    if duplicate_gvkey > 0:
        logger.info("Sample duplicate gvkeys:")
        duplicate_gvkey_df = df[df['gvkey'].duplicated(keep=False)].sort_values('gvkey')
        for gvkey in duplicate_gvkey_df['gvkey'].unique()[:5]:
            tics = duplicate_gvkey_df[duplicate_gvkey_df['gvkey'] == gvkey]['tic'].tolist()
            logger.info(f"  GVKEY {gvkey}: {tics}")
    
    # Summary
    logger.info("=" * 50)
    logger.info("QUALITY SUMMARY")
    logger.info("=" * 50)
    
    if missing_tic == 0 and missing_gvkey == 0 and duplicate_tic == 0:
        logger.info("✅ No missing values or duplicate tickers")
    else:
        logger.warning("⚠️ Found data quality issues")
    
    if duplicate_gvkey == 0:
        logger.info("✅ No duplicate gvkeys")
    else:
        logger.warning(f"⚠️ Found {duplicate_gvkey} duplicate gvkeys")
    
    if len(multiple_gvkeys) == 0:
        logger.info("✅ No tickers with multiple gvkeys")
    else:
        logger.info(f"ℹ️ Found {len(multiple_gvkeys)} tickers with multiple gvkeys (resolved by latest date)")


def show_sample_mappings(df: pd.DataFrame, n: int = 20) -> None:
    """Show sample mappings."""
    logger.info("=" * 50)
    logger.info(f"SAMPLE MAPPINGS (first {n} records)")
    logger.info("=" * 50)
    
    sample_df = df.head(n)
    for _, row in sample_df.iterrows():
        logger.info(f"{row['tic']:>6} -> {row['gvkey']:>8} (date: {row['latest_date']})")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Verify tic-gvkey mapping table quality')
    parser.add_argument('--mapping_file', type=str, default='tic_gvkey_mapping_2025.csv',
                       help='Path to the mapping CSV file')
    parser.add_argument('--sample_size', type=int, default=20,
                       help='Number of sample mappings to display')
    
    args = parser.parse_args()
    
    try:
        # Load mapping table
        df = load_mapping(args.mapping_file)
        
        # Verify quality
        verify_mapping_quality(df)
        
        # Show samples
        show_sample_mappings(df, args.sample_size)
        
        logger.info("=" * 50)
        logger.info("VERIFICATION COMPLETED")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Error during verification: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 