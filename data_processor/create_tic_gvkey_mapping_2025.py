#!/usr/bin/env python3
"""
create_tic_gvkey_mapping_2025.py

This script creates a mapping table between ticker symbols (tic) and gvkey identifiers
from the daily price data file. For cases where multiple gvkeys correspond to the same
ticker, the most recent one (based on datadate) is selected.

Usage:
    python create_tic_gvkey_mapping_2025.py --price_file "sp500_tickers_daily_price_20250712.csv"

Author: FinRL Team
Date: 2025
"""

import argparse
import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_price_data(price_file: str) -> pd.DataFrame:
    """
    Load price data from CSV file.
    
    Args:
        price_file (str): Path to the price data CSV file
        
    Returns:
        pd.DataFrame: DataFrame with price data
    """
    logger.info(f"Loading price data from: {price_file}")
    
    if not os.path.isfile(price_file):
        raise FileNotFoundError(f"Price file {price_file} not found.")
    
    # Load only necessary columns to save memory
    required_columns = ['tic', 'datadate', 'gvkey']
    
    try:
        # First, let's check what columns are available
        sample_df = pd.read_csv(price_file, nrows=5)
        logger.info(f"Available columns: {list(sample_df.columns)}")
        
        # Load the data with required columns
        df = pd.read_csv(price_file, usecols=required_columns)
        
        logger.info(f"Successfully loaded price data with shape: {df.shape}")
        logger.info(f"Date range: {df['datadate'].min()} to {df['datadate'].max()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading price data: {e}")
        raise


def filter_2025_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter data to include only 2025 records.
    
    Args:
        df (pd.DataFrame): Price data DataFrame
        
    Returns:
        pd.DataFrame: Filtered DataFrame with only 2025 data
    """
    logger.info("Filtering data for 2025...")
    
    # Convert datadate to datetime
    df['datadate'] = pd.to_datetime(df['datadate'])
    
    # Filter for 2025 data
    df_2025 = df[df['datadate'].dt.year == 2025].copy()
    
    logger.info(f"2025 data shape: {df_2025.shape}")
    logger.info(f"2025 date range: {df_2025['datadate'].min()} to {df_2025['datadate'].max()}")
    
    if df_2025.empty:
        logger.warning("No 2025 data found. Using all available data instead.")
        return df
    
    return df_2025


def create_mapping_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create mapping table between tic and gvkey.
    For multiple gvkeys per tic, select the most recent one.
    
    Args:
        df (pd.DataFrame): Price data DataFrame
        
    Returns:
        pd.DataFrame: Mapping table with columns [tic, gvkey, latest_date, count_gvkeys]
    """
    logger.info("Creating mapping table...")
    
    # Group by tic and find the most recent record for each
    mapping_data = []
    
    for tic, group in df.groupby('tic'):
        # Sort by datadate to get the most recent record
        group_sorted = group.sort_values('datadate', ascending=False)
        
        # Get the most recent record
        latest_record = group_sorted.iloc[0]
        
        # Count unique gvkeys for this tic
        unique_gvkeys = group['gvkey'].nunique()
        
        mapping_data.append({
            'tic': tic,
            'gvkey': latest_record['gvkey'],
            'latest_date': latest_record['datadate'],
            'count_gvkeys': unique_gvkeys
        })
    
    # Create DataFrame
    mapping_df = pd.DataFrame(mapping_data)
    
    # Sort by tic for better readability
    mapping_df = mapping_df.sort_values('tic').reset_index(drop=True)
    
    logger.info(f"Created mapping table with {len(mapping_df)} unique tickers")
    
    # Log statistics
    multiple_gvkeys = mapping_df[mapping_df['count_gvkeys'] > 1]
    if not multiple_gvkeys.empty:
        logger.info(f"Found {len(multiple_gvkeys)} tickers with multiple gvkeys:")
        for _, row in multiple_gvkeys.head(10).iterrows():
            logger.info(f"  {row['tic']}: {row['count_gvkeys']} gvkeys, selected gvkey: {row['gvkey']} (date: {row['latest_date']})")
    
    return mapping_df


def validate_mapping(mapping_df: pd.DataFrame) -> None:
    """
    Validate the mapping table for data quality.
    
    Args:
        mapping_df (pd.DataFrame): Mapping table DataFrame
    """
    logger.info("Validating mapping table...")
    
    # Check for missing values
    missing_tic = mapping_df['tic'].isna().sum()
    missing_gvkey = mapping_df['gvkey'].isna().sum()
    
    if missing_tic > 0:
        logger.warning(f"Found {missing_tic} records with missing tic")
    
    if missing_gvkey > 0:
        logger.warning(f"Found {missing_gvkey} records with missing gvkey")
    
    # Check for duplicate tics
    duplicate_tics = mapping_df['tic'].duplicated().sum()
    if duplicate_tics > 0:
        logger.error(f"Found {duplicate_tics} duplicate tic entries")
    else:
        logger.info("No duplicate tic entries found")
    
    # Check for duplicate gvkeys
    duplicate_gvkeys = mapping_df['gvkey'].duplicated().sum()
    if duplicate_gvkeys > 0:
        logger.warning(f"Found {duplicate_gvkeys} duplicate gvkey entries (this may be normal)")
    
    # Summary statistics
    logger.info(f"Mapping table summary:")
    logger.info(f"  Total unique tickers: {len(mapping_df)}")
    logger.info(f"  Total unique gvkeys: {mapping_df['gvkey'].nunique()}")
    logger.info(f"  Tickers with multiple gvkeys: {(mapping_df['count_gvkeys'] > 1).sum()}")


def save_mapping_table(mapping_df: pd.DataFrame, output_file: str = "tic_gvkey_mapping_2025.csv") -> str:
    """
    Save the mapping table to CSV file.
    
    Args:
        mapping_df (pd.DataFrame): Mapping table DataFrame
        output_file (str): Output filename
        
    Returns:
        str: Path to the saved file
    """
    logger.info(f"Saving mapping table to: {output_file}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save to CSV
    mapping_df.to_csv(output_file, index=False)
    
    logger.info(f"Successfully saved mapping table with {len(mapping_df)} records")
    
    return output_file


def main():
    """
    Main function to create the mapping table.
    """
    parser = argparse.ArgumentParser(
        description='Create gvkey-ticker mapping table from price data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python create_tic_gvkey_mapping_2025.py --price_file "sp500_tickers_daily_price_20250712.csv"
    python create_tic_gvkey_mapping_2025.py --price_file "data.csv" --output "my_mapping.csv"
        """
    )
    
    parser.add_argument(
        '--price_file',
        type=str,
        required=True,
        help='Path to the price data CSV file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='tic_gvkey_mapping_2025.csv',
        help='Output filename for the mapping table (default: tic_gvkey_mapping_2025.csv)'
    )
    
    parser.add_argument(
        '--use_all_data',
        action='store_true',
        help='Use all available data instead of filtering for 2025 only'
    )
    
    args = parser.parse_args()
    
    try:
        logger.info("=" * 60)
        logger.info("GVKEY-Ticker Mapping Table Creator")
        logger.info("=" * 60)
        logger.info(f"Price file: {args.price_file}")
        logger.info(f"Output file: {args.output}")
        logger.info(f"Use all data: {args.use_all_data}")
        logger.info("-" * 60)
        
        # Load price data
        df = load_price_data(args.price_file)
        
        # Filter for 2025 data (unless use_all_data is specified)
        if not args.use_all_data:
            df = filter_2025_data(df)
        
        # Create mapping table
        mapping_df = create_mapping_table(df)
        
        # Validate mapping
        validate_mapping(mapping_df)
        
        # Save mapping table
        output_path = save_mapping_table(mapping_df, args.output)
        
        logger.info("=" * 60)
        logger.info("Mapping table creation completed successfully!")
        logger.info(f"Output saved to: {output_path}")
        logger.info("=" * 60)
        
        # Display sample of the mapping
        logger.info("\nSample mapping (first 10 records):")
        print(mapping_df.head(10).to_string(index=False))
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 