#!/usr/bin/env python3
"""
Step1_get_sp500_ticker.py

This script processes S&P 500 historical components data to extract unique stock tickers.
It reads a CSV file containing historical S&P 500 components and outputs a text file
with unique ticker symbols for use in WRDS queries.

Usage:
    python Step1_get_sp500_ticker.py --Stock_Index_His_file "S&P 500 Historical Components & Changes(08-12-2022).csv" --output_filename "sp500_tickers"

Author: FinRL Team
Date: 2024
"""

import argparse
import os
import pandas as pd
from datetime import datetime


def get_table(filename):
    """
    Read the CSV file containing S&P 500 historical components data.
    
    Args:
        filename (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: DataFrame with date as index and tickers column
    """
    if os.path.isfile(filename):
        df = pd.read_csv(filename, index_col='date')
        return df
    else:
        raise FileNotFoundError(f"File {filename} not found.")


def process_tickers(df):
    """
    Process the tickers data to extract unique stock symbols.
    
    Args:
        df (pandas.DataFrame): DataFrame containing tickers data
        
    Returns:
        list: List of unique ticker symbols
    """
    print(f"Processing data from {df.index[0]} to {df.index[-1]}")
    print(f"Total number of records: {df.shape[0]}")
    
    # Convert ticker column from csv to list, then sort
    df['tickers'] = df['tickers'].apply(lambda x: sorted(x.split(',')))
    
    # Replace SYMBOL-yyyymm with SYMBOL (remove date suffixes)
    df['tickers'] = [[ticker.split('-')[0] for ticker in tickers] for tickers in df['tickers']]
    
    # Merge all tickers together into a list
    tickers_list = []
    for i in range(df['tickers'].shape[0]):
        tickers_list = tickers_list + df['tickers'].iloc[i]
    
    print(f"Total number of ticker entries: {len(tickers_list)}")
    
    # Get unique tickers and sort them
    tickers_list_set = list(sorted(set(tickers_list)))
    
    print(f"Total number of unique tickers: {len(tickers_list_set)}")
    
    return tickers_list_set


def save_tickers_to_file(tickers_list, output_filename):
    """
    Save the unique tickers to a text file.
    
    Args:
        tickers_list (list): List of unique ticker symbols
        output_filename (str): Base name for the output file (without extension)
    """
    # Convert to DataFrame for easier saving
    tickers_df = pd.DataFrame(tickers_list)
    
    # Create output filename with .txt extension
    output_file = f"{output_filename}.txt"
    
    # Save to txt file for WRDS queries
    tickers_df.to_csv(output_file, header=None, index=None, sep=' ', mode='w')
    
    print(f"Unique tickers saved to: {output_file}")
    print(f"File contains {len(tickers_list)} unique ticker symbols")


def main():
    """
    Main function to process S&P 500 tickers data.
    """
    # Set up command line argument parser
    parser = argparse.ArgumentParser(
        description='Extract unique stock tickers from S&P 500 historical components data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python Step1_get_sp500_ticker.py --Stock_Index_His_file "S&P 500 Historical Components & Changes(08-12-2022).csv"
    python Step1_get_sp500_ticker.py --Stock_Index_His_file "data.csv" --output_filename "my_tickers"
        """
    )
    
    parser.add_argument(
        '--Stock_Index_His_file',
        type=str,
        required=True,
        help='Path to the S&P 500 historical components CSV file'
    )
    
    parser.add_argument(
        '--output_filename',
        type=str,
        default='sp500_tickers',
        help='Base name for the output file (default: sp500_tickers)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        print("=" * 60)
        print("S&P 500 Ticker Extraction Tool")
        print("=" * 60)
        print(f"Input file: {args.Stock_Index_His_file}")
        print(f"Output file: {args.output_filename}.txt")
        print("-" * 60)
        
        # Read the data
        print("Reading input file...")
        df = get_table(args.Stock_Index_His_file)
        print(f"Successfully loaded data with shape: {df.shape}")
        
        # Process the tickers
        print("\nProcessing tickers...")
        unique_tickers = process_tickers(df)
        
        # Save to file
        print("\nSaving results...")
        save_tickers_to_file(unique_tickers, args.output_filename)
        
        print("\n" + "=" * 60)
        print("Processing completed successfully!")
        print("=" * 60)
        
        # Display some sample tickers
        print(f"\nSample tickers (first 10): {unique_tickers[:10]}")
        print(f"Sample tickers (last 10): {unique_tickers[-10:]}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 