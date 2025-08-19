#!/usr/bin/env python3
"""
Base Index Data Downloader

This script downloads daily index data for SPX (S&P 500) and QQQ (Nasdaq-100 ETF)
from Yahoo Finance API using yfinance library.


"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings("ignore")

def download_index_data(symbol, start_date, end_date=None, output_dir="./"):
    """
    Download daily index data from Yahoo Finance
    
    Parameters:
    -----------
    symbol : str
        Stock symbol (e.g., '^GSPC' for SPX, 'QQQ' for Nasdaq-100 ETF)
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str, optional
        End date in 'YYYY-MM-DD' format (default: today)
    output_dir : str
        Output directory for CSV files
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with daily index data
    """
    
    # Set end date to today if not provided
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Downloading {symbol} data from {start_date} to {end_date}...")
    
    try:
        # Download data using yfinance
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        
        if data.empty:
            print(f"Warning: No data found for {symbol}")
            return None
            
        # Reset index to make date a column
        data = data.reset_index()
        
        # Rename columns to match expected format
        data = data.rename(columns={
            'Date': 'date',
            'Close': 'close',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Volume': 'volume'
        })
        
        # Convert date to string format
        data['date'] = data['date'].dt.strftime('%Y-%m-%d')
        
        # Select only required columns for backtesting
        result = data[['date', 'close']].copy()
        
        # Remove any rows with NaN values
        result = result.dropna()
        
        print(f"Successfully downloaded {len(result)} records for {symbol}")
        
        return result
        
    except Exception as e:
        print(f"Error downloading data for {symbol}: {str(e)}")
        return None

def download_spx_data(start_date="2000-01-01", end_date=None, output_dir="./"):
    """
    Download S&P 500 index data
    
    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str, optional
        End date in 'YYYY-MM-DD' format
    output_dir : str
        Output directory for CSV files
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with SPX daily data
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Download SPX data (^GSPC is the Yahoo Finance symbol for S&P 500)
    spx_data = download_index_data('^GSPC', start_date, end_date, output_dir)
    
    if spx_data is not None:
        # Save to CSV file
        output_file = os.path.join(output_dir, 'SPX.csv')
        spx_data.to_csv(output_file, index=False)
        print(f"SPX data saved to {output_file}")
        
        # Display summary statistics
        print("\nSPX Data Summary:")
        print(f"Date range: {spx_data['date'].min()} to {spx_data['date'].max()}")
        print(f"Total records: {len(spx_data)}")
        print(f"Price range: ${spx_data['close'].min():.2f} to ${spx_data['close'].max():.2f}")
        print(f"Average price: ${spx_data['close'].mean():.2f}")
        
    return spx_data

def download_qqq_data(start_date="2000-01-01", end_date=None, output_dir="./"):
    """
    Download Nasdaq-100 ETF (QQQ) data
    
    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str, optional
        End date in 'YYYY-MM-DD' format
    output_dir : str
        Output directory for CSV files
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with QQQ daily data
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Download QQQ data
    qqq_data = download_index_data('QQQ', start_date, end_date, output_dir)
    
    if qqq_data is not None:
        # Save to CSV file
        output_file = os.path.join(output_dir, 'QQQ.csv')
        qqq_data.to_csv(output_file, index=False)
        print(f"QQQ data saved to {output_file}")
        
        # Display summary statistics
        print("\nQQQ Data Summary:")
        print(f"Date range: {qqq_data['date'].min()} to {qqq_data['date'].max()}")
        print(f"Total records: {len(qqq_data)}")
        print(f"Price range: ${qqq_data['close'].min():.2f} to ${qqq_data['close'].max():.2f}")
        print(f"Average price: ${qqq_data['close'].mean():.2f}")
        
    return qqq_data

def download_all_indices(start_date="2000-01-01", end_date=None, output_dir="./"):
    """
    Download both SPX and QQQ data
    
    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str, optional
        End date in 'YYYY-MM-DD' format
    output_dir : str
        Output directory for CSV files
        
    Returns:
    --------
    tuple
        (spx_data, qqq_data) DataFrames
    """
    
    print("=" * 60)
    print("DOWNLOADING BASE INDEX DATA")
    print("=" * 60)
    
    # Download SPX data
    spx_data = download_spx_data(start_date, end_date, output_dir)
    
    print("\n" + "-" * 40)
    
    # Download QQQ data
    qqq_data = download_qqq_data(start_date, end_date, output_dir)
    
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETED")
    print("=" * 60)
    
    return spx_data, qqq_data

def verify_data_compatibility(spx_data, qqq_data):
    """
    Verify that downloaded data is compatible with backtesting requirements
    
    Parameters:
    -----------
    spx_data : pd.DataFrame
        SPX data DataFrame
    qqq_data : pd.DataFrame
        QQQ data DataFrame
        
    Returns:
    --------
    bool
        True if data is compatible, False otherwise
    """
    
    print("\nVerifying data compatibility...")
    
    # Check required columns
    required_columns = ['date', 'close']
    
    for name, data in [('SPX', spx_data), ('QQQ', qqq_data)]:
        if data is None:
            print(f"❌ {name} data is None")
            return False
            
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            print(f"❌ {name} missing columns: {missing_cols}")
            return False
            
        print(f"✅ {name} has required columns: {list(data.columns)}")
    
    # Check data types
    for name, data in [('SPX', spx_data), ('QQQ', qqq_data)]:
        if data['date'].dtype != 'object':  # Should be string
            print(f"⚠️  {name} date column is not string type")
        if not pd.api.types.is_numeric_dtype(data['close']):
            print(f"❌ {name} close column is not numeric")
            return False
            
        print(f"✅ {name} data types are correct")
    
    # Check for missing values
    for name, data in [('SPX', spx_data), ('QQQ', qqq_data)]:
        missing_count = data.isnull().sum().sum()
        if missing_count > 0:
            print(f"⚠️  {name} has {missing_count} missing values")
        else:
            print(f"✅ {name} has no missing values")
    
    print("✅ Data compatibility verification completed")
    return True

def main():
    """
    Main function to download and verify index data
    """
    
    # Set parameters
    start_date = "2000-01-01"
    output_dir = "./output/"  # Output to current directory
    
    print("Base Index Data Downloader")
    print("=" * 40)
    print(f"Start Date: {start_date}")
    print(f"Output Directory: {output_dir}")
    print()
    
    # Download all indices
    spx_data, qqq_data = download_all_indices(start_date, output_dir=output_dir)
    
    # Verify data compatibility
    if spx_data is not None and qqq_data is not None:
        verify_data_compatibility(spx_data, qqq_data)
        
        print("\n" + "=" * 60)
        print("FILES CREATED:")
        print("=" * 60)
        print(f"✅ SPX.csv - S&P 500 index data ({len(spx_data)} records)")
        print(f"✅ QQQ.csv - Nasdaq-100 ETF data ({len(qqq_data)} records)")
        print("\nThese files are ready for use in fundamental_back_testing.ipynb")
        
    else:
        print("\n❌ Failed to download some data. Please check your internet connection and try again.")

if __name__ == "__main__":
    main()