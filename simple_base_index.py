#!/usr/bin/env python3
"""
Simplified Base Index Data Downloader
"""

import yfinance as yf
import pandas as pd
import os

def main():
    print("Starting simplified base index downloader...")
    
    # Test SPX download
    print("Downloading SPX data...")
    try:
        ticker = yf.Ticker('^GSPC')
        data = ticker.history(start='2020-01-01', end='2020-12-31')
        
        if not data.empty:
            print(f"✅ Downloaded {len(data)} records for SPX")
            
            # Reset index and rename columns
            data = data.reset_index()
            data = data.rename(columns={
                'Date': 'date',
                'Close': 'close'
            })
            
            # Convert date to string
            data['date'] = data['date'].dt.strftime('%Y-%m-%d')
            
            # Select only required columns
            result = data[['date', 'close']].copy()
            
            # Save to CSV
            output_file = 'SPX_test.csv'
            result.to_csv(output_file, index=False)
            print(f"✅ Saved to {output_file}")
            print(f"Sample data:\n{result.head()}")
            
        else:
            print("❌ No data downloaded")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 