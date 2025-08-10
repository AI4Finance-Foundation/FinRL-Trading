#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stock Selection Script
Convert Jupyter notebook stock selection functionality to command line script
"""

import pandas as pd
import time
import os
import argparse
import sys
from pathlib import Path


def create_directory_if_not_exists(directory_path):
    """
    Create directory if it doesn't exist
    
    Args:
        directory_path (str): Directory path
    """
    path = Path(directory_path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory_path}")
    else:
        print(f"✓ Directory already exists: {directory_path}")


def run_stock_selection(data_path, output_path):
    """
    Run stock selection model
    
    Args:
        data_path (str): Input sector files directory
        output_path (str): Output directory
    """
    # Define sector range
    sectors = range(10, 65, 5)
    
    # Set data directory path
    DATA_DIR = data_path
    FUNDAMENTAL_FILE = os.path.join(DATA_DIR, "final_ratios.csv")
    
    print(f"Using data directory: {DATA_DIR}")
    print(f"Fundamental data file: {FUNDAMENTAL_FILE}")
    print(f"Output directory: {output_path}")
    print(f"Sector range: {list(sectors)}")
    
    # Check if input file exists
    if not os.path.exists(FUNDAMENTAL_FILE):
        print(f"Error: Fundamental data file does not exist: {FUNDAMENTAL_FILE}")
        sys.exit(1)
    
    # Create output directory
    create_directory_if_not_exists(output_path)
    
    # gvkey is unique identifier
    df_dict = {'gvkey': [], 'predicted_return': [], 'trade_date': []}
    
    # ===== Run stock selection for all sectors in my_outputs directory =====
    start = time.time()
    print("\nStarting stock selection model for all sectors...")
    
    for sector in sectors:
        sector_file = os.path.join(DATA_DIR, f"sector{sector}.xlsx")
        print(f"\nProcessing sector{sector}...")
        
        # Check if sector file exists
        if not os.path.exists(sector_file):
            print(f"   Warning: Sector file does not exist, skipping: {sector_file}")
            continue
        
        # Run model training - using files from specified directory
        cmd = f"python fundamental_run_model.py -sector_name sector{sector} -tic_column gvkey -fundamental {FUNDAMENTAL_FILE} -sector {sector_file}"
        print(f"Executing command: {cmd}")
        
        result = os.system(cmd)
        if result != 0:
            print(f" sector{sector} model training failed")
            continue
        
        # Read prediction results
        result_file = f"results/sector{sector}/df_predict_best.csv"
        if not os.path.exists(result_file):
            print(f" Prediction result file does not exist: {result_file}")
            continue
            
        try:
            df = pd.read_csv(result_file, index_col=0)
            print(f"  Reading prediction results: {df.shape[0]} dates, {df.shape[1]} stocks")
            
            for idx in df.index:
                predicted_return = df.loc[idx]
                top_q = predicted_return.quantile(0.75)
                predicted_return = predicted_return[predicted_return >= top_q]
                for gvkey in predicted_return.index:
                    df_dict["gvkey"].append(gvkey)
                    df_dict["predicted_return"].append(predicted_return[gvkey])
                    df_dict["trade_date"].append(idx)
            
            print(f" sector{sector} processing completed")
            
        except Exception as e:
            print(f"Error processing sector{sector}: {str(e)}")
            continue
    
    end = time.time()
    
    print(f"\nTotal time: {(end-start)/60:.2f} minutes")
    print(f"Processing completed! Total records: {len(df_dict['gvkey'])}")
    
    # Create result DataFrame
    df_result = pd.DataFrame(df_dict)
    
    # Save results to CSV file
    output_file = os.path.join(output_path, "stock_selected.csv")
    df_result.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
    
    # Display result statistics
    if len(df_result) > 0:
        print(f"\nResult statistics:")
        print(f"  Total records: {len(df_result)}")
        print(f"  Unique stocks: {df_result['gvkey'].nunique()}")
        print(f"  Date range: {df_result['trade_date'].min()} to {df_result['trade_date'].max()}")
        print(f"  Predicted return range: {df_result['predicted_return'].min():.4f} to {df_result['predicted_return'].max():.4f}")
    else:
        print("\nWarning: No stock selection results generated")


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(
        description="Stock Selection Script - Run stock selection model for all sectors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python stock_selection.py --data_path "./data_processor/my_outputs" --output_path "./result"
  python stock_selection.py --data_path "./data_processor/my_outputs"
  python stock_selection.py --help
        """
    )
    
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Input sector files directory path (e.g., "./data_processor/my_outputs")'
    )
    
    parser.add_argument(
        '--output_path',
        type=str,
        default='./result',
        help='Output directory path (default: "./result")'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Stock Selection Script")
    print("=" * 60)
    
    # Run stock selection
    run_stock_selection(args.data_path, args.output_path)
    
    print("\n" + "=" * 60)
    print("Processing completed!")
    print("=" * 60)


if __name__ == "__main__":
    main() 