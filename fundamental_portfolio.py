#!/usr/bin/env python3
"""
Fundamental Portfolio Optimization Script

This script implements portfolio optimization using fundamental analysis and machine learning predictions.
It reads stock price data and selected stock predictions, then performs portfolio optimization using PyPortfolioOpt.

Author: AI Assistant
Date: 2024
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import argparse
import os
import time
import pickle
from datetime import datetime
from pandas.tseries.offsets import BDay

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt import expected_returns
from pypfopt import objective_functions


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Fundamental Portfolio Optimization using PyPortfolioOpt',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fundamental_portfolio.py --stocks_price data_processor/sp500_tickers_daily_price_20250712.csv --stock_selected result/stock_selected.csv
  python fundamental_portfolio.py --stocks_price data_processor/sp500_tickers_daily_price_20250712.csv --stock_selected result/stock_selected.csv --output_dir ./my_output
        """
    )
    
    parser.add_argument(
        '--stocks_price', 
        type=str, 
        required=True,
        help='Path to stock price data CSV file (e.g., sp500_tickers_daily_price_20250712.csv)'
    )
    
    parser.add_argument(
        '--stock_selected', 
        type=str, 
        required=True,
        help='Path to selected stock predictions CSV file (e.g., stock_selected.csv)'
    )
    
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='./output',
        help='Output directory for results (default: ./output)'
    )
    
    return parser.parse_args()


def load_and_preprocess_data(stocks_price_path, stock_selected_path):
    """
    Load and preprocess input data with memory optimization.
    
    Args:
        stocks_price_path (str): Path to stock price data file
        stock_selected_path (str): Path to selected stock predictions file
    
    Returns:
        tuple: (df_price, selected_stock)
    """
    print("=" * 60)
    print("Loading and preprocessing data...")
    print("=" * 60)
    
    # Load price data with memory optimization
    print(f"Loading price data from: {stocks_price_path}")
    print("Loading only necessary columns to save memory...")
    
    # Define necessary columns for price data
    price_columns = ['gvkey', 'datadate', 'prccd', 'ajexdi']
    
    try:
        df_price = pd.read_csv(stocks_price_path, usecols=price_columns)
        print(f"✓ Price data loaded successfully")
        print(f"  Shape: {df_price.shape}")
        print(f"  Memory usage: {df_price.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    except FileNotFoundError:
        raise FileNotFoundError(f"Price data file not found: {stocks_price_path}")
    except Exception as e:
        raise Exception(f"Error loading price data: {str(e)}")
    
    # Calculate adjusted price
    print("Calculating adjusted prices...")
    df_price['adj_price'] = df_price['prccd'] / df_price['ajexdi']
    
    # Keep only necessary columns
    df_price = df_price[["gvkey", "datadate", 'adj_price']]
    
    # Convert datadate to datetime for efficient processing
    # Handle both YYYY-MM-DD and YYYYMMDD formats
    try:
        df_price['datadate'] = pd.to_datetime(df_price['datadate'], format="%Y-%m-%d")
    except ValueError:
        try:
            df_price['datadate'] = pd.to_datetime(df_price['datadate'], format="%Y%m%d")
        except ValueError:
            # Let pandas infer the format
            df_price['datadate'] = pd.to_datetime(df_price['datadate'])
    
    print(f"✓ Price data processed")
    print(f"  Final shape: {df_price.shape}")
    print(f"  Number of unique stocks: {len(df_price.gvkey.unique())}")
    
    # Load selected stock data
    print(f"\nLoading selected stock data from: {stock_selected_path}")
    
    try:
        selected_stock = pd.read_csv(stock_selected_path)
        print(f"✓ Selected stock data loaded successfully")
        print(f"  Shape: {selected_stock.shape}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Selected stock file not found: {stock_selected_path}")
    except Exception as e:
        raise Exception(f"Error loading selected stock data: {str(e)}")
    
    # Filter data early to reduce memory usage
    print("Filtering data for dates >= 2018-03-01...")
    selected_stock = selected_stock[selected_stock.trade_date >= '2018-03-01'].reset_index(drop=True)
    
    # Convert trade_date to datetime for efficient comparison
    selected_stock['trade_date'] = pd.to_datetime(selected_stock['trade_date'])
    
    print(f"✓ Selected stock data processed")
    print(f"  Final shape: {selected_stock.shape}")
    print(f"  Memory usage: {selected_stock.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"  Number of unique stocks selected: {len(selected_stock.gvkey.unique())}")
    
    return df_price, selected_stock


def get_trade_dates(selected_stock):
    """
    Extract unique trade dates from selected stock data.
    
    Args:
        selected_stock (DataFrame): Selected stock data
    
    Returns:
        list: Unique trade dates
    """
    print("\n" + "=" * 60)
    print("Extracting trade dates...")
    print("=" * 60)
    
    trade_date = selected_stock.trade_date.unique()
    trade_date = sorted(trade_date)
    
    print(f"✓ Found {len(trade_date)} unique trade dates")
    print(f"  Date range: {trade_date[0].strftime('%Y-%m-%d')} to {trade_date[-1].strftime('%Y-%m-%d')}")
    
    return trade_date


def calculate_historical_returns(df_price, selected_stock, trade_date):
    """
    Calculate historical daily returns for selected stocks in each trade period.
    
    Args:
        df_price (DataFrame): Price data
        selected_stock (DataFrame): Selected stock data
        trade_date (list): List of trade dates
    
    Returns:
        tuple: (all_return_table, all_stocks_info)
    """
    print("\n" + "=" * 60)
    print("Calculating historical returns...")
    print("=" * 60)
    
    start_time = time.time()
    
    all_return_table = {}
    all_stocks_info = {}
    
    # Get all unique dates from price data
    all_date = df_price.datadate.unique()
    all_date = pd.to_datetime(all_date)
    
    print(f"Processing {len(trade_date)} trade periods...")
    
    for i, current_trade_date in enumerate(trade_date):
        print(f"  Processing trade date {i+1}/{len(trade_date)}: {current_trade_date.strftime('%Y-%m-%d')}")
        
        # Match trading date
        index = selected_stock.trade_date == current_trade_date
        
        # Get the corresponding trade period's selected stocks' name
        stocks_name = selected_stock.gvkey[selected_stock.trade_date == current_trade_date].values
        
        # Get stock information for this period
        temp_info = selected_stock[selected_stock.trade_date == current_trade_date]
        temp_info = temp_info.reset_index(drop=True)
        all_stocks_info[current_trade_date] = temp_info
        
        # Get the corresponding trade period's selected stocks' predicted return
        asset_expected_return = selected_stock[index].predicted_return.values
        
        # Determine the business date (one year before current trade date)
        tradedate = pd.to_datetime(current_trade_date)
        ts = datetime(tradedate.year - 1, tradedate.month, tradedate.day)
        bd = pd.tseries.offsets.BusinessDay(n=1)
        new_timestamp = ts - bd
        
        # Get dates for the past year
        get_date_index = (all_date < tradedate) & (all_date > new_timestamp)
        get_date = all_date[get_date_index]
        
        # Get adjusted price table
        return_table = pd.DataFrame()
        
        for m, stock_name in enumerate(stocks_name):
            # Get stock's historical price data
            index_tic = (df_price.gvkey == stock_name)
            sp500_temp = df_price[index_tic]
            
            # Create merge table
            merge_left_data_table = pd.DataFrame(get_date)
            merge_left_data_table.columns = ['datadate']
            
            # Merge with price data
            # sp500_temp.datadate is already converted to datetime in load_and_preprocess_data
            temp_price = merge_left_data_table.merge(sp500_temp, on=['datadate'], how='left')
            
            # Calculate daily returns
            temp_price = temp_price.dropna()
            temp_price['daily_return'] = temp_price.adj_price.pct_change()
            
            # Append to return table
            return_table = pd.concat([return_table, temp_price], ignore_index=True)
        
        all_return_table[current_trade_date] = return_table
    
    end_time = time.time()
    print(f"✓ Historical returns calculation completed")
    print(f"  Time taken: {(end_time - start_time) / 60:.2f} minutes")
    
    return all_return_table, all_stocks_info


def save_intermediate_results(all_return_table, all_stocks_info, output_dir):
    """
    Save intermediate results to pickle files.
    
    Args:
        all_return_table (dict): Historical return data
        all_stocks_info (dict): Stock information data
        output_dir (str): Output directory
    """
    print("\n" + "=" * 60)
    print("Saving intermediate results...")
    print("=" * 60)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to pickle files
    return_table_path = os.path.join(output_dir, 'all_return_table.pickle')
    stocks_info_path = os.path.join(output_dir, 'all_stocks_info.pickle')
    
    with open(return_table_path, 'wb') as handle:
        pickle.dump(all_return_table, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(stocks_info_path, 'wb') as handle:
        pickle.dump(all_stocks_info, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"✓ Intermediate results saved to {output_dir}")
    print(f"  - all_return_table.pickle")
    print(f"  - all_stocks_info.pickle")


def load_intermediate_results(output_dir):
    """
    Load intermediate results from pickle files.
    
    Args:
        output_dir (str): Output directory
    
    Returns:
        tuple: (all_return_table, all_stocks_info)
    """
    print("\n" + "=" * 60)
    print("Loading intermediate results...")
    print("=" * 60)
    
    return_table_path = os.path.join(output_dir, 'all_return_table.pickle')
    stocks_info_path = os.path.join(output_dir, 'all_stocks_info.pickle')
    
    try:
        with open(return_table_path, 'rb') as handle:
            all_return_table = pickle.load(handle)
        
        with open(stocks_info_path, 'rb') as handle:
            all_stocks_info = pickle.load(handle)
        
        print(f"✓ Intermediate results loaded successfully")
        print(f"  Number of trade periods: {len(all_stocks_info)}")
        
        return all_return_table, all_stocks_info
        
    except FileNotFoundError:
        print("⚠ Intermediate files not found. Please run the calculation first.")
        return None, None


def perform_portfolio_optimization(all_stocks_info, all_return_table, trade_date, output_dir):
    """
    Perform portfolio optimization using PyPortfolioOpt.
    
    Args:
        all_stocks_info (dict): Stock information data
        all_return_table (dict): Historical return data
        trade_date (list): List of trade dates
        output_dir (str): Output directory
    """
    print("\n" + "=" * 60)
    print("Performing portfolio optimization...")
    print("=" * 60)
    
    start_time = time.time()
    
    stocks_weight_table = pd.DataFrame([])
    
    print(f"Processing {len(trade_date)} trade periods...")
    
    for i, current_trade_date in enumerate(trade_date):
        print(f"  Optimizing portfolio {i+1}/{len(trade_date)}: {current_trade_date.strftime('%Y-%m-%d')}")
        
        # Get selected stocks information
        p1_alldata = all_stocks_info[current_trade_date]
        p1_alldata = p1_alldata.sort_values('gvkey').reset_index(drop=True)
        
        # Get selected stocks ticker
        p1_stock = p1_alldata.gvkey
        
        # Get predicted return from selected stocks
        p1_predicted_return = p1_alldata.pivot_table(
            index='trade_date', columns='gvkey', values='predicted_return'
        )
        
        # Get the 1-year historical return
        p1_return_table = all_return_table[current_trade_date]
        p1_return_table_pivot = p1_return_table.pivot_table(
            index='datadate', columns='gvkey', values='daily_return'
        )
        
        # Find common stocks between predicted and historical data
        selected_stocks = list(set(p1_predicted_return.columns).intersection(p1_return_table_pivot.columns))
        
        # Filter data for common stocks
        p1_predicted_return = p1_predicted_return.loc[:, selected_stocks]
        p1_return_table_pivot = p1_return_table_pivot.loc[:, selected_stocks]
        
        # Calculate covariance matrix
        S = risk_models.sample_cov(p1_return_table_pivot)
        mu = p1_predicted_return.T.values
        
        # Mean-variance optimization
        ef_mean = EfficientFrontier(mu, S, weight_bounds=(0, 0.05))
        raw_weights_mean = ef_mean.nonconvex_objective(
            objective_functions.sharpe_ratio,
            objective_args=(ef_mean.expected_returns, ef_mean.cov_matrix),
            weights_sum_to_one=True
        )
        cleaned_weights_mean = ef_mean.clean_weights()
        
        # Minimum variance optimization
        ef_min = EfficientFrontier([0] * len(mu), S, weight_bounds=(0, 0.05))
        raw_weights_min = ef_min.nonconvex_objective(
            objective_functions.sharpe_ratio,
            objective_args=(ef_min.expected_returns, ef_min.cov_matrix),
            weights_sum_to_one=True
        )
        cleaned_weights_min = ef_min.clean_weights()
        
        # Assign weights to stocks
        idx = np.isin(p1_alldata.gvkey, selected_stocks)
        
        p1_alldata["mean_weight"] = 0
        p1_alldata.loc[idx, 'mean_weight'] = list(cleaned_weights_mean.values())
        
        p1_alldata["min_weight"] = 0
        p1_alldata.loc[idx, 'min_weight'] = list(cleaned_weights_min.values())
        
        p1_alldata["equal_weight"] = 0
        p1_alldata.loc[idx, 'equal_weight'] = np.ones(len(cleaned_weights_mean.values())) / len(cleaned_weights_mean.values())
        
        # Append to weight table
        stocks_weight_table = pd.concat([stocks_weight_table, pd.DataFrame(p1_alldata)], ignore_index=True)
        
        print(f"    ✓ Portfolio optimized for {len(selected_stocks)} stocks")
    
    end_time = time.time()
    print(f"✓ Portfolio optimization completed")
    print(f"  Time taken: {(end_time - start_time) / 60:.2f} minutes")
    
    return stocks_weight_table


def save_results(stocks_weight_table, output_dir):
    """
    Save optimization results to Excel files.
    
    Args:
        stocks_weight_table (DataFrame): Portfolio weights data
        output_dir (str): Output directory
    """
    print("\n" + "=" * 60)
    print("Saving results...")
    print("=" * 60)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save mean-variance optimization results
    df_mean = pd.DataFrame()
    df_mean['trade_date'] = stocks_weight_table['trade_date']
    df_mean['gvkey'] = stocks_weight_table['gvkey']
    df_mean['weights'] = stocks_weight_table['mean_weight']
    df_mean['predicted_return'] = stocks_weight_table['predicted_return']
    
    mean_output_path = os.path.join(output_dir, "mean_weighted.xlsx")
    df_mean.to_excel(mean_output_path, index=False)
    
    # Save minimum variance optimization results
    df_min = pd.DataFrame()
    df_min['trade_date'] = stocks_weight_table['trade_date']
    df_min['gvkey'] = stocks_weight_table['gvkey']
    df_min['weights'] = stocks_weight_table['min_weight']
    df_min['predicted_return'] = stocks_weight_table['predicted_return']
    
    min_output_path = os.path.join(output_dir, "minimum_weighted.xlsx")
    df_min.to_excel(min_output_path, index=False)
    
    # Save equal weight results
    df_equal = pd.DataFrame()
    df_equal['trade_date'] = stocks_weight_table['trade_date']
    df_equal['gvkey'] = stocks_weight_table['gvkey']
    df_equal['weights'] = stocks_weight_table['equal_weight']
    df_equal['predicted_return'] = stocks_weight_table['predicted_return']
    
    equal_output_path = os.path.join(output_dir, "equally_weighted.xlsx")
    df_equal.to_excel(equal_output_path, index=False)
    
    print(f"✓ Results saved to {output_dir}")
    print(f"  - mean_weighted.xlsx")
    print(f"  - minimum_weighted.xlsx")
    print(f"  - equally_weighted.xlsx")
    
    # Print summary statistics
    print(f"\nSummary:")
    print(f"  Total records: {len(stocks_weight_table)}")
    print(f"  Number of trade dates: {len(stocks_weight_table['trade_date'].unique())}")
    print(f"  Number of unique stocks: {len(stocks_weight_table['gvkey'].unique())}")


def main():
    """Main function to run the portfolio optimization."""
    print("=" * 80)
    print("Fundamental Portfolio Optimization")
    print("=" * 80)
    
    # Parse command line arguments
    args = parse_arguments()
    
    print(f"Input files:")
    print(f"  Stock price data: {args.stocks_price}")
    print(f"  Selected stock data: {args.stock_selected}")
    print(f"  Output directory: {args.output_dir}")
    
    try:
        # Load and preprocess data
        df_price, selected_stock = load_and_preprocess_data(args.stocks_price, args.stock_selected)
        
        # Get trade dates
        trade_date = get_trade_dates(selected_stock)
        
        # Check if intermediate results exist
        all_return_table, all_stocks_info = load_intermediate_results(args.output_dir)
        
        if all_return_table is None:
            # Calculate historical returns
            all_return_table, all_stocks_info = calculate_historical_returns(
                df_price, selected_stock, trade_date
            )
            
            # Save intermediate results
            save_intermediate_results(all_return_table, all_stocks_info, args.output_dir)
        
        # Perform portfolio optimization
        stocks_weight_table = perform_portfolio_optimization(
            all_stocks_info, all_return_table, trade_date, args.output_dir
        )
        
        # Save results
        save_results(stocks_weight_table, args.output_dir)
        
        print("\n" + "=" * 80)
        print("Portfolio optimization completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("Please check your input files and try again.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 