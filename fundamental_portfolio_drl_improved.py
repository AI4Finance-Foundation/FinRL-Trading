#!/usr/bin/env python3
"""
Fundamental Portfolio Optimization using Deep Reinforcement Learning (DRL)
Improved version with command line arguments support
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

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.preprocessor.preprocessors import data_split
from finrl import config

from rl_model import run_models
# ==== ADD：Temp directory ====
CACHE_DIR = "./cache"
CKPT_DIR  = "./checkpoints"  # 仅供可读性；训练保存由 rl_model.py 负责
RESULTS_DIR = "./results"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CKPT_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==== ADD：Deterministic & Random Seed ====
import random
import torch

def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # CuDNN Deterministic: Same input, same output (slightly sacrifice speed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

# ==== ADD：Atomic Write ====
def atomic_to_csv(df: pd.DataFrame, path: str, index: bool | None = None):
    tmp = path + ".tmp"
    df.to_csv(tmp, index=(True if index is None else index))
    os.replace(tmp, path)

def atomic_to_parquet(df: pd.DataFrame, path: str, index: bool = False):
    tmp = path + ".tmp"
    df.to_parquet(tmp, index=index)
    os.replace(tmp, path)

# ==== ADD：FeatureEngineer Cache Tool ====
import hashlib

def _hash_list(values) -> str:
    s = ",".join(map(str, sorted(list(values))))
    return hashlib.md5(s.encode()).hexdigest()[:10]

def load_or_build_fe_features(df_src: pd.DataFrame,
                              p1_stock: pd.Series,
                              earliest_date: pd.Timestamp,
                              end_exclusive: pd.Timestamp) -> pd.DataFrame:
    """
    Only cache FeatureEngineer.preprocess_data() output (without cov_list/return_list).
    key is determined by (earliest_date, end_exclusive, stock set hash).
    """
    key = f"{earliest_date.date()}_{end_exclusive.date()}_{_hash_list(p1_stock)}"
    feat_path = f"{CACHE_DIR}/fe_{key}.parquet"

    if os.path.exists(feat_path):
        return pd.read_parquet(feat_path)

    # —— Original logic: slice + FE preprocess ——
    df_ = df_src[df_src['tic'].isin(p1_stock) &
                 (df_src['date'] >= earliest_date) &
                 (df_src['date'] < end_exclusive)]
    if df_.empty:
        return df_

    fe = FeatureEngineer(use_technical_indicator=True,
                         use_turbulence=False,
                         user_defined_feature=False)
    df_ = fe.preprocess_data(df_)
    df_ = df_.sort_values(['date', 'tic'], ignore_index=True)
    # Keep the factorized index for lookback later
    df_.index = df_.date.factorize()[0]

    # Cache FE output (cov_list/return_list still calculated as before)
    atomic_to_parquet(df_, feat_path, index=False)
    return df_


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='DRL Portfolio Optimization with command line arguments'
    )
    
    parser.add_argument(
        '--stocks_price', 
        type=str, 
        required=True,
        help='Path to stock price data CSV file'
    )
    
    parser.add_argument(
        '--stock_selected', 
        type=str, 
        required=True,
        help='Path to selected stock predictions CSV file'
    )
    
    parser.add_argument(
        '--return_table', 
        type=str, 
        default='all_return_table.pickle',
        help='Path to return table pickle file'
    )
    
    parser.add_argument(
        '--stocks_info', 
        type=str, 
        default='all_stocks_info.pickle',
        help='Path to stocks info pickle file'
    )
    
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='./drl_output',
        help='Output directory for results'
    )
    
    return parser.parse_args()


def main():
    

    """Main function."""
    set_global_seed(42)
    args = parse_arguments()
    
    print("=" * 60)
    print("DRL Portfolio Optimization")
    print("=" * 60)
    print(f"Stock price data: {args.stocks_price}")
    print(f"Selected stock data: {args.stock_selected}")
    print(f"Return table: {args.return_table}")
    print(f"Stocks info: {args.stocks_info}")
    print(f"Output directory: {args.output_dir}")
    
    # Load price data
    print("\nLoading price data...")
    df_price = pd.read_csv(args.stocks_price)
    df_price['adjcp'] = df_price['prccd'] / df_price['ajexdi']
    
    # Create standardized dataframe
    df = df_price[['datadate', 'prcod', 'prccd', 'prchd', 'prcld', 'adjcp', 'cshtrd', 'gvkey']]
    df.columns = ['date', 'open', 'close', 'high', 'low', 'adjcp', 'volume', 'gvkey']
    df['tic'] = df['gvkey']
    
    # Convert date format
    try:
        df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d")
    except ValueError:
        df['date'] = pd.to_datetime(df['date'], format="%Y%m%d")
    
    df['day'] = [x.weekday() for x in df['date']]
    df.drop_duplicates(['gvkey', 'date'], inplace=True)
    
    # Load selected stock data
    print("Loading selected stock data...")
    selected_stock = pd.read_csv(args.stock_selected)
    
    # Load intermediate data
    print("Loading intermediate data...")
    with open(args.return_table, 'rb') as handle:
        all_return_table = pickle.load(handle)
    
    with open(args.stocks_info, 'rb') as handle:
        all_stocks_info = pickle.load(handle)
    
    # Process data
    trade_date = selected_stock.trade_date.unique()
    df_dict = {'trade_date': [], 'gvkey': [], 'weights': []}
    testing_window = pd.Timedelta(np.timedelta64(1, 'Y'))
    max_rolling_window = pd.Timedelta(np.timedelta64(10, 'Y'))
    
    print(f"Processing {len(trade_date)-1} trade periods...")
    
    for idx in range(1, len(trade_date)):
        print(f"  Processing period {idx}: {trade_date[idx-1]} to {trade_date[idx]}")
        
        # Get stock data
        p1_alldata = all_stocks_info[trade_date[idx-1]]
        p1_alldata = p1_alldata.sort_values('gvkey').reset_index(drop=True)
        p1_stock = p1_alldata.gvkey
        
        # Filter data
        earliest_date = pd.to_datetime(trade_date[idx-1]) - max_rolling_window
        df_ = df[df['tic'].isin(p1_stock) & (df['date'] >= earliest_date) & (df['date'] < trade_date[idx])]
        
        if len(df_) == 0:
            continue
        
        # Feature engineering
        fe = FeatureEngineer(use_technical_indicator=True, use_turbulence=False, user_defined_feature=False)
        df_ = fe.preprocess_data(df_)
        # zscore normalize indicators
        #df_ = zscore_normalize_indicators(df_, config.INDICATORS)
        df_ = df_.sort_values(['date', 'tic'], ignore_index=True)
        df_.index = df_.date.factorize()[0]
        
        # Calculate covariance
        cov_list = []
        return_list = []
        lookback = 252
        
        for i in range(lookback, len(df_.index.unique())):
            data_lookback = df_.loc[i-lookback:i, :]
            price_lookback = data_lookback.pivot_table(index='date', columns='tic', values='close')
            return_lookback = price_lookback.pct_change().dropna()
            return_list.append(return_lookback)
            covs = return_lookback.cov().values
            cov_list.append(covs)
        
        # Merge data
        df_cov = pd.DataFrame({
            'date': df_.date.unique()[lookback:],
            'cov_list': cov_list,
            'return_list': return_list
        })
        df_ = df_.merge(df_cov, on='date')
        df_ = df_.sort_values(['date', 'tic']).reset_index(drop=True)
        
        # Environment setup
        stock_dimension = len(df_.tic.unique())
        env_kwargs = {
            "hmax": 100,
            "initial_amount": 1000000,
            "transaction_cost_pct": 0.001,
            "state_space": stock_dimension,
            "stock_dim": stock_dimension,
            "tech_indicator_list": config.INDICATORS,
            "action_space": stock_dimension,
            "reward_scaling": 1e-4
        }
        
        # Train models
        a2c_model, ppo_model, ddpg_model, td3_model, sac_model, best_model = run_models(
            df_, "date", pd.to_datetime(trade_date[idx-1]), env_kwargs, 
            testing_window, max_rolling_window
        )
        
        # Make predictions
        trade = data_split(df_, pd.to_datetime(trade_date[idx-1]), pd.to_datetime(trade_date[idx]))
        e_trade_gym = StockPortfolioEnv(df=trade, **env_kwargs)
        df_daily_return, df_actions = DRLAgent.DRL_prediction(model=a2c_model, environment=e_trade_gym)
        
        # Store results
        for i in range(len(df_actions)):
            for j in df_actions.columns:
                df_dict['trade_date'].append(df_actions.index[i])
                df_dict['gvkey'].append(j)
                df_dict['weights'].append(df_actions.loc[df_actions.index[i], j])
    
    # Save results
    df_rl = pd.DataFrame(df_dict)
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "drl_weight.csv")
    df_rl.to_csv(output_path, index=False)
    
    print(f"\n✓ Results saved to: {output_path}")
    print(f"  Total records: {len(df_rl)}")


if __name__ == "__main__":
    main() 