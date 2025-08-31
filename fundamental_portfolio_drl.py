import warnings
warnings.filterwarnings("ignore")
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import pandas as pd
import numpy as np
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt import expected_returns
from datetime import datetime
from pandas.tseries.offsets import BDay

# Try to import gymnasium instead of gym for compatibility
try:
    import gymnasium as gym
except ImportError:
    import gym

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.preprocessor.preprocessors import data_split
from finrl import config
import pickle
import random
import torch

import time
from rl_model import run_models
# ==== ADD：Temp directory ====
#CACHE_DIR = "./cache"
#CKPT_DIR  = "./checkpoints"  # For readability; training saved by rl_model.py
RESULTS_DIR = "./results"
#os.makedirs(CACHE_DIR, exist_ok=True)
#os.makedirs(CKPT_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==== ADD：Deterministic & Random Seed ====
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
# ==== ADD：Fix for crash at end of vec env ====
def _safe_DRL_prediction(model, environment, deterministic=True):
    """
    Run a test episode and ALWAYS return (account_value_df, actions_df),
    even if the vec env ends early.
    """
    print(f"=== DEBUG:get into  _safe_DRL_prediction ===")
    test_env, test_obs = environment.get_sb_env()
    test_env.reset()
    n_steps = len(environment.df.index.unique())
    max_steps = n_steps - 1

    account_memory = None
    actions_memory = None

    for i in range(n_steps):
        action, _ = model.predict(test_obs, deterministic=deterministic)
        test_obs, rewards, dones, info = test_env.step(action)

        # Fetch memories either at last step or when done happens early
        if (i == max_steps) or dones[0]:
            account_memory = test_env.env_method("save_asset_memory")
            actions_memory = test_env.env_method("save_action_memory")
            if dones[0]:
                print("hit end!")
            break

    # Fallback: if for any reason memories weren't fetched in the loop
    if account_memory is None:
        account_memory = test_env.env_method("save_asset_memory")
    if actions_memory is None:
        actions_memory = test_env.env_method("save_action_memory")

    # env_method returns list-of-envs; take the first
    return account_memory[0], actions_memory[0]

# Apply the patch
DRLAgent.DRL_prediction = staticmethod(_safe_DRL_prediction)

# ==== ADD：Atomic Write ====
def atomic_to_csv(df: pd.DataFrame, path: str, index: bool | None = None):
    tmp = path + ".tmp"
    df.to_csv(tmp, index=(True if index is None else index))
    os.replace(tmp, path)

#def atomic_to_parquet(df: pd.DataFrame, path: str, index: bool = False):
#    tmp = path + ".tmp"
#    df.to_parquet(tmp, index=index)
#    os.replace(tmp, path)


def atomic_write_json(obj: dict, path: str):
    import json
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)

# ==== ADD：Progress Tracking ====
import json
PROGRESS_PATH = f"{RESULTS_DIR}/progress.json"

def load_progress():
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"last_idx": -1, "last_trade_date": None, "df_dict": None}  # add df_dict field

def save_progress(idx, trade_date, df_dict):  # add df_dict parameter
    atomic_write_json({
        "last_idx": idx, 
        "last_trade_date": str(trade_date.date()),
        "df_dict": df_dict  # save df_dict state
    }, PROGRESS_PATH)
    print(f"Saved progress to {PROGRESS_PATH}")


"""


import hashlib

def _hash_list(values) -> str:
    s = ",".join(map(str, sorted(list(values))))
    return hashlib.md5(s.encode()).hexdigest()[:10]

def load_or_build_fe_features(df_src: pd.DataFrame,
                              p1_stock: pd.Series,
                              earliest_date: pd.Timestamp,
                              end_exclusive: pd.Timestamp) -> pd.DataFrame:

#    Only cache FeatureEngineer.preprocess_data() output (without cov_list/return_list).
#    key is determined by (earliest_date, end_exclusive, stock set hash).

    key = f"{earliest_date.date()}_{end_exclusive.date()}_{_hash_list(p1_stock)}"
    feat_path = f"{CACHE_DIR}/fe_{key}.parquet"

    if os.path.exists(feat_path):
        print(f"Loading cached FE features from {feat_path}")
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
    print(f"Cached FE features to {feat_path}")
    return df_
"""
def check_per_date_stock_coverage(df_, stock_dim):
    stock_counts = df_.groupby("date")["tic"].nunique()
    invalid_dates = stock_counts[stock_counts != stock_dim]
    if not invalid_dates.empty:
        print("[WARNING] Found dates with missing stocks:")
        print(invalid_dates)
        return df_[df_["date"].isin(stock_counts[stock_counts == stock_dim].index)]
    return df_

def zscore_normalize_indicators(df: pd.DataFrame, indicators: list[str]) -> pd.DataFrame:
    """
    Global Z-score normalization for technical indicators: --》 improve RL performance
      x' = (x - mean_all) / std_all
    - Only applies to columns listed in `indicators`
    - Safely handles inf/NaN; zero-variance columns become 0
    """
    df = df.copy()
    for col in indicators:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        mu = vals.mean(skipna=True)
        sigma = vals.std(skipna=True)
        if sigma and sigma > 0:
            df[col] = (vals - mu) / sigma
        else:
            # if no variance (or all NaN), set to 0 to avoid NaN spillover
            df[col] = 0.0
    return df

def compute_and_save_performance(df_daily_return: pd.DataFrame,
                                 df_actions: pd.DataFrame,
                                 out_prefix: str = "backtest",
                                 results_dir: str = "results",
                                 rf_annual: float = 0.0,
                                 trading_days: int = 252) -> pd.DataFrame:
    """
    Calculate backtest performance metrics and save them to files:
      - Total return, annualized return, annualized volatility, Sharpe ratio, max drawdown
      - Daily weight sum (check if ≈ 1)
      - Turnover (total and average daily)

    Saves:
      - <results_dir>/{out_prefix}_summary.csv: single-line summary table
      - <results_dir>/{out_prefix}_equity_curve.csv: equity curve and drawdown
      - <results_dir>/{out_prefix}_turnover.csv: daily turnover
      - <results_dir>/{out_prefix}_weights_sum.csv: daily sum of portfolio weights

    Parameters
    ----------
    df_daily_return : DataFrame
        Must contain 'daily_return' column; optionally a 'date' column (otherwise index is used).
    df_actions : DataFrame
        Index is date, columns are stock weights.
    out_prefix : str
        Prefix for saved files.
    results_dir : str
        Output directory.
    rf_annual : float
        Annualized risk-free rate for Sharpe calculation (e.g., 0.02 for 2%).
    trading_days : int
        Number of trading days used for annualization (default 252).

    Returns
    -------
    summary_df : DataFrame
        One-row DataFrame with key metrics.
    """
    print("[DEBUG] df_daily_return type:", type(df_daily_return))
    if df_daily_return is not None:
        print("[DEBUG] df_daily_return shape:", df_daily_return.shape)
        print("[DEBUG] df_daily_return columns:", list(df_daily_return.columns))
        print("[DEBUG] df_daily_return head:\n", df_daily_return.head())

    if not isinstance(df_daily_return, pd.DataFrame) or df_daily_return.empty:
        raise ValueError("df_daily_return is None or empty")
    if "daily_return" not in df_daily_return.columns:
        raise ValueError("df_daily_return must contain 'daily_return'")
    if "date" in df_daily_return.columns:
        df_daily_return = df_daily_return.sort_values("date").set_index("date")
    import os
    os.makedirs(results_dir, exist_ok=True)


    # --- Prepare daily returns ---
    dr = df_daily_return.copy()
    if "date" in dr.columns:
        dr = dr.sort_values("date")
        dr.set_index("date", inplace=True)
    dr = dr[["daily_return"]].dropna()

    # --- Equity curve & drawdown ---
    equity = (1.0 + dr["daily_return"]).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_drawdown = drawdown.min() if len(drawdown) > 0 else np.nan

    # --- Annualized return, volatility, Sharpe ---
    n = len(dr)
    if n > 0:
        total_return = equity.iloc[-1] - 1.0
        ann_return = (equity.iloc[-1]) ** (trading_days / n) - 1.0
        ann_vol = dr["daily_return"].std() * np.sqrt(trading_days)
        rf_daily = (1.0 + rf_annual) ** (1.0 / trading_days) - 1.0
        excess_daily = dr["daily_return"] - rf_daily
        ann_excess_ret = excess_daily.mean() * trading_days
        sharpe = ann_excess_ret / ann_vol if ann_vol and ann_vol > 0 else np.nan
    else:
        total_return = ann_return = ann_vol = sharpe = np.nan

    # --- Weight sum check ---
    if df_actions is not None and not df_actions.empty:
        weights_sum = df_actions.sum(axis=1)
        weights_sum.to_frame("weights_sum").to_csv(
            os.path.join(results_dir, f"{out_prefix}_weights_sum.csv")
        )
        weights_sum_min = float(weights_sum.min())
        weights_sum_max = float(weights_sum.max())
        weights_sum_mean = float(weights_sum.mean())
    else:
        weights_sum_min = weights_sum_max = weights_sum_mean = np.nan

    # --- Turnover calculation ---
    # turnover_t = sum(|w_t - w_{t-1}|) / 2
    if df_actions is not None and len(df_actions) > 1:
        actions_sorted = df_actions.copy().sort_index()
        dw = actions_sorted.diff().abs()
        turnover_series = dw.sum(axis=1) / 2.0
        turnover_series = turnover_series.dropna()
        turnover_series.to_frame("turnover").to_csv(
            os.path.join(results_dir, f"{out_prefix}_turnover.csv")
        )
        total_turnover = float(turnover_series.sum())
        avg_daily_turnover = float(turnover_series.mean())
    else:
        total_turnover = avg_daily_turnover = np.nan

    # --- Save equity curve & drawdown ---
    eq_df = pd.DataFrame(
        {
            "equity": equity,
            "drawdown": drawdown,
            "daily_return": dr["daily_return"],
        }
    )
    eq_df.to_csv(os.path.join(results_dir, f"{out_prefix}_equity_curve.csv"))

    # --- Summary table ---
    summary = {
        "n_days": n,
        "total_return": float(total_return) if pd.notna(total_return) else np.nan,
        "annual_return": float(ann_return) if pd.notna(ann_return) else np.nan,
        "annual_vol": float(ann_vol) if pd.notna(ann_vol) else np.nan,
        "sharpe": float(sharpe) if pd.notna(sharpe) else np.nan,
        "max_drawdown": float(max_drawdown) if pd.notna(max_drawdown) else np.nan,
        "weights_sum_min": weights_sum_min,
        "weights_sum_mean": weights_sum_mean,
        "weights_sum_max": weights_sum_max,
        "total_turnover": total_turnover,
        "avg_daily_turnover": avg_daily_turnover,
    }
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(results_dir, f"{out_prefix}_summary.csv"), index=False)
    return summary_df
def main():
    # read price data

    set_global_seed(42)

    print("Loading price data...")
    usecols = [
        "datadate", "prcod", "prccd", "prchd", "prcld", "cshtrd", "ajexdi", "gvkey"
    ]

    dtypes = {
        "prcod": "float32",
        "prccd": "float32",
        "prchd": "float32",
        "prcld": "float32",
        "cshtrd": "float32",   # volume as float32 is fine
        "ajexdi": "float32",
        "gvkey": "int32",
    }

    # If you have pandas>=2.0 and pyarrow installed, this is the most memory-efficient:
    # df_price = pd.read_csv(
    #     "./data_processor/sp500_tickers_daily_price_20250712.csv",
    #     usecols=usecols,
    #     dtype=dtypes,
    #     parse_dates=["datadate"],
    #     engine="pyarrow",
    # )

    # Otherwise, use the C engine with low_memory off:
    df_price = pd.read_csv(
        "./data_processor/sp500_tickers_daily_price_20250712.csv",
        usecols=usecols,
        dtype=dtypes,
        parse_dates=["datadate"],
        low_memory=False,
        engine="c",
    )

    #df_price = pd.read_csv("./data_processor/sp500_tickers_daily_price_20250712.csv")
    print(f"Price data loaded: {df_price.shape}")
    print(f"Price data columns: {list(df_price.columns)}")
    print(f"Sample data:")
    print(df_price.head())

    df_price['adjcp'] = df_price['prccd'] / df_price['ajexdi']

    df_price['date'] = df_price['datadate']
    df_price['open'] = df_price['prcod']
    df_price['close'] = df_price['prccd']
    df_price['high'] = df_price['prchd']
    df_price['low'] = df_price['prcld']
    df_price['volume'] =df_price['cshtrd']

    df = df_price[['date', 'open', 'close', 'high', 'low','adjcp','volume', 'gvkey']]
    print(f"Processed data shape: {df.shape}")
    print(f"Processed data columns: {list(df.columns)}")

    df['tic'] = df_price['gvkey']

    # Fix date format conversion to handle both YYYY-MM-DD and YYYYMMDD formats
    try:
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    except ValueError:
        try:
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        except ValueError:
            # Let pandas infer the format
            df['date'] = pd.to_datetime(df['date'])

    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Number of unique stocks: {len(df['gvkey'].unique())}")

    df['day'] = [x.weekday() for x in df['date']]
    df.drop_duplicates(['gvkey', 'date'], inplace=True)
    print(f"After removing duplicates: {df.shape}")
    selected_stock = pd.read_csv("./result/stock_selected.csv")

    # Convert trade_date to datetime and filter from 2018-03-01 to current date
    selected_stock['trade_date'] = pd.to_datetime(selected_stock['trade_date'])
    print(f"Original selected_stock shape: {selected_stock.shape}")

    # Filter data from 2018-03-01 to current date (should not include future dates)
    current_date = pd.Timestamp.now().normalize()  # current date 
    selected_stock = selected_stock[
        (selected_stock.trade_date >= '2018-03-01') & 
        (selected_stock.trade_date <= current_date)
    ].reset_index(drop=True)
    print(f"Filtered selected_stock shape (2018-03-01 to {current_date}): {selected_stock.shape}")

    with open('./output/all_return_table.pickle', 'rb') as handle:
        all_return_table = pickle.load(handle)

    with open('./output/all_stocks_info.pickle', 'rb') as handle:
        all_stocks_info = pickle.load(handle)

    # Get only the trade dates that exist in all_stocks_info
    available_trade_dates = list(all_stocks_info.keys())
    trade_date = [pd.to_datetime(date) for date in available_trade_dates]
    trade_date = sorted(trade_date)

    print("Available keys in all_stocks_info:")
    print(trade_date[:5])  # Show first 5 keys
    print(f"Total keys: {len(trade_date)}")

    # Verify that our filtered selected_stock matches all_stocks_info
    filtered_trade_dates = selected_stock.trade_date.unique()
    print(f"Filtered trade dates from selected_stock: {len(filtered_trade_dates)}")
    print(f"First filtered date: {filtered_trade_dates[0]}")
    print(f"Last filtered date: {filtered_trade_dates[-1]}")


    prog = load_progress()
    start_idx = max(1, prog.get("last_idx", -1) + 1)
    df_dict = prog.get("df_dict", {'trade_date':[], 'gvkey':[], 'weights':[]})  # 恢复df_dict

    # 1 year
    #testing_window = pd.Timedelta(np.timedelta64(1,'Y'))
    testing_window = pd.Timedelta(days=365)  # 1 year --
    #max_rolling_window = pd.Timedelta(np.timedelta64(10, 'Y'))
    max_rolling_window = pd.Timedelta(days=1095)  # 10 years -->change from 10year to 3 year

    print(f"Number of trade dates used (should be ~31): {len(trade_date)}")
    # ==== ADD：Progress Tracking ====
    #prog = load_progress()
    #start_idx = max(1, prog.get("last_idx", -1) + 1)

    for idx in range(start_idx, len(trade_date)):
        current_trade_date = trade_date[idx-1]
    #for idx in range(1, len(trade_date)):
    #    current_trade_date = trade_date[idx-1]
        #
        # Check if the key exists in all_stocks_info
        if current_trade_date not in all_stocks_info:
            print(f"Warning: {current_trade_date} not found in all_stocks_info. Skipping...")
            continue
        
        p1_alldata = all_stocks_info[current_trade_date]
        p1_alldata = p1_alldata.sort_values('gvkey')
        p1_alldata = p1_alldata.reset_index()
        del p1_alldata['index']
        p1_stock = p1_alldata.gvkey

        earliest_date = current_trade_date - max_rolling_window

        
        df_ = df[df['tic'].isin(p1_stock) & (df['date'] >= earliest_date) & (df['date'] < trade_date[idx])]
        print(f"Processing trade date {idx}: {current_trade_date}")
        print(f"Data shape: {df_.shape}")
        
        if df_.empty:
            print(f"Warning: No data for trade date {current_trade_date}. Skipping...")
            continue
        fe = FeatureEngineer(
                        use_technical_indicator=True,
                        use_turbulence=False,
                        user_defined_feature = False)

        df_ = fe.preprocess_data(df_)

        df_=df_.sort_values(['date','tic'],ignore_index=True)
        df_.index = df_.date.factorize()[0]
             
        cov_list = []
        return_list = []

        # look back is one year
        lookback=252
        for i in range(lookback,len(df_.index.unique())):
            data_lookback = df_.loc[i-lookback:i,:]
            price_lookback=data_lookback.pivot_table(index = 'date',columns = 'tic', values = 'close')
            return_lookback = price_lookback.pct_change().dropna()
            return_list.append(return_lookback)

            covs = return_lookback.cov().values
            cov_list.append(covs)

    
        df_cov = pd.DataFrame({'date':df_.date.unique()[lookback:],'cov_list':cov_list,'return_list':return_list})
        df_ = df_.merge(df_cov, on='date')
        df_ = df_.sort_values(['date','tic']).reset_index(drop=True)

        stock_dimension = len(df_.tic.unique())
        # lxy: adjust state_space =stock_dim^2 + stock_dim * len(indicators)
        # lxy: vols + top-K eigenvalues + per-stock tech indicators 
    # state_space = stock_dimension ** 2 + stock_dimension * len(config.INDICATORS)
        K_EIG = 10  #  must be <= stock_dimension
        state_space = stock_dimension + K_EIG + stock_dimension * len(config.INDICATORS)
        env_kwargs = {
        "hmax": 100, 
        "initial_amount": 1000000, 
        "transaction_cost_pct": 0.001, 
        "state_space": state_space, 
        "stock_dim": stock_dimension, 
        "tech_indicator_list": config.INDICATORS, 
        "action_space": stock_dimension, 
        "reward_scaling": 1e-4,
        "k_eig": K_EIG   # <--- pass K down to the environment
        }

        
        try:
                # before calling run_models, rename column name
            print(f"=== DEBUG: Before run_models ===")
            print(f"Before rename - df_ columns: {list(df_.columns)}")
            print(f"Before rename - df_ has 'date' column: {'date' in df_.columns}")
            print(f"Before rename - df_ has 'datadate' column: {'datadate' in df_.columns}")
            print(f"Before rename - df_ shape: {df_.shape}")
            print(f"Before rename - df_ sample data:")
            print(df_.head(2))
            
            #df_ = df_.rename(columns={'date': 'datadate'})
            print(f"=== DEBUG: After rename ===")
            print(f"After rename - df_ columns: {list(df_.columns)}")
            print(f"After rename - df_ has 'date' column: {'date' in df_.columns}")
            print(f"After rename - df_ has 'datadate' column: {'datadate' in df_.columns}")
            
            print(f"=== DEBUG: Calling run_models ===")
            print(f"Calling run_models with date_column='datadate'")
            #print(f"Stock count used in training: {len(df_.tic.unique())}")
            #print(f"Stock list: {df_.tic.unique()}")
            df_ = check_per_date_stock_coverage(df_, stock_dimension)
            # move td3 and sac model td3_model,sac_model,
            try:
                a2c_model,ppo_model,ddpg_model,best_model = run_models(df_, "date", current_trade_date, env_kwargs,testing_window, max_rolling_window)
                print(f"=== DEBUG: run_models completed successfully ===")
            except Exception as run_models_error:
                print(f"=== DEBUG: run_models failed ===")
                print(f"Error in run_models: {str(run_models_error)}")
                print(f"Error type: {type(run_models_error)}")
                import traceback
                print(f"Traceback:")
                #traceback.print_exc()
                raise run_models_error
            
            # now df_ has 'datadate' column, use it directly
            print(f"=== DEBUG: Trading data ===")
            print(f"Before data_split - df_ columns: {list(df_.columns)}")
            print(f"Before data_split - df_ shape: {df_.shape}")
            print(f"current_trade_date: {current_trade_date}")
            print(f"trade_date[idx]: {trade_date[idx]}")
            
            trade = data_split(df_, current_trade_date, trade_date[idx])
            print(f"=== DEBUG: After trade date data_split ===")
            print(f"After data_split - trade shape: {trade.shape if hasattr(trade, 'shape') else 'No shape'}")
            print(f"After data_split - trade columns: {list(trade.columns) if hasattr(trade, 'columns') else 'No columns'}")
            print(f"After data_split - trade type: {type(trade)}")
            
            #print(f"=== DEBUG: Before StockPortfolioEnv ===")
            e_trade_gym = StockPortfolioEnv(df = trade, **env_kwargs)
            print(f"=== DEBUG: StockPortfolioEnv created successfully ===")
            
            #print(f"=== DEBUG: Before DRL_prediction ===")
            #print(f"Predicting with model: A2C")
    #using best model as call back test , if best model is null , then select a2c model
            print("=== DEBUG: Before DRL_prediction ===")
            # ==== ADD: Use best_model if available; fallback to A2C if best_model is None ====
            model_for_backtest = best_model if best_model is not None else a2c_model
            #print(f"best model is {best_model}")
            if model_for_backtest is None:
                raise RuntimeError("No model available for backtesting (best_model and a2c_model are both None).")

            model_name = type(model_for_backtest).__name__
            print(f"trade date Predicting with model: {model_name}")

            df_daily_return, df_actions = DRLAgent.DRL_prediction(
                model=model_for_backtest, environment=e_trade_gym
            )
    #        df_daily_return, df_actions = DRLAgent.DRL_prediction(
    #        model=a2c_model, environment=e_trade_gym
    #        )
            print(f"tradedate  df_daily_return.shape: {df_daily_return.shape}")
            print(f"tradedatedf_actions.shape: {df_actions.shape}")
            print(f"=== DEBUG: DRL_prediction completed successfully ===")
            print(f"df_actions shape: {df_actions.shape if hasattr(df_actions, 'shape') else 'No shape'}")
            print(f"df_actions columns: {list(df_actions.columns) if hasattr(df_actions, 'columns') else 'No columns'}")
            
            # weight accumulation
            for i in range(len(df_actions)):
                for j in df_actions.columns:
                    df_dict['trade_date'].append(df_actions.index[i])
                    df_dict['gvkey'].append(j)
                    df_dict['weights'].append(df_actions.loc[df_actions.index[i], j])
            
            out_prefix = f"bt_{current_trade_date.strftime('%Y%m%d')}_{trade_date[idx].strftime('%Y%m%d')}"
            print(f"[PERF CALL] calling compute_and_save_performance for {out_prefix}")
            print("[CHECK] df_daily_return columns before perf:", list(df_daily_return.columns) if df_daily_return is not None else None)
            print("[CHECK] df_daily_return shape before perf:", df_daily_return.shape if df_daily_return is not None else None)
            print("[CHECK] df_daily_return head before perf:\n", df_daily_return.head() if df_daily_return is not None else None)


            try:
                summary_df = compute_and_save_performance(
                    df_daily_return=df_daily_return,
                    df_actions=df_actions,
                    out_prefix=out_prefix,
                    results_dir="results",
                    rf_annual=0.02,
                    trading_days=252
                )
                print(summary_df)
                print(f"[PERF DONE] compute_and_save_performance finished for {out_prefix}")

            except Exception as perf_err:
                print(f"[PERF ERROR] {out_prefix}: {perf_err}")

            
        except Exception as e:
            print(f"[PERF SKIP] compute_and_save_performance skipped due to error in {current_trade_date}")
            print(f"Error processing trade date {current_trade_date}: {str(e)}")
            
            # Add detailed debugging information for array dimension mismatch
            if "array dimensions" in str(e) and "concatenation axis" in str(e):
                print("\n=== ARRAY DIMENSION DEBUG INFO ===")
                print(f"Current trade date: {current_trade_date}")
                print(f"DataFrame shape before run_models: {df_.shape}")
                print(f"Number of unique stocks: {len(df_.tic.unique())}")
                # make sure right column name
                if 'date' in df_.columns:
                    print(f"Number of unique dates: {len(df_.date.unique())}")
                elif 'datadate' in df_.columns:
                    print(f"Number of unique dates: {len(df_.datadate.unique())}")
                
                # Check data structure
                print(f"\nDataFrame columns: {list(df_.columns)}")
                print(f"DataFrame dtypes: {df_.dtypes}")
                
                # Check for any NaN values
                print(f"\nNaN values in DataFrame:")
                print(df_.isnull().sum())
                
                # Check data distribution by stock
                stock_counts = df_.groupby('tic').size()
                print(f"\nData points per stock:")
                print(f"Min: {stock_counts.min()}")
                print(f"Max: {stock_counts.max()}")
                print(f"Mean: {stock_counts.mean():.2f}")
                print(f"Stocks with < 252 data points: {(stock_counts < 252).sum()}")
                
                # Check date range for each stock
                print(f"\nDate range analysis:")
                for tic in df_.tic.unique()[:5]:  # Show first 5 stocks
                    stock_data = df_[df_.tic == tic]
                    # make sure right column name
                    date_col = 'date' if 'date' in stock_data.columns else 'datadate'
                    print(f"Stock {tic}: {stock_data[date_col].min()} to {stock_data[date_col].max()} ({len(stock_data)} records)")
                
                # Check if there are any stocks with insufficient data
                insufficient_stocks = stock_counts[stock_counts < 252]
                if len(insufficient_stocks) > 0:
                    print(f"\nStocks with insufficient data (< 252 records):")
                    print(insufficient_stocks.head(10))
        finally: 
            save_progress(idx, trade_date[idx], df_dict)  #  save df_dict           



    # save the accumulated weights data after the loop
    df_rl = pd.DataFrame(df_dict)
    df_rl.to_csv("./results/drl_weight.csv")
    print("DRL weights saved to drl_weight.csv")




    # add debug info at the end of the file
    print(f"\nDebug: df_dict contents:")
    print(f"  trade_date entries: {len(df_dict['trade_date'])}")
    print(f"  gvkey entries: {len(df_dict['gvkey'])}")
    print(f"  weights entries: {len(df_dict['weights'])}")

    if len(df_dict['trade_date']) == 0:
        print("Warning: No data was processed. df_dict is empty.")
        print("This could be due to:")
        print("1. All trade dates were skipped")
        print("2. No data available for the selected stocks")
        print("3. Errors in the DRL model training/prediction")
    else:
        #df_rl = pd.DataFrame(df_dict)
        #df_rl.to_csv("drl_weight.csv")
        #print(f"DRL weights saved to drl_weight.csv")
        print(f"Final DataFrame shape: {df_rl.shape}")

if __name__ == "__main__":
    # Windows/Colab/多数环境都推荐 spawn；SB3 的 SubprocVecEnv 也默认用 spawn
    import multiprocessing as mp
    from multiprocessing import freeze_support
    freeze_support()  # 不是必须，但对某些环境友好

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # 已设置过启动方式会抛 RuntimeError，忽略即可
        pass

    main() 