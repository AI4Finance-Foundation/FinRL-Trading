# Fundamental Portfolio DRL System Documentation

## Overview

This document describes the Deep Reinforcement Learning (DRL) based portfolio management system implemented in `fundamental_portfolio_drl.py` and `rl_model.py`. The system implements a rolling time window strategy for automated stock portfolio allocation using multiple reinforcement learning algorithms.

## File Structure and Purpose

### 1. `fundamental_portfolio_drl.py` - Main Orchestration File

**Purpose**: Main execution script that orchestrates the entire DRL portfolio management pipeline.

**Key Responsibilities**:
- Data loading and preprocessing
- Rolling time window strategy implementation
- Model training coordination
- Backtesting and performance evaluation
- Results generation and storage

### 2. `rl_model.py` - Reinforcement Learning Model Training

**Purpose**: Handles the training, evaluation, and selection of multiple reinforcement learning algorithms.

**Key Responsibilities**:
- Model training for A2C, PPO, and DDPG algorithms
- Model checkpoint management
- Performance evaluation and best model selection
- Data preparation for training and testing

## Core Functions Analysis

### `fundamental_portfolio_drl.py` Functions

#### Data Management Functions

```python
def set_global_seed(seed: int = 42)
```
- **Purpose**: Ensures reproducible results across runs
- **Input**: Random seed value
- **Output**: Sets global random seeds for numpy, torch, and random modules

```python
def load_or_build_fe_features(df_src, p1_stock, earliest_date, end_exclusive)
```
- **Purpose**: Caches feature engineering results to avoid recomputation
- **Input**: Source data, stock list, date range
- **Output**: Preprocessed data with technical indicators
- **Cache Strategy**: Uses MD5 hash of parameters as cache key

```python
def atomic_to_csv(df, path, index=None)
def atomic_to_parquet(df, path, index=False)
def atomic_write_json(obj, path)
```
- **Purpose**: Thread-safe file writing operations
- **Input**: Data to write, file path
- **Output**: Safely written files with temporary file approach

#### Progress Tracking Functions

```python
def load_progress()
def save_progress(idx, trade_date)
```
- **Purpose**: Resume execution from last successful point
- **Input/Output**: JSON file tracking last processed trade date
- **Use Case**: Enables long-running processes to resume after interruption

#### Data Validation Functions

```python
def check_per_date_stock_coverage(df_, stock_dim)
```
- **Purpose**: Validates data completeness for each trading date
- **Input**: DataFrame, expected stock dimension
- **Output**: Filtered DataFrame with complete data
- **Validation**: Ensures each date has exactly `stock_dim` stocks

#### Performance Analysis Functions

```python
def compute_and_save_performance(df_daily_return, df_actions, out_prefix, results_dir, rf_annual, trading_days)
```
- **Purpose**: Comprehensive backtesting performance evaluation
- **Input**: Daily returns, portfolio actions, parameters
- **Output**: Multiple CSV files with performance metrics
- **Metrics**: Total return, Sharpe ratio, max drawdown, turnover, etc.

### `rl_model.py` Functions

#### Model Training Functions

```python
def train_a2c(agent, USE_GPU)
def train_ppo(agent, USE_GPU)
def train_ddpg(agent, USE_GPU)
```
- **Purpose**: Train specific RL algorithms
- **Input**: DRLAgent instance, GPU flag
- **Output**: Trained model instances
- **Parameters**: Optimized for GPU/CPU environments

#### Checkpoint Management Functions

```python
def save_ckpt(model, save_dir, algo_name, keep_last_n_dirs=3)
def load_ckpt(env, save_dir, algo_name, device="auto")
```
- **Purpose**: Model persistence and loading
- **Input**: Model, directory, algorithm name
- **Output**: Saved/loaded model files
- **Cleanup**: Automatically removes old checkpoints

#### Data Preparation Functions

```python
def prepare_rolling_train(df, date_column, testing_window, max_rolling_window, trade_date)
def prepare_rolling_test(df, date_column, testing_window, max_rolling_window, trade_date)
```
- **Purpose**: Split data into training and testing windows
- **Input**: Full dataset, time parameters
- **Output**: Training and testing datasets
- **Strategy**: Rolling window approach for time series data

#### Main Orchestration Function

```python
def run_models(df, date_column, trade_date, env_kwargs, testing_window, max_rolling_window)
```
- **Purpose**: Coordinates entire model training and evaluation pipeline
- **Input**: Preprocessed data, parameters
- **Output**: Trained models (A2C, PPO, DDPG) and best model
- **Process**: 
  1. Prepare training/testing data
  2. Train multiple models
  3. Evaluate on test set
  4. Select best performing model

## Input/Output Analysis

### Input Data Sources

1. **Price Data**: `./data_processor/sp500_tickers_daily_price_20250712.csv`
   - Contains: OHLCV data, adjusted prices, volume
   - Format: CSV with date, stock identifier, price columns

2. **Stock Selection**: `./result/stock_selected.csv`
   - Contains: Selected stocks for each trade date
   - Format: CSV with trade_date and stock identifiers

3. **Preprocessed Data**: `./output/all_stocks_info.pickle`
   - Contains: Preprocessed stock information
   - Format: Pickle file with dictionary structure

### Output Files

#### Performance Reports (`./results/`)
- `bt_YYYYMMDD_YYYYMMDD_summary.csv`: Key performance metrics
- `bt_YYYYMMDD_YYYYMMDD_equity_curve.csv`: Portfolio value over time
- `bt_YYYYMMDD_YYYYMMDD_turnover.csv`: Trading activity metrics
- `bt_YYYYMMDD_YYYYMMDD_weights_sum.csv`: Portfolio weight validation

#### Model Outputs
- `drl_weight.csv`: Portfolio weights for each trade date
- `./checkpoints/`: Model checkpoints for each trade date
- `./cache/`: Cached feature engineering results

#### Progress Tracking
- `./results/progress.json`: Execution progress for resumption

## Connection with Other Project Components

### 1. Data Processing Pipeline

```
data_processor/ → fundamental_portfolio_drl.py
├── sp500_tickers_daily_price_20250712.csv (price data)
├── stock_selection.py (stock selection logic)
└── outputs/ (preprocessed data)
```

### 2. FinRL Framework Integration

```python
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
```

- **DRLAgent**: Provides RL algorithm implementations
- **StockPortfolioEnv**: Custom environment for portfolio optimization
- **FeatureEngineer**: Technical indicator calculation

### 3. Environment Configuration

The system uses a custom `StockPortfolioEnv` with the following state space:
```python
state_space = stock_dimension + K_EIG + stock_dimension * len(config.INDICATORS)
```

Where:
- `stock_dimension`: Number of stocks
- `K_EIG`: Top-K eigenvalues (default: 10)
- `config.INDICATORS`: Technical indicators per stock

### 4. Model Architecture

The system implements a multi-algorithm approach:

1. **A2C (Advantage Actor-Critic)**: On-policy algorithm
2. **PPO (Proximal Policy Optimization)**: On-policy algorithm  
3. **DDPG (Deep Deterministic Policy Gradient)**: Off-policy algorithm

Each model is trained independently and the best performer is selected for backtesting.

## Execution Flow

### 1. Initialization Phase
```python
# Set random seeds
set_global_seed(42)

# Load and preprocess data
df_price = pd.read_csv("./data_processor/sp500_tickers_daily_price_20250712.csv")
selected_stock = pd.read_csv("./result/stock_selected.csv")
```

### 2. Rolling Window Loop
```python
for idx in range(start_idx, len(trade_date)):
    current_trade_date = trade_date[idx-1]
    
    # 1. Data preparation
    df_ = load_or_build_fe_features(df, p1_stock, earliest_date, trade_date[idx])
    
    # 2. Model training
    a2c_model, ppo_model, ddpg_model, best_model = run_models(...)
    
    # 3. Backtesting
    df_daily_return, df_actions = DRLAgent.DRL_prediction(...)
    
    # 4. Performance calculation
    compute_and_save_performance(...)
```

### 3. Model Training Process (in `rl_model.py`)
```python
def run_models(...):
    # 1. Prepare data
    X_train = prepare_rolling_train(...)
    X_test = prepare_rolling_test(...)
    
    # 2. Create environment
    e_train_gym = StockPortfolioEnv(df=X_train, **env_kwargs)
    
    # 3. Train models
    a2c_model = train_a2c(agent, USE_GPU)
    ppo_model = train_ppo(agent, USE_GPU)
    ddpg_model = train_ddpg(agent, USE_GPU)
    
    # 4. Evaluate and select best
    best_model = evaluate_and_select_best(...)
```

## Key Features

### 1. Robustness Features
- **Progress Tracking**: Resume from interruption
- **Atomic Writes**: Prevent data corruption
- **Cache Management**: Avoid redundant computations
- **Error Handling**: Comprehensive exception handling

### 2. Performance Optimizations
- **GPU Support**: Automatic GPU/CPU detection
- **Checkpoint Management**: Efficient model persistence
- **Memory Management**: Optimized data loading with dtypes

### 3. Validation Features
- **Data Integrity**: Stock coverage validation
- **Weight Validation**: Portfolio weight sum checks
- **Performance Metrics**: Comprehensive backtesting analysis

## Configuration Parameters

### Environment Parameters
```python
env_kwargs = {
    "hmax": 100,                    # Maximum shares per trade
    "initial_amount": 1000000,      # Initial portfolio value
    "transaction_cost_pct": 0.001,  # Transaction costs
    "state_space": state_space,     # State dimension
    "stock_dim": stock_dimension,   # Number of stocks
    "tech_indicator_list": config.INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4,
    "k_eig": K_EIG                  # Top-K eigenvalues
}
```

### Time Window Parameters
```python
testing_window = pd.Timedelta(days=365)      # 1 year testing
max_rolling_window = pd.Timedelta(days=1095) # 3 years training
lookback = 252                               # 1 year lookback for covariance
```

## Dependencies

### Core Dependencies
- `pandas`, `numpy`: Data manipulation
- `torch`: Deep learning framework
- `stable-baselines3`: RL algorithms
- `finrl`: Financial RL framework
- `gymnasium`: RL environment interface

### Optional Dependencies
- `pypfopt`: Portfolio optimization (imported but not used)
- `matplotlib`: Plotting (used in environment)
- `sklearn`: Machine learning utilities

## Usage Example

```python
# Run the complete pipeline
python fundamental_portfolio_drl.py

# The system will:
# 1. Load data and initialize
# 2. Process each trade date in rolling window
# 3. Train models and select best
# 4. Generate backtesting results
# 5. Save all outputs to results/ directory
```

## Output Interpretation

### Performance Metrics
- **Total Return**: Overall portfolio performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Maximum portfolio decline
- **Turnover**: Trading activity level
- **Weight Sum**: Portfolio allocation validation

### Model Selection
The system automatically selects the best performing model based on cumulative returns during the testing window, ensuring optimal performance for backtesting.

## Future Enhancements

1. **Additional Algorithms**: TD3, SAC implementation
2. **Risk Management**: VaR, CVaR constraints
3. **Multi-Asset**: Extend to bonds, commodities
4. **Real-time Trading**: Live market integration
5. **Ensemble Methods**: Combine multiple model predictions 