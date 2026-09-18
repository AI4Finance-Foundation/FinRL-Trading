
<div align="center">

# FinRL-X

### An AI-Native Modular Infrastructure for Quantitative Trading

<img src="https://github.com/user-attachments/assets/80fe89bb-fb09-4267-b29a-76030512f8cf" width="420">

[![Paper](https://img.shields.io/badge/Paper-arXiv_2603.21330-b31b1b?style=for-the-badge)](https://arxiv.org/abs/2603.21330)
&nbsp;
[![PyPI](https://img.shields.io/badge/PyPI-finrl--trading-3775A9?style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/finrl-trading/)

[![Python 3.11](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
![License](https://img.shields.io/github/license/AI4Finance-Foundation/FinRL-Trading.svg?color=brightgreen)
[![Downloads](https://static.pepy.tech/badge/finrl-trading)](https://pepy.tech/project/finrl-trading)
[![Downloads](https://static.pepy.tech/badge/finrl-trading/week)](https://pepy.tech/project/finrl-trading)
[![Join Discord](https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white)](https://discord.gg/trsr8SXpW5)

![](https://img.shields.io/github/issues-raw/AI4Finance-Foundation/FinRL-Trading?label=Issues)
![](https://img.shields.io/github/issues-pr-raw/AI4Finance-Foundation/FinRL-Trading?label=PRs)
![Visitors](https://api.visitorbadge.io/api/VisitorHit?user=AI4Finance-Foundation&repo=FinRL-Trading&countColor=%23B17A)

*A deployment-consistent trading system that unifies data processing, strategy composition, backtesting, and broker execution through a weight-centric interface.*

[Paper](https://arxiv.org/abs/2603.21330) | [Quick Start](#quick-start) | [Strategies](#strategies) | [Results](#results) | [Discord](https://discord.gg/trsr8SXpW5)

</div>

---

## About

**FinRL-X** is a next-generation, **AI-native** quantitative trading infrastructure that redefines how researchers and practitioners build, test, and deploy algorithmic trading strategies. 

Introduced in our paper *"FinRL-X: An AI-Native Modular Infrastructure for Quantitative Trading"* ([arXiv:2603.21330](https://arxiv.org/abs/2603.21330)), FinRL-X succeeds the original [FinRL](https://github.com/AI4Finance-Foundation/FinRL) framework with a fully modernized architecture designed for the LLM and agentic AI era.

> FinRL-X is **not just a library** — it is a full-stack trading platform engineered around modularity, reproducibility, and production-readiness, supporting everything from ML-based stock selection and professional backtesting to live brokerage execution.

At its core is a **weight-centric architecture** — the target portfolio weight vector is the sole interface contract between strategy logic and downstream execution:

$$w_t = \mathcal{R}_t\bigl(\mathcal{T}_t\bigl(\mathcal{A}_t\bigl(\mathcal{S}_t(\mathcal{X}_{\le t})\bigr)\bigr)\bigr)$$

where $\mathcal{S}$ denotes stock selection, $\mathcal{A}$ portfolio allocation, $\mathcal{T}$ timing adjustment, and $\mathcal{R}$ portfolio-level risk overlay. Each transformation is contract-preserving — you can swap any module (e.g. equal-weight $\to$ DRL allocator) without touching the rest of the pipeline, and the same weights flow identically through backtesting and live execution.

---

## Architecture

<div align="center">
  <img src="https://github.com/AI4Finance-Foundation/FinRL-Trading/blob/master/figs/FinRL_X_Framework.png" width="880"/>
</div>

<br/>

| Layer | Role | Components |
|:------|:-----|:-----------|
| **Data** | Unified market data pipeline | FMP, Yahoo Finance, WRDS; optional [Adanos market sentiment](docs/adanos_sentiment_features.md); LLM sentiment preprocessing; SQLite cache |
| **Strategy** | Weight-centric signal generation | Stock selection, portfolio allocation, timing adjustment, risk overlay |
| **Backtest** | Offline evaluation | `bt`-powered engine with multi-benchmark comparison and transaction costs |
| **Execution** | Live/paper trading | Alpaca multi-account integration with pre-trade risk checks |

```
finrl-trading/
├── src/
│   ├── config/                     # ⚙️  Centralized configuration management
│   │   └── settings.py             #     Pydantic-based settings + environment variables
│   ├── data/                       # 🗄️  Data acquisition and processing
│   │   ├── data_fetcher.py         #     Multi-source integration (Yahoo / FMP / WRDS)
│   │   ├── data_processor.py       #     Feature engineering & data cleaning
│   │   └── data_store.py           #     SQLite persistence with caching
│   ├── backtest/                   # 📊  Backtesting engine
│   │   └── backtest_engine.py      #     bt-powered engine with benchmark comparison
│   ├── strategies/                 # 🤖  Trading strategies
│   │   ├── base_strategy.py        #     Abstract strategy framework
│   │   └── ml_strategy.py          #     Random Forest stock selection
│   ├── trading/                    # 💰  Live trading execution
│   │   ├── alpaca_manager.py       #     Alpaca API integration (multi-account)
│   │   ├── trade_executor.py       #     Order management & risk controls
│   │   └── performance_analyzer.py #     Real-time P&L tracking
│   └── main.py                     # 🚀  CLI entry point
├── examples/
│   ├── FinRL_Full_Workflow.ipynb   # 📓  Complete workflow tutorial (start here!)
│   └── README.md
├── data/                           # Runtime data storage (gitignored)
├── logs/                           # Application logs (gitignored)
├── requirements.txt
└── setup.py
```

---

## Strategies

FinRL-X implements three use cases from the paper, each demonstrating different compositions of the weight-centric pipeline.

### Use Case 1 — Portfolio Allocation Paradigms

Compares heterogeneous allocation methods under a unified interface:

| Method | Type | Description |
|:-------|:-----|:------------|
| Equal Weight | Classical | Uniform 1/N allocation |
| Mean-Variance | Classical | Markowitz optimization |
| Minimum Variance | Classical | Minimize portfolio volatility |
| DRL Allocator | Learning | PPO/SAC continuous weight generation |
| KAMA Timing | Signal | Kaufman adaptive trend overlay |

All methods output the same weight vector, making them directly composable with timing and risk overlays.

<div align="center">
  <img src="https://github.com/AI4Finance-Foundation/FinRL-Trading/blob/master/figs/DRL_Timing_Backtest.png" width="900"/>
</div>

### Use Case 2 — Rolling Stock Selection + DRL

Quarterly selection of top-25% NASDAQ-100 stocks via ML fundamental scoring, combined with DRL-based portfolio allocation. Strict no-lookahead semantics prevent data leakage.

### Use Case 3 — Adaptive Multi-Asset Rotation

A research-grade, walk-forward-safe rotation strategy with daily risk monitoring:

| Component | Detail |
|:----------|:-------|
| **Asset Groups** | Growth Tech, Real Assets, Defensive — max 2 active per week |
| **Group Selection** | Information Ratio relative to QQQ benchmark |
| **Intra-Group Ranking** | Residual momentum with robust Z-score exception handling |
| **Market Regime** | Slow regime (26-week trend + VIX) + Fast Risk-Off (3-day shock) |
| **Risk Controls** | Trailing stop-loss, absolute stop-loss, cooldown periods |
| **Rebalance** | Weekly (full) + daily monitoring (fast risk-off, stop-loss adjustments) |

```bash
# Run the adaptive rotation backtest
./deploy.sh --strategy adaptive_rotation --mode backtest --start 2023-01-01 --end 2024-12-31

# Paper trade with Alpaca
./deploy.sh --strategy adaptive_rotation --mode paper --dry-run
```

---

## Results

### Historical Backtest (Jan 2018 – Oct 2025)

<div align="center">
  <img src="https://github.com/AI4Finance-Foundation/FinRL-Trading/blob/master/figs/All_Backtests_v2.png" width="900"/>
</div>

| Metric | Rolling Strategy | Adaptive Rotation | QQQ | SPY |
|:-------|:---:|:---:|:---:|:---:|
| Cumulative Return | 5.98x | 4.80x | 4.02x | 2.80x |
| Annualized Return | 25.85% | 22.32% | 19.56% | 14.14% |
| Annualized Volatility | 27.85% | 20.30% | 24.20% | 19.61% |
| **Sharpe Ratio** | 0.93 | **1.10** | 0.81 | 0.72 |
| Max Drawdown | -38.95% | **-21.46%** | -35.12% | -33.72% |
| **Calmar Ratio** | 0.66 | **1.04** | 0.56 | 0.42 |
| Win Rate | 54.36% | 54.77% | 56.25% | 55.28% |

### Paper Trading (Oct 2025 – Mar 2026)

<div align="center">
  <img src="https://github.com/AI4Finance-Foundation/FinRL-Trading/blob/master/figs/Paper_Trading.png" width="900"/>
</div>

Ensemble of Rolling Selection + Adaptive Rotation deployed on Alpaca paper trading:

| Metric | Strategy | SPY | QQQ |
|:-------|:---:|:---:|:---:|
| Cumulative Return | **1.20x** | 0.97x | 0.95x |
| Total Return | **+19.76%** | -2.51% | -4.79% |
| Annualized Return | **62.16%** | -6.60% | -12.32% |
| Annualized Volatility | 31.75% | 11.96% | 16.79% |
| **Sharpe Ratio** | **1.96** | -0.55 | -0.73 |
| Max Drawdown | -12.22% | -5.35% | -7.88% |
| **Calmar Ratio** | **5.09** | -1.23 | -1.56 |
| Win Rate | **64.89%** | 52.13% | 54.02% |

### Dynamic Sector Rotation

<div align="center">
  <img src="https://github.com/AI4Finance-Foundation/FinRL-Trading/blob/master/figs/Sector_Rotation_Standalone.png" width="900"/>
</div>

The Adaptive Rotation strategy dynamically shifts capital across three asset groups — **Growth Tech**, **Real Assets**, and **Defensive** — based on market regime signals. During risk-on regimes, the portfolio tilts toward high-momentum growth and commodity plays; when regime detection flags risk-off or fast risk-off conditions, capital rotates into bonds and utilities with an automatic cash buffer. Weekly rebalancing is complemented by daily stop-loss and fast risk-off monitoring, enabling rapid de-risking without waiting for the next scheduled rebalance.

---

## Quick Start

### Option A — One-Command Deploy

`deploy.sh` handles everything automatically: dependency check, data download, and strategy execution.

```bash
git clone https://github.com/AI4Finance-Foundation/FinRL-Trading.git
cd FinRL-Trading

# Backtest (downloads data + runs strategy)
./deploy.sh --strategy adaptive_rotation --mode backtest

# Custom date range
./deploy.sh --strategy adaptive_rotation --mode backtest --start 2020-01-01 --end 2025-12-31

# Single date signal
./deploy.sh --strategy adaptive_rotation --mode single --date 2024-12-31

# Paper trading (requires Alpaca credentials in .env)
./deploy.sh --strategy adaptive_rotation --mode paper --dry-run   # preview
./deploy.sh --strategy adaptive_rotation --mode paper              # execute

# See all options
./deploy.sh --help
```

### Option B — Manual Setup with venv

```bash
# 1. Clone
git clone https://github.com/AI4Finance-Foundation/FinRL-Trading.git
cd FinRL-Trading

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Configure API keys for paper trading
cp .env.example .env
# Edit .env — set APCA_API_KEY, APCA_API_SECRET, etc.
```

#### Running Strategies via Python

```bash
# Backtest — Adaptive Rotation (2023-01-01 to 2024-12-31)
python src/strategies/run_adaptive_rotation_strategy.py \
    --config src/strategies/AdaptiveRotationConf_v1.2.1.yaml \
    --backtest --start 2023-01-01 --end 2024-12-31

# Single date signal
python src/strategies/run_adaptive_rotation_strategy.py \
    --config src/strategies/AdaptiveRotationConf_v1.2.1.yaml \
    --date 2024-12-31

# Full workflow tutorial (Jupyter)
jupyter notebook examples/FinRL_Full_Workflow.ipynb
```

> **Note:** Data files (`{SYMBOL}_daily.csv`) must exist under `data/fmp_daily/` before running.
> `deploy.sh` downloads them automatically; for manual setup, either run `deploy.sh --strategy adaptive_rotation --mode backtest` once, or prepare CSV files with columns `date,open,high,low,close,volume`.

### Configuration

```bash
cp .env.example .env
```

```bash
# Alpaca (required for paper/live trading)
APCA_API_KEY=your_key
APCA_API_SECRET=your_secret
APCA_BASE_URL=https://paper-api.alpaca.markets

# Data source (optional; Yahoo Finance is the free default)
FMP_API_KEY=your_fmp_key
```

### Python API

```python
# Data
from src.data.data_fetcher import get_data_manager
manager = get_data_manager()
prices = manager.get_price_data(['AAPL', 'MSFT'], '2020-01-01', '2024-12-31')

# Strategy
from src.strategies.ml_strategy import MLStockSelectorStrategy
strategy = MLStockSelectorStrategy(config)
result = strategy.generate_weights(data)

# Backtest
from src.backtest.backtest_engine import BacktestEngine, BacktestConfig
engine = BacktestEngine(BacktestConfig(start_date='2020-01-01', end_date='2024-12-31'))
result = engine.run_backtest("My Strategy", weights, prices)

# Live trade
from src.trading.alpaca_manager import create_alpaca_account_from_env, AlpacaManager
alpaca = AlpacaManager([create_alpaca_account_from_env()])
alpaca.execute_portfolio_rebalance(target_weights={'AAPL': 0.3, 'MSFT': 0.7})
```

---

## Evolution from FinRL

| | FinRL (2020) | FinRL-X (2026) |
|:---|:---|:---|
| **Paradigm** | DRL-only | AI-Native (ML + DRL + LLM-ready) |
| **Architecture** | Coupled monolith | Decoupled modular layers |
| **Interface** | Gym state/action spaces | Weight-centric contract |
| **Data** | 14 manual processors | Auto-select: Yahoo / FMP / WRDS |
| **Backtesting** | Hand-rolled loops | `bt` engine + multi-benchmark |
| **Live Trading** | Basic Alpaca | Multi-account + risk controls |
| **Config** | `config.py` | Pydantic + `.env` |
| **Paper** | [arXiv:2011.09607](https://arxiv.org/abs/2011.09607) | [arXiv:2603.21330](https://arxiv.org/abs/2603.21330) |

### Migration from FinRL

```
FinRL                                →  FinRL-X
─────────────────────────────────────────────────────────────
finrl/meta/data_processor.py         →  src/data/data_fetcher.py
finrl/train.py                       →  strategy.generate_weights()
finrl/trade.py                       →  TradeExecutor.execute_portfolio_rebalance()
config.py + config_tickers.py        →  src/config/settings.py (Pydantic + .env)
gym.Env subclassing                  →  BaseStrategy.generate_weights()
```

---

## Comparison with Existing Platforms

| Feature | FinRL-X | [Qlib](https://github.com/microsoft/qlib) | [TradingAgents](https://github.com/TauricResearch/TradingAgents) | [Zipline](https://github.com/quantopian/zipline)/[Backtrader](https://github.com/mementum/backtrader) | [QuantConnect Lean](https://github.com/QuantConnect/Lean) |
|:--------|:-------:|:----:|:-------------:|:------------------:|:-----------------:|
| Primary Orientation | End-to-End System | ML Research | Agent-Based Trading | Backtesting | End-to-End Platform |
| Broker Integration | Yes | - | - | - | Yes |
| Deployment-Consistent Interface | Yes | - | - | - | Partial |
| Reinforcement Learning Support | Yes | Limited | Yes | - | Partial |
| Modular Strategy Pipeline | Yes | - | - | - | Partial |
| Portfolio-Level Risk Overlay | Yes | - | - | - | Partial |
| Open Source License | Apache 2.0 | MIT | Apache 2.0 | Apache 2.0 | Apache 2.0 |

---

## Contributing

```bash
git checkout -b feature/your-feature
pip install -r requirements.txt
# make changes, add tests
git commit -m "Add: your feature"
git push origin feature/your-feature
# open a Pull Request
```

Adding a custom strategy:

```python
from src.strategies.base_strategy import BaseStrategy, StrategyConfig, StrategyResult

class MyStrategy(BaseStrategy):
    def generate_weights(self, data, **kwargs) -> StrategyResult:
        # your alpha logic — return portfolio weights
        pass
```

---

## Citation

```bibtex
@inproceedings{yang2026finrlx,
  title     = {FinRL-X: An AI-Native Modular Infrastructure for Quantitative Trading},
  author    = {Yang, Hongyang and Zhang, Boyu and She, Yang and Liao, Xinyu and Zhang, Xiaoli},
  booktitle = {Proceedings of the 2nd International Workshop on Decision Making and Optimization in Financial Technologies (DMO-FinTech)},
  year      = {2026},
  note      = {Workshop at PAKDD 2026}
}
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## Disclaimer

This software is for **educational and research purposes only**. Not financial advice. Always consult qualified professionals before making investment decisions. Past performance does not guarantee future results.

---

<div align="center">

**[AI4Finance Foundation](https://github.com/AI4Finance-Foundation)**

<img src="https://github.com/AI4Finance-Foundation/FinGPT/assets/31713746/e0371951-1ce1-488e-aa25-0992dafcc139" width="200"/>

</div>
