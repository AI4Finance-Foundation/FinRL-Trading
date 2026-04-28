"""
Backtest Engine Module - Powered by bt library
==============================================

Implements portfolio backtesting functionality using bt library:
- Professional backtesting framework
- Portfolio return calculation
- Risk metrics computation
- Performance analysis
- Benchmark comparison
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import bt

logger = logging.getLogger(__name__)

# Import data source manager
# add project root to path
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
print(f"project_root: {project_root}")
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))
from src.data.data_fetcher import fetch_price_data



@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    start_date: str
    end_date: str
    rebalance_freq: str = 'Q'  # Q: quarterly, M: monthly, W: weekly
    initial_capital: float = 1000000.0
    transaction_cost: float = 0.001  # 0.1% per trade
    benchmark_tickers: List[str] = None
    # Optional: a bt.core.CostModel instance (e.g. bt.AlmgrenChrissCostModel(...)).
    # When set, the engine passes it to bt.Backtest as `commissions=` together
    # with auto-built volume / volatility frames. When None, the legacy
    # transaction_cost flat-rate callable path is used (existing behaviour).
    cost_model: Optional[Any] = None
    # Rolling window for realised-volatility estimation (log-return stdev).
    volatility_window: int = 20
    # Forwarded to bt.Backtest. Default True preserves existing behaviour.
    # Set False for large-notional books — bt's integer-share allocator can
    # fail to converge with non-zero commissions at >~$100M (the bisection
    # in bt.core.SecurityBase.allocate stops approaching the target outlay).
    # bt's own cost-model example uses fractional shares for the same reason.
    integer_positions: bool = True

    def __post_init__(self):
        if self.benchmark_tickers is None:
            self.benchmark_tickers = ['SPY', 'QQQ']


@dataclass
class BacktestResult:
    """Results from backtesting."""
    strategy_name: str
    portfolio_returns: pd.Series
    portfolio_values: pd.Series
    weights_history: pd.DataFrame
    trades: pd.DataFrame
    metrics: Dict[str, float]
    benchmark_returns: Dict[str, pd.Series] = None
    annualized_return: float = 0.0
    benchmark_annualized: Dict[str, float] = None
    benchmark_metrics: Dict[str, Dict[str, float]] = None  # New field for full benchmark metrics

    def __post_init__(self):
        if self.benchmark_returns is None:
            self.benchmark_returns = {}
        if self.benchmark_annualized is None:
            self.benchmark_annualized = {}
        if self.benchmark_metrics is None:
            self.benchmark_metrics = {}

    def to_metrics_dataframe(self) -> pd.DataFrame:
        """Combine all metrics into a DataFrame for comparison."""
        # Start with main strategy metrics
        all_metrics = {self.strategy_name: self.metrics}
        
        # Add benchmark metrics
        all_metrics.update(self.benchmark_metrics)
        
        # Create DataFrame with strategies as index
        df = pd.DataFrame.from_dict(all_metrics, orient='index')
        
        # Add annualized return as a column for consistency
        annualized = {self.strategy_name: self.annualized_return}
        annualized.update(self.benchmark_annualized)
        df['annualized_return'] = pd.Series(annualized)
        
        # Sort columns alphabetically or by importance
        return df[sorted(df.columns)]


class BacktestEngine:
    """Professional backtesting engine powered by bt library."""

    def __init__(self, config: BacktestConfig):
        """
        Initialize backtest engine.

        Args:
            config: Backtest configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        # Initialize data source manager
        # self.data_manager = get_data_manager()

    def run_backtest(self, strategy_name: str, price_data: pd.DataFrame,
                     weight_signals: pd.DataFrame) -> BacktestResult:
        """
        Run backtest for a strategy using bt library.

        Args:
            strategy_name: Name of the strategy
            price_data: Historical price data (wide format: dates index, tickers columns)
            weight_signals: Strategy weight signals (dates index, tickers columns)

        Returns:
            BacktestResult with performance metrics
        """
        self.logger.info(f"Running backtest for {strategy_name} using bt library")

        try:
            # Prepare price data for bt
            price_data_clean = self._prepare_price_data_for_bt(price_data)
            self.logger.info(f"Price data prepared for bt: shape {price_data_clean.shape}")

            # Align and forward-fill weight signals to trading calendar
            ws = weight_signals.copy()
            if not isinstance(ws.index, pd.DatetimeIndex):
                ws.index = pd.to_datetime(ws.index)
            ws = ws.sort_index()
            # Keep only tickers present in price data
            common_cols = [c for c in ws.columns if c in price_data_clean.columns]
            ws = ws[common_cols]
            # Reindex to full trading index and forward-fill weights
            ws = ws.reindex(price_data_clean.index).ffill()
            # Fill any remaining NaNs with 0 (before first signal)
            ws = ws.fillna(0.0)
            # Normalize rows where sum > 0
            row_sum = ws.sum(axis=1)
            nonzero = row_sum > 0
            if nonzero.any():
                ws.loc[nonzero] = ws.loc[nonzero].div(row_sum[nonzero], axis=0)

            # Create bt strategy from aligned weights
            bt_strategy = self._create_bt_strategy(strategy_name, ws)

            # Build the bt.Backtest. When a CostModel is configured, route it
            # through the new bt cost-model API together with volume +
            # volatility frames; otherwise keep the original flat-rate path.
            backtest = self._build_bt_backtest(
                bt_strategy, price_data_clean, price_data
            )

            # Run backtest
            result = bt.run(backtest)

            # Extract results
            strategy_result = result[strategy_name]

            # Create our result format
            portfolio_values = strategy_result.prices
            portfolio_returns = portfolio_values.pct_change().dropna()

            # Scale to match initial capital if needed
            if len(portfolio_values) > 0:
                initial_value = portfolio_values.iloc[0]
                if abs(initial_value - self.config.initial_capital) > 1:
                    scale_factor = self.config.initial_capital / initial_value
                    portfolio_values = portfolio_values * scale_factor

            # Optional: basic sanity logging for trades/weights
            try:
                first_date = portfolio_values.index.min()
                last_date = portfolio_values.index.max()
                self.logger.info(f"Backtest period: {first_date.date()} -> {last_date.date()}")
                # bt does not expose trades API directly; prices change implies positions exist
                self.logger.info(f"First value: {float(portfolio_values.iloc[0]):.2f}, Last value: {float(portfolio_values.iloc[-1]):.2f}")
            except Exception:
                pass

            # Calculate comprehensive metrics
            metrics = self._calculate_comprehensive_metrics(strategy_result, portfolio_returns)

            # Calculate annualized return
            annualized_return = metrics.get('annual_return', 0.0)

            # Get benchmark returns and annualized
            # benchmark_returns = self._get_benchmark_returns(price_data_clean)
            benchmark_metrics = self._get_benchmark_metrics(price_data_clean)
            benchmark_returns = {bm: metrics.get('total_return', 0.0) for bm, metrics in benchmark_metrics.items()}
            benchmark_annualized = {bm: metrics.get('annual_return', 0.0) for bm, metrics in benchmark_metrics.items()}

            # Create result object
            result_obj = BacktestResult(
                strategy_name=strategy_name,
                portfolio_returns=portfolio_returns,
                portfolio_values=portfolio_values,
                weights_history=pd.DataFrame(),  # bt handles this internally
                trades=pd.DataFrame(),  # bt handles this internally
                metrics=metrics,
                benchmark_returns=benchmark_returns,
                annualized_return=annualized_return,
                benchmark_annualized=benchmark_annualized,
                benchmark_metrics=benchmark_metrics  # Add this
            )

            self.logger.info(f"Backtest completed. Annualized return: {annualized_return:.2%}")
            for bm, ann in benchmark_annualized.items():
                self.logger.info(f"Benchmark {bm} annualized: {ann:.2%}")

            return result_obj

        except Exception as e:
            self.logger.error(f"Error in bt backtest: {e}")
            raise

    def _prepare_price_data_for_bt(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare price data for bt library format."""
        # Assume input is long format with 'date', 'tic', 'adj_close'
        if 'tic' in price_data.columns and 'adj_close' in price_data.columns:
            price_data = price_data.pivot(index='datadate', columns='tic', values='adj_close')
        price_data.index = pd.to_datetime(price_data.index)
        price_data = price_data.ffill().dropna(how='all')
        return price_data

    def _build_volume_volatility(self, price_data_long: pd.DataFrame,
                                  price_data_wide: pd.DataFrame
                                  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Build volume and rolling-volatility frames aligned to price_data_wide.

        - Volume comes from the `cshtrd` column produced by data_fetcher
          (already in the SQLite price_data table as the `volume` column).
        - Volatility is the rolling stdev of log-returns over
          `config.volatility_window` business days, computed from `adj_close`.
        Both frames are reindexed onto `price_data_wide.index` and
        `price_data_wide.columns` so they line up with the price frame bt sees.
        """
        idx = price_data_wide.index
        cols = price_data_wide.columns

        # ---- volume ----
        missing = [c for c in ('cshtrd', 'tic', 'datadate')
                   if c not in price_data_long.columns]
        if missing:
            raise ValueError(
                f"cost_model is configured but price_data is missing required "
                f"column(s) {missing}; nonlinear cost models need per-ticker "
                f"daily volume (cshtrd) to compute participation rate."
            )
        vol_long = price_data_long[['datadate', 'tic', 'cshtrd']].copy()
        vol_long['datadate'] = pd.to_datetime(vol_long['datadate'])
        volume_df = (vol_long
                     .pivot_table(index='datadate', columns='tic',
                                  values='cshtrd', aggfunc='last')
                     .reindex(index=idx, columns=cols)
                     .ffill())

        # Cost models divide by V; avoid zeros/NaNs.
        volume_df = volume_df.replace(0, np.nan).ffill().fillna(1.0)

        # ---- volatility ----
        win = max(int(self.config.volatility_window), 2)
        log_ret = np.log(price_data_wide / price_data_wide.shift()).fillna(0.0)
        vol_df = (log_ret.rolling(win, min_periods=1)
                          .std()
                          .reindex(index=idx, columns=cols))
        # Cost models multiply by sigma; first row will be NaN → 0 fee that day.
        vol_df = vol_df.fillna(0.0)

        return volume_df, vol_df

    def _build_bt_backtest(self, bt_strategy, price_data_wide: pd.DataFrame,
                           price_data_long: pd.DataFrame) -> "bt.Backtest":
        """Construct a bt.Backtest, switching to the CostModel API when configured."""
        cost_model = getattr(self.config, 'cost_model', None)
        integer_positions = getattr(self.config, 'integer_positions', True)
        if cost_model is not None and isinstance(cost_model, bt.core.CostModel):
            volume_df, volatility_df = self._build_volume_volatility(
                price_data_long, price_data_wide
            )
            self.logger.info(
                f"Using bt CostModel: {type(cost_model).__name__} "
                f"(volume {volume_df.shape}, volatility {volatility_df.shape})"
            )
            return bt.Backtest(
                bt_strategy,
                price_data_wide,
                initial_capital=self.config.initial_capital,
                commissions=cost_model,
                volume=volume_df,
                volatility=volatility_df,
                integer_positions=integer_positions,
            )
        # Legacy fixed-fee path (unchanged default behaviour).
        return bt.Backtest(
            bt_strategy,
            price_data_wide,
            initial_capital=self.config.initial_capital,
            commissions=lambda q, p: abs(q) * p * self.config.transaction_cost,
            integer_positions=integer_positions,
        )

    def _create_bt_strategy(self, strategy_name: str, weight_signals: pd.DataFrame) -> bt.Strategy:
        """Create bt strategy from weight signals."""
        # Ensure weight_signals has datetime index
        if not isinstance(weight_signals.index, pd.DatetimeIndex):
            weight_signals.index = pd.to_datetime(weight_signals.index)
        
        # Normalize weights to sum to 1 per row
        weight_signals = weight_signals.div(weight_signals.sum(axis=1), axis=0).fillna(0)
        tw = weight_signals.sort_index()

        strategy = bt.Strategy(
            strategy_name,
            [
                bt.algos.RunAfterDate(tw.index.min()),
                bt.algos.RunOnDate(*tw.index.tolist()),
                bt.algos.SelectThese(list(tw.columns)),
                bt.algos.WeighTarget(tw),
                bt.algos.Rebalance()
            ]
        )
        return strategy

    def _calculate_comprehensive_metrics(self, bt_result, portfolio_returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive metrics from bt result."""
        metrics = {}

        try:
            # Basic metrics from bt
            metrics['total_return'] = bt_result.total_return
            metrics['annual_return'] = bt_result.cagr
            metrics['annual_volatility'] = bt_result.yearly_vol
            metrics['max_drawdown'] = bt_result.max_drawdown
            metrics['sharpe_ratio'] = bt_result.yearly_sharpe
            metrics['sortino_ratio'] = bt_result.yearly_sortino
            metrics['skewness'] = bt_result.yearly_skew
            metrics['kurtosis'] = bt_result.yearly_kurt

        except Exception as e:
            self.logger.error(f"Error calculating comprehensive metrics: {e}")
            # Fallback to basic metrics
            metrics = self._calculate_basic_metrics(portfolio_returns, bt_result.prices)

        # 当样本期不足一年或 bt 返回 NaN 时，使用日频回填并补充月频指标
        try:
            prices_series = bt_result.prices
            metrics = self._backfill_short_period_metrics(metrics, portfolio_returns, prices_series)
        except Exception:
            pass

        return metrics

    def _backfill_short_period_metrics(self, metrics: Dict[str, float],
                                        returns: pd.Series,
                                        portfolio_values: pd.Series) -> Dict[str, float]:
        """Backfill NaN yearly metrics using daily-based estimates and add monthly metrics."""
        if returns is None or len(returns) == 0:
            return metrics

        # 年化回填（基于日频）
        num_days = len(returns)
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1 if len(portfolio_values) > 0 else returns.add(1).prod() - 1
        est_annual_return = (1 + total_return) ** (252 / max(num_days, 1)) - 1
        daily_vol = float(returns.std()) if len(returns) > 1 else 0.0
        annual_vol = daily_vol * np.sqrt(252)

        def is_nan(x: Any) -> bool:
            try:
                return x is None or (isinstance(x, float) and (np.isnan(x) or not np.isfinite(x)))
            except Exception:
                return False

        if is_nan(metrics.get('annual_return')):
            metrics['annual_return'] = est_annual_return
        if is_nan(metrics.get('annual_volatility')):
            metrics['annual_volatility'] = annual_vol
        if is_nan(metrics.get('max_drawdown')):
            metrics['max_drawdown'] = self._calculate_max_drawdown(portfolio_values)

        # 夏普与索提诺（基于日频回填）
        if is_nan(metrics.get('sharpe_ratio')):
            if annual_vol > 0:
                metrics['sharpe_ratio'] = metrics['annual_return'] / annual_vol
            else:
                metrics['sharpe_ratio'] = 0.0

        if is_nan(metrics.get('sortino_ratio')):
            downside = returns[returns < 0]
            downside_std = float(downside.std()) if len(downside) > 1 else 0.0
            annual_downside = downside_std * np.sqrt(252)
            if annual_downside > 0:
                metrics['sortino_ratio'] = metrics['annual_return'] / annual_downside
            else:
                metrics['sortino_ratio'] = 0.0

        # 偏度与峰度（基于日频回填）
        if is_nan(metrics.get('skewness')):
            metrics['skewness'] = float(returns.skew()) if len(returns) > 1 else 0.0
        if is_nan(metrics.get('kurtosis')):
            metrics['kurtosis'] = float(returns.kurtosis()) if len(returns) > 1 else 0.0

        # 计算月频指标以供短期样本参考
        monthly_metrics = self._calculate_monthly_metrics(returns)
        metrics.update(monthly_metrics)

        return metrics

    def _calculate_monthly_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Compute monthly-level metrics as complementary stats for short samples."""
        if returns is None or len(returns) == 0:
            return {
                'monthly_return': 0.0,
                'monthly_volatility': 0.0,
                'monthly_sharpe': 0.0,
                'monthly_sortino': 0.0
            }

        # 将日收益聚合为月收益
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        if monthly_returns.empty:
            return {
                'monthly_return': 0.0,
                'monthly_volatility': 0.0,
                'monthly_sharpe': 0.0,
                'monthly_sortino': 0.0
            }

        monthly_vol = float(monthly_returns.std()) if len(monthly_returns) > 1 else 0.0
        # 月度“Sharpe/Sortino”供参考（简单均值/波动）
        monthly_mean = float(monthly_returns.mean())
        monthly_sharpe = (monthly_mean / monthly_vol) if monthly_vol > 0 else 0.0
        monthly_downside = monthly_returns[monthly_returns < 0]
        monthly_downside_std = float(monthly_downside.std()) if len(monthly_downside) > 1 else 0.0
        monthly_sortino = (monthly_mean / monthly_downside_std) if monthly_downside_std > 0 else 0.0

        return {
            'monthly_return': monthly_mean,
            'monthly_volatility': monthly_vol,
            'monthly_sharpe': monthly_sharpe,
            'monthly_sortino': monthly_sortino
        }

    def _calculate_basic_metrics(self, returns: pd.Series, portfolio_values: pd.Series) -> Dict[str, float]:
        """Calculate basic metrics as fallback."""
        if len(returns) == 0:
            return {}

        # Basic metrics
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        num_days = len(returns)
        annual_return = (1 + total_return) ** (252 / num_days) - 1

        annual_volatility = returns.std() * np.sqrt(252)

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': annual_return / annual_volatility if annual_volatility > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(portfolio_values),
            'skewness': returns.skew() if len(returns) > 1 else 0,
            'kurtosis': returns.kurtosis() if len(returns) > 1 else 0
        }

    def _calculate_max_drawdown(self, portfolio_values: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(portfolio_values) == 0:
            return 0.0

        cumulative = portfolio_values / portfolio_values.iloc[0]
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()


    def _get_benchmark_metrics(self, price_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Compute benchmark metrics using bt library with buy-and-hold strategy."""
        benchmark_metrics = {}
        start_date = price_data.index.min().strftime('%Y-%m-%d')
        end_date = price_data.index.max().strftime('%Y-%m-%d')

        for ticker in self.config.benchmark_tickers:
            try:
                # Fetch benchmark price data
                bm_data = fetch_price_data([ticker], start_date, end_date)
                if bm_data.empty:
                    self.logger.warning(f"No data for benchmark {ticker}, skipping")
                    continue

                # Prepare price data (single column)
                bm_prices = bm_data.pivot(index='datadate', columns='tic', values='adj_close')
                bm_prices.index = pd.to_datetime(bm_prices.index)
                bm_prices = bm_prices.ffill().dropna(how='all')

                if bm_prices.empty or ticker not in bm_prices.columns:
                    self.logger.warning(f"No valid price data for {ticker}")
                    continue

                bm_prices = bm_prices[[ticker]]  # Single column DataFrame

                # Create buy-and-hold strategy
                bh_strategy = bt.Strategy(
                    f'{ticker}_BuyHold',
                    [
                        bt.algos.RunOnce(),
                        bt.algos.SelectAll(),
                        bt.algos.WeighEqually(),
                        bt.algos.Rebalance()
                    ]
                )

                # Create and run backtest (use the same cost-model path the
                # main strategy uses, so benchmark and strategy stay
                # apples-to-apples).
                backtest = self._build_bt_backtest(bh_strategy, bm_prices, bm_data)
                result = bt.run(backtest)

                # Extract results
                strategy_result = result[f'{ticker}_BuyHold']
                portfolio_values = strategy_result.prices
                portfolio_returns = portfolio_values.pct_change().dropna()

                # Scale if needed
                if len(portfolio_values) > 0:
                    initial_value = portfolio_values.iloc[0]
                    if abs(initial_value - self.config.initial_capital) > 1:
                        scale_factor = self.config.initial_capital / initial_value
                        portfolio_values = portfolio_values * scale_factor

                # Calculate metrics
                metrics = self._calculate_comprehensive_metrics(strategy_result, portfolio_returns)
                benchmark_metrics[ticker] = metrics

                self.logger.info(f"Computed bt metrics for benchmark {ticker}")

            except Exception as e:
                self.logger.error(f"Error computing bt metrics for {ticker}: {e}")
                benchmark_metrics[ticker] = {}

        return benchmark_metrics

    def plot_results(self, result: BacktestResult, save_path: Optional[str] = None):
        """Plot backtest results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Portfolio value
        result.portfolio_values.plot(ax=axes[0, 0], title='Portfolio Value')
        axes[0, 0].set_ylabel('Value ($)')

        # Returns distribution
        result.portfolio_returns.plot.hist(ax=axes[0, 1], bins=50, title='Return Distribution')

        # Drawdown
        cumulative = (1 + result.portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        drawdown.plot(ax=axes[1, 0], title='Drawdown', color='red')

        # Rolling Sharpe ratio
        rolling_sharpe = result.portfolio_returns.rolling(252).apply(
            lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
        )
        rolling_sharpe.plot(ax=axes[1, 1], title='Rolling Sharpe Ratio (252-day)')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Backtest plot saved to {save_path}")
        else:
            plt.show()

        plt.close()


def run_multiple_backtests(strategies: List, price_data: pd.DataFrame,
                          weight_signals: List[pd.DataFrame],
                          config: BacktestConfig) -> Dict[str, BacktestResult]:
    """
    Run backtests for multiple strategies.

    Args:
        strategies: List of strategy instances
        price_data: Historical price data
        weight_signals: List of weight signals for each strategy
        config: Backtest configuration

    Returns:
        Dictionary of backtest results by strategy name
    """
    engine = BacktestEngine(config)
    results = {}

    for strategy, signals in zip(strategies, weight_signals):
        try:
            result = engine.run_backtest(strategy.config.name, price_data, signals)
            results[strategy.config.name] = result
            logger.info(f"Completed backtest for {strategy.config.name}")
        except Exception as e:
            logger.error(f"Backtest failed for {strategy.config.name}: {e}")
            continue

    return results


if __name__ == "__main__":
    # 示例：读取选股权重文件（gvkey 作为 ticker，weight 为权重，date 为目标调仓日）
    logging.basicConfig(level=logging.INFO)

    # 1) 读取权重 csv（示例路径：data/ml_weights_single.csv）
    weights_csv_path = Path(project_root) / 'data' / 'ml_weights.csv'
    weights_raw = pd.read_csv(weights_csv_path)
    if 'date' not in weights_raw.columns or 'gvkey' not in weights_raw.columns or 'weight' not in weights_raw.columns:
        raise ValueError("权重文件缺少必要列：date / gvkey / weight")

    # 统一类型与排序
    weights_raw['date'] = pd.to_datetime(weights_raw['date'])
    weights_raw['gvkey'] = weights_raw['gvkey'].astype(str)
    weights_raw = weights_raw.sort_values(['date', 'gvkey'])

    # 2) 生成调仓信号矩阵（行：调仓日，列：ticker，值：权重）
    weight_signals = (
        weights_raw
        .pivot_table(index='date', columns='gvkey', values='weight', aggfunc='sum')
        .fillna(0.0)
        .sort_index()
    )

    # 过滤掉全 0 的调仓日（无持仓）
    if not weight_signals.empty:
        weight_signals = weight_signals.loc[(weight_signals.sum(axis=1) > 0.0)]

    if weight_signals.empty:
        raise ValueError("权重矩阵为空，请检查权重文件是否包含有效数据。")

    # 3) 配置回测区间（自动覆盖为权重日期的最小/最大范围）
    # 回测日期重新配置，向前或向后延申一个季度
    # start_date = (weight_signals.index.min() - pd.Timedelta(days=90)).strftime('%Y-%m-%d')
    # end_date = (weight_signals.index.max() + pd.Timedelta(days=90)).strftime('%Y-%m-%d')
    start_date = weight_signals.index.min().strftime('%Y-%m-%d')
    end_date = weight_signals.index.max().strftime('%Y-%m-%d')

    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        rebalance_freq='Q',
        initial_capital=1000000.0
    )

    # 4) 拉取价格数据（使用权重里的所有 ticker）
    # data_manager = get_data_manager()
    tickers = weight_signals.columns.tolist()
    price_data = fetch_price_data(tickers, config.start_date, config.end_date)

    # 5) 对齐调仓日到交易日，并裁剪列到有价格数据的股票集合
    engine = BacktestEngine(config)
    price_data_bt = engine._prepare_price_data_for_bt(price_data)

    # 仅保留价格可用的列
    common_cols = [c for c in weight_signals.columns if c in price_data_bt.columns]
    if not common_cols:
        raise ValueError("价格数据与权重中的股票列没有交集，请检查 ticker 是否匹配数据源。")
    weight_signals = weight_signals[common_cols]

    # 将目标调仓日对齐到下一个可交易日（bfill）
    trading_index = price_data_bt.index
    pos = trading_index.get_indexer(weight_signals.index, method='bfill')
    mask = pos != -1
    weight_signals = weight_signals.iloc[mask]
    weight_signals.index = trading_index[pos[mask]]

    # 逐个调仓日，移除当日价格为 NaN 的票，再归一化
    if not weight_signals.empty:
        aligned = []
        for dt, row in weight_signals.iterrows():
            if dt not in price_data_bt.index:
                continue
            prices_today = price_data_bt.loc[dt, row.index]
            valid_cols = prices_today.dropna().index.tolist()
            if len(valid_cols) == 0:
                continue
            row_valid = row[valid_cols]
            s = row_valid.sum()
            if s <= 0:
                continue
            aligned.append((dt, (row_valid / s)))
        if aligned:
            weight_signals = pd.DataFrame({dt: vec for dt, vec in aligned}).T.sort_index()
            weight_signals.index.name = None
        else:
            weight_signals = pd.DataFrame()

    # 删除在该调仓日没有任何可交易票的行，并行内归一化
    if not weight_signals.empty:
        # 去掉全 0 行
        weight_signals = weight_signals.loc[(weight_signals.sum(axis=1) > 0.0)]
        # 再次按行归一化，确保权重和为 1
        weight_signals = weight_signals.div(weight_signals.sum(axis=1), axis=0).fillna(0)

    if weight_signals.empty:
        raise ValueError("对齐交易日后权重矩阵为空，可能所有调仓日都非交易日或票无价格数据。")

    print(f"有效调仓日数量: {len(weight_signals)}，有效股票数: {len(weight_signals.columns)}")

    # 6) 运行回测
    result = engine.run_backtest("ML Weights Strategy", price_data, weight_signals)

    # 7) 输出结果
    print(f"组合年化收益: {result.annualized_return:.2%}")
    for bm, ann in (result.benchmark_annualized or {}).items():
        print(f"基准 {bm} 年化收益: {ann:.2%}")

    metrics_df = result.to_metrics_dataframe()
    print("\n指标对比表：")
    print(metrics_df)
