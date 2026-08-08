'''Backtest engine module for portfolio backtesting using the bt library.'''

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
import argparse

logger = logging.getLogger(__name__)

from src.data.data_fetcher import fetch_price_data


@dataclass
class BacktestConfig:
    """Configuration for backtesting engine."""
    start_date: str
    end_date: str
    rebalance_freq: str = 'Q'
    initial_capital: float = 1000000.0
    transaction_cost: float = 0.001
    benchmark_tickers: List[str] = None
    cost_model: Optional[Any] = None
    volatility_window: int = 20
    integer_positions: bool = True

    def __post_init__(self):
        self.benchmark_tickers = self.benchmark_tickers or []


@dataclass
class BacktestResult:
    """Result of a backtest."""
    strategy_name: str
    portfolio_returns: pd.Series
    portfolio_values: pd.Series
    weights_history: pd.DataFrame
    trades: pd.DataFrame
    metrics: Dict[str, float]
    benchmark_returns: Dict[str, pd.Series] = None
    annualized_return: float = 0.0
    benchmark_annualized: Dict[str, float] = None
    benchmark_metrics: Dict[str, Dict[str, float]] = None

    def __post_init__(self):
        if self.benchmark_returns is None:
            self.benchmark_returns = {}
        if self.benchmark_annualized is None:
            self.benchmark_annualized = {}
        if self.benchmark_metrics is None:
            self.benchmark_metrics = {}

    def to_metrics_dataframe(self) -> pd.DataFrame:
        all_metrics = {self.strategy_name: self.metrics}
        all_metrics.update(self.benchmark_metrics)
        df = pd.DataFrame.from_dict(all_metrics, orient='index')
        annualized = {self.strategy_name: self.annualized_return}
        annualized.update(self.benchmark_annualized)
        df['annualized_return'] = pd.Series(annualized)
        return df[sorted(df.columns)]


class BacktestEngine:

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def run_backtest(self, strategy_name: str, price_data: pd.DataFrame,
                     weight_signals: pd.DataFrame) -> BacktestResult:
        self.logger.info(f"Running backtest for {strategy_name} using bt library")
        try:
            # Prepare price data
            price_data_clean = self._prepare_price_data_for_bt(price_data)
            self.logger.info(f"Price data prepared: shape {price_data_clean.shape}")

            # Align weight signals
            ws = weight_signals.copy()
            if not isinstance(ws.index, pd.DatetimeIndex):
                ws.index = pd.to_datetime(ws.index)
            ws = ws.sort_index()
            common_cols = [c for c in ws.columns if c in price_data_clean.columns]
            ws = ws[common_cols]
            ws = ws.reindex(price_data_clean.index).ffill()
            ws = ws.fillna(0.0)
            row_sum = ws.sum(axis=1)
            nonzero = row_sum > 0
            if nonzero.any():
                ws.loc[nonzero] = ws.loc[nonzero].div(row_sum[nonzero], axis=0)

            # Create strategy
            bt_strategy = self._create_bt_strategy(strategy_name, ws)

            # Build backtest
            backtest = self._build_bt_backtest(bt_strategy, price_data_clean, price_data)

            # Run
            result = bt.run(backtest)
            strategy_result = result[strategy_name]

            # Extract portfolio values and returns
            portfolio_values = strategy_result.prices
            portfolio_returns = portfolio_values.pct_change().dropna()

            # Scale to initial capital
            if len(portfolio_values) > 0:
                initial_value = portfolio_values.iloc[0]
                if abs(initial_value - self.config.initial_capital) > 1:
                    scale_factor = self.config.initial_capital / initial_value
                    portfolio_values = portfolio_values * scale_factor

            # Calculate metrics
            metrics = self._calculate_comprehensive_metrics(strategy_result, portfolio_returns)
            annualized_return = metrics.get('annual_return', 0.0)

            # Benchmark metrics
            benchmark_metrics = self._get_benchmark_metrics(price_data_clean)
            benchmark_returns = {}
            for bm, bm_metrics in benchmark_metrics.items():
                # Generate a return series for this benchmark
                try:
                    bm_data = fetch_price_data([bm], price_data.index.min().strftime('%Y-%m-%d'), price_data.index.max().strftime('%Y-%m-%d'))
                    if not bm_data.empty:
                        bm_prices = bm_data.pivot(index='datadate', columns='tic', values='adj_close')
                        bm_prices.index = pd.to_datetime(bm_prices.index)
                        bm_prices = bm_prices.ffill().dropna(how='all')
                        if bm in bm_prices.columns:
                            bm_series = bm_prices[bm].pct_change().dropna()
                            benchmark_returns[bm] = bm_series
                        else:
                            benchmark_returns[bm] = pd.Series(dtype=float)
                    else:
                        benchmark_returns[bm] = pd.Series(dtype=float)
                except Exception:
                    benchmark_returns[bm] = pd.Series(dtype=float)
            benchmark_annualized = {bm: metrics.get('annual_return', 0.0) for bm, metrics in benchmark_metrics.items()}

            result_obj = BacktestResult(
                strategy_name=strategy_name,
                portfolio_returns=portfolio_returns,
                portfolio_values=portfolio_values,
                weights_history=pd.DataFrame(),
                trades=pd.DataFrame(),
                metrics=metrics,
                benchmark_returns=benchmark_returns,
                annualized_return=annualized_return,
                benchmark_annualized=benchmark_annualized,
                benchmark_metrics=benchmark_metrics
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
        # Copy to avoid modifying original
        data = price_data.copy()
        # Check required columns
        if 'tic' not in data.columns or 'adj_close' not in data.columns:
            # Assume already wide format
            if isinstance(data.index, pd.DatetimeIndex):
                return data.ffill().dropna(how='all')
            else:
                raise ValueError("Price data must have 'tic' and 'adj_close' columns or be wide with datetime index")
        # Normalize date column: accept date or datadate
        if 'datadate' not in data.columns and 'date' in data.columns:
            data = data.rename(columns={'date': 'datadate'})
        if 'datadate' not in data.columns:
            raise ValueError("Missing date column: expected 'datadate' or 'date'")
        # Pivot
        price_data_wide = data.pivot(index='datadate', columns='tic', values='adj_close')
        price_data_wide.index = pd.to_datetime(price_data_wide.index)
        price_data_wide = price_data_wide.ffill().dropna(how='all')
        return price_data_wide

    def _build_volume_volatility(self, price_data_long: pd.DataFrame,
                                  price_data_wide: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Build volume and rolling-volatility frames aligned to price_data_wide."""
        idx = price_data_wide.index
        cols = price_data_wide.columns

        # Volume from cshtrd column
        missing = [c for c in ('cshtrd', 'tic', 'datadate') if c not in price_data_long.columns]
        if missing:
            raise ValueError(
                f"cost_model is configured but price_data is missing required columns {missing}; "
                "nonlinear cost models need per-ticker daily volume (cshtrd)."
            )
        vol_long = price_data_long[['datadate', 'tic', 'cshtrd']].copy()
        vol_long['datadate'] = pd.to_datetime(vol_long['datadate'])
        volume_df = (vol_long
                     .pivot_table(index='datadate', columns='tic', values='cshtrd', aggfunc='last')
                     .reindex(index=idx, columns=cols)
                     .ffill())
        volume_df = volume_df.replace(0, np.nan).ffill().fillna(1.0)

        # Volatility
        win = max(int(self.config.volatility_window), 2)
        log_ret = np.log(price_data_wide / price_data_wide.shift()).fillna(0.0)
        vol_df = (log_ret.rolling(win, min_periods=1).std()
                  .reindex(index=idx, columns=cols))
        vol_df = vol_df.fillna(0.0)

        return volume_df, vol_df

    def _build_bt_backtest(self, bt_strategy, price_data_wide: pd.DataFrame,
                           price_data_long: pd.DataFrame) -> "bt.Backtest":
        cost_model = getattr(self.config, 'cost_model', None)
        integer_positions = getattr(self.config, 'integer_positions', True)
        if cost_model is not None and isinstance(cost_model, bt.core.CostModel):
            volume_df, volatility_df = self._build_volume_volatility(price_data_long, price_data_wide)
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
                integer_positions=integer_positions
            )
        # Legacy fixed-fee path
        return bt.Backtest(
            bt_strategy,
            price_data_wide,
            initial_capital=self.config.initial_capital,
            commissions=lambda q, p: abs(q) * p * self.config.transaction_cost,
            integer_positions=integer_positions
        )

    def _create_bt_strategy(self, strategy_name: str, weight_signals: pd.DataFrame) -> bt.Strategy:
        if not isinstance(weight_signals.index, pd.DatetimeIndex):
            weight_signals.index = pd.to_datetime(weight_signals.index)
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
        metrics = {}
        try:
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
            metrics = self._calculate_basic_metrics(portfolio_returns, bt_result.prices)

        metrics = self._backfill_short_period_metrics(metrics, portfolio_returns, bt_result.prices)
        return metrics

    def _backfill_short_period_metrics(self, metrics: Dict[str, float],
                                        returns: pd.Series,
                                        portfolio_values: pd.Series) -> Dict[str, float]:
        if returns is None or len(returns) == 0:
            return metrics

        num_days = len(returns)
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1 if len(portfolio_values) > 0 else returns.add(1).prod() - 1
        est_annual_return = (1 + total_return) ** (252 / max(num_days, 1)) - 1
        daily_vol = float(returns.std()) if len(returns) > 1 else 0.0
        annual_vol = daily_vol * np.sqrt(252)

        def is_nan(x):
            try:
                return x is None or (isinstance(x, float) and (np.isnan(x) or not np.isfinite(x)))
            except:
                return False

        if is_nan(metrics.get('annual_return')):
            metrics['annual_return'] = est_annual_return
        if is_nan(metrics.get('annual_volatility')):
            metrics['annual_volatility'] = annual_vol
        if is_nan(metrics.get('max_drawdown')):
            metrics['max_drawdown'] = self._calculate_max_drawdown(portfolio_values)

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

        if is_nan(metrics.get('skewness')):
            metrics['skewness'] = float(returns.skew()) if len(returns) > 1 else 0.0
        if is_nan(metrics.get('kurtosis')):
            metrics['kurtosis'] = float(returns.kurtosis()) if len(returns) > 1 else 0.0

        monthly_metrics = self._calculate_monthly_metrics(returns)
        metrics.update(monthly_metrics)
        return metrics

    def _calculate_monthly_metrics(self, returns: pd.Series) -> Dict[str, float]:
        if returns.empty:
            return {
                'monthly_return': 0.0,
                'monthly_volatility': 0.0,
                'monthly_sharpe': 0.0,
                'monthly_sortino': 0.0
            }
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        if monthly_returns.empty:
            return {
                'monthly_return': 0.0,
                'monthly_volatility': 0.0,
                'monthly_sharpe': 0.0,
                'monthly_sortino': 0.0
            }
        monthly_vol = float(monthly_returns.std()) if len(monthly_returns) > 1 else 0.0
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
        if len(returns) == 0:
            return {}
        num_days = len(returns)
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        annual_return = (1 + total_return) ** (252 / num_days) - 1
        annual_volatility = returns.std() * np.sqrt(252)
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': annual_return / annual_volatility if annual_volatility > 0 else 0.0,
            'max_drawdown': self._calculate_max_drawdown(portfolio_values),
            'skewness': returns.skew() if len(returns) > 1 else 0.0,
            'kurtosis': returns.kurtosis() if len(returns) > 1 else 0.0
        }

    def _calculate_max_drawdown(self, portfolio_values: pd.Series) -> float:
        if len(portfolio_values) == 0:
            return 0.0
        cumulative = portfolio_values / portfolio_values.iloc[0]
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def _get_benchmark_metrics(self, price_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        if not self.config.benchmark_tickers:
            return {}
        benchmark_metrics = {}
        start_date = price_data.index.min().strftime('%Y-%m-%d')
        end_date = price_data.index.max().strftime('%Y-%m-%d')

        for ticker in self.config.benchmark_tickers:
            try:
                bm_data = fetch_price_data([ticker], start_date, end_date)
                if bm_data.empty:
                    self.logger.warning(f"No data for benchmark {ticker}, skipping")
                    continue
                bm_prices = bm_data.pivot(index='datadate', columns='tic', values='adj_close')
                bm_prices.index = pd.to_datetime(bm_prices.index)
                bm_prices = bm_prices.ffill().dropna(how='all')
                if bm_prices.empty or ticker not in bm_prices.columns:
                    self.logger.warning(f"No valid price data for {ticker}")
                    continue
                bm_prices = bm_prices[[ticker]]

                bh_strategy = bt.Strategy(
                    f'{ticker}_BuyHold',
                    [
                        bt.algos.RunOnce(),
                        bt.algos.SelectAll(),
                        bt.algos.WeighEqually(),
                        bt.algos.Rebalance()
                    ]
                )
                backtest = self._build_bt_backtest(bh_strategy, bm_prices, bm_data)
                result = bt.run(backtest)
                strategy_result = result[f'{ticker}_BuyHold']
                bm_returns = strategy_result.prices.pct_change().dropna()
                metrics = self._calculate_comprehensive_metrics(strategy_result, bm_returns)
                benchmark_metrics[ticker] = metrics
                self.logger.info(f"Computed bt metrics for benchmark {ticker}")
            except Exception as e:
                self.logger.error(f"Error computing bt metrics for {ticker}: {e}")
                benchmark_metrics[ticker] = {}
        return benchmark_metrics

    def plot_results(self, result: BacktestResult, save_path: Optional[str] = None):
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
    engine = BacktestEngine(config)
    results = {}
    for strategy, signals in zip(strategies, weight_signals):
        try:
            result = engine.run_backtest(strategy.config.name, price_data, signals)
            results[strategy.config.name] = result
            logger.info(f"Completed backtest for {strategy.config.name}")
        except Exception as e:
            logger.error(f"Backtest failed for {strategy.config.name}: {e}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run portfolio backtest")
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--start', type=str, default=None, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None, help='End date (YYYY-MM-DD)')
    parser.add_argument('--symbols', type=str, nargs='+', default=None, help='Symbols to backtest')
    args = parser.parse_args()

    start_date = args.start or '2023-01-01'
    end_date = args.end or '2024-01-01'

    print(f"Running backtest from {start_date} to {end_date}")

    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date
    )
    engine = BacktestEngine(config)

    if args.symbols:
        symbols = args.symbols
        print(f"Symbols: {', '.join(symbols)}")
        price_data = fetch_price_data(symbols, start_date, end_date)
        if not price_data.empty:
            weights = pd.DataFrame(1.0 / len(symbols), index=price_data.index, columns=symbols)
            result = engine.run_backtest("Equal Weight Strategy", price_data, weights)
        else:
            print("No price data available. Try different symbols or dates.")
            return
    else:
        print("Using default config (no --symbols specified)")
        symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT']
        price_data = fetch_price_data(symbols, start_date, end_date)
        if price_data.empty:
            print("No price data available. Try with --symbols flag.")
            return
        weights = pd.DataFrame(1.0 / len(symbols), index=price_data.index, columns=symbols)
        result = engine.run_backtest("Equal Weight Strategy", price_data, weights)

    print(f"\n{'='*50}")
    print(f"BACKTEST RESULTS")
    print(f"{'='*50}")
    print(f"  Total Return: {result.metrics.get('total_return', 0):.2%}")
    print(f"  Annualized Return: {result.annualized_return:.2%}")
    print(f"  Sharpe Ratio: {result.metrics.get('sharpe_ratio', 0):.2f}")
    print(f"  Max Drawdown: {result.metrics.get('max_drawdown', 0):.2%}")
    print(f"  Volatility: {result.metrics.get('annual_volatility', 0):.2%}")

    if result.benchmark_metrics:
        print(f"\nBenchmark Comparison:")
        metrics_df = result.to_metrics_dataframe()
        print(metrics_df.to_string())


if __name__ == "__main__":
    main()