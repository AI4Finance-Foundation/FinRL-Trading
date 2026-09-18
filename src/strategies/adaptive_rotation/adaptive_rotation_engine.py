"""
Adaptive Rotation Engine
========================

Main strategy engine that orchestrates all modules.

High-Level API:
    engine = AdaptiveRotationEngine(config)
    weights, audit_log = engine.run(price_data, as_of_date)

Author: Adaptive Rotation Strategy Team
Version: 1.2.1
"""

import pandas as pd
import json
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .config_loader import AdaptiveRotationConfig, load_config
from .data_preprocessor import DataPreprocessor, get_data_as_of_date
from .market_regime import detect_market_regime
from .group_strength import analyze_group_strength, compute_group_returns
from .intra_group_ranking import IntraGroupRanker
from .exception_framework import ExceptionDetector
from .risk_manager import RiskManager, PositionState
from .portfolio_builder import PortfolioBuilder, PortfolioWeights


# ============================================================================
# Audit Log
# ============================================================================

@dataclass
class AuditLog:
    """Complete decision trace for explainability"""
    date: pd.Timestamp
    
    # Input summary
    data_summary: Dict[str, Any]
    
    # Market regime
    regime: Dict[str, Any]
    
    # Signal generation
    group_strength: Dict[str, Any]
    asset_ranking: Dict[str, Any]
    
    # Exception framework
    exceptions: Dict[str, Any]
    
    # Portfolio construction
    portfolio: Dict[str, Any]
    
    # Risk management
    risk: Dict[str, Any]
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_json(self, path: str, indent: int = 2):
        """Save as JSON file"""
        output = self.to_dict()
        
        # Convert Timestamp to string
        if isinstance(output['date'], pd.Timestamp):
            output['date'] = output['date'].strftime('%Y-%m-%d')
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(output, f, indent=indent, default=str)


# ============================================================================
# Main Engine
# ============================================================================

class AdaptiveRotationEngine:
    """
    Main strategy engine
    
    Examples:
        >>> # Initialize
        >>> engine = AdaptiveRotationEngine(
        ...     config_path="src/strategies/AdaptiveRotationConf_v1.2.1.yaml"
        ... )
        >>> 
        >>> # Run for a single decision point
        >>> weights, audit_log = engine.run(
        ...     price_data=weekly_data,
        ...     as_of_date="2024-02-01"
        ... )
        >>> 
        >>> # Access results
        >>> print(f"Invested: {weights.get_invested_weight():.1%}")
        >>> print(f"Cash: {weights.cash_weight:.1%}")
        >>> print(f"Regime: {weights.regime_state}")
    """
    
    def __init__(
        self,
        config: Optional[Union[str, Path, AdaptiveRotationConfig]] = None,
        config_path: Optional[Union[str, Path]] = None,
        data_preprocessor: Optional[DataPreprocessor] = None,
    ):
        """
        Initialize strategy engine
        
        Args:
            config: Config object, path to YAML, or None
            config_path: Alternative way to specify config path
            data_preprocessor: Optional DataPreprocessor instance for accessing daily data
        
        Note:
            Provide either `config` or `config_path`, not both
        """
        # Load configuration
        if config_path is not None:
            self.config = load_config(str(config_path))
        elif isinstance(config, (str, Path)):
            self.config = load_config(str(config))
        elif isinstance(config, AdaptiveRotationConfig):
            self.config = config
        else:
            raise ValueError("Must provide either config or config_path")
        
        # Store data preprocessor for accessing daily data
        self.data_preprocessor = data_preprocessor
        
        # Initialize sub-modules
        self._init_modules()
        
        # State tracking
        self._current_positions: Dict[str, PositionState] = {}
    
    def _init_modules(self):
        """Initialize all strategy modules"""
        # Regime detection
        self.regime_config = self.config.market_regime
        self.fast_config = self.config.fast_risk_off
        
        # Ranking
        self.intra_group_ranker = IntraGroupRanker(
            lookback_weeks=self.config.ranking.zscore_window or 12,
            robust=self.config.ranking.robust
        )
        
        # Exception detection
        self.exception_detector = ExceptionDetector.from_config(self.config)
        
        # Risk management
        self.risk_manager = RiskManager.from_config(self.config)
        
        # Portfolio construction
        self.portfolio_builder = PortfolioBuilder(self.config)
    
    def run(
        self,
        price_data: Union[pd.DataFrame, Dict[str, pd.Series]],
        as_of_date: Union[str, pd.Timestamp],
        current_positions: Optional[Dict[str, PositionState]] = None,
        mode: str = "backtest"
    ) -> Tuple[PortfolioWeights, AuditLog]:
        """
        Generate portfolio weights for a single decision point
        
        Args:
            price_data: Price data (DataFrame or Dict of Series)
                       If DataFrame: columns = [date, symbol, close, ...]
                       If Dict: {symbol: price_series}
            as_of_date: Decision date
            current_positions: Current positions for stop-loss tracking
            mode: "backtest" or "live"
        
        Returns:
            Tuple of (PortfolioWeights, AuditLog)
        
        Examples:
            >>> weights, audit = engine.run(
            ...     price_data=weekly_data,
            ...     as_of_date="2024-02-01"
            ... )
        """
        # Convert date
        if isinstance(as_of_date, str):
            as_of_date = pd.Timestamp(as_of_date)
        
        # Convert price_data to dict format if needed
        if isinstance(price_data, pd.DataFrame):
            prices_dict = self._dataframe_to_dict(price_data)
        else:
            prices_dict = price_data
        
        # Get point-in-time data
        prices_as_of = get_data_as_of_date(prices_dict, as_of_date)
        
        # Initialize audit sections
        audit_data = {
            'data_summary': {
                'symbols_available': len(prices_as_of),
                'date_range': (
                    min(s.index[0] for s in prices_as_of.values() if len(s) > 0).strftime('%Y-%m-%d'),
                    max(s.index[-1] for s in prices_as_of.values() if len(s) > 0).strftime('%Y-%m-%d')
                ),
                'as_of_date': as_of_date.strftime('%Y-%m-%d')
            }
        }
        
        # === STEP 1: Market Regime Detection ===
        regime_result = self._detect_regime(prices_as_of, as_of_date)
        audit_data['regime'] = self._audit_regime(regime_result)
        
        # === STEP 2: Group Strength Analysis ===
        group_strength = self._analyze_group_strength(prices_as_of, as_of_date)
        audit_data['group_strength'] = self._audit_group_strength(group_strength)
        
        # === STEP 3: Intra-Group Ranking ===
        group_rankings = self._rank_assets_in_groups(
            prices_as_of,
            group_strength.active_groups,
            as_of_date
        )
        audit_data['asset_ranking'] = self._audit_asset_ranking(group_rankings)
        
        # === STEP 4: Exception Detection ===
        exceptions = self._detect_exceptions(
            group_rankings, 
            prices_as_of,  # Pass price data for strong signal rule
            as_of_date
        )
        audit_data['exceptions'] = self._audit_exceptions(exceptions)
        
        # === STEP 5: Risk Management ===
        if current_positions is None:
            current_positions = self._current_positions
        
        risk_result = self._check_stops(current_positions, prices_as_of, as_of_date)
        audit_data['risk'] = self._audit_risk(risk_result)
        
        # Update positions
        self._current_positions = risk_result.updated_positions
        
        # === STEP 6: Portfolio Construction ===
        portfolio_result = self.portfolio_builder.build(
            regime_result=regime_result,
            group_strength=group_strength,
            group_rankings=group_rankings,
            exceptions=exceptions,
            as_of_date=as_of_date
        )
        
        audit_data['portfolio'] = self._audit_portfolio(portfolio_result)
        audit_data['metadata'] = {
            'mode': mode,
            'strategy_version': self.config.strategy.version,
            'config_hash': self.config.compute_config_hash()
        }
        
        # Create audit log
        audit_log = AuditLog(
            date=as_of_date,
            **audit_data
        )
        
        return portfolio_result.portfolio, audit_log
    
    def _dataframe_to_dict(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Convert DataFrame to dict of Series"""
        # Assume df has columns: date, symbol, close
        prices_dict = {}
        
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].copy()
            symbol_data = symbol_data.set_index('date')
            prices_dict[symbol] = symbol_data['close']
        
        return prices_dict
    
    def _detect_regime(
        self,
        prices: Dict[str, pd.Series],
        as_of_date: pd.Timestamp
    ):
        """Run market regime detection"""
        # Weekly data for slow regime detection
        spx_weekly = prices.get("^GSPC", pd.Series())
        vix_weekly = prices.get("^VIX", pd.Series())
        qqq_weekly = prices.get("QQQ", pd.Series())
        
        # Try to get daily data for Fast Risk-Off detection
        spx_daily = spx_weekly  # Default fallback
        qqq_daily = qqq_weekly  # Default fallback
        vix_daily = vix_weekly  # Default fallback
        
        if self.data_preprocessor is not None:
            try:
                daily_data = self.data_preprocessor.get_daily_data_as_of(
                    as_of_date=as_of_date,
                    symbols=["^GSPC", "^VIX", "QQQ"]
                )
                spx_daily = daily_data.get("^GSPC", spx_weekly)
                vix_daily = daily_data.get("^VIX", vix_weekly)
                qqq_daily = daily_data.get("QQQ", qqq_weekly)
            except Exception as e:
                # If daily data not available, fall back to weekly
                print(f"Warning: Could not load daily data, using weekly as proxy: {e}")
        
        return detect_market_regime(
            spx_weekly=spx_weekly,
            vix_weekly=vix_weekly,
            spx_daily=spx_daily,
            qqq_daily=qqq_daily,
            vix_daily=vix_daily,
            as_of_date=as_of_date,
            config=self.config
        )
    
    def _analyze_group_strength(
        self,
        prices: Dict[str, pd.Series],
        as_of_date: pd.Timestamp
    ):
        """Run group strength analysis"""
        return analyze_group_strength(prices, self.config, as_of_date)
    
    def _rank_assets_in_groups(
        self,
        prices: Dict[str, pd.Series],
        active_groups: List[str],
        as_of_date: pd.Timestamp
    ):
        """Rank assets within each active group"""
        # Compute group returns
        group_returns_dict = {}
        group_members_dict = {}
    
        for group_name in active_groups:
            if group_name not in self.config.asset_groups:
                continue
    
            group_config = self.config.asset_groups[group_name]
            group_ret = compute_group_returns(
                prices,
                group_config.symbols,
                lookback_periods=None  # Use all available
            )
            group_returns_dict[group_name] = group_ret
            group_members_dict[group_name] = self.config.get_group_symbols(group_name)
    
        # Rank assets
        return self.intra_group_ranker.rank_multiple_groups(
            asset_returns_dict={sym: prices[sym].pct_change() for sym in prices.keys()},
            group_returns_dict=group_returns_dict,
            group_members_dict=group_members_dict,
            active_groups=active_groups,
            as_of_date=as_of_date,
            top_n=self.config.ranking.top_n_per_group
        )
    
    def _detect_exceptions(
        self,
        group_rankings: Dict,
        prices_as_of: Dict[str, pd.Series],
        as_of_date: pd.Timestamp
    ):
        """Detect exception assets"""
        # Collect Z-scores from all groups
        all_zscores = {}
        
        for group_name, ranking in group_rankings.items():
            for symbol, score in ranking.asset_scores.items():
                if score.is_valid:
                    # For exception detection, we need historical Z-scores
                    # For now, use latest Z-score as a series
                    # TODO: In full implementation, maintain Z-score history
                    all_zscores[symbol] = pd.Series(
                        [score.zscore],
                        index=[as_of_date]
                    )
        
        # Get benchmark prices for strong signal rule
        # Use QQQ as benchmark (configurable via config in future)
        benchmark_symbol = "QQQ"
        benchmark_prices = prices_as_of.get(benchmark_symbol, None)
        
        return self.exception_detector.detect_exceptions(
            all_zscores,
            as_of_date,
            asset_prices=prices_as_of,
            benchmark_prices=benchmark_prices,
        )
    
    def _check_stops(
        self,
        positions: Dict[str, PositionState],
        prices: Dict[str, pd.Series],
        as_of_date: pd.Timestamp
    ):
        """Check stop-loss conditions"""
        # Get current prices
        current_prices = {}
        for symbol, price_series in prices.items():
            if len(price_series) > 0:
                current_prices[symbol] = float(price_series.iloc[-1])
        
        return self.risk_manager.check_stops(
            positions,
            current_prices,
            as_of_date
        )
    
    # === Audit Log Helpers ===
    
    def _audit_regime(self, regime_result) -> Dict:
        """Create regime audit section"""
        return {
            'slow_regime': {
                'state': regime_result.slow_regime.state.value,
                'group_cap': regime_result.slow_regime.group_cap,
                'cash_floor': regime_result.slow_regime.cash_floor,
                'is_persistent': regime_result.slow_regime.is_persistent,
                'signals': {
                    'trend_deterioration': regime_result.slow_regime.signals.trend_deterioration,
                    'drawdown_stress': regime_result.slow_regime.signals.drawdown_stress,
                    'volatility_stress': regime_result.slow_regime.signals.volatility_stress,
                    'risk_score': regime_result.slow_regime.signals.risk_score,
                }
            },
            'fast_risk_off': {
                'is_active': regime_result.fast_risk_off.is_active,
                'days_remaining': regime_result.fast_risk_off.days_remaining,
                'price_shock': regime_result.fast_risk_off.price_shock,
                'volatility_shock': regime_result.fast_risk_off.volatility_shock,
            },
            'effective': {
                'state': regime_result.effective_state,
                'group_cap': regime_result.effective_group_cap,
                'cash_floor': regime_result.effective_cash_floor,
            }
        }
    
    def _audit_group_strength(self, group_strength) -> Dict:
        """Create group strength audit section"""
        return {
            'ranked_groups': group_strength.ranked_groups,
            'active_groups': group_strength.active_groups,
            'metrics': {
                name: {
                    'information_ratio': metrics.information_ratio,
                    'excess_return': metrics.excess_return,
                    'is_valid': metrics.is_valid,
                    'rank': metrics.rank
                }
                for name, metrics in group_strength.groups.items()
            }
        }
    
    def _audit_asset_ranking(self, group_rankings: Dict) -> Dict:
        """Create asset ranking audit section"""
        return {
            group_name: {
                'ranked_assets': ranking.ranked_assets,
                'top_n_selected': ranking.top_n_assets,
                'scores': {
                    symbol: {
                        'zscore': score.zscore,
                        'residual_momentum': score.residual_momentum,
                        'rank': score.rank
                    }
                    for symbol, score in ranking.asset_scores.items()
                    if score.is_valid
                }
            }
            for group_name, ranking in group_rankings.items()
        }
    
    def _audit_exceptions(self, exceptions) -> Dict:
        """Create exceptions audit section"""
        return {
            'qualified_count': len(exceptions.exceptions),
            'qualified_symbols': exceptions.get_qualified_symbols(),
            'candidates': {
                symbol: {
                    'trigger_count': exc.trigger_count,
                    'qualifies': exc.qualifies,
                    'latest_zscore': exc.latest_zscore,
                    'trigger_dates': exc.trigger_dates,
                    # Strong signal info
                    'strong_signal_qualified': exc.strong_signal_qualified,
                    'strong_signal_return': exc.strong_signal_return,
                    'strong_signal_benchmark_return': exc.strong_signal_benchmark_return,
                    'strong_signal_reason': exc.strong_signal_reason,
                }
                for symbol, exc in exceptions.candidates.items()  # Record ALL candidates
            }
        }
    
    def _audit_risk(self, risk_result) -> Dict:
        """Create risk audit section"""
        return {
            'stops_triggered': len(risk_result.triggered_stops),
            'stopped_symbols': risk_result.get_stopped_symbols(),
            'cooldowns_active': {
                sym: date.strftime('%Y-%m-%d')
                for sym, date in risk_result.cooldowns_active.items()
            },
            'stop_details': [
                {
                    'symbol': stop.symbol,
                    'type': stop.trigger_type,
                    'loss_from_entry': stop.loss_from_entry_pct,
                    'loss_from_peak': stop.loss_from_peak_pct
                }
                for stop in risk_result.triggered_stops
            ]
        }
    
    def _audit_portfolio(self, portfolio_result) -> Dict:
        """Create portfolio audit section"""
        portfolio = portfolio_result.portfolio
        
        return {
            'weights': portfolio.weights,
            'cash_weight': portfolio.cash_weight,
            'invested_weight': portfolio.get_invested_weight(),
            'active_groups': portfolio.active_groups,
            'exception_symbols': portfolio.exception_symbols,
            'regime_state': portfolio.regime_state,
            'risk_budget': portfolio.risk_budget,
            'group_budgets': portfolio.group_budgets,
            'constraints': portfolio_result.constraints_applied,
            'warnings': portfolio_result.warnings
        }
    
    def get_current_positions(self) -> Dict[str, PositionState]:
        """Get current position states"""
        return self._current_positions.copy()
    
    def get_config(self) -> AdaptiveRotationConfig:
        """Get strategy configuration"""
        return self.config
    
    @staticmethod
    def export_weights_to_dataframe(
        results: List[Dict],
        all_symbols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Export portfolio weights to a DataFrame for analysis
        
        Args:
            results: List of backtest results containing weights and audit logs
            all_symbols: Optional list of all symbols to include as columns.
                        If None, will auto-detect from results.
        
        Returns:
            DataFrame with columns: date, symbol1, symbol2, ..., cash, regime
        
        Examples:
            >>> results = engine.run_backtest(...)
            >>> df = AdaptiveRotationEngine.export_weights_to_dataframe(results)
            >>> df.to_csv('portfolio_weights.csv', index=False)
        """
        # Auto-detect all symbols if not provided
        if all_symbols is None:
            all_symbols_set = set()
            for r in results:
                all_symbols_set.update(r['weights'].weights.keys())
            all_symbols = sorted(list(all_symbols_set))
        
        # Build rows
        rows = []
        for r in results:
            row = {'date': r['date'].strftime('%Y-%m-%d')}
            
            # Add weight for each symbol (0 if not present)
            weights_dict = r['weights'].weights
            for symbol in all_symbols:
                row[symbol] = weights_dict.get(symbol, 0.0)
            
            # Add cash and regime
            row['cash'] = r['weights'].cash_weight
            row['regime'] = r['weights'].regime_state
            
            rows.append(row)
        
        # Create DataFrame with specific column order
        columns = ['date'] + all_symbols + ['cash', 'regime']
        df = pd.DataFrame(rows, columns=columns)
        
        return df


if __name__ == "__main__":
    """Quick test of engine"""
    
    print("Testing Adaptive Rotation Engine")
    print("=" * 60)
    
    # Initialize engine
    print("\n[Test] Initializing engine...")
    engine = AdaptiveRotationEngine(
        config_path="src/strategies/AdaptiveRotationConf_v1.2.1.yaml"
    )
    print("  ✓ Engine initialized")
    
    # Create minimal test data
    dates = pd.date_range("2023-01-01", "2024-12-31", freq="W-FRI")
    
    prices = {}
    
    # Required symbols
    for sym in ["^GSPC", "^VIX", "QQQ"]:
        prices[sym] = pd.Series(
            100 + pd.Series(range(len(dates))).values * 0.5,
            index=dates
        )
    
    # Sample assets
    for sym in ["AAPL", "MSFT", "NVDA", "XOM", "CVX"]:
        prices[sym] = pd.Series(
            100 + pd.Series(range(len(dates))).values * 0.3,
            index=dates
        )
    
    print("\n[Test] Running strategy...")
    test_date = pd.Timestamp("2024-06-30")
    
    try:
        weights, audit = engine.run(
            price_data=prices,
            as_of_date=test_date
        )
        
        print(f"  ✓ Strategy run complete")
        print(f"\n  Portfolio:")
        print(f"    Invested: {weights.get_invested_weight():.1%}")
        print(f"    Cash: {weights.cash_weight:.1%}")
        print(f"    Regime: {weights.regime_state}")
        print(f"    Active Groups: {weights.active_groups}")
        print(f"    Positions: {len(weights.weights)}")
        
        print(f"\n  Audit Log:")
        print(f"    Regime: {audit.regime['effective']['state']}")
        print(f"    Groups Ranked: {len(audit.group_strength['ranked_groups'])}")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("[PASS] Engine test complete!")
