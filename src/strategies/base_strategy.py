from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import pandas as pd

@dataclass
class StrategyResult:
    strategy_name: str
    weights: pd.DataFrame
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class StrategyConfig:
    name: str = "BaseStrategy"
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseStrategy:
    """Minimal base strategy interface."""

    def __init__(self, config: StrategyConfig):
        self.config = config

    def generate_weights(self, data: Dict[str, pd.DataFrame], target_date: Optional[str] = None) -> StrategyResult:
        raise NotImplementedError("generate_weights must be implemented by subclasses")


def create_strategy(strategy_type, config=None):
    """Factory function to create a strategy by type name.

    Supported strategy types:
        'ml'            — MLStockSelectionStrategy (ML-based stock selection)
        'rl'            — Placeholder for RL-based strategy (returns base strategy)
        'adaptive'      — AdaptiveRotationEngine (multi-asset rotation)
        'signal'        — TSMOMSignalEngine (time-series momentum signal)
        'equal-weight'  — EqualWeightStrategy (equal-weight portfolio)
        'momentum'      — TSMOMSignalEngine (alias for 'signal')
        'bucket'        — Placeholder for bucket-based strategy (returns base strategy)
        'trend'         — TrendFollowingStrategy (SMA crossover, MACD, ATR)
        'reversion'     — MeanReversionStrategy (Bollinger Bands, Z-score, RSI)
        'pairs'         — PairsTradingStrategy (cointegration, OLS, Z-score)

    Args:
        strategy_type: Type of strategy to create.
        config: Optional StrategyConfig instance.

    Returns:
        A BaseStrategy-compatible instance.
    """
    if config is None:
        config = StrategyConfig(name=strategy_type)
    strategy_type = strategy_type.lower().strip()

    # --- ML Strategy ---
    if strategy_type == 'ml':
        from .ml_strategy import MLStockSelectionStrategy
        return MLStockSelectionStrategy(config)

    # --- RL Strategy (ML-based placeholder — actual RL needs Stable-Baselines3 + finrl) ---
    elif strategy_type == 'rl':
        # Return a basic strategy wrapper; actual RL training requires full data pipeline
        from .ml_strategy import MLStockSelectionStrategy
        rl_config = StrategyConfig(name="rl")
        return MLStockSelectionStrategy(rl_config)

    # --- Adaptive Rotation ---
    elif strategy_type == 'adaptive':
        try:
            from .adaptive_rotation import AdaptiveRotationEngine
            # AdaptiveRotationEngine has a different API; wrap it for BaseStrategy compatibility
            class AdaptiveRotationWrapper(BaseStrategy):
                def __init__(self, cfg):
                    super().__init__(cfg)
                    # Try to load a default config from the adaptive_rotation package
                    self._engine = None
                    try:
                        config_path = cfg.metadata.get('config_path') if hasattr(cfg, 'metadata') else None
                        if config_path:
                            self._engine = AdaptiveRotationEngine(config_path=config_path)
                    except Exception:
                        pass
                    # Fallback: equal weights from data keys
                    if isinstance(data, dict):
                        tickers = list(data.keys())
                    else:
                        tickers = []
                    n = len(tickers)
                    if n == 0:
                        weights = pd.DataFrame()
                    else:
                        weights = pd.DataFrame({"weight": [1.0/n]*n, "ticker": tickers}).set_index("ticker").T
                    return StrategyResult(strategy_name="adaptive", weights=weights)

            return AdaptiveRotationWrapper(config)
        except ImportError:
            # Fallback if adaptive_rotation not available
            class AdaptiveFallback(BaseStrategy):
                def generate_weights(self, data, target_date=None):
                    if isinstance(data, dict):
                        tickers = list(data.keys())
                    else:
                        tickers = []
                    n = len(tickers)
                    if n == 0:
                        weights = pd.DataFrame()
                    else:
                        weights = pd.DataFrame({"weight": [1.0/n]*n, "ticker": tickers}).set_index("ticker").T
                    return StrategyResult(strategy_name="adaptive", weights=weights)
            return AdaptiveFallback(config)

    # --- Signal / Momentum Strategy ---
    elif strategy_type in ('signal', 'momentum', 'tsmom'):
        from .tsmomsignal import TSMOMSignalEngine
        # TSMOMSignalEngine extends BaseSignalEngine, not BaseStrategy.
        # Wrap it for BaseStrategy compatibility.
        class TSMOMWrapper(BaseStrategy):
            def __init__(self, cfg):
                super().__init__(cfg)
                self._engine = TSMOMSignalEngine(
                    strategy_name=cfg.name,
                    lookback_months=12,
                    neutral_band=0.10,
                )

            def generate_weights(self, data, target_date=None):
                if isinstance(data, dict):
                    # Try to extract a single ticker's DataFrame for signal generation
                    for ticker, df in data.items():
                        if isinstance(df, pd.DataFrame) and 'close' in df.columns:
                            sig = self._engine.generate_signal_one_ticker(df)
                            if sig is not None and len(sig) > 0:
                                direction = sig.iloc[-1]
                                if direction != 0:
                                    # Single-asset momentum signal
                                    weights = pd.DataFrame(
                                        {"weight": [1.0], "ticker": [ticker]}
                                    ).set_index("ticker").T
                                else:
                                    weights = pd.DataFrame()
                                return StrategyResult(
                                    strategy_name=cfg.name,
                                    weights=weights,
                                    metadata={'signal': int(direction)},
                                )
                # Fallback: equal weights
                if isinstance(data, dict):
                    tickers = list(data.keys())
                else:
                    tickers = []
                n = len(tickers)
                if n == 0:
                    weights = pd.DataFrame()
                else:
                    weights = pd.DataFrame({"weight": [1.0/n]*n, "ticker": tickers}).set_index("ticker").T
                return StrategyResult(strategy_name=cfg.name, weights=weights)

        return TSMOMWrapper(config)

    # --- Bucket Strategy ---
    elif strategy_type == 'bucket':
        from .ml_strategy import MLStockSelectionStrategy
        bucket_config = StrategyConfig(name="bucket")
        return MLStockSelectionStrategy(bucket_config)

    # --- Equal Weight Strategy ---
    elif strategy_type in ('equal-weight', 'equal_weight', 'equal'):
        class EqualWeightStrategy(BaseStrategy):
            def generate_weights(self, data, target_date=None):
                if isinstance(data, dict):
                    tickers = list(data.keys())
                else:
                    raise ValueError("Data must be a dict of DataFrames")
                n = len(tickers)
                if n == 0:
                    weights = pd.DataFrame()
                else:
                    weights = pd.DataFrame({"weight": [1.0/n]*n, "ticker": tickers}).set_index("ticker").T
                return StrategyResult(strategy_name="equal_weight", weights=weights)
        return EqualWeightStrategy(config)

    # --- Trend Following Strategy ---
    elif strategy_type == 'trend':
        from .trend_following import TrendFollowingStrategy
        return TrendFollowingStrategy(config)

    # --- Mean Reversion Strategy ---
    elif strategy_type == 'reversion':
        from .mean_reversion import MeanReversionStrategy
        return MeanReversionStrategy(config)

    # --- Pairs Trading Strategy ---
    elif strategy_type == 'pairs':
        from .pairs_trading import PairsTradingStrategy
        return PairsTradingStrategy(config)

    # --- Unknown: raise error immediately ---
    else:
        raise ValueError(f"Unknown strategy_type: {strategy_type}")