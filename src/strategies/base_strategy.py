from __future__ import annotations
from dataclasses import dataclass
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

class BaseStrategy:
    """Minimal base strategy interface."""

    def __init__(self, config: StrategyConfig):
        self.config = config

    def generate_weights(self, data: Dict[str, pd.DataFrame], target_date: Optional[str] = None) -> StrategyResult:
        raise NotImplementedError("generate_weights must be implemented by subclasses")

class EqualWeightStrategy(BaseStrategy):
    """A strategy class that gives equal weight to all stocks"""

    def generate_weights(self, data: Dict[str, pd.DataFrame], target_date: Optional[str] = None) -> StrategyResult:
        tickers = data["fundamentals"]["gvkey"].unique()
        weight = 1.0 / len(tickers)
        weights_df = pd.DataFrame({
            "gvkey": tickers,
            "weight": [weight] * len(tickers)
        })
        return StrategyResult(strategy_name=self.config.name, weights=weights_df)


def create_strategy(strategy_type, config):
    strategies = {
        "equal_weight": EqualWeightStrategy,
    }

    strategy_class = strategies.get(strategy_type)
    if strategy_class == None:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
    return strategy_class(config)
