from pathlib import Path

from src.strategies.adaptive_rotation.adaptive_rotation_engine import (
    AdaptiveRotationEngine,
)
from src.strategies.adaptive_rotation.config_loader import load_config


def test_ranker_uses_configured_zscore_window_instead_of_selection_count():
    config_path = (
        Path(__file__).parents[1]
        / "src"
        / "strategies"
        / "AdaptiveRotationConf_v1.2.2.yaml"
    )
    config = load_config(str(config_path))

    engine = AdaptiveRotationEngine(config)

    assert config.ranking.top_n_per_group == 3
    assert config.ranking.zscore_window == 12
    assert engine.intra_group_ranker.lookback_weeks == config.ranking.zscore_window
