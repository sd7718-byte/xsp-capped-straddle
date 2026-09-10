from datetime import time

import pytest

from xsp_straddle.config import BacktestConfig, StrategyConfig


def test_defaults_match_strategy_brief():
    config = StrategyConfig()
    assert (config.entry_dte_min, config.entry_dte_max, config.exit_dte) == (35, 45, 7)
    assert config.target_wing_abs_delta == pytest.approx(0.10)
    assert config.scheduled_hedge_time == time(15, 45)
    assert config.entry_weekday == 2
    assert config.option_multiplier == pytest.approx(100.0)


def test_invalid_configurations_fail_fast():
    with pytest.raises(ValueError):
        StrategyConfig(entry_dte_min=7)
    with pytest.raises(ValueError):
        StrategyConfig(max_leg_spread_fraction=1.1)
    with pytest.raises(ValueError):
        BacktestConfig(initial_capital=0.0)
