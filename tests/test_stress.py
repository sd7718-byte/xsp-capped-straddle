import pytest

from xsp_straddle.config import StrategyConfig
from xsp_straddle.costs import CostAssumptions
from xsp_straddle.risk import SHORT
from xsp_straddle.signal import analyze_snapshot
from xsp_straddle.stress import stress_short_package

from .test_signal import _chain


def test_short_stress_is_positive_and_uses_required_default_scenario():
    chain, expiration = _chain()
    decision = analyze_snapshot(
        chain,
        expiration,
        0.10,
        0.12,
        CostAssumptions(0.0, 0.0),
    )
    stressed = stress_short_package(decision.analysis)
    estimate = stressed.as_risk_estimate()
    assert stressed.stressed_spot == pytest.approx(0.92 * decision.analysis.spot)
    assert estimate.underlying_move_fraction == pytest.approx(-0.08)
    assert estimate.volatility_point_increase == pytest.approx(15.0)
    assert stressed.loss_cash > 0.0


def test_commission_increases_stressed_loss_dollar_for_dollar():
    chain, expiration = _chain()
    analysis = analyze_snapshot(
        chain,
        expiration,
        0.10,
        0.12,
        CostAssumptions(0.0, 0.0),
    ).analysis
    base = stress_short_package(analysis)
    costly = stress_short_package(analysis, close_commission_cash=4.0)
    assert costly.loss_cash == pytest.approx(base.loss_cash + 4.0)
