import math

import pandas as pd
import pytest

from xsp_straddle.config import StrategyConfig
from xsp_straddle.costs import CostAssumptions
from xsp_straddle.pricing import OptionType, bs_price
from xsp_straddle.signal import SignalSide, analyze_snapshot


def _chain(vol=0.20):
    timestamp = pd.Timestamp("2026-01-07T20:45:00Z")
    expiration = pd.Timestamp("2026-02-20T21:00:00Z")
    maturity = (expiration - timestamp).total_seconds() / (365.25 * 86400.0)
    spot, rate, dividend = 500.0, 0.04, 0.01
    forward = spot * math.exp((rate - dividend) * maturity)
    rows = []
    for strike in range(440, 561, 10):
        skew_vol = vol - 0.12 * math.log(strike / forward)
        for right, kind in [("C", OptionType.CALL), ("P", OptionType.PUT)]:
            mid = bs_price(spot, strike, maturity, rate, skew_vol, kind, dividend)
            half_spread = min(0.002 * mid, 0.005)
            rows.append(
                {
                    "symbol": "XSP",
                    "settlement": "PM",
                    "timestamp": timestamp,
                    "expiration": expiration,
                    "strike": float(strike),
                    "option_type": right,
                    "bid": mid - half_spread,
                    "ask": mid + half_spread,
                    "spot": spot,
                    "hedge_bid": spot - 0.01,
                    "hedge_ask": spot + 0.01,
                    "rate": rate,
                    "dividend_yield": dividend,
                }
            )
    return pd.DataFrame(rows), expiration


def test_high_pessimistic_forecast_produces_exact_long_package_signal():
    chain, expiration = _chain()
    decision = analyze_snapshot(
        chain,
        expiration,
        forecast_lower_vol=0.30,
        forecast_upper_vol=0.34,
        cost_assumptions=CostAssumptions(0.0, 0.0),
    )
    assert decision.side is SignalSide.LONG
    assert decision.long_edge > 0.0
    assert decision.analysis.spread_ok
    assert decision.analysis.definition.lower_strike < decision.analysis.definition.atm_strike
    assert decision.analysis.definition.upper_strike > decision.analysis.definition.atm_strike
    assert len(decision.analysis.legs) == 4
    for leg in decision.analysis.legs:
        assert leg.bid_iv <= leg.mid_iv <= leg.ask_iv


def test_low_pessimistic_forecast_produces_short_signal_with_protective_wings():
    chain, expiration = _chain()
    decision = analyze_snapshot(
        chain,
        expiration,
        forecast_lower_vol=0.10,
        forecast_upper_vol=0.12,
        cost_assumptions=CostAssumptions(0.0, 0.0),
    )
    assert decision.side is SignalSide.SHORT
    assert decision.short_edge > 0.0
    assert decision.analysis.definition.lower_strike < decision.analysis.definition.atm_strike
    assert decision.analysis.definition.atm_strike < decision.analysis.definition.upper_strike


def test_approximation_can_only_pre_filter_and_never_creates_a_trade():
    chain, expiration = _chain()
    decision = analyze_snapshot(
        chain,
        expiration,
        forecast_lower_vol=0.21,
        forecast_upper_vol=0.21,
        cost_assumptions=CostAssumptions(0.0, 0.0),
        approximate_bid_iv=0.19,
        approximate_ask_iv=0.20,
    )
    assert decision.side is SignalSide.NONE
    assert decision.reason == "APPROXIMATE_PREFILTER_NO_EDGE"
    assert decision.analysis is None


def test_any_selected_leg_over_ten_percent_spread_rejects_package():
    chain, expiration = _chain()
    baseline = analyze_snapshot(
        chain,
        expiration,
        0.30,
        0.34,
        CostAssumptions(0.0, 0.0),
    )
    upper = baseline.analysis.definition.upper_strike
    mask = (chain["strike"] == upper) & (chain["option_type"] == "C")
    mid = float((chain.loc[mask, "bid"].iloc[0] + chain.loc[mask, "ask"].iloc[0]) / 2)
    chain.loc[mask, "bid"] = 0.90 * mid
    chain.loc[mask, "ask"] = 1.10 * mid
    rejected = analyze_snapshot(
        chain,
        expiration,
        0.30,
        0.34,
        CostAssumptions(0.0, 0.0),
    )
    assert rejected.side is SignalSide.NONE
    assert rejected.reason == "LEG_SPREAD_TOO_WIDE"


def test_positive_projected_cost_can_remove_small_edge():
    chain, expiration = _chain()
    no_cost = analyze_snapshot(
        chain,
        expiration,
        0.21,
        0.23,
        CostAssumptions(0.0, 0.0),
    )
    costly = analyze_snapshot(
        chain,
        expiration,
        0.21,
        0.23,
        CostAssumptions(10.0, 100.0),
    )
    assert costly.long_edge < no_cost.long_edge
    assert costly.short_edge < no_cost.short_edge
