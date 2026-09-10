import math

import pytest

from xsp_straddle.market import (
    BidAskQuote,
    CappedStraddle,
    MarketDataError,
    OptionChain,
    OptionQuote,
    closest_atm_strike,
    executable_package_prices,
    passes_spread_filter,
    select_capped_straddle,
)
from xsp_straddle.pricing import OptionType


def quote(
    strike: float,
    option_type: str,
    bid: float,
    ask: float,
    delta=None,
) -> OptionQuote:
    return OptionQuote(strike, option_type, bid, ask, delta=delta)


def sample_chain() -> OptionChain:
    return OptionChain(
        quotes=[
            quote(85, "put", 0.35, 0.37, -0.04),
            quote(90, "put", 1.45, 1.55, -0.105),
            quote(95, "put", 3.1, 3.2, -0.24),
            quote(100, "put", 8.5, 9.0, -0.48),
            quote(105, "put", 11.8, 12.1, -0.66),
            quote(95, "call", 13.2, 13.5, 0.73),
            quote(100, "call", 9.5, 10.0, 0.52),
            quote(105, "call", 5.4, 5.6, 0.29),
            quote(110, "call", 1.9, 2.1, 0.095),
            quote(115, "call", 0.75, 0.79, 0.045),
        ]
    )


def test_bid_ask_quote_validates_and_exposes_spreads() -> None:
    market = BidAskQuote(1.9, 2.1)
    assert market.mid == 2.0
    assert market.midpoint == 2.0
    assert market.spread == pytest.approx(0.2)
    assert market.relative_spread() == pytest.approx(0.1)
    assert market.relative_spread("ask") == pytest.approx(0.2 / 2.1)

    with pytest.raises(MarketDataError, match="below bid"):
        BidAskQuote(2.0, 1.0)
    with pytest.raises(MarketDataError, match="non-negative"):
        BidAskQuote(-0.01, 1.0)
    with pytest.raises(MarketDataError, match="finite"):
        BidAskQuote(1.0, math.nan)


def test_option_quote_normalizes_type_and_validates_analytics() -> None:
    market = OptionQuote(100, "C", 2, 2.2, delta=0.4, implied_volatility=0.21)
    assert market.option_type is OptionType.CALL
    assert isinstance(market.strike, float)
    assert market.mid == 2.1

    with pytest.raises(MarketDataError, match="call delta"):
        OptionQuote(100, "call", 2, 2.2, delta=-0.1)
    with pytest.raises(MarketDataError, match="put delta"):
        OptionQuote(100, "put", 2, 2.2, delta=0.1)
    with pytest.raises(MarketDataError, match="implied_volatility"):
        OptionQuote(100, "put", 2, 2.2, implied_volatility=-0.1)
    with pytest.raises(MarketDataError, match="option_type"):
        OptionQuote(100, "future", 2, 2.2)


def test_option_chain_normalizes_sorts_and_rejects_duplicates() -> None:
    chain = sample_chain()
    assert isinstance(chain.quotes, tuple)
    assert [item.strike for item in chain.calls] == [95, 100, 105, 110, 115]
    assert [item.strike for item in chain.puts] == [85, 90, 95, 100, 105]
    assert chain.strikes == (85, 90, 95, 100, 105, 110, 115)
    assert chain.common_strikes == (95, 100, 105)
    assert chain.get(100, "c").bid == 9.5
    with pytest.raises(MarketDataError, match="missing put"):
        chain.get(110, "put")

    duplicate = quote(100, "call", 9.4, 10.1, 0.51)
    with pytest.raises(MarketDataError, match="duplicate call"):
        OptionChain(list(chain.quotes) + [duplicate])
    with pytest.raises(MarketDataError, match="cannot be empty"):
        OptionChain([])


def test_closest_atm_strike_is_deterministic_on_ties() -> None:
    assert closest_atm_strike([105, 95, 100], 102.5) == 100
    assert closest_atm_strike([105, 95, 100], 103.0) == 105
    with pytest.raises(MarketDataError, match="empty"):
        closest_atm_strike([], 100)


def test_select_capped_straddle_uses_forward_and_closest_delta_wings() -> None:
    package = select_capped_straddle(sample_chain(), forward=101.0)
    assert package.atm_strike == 100
    assert package.atm_call.option_type is OptionType.CALL
    assert package.atm_put.option_type is OptionType.PUT
    assert package.lower_put.strike == 90
    assert package.upper_call.strike == 110
    assert package.legs == (
        package.atm_call,
        package.atm_put,
        package.upper_call,
        package.lower_put,
    )
    assert package.delta == pytest.approx(0.05)


def test_wing_selection_uses_nearest_strike_as_delta_tie_breaker() -> None:
    chain = sample_chain()
    amended = [item for item in chain.quotes if item.strike not in (85, 115)]
    amended.extend(
        [
            quote(85, "put", 0.35, 0.37, -0.08),
            quote(115, "call", 0.75, 0.79, 0.12),
        ]
    )
    # Make the nearer wings equally distant from 10 delta.
    amended = [
        quote(q.strike, q.option_type.value, q.bid, q.ask, -0.12)
        if q.strike == 90 and q.option_type is OptionType.PUT
        else quote(q.strike, q.option_type.value, q.bid, q.ask, 0.08)
        if q.strike == 110 and q.option_type is OptionType.CALL
        else q
        for q in amended
    ]
    package = select_capped_straddle(OptionChain(amended), forward=100)
    assert package.lower_put.strike == 90
    assert package.upper_call.strike == 110


def test_selection_requires_common_atm_pair_and_delta_enriched_wings() -> None:
    no_common = OptionChain(
        [quote(90, "put", 1, 1.1, -0.1), quote(110, "call", 1, 1.1, 0.1)]
    )
    with pytest.raises(MarketDataError, match="no strike"):
        select_capped_straddle(no_common, 100)

    quotes_without_put_delta = [
        OptionQuote(q.strike, q.option_type, q.bid, q.ask, None)
        if q.option_type is OptionType.PUT and q.strike < 100
        else q
        for q in sample_chain().quotes
    ]
    with pytest.raises(MarketDataError, match="delta-enriched put"):
        select_capped_straddle(OptionChain(quotes_without_put_delta), 100)


def test_capped_straddle_validates_leg_types_and_strike_ordering() -> None:
    package = select_capped_straddle(sample_chain(), 100)
    with pytest.raises(MarketDataError, match="atm_call must be a call"):
        CappedStraddle(
            package.atm_put,
            package.atm_put,
            package.upper_call,
            package.lower_put,
        )
    with pytest.raises(MarketDataError, match="bracket"):
        CappedStraddle(
            package.atm_call,
            package.atm_put,
            quote(99, "call", 1, 1.1, 0.1),
            package.lower_put,
        )


def test_executable_package_prices_follow_correct_sides_and_multiplier() -> None:
    package = select_capped_straddle(sample_chain(), 100)
    points = executable_package_prices(package)
    assert points.buy == pytest.approx(10.0 + 9.0 - 1.9 - 1.45)
    assert points.sell == pytest.approx(9.5 + 8.5 - 2.1 - 1.55)
    assert points.mid == pytest.approx(9.75 + 8.75 - 2.0 - 1.5)
    assert points.multiplier == 1.0
    assert points.width == pytest.approx(points.buy - points.sell)

    currency = executable_package_prices(package, multiplier=100.0)
    assert currency.buy == pytest.approx(100.0 * points.buy)
    assert currency.sell == pytest.approx(100.0 * points.sell)
    assert currency.mid == pytest.approx(100.0 * points.mid)
    assert currency.multiplier == 100.0
    with pytest.raises(MarketDataError, match="multiplier"):
        executable_package_prices(package, multiplier=0)


def test_spread_filter_checks_every_leg_and_can_be_applied_during_selection() -> None:
    package = select_capped_straddle(sample_chain(), 100)
    # Upper call is 10% of midpoint; ATM call is roughly 5.1%.
    assert passes_spread_filter(package, 0.101)
    assert not passes_spread_filter(package, 0.099)
    assert passes_spread_filter(package, 0.096, reference="ask")
    with pytest.raises(MarketDataError, match="wider"):
        select_capped_straddle(sample_chain(), 100, max_relative_spread=0.09)

    zero = BidAskQuote(0, 0)
    assert zero.relative_spread() == 0
    assert math.isinf(BidAskQuote(0, 0.1).relative_spread("bid"))
    with pytest.raises(MarketDataError, match="reference"):
        zero.relative_spread("last")

