import math

import pytest

from xsp_straddle.pricing import (
    ImpliedVolatilityError,
    NoArbitrageError,
    OptionType,
    PricingError,
    bs_delta,
    bs_forward_greeks,
    bs_forward_price,
    bs_gamma,
    bs_greeks,
    bs_price,
    bs_vega,
    estimate_forward,
    forward_no_arbitrage_bounds,
    implied_vol_brent,
    implied_vol_brent_forward,
    no_arbitrage_bounds,
)


def test_option_type_parses_common_spellings() -> None:
    assert OptionType.parse("CALL") is OptionType.CALL
    assert OptionType.parse(" c ") is OptionType.CALL
    assert OptionType.parse("p") is OptionType.PUT
    with pytest.raises(ValueError, match="option_type"):
        OptionType.parse("straddle")


@pytest.mark.parametrize(
    "option_type, expected_price, expected_delta",
    [
        ("call", 10.450583572185565, 0.6368306511756191),
        ("put", 5.573526022256971, -0.3631693488243809),
    ],
)
def test_black_scholes_matches_reference_values(
    option_type: str, expected_price: float, expected_delta: float
) -> None:
    result = bs_greeks(100.0, 100.0, 1.0, 0.05, 0.20, option_type)
    assert result.price == pytest.approx(expected_price, abs=1e-12)
    assert result.delta == pytest.approx(expected_delta, abs=1e-12)
    assert result.gamma == pytest.approx(0.018762017345846895, abs=1e-14)
    assert result.vega == pytest.approx(37.52403469169379, abs=1e-12)
    assert bs_price(100, 100, 1, 0.05, 0.2, option_type) == result.price
    assert bs_delta(100, 100, 1, 0.05, 0.2, option_type) == result.delta
    assert bs_gamma(100, 100, 1, 0.05, 0.2, option_type) == result.gamma
    assert bs_vega(100, 100, 1, 0.05, 0.2, option_type) == result.vega


def test_spot_put_call_parity_and_forward_convention_agree() -> None:
    spot = 123.0
    strike = 117.0
    tenor = 0.63
    rate = 0.031
    dividend_yield = 0.014
    volatility = 0.27
    discount = math.exp(-rate * tenor)
    forward = spot * math.exp((rate - dividend_yield) * tenor)

    call = bs_price(
        spot, strike, tenor, rate, volatility, "call", dividend_yield
    )
    put = bs_price(spot, strike, tenor, rate, volatility, "put", dividend_yield)
    assert call - put == pytest.approx(
        spot * math.exp(-dividend_yield * tenor) - strike * discount,
        abs=2e-14,
    )
    assert bs_forward_price(
        forward, strike, tenor, discount, volatility, "call"
    ) == pytest.approx(call, abs=5e-14)
    assert bs_forward_price(
        forward, strike, tenor, discount, volatility, "put"
    ) == pytest.approx(put, abs=5e-14)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_spot_greeks_match_centered_finite_differences(option_type: str) -> None:
    spot = 97.0
    strike = 102.0
    tenor = 0.37
    rate = -0.002
    dividend_yield = 0.011
    volatility = 0.34
    greeks = bs_greeks(
        spot, strike, tenor, rate, volatility, option_type, dividend_yield
    )

    spot_step = 1e-2
    price_up = bs_price(
        spot + spot_step,
        strike,
        tenor,
        rate,
        volatility,
        option_type,
        dividend_yield,
    )
    price_at = bs_price(
        spot, strike, tenor, rate, volatility, option_type, dividend_yield
    )
    price_down = bs_price(
        spot - spot_step,
        strike,
        tenor,
        rate,
        volatility,
        option_type,
        dividend_yield,
    )
    assert greeks.delta == pytest.approx(
        (price_up - price_down) / (2.0 * spot_step), rel=2e-8
    )
    assert greeks.gamma == pytest.approx(
        (price_up - 2.0 * price_at + price_down) / (spot_step * spot_step),
        rel=2e-6,
    )

    vol_step = 1e-5
    vol_up = bs_price(
        spot,
        strike,
        tenor,
        rate,
        volatility + vol_step,
        option_type,
        dividend_yield,
    )
    vol_down = bs_price(
        spot,
        strike,
        tenor,
        rate,
        volatility - vol_step,
        option_type,
        dividend_yield,
    )
    assert greeks.vega == pytest.approx(
        (vol_up - vol_down) / (2.0 * vol_step), rel=2e-9
    )


def test_forward_greeks_use_forward_delta_convention() -> None:
    forward = 101.0
    strike = 99.0
    tenor = 0.4
    discount = 0.98
    volatility = 0.22
    call = bs_forward_greeks(
        forward, strike, tenor, discount, volatility, "call"
    )
    put = bs_forward_greeks(
        forward, strike, tenor, discount, volatility, "put"
    )
    step = 1e-3
    finite_delta = (
        bs_forward_price(
            forward + step, strike, tenor, discount, volatility, "call"
        )
        - bs_forward_price(
            forward - step, strike, tenor, discount, volatility, "call"
        )
    ) / (2.0 * step)
    assert call.delta == pytest.approx(finite_delta, rel=1e-9)
    assert call.delta - put.delta == pytest.approx(discount, abs=1e-15)
    assert call.gamma == put.gamma
    assert call.vega == put.vega


def test_expiry_and_zero_volatility_limits_are_well_defined() -> None:
    assert bs_greeks(105, 100, 0, 0.03, 0.2, "call").price == 5.0
    assert bs_greeks(105, 100, 0, 0.03, 0.2, "call").delta == 1.0
    assert bs_greeks(95, 100, 0, 0.03, 0.2, "put").price == 5.0
    assert bs_greeks(100, 100, 0, 0.03, 0.2, "call").delta == 0.5

    deterministic = bs_greeks(100, 90, 1.0, 0.02, 0.0, "call", 0.01)
    assert deterministic.price == pytest.approx(
        100 * math.exp(-0.01) - 90 * math.exp(-0.02)
    )
    assert deterministic.delta == pytest.approx(math.exp(-0.01))
    assert deterministic.gamma == 0.0
    assert deterministic.vega == 0.0


def test_no_arbitrage_bounds_include_dividends_and_negative_rates() -> None:
    call_bounds = no_arbitrage_bounds(100, 103, 0.5, -0.01, "call", 0.02)
    put_bounds = no_arbitrage_bounds(100, 103, 0.5, -0.01, "put", 0.02)
    pv_spot = 100 * math.exp(-0.02 * 0.5)
    pv_strike = 103 * math.exp(0.01 * 0.5)
    assert call_bounds == (0.0, pv_spot)
    assert put_bounds == (pytest.approx(pv_strike - pv_spot), pv_strike)
    assert forward_no_arbitrage_bounds(105, 100, 0.97, "call") == (
        pytest.approx(4.85),
        pytest.approx(101.85),
    )


@pytest.mark.parametrize(
    "spot,strike,tenor,rate,dividend,volatility,option_type",
    [
        (100.0, 100.0, 1.0, 0.05, 0.0, 0.20, "call"),
        (100.0, 100.0, 1.0, 0.05, 0.0, 0.20, "put"),
        (72.0, 110.0, 0.08, -0.01, 0.02, 0.55, "call"),
        (140.0, 90.0, 1.8, 0.01, 0.03, 0.48, "put"),
        (100.0, 130.0, 0.7, 0.0, 0.0, 2.40, "call"),
    ],
)
def test_brent_implied_vol_recovers_generating_volatility(
    spot: float,
    strike: float,
    tenor: float,
    rate: float,
    dividend: float,
    volatility: float,
    option_type: str,
) -> None:
    price = bs_price(
        spot, strike, tenor, rate, volatility, option_type, dividend
    )
    solved = implied_vol_brent(
        price,
        spot,
        strike,
        tenor,
        rate,
        option_type,
        dividend,
        initial_upper=0.10,
    )
    assert solved == pytest.approx(volatility, abs=2e-10)


def test_forward_brent_implied_vol_recovers_generating_volatility() -> None:
    target = bs_forward_price(420.0, 400.0, 0.25, 0.992, 0.31, "put")
    solved = implied_vol_brent_forward(
        target, 420.0, 400.0, 0.25, 0.992, "put", initial_upper=0.05
    )
    assert solved == pytest.approx(0.31, abs=1e-11)


def test_implied_vol_validates_bounds_and_boundary_cases() -> None:
    lower, upper = no_arbitrage_bounds(100, 80, 1, 0.03, "call")
    assert implied_vol_brent(lower, 100, 80, 1, 0.03, "call") == 0.0
    with pytest.raises(NoArbitrageError, match="outside no-arbitrage bounds"):
        implied_vol_brent(lower - 0.01, 100, 80, 1, 0.03, "call")
    with pytest.raises(NoArbitrageError, match="outside no-arbitrage bounds"):
        implied_vol_brent(upper + 0.01, 100, 80, 1, 0.03, "call")
    with pytest.raises(ImpliedVolatilityError, match="no finite"):
        implied_vol_brent(upper, 100, 80, 1, 0.03, "call")
    with pytest.raises(ImpliedVolatilityError, match="undefined at expiry"):
        implied_vol_brent(5.0, 105, 100, 0, 0.03, "call")


def test_implied_vol_reports_failed_bracket_and_iteration_limit() -> None:
    price = bs_price(100, 100, 1, 0, 2.0, "call")
    with pytest.raises(ImpliedVolatilityError, match="could not bracket"):
        implied_vol_brent(
            price,
            100,
            100,
            1,
            0,
            "call",
            initial_upper=0.1,
            max_volatility=0.2,
        )

    ordinary_price = bs_price(100, 105, 0.8, 0.01, 0.37, "put")
    with pytest.raises(ImpliedVolatilityError, match="did not converge"):
        implied_vol_brent(
            ordinary_price,
            100,
            105,
            0.8,
            0.01,
            "put",
            max_iterations=1,
        )


def test_estimate_forward_from_put_call_parity_with_rate_or_discount() -> None:
    expected_forward = 104.25
    strike = 100.0
    tenor = 0.4
    rate = 0.03
    discount = math.exp(-rate * tenor)
    put = 3.75
    call = put + discount * (expected_forward - strike)
    assert estimate_forward(call, put, strike, tenor, rate) == pytest.approx(
        expected_forward
    )
    assert estimate_forward(
        call, put, strike, tenor, discount_factor=discount
    ) == pytest.approx(expected_forward)

    with pytest.raises(NoArbitrageError, match="non-positive forward"):
        estimate_forward(0.0, 200.0, 100.0, 1.0)


@pytest.mark.parametrize(
    "function,args",
    [
        (bs_price, (0, 100, 1, 0.01, 0.2, "call")),
        (bs_price, (100, -1, 1, 0.01, 0.2, "call")),
        (bs_price, (100, 100, -1, 0.01, 0.2, "call")),
        (bs_price, (100, 100, 1, 0.01, -0.2, "call")),
        (bs_forward_price, (100, 100, 1, 0, 0.2, "call")),
    ],
)
def test_pricing_rejects_invalid_numeric_inputs(function, args) -> None:
    with pytest.raises(PricingError):
        function(*args)
