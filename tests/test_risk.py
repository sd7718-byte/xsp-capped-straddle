import math

import pytest

from xsp_straddle.risk import (
    CappedPackage,
    LegQuote,
    RiskLimits,
    ShortStressEstimate,
    TradeRiskInput,
    evaluate_trade_risk,
    size_trade,
)


def make_package(
    expiry="2026-10-16",
    atm_call=None,
    atm_put=None,
    upper_call=None,
    lower_put=None,
):
    return CappedPackage(
        expiry=expiry,
        atm_call=atm_call or LegQuote("atm_call", "call", 100.0, 4.90, 5.10),
        atm_put=atm_put or LegQuote("atm_put", "put", 100.0, 4.90, 5.10),
        upper_call=upper_call
        or LegQuote("upper_call", "call", 110.0, 1.00, 1.04),
        lower_put=lower_put or LegQuote("lower_put", "put", 90.0, 1.00, 1.04),
    )


def make_trade(**overrides):
    values = {
        "capital": 1_000_000.0,
        "direction": "long",
        "package": make_package(),
        "one_vol_point_vega_loss_per_package": 250.0,
    }
    values.update(overrides)
    return TradeRiskInput(**values)


def test_package_is_structurally_capped_and_uses_executable_prices():
    package = make_package()

    assert package.executable_buy_price == pytest.approx(8.20)
    assert package.executable_sell_price == pytest.approx(7.72)
    assert package.long_debit_per_package == pytest.approx(820.0)
    assert [leg.label for leg in package.legs] == [
        "atm_call",
        "atm_put",
        "upper_call",
        "lower_put",
    ]


@pytest.mark.parametrize(
    "overrides, error",
    [
        (
            {"atm_put": LegQuote("bad_atm_put", "put", 99.0, 4.9, 5.1)},
            "must match",
        ),
        (
            {"upper_call": LegQuote("bad_upper", "call", 99.0, 1.0, 1.04)},
            "above",
        ),
        (
            {"lower_put": LegQuote("bad_lower", "put", 101.0, 1.0, 1.04)},
            "below",
        ),
    ],
)
def test_package_rejects_uncapped_or_misaligned_geometry(overrides, error):
    with pytest.raises(ValueError, match=error):
        make_package(**overrides)


def test_leg_quote_validates_market_data_and_relative_spread():
    quote = LegQuote("x", "call", 100.0, 0.95, 1.05)
    assert quote.midpoint == pytest.approx(1.0)
    assert quote.relative_spread == pytest.approx(0.10)

    with pytest.raises(ValueError, match="ask"):
        LegQuote("bad", "put", 100.0, 2.0, 1.0)
    with pytest.raises(ValueError, match="option_type"):
        LegQuote("bad", "straddle", 100.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="finite"):
        LegQuote("bad", "put", 100.0, math.nan, 1.0)


def test_one_vol_point_vega_limit_sizes_to_tenth_percent_of_capital():
    sizing = size_trade(make_trade())

    assert sizing.max_packages == 4  # $1,000 budget / $250 per package
    assert sizing.binding_constraints == ("ONE_VOL_VEGA",)
    assert not sizing.rejected
    vega = next(item for item in sizing.constraints if item.code == "ONE_VOL_VEGA")
    assert vega.limit_dollars == pytest.approx(1_000.0)
    assert vega.max_packages == 4


def test_long_debit_limit_sizes_to_half_percent_of_capital():
    sizing = size_trade(
        make_trade(one_vol_point_vega_loss_per_package=40.0)
    )

    # $5,000 debit budget / $820 executable debit.
    assert sizing.max_packages == 6
    assert sizing.binding_constraints == ("LONG_OPTION_DEBIT",)


def test_existing_exposure_reduces_remaining_capacity():
    sizing = size_trade(
        make_trade(
            one_vol_point_vega_loss_per_package=100.0,
            existing_one_vol_point_vega_loss=400.0,
            existing_long_option_debit=900.0,
        )
    )

    assert sizing.max_packages == 5  # debit headroom: (5000 - 900) / 820
    assert sizing.binding_constraints == ("LONG_OPTION_DEBIT",)


def test_short_stress_is_limited_to_one_percent_per_expiry():
    trade = make_trade(
        direction="short",
        one_vol_point_vega_loss_per_package=100.0,
        short_stress=ShortStressEstimate(loss_per_package=3_000.0),
        existing_short_stress_loss_for_expiry=1_000.0,
    )
    sizing = size_trade(trade)

    assert sizing.max_packages == 3  # ($10,000 - $1,000) / $3,000
    assert sizing.binding_constraints == ("SHORT_STRESS_PER_EXPIRY",)


def test_short_trade_requires_exact_joint_stress_scenario():
    missing = size_trade(make_trade(direction="short"))
    assert missing.max_packages == 0
    assert "SHORT_STRESS_REQUIRED" in missing.reason_codes

    mismatch = size_trade(
        make_trade(
            direction="short",
            short_stress=ShortStressEstimate(
                loss_per_package=1_000.0,
                underlying_move_fraction=-0.07,
                volatility_point_increase=15.0,
            ),
        )
    )
    assert mismatch.max_packages == 0
    assert "SHORT_STRESS_SCENARIO_MISMATCH" in mismatch.reason_codes


def test_maximum_three_distinct_overlapping_expirations():
    active = ("2026-09-18", "2026-09-25", "2026-10-02")
    rejected = size_trade(make_trade(active_expirations=active))
    assert rejected.max_packages == 0
    assert "MAX_OVERLAPPING_EXPIRATIONS" in rejected.reason_codes

    same_expiry = make_package(expiry="2026-10-02")
    allowed = size_trade(make_trade(active_expirations=active, package=same_expiry))
    assert allowed.max_packages > 0
    assert "MAX_OVERLAPPING_EXPIRATIONS" not in allowed.reason_codes


def test_leg_spread_over_ten_percent_rejects_whole_package():
    exactly_ten = LegQuote("wing", "call", 110.0, 0.95, 1.05)
    accepted = size_trade(make_trade(package=make_package(upper_call=exactly_ten)))
    assert accepted.max_packages > 0

    too_wide = LegQuote("wide_wing", "call", 110.0, 0.94, 1.06)
    rejected = size_trade(make_trade(package=make_package(upper_call=too_wide)))
    assert rejected.max_packages == 0
    assert rejected.reason_codes == ("LEG_SPREAD_TOO_WIDE",)
    assert rejected.violations[0].leg == "wide_wing"
    assert rejected.violations[0].observed == pytest.approx(0.12)


def test_zero_whole_package_capacity_has_specific_reason():
    sizing = size_trade(
        make_trade(existing_one_vol_point_vega_loss=1_000.0)
    )

    assert sizing.max_packages == 0
    assert "ONE_VOL_VEGA_CAPACITY_EXHAUSTED" in sizing.reason_codes


def test_requested_quantity_is_approved_or_rejected_without_partial_approval():
    trade = make_trade()

    approved = evaluate_trade_risk(trade, requested_packages=4)
    assert approved.approved
    assert approved.approved_packages == 4
    assert approved.recommended_packages == 4

    rejected = evaluate_trade_risk(trade, requested_packages=5)
    assert not rejected.approved
    assert rejected.approved_packages == 0
    assert rejected.recommended_packages == 4
    assert rejected.reason_codes == ("REQUESTED_SIZE_EXCEEDS_LIMIT",)
    assert rejected.to_dict()["sizing"]["max_packages"] == 4


@pytest.mark.parametrize("quantity", [0, -1, 1.5, True])
def test_requested_quantity_must_be_a_positive_integer(quantity):
    with pytest.raises(ValueError, match="positive integer"):
        evaluate_trade_risk(make_trade(), quantity)


def test_limits_are_configurable_and_validated():
    sizing = size_trade(
        make_trade(),
        RiskLimits(one_vol_vega_fraction=0.002),
    )
    assert sizing.max_packages == 6  # debit, rather than vega, is now binding

    with pytest.raises(ValueError, match=r"between|cannot exceed|non-negative"):
        RiskLimits(long_debit_fraction=-0.01)
