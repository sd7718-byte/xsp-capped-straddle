import pytest

from xsp_straddle.costs import CostAssumptions, project_round_trip_costs


def test_projected_long_costs_include_exit_liquidity_and_eight_commissions():
    costs = project_round_trip_costs(
        package_buy=5.2,
        package_sell=4.8,
        package_mid=5.0,
        direction=1,
        assumptions=CostAssumptions(0.65, 10.0, 2.0),
        option_multiplier=100,
    )
    assert costs.option_commissions == pytest.approx(8 * 0.65 / 100)
    assert costs.closing_liquidity == pytest.approx(0.2)
    assert costs.hedge_costs == pytest.approx(0.1)
    assert costs.other_costs == pytest.approx(0.02)


def test_short_uses_buy_side_for_projected_close_and_doubling_is_componentwise():
    costs = project_round_trip_costs(
        5.3,
        4.9,
        5.0,
        -1,
        CostAssumptions(0.0, 0.0),
    )
    assert costs.closing_liquidity == pytest.approx(0.3)
    assert costs.doubled().total == pytest.approx(2.0 * costs.total)


def test_invalid_package_market_is_rejected():
    with pytest.raises(ValueError):
        project_round_trip_costs(4.0, 5.0, 4.5, 1, CostAssumptions(0.0, 0.0))
