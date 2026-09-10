import pytest

from xsp_straddle.hedging import (
    capped_package_delta,
    convert_dollar_delta_to_futures,
    desired_hedge_units,
    execute_underlying_hedge,
    residual_delta_exceeds_trigger,
)


def test_package_delta_and_long_short_hedges_have_opposite_signs():
    delta = capped_package_delta(0.55, -0.45, 0.10, -0.10)
    assert delta == pytest.approx(0.10)
    assert desired_hedge_units(delta, 2, 100, 1) == pytest.approx(-20.0)
    assert desired_hedge_units(delta, 2, 100, -1) == pytest.approx(20.0)


def test_trigger_is_strictly_greater_than_ten_shares_per_package():
    assert not residual_delta_exceeds_trigger(0.0, 20.0, 2, 100, 0.10)
    assert residual_delta_exceeds_trigger(0.0, 20.01, 2, 100, 0.10)


def test_hedge_execution_uses_adverse_quote_and_commission():
    buy = execute_underlying_hedge(0.0, 10.0, 99.9, 100.1, 0.01)
    assert buy.execution_price == pytest.approx(100.1)
    assert buy.cash_change == pytest.approx(-1001.1)
    sell = execute_underlying_hedge(10.0, 0.0, 99.9, 100.1, 0.01)
    assert sell.execution_price == pytest.approx(99.9)
    assert sell.cash_change == pytest.approx(998.9)


def test_mes_es_conversion_uses_dollar_delta_and_reports_residual():
    # -100 XSP-equivalent units at 500 map to -2 MES contracts at ES 5000.
    converted = convert_dollar_delta_to_futures(-100, 500, 5000, 5)
    assert converted.contracts == -2
    assert converted.residual_dollar_delta == pytest.approx(0.0)
