import pytest

from xsp_straddle.diagnostics import (
    BacktestComparison,
    DiagnosticThresholds,
    ReturnAttribution,
    ScenarioResult,
    SensitivityResult,
    TradeContribution,
    evaluate_backtest,
    overnight_gap_share,
    profit_disappears,
    top_short_vol_profit_share,
)


def scenario(name, profit):
    return ScenarioResult(name=name, net_profit=profit)


def comparison(**overrides):
    values = {
        "approximate_iv": scenario("approximate_iv", 120.0),
        "exact_iv": scenario("exact_iv", 100.0),
        "midpoint_fills": scenario("midpoint", 110.0),
        "executable_fills": scenario("executable", 100.0),
        "doubled_costs": scenario("double_cost", 20.0),
        "trade_contributions": (
            TradeContribution("s1", "short_vol", 15.0),
            TradeContribution("s2", "short", 10.0),
            TradeContribution("l1", "long_vol", 75.0),
        ),
        "return_attribution": ReturnAttribution(100.0, 20.0),
        "dte_sensitivities": (
            SensitivityResult("35_dte", 80.0),
            SensitivityResult("45_dte", 90.0),
        ),
        "hedge_frequency_sensitivities": (
            SensitivityResult("close_only", 70.0),
            SensitivityResult("delta_0.08", 75.0),
        ),
    }
    values.update(overrides)
    return BacktestComparison(**values)


def test_robust_backtest_is_accepted_with_auditable_metrics():
    report = evaluate_backtest(comparison())

    assert report.accepted
    assert not report.rejected
    assert report.reason_codes == ()
    assert report.metrics.top_short_vol_profit_share == pytest.approx(0.25)
    assert report.metrics.overnight_gap_share == pytest.approx(0.20)
    assert report.metrics.weakest_dte_profit == pytest.approx(80.0)
    assert report.metrics.weakest_hedge_frequency_profit == pytest.approx(70.0)
    assert "DOUBLED_COSTS_ROBUST" in report.passed_checks
    assert report.to_dict()["accepted"] is True


def test_rejects_when_exact_iv_removes_approximate_profit():
    report = evaluate_backtest(
        comparison(exact_iv=scenario("exact_iv", 0.0))
    )

    assert "EXACT_IV_REMOVES_PROFIT" in report.reason_codes
    reason = report.reasons[0]
    assert reason.observed == 0.0
    assert reason.threshold == 0.0
    assert reason.scenario == "exact_iv"


def test_exact_iv_rule_only_fires_when_reference_was_profitable():
    report = evaluate_backtest(
        comparison(
            approximate_iv=scenario("approximate_iv", -10.0),
            exact_iv=scenario("exact_iv", -20.0),
        )
    )
    assert "EXACT_IV_REMOVES_PROFIT" not in report.reason_codes


def test_rejects_profit_that_exists_only_at_midpoint_fills():
    report = evaluate_backtest(
        comparison(
            executable_fills=scenario("executable", -1.0),
            doubled_costs=scenario("double_cost", 1.0),
        )
    )

    assert "MIDPOINT_ONLY_PROFIT" in report.reason_codes
    assert "EXECUTABLE_BASELINE_UNPROFITABLE" in report.reason_codes


def test_rejects_when_doubled_transaction_costs_remove_profit():
    report = evaluate_backtest(
        comparison(doubled_costs=scenario("double_cost", -0.01))
    )

    assert report.reason_codes == ("DOUBLED_COSTS_UNPROFITABLE",)


def test_rejects_concentration_in_top_few_short_vol_trades():
    contributions = (
        TradeContribution("s1", "short", 40.0),
        TradeContribution("s2", "short", 20.0),
        TradeContribution("s3", "short", 10.0),
        TradeContribution("l1", "long", 30.0),
    )
    report = evaluate_backtest(comparison(trade_contributions=contributions))

    assert "SHORT_VOL_PROFIT_CONCENTRATION" in report.reason_codes
    reason = next(
        item for item in report.reasons if item.code == "SHORT_VOL_PROFIT_CONCENTRATION"
    )
    assert reason.observed == pytest.approx(0.70)
    assert reason.threshold == pytest.approx(0.50)


def test_concentration_at_threshold_is_not_most_profit():
    contributions = (
        TradeContribution("s1", "short", 30.0),
        TradeContribution("s2", "short", 20.0),
        TradeContribution("l1", "long", 50.0),
    )
    report = evaluate_backtest(comparison(trade_contributions=contributions))
    assert "SHORT_VOL_PROFIT_CONCENTRATION" not in report.reason_codes


def test_concentration_uses_winners_not_short_trade_netting():
    contributions = (
        TradeContribution("winner", "short", 80.0),
        TradeContribution("loser", "short", -70.0),
        TradeContribution("long", "long", 90.0),
    )
    share = top_short_vol_profit_share(contributions, lifetime_profit=100.0)
    assert share == pytest.approx(0.80)


def test_rejects_when_overnight_gaps_dominate_delta_hedged_return():
    report = evaluate_backtest(
        comparison(return_attribution=ReturnAttribution(100.0, -60.0))
    )

    assert "OVERNIGHT_GAPS_DOMINATE" in report.reason_codes
    assert report.metrics.overnight_gap_share == pytest.approx(0.60)
    assert overnight_gap_share(ReturnAttribution(-100.0, 60.0)) == pytest.approx(0.60)


def test_near_zero_total_return_has_finite_gap_ratio():
    ratio = overnight_gap_share(ReturnAttribution(0.0, 1.0), tolerance=1e-6)
    assert ratio == pytest.approx(1_000_000.0)


def test_rejects_dte_and_hedge_frequency_reversals_separately():
    report = evaluate_backtest(
        comparison(
            dte_sensitivities=(SensitivityResult("34_dte", -1.0),),
            hedge_frequency_sensitivities=(
                SensitivityResult("twice_daily", -2.0),
            ),
        )
    )

    assert "DTE_SENSITIVITY_REVERSAL" in report.reason_codes
    assert "HEDGE_FREQUENCY_SENSITIVITY_REVERSAL" in report.reason_codes
    dte_reason = next(
        item for item in report.reasons if item.code == "DTE_SENSITIVITY_REVERSAL"
    )
    assert dte_reason.scenario == "34_dte"


def test_missing_required_evidence_is_machine_readable_rejection():
    report = evaluate_backtest(
        comparison(
            trade_contributions=(),
            dte_sensitivities=(),
            hedge_frequency_sensitivities=(),
        )
    )

    assert "MISSING_TRADE_CONTRIBUTIONS" in report.reason_codes
    assert "MISSING_DTE_SENSITIVITY" in report.reason_codes
    assert "MISSING_HEDGE_FREQUENCY_SENSITIVITY" in report.reason_codes


def test_missing_evidence_requirements_can_be_disabled_explicitly():
    thresholds = DiagnosticThresholds(
        require_trade_contributions=False,
        require_dte_sensitivities=False,
        require_hedge_frequency_sensitivities=False,
    )
    report = evaluate_backtest(
        comparison(
            trade_contributions=(),
            dte_sensitivities=(),
            hedge_frequency_sensitivities=(),
        ),
        thresholds,
    )
    assert report.accepted


def test_profit_floor_and_tolerance_are_configurable():
    thresholds = DiagnosticThresholds(profit_floor=10.0, comparison_tolerance=0.1)
    assert profit_disappears(
        scenario("reference", 10.2),
        scenario("alternative", 10.1),
        thresholds,
    )
    assert not profit_disappears(
        scenario("reference", 10.1),
        scenario("alternative", 0.0),
        thresholds,
    )


def test_thresholds_and_evidence_validate_bad_inputs():
    with pytest.raises(ValueError, match="between"):
        DiagnosticThresholds(max_overnight_gap_share=1.1)
    with pytest.raises(ValueError, match="positive integer"):
        DiagnosticThresholds(short_vol_top_n=0)
    with pytest.raises(ValueError, match="unique"):
        comparison(
            trade_contributions=(
                TradeContribution("duplicate", "short", 1.0),
                TradeContribution("duplicate", "long", 2.0),
            )
        )

