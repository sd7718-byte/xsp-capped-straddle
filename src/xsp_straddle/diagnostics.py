"""Auditable rejection diagnostics for capped-straddle backtests.

The checks in this module intentionally answer a narrow question: does a
reported backtest survive the six falsification tests specified by the research
brief?  They do not manufacture a performance score or silently average away a
failed scenario.  Every rejection has a stable code, observed value, threshold
and optional scenario context suitable for JSON output.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple


def _finite(value: float, field_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must be a finite number" % field_name) from exc
    if not math.isfinite(result):
        raise ValueError("%s must be a finite number" % field_name)
    return result


@dataclass(frozen=True)
class ScenarioResult:
    """Net result from one otherwise-comparable backtest scenario."""

    name: str
    net_profit: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("scenario name must be a non-empty string")
        object.__setattr__(self, "net_profit", _finite(self.net_profit, "net_profit"))


@dataclass(frozen=True)
class TradeContribution:
    """Net lifetime contribution of one trade under executable assumptions."""

    trade_id: str
    direction: str
    net_profit: float

    def __post_init__(self) -> None:
        if not isinstance(self.trade_id, str) or not self.trade_id.strip():
            raise ValueError("trade_id must be a non-empty string")
        if self.direction not in ("long", "short", "long_vol", "short_vol"):
            raise ValueError(
                "direction must be 'long', 'short', 'long_vol', or 'short_vol'"
            )
        object.__setattr__(self, "net_profit", _finite(self.net_profit, "net_profit"))

    @property
    def is_short_vol(self) -> bool:
        return self.direction in ("short", "short_vol")


@dataclass(frozen=True)
class ReturnAttribution:
    """Like-for-like return or P&L attribution in a single unit.

    Both fields may be dollars, return points, or percentage points, but must
    use the same unit.  ``delta_hedged_return`` is the complete return and
    ``overnight_gap_return`` is the signed component attributed to closed-market
    gaps.
    """

    delta_hedged_return: float
    overnight_gap_return: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "delta_hedged_return",
            _finite(self.delta_hedged_return, "delta_hedged_return"),
        )
        object.__setattr__(
            self,
            "overnight_gap_return",
            _finite(self.overnight_gap_return, "overnight_gap_return"),
        )


@dataclass(frozen=True)
class SensitivityResult:
    """Result from one nearby DTE or hedge-frequency setting."""

    setting: str
    net_profit: float

    def __post_init__(self) -> None:
        if not isinstance(self.setting, str) or not self.setting.strip():
            raise ValueError("sensitivity setting must be a non-empty string")
        object.__setattr__(self, "net_profit", _finite(self.net_profit, "net_profit"))


@dataclass(frozen=True)
class BacktestComparison:
    """Evidence bundle needed to run all rejection diagnostics.

    The five scenario fields should differ only in the assumption named by the
    field.  DTE and hedge-frequency sequences should contain nearby, plausible
    alternatives rather than broad parameter searches.
    """

    approximate_iv: ScenarioResult
    exact_iv: ScenarioResult
    midpoint_fills: ScenarioResult
    executable_fills: ScenarioResult
    doubled_costs: ScenarioResult
    trade_contributions: Tuple[TradeContribution, ...]
    return_attribution: ReturnAttribution
    dte_sensitivities: Tuple[SensitivityResult, ...]
    hedge_frequency_sensitivities: Tuple[SensitivityResult, ...]

    def __post_init__(self) -> None:
        for name in (
            "approximate_iv",
            "exact_iv",
            "midpoint_fills",
            "executable_fills",
            "doubled_costs",
        ):
            if not isinstance(getattr(self, name), ScenarioResult):
                raise TypeError("%s must be a ScenarioResult" % name)
        if not isinstance(self.return_attribution, ReturnAttribution):
            raise TypeError("return_attribution must be a ReturnAttribution")

        contributions = tuple(self.trade_contributions)
        dte = tuple(self.dte_sensitivities)
        hedge = tuple(self.hedge_frequency_sensitivities)
        if any(not isinstance(item, TradeContribution) for item in contributions):
            raise TypeError("trade_contributions must contain TradeContribution values")
        if any(not isinstance(item, SensitivityResult) for item in dte):
            raise TypeError("dte_sensitivities must contain SensitivityResult values")
        if any(not isinstance(item, SensitivityResult) for item in hedge):
            raise TypeError(
                "hedge_frequency_sensitivities must contain SensitivityResult values"
            )
        trade_ids = [item.trade_id for item in contributions]
        if len(trade_ids) != len(set(trade_ids)):
            raise ValueError("trade_contributions must have unique trade_id values")
        object.__setattr__(self, "trade_contributions", contributions)
        object.__setattr__(self, "dte_sensitivities", dte)
        object.__setattr__(self, "hedge_frequency_sensitivities", hedge)


@dataclass(frozen=True)
class DiagnosticThresholds:
    """Transparent decision boundaries for the rejection rules.

    At the defaults, profit must be strictly positive, the largest three
    profitable short-vol trades may not explain more than half of executable
    lifetime profit, and overnight gaps may not explain more than half of the
    absolute delta-hedged result.
    """

    profit_floor: float = 0.0
    short_vol_top_n: int = 3
    max_top_short_vol_profit_share: float = 0.50
    max_overnight_gap_share: float = 0.50
    comparison_tolerance: float = 1e-9
    require_trade_contributions: bool = True
    require_dte_sensitivities: bool = True
    require_hedge_frequency_sensitivities: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "profit_floor", _finite(self.profit_floor, "profit_floor"))
        if isinstance(self.short_vol_top_n, bool) or not isinstance(
            self.short_vol_top_n,
            int,
        ):
            raise ValueError("short_vol_top_n must be a positive integer")
        if self.short_vol_top_n <= 0:
            raise ValueError("short_vol_top_n must be a positive integer")
        for name in (
            "max_top_short_vol_profit_share",
            "max_overnight_gap_share",
        ):
            value = _finite(getattr(self, name), name)
            if not 0.0 <= value <= 1.0:
                raise ValueError("%s must be between 0 and 1" % name)
            object.__setattr__(self, name, value)
        tolerance = _finite(self.comparison_tolerance, "comparison_tolerance")
        if tolerance <= 0.0:
            raise ValueError("comparison_tolerance must be positive")
        object.__setattr__(self, "comparison_tolerance", tolerance)
        for name in (
            "require_trade_contributions",
            "require_dte_sensitivities",
            "require_hedge_frequency_sensitivities",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError("%s must be a bool" % name)


@dataclass(frozen=True)
class DiagnosticReason:
    """One stable rejection reason with numerical evidence."""

    code: str
    message: str
    observed: Optional[float] = None
    threshold: Optional[float] = None
    scenario: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "observed": self.observed,
            "threshold": self.threshold,
            "scenario": self.scenario,
        }


@dataclass(frozen=True)
class DiagnosticMetrics:
    """Intermediate metrics retained to make a report independently auditable."""

    top_short_vol_profit_share: Optional[float]
    overnight_gap_share: float
    weakest_dte_profit: Optional[float]
    weakest_hedge_frequency_profit: Optional[float]

    def to_dict(self) -> dict:
        return {
            "top_short_vol_profit_share": self.top_short_vol_profit_share,
            "overnight_gap_share": self.overnight_gap_share,
            "weakest_dte_profit": self.weakest_dte_profit,
            "weakest_hedge_frequency_profit": self.weakest_hedge_frequency_profit,
        }


@dataclass(frozen=True)
class DiagnosticReport:
    """Final pass/reject decision and all non-short-circuited reasons."""

    accepted: bool
    reasons: Tuple[DiagnosticReason, ...]
    passed_checks: Tuple[str, ...]
    metrics: DiagnosticMetrics

    @property
    def rejected(self) -> bool:
        return not self.accepted

    @property
    def reason_codes(self) -> Tuple[str, ...]:
        return tuple(reason.code for reason in self.reasons)

    def to_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "rejected": self.rejected,
            "reasons": [reason.to_dict() for reason in self.reasons],
            "passed_checks": list(self.passed_checks),
            "metrics": self.metrics.to_dict(),
        }


def is_profitable(
    result: ScenarioResult,
    thresholds: Optional[DiagnosticThresholds] = None,
) -> bool:
    """Return whether a scenario clears the configured profit floor."""

    active = thresholds if thresholds is not None else DiagnosticThresholds()
    return result.net_profit > active.profit_floor + active.comparison_tolerance


def profit_disappears(
    profitable_assumption: ScenarioResult,
    conservative_assumption: ScenarioResult,
    thresholds: Optional[DiagnosticThresholds] = None,
) -> bool:
    """Return true when a profitable reference becomes non-profitable."""

    active = thresholds if thresholds is not None else DiagnosticThresholds()
    return is_profitable(profitable_assumption, active) and not is_profitable(
        conservative_assumption,
        active,
    )


def top_short_vol_profit_share(
    contributions: Tuple[TradeContribution, ...],
    lifetime_profit: float,
    top_n: int = 3,
    tolerance: float = 1e-9,
) -> Optional[float]:
    """Share of net lifetime profit supplied by the top profitable short trades.

    ``None`` means lifetime profit is not positive enough for a meaningful
    concentration ratio.  Negative short-trade contributions are deliberately
    excluded from the numerator; subtracting them would conceal dependence on a
    small number of winning trades.
    """

    total = _finite(lifetime_profit, "lifetime_profit")
    tol = _finite(tolerance, "tolerance")
    if tol <= 0.0:
        raise ValueError("tolerance must be positive")
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n <= 0:
        raise ValueError("top_n must be a positive integer")
    if total <= tol:
        return None
    short_winners = sorted(
        (
            contribution.net_profit
            for contribution in contributions
            if contribution.is_short_vol and contribution.net_profit > 0.0
        ),
        reverse=True,
    )
    return sum(short_winners[:top_n]) / total


def overnight_gap_share(
    attribution: ReturnAttribution,
    tolerance: float = 1e-9,
) -> float:
    """Absolute overnight contribution divided by absolute total hedged return."""

    tol = _finite(tolerance, "tolerance")
    if tol <= 0.0:
        raise ValueError("tolerance must be positive")
    denominator = max(abs(attribution.delta_hedged_return), tol)
    return abs(attribution.overnight_gap_return) / denominator


def _weakest(sensitivities: Tuple[SensitivityResult, ...]) -> Optional[SensitivityResult]:
    if not sensitivities:
        return None
    # Setting is a deterministic tie-breaker, independent of input ordering.
    return min(sensitivities, key=lambda result: (result.net_profit, result.setting))


def evaluate_backtest(
    comparison: BacktestComparison,
    thresholds: Optional[DiagnosticThresholds] = None,
) -> DiagnosticReport:
    """Run every rejection rule and return all failures in a stable order."""

    if not isinstance(comparison, BacktestComparison):
        raise TypeError("comparison must be a BacktestComparison")
    active = thresholds if thresholds is not None else DiagnosticThresholds()
    reasons = []
    passed = []

    if profit_disappears(comparison.approximate_iv, comparison.exact_iv, active):
        reasons.append(
            DiagnosticReason(
                code="EXACT_IV_REMOVES_PROFIT",
                message="profit disappears when exact IV replaces approximate IV",
                observed=comparison.exact_iv.net_profit,
                threshold=active.profit_floor,
                scenario=comparison.exact_iv.name,
            )
        )
    else:
        passed.append("EXACT_IV_ROBUST")

    if profit_disappears(comparison.midpoint_fills, comparison.executable_fills, active):
        reasons.append(
            DiagnosticReason(
                code="MIDPOINT_ONLY_PROFIT",
                message="profit exists at midpoint fills but not executable fills",
                observed=comparison.executable_fills.net_profit,
                threshold=active.profit_floor,
                scenario=comparison.executable_fills.name,
            )
        )
    else:
        passed.append("EXECUTABLE_FILL_ROBUST")

    if not is_profitable(comparison.executable_fills, active):
        reasons.append(
            DiagnosticReason(
                code="EXECUTABLE_BASELINE_UNPROFITABLE",
                message="the executable-fill baseline does not clear the profit floor",
                observed=comparison.executable_fills.net_profit,
                threshold=active.profit_floor,
                scenario=comparison.executable_fills.name,
            )
        )
    else:
        passed.append("EXECUTABLE_BASELINE_PROFITABLE")

    if not is_profitable(comparison.doubled_costs, active):
        reasons.append(
            DiagnosticReason(
                code="DOUBLED_COSTS_UNPROFITABLE",
                message="the strategy is unprofitable with transaction costs doubled",
                observed=comparison.doubled_costs.net_profit,
                threshold=active.profit_floor,
                scenario=comparison.doubled_costs.name,
            )
        )
    else:
        passed.append("DOUBLED_COSTS_ROBUST")

    concentration = top_short_vol_profit_share(
        comparison.trade_contributions,
        comparison.executable_fills.net_profit,
        top_n=active.short_vol_top_n,
        tolerance=active.comparison_tolerance,
    )
    if not comparison.trade_contributions and active.require_trade_contributions:
        reasons.append(
            DiagnosticReason(
                code="MISSING_TRADE_CONTRIBUTIONS",
                message="trade-level contributions are required to test profit concentration",
            )
        )
    elif (
        concentration is not None
        and concentration
        > active.max_top_short_vol_profit_share + active.comparison_tolerance
    ):
        reasons.append(
            DiagnosticReason(
                code="SHORT_VOL_PROFIT_CONCENTRATION",
                message="a few profitable short-vol trades explain too much lifetime profit",
                observed=concentration,
                threshold=active.max_top_short_vol_profit_share,
                scenario="top_%d_short_vol_trades" % active.short_vol_top_n,
            )
        )
    else:
        passed.append("SHORT_VOL_PROFIT_DIVERSIFIED")

    gap_share = overnight_gap_share(
        comparison.return_attribution,
        tolerance=active.comparison_tolerance,
    )
    if gap_share > active.max_overnight_gap_share + active.comparison_tolerance:
        reasons.append(
            DiagnosticReason(
                code="OVERNIGHT_GAPS_DOMINATE",
                message="overnight gaps dominate the absolute delta-hedged return",
                observed=gap_share,
                threshold=active.max_overnight_gap_share,
            )
        )
    else:
        passed.append("OVERNIGHT_GAP_ATTRIBUTION_ACCEPTABLE")

    baseline_profitable = is_profitable(comparison.executable_fills, active)
    weakest_dte = _weakest(comparison.dte_sensitivities)
    if weakest_dte is None and active.require_dte_sensitivities:
        reasons.append(
            DiagnosticReason(
                code="MISSING_DTE_SENSITIVITY",
                message="nearby entry-DTE sensitivity results are required",
            )
        )
    elif (
        weakest_dte is not None
        and baseline_profitable
        and not is_profitable(
            ScenarioResult(weakest_dte.setting, weakest_dte.net_profit),
            active,
        )
    ):
        reasons.append(
            DiagnosticReason(
                code="DTE_SENSITIVITY_REVERSAL",
                message="a nearby entry-DTE setting reverses baseline profitability",
                observed=weakest_dte.net_profit,
                threshold=active.profit_floor,
                scenario=weakest_dte.setting,
            )
        )
    else:
        passed.append("DTE_SENSITIVITY_ROBUST")

    weakest_hedge = _weakest(comparison.hedge_frequency_sensitivities)
    if weakest_hedge is None and active.require_hedge_frequency_sensitivities:
        reasons.append(
            DiagnosticReason(
                code="MISSING_HEDGE_FREQUENCY_SENSITIVITY",
                message="nearby hedge-frequency sensitivity results are required",
            )
        )
    elif (
        weakest_hedge is not None
        and baseline_profitable
        and not is_profitable(
            ScenarioResult(weakest_hedge.setting, weakest_hedge.net_profit),
            active,
        )
    ):
        reasons.append(
            DiagnosticReason(
                code="HEDGE_FREQUENCY_SENSITIVITY_REVERSAL",
                message="a nearby hedge-frequency setting reverses baseline profitability",
                observed=weakest_hedge.net_profit,
                threshold=active.profit_floor,
                scenario=weakest_hedge.setting,
            )
        )
    else:
        passed.append("HEDGE_FREQUENCY_SENSITIVITY_ROBUST")

    metrics = DiagnosticMetrics(
        top_short_vol_profit_share=concentration,
        overnight_gap_share=gap_share,
        weakest_dte_profit=None if weakest_dte is None else weakest_dte.net_profit,
        weakest_hedge_frequency_profit=(
            None if weakest_hedge is None else weakest_hedge.net_profit
        ),
    )
    return DiagnosticReport(
        accepted=not reasons,
        reasons=tuple(reasons),
        passed_checks=tuple(passed),
        metrics=metrics,
    )


__all__ = [
    "BacktestComparison",
    "DiagnosticMetrics",
    "DiagnosticReason",
    "DiagnosticReport",
    "DiagnosticThresholds",
    "ReturnAttribution",
    "ScenarioResult",
    "SensitivityResult",
    "TradeContribution",
    "evaluate_backtest",
    "is_profitable",
    "overnight_gap_share",
    "profit_disappears",
    "top_short_vol_profit_share",
]
