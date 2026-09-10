"""Deterministic pre-trade risk controls for the capped XSP straddle.

The module deliberately accepts dollar losses that were calculated by a pricing
engine rather than embedding a pricing model.  That keeps the controls pure and
makes the unit convention explicit:

* ``one_vol_point_vega_loss_per_package`` is dollars lost by one package for a
  one percentage-point adverse volatility move (the option multiplier is
  already included), and
* ``ShortStressEstimate.loss_per_package`` is the positive dollar loss for the
  complete package under the stated joint spot/volatility scenario.

Raw short straddles cannot be passed to these functions.  A trade must be a
validated :class:`CappedPackage`, containing the two protective wings.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple


LONG = "long"
SHORT = "short"
CALL = "call"
PUT = "put"


def _finite(value: float, field_name: str) -> float:
    """Return ``value`` as a float, raising a useful error if it is not finite."""

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must be a finite number" % field_name) from exc
    if not math.isfinite(result):
        raise ValueError("%s must be a finite number" % field_name)
    return result


def _nonnegative(value: float, field_name: str) -> float:
    result = _finite(value, field_name)
    if result < 0.0:
        raise ValueError("%s must be non-negative" % field_name)
    return result


@dataclass(frozen=True)
class LegQuote:
    """Minimal quote data needed by the independent risk layer.

    ``label`` is an audit identifier; it need not be an exchange symbol.  The
    premium used for the spread test is the quote midpoint.  A zero quote has a
    zero relative spread, while a positive ask with a zero midpoint is treated
    as infinitely wide.
    """

    label: str
    option_type: str
    strike: float
    bid: float
    ask: float

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be a non-empty string")
        if self.option_type not in (CALL, PUT):
            raise ValueError("option_type must be 'call' or 'put'")
        strike = _finite(self.strike, "strike")
        bid = _nonnegative(self.bid, "bid")
        ask = _nonnegative(self.ask, "ask")
        if strike <= 0.0:
            raise ValueError("strike must be positive")
        if ask < bid:
            raise ValueError("ask must be greater than or equal to bid")
        object.__setattr__(self, "strike", strike)
        object.__setattr__(self, "bid", bid)
        object.__setattr__(self, "ask", ask)

    @property
    def midpoint(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def relative_spread(self) -> float:
        midpoint = self.midpoint
        if midpoint == 0.0:
            return 0.0 if self.spread == 0.0 else math.inf
        return self.spread / midpoint


@dataclass(frozen=True)
class CappedPackage:
    """A one-to-one ATM straddle with an upper-call and lower-put cap."""

    expiry: str
    atm_call: LegQuote
    atm_put: LegQuote
    upper_call: LegQuote
    lower_put: LegQuote
    multiplier: int = 100

    def __post_init__(self) -> None:
        if not isinstance(self.expiry, str) or not self.expiry.strip():
            raise ValueError("expiry must be a non-empty string")
        if isinstance(self.multiplier, bool) or not isinstance(self.multiplier, int):
            raise ValueError("multiplier must be a positive integer")
        if self.multiplier <= 0:
            raise ValueError("multiplier must be a positive integer")
        if self.atm_call.option_type != CALL:
            raise ValueError("atm_call must be a call")
        if self.atm_put.option_type != PUT:
            raise ValueError("atm_put must be a put")
        if self.upper_call.option_type != CALL:
            raise ValueError("upper_call must be a call")
        if self.lower_put.option_type != PUT:
            raise ValueError("lower_put must be a put")
        if not math.isclose(
            self.atm_call.strike,
            self.atm_put.strike,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("ATM call and put strikes must match")
        atm_strike = self.atm_call.strike
        if not self.lower_put.strike < atm_strike:
            raise ValueError("lower put wing must be below the ATM strike")
        if not self.upper_call.strike > atm_strike:
            raise ValueError("upper call wing must be above the ATM strike")

    @property
    def legs(self) -> Tuple[LegQuote, LegQuote, LegQuote, LegQuote]:
        return (self.atm_call, self.atm_put, self.upper_call, self.lower_put)

    @property
    def executable_buy_price(self) -> float:
        """Debit in index points to buy the complete capped package."""

        return (
            self.atm_call.ask
            + self.atm_put.ask
            - self.upper_call.bid
            - self.lower_put.bid
        )

    @property
    def executable_sell_price(self) -> float:
        """Credit in index points to sell the complete capped package."""

        return (
            self.atm_call.bid
            + self.atm_put.bid
            - self.upper_call.ask
            - self.lower_put.ask
        )

    @property
    def long_debit_per_package(self) -> float:
        """Non-negative cash debit, including the option multiplier."""

        return max(0.0, self.executable_buy_price * self.multiplier)


@dataclass(frozen=True)
class ShortStressEstimate:
    """Priced loss for the required joint overnight short-vol stress."""

    loss_per_package: float
    underlying_move_fraction: float = -0.08
    volatility_point_increase: float = 15.0

    def __post_init__(self) -> None:
        loss = _nonnegative(self.loss_per_package, "loss_per_package")
        spot_move = _finite(self.underlying_move_fraction, "underlying_move_fraction")
        vol_move = _nonnegative(
            self.volatility_point_increase,
            "volatility_point_increase",
        )
        object.__setattr__(self, "loss_per_package", loss)
        object.__setattr__(self, "underlying_move_fraction", spot_move)
        object.__setattr__(self, "volatility_point_increase", vol_move)


@dataclass(frozen=True)
class RiskLimits:
    """Research risk defaults expressed as fractions of current capital."""

    one_vol_vega_fraction: float = 0.001
    long_debit_fraction: float = 0.005
    short_stress_fraction_per_expiry: float = 0.01
    max_overlapping_expirations: int = 3
    max_leg_relative_spread: float = 0.10
    short_stress_underlying_move_fraction: float = -0.08
    short_stress_volatility_point_increase: float = 15.0

    def __post_init__(self) -> None:
        for name in (
            "one_vol_vega_fraction",
            "long_debit_fraction",
            "short_stress_fraction_per_expiry",
            "max_leg_relative_spread",
        ):
            value = _nonnegative(getattr(self, name), name)
            object.__setattr__(self, name, value)
        if self.one_vol_vega_fraction > 1.0:
            raise ValueError("one_vol_vega_fraction cannot exceed 1")
        if self.long_debit_fraction > 1.0:
            raise ValueError("long_debit_fraction cannot exceed 1")
        if self.short_stress_fraction_per_expiry > 1.0:
            raise ValueError("short_stress_fraction_per_expiry cannot exceed 1")
        if isinstance(self.max_overlapping_expirations, bool) or not isinstance(
            self.max_overlapping_expirations,
            int,
        ):
            raise ValueError("max_overlapping_expirations must be a positive integer")
        if self.max_overlapping_expirations <= 0:
            raise ValueError("max_overlapping_expirations must be a positive integer")
        spot_move = _finite(
            self.short_stress_underlying_move_fraction,
            "short_stress_underlying_move_fraction",
        )
        vol_move = _nonnegative(
            self.short_stress_volatility_point_increase,
            "short_stress_volatility_point_increase",
        )
        if spot_move >= 0.0:
            raise ValueError("short_stress_underlying_move_fraction must be negative")
        object.__setattr__(self, "short_stress_underlying_move_fraction", spot_move)
        object.__setattr__(self, "short_stress_volatility_point_increase", vol_move)


@dataclass(frozen=True)
class TradeRiskInput:
    """All exposures needed to size one proposed package trade.

    Existing vega is portfolio-wide.  Existing debit is the capital already
    committed to long option packages.  Existing short stress must include only
    positions sharing the proposed package's expiry.
    """

    capital: float
    direction: str
    package: CappedPackage
    one_vol_point_vega_loss_per_package: float
    short_stress: Optional[ShortStressEstimate] = None
    active_expirations: Tuple[str, ...] = ()
    existing_one_vol_point_vega_loss: float = 0.0
    existing_long_option_debit: float = 0.0
    existing_short_stress_loss_for_expiry: float = 0.0

    def __post_init__(self) -> None:
        capital = _finite(self.capital, "capital")
        if capital <= 0.0:
            raise ValueError("capital must be positive")
        if self.direction not in (LONG, SHORT):
            raise ValueError("direction must be 'long' or 'short'")
        if not isinstance(self.package, CappedPackage):
            raise TypeError("package must be a validated CappedPackage")
        vega = _finite(
            self.one_vol_point_vega_loss_per_package,
            "one_vol_point_vega_loss_per_package",
        )
        if vega <= 0.0:
            raise ValueError("one_vol_point_vega_loss_per_package must be positive")
        expirations = tuple(self.active_expirations)
        if any(not isinstance(expiry, str) or not expiry.strip() for expiry in expirations):
            raise ValueError("active_expirations must contain non-empty strings")
        object.__setattr__(self, "capital", capital)
        object.__setattr__(self, "one_vol_point_vega_loss_per_package", vega)
        object.__setattr__(self, "active_expirations", expirations)
        for name in (
            "existing_one_vol_point_vega_loss",
            "existing_long_option_debit",
            "existing_short_stress_loss_for_expiry",
        ):
            value = _nonnegative(getattr(self, name), name)
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class RiskViolation:
    """Stable, machine-readable explanation of a failed risk control."""

    code: str
    message: str
    observed: Optional[float] = None
    limit: Optional[float] = None
    leg: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "observed": self.observed,
            "limit": self.limit,
            "leg": self.leg,
        }


@dataclass(frozen=True)
class SizingConstraint:
    """Headroom and integer capacity for one quantitative limit."""

    code: str
    limit_dollars: float
    existing_exposure_dollars: float
    exposure_per_package_dollars: float
    remaining_dollars: float
    max_packages: int

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "limit_dollars": self.limit_dollars,
            "existing_exposure_dollars": self.existing_exposure_dollars,
            "exposure_per_package_dollars": self.exposure_per_package_dollars,
            "remaining_dollars": self.remaining_dollars,
            "max_packages": self.max_packages,
        }


@dataclass(frozen=True)
class RiskSizing:
    """Maximum whole-package size after every hard and quantitative check."""

    max_packages: int
    constraints: Tuple[SizingConstraint, ...]
    binding_constraints: Tuple[str, ...]
    violations: Tuple[RiskViolation, ...]

    @property
    def rejected(self) -> bool:
        return self.max_packages == 0

    @property
    def reason_codes(self) -> Tuple[str, ...]:
        return tuple(violation.code for violation in self.violations)

    def to_dict(self) -> dict:
        return {
            "max_packages": self.max_packages,
            "constraints": [constraint.to_dict() for constraint in self.constraints],
            "binding_constraints": list(self.binding_constraints),
            "violations": [violation.to_dict() for violation in self.violations],
        }


@dataclass(frozen=True)
class RiskDecision:
    """Approval decision for a specifically requested integer quantity."""

    approved: bool
    requested_packages: int
    approved_packages: int
    recommended_packages: int
    sizing: RiskSizing
    violations: Tuple[RiskViolation, ...]

    @property
    def reason_codes(self) -> Tuple[str, ...]:
        return tuple(violation.code for violation in self.violations)

    def to_dict(self) -> dict:
        return {
            "approved": self.approved,
            "requested_packages": self.requested_packages,
            "approved_packages": self.approved_packages,
            "recommended_packages": self.recommended_packages,
            "sizing": self.sizing.to_dict(),
            "violations": [violation.to_dict() for violation in self.violations],
        }


def _integer_capacity(
    code: str,
    limit_dollars: float,
    existing_exposure_dollars: float,
    exposure_per_package_dollars: float,
) -> SizingConstraint:
    remaining = limit_dollars - existing_exposure_dollars
    # Tiny floating-point noise at an exact boundary must not remove a package.
    scale = max(1.0, abs(limit_dollars), abs(existing_exposure_dollars))
    adjusted_remaining = remaining + 1e-12 * scale
    if adjusted_remaining <= 0.0:
        capacity = 0
    elif exposure_per_package_dollars == 0.0:
        # The universal vega control is always positive, so another constraint
        # supplies a finite cap.  This sentinel is internal and deterministic.
        capacity = 2**63 - 1
    else:
        capacity = max(0, int(math.floor(adjusted_remaining / exposure_per_package_dollars)))
    return SizingConstraint(
        code=code,
        limit_dollars=limit_dollars,
        existing_exposure_dollars=existing_exposure_dollars,
        exposure_per_package_dollars=exposure_per_package_dollars,
        remaining_dollars=max(0.0, remaining),
        max_packages=capacity,
    )


def _hard_violations(
    trade: TradeRiskInput,
    limits: RiskLimits,
) -> Tuple[RiskViolation, ...]:
    violations = []

    resulting_expirations = set(trade.active_expirations)
    resulting_expirations.add(trade.package.expiry)
    if len(resulting_expirations) > limits.max_overlapping_expirations:
        violations.append(
            RiskViolation(
                code="MAX_OVERLAPPING_EXPIRATIONS",
                message="proposed trade exceeds the overlapping-expiration limit",
                observed=float(len(resulting_expirations)),
                limit=float(limits.max_overlapping_expirations),
            )
        )

    for leg in trade.package.legs:
        if leg.relative_spread > limits.max_leg_relative_spread + 1e-12:
            violations.append(
                RiskViolation(
                    code="LEG_SPREAD_TOO_WIDE",
                    message="leg bid/ask spread exceeds the allowed share of midpoint premium",
                    observed=leg.relative_spread,
                    limit=limits.max_leg_relative_spread,
                    leg=leg.label,
                )
            )

    if trade.direction == SHORT:
        if trade.short_stress is None:
            violations.append(
                RiskViolation(
                    code="SHORT_STRESS_REQUIRED",
                    message="short-vol trades require a priced overnight spot/volatility stress",
                )
            )
        else:
            scenario_matches = math.isclose(
                trade.short_stress.underlying_move_fraction,
                limits.short_stress_underlying_move_fraction,
                rel_tol=0.0,
                abs_tol=1e-12,
            ) and math.isclose(
                trade.short_stress.volatility_point_increase,
                limits.short_stress_volatility_point_increase,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            if not scenario_matches:
                violations.append(
                    RiskViolation(
                        code="SHORT_STRESS_SCENARIO_MISMATCH",
                        message="short stress must combine the configured spot and volatility shocks",
                    )
                )
    return tuple(violations)


def size_trade(
    trade: TradeRiskInput,
    limits: Optional[RiskLimits] = None,
) -> RiskSizing:
    """Return the largest allowed number of whole capped packages.

    Hard failures (liquidity, overlap, an absent/mismatched short stress) force a
    zero size.  Otherwise the minimum capacity across portfolio vega, long debit
    and same-expiry short stress is used.
    """

    active_limits = limits if limits is not None else RiskLimits()
    hard_violations = list(_hard_violations(trade, active_limits))

    constraints = [
        _integer_capacity(
            code="ONE_VOL_VEGA",
            limit_dollars=trade.capital * active_limits.one_vol_vega_fraction,
            existing_exposure_dollars=trade.existing_one_vol_point_vega_loss,
            exposure_per_package_dollars=trade.one_vol_point_vega_loss_per_package,
        )
    ]

    if trade.direction == LONG:
        constraints.append(
            _integer_capacity(
                code="LONG_OPTION_DEBIT",
                limit_dollars=trade.capital * active_limits.long_debit_fraction,
                existing_exposure_dollars=trade.existing_long_option_debit,
                exposure_per_package_dollars=trade.package.long_debit_per_package,
            )
        )
    elif trade.short_stress is not None:
        constraints.append(
            _integer_capacity(
                code="SHORT_STRESS_PER_EXPIRY",
                limit_dollars=(
                    trade.capital * active_limits.short_stress_fraction_per_expiry
                ),
                existing_exposure_dollars=trade.existing_short_stress_loss_for_expiry,
                exposure_per_package_dollars=trade.short_stress.loss_per_package,
            )
        )

    quantitative_max = min(constraint.max_packages for constraint in constraints)
    if quantitative_max == 0:
        for constraint in constraints:
            if constraint.max_packages == 0:
                hard_violations.append(
                    RiskViolation(
                        code=constraint.code + "_CAPACITY_EXHAUSTED",
                        message="no whole package fits within this risk budget",
                        observed=constraint.existing_exposure_dollars
                        + constraint.exposure_per_package_dollars,
                        limit=constraint.limit_dollars,
                    )
                )

    max_packages = 0 if hard_violations else quantitative_max
    binding = tuple(
        constraint.code
        for constraint in constraints
        if constraint.max_packages == quantitative_max
    )
    return RiskSizing(
        max_packages=max_packages,
        constraints=tuple(constraints),
        binding_constraints=binding,
        violations=tuple(hard_violations),
    )


def evaluate_trade_risk(
    trade: TradeRiskInput,
    requested_packages: int,
    limits: Optional[RiskLimits] = None,
) -> RiskDecision:
    """Approve a requested whole-package quantity or return a zero approval."""

    if isinstance(requested_packages, bool) or not isinstance(requested_packages, int):
        raise ValueError("requested_packages must be a positive integer")
    if requested_packages <= 0:
        raise ValueError("requested_packages must be a positive integer")

    sizing = size_trade(trade, limits)
    violations = list(sizing.violations)
    if not violations and requested_packages > sizing.max_packages:
        violations.append(
            RiskViolation(
                code="REQUESTED_SIZE_EXCEEDS_LIMIT",
                message="requested package count exceeds the risk-sized maximum",
                observed=float(requested_packages),
                limit=float(sizing.max_packages),
            )
        )
    approved = not violations
    return RiskDecision(
        approved=approved,
        requested_packages=requested_packages,
        approved_packages=requested_packages if approved else 0,
        recommended_packages=min(requested_packages, sizing.max_packages),
        sizing=sizing,
        violations=tuple(violations),
    )


__all__ = [
    "CALL",
    "PUT",
    "LONG",
    "SHORT",
    "CappedPackage",
    "LegQuote",
    "RiskDecision",
    "RiskLimits",
    "RiskSizing",
    "RiskViolation",
    "ShortStressEstimate",
    "SizingConstraint",
    "TradeRiskInput",
    "evaluate_trade_risk",
    "size_trade",
]
