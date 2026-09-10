"""Delta-hedging mechanics shared by live signal simulation and backtests."""

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Sequence, Tuple


@dataclass(frozen=True)
class HedgeConversion:
    contracts: int
    target_dollar_delta: float
    achieved_dollar_delta: float
    residual_dollar_delta: float


@dataclass(frozen=True)
class HedgeFill:
    units_before: float
    units_after: float
    traded_units: float
    execution_price: float
    commission: float
    cash_change: float


def capped_package_delta(
    atm_call_delta: float,
    atm_put_delta: float,
    upper_call_delta: float,
    lower_put_delta: float,
) -> float:
    """Delta of ``C(K0)+P(K0)-C(KU)-P(KL)`` per package."""

    values = (atm_call_delta, atm_put_delta, upper_call_delta, lower_put_delta)
    if not all(isfinite(float(value)) for value in values):
        raise ValueError("all leg deltas must be finite")
    return float(atm_call_delta + atm_put_delta - upper_call_delta - lower_put_delta)


def desired_hedge_units(
    package_delta: float,
    packages: int,
    option_multiplier: float = 100.0,
    direction: int = 1,
) -> float:
    """Underlying-equivalent units required to neutralize the option delta.

    ``direction`` is ``1`` for a long package and ``-1`` for a short package.
    """

    if direction not in (-1, 1):
        raise ValueError("direction must be 1 (long) or -1 (short)")
    if packages < 0:
        raise ValueError("packages cannot be negative")
    if not isfinite(option_multiplier) or option_multiplier <= 0.0:
        raise ValueError("option_multiplier must be positive and finite")
    return -direction * packages * option_multiplier * float(package_delta)


def residual_delta_exceeds_trigger(
    current_hedge_units: float,
    target_hedge_units: float,
    packages: int,
    option_multiplier: float = 100.0,
    threshold_delta_per_package: float = 0.10,
) -> bool:
    """Whether residual delta breaches the per-package intraday trigger."""

    if packages <= 0:
        return False
    threshold_units = packages * option_multiplier * threshold_delta_per_package
    return abs(float(target_hedge_units) - float(current_hedge_units)) > threshold_units


def execute_underlying_hedge(
    current_units: float,
    target_units: float,
    bid: float,
    ask: float,
    commission_per_unit: float = 0.0,
) -> HedgeFill:
    """Execute a spot-equivalent hedge at the adverse side of the market."""

    if bid <= 0.0 or ask <= 0.0 or bid > ask:
        raise ValueError("hedge bid/ask must be positive and non-crossed")
    if commission_per_unit < 0.0:
        raise ValueError("commission_per_unit cannot be negative")
    traded = float(target_units) - float(current_units)
    execution_price = ask if traded > 0.0 else bid if traded < 0.0 else (bid + ask) / 2.0
    commission = abs(traded) * commission_per_unit
    cash_change = -traded * execution_price - commission
    return HedgeFill(
        units_before=float(current_units),
        units_after=float(target_units),
        traded_units=traded,
        execution_price=execution_price,
        commission=commission,
        cash_change=cash_change,
    )


def convert_dollar_delta_to_futures(
    target_underlying_units: float,
    reference_spot: float,
    futures_price: float,
    futures_multiplier: float,
) -> HedgeConversion:
    """Round a target dollar delta to the nearest MES/ES-style contract count.

    Pass multiplier ``5`` for MES or ``50`` for ES.  The reference and futures
    prices may differ (for example XSP versus the full-sized S&P 500 index); the
    conversion is therefore performed in dollar-delta space.
    """

    if reference_spot <= 0.0 or futures_price <= 0.0 or futures_multiplier <= 0.0:
        raise ValueError("prices and futures_multiplier must be positive")
    target = float(target_underlying_units) * reference_spot
    per_contract = futures_price * futures_multiplier
    contracts = int(round(target / per_contract))
    achieved = contracts * per_contract
    return HedgeConversion(
        contracts=contracts,
        target_dollar_delta=target,
        achieved_dollar_delta=achieved,
        residual_dollar_delta=target - achieved,
    )

