"""Conservative pre-filter for a caller-supplied approximate implied volatility.

The paper-specific approximate-IV formula is intentionally not implemented here:
it was not included in the strategy brief.  This module only applies the uniform
error envelope from the brief.  Every executable signal must subsequently use an
exact implied-volatility inversion.
"""

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Optional


APPROX_IV_LOWER_MULTIPLIER = 0.9599
APPROX_IV_UPPER_MULTIPLIER = 1.1284


class PreliminarySide(str, Enum):
    """Outcome of the conservative approximate-IV screen."""

    LONG = "long"
    SHORT = "short"
    NONE = "none"


@dataclass(frozen=True)
class ApproximateIVEnvelope:
    """Guaranteed exact-IV envelope induced by an approximate IV.

    All volatilities are decimal annualized values (``0.20`` means 20%).
    """

    lower: float
    upper: float


@dataclass(frozen=True)
class PreliminaryScreen:
    side: PreliminarySide
    long_margin: float
    short_margin: float
    exact_bid_envelope: ApproximateIVEnvelope
    exact_ask_envelope: ApproximateIVEnvelope


def _validate_vol(name: str, value: float) -> float:
    value = float(value)
    if not isfinite(value) or value < 0.0:
        raise ValueError("{} must be a finite, non-negative decimal volatility".format(name))
    return value


def exact_iv_envelope(approximate_iv: float) -> ApproximateIVEnvelope:
    """Return the paper's uniform lower/upper bounds for exact Black-Scholes IV."""

    approximate_iv = _validate_vol("approximate_iv", approximate_iv)
    return ApproximateIVEnvelope(
        lower=APPROX_IV_LOWER_MULTIPLIER * approximate_iv,
        upper=APPROX_IV_UPPER_MULTIPLIER * approximate_iv,
    )


def conservative_preliminary_screen(
    approximate_bid_iv: float,
    approximate_ask_iv: float,
    forecast_lower_vol: float,
    forecast_upper_vol: float,
    cost_in_vol_points: float,
) -> PreliminaryScreen:
    """Apply the brief's conservative long/short pre-filter.

    ``cost_in_vol_points`` is expressed as decimal volatility.  The function
    reports margins even when neither side passes, which is useful for audit
    logs.  If both sides appear to pass because of inconsistent input bounds,
    the function raises instead of selecting an arbitrary direction.
    """

    approximate_bid_iv = _validate_vol("approximate_bid_iv", approximate_bid_iv)
    approximate_ask_iv = _validate_vol("approximate_ask_iv", approximate_ask_iv)
    forecast_lower_vol = _validate_vol("forecast_lower_vol", forecast_lower_vol)
    forecast_upper_vol = _validate_vol("forecast_upper_vol", forecast_upper_vol)
    cost_in_vol_points = _validate_vol("cost_in_vol_points", cost_in_vol_points)
    if approximate_bid_iv > approximate_ask_iv:
        raise ValueError("approximate bid IV cannot exceed approximate ask IV")
    if forecast_lower_vol > forecast_upper_vol:
        raise ValueError("forecast lower bound cannot exceed upper bound")

    bid_envelope = exact_iv_envelope(approximate_bid_iv)
    ask_envelope = exact_iv_envelope(approximate_ask_iv)
    long_margin = forecast_lower_vol - ask_envelope.upper - cost_in_vol_points
    short_margin = bid_envelope.lower - cost_in_vol_points - forecast_upper_vol
    passes_long = long_margin > 0.0
    passes_short = short_margin > 0.0
    if passes_long and passes_short:
        raise ValueError("inconsistent inputs make both long and short screens pass")
    side = (
        PreliminarySide.LONG
        if passes_long
        else PreliminarySide.SHORT
        if passes_short
        else PreliminarySide.NONE
    )
    return PreliminaryScreen(
        side=side,
        long_margin=long_margin,
        short_margin=short_margin,
        exact_bid_envelope=bid_envelope,
        exact_ask_envelope=ask_envelope,
    )


def maybe_screen(
    approximate_bid_iv: Optional[float],
    approximate_ask_iv: Optional[float],
    forecast_lower_vol: float,
    forecast_upper_vol: float,
    cost_in_vol_points: float,
) -> Optional[PreliminaryScreen]:
    """Run the screen when both approximate quotes exist, otherwise return ``None``.

    Skipping this computational pre-filter does not relax final execution rules;
    the caller must proceed to exact IV and package-level repricing.
    """

    if approximate_bid_iv is None and approximate_ask_iv is None:
        return None
    if approximate_bid_iv is None or approximate_ask_iv is None:
        raise ValueError("approximate bid and ask IV must be supplied together")
    return conservative_preliminary_screen(
        approximate_bid_iv,
        approximate_ask_iv,
        forecast_lower_vol,
        forecast_upper_vol,
        cost_in_vol_points,
    )

