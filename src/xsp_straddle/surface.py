"""Exact-IV total-variance surface and conservative parallel variance shifts."""

from dataclasses import dataclass
from math import exp, isfinite, log, sqrt
from typing import Iterable, Sequence, Tuple

import numpy as np


class SurfaceError(ValueError):
    """Raised for an invalid or non-positive shifted volatility surface."""


@dataclass(frozen=True)
class SurfacePoint:
    strike: float
    implied_vol: float


class TotalVarianceSurface:
    """Piecewise-linear total variance over log-forward moneyness.

    Inputs must be exact Black-Scholes implied volatilities from the current
    market.  Interpolation occurs in total variance, with flat extrapolation
    beyond the observed wings to avoid manufacturing extreme skew.  The fair
    surface adds ``T * (q**2 - sigma_atm**2)`` at every moneyness, preserving the
    currently observed skew as specified in the strategy brief.
    """

    def __init__(
        self,
        forward: float,
        maturity_years: float,
        points: Sequence[SurfacePoint],
    ) -> None:
        if not isfinite(forward) or forward <= 0.0:
            raise SurfaceError("forward must be positive and finite")
        if not isfinite(maturity_years) or maturity_years <= 0.0:
            raise SurfaceError("maturity_years must be positive and finite")
        if len(points) < 2:
            raise SurfaceError("at least two surface points are required")
        ordered = sorted(points, key=lambda point: point.strike)
        strikes = np.asarray([float(point.strike) for point in ordered], dtype=float)
        vols = np.asarray([float(point.implied_vol) for point in ordered], dtype=float)
        if not np.isfinite(strikes).all() or (strikes <= 0.0).any():
            raise SurfaceError("surface strikes must be positive and finite")
        if len(np.unique(strikes)) != len(strikes):
            raise SurfaceError("surface strikes must be unique")
        if not np.isfinite(vols).all() or (vols <= 0.0).any():
            raise SurfaceError("surface implied volatilities must be positive and finite")
        self.forward = float(forward)
        self.maturity_years = float(maturity_years)
        self._strikes = strikes
        self._log_moneyness = np.log(strikes / self.forward)
        self._total_variance = self.maturity_years * vols * vols

    @property
    def strikes(self) -> Tuple[float, ...]:
        return tuple(float(value) for value in self._strikes)

    def market_total_variance(self, strike: float) -> float:
        if not isfinite(strike) or strike <= 0.0:
            raise SurfaceError("strike must be positive and finite")
        k = log(float(strike) / self.forward)
        return float(np.interp(k, self._log_moneyness, self._total_variance))

    def market_iv(self, strike: float) -> float:
        return sqrt(self.market_total_variance(strike) / self.maturity_years)

    @property
    def atm_iv(self) -> float:
        """Interpolated exact IV at zero log-forward moneyness."""

        return self.market_iv(self.forward)

    def shifted_total_variance(self, strike: float, forecast_vol: float) -> float:
        if not isfinite(forecast_vol) or forecast_vol <= 0.0:
            raise SurfaceError("forecast_vol must be positive and finite")
        shift = self.maturity_years * (float(forecast_vol) ** 2 - self.atm_iv**2)
        shifted = self.market_total_variance(strike) + shift
        if shifted <= 0.0 or not isfinite(shifted):
            raise SurfaceError(
                "forecast shift produces non-positive total variance at strike {}".format(strike)
            )
        return shifted

    def shifted_iv(self, strike: float, forecast_vol: float) -> float:
        return sqrt(self.shifted_total_variance(strike, forecast_vol) / self.maturity_years)

    def shifted_points(self, forecast_vol: float) -> Tuple[SurfacePoint, ...]:
        return tuple(
            SurfacePoint(strike, self.shifted_iv(strike, forecast_vol))
            for strike in self.strikes
        )


def build_surface(
    forward: float,
    maturity_years: float,
    strike_iv_pairs: Iterable[Tuple[float, float]],
) -> TotalVarianceSurface:
    """Convenience constructor from ``(strike, exact_iv)`` pairs."""

    return TotalVarianceSurface(
        forward,
        maturity_years,
        [SurfacePoint(float(strike), float(iv)) for strike, iv in strike_iv_pairs],
    )

