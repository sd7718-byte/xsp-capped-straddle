"""Joint overnight spot/volatility stress for a delta-hedged short package."""

from dataclasses import dataclass
from math import isfinite

from .config import StrategyConfig
from .pricing import bs_price
from .risk import ShortStressEstimate
from .signal import PackageAnalysis


@dataclass(frozen=True)
class ShortStressBreakdown:
    stressed_spot: float
    stressed_package_value_points: float
    option_pnl_cash: float
    hedge_units: float
    hedge_pnl_cash: float
    close_commission_cash: float
    net_pnl_cash: float
    loss_cash: float

    def as_risk_estimate(
        self,
        spot_move_fraction: float = -0.08,
        volatility_point_increase: float = 15.0,
    ) -> ShortStressEstimate:
        return ShortStressEstimate(
            loss_per_package=self.loss_cash,
            underlying_move_fraction=spot_move_fraction,
            volatility_point_increase=volatility_point_increase,
        )


def stress_short_package(
    analysis: PackageAnalysis,
    config: StrategyConfig = StrategyConfig(),
    spot_move_fraction: float = -0.08,
    volatility_increase: float = 0.15,
    close_commission_cash: float = 0.0,
) -> ShortStressBreakdown:
    """Price the required ``-8%`` spot / ``+15`` vol-point short stress.

    The volatility shock is added to every current exact surface leg IV and the
    skew is therefore retained.  The immediately established delta hedge for a
    short package is included.  Results are for one package and use the
    configured option multiplier.
    """

    if not isinstance(analysis, PackageAnalysis):
        raise TypeError("analysis must be a PackageAnalysis")
    if not isfinite(spot_move_fraction) or spot_move_fraction <= -1.0:
        raise ValueError("spot_move_fraction must be finite and greater than -1")
    if not isfinite(volatility_increase) or volatility_increase < 0.0:
        raise ValueError("volatility_increase must be finite and non-negative")
    if not isfinite(close_commission_cash) or close_commission_cash < 0.0:
        raise ValueError("close_commission_cash must be finite and non-negative")
    stressed_spot = analysis.spot * (1.0 + spot_move_fraction)
    values = []
    for leg in analysis.legs:
        values.append(
            bs_price(
                stressed_spot,
                leg.strike,
                analysis.maturity_years,
                analysis.rate,
                leg.surface_iv + volatility_increase,
                leg.option_type,
                analysis.dividend_yield,
            )
        )
    stressed_package = values[0] + values[1] - values[2] - values[3]
    multiplier = config.option_multiplier
    option_pnl = (analysis.package_sell - stressed_package) * multiplier
    # A short package has the reverse of the long package's hedge sign.
    hedge_units = multiplier * analysis.package_delta
    hedge_pnl = hedge_units * (stressed_spot - analysis.spot)
    net = option_pnl + hedge_pnl - close_commission_cash
    return ShortStressBreakdown(
        stressed_spot=stressed_spot,
        stressed_package_value_points=stressed_package,
        option_pnl_cash=option_pnl,
        hedge_units=hedge_units,
        hedge_pnl_cash=hedge_pnl,
        close_commission_cash=close_commission_cash,
        net_pnl_cash=net,
        loss_cash=max(0.0, -net),
    )

