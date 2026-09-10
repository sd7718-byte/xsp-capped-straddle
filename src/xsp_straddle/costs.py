"""Projected and realized transaction-cost accounting in consistent units."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class CostAssumptions:
    """User-supplied cost assumptions; no broker-specific schedule is implied."""

    option_commission_per_contract: float
    projected_hedge_cost_cash_per_package: float
    other_cost_cash_per_package: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.option_commission_per_contract,
            self.projected_hedge_cost_cash_per_package,
            self.other_cost_cash_per_package,
        )
        if not all(isfinite(float(value)) and value >= 0.0 for value in values):
            raise ValueError("all cost assumptions must be finite and non-negative")


@dataclass(frozen=True)
class ProjectedPackageCosts:
    """Round-trip costs expressed in option index points per package."""

    option_commissions: float
    closing_liquidity: float
    hedge_costs: float
    other_costs: float

    @property
    def total(self) -> float:
        return self.option_commissions + self.closing_liquidity + self.hedge_costs + self.other_costs

    def doubled(self) -> "ProjectedPackageCosts":
        return ProjectedPackageCosts(
            option_commissions=2.0 * self.option_commissions,
            closing_liquidity=2.0 * self.closing_liquidity,
            hedge_costs=2.0 * self.hedge_costs,
            other_costs=2.0 * self.other_costs,
        )


def project_round_trip_costs(
    package_buy: float,
    package_sell: float,
    package_mid: float,
    direction: int,
    assumptions: CostAssumptions,
    option_multiplier: float = 100.0,
    legs_per_package: int = 4,
) -> ProjectedPackageCosts:
    """Project all costs not already captured by the executable entry price.

    ``package_buy``/``package_sell`` already include the adverse entry leg
    quotes.  The function therefore adds a projected executable closing spread,
    both entry and exit commissions, hedge costs, and explicit miscellaneous
    costs.  Cash costs are divided by the option multiplier into index points.
    """

    if direction not in (-1, 1):
        raise ValueError("direction must be 1 (long) or -1 (short)")
    if option_multiplier <= 0.0 or not isfinite(option_multiplier):
        raise ValueError("option_multiplier must be positive and finite")
    if legs_per_package <= 0:
        raise ValueError("legs_per_package must be positive")
    if not (package_sell <= package_mid <= package_buy):
        raise ValueError("package prices must satisfy sell <= mid <= buy")
    close_liquidity = package_mid - package_sell if direction == 1 else package_buy - package_mid
    commissions_cash = 2.0 * legs_per_package * assumptions.option_commission_per_contract
    return ProjectedPackageCosts(
        option_commissions=commissions_cash / option_multiplier,
        closing_liquidity=close_liquidity,
        hedge_costs=assumptions.projected_hedge_cost_cash_per_package / option_multiplier,
        other_costs=assumptions.other_cost_cash_per_package / option_multiplier,
    )

