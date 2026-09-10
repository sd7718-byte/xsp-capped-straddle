"""Explicit research defaults from the strategy brief."""

from dataclasses import dataclass
from datetime import time


@dataclass(frozen=True)
class StrategyConfig:
    entry_dte_min: int = 35
    entry_dte_max: int = 45
    exit_dte: int = 7
    target_wing_abs_delta: float = 0.10
    max_leg_spread_fraction: float = 0.10
    option_multiplier: float = 100.0
    scheduled_hedge_time: time = time(15, 45)
    market_timezone: str = "America/New_York"
    intraday_delta_trigger_per_package: float = 0.10
    entry_weekday: int = 2  # Monday=0, Wednesday=2.
    day_count: float = 365.25

    def __post_init__(self) -> None:
        if self.entry_dte_min <= self.exit_dte:
            raise ValueError("entry_dte_min must exceed exit_dte")
        if self.entry_dte_max < self.entry_dte_min:
            raise ValueError("entry DTE range is invalid")
        if not (0.0 < self.target_wing_abs_delta < 0.5):
            raise ValueError("target_wing_abs_delta must be between 0 and 0.5")
        if not (0.0 < self.max_leg_spread_fraction <= 1.0):
            raise ValueError("max_leg_spread_fraction must be in (0, 1]")
        if self.option_multiplier <= 0.0:
            raise ValueError("option_multiplier must be positive")
        if self.intraday_delta_trigger_per_package <= 0.0:
            raise ValueError("intraday delta trigger must be positive")
        if self.entry_weekday not in range(7):
            raise ValueError("entry_weekday must use Python weekday numbering")
        if self.day_count <= 0.0:
            raise ValueError("day_count must be positive")


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float
    strategy: StrategyConfig = StrategyConfig()
    hedge_commission_per_unit: float = 0.0
    accrue_cash_interest: bool = True
    force_close_at_end: bool = True

    def __post_init__(self) -> None:
        if self.initial_capital <= 0.0:
            raise ValueError("initial_capital must be positive")
        if self.hedge_commission_per_unit < 0.0:
            raise ValueError("hedge_commission_per_unit cannot be negative")

