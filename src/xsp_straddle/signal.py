"""Exact-IV, package-level signal construction for one synchronized chain."""

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import StrategyConfig
from .costs import CostAssumptions, ProjectedPackageCosts, project_round_trip_costs
from .filtering import PreliminaryScreen, PreliminarySide, maybe_screen
from .market import (
    CappedStraddle,
    OptionChain,
    OptionQuote,
    executable_package_prices,
    passes_spread_filter,
    select_capped_straddle,
)
from .pricing import (
    OptionType,
    PricingError,
    bs_greeks,
    bs_price,
    estimate_forward,
    implied_vol_brent,
)
from .surface import SurfaceError, SurfacePoint, TotalVarianceSurface


class SignalSide(str, Enum):
    LONG = "long"
    SHORT = "short"
    NONE = "none"


@dataclass(frozen=True)
class PackageDefinition:
    expiration: pd.Timestamp
    atm_strike: float
    lower_strike: float
    upper_strike: float


@dataclass(frozen=True)
class ExactLegAnalysis:
    role: str
    strike: float
    option_type: OptionType
    bid: float
    ask: float
    bid_iv: float
    mid_iv: float
    ask_iv: float
    surface_iv: float
    delta: float
    gamma: float
    vega: float


@dataclass(frozen=True)
class PackageAnalysis:
    definition: PackageDefinition
    timestamp: pd.Timestamp
    spot: float
    forward: float
    forward_dispersion: float
    maturity_years: float
    rate: float
    dividend_yield: float
    exact_atm_iv: float
    package_buy: float
    package_sell: float
    package_mid: float
    fair_at_lower_vol: float
    fair_at_upper_vol: float
    package_delta: float
    package_gamma: float
    package_vega: float
    legs: Tuple[ExactLegAnalysis, ...]
    long_costs: ProjectedPackageCosts
    short_costs: ProjectedPackageCosts
    spread_ok: bool

    @property
    def one_vol_point_vega_loss_cash(self) -> float:
        """Absolute one-percentage-point vega move for one package in dollars."""

        return abs(self.package_vega) * 0.01


@dataclass(frozen=True)
class SignalDecision:
    side: SignalSide
    reason: str
    long_edge: Optional[float]
    short_edge: Optional[float]
    analysis: Optional[PackageAnalysis]
    preliminary: Optional[PreliminaryScreen] = None

    @property
    def tradeable(self) -> bool:
        return self.side is not SignalSide.NONE


def _option_type(value: str) -> OptionType:
    value = str(value).upper()
    if value in ("C", "CALL"):
        return OptionType.CALL
    if value in ("P", "PUT"):
        return OptionType.PUT
    raise ValueError("unknown option type {!r}".format(value))


def _snapshot_state(
    snapshot: pd.DataFrame, expiration: pd.Timestamp, config: StrategyConfig
) -> Tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp, float, float, float, float]:
    if snapshot.empty:
        raise ValueError("snapshot cannot be empty")
    frame = snapshot.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["expiration"] = pd.to_datetime(frame["expiration"], utc=True)
    expiration = pd.to_datetime(expiration, utc=True)
    frame = frame[frame["expiration"] == expiration]
    if frame.empty:
        raise ValueError("snapshot has no rows for expiration {}".format(expiration))
    timestamps = frame["timestamp"].unique()
    if len(timestamps) != 1:
        raise ValueError("snapshot must contain exactly one timestamp")
    timestamp = pd.Timestamp(timestamps[0])
    maturity = (expiration - timestamp).total_seconds() / (config.day_count * 86400.0)
    if maturity <= 0.0:
        raise ValueError("expiration must be after timestamp")
    state = frame[["spot", "rate", "dividend_yield"]].drop_duplicates()
    if len(state) != 1:
        raise ValueError("spot/rate state must be synchronized within snapshot")
    spot, rate, dividend = (float(value) for value in state.iloc[0])
    if spot <= 0.0:
        raise ValueError("spot must be positive")
    return frame, timestamp, expiration, maturity, spot, rate, dividend


def _estimate_snapshot_forward(frame: pd.DataFrame, maturity: float, rate: float) -> Tuple[float, float]:
    forwards: List[float] = []
    for strike, group in frame.groupby("strike"):
        by_type = {str(row.option_type).upper(): row for row in group.itertuples()}
        call = by_type.get("C") or by_type.get("CALL")
        put = by_type.get("P") or by_type.get("PUT")
        if call is None or put is None:
            continue
        call_mid = (float(call.bid) + float(call.ask)) / 2.0
        put_mid = (float(put.bid) + float(put.ask)) / 2.0
        try:
            value = estimate_forward(call_mid, put_mid, float(strike), maturity, rate)
        except PricingError:
            continue
        if isfinite(value) and value > 0.0:
            forwards.append(float(value))
    if not forwards:
        raise ValueError("put-call parity produced no valid forward estimates")
    forward = float(np.median(np.asarray(forwards)))
    dispersion = float(np.median(np.abs(np.asarray(forwards) - forward)))
    return forward, dispersion


def _solve_mid_quotes(
    frame: pd.DataFrame,
    spot: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
) -> Tuple[Dict[Tuple[float, OptionType], OptionQuote], Dict[Tuple[float, OptionType], float]]:
    quotes: Dict[Tuple[float, OptionType], OptionQuote] = {}
    ivs: Dict[Tuple[float, OptionType], float] = {}
    for row in frame.itertuples():
        option_type = _option_type(row.option_type)
        strike = float(row.strike)
        bid = float(row.bid)
        ask = float(row.ask)
        mid = (bid + ask) / 2.0
        try:
            iv = implied_vol_brent(
                mid,
                spot,
                strike,
                maturity,
                rate,
                option_type,
                dividend_yield,
            )
            delta = bs_greeks(
                spot, strike, maturity, rate, iv, option_type, dividend_yield
            ).delta
        except PricingError:
            continue
        key = (strike, option_type)
        quotes[key] = OptionQuote(
            strike=strike,
            option_type=option_type,
            bid=bid,
            ask=ask,
            delta=delta,
            implied_volatility=iv,
        )
        ivs[key] = iv
    return quotes, ivs


def _package_from_definition(
    quotes: Mapping[Tuple[float, OptionType], OptionQuote], definition: PackageDefinition
) -> CappedStraddle:
    try:
        return CappedStraddle(
            atm_call=quotes[(definition.atm_strike, OptionType.CALL)],
            atm_put=quotes[(definition.atm_strike, OptionType.PUT)],
            upper_call=quotes[(definition.upper_strike, OptionType.CALL)],
            lower_put=quotes[(definition.lower_strike, OptionType.PUT)],
        )
    except KeyError as exc:
        raise ValueError("fixed package leg is missing or has no valid exact midpoint IV") from exc


def _surface_from_exact_ivs(
    ivs: Mapping[Tuple[float, OptionType], float],
    forward: float,
    maturity: float,
) -> TotalVarianceSurface:
    strikes = sorted({key[0] for key in ivs})
    points: List[SurfacePoint] = []
    for strike in strikes:
        put_iv = ivs.get((strike, OptionType.PUT))
        call_iv = ivs.get((strike, OptionType.CALL))
        if strike < forward and put_iv is not None:
            iv = put_iv
        elif strike > forward and call_iv is not None:
            iv = call_iv
        else:
            available = [value for value in (put_iv, call_iv) if value is not None]
            if not available:
                continue
            iv = float(np.mean(available))
        points.append(SurfacePoint(strike, iv))
    return TotalVarianceSurface(forward, maturity, points)


def _exact_leg(
    role: str,
    quote: OptionQuote,
    surface: TotalVarianceSurface,
    spot: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
) -> ExactLegAnalysis:
    bid_iv = implied_vol_brent(
        quote.bid,
        spot,
        quote.strike,
        maturity,
        rate,
        quote.option_type,
        dividend_yield,
    )
    ask_iv = implied_vol_brent(
        quote.ask,
        spot,
        quote.strike,
        maturity,
        rate,
        quote.option_type,
        dividend_yield,
    )
    mid_iv = float(quote.implied_volatility)
    surface_iv = surface.market_iv(quote.strike)
    greeks = bs_greeks(
        spot,
        quote.strike,
        maturity,
        rate,
        surface_iv,
        quote.option_type,
        dividend_yield,
    )
    return ExactLegAnalysis(
        role=role,
        strike=quote.strike,
        option_type=quote.option_type,
        bid=quote.bid,
        ask=quote.ask,
        bid_iv=bid_iv,
        mid_iv=mid_iv,
        ask_iv=ask_iv,
        surface_iv=surface_iv,
        delta=greeks.delta,
        gamma=greeks.gamma,
        vega=greeks.vega,
    )


def _fair_package_value(
    package: CappedStraddle,
    surface: TotalVarianceSurface,
    forecast_vol: float,
    spot: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
) -> float:
    values = []
    for quote in package.legs:
        shifted_iv = surface.shifted_iv(quote.strike, forecast_vol)
        values.append(
            bs_price(
                spot,
                quote.strike,
                maturity,
                rate,
                shifted_iv,
                quote.option_type,
                dividend_yield,
            )
        )
    return values[0] + values[1] - values[2] - values[3]


def analyze_snapshot(
    snapshot: pd.DataFrame,
    expiration: pd.Timestamp,
    forecast_lower_vol: float,
    forecast_upper_vol: float,
    cost_assumptions: CostAssumptions,
    config: Optional[StrategyConfig] = None,
    definition: Optional[PackageDefinition] = None,
    approximate_bid_iv: Optional[float] = None,
    approximate_ask_iv: Optional[float] = None,
    prefilter_cost_vol: float = 0.0,
) -> SignalDecision:
    """Build one conservative package signal from synchronized executable quotes.

    Approximate IVs are optional because the paper-specific approximation was
    not supplied.  If provided, they are used only as a computational pre-filter.
    The final decision always solves selected bid, midpoint, and ask IVs with
    Brent and reprices all four legs from the shifted exact-IV surface.
    """

    active = config if config is not None else StrategyConfig()
    if forecast_lower_vol <= 0.0 or forecast_upper_vol <= 0.0:
        raise ValueError("forecast volatility bounds must be positive")
    if forecast_lower_vol > forecast_upper_vol:
        raise ValueError("forecast lower bound cannot exceed upper bound")
    frame, timestamp, expiration, maturity, spot, rate, dividend = _snapshot_state(
        snapshot, expiration, active
    )
    forward, dispersion = _estimate_snapshot_forward(frame, maturity, rate)
    preliminary = maybe_screen(
        approximate_bid_iv,
        approximate_ask_iv,
        forecast_lower_vol,
        forecast_upper_vol,
        prefilter_cost_vol,
    )
    if preliminary is not None and preliminary.side is PreliminarySide.NONE:
        return SignalDecision(
            side=SignalSide.NONE,
            reason="APPROXIMATE_PREFILTER_NO_EDGE",
            long_edge=None,
            short_edge=None,
            analysis=None,
            preliminary=preliminary,
        )

    quotes, mid_ivs = _solve_mid_quotes(frame, spot, maturity, rate, dividend)
    if definition is None:
        package = select_capped_straddle(
            OptionChain(quotes=list(quotes.values())),
            forward,
            target_abs_delta=active.target_wing_abs_delta,
        )
        definition = PackageDefinition(
            expiration=expiration,
            atm_strike=package.atm_call.strike,
            lower_strike=package.lower_put.strike,
            upper_strike=package.upper_call.strike,
        )
    else:
        if pd.to_datetime(definition.expiration, utc=True) != expiration:
            raise ValueError("fixed package expiration does not match requested expiration")
        package = _package_from_definition(quotes, definition)

    surface = _surface_from_exact_ivs(mid_ivs, forward, maturity)
    roles = ("atm_call", "atm_put", "upper_call", "lower_put")
    legs = tuple(
        _exact_leg(role, quote, surface, spot, maturity, rate, dividend)
        for role, quote in zip(roles, package.legs)
    )
    signs = (1.0, 1.0, -1.0, -1.0)
    package_delta = sum(sign * leg.delta for sign, leg in zip(signs, legs))
    package_gamma = sum(sign * leg.gamma for sign, leg in zip(signs, legs))
    # Convert vega and all other Greeks to one package's cash units.
    package_vega = active.option_multiplier * sum(
        sign * leg.vega for sign, leg in zip(signs, legs)
    )
    prices = executable_package_prices(package, multiplier=1.0)
    fair_lower = _fair_package_value(
        package, surface, forecast_lower_vol, spot, maturity, rate, dividend
    )
    fair_upper = _fair_package_value(
        package, surface, forecast_upper_vol, spot, maturity, rate, dividend
    )
    long_costs = project_round_trip_costs(
        prices.buy,
        prices.sell,
        prices.mid,
        1,
        cost_assumptions,
        active.option_multiplier,
    )
    short_costs = project_round_trip_costs(
        prices.buy,
        prices.sell,
        prices.mid,
        -1,
        cost_assumptions,
        active.option_multiplier,
    )
    spread_ok = passes_spread_filter(
        package,
        max_relative_spread=active.max_leg_spread_fraction,
        reference="mid",
    )
    analysis = PackageAnalysis(
        definition=definition,
        timestamp=timestamp,
        spot=spot,
        forward=forward,
        forward_dispersion=dispersion,
        maturity_years=maturity,
        rate=rate,
        dividend_yield=dividend,
        exact_atm_iv=surface.atm_iv,
        package_buy=prices.buy,
        package_sell=prices.sell,
        package_mid=prices.mid,
        fair_at_lower_vol=fair_lower,
        fair_at_upper_vol=fair_upper,
        package_delta=package_delta,
        package_gamma=package_gamma,
        package_vega=package_vega,
        legs=legs,
        long_costs=long_costs,
        short_costs=short_costs,
        spread_ok=spread_ok,
    )
    long_edge = fair_lower - prices.buy - long_costs.total
    short_edge = prices.sell - fair_upper - short_costs.total
    if not spread_ok:
        return SignalDecision(
            SignalSide.NONE,
            "LEG_SPREAD_TOO_WIDE",
            long_edge,
            short_edge,
            analysis,
            preliminary,
        )

    allow_long = preliminary is None or preliminary.side is PreliminarySide.LONG
    allow_short = preliminary is None or preliminary.side is PreliminarySide.SHORT
    long_passes = allow_long and long_edge > 0.0
    short_passes = allow_short and short_edge > 0.0
    if long_passes and short_passes:
        return SignalDecision(
            SignalSide.NONE,
            "CONFLICTING_PACKAGE_SIGNALS",
            long_edge,
            short_edge,
            analysis,
            preliminary,
        )
    if long_passes:
        return SignalDecision(
            SignalSide.LONG,
            "LONG_EDGE_POSITIVE",
            long_edge,
            short_edge,
            analysis,
            preliminary,
        )
    if short_passes:
        return SignalDecision(
            SignalSide.SHORT,
            "SHORT_EDGE_POSITIVE",
            long_edge,
            short_edge,
            analysis,
            preliminary,
        )
    return SignalDecision(
        SignalSide.NONE,
        "PACKAGE_EDGE_NONPOSITIVE",
        long_edge,
        short_edge,
        analysis,
        preliminary,
    )

