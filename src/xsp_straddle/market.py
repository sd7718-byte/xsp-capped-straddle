"""Validated option quotes and capped-straddle market selection.

Prices returned by this module are in option points unless a caller supplies a
``multiplier`` to :func:`executable_package_prices`.  No XSP contract
multiplier is embedded here; this keeps research and execution sizing explicit.
"""

from dataclasses import dataclass
import math
from typing import Iterable, Optional, Sequence, Tuple, Union

from .pricing import OptionType


class MarketDataError(ValueError):
    """Raised when quotes are invalid or a package cannot be selected."""


def _finite(name: str, value: float) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        raise MarketDataError("{} must be a finite real number".format(name))
    if not math.isfinite(converted):
        raise MarketDataError("{} must be a finite real number".format(name))
    return converted


def _positive(name: str, value: float) -> float:
    converted = _finite(name, value)
    if converted <= 0.0:
        raise MarketDataError("{} must be strictly positive".format(name))
    return converted


def _nonnegative(name: str, value: float) -> float:
    converted = _finite(name, value)
    if converted < 0.0:
        raise MarketDataError("{} must be non-negative".format(name))
    return converted


def _relative_spread(bid: float, ask: float, reference: str) -> float:
    normalized = reference.strip().lower()
    spread = ask - bid
    if normalized in ("mid", "midpoint", "premium"):
        denominator = 0.5 * (bid + ask)
    elif normalized == "ask":
        denominator = ask
    elif normalized == "bid":
        denominator = bid
    else:
        raise MarketDataError("reference must be 'mid', 'ask', or 'bid'")
    if denominator == 0.0:
        return 0.0 if spread == 0.0 else math.inf
    return spread / denominator


@dataclass(frozen=True)
class BidAskQuote:
    """A validated non-negative two-sided quote."""

    bid: float
    ask: float

    def __post_init__(self) -> None:
        bid = _nonnegative("bid", self.bid)
        ask = _nonnegative("ask", self.ask)
        if ask < bid:
            raise MarketDataError(
                "ask ({:.12g}) cannot be below bid ({:.12g})".format(ask, bid)
            )
        object.__setattr__(self, "bid", bid)
        object.__setattr__(self, "ask", ask)

    @property
    def mid(self) -> float:
        """Arithmetic midpoint."""

        return 0.5 * (self.bid + self.ask)

    @property
    def midpoint(self) -> float:
        """Alias for :attr:`mid`."""

        return self.mid

    @property
    def spread(self) -> float:
        """Ask less bid, in price points."""

        return self.ask - self.bid

    def relative_spread(self, reference: str = "mid") -> float:
        """Return spread divided by mid (default), ask, or bid premium."""

        return _relative_spread(self.bid, self.ask, reference)


@dataclass(frozen=True)
class OptionQuote:
    """One European option quote at a strike.

    ``delta`` and ``implied_volatility`` are optional so raw quote ingestion is
    possible before analytics are fitted.  Wing selection requires delta.
    """

    strike: float
    option_type: Union[OptionType, str]
    bid: float
    ask: float
    delta: Optional[float] = None
    implied_volatility: Optional[float] = None

    def __post_init__(self) -> None:
        strike = _positive("strike", self.strike)
        try:
            kind = OptionType.parse(self.option_type)
        except ValueError as exc:
            raise MarketDataError(str(exc))
        bid = _nonnegative("bid", self.bid)
        ask = _nonnegative("ask", self.ask)
        if ask < bid:
            raise MarketDataError(
                "ask ({:.12g}) cannot be below bid ({:.12g}) at strike "
                "{:.12g}".format(ask, bid, strike)
            )

        delta = self.delta
        if delta is not None:
            delta = _finite("delta", delta)
            if kind is OptionType.CALL and not 0.0 <= delta <= 1.0:
                raise MarketDataError("call delta must be between 0 and 1")
            if kind is OptionType.PUT and not -1.0 <= delta <= 0.0:
                raise MarketDataError("put delta must be between -1 and 0")

        volatility = self.implied_volatility
        if volatility is not None:
            volatility = _nonnegative("implied_volatility", volatility)

        object.__setattr__(self, "strike", strike)
        object.__setattr__(self, "option_type", kind)
        object.__setattr__(self, "bid", bid)
        object.__setattr__(self, "ask", ask)
        object.__setattr__(self, "delta", delta)
        object.__setattr__(self, "implied_volatility", volatility)

    @property
    def mid(self) -> float:
        """Arithmetic midpoint."""

        return 0.5 * (self.bid + self.ask)

    @property
    def midpoint(self) -> float:
        """Alias for :attr:`mid`."""

        return self.mid

    @property
    def spread(self) -> float:
        """Ask less bid, in price points."""

        return self.ask - self.bid

    def relative_spread(self, reference: str = "mid") -> float:
        """Return spread divided by mid (default), ask, or bid premium."""

        return _relative_spread(self.bid, self.ask, reference)


@dataclass(frozen=True)
class OptionChain:
    """A single-expiry collection containing at most one quote per contract."""

    quotes: Sequence[OptionQuote]

    def __post_init__(self) -> None:
        try:
            quotes = tuple(self.quotes)
        except TypeError:
            raise MarketDataError("quotes must be an iterable of OptionQuote")
        if not quotes:
            raise MarketDataError("option chain cannot be empty")
        seen = set()
        for quote in quotes:
            if not isinstance(quote, OptionQuote):
                raise MarketDataError("every chain member must be an OptionQuote")
            key = (quote.option_type, quote.strike)
            if key in seen:
                raise MarketDataError(
                    "duplicate {} quote at strike {:.12g}".format(
                        quote.option_type.value, quote.strike
                    )
                )
            seen.add(key)
        object.__setattr__(self, "quotes", quotes)

    @property
    def calls(self) -> Tuple[OptionQuote, ...]:
        """Call quotes ordered by ascending strike."""

        return tuple(
            sorted(
                (q for q in self.quotes if q.option_type is OptionType.CALL),
                key=lambda q: q.strike,
            )
        )

    @property
    def puts(self) -> Tuple[OptionQuote, ...]:
        """Put quotes ordered by ascending strike."""

        return tuple(
            sorted(
                (q for q in self.quotes if q.option_type is OptionType.PUT),
                key=lambda q: q.strike,
            )
        )

    @property
    def strikes(self) -> Tuple[float, ...]:
        """All distinct strikes in ascending order."""

        return tuple(sorted({quote.strike for quote in self.quotes}))

    @property
    def common_strikes(self) -> Tuple[float, ...]:
        """Ascending strikes having both a call and put quote."""

        call_strikes = {quote.strike for quote in self.calls}
        put_strikes = {quote.strike for quote in self.puts}
        return tuple(sorted(call_strikes & put_strikes))

    def get(
        self, strike: float, option_type: Union[OptionType, str]
    ) -> OptionQuote:
        """Return a quote by exact strike/type or raise :class:`MarketDataError`."""

        requested_strike = _positive("strike", strike)
        try:
            kind = OptionType.parse(option_type)
        except ValueError as exc:
            raise MarketDataError(str(exc))
        for quote in self.quotes:
            if quote.strike == requested_strike and quote.option_type is kind:
                return quote
        raise MarketDataError(
            "missing {} quote at strike {:.12g}".format(
                kind.value, requested_strike
            )
        )


@dataclass(frozen=True)
class CappedStraddle:
    """Long ATM call/put and short upper-call/lower-put wings."""

    atm_call: OptionQuote
    atm_put: OptionQuote
    upper_call: OptionQuote
    lower_put: OptionQuote

    def __post_init__(self) -> None:
        for field_name in ("atm_call", "atm_put", "upper_call", "lower_put"):
            if not isinstance(getattr(self, field_name), OptionQuote):
                raise MarketDataError("{} must be an OptionQuote".format(field_name))
        if self.atm_call.option_type is not OptionType.CALL:
            raise MarketDataError("atm_call must be a call")
        if self.atm_put.option_type is not OptionType.PUT:
            raise MarketDataError("atm_put must be a put")
        if self.upper_call.option_type is not OptionType.CALL:
            raise MarketDataError("upper_call must be a call")
        if self.lower_put.option_type is not OptionType.PUT:
            raise MarketDataError("lower_put must be a put")
        if self.atm_call.strike != self.atm_put.strike:
            raise MarketDataError("ATM call and put must have the same strike")
        if not (
            self.lower_put.strike
            < self.atm_call.strike
            < self.upper_call.strike
        ):
            raise MarketDataError("wing strikes must bracket the ATM strike")

    @property
    def atm_strike(self) -> float:
        """Shared strike of the long call and put."""

        return self.atm_call.strike

    @property
    def legs(self) -> Tuple[OptionQuote, OptionQuote, OptionQuote, OptionQuote]:
        """Legs ordered ATM call, ATM put, upper call, lower put."""

        return self.atm_call, self.atm_put, self.upper_call, self.lower_put

    @property
    def delta(self) -> float:
        """Package delta, provided all four leg deltas are available."""

        if any(leg.delta is None for leg in self.legs):
            raise MarketDataError("all four leg deltas are required for package delta")
        return (
            float(self.atm_call.delta)
            + float(self.atm_put.delta)
            - float(self.upper_call.delta)
            - float(self.lower_put.delta)
        )


@dataclass(frozen=True)
class PackagePrices:
    """Executable long/short and midpoint values, scaled by ``multiplier``."""

    buy: float
    sell: float
    mid: float
    multiplier: float

    @property
    def width(self) -> float:
        """Round-trip executable quote width (buy less sell)."""

        return self.buy - self.sell


def closest_atm_strike(strikes: Iterable[float], forward: float) -> float:
    """Return the listed strike closest to ``forward``.

    Ties resolve to the lower strike, making selection deterministic.
    """

    fwd = _positive("forward", forward)
    try:
        validated = tuple(_positive("strike", strike) for strike in strikes)
    except TypeError:
        raise MarketDataError("strikes must be an iterable")
    if not validated:
        raise MarketDataError("cannot select ATM strike from an empty set")
    return min(set(validated), key=lambda strike: (abs(strike - fwd), strike))


def select_capped_straddle(
    chain: OptionChain,
    forward: float,
    target_abs_delta: float = 0.10,
    *,
    max_relative_spread: Optional[float] = None,
    spread_reference: str = "mid",
) -> CappedStraddle:
    """Select the common-strike ATM pair and approximately 10-delta wings.

    The lower wing is the out-of-the-money put below ATM whose absolute delta
    is closest to ``target_abs_delta``; the upper wing is selected analogously
    from calls above ATM.  Ties choose the wing nearer ATM, then lower strike.
    Delta-enriched quotes are therefore required for wing candidates.
    """

    if not isinstance(chain, OptionChain):
        raise MarketDataError("chain must be an OptionChain")
    target = _positive("target_abs_delta", target_abs_delta)
    if target >= 1.0:
        raise MarketDataError("target_abs_delta must be less than 1")
    if not chain.common_strikes:
        raise MarketDataError("chain has no strike with both a call and put")

    atm_strike = closest_atm_strike(chain.common_strikes, forward)
    atm_call = chain.get(atm_strike, OptionType.CALL)
    atm_put = chain.get(atm_strike, OptionType.PUT)
    lower_candidates = [
        quote
        for quote in chain.puts
        if quote.strike < atm_strike and quote.delta is not None
    ]
    upper_candidates = [
        quote
        for quote in chain.calls
        if quote.strike > atm_strike and quote.delta is not None
    ]
    if not lower_candidates:
        raise MarketDataError("no delta-enriched put wing exists below ATM")
    if not upper_candidates:
        raise MarketDataError("no delta-enriched call wing exists above ATM")

    lower_put = min(
        lower_candidates,
        key=lambda quote: (
            # Quantize the distance so mathematically equal deviations such as
            # 0.08 and 0.12 around a 0.10 target reach the documented strike
            # tie-breaker instead of depending on binary-float noise.
            round(abs(abs(float(quote.delta)) - target), 12),
            abs(quote.strike - atm_strike),
            quote.strike,
        ),
    )
    upper_call = min(
        upper_candidates,
        key=lambda quote: (
            round(abs(abs(float(quote.delta)) - target), 12),
            abs(quote.strike - atm_strike),
            quote.strike,
        ),
    )
    package = CappedStraddle(atm_call, atm_put, upper_call, lower_put)
    if max_relative_spread is not None and not passes_spread_filter(
        package, max_relative_spread, reference=spread_reference
    ):
        raise MarketDataError(
            "selected package contains a leg wider than the configured "
            "relative-spread limit"
        )
    return package


def executable_package_prices(
    package: CappedStraddle, multiplier: float = 1.0
) -> PackagePrices:
    """Return executable buy, executable sell and midpoint package prices.

    For ``G = C0 + P0 - CU - PL`` the long-package buy lifts the ATM asks and
    sells the wings at their bids.  The sell price does the reverse.  The
    caller-supplied ``multiplier`` scales point values to currency values when
    desired; its default is one and no product-specific constant is assumed.
    """

    if not isinstance(package, CappedStraddle):
        raise MarketDataError("package must be a CappedStraddle")
    scale = _positive("multiplier", multiplier)
    buy_points = (
        package.atm_call.ask
        + package.atm_put.ask
        - package.upper_call.bid
        - package.lower_put.bid
    )
    sell_points = (
        package.atm_call.bid
        + package.atm_put.bid
        - package.upper_call.ask
        - package.lower_put.ask
    )
    mid_points = (
        package.atm_call.mid
        + package.atm_put.mid
        - package.upper_call.mid
        - package.lower_put.mid
    )
    return PackagePrices(
        buy=scale * buy_points,
        sell=scale * sell_points,
        mid=scale * mid_points,
        multiplier=scale,
    )


def passes_spread_filter(
    package: CappedStraddle,
    max_relative_spread: float = 0.10,
    *,
    reference: str = "mid",
) -> bool:
    """Return whether every leg's spread is within the configured limit.

    Relative spread is ``(ask-bid)/mid`` by default.  The comparison is
    inclusive, so a leg exactly at the limit passes.  A positive spread on a
    zero reference premium has infinite relative width and fails.
    """

    if not isinstance(package, CappedStraddle):
        raise MarketDataError("package must be a CappedStraddle")
    limit = _nonnegative("max_relative_spread", max_relative_spread)
    return all(leg.relative_spread(reference) <= limit for leg in package.legs)


# Short aliases useful in research notebooks and integration code.
package_prices = executable_package_prices
package_passes_spread_filter = passes_spread_filter
