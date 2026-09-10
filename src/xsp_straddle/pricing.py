"""Dependency-free European option pricing primitives.

Two Black--Scholes conventions are exposed deliberately rather than hidden
behind an ambiguous ``underlying`` argument:

* :func:`bs_greeks` takes spot ``S``, continuously-compounded rate ``r`` and
  continuous dividend yield ``q``.  Its delta and gamma are derivatives with
  respect to spot.
* :func:`bs_forward_greeks` takes a forward ``F`` and discount factor ``D``.
  Its delta and gamma are derivatives with respect to the forward.  Prices and
  vega are present values in both conventions.

Vega is quoted per unit volatility (a change of ``1.0``), not per one
volatility point.  Divide it by 100 for the price effect of one vol point.
"""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Callable, Optional, Tuple, Union


Number = Union[int, float]


class OptionType(str, Enum):
    """European option right."""

    CALL = "call"
    PUT = "put"

    @classmethod
    def parse(cls, value: Union["OptionType", str]) -> "OptionType":
        """Return an :class:`OptionType` from common string spellings."""

        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in ("call", "c"):
                return cls.CALL
            if normalized in ("put", "p"):
                return cls.PUT
        raise ValueError("option_type must be 'call'/'c' or 'put'/'p'")


class PricingError(ValueError):
    """Base class for invalid pricing inputs or unavailable prices."""


class NoArbitrageError(PricingError):
    """Raised when an option price violates static no-arbitrage bounds."""


class ImpliedVolatilityError(PricingError):
    """Raised when implied volatility is undefined or cannot be solved."""


@dataclass(frozen=True)
class Greeks:
    """Option present value and first/second-order sensitivities."""

    price: float
    delta: float
    gamma: float
    vega: float


def _finite(name: str, value: Number) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        raise PricingError("{} must be a finite real number".format(name))
    if not math.isfinite(converted):
        raise PricingError("{} must be a finite real number".format(name))
    return converted


def _positive(name: str, value: Number) -> float:
    converted = _finite(name, value)
    if converted <= 0.0:
        raise PricingError("{} must be strictly positive".format(name))
    return converted


def _nonnegative(name: str, value: Number) -> float:
    converted = _finite(name, value)
    if converted < 0.0:
        raise PricingError("{} must be non-negative".format(name))
    return converted


def _normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _spot_inputs(
    spot: Number,
    strike: Number,
    time_to_expiry: Number,
    rate: Number,
    volatility: Number,
    dividend_yield: Number,
) -> Tuple[float, float, float, float, float, float]:
    return (
        _positive("spot", spot),
        _positive("strike", strike),
        _nonnegative("time_to_expiry", time_to_expiry),
        _finite("rate", rate),
        _nonnegative("volatility", volatility),
        _finite("dividend_yield", dividend_yield),
    )


def _forward_inputs(
    forward: Number,
    strike: Number,
    time_to_expiry: Number,
    discount_factor: Number,
    volatility: Number,
) -> Tuple[float, float, float, float, float]:
    return (
        _positive("forward", forward),
        _positive("strike", strike),
        _nonnegative("time_to_expiry", time_to_expiry),
        _positive("discount_factor", discount_factor),
        _nonnegative("volatility", volatility),
    )


def bs_greeks(
    spot: Number,
    strike: Number,
    time_to_expiry: Number,
    rate: Number,
    volatility: Number,
    option_type: Union[OptionType, str],
    dividend_yield: Number = 0.0,
) -> Greeks:
    """Return spot-convention Black--Scholes price, delta, gamma and vega.

    ``rate`` and ``dividend_yield`` are continuously compounded annual rates;
    ``time_to_expiry`` is in years.  At expiry, or at zero volatility, the
    limiting deterministic price/delta is returned and gamma/vega are zero.
    At the deterministic strike kink, delta uses the symmetric half-delta
    convention.
    """

    kind = OptionType.parse(option_type)
    s, k, t, r, sigma, q = _spot_inputs(
        spot, strike, time_to_expiry, rate, volatility, dividend_yield
    )
    discount_q = math.exp(-q * t)
    discount_r = math.exp(-r * t)
    pv_spot = s * discount_q
    pv_strike = k * discount_r

    if t == 0.0 or sigma == 0.0:
        difference = pv_spot - pv_strike
        if difference > 0.0:
            call_delta = discount_q
        elif difference < 0.0:
            call_delta = 0.0
        else:
            call_delta = 0.5 * discount_q
        if kind is OptionType.CALL:
            return Greeks(max(difference, 0.0), call_delta, 0.0, 0.0)
        return Greeks(max(-difference, 0.0), call_delta - discount_q, 0.0, 0.0)

    root_t = math.sqrt(t)
    sigma_root_t = sigma * root_t
    d1 = (math.log(s / k) + (r - q + 0.5 * sigma * sigma) * t) / sigma_root_t
    d2 = d1 - sigma_root_t
    pdf_d1 = _normal_pdf(d1)

    call_price = pv_spot * _normal_cdf(d1) - pv_strike * _normal_cdf(d2)
    call_delta = discount_q * _normal_cdf(d1)
    gamma = discount_q * pdf_d1 / (s * sigma_root_t)
    vega = pv_spot * pdf_d1 * root_t
    if kind is OptionType.CALL:
        return Greeks(call_price, call_delta, gamma, vega)

    put_price = pv_strike * _normal_cdf(-d2) - pv_spot * _normal_cdf(-d1)
    put_delta = call_delta - discount_q
    return Greeks(put_price, put_delta, gamma, vega)


def bs_price(
    spot: Number,
    strike: Number,
    time_to_expiry: Number,
    rate: Number,
    volatility: Number,
    option_type: Union[OptionType, str],
    dividend_yield: Number = 0.0,
) -> float:
    """Return a European option price under the spot convention."""

    return bs_greeks(
        spot, strike, time_to_expiry, rate, volatility, option_type, dividend_yield
    ).price


def bs_delta(
    spot: Number,
    strike: Number,
    time_to_expiry: Number,
    rate: Number,
    volatility: Number,
    option_type: Union[OptionType, str],
    dividend_yield: Number = 0.0,
) -> float:
    """Return derivative of option present value with respect to spot."""

    return bs_greeks(
        spot, strike, time_to_expiry, rate, volatility, option_type, dividend_yield
    ).delta


def bs_gamma(
    spot: Number,
    strike: Number,
    time_to_expiry: Number,
    rate: Number,
    volatility: Number,
    option_type: Union[OptionType, str] = OptionType.CALL,
    dividend_yield: Number = 0.0,
) -> float:
    """Return second derivative of option present value with respect to spot."""

    return bs_greeks(
        spot, strike, time_to_expiry, rate, volatility, option_type, dividend_yield
    ).gamma


def bs_vega(
    spot: Number,
    strike: Number,
    time_to_expiry: Number,
    rate: Number,
    volatility: Number,
    option_type: Union[OptionType, str] = OptionType.CALL,
    dividend_yield: Number = 0.0,
) -> float:
    """Return derivative of price per unit change in volatility."""

    return bs_greeks(
        spot, strike, time_to_expiry, rate, volatility, option_type, dividend_yield
    ).vega


def bs_forward_greeks(
    forward: Number,
    strike: Number,
    time_to_expiry: Number,
    discount_factor: Number,
    volatility: Number,
    option_type: Union[OptionType, str],
) -> Greeks:
    """Return Black--76 price and greeks under the forward convention.

    Delta and gamma are derivatives with respect to ``forward``, not spot.
    ``discount_factor`` discounts expiry cash flows to the valuation time.
    """

    kind = OptionType.parse(option_type)
    f, k, t, discount, sigma = _forward_inputs(
        forward, strike, time_to_expiry, discount_factor, volatility
    )
    difference = f - k
    if t == 0.0 or sigma == 0.0:
        if difference > 0.0:
            call_delta = discount
        elif difference < 0.0:
            call_delta = 0.0
        else:
            call_delta = 0.5 * discount
        if kind is OptionType.CALL:
            return Greeks(discount * max(difference, 0.0), call_delta, 0.0, 0.0)
        return Greeks(
            discount * max(-difference, 0.0), call_delta - discount, 0.0, 0.0
        )

    root_t = math.sqrt(t)
    sigma_root_t = sigma * root_t
    d1 = (math.log(f / k) + 0.5 * sigma * sigma * t) / sigma_root_t
    d2 = d1 - sigma_root_t
    pdf_d1 = _normal_pdf(d1)
    call_price = discount * (f * _normal_cdf(d1) - k * _normal_cdf(d2))
    call_delta = discount * _normal_cdf(d1)
    gamma = discount * pdf_d1 / (f * sigma_root_t)
    vega = discount * f * pdf_d1 * root_t
    if kind is OptionType.CALL:
        return Greeks(call_price, call_delta, gamma, vega)
    put_price = discount * (k * _normal_cdf(-d2) - f * _normal_cdf(-d1))
    return Greeks(put_price, call_delta - discount, gamma, vega)


def bs_forward_price(
    forward: Number,
    strike: Number,
    time_to_expiry: Number,
    discount_factor: Number,
    volatility: Number,
    option_type: Union[OptionType, str],
) -> float:
    """Return a European option present value under the forward convention."""

    return bs_forward_greeks(
        forward, strike, time_to_expiry, discount_factor, volatility, option_type
    ).price


def bs_forward_delta(
    forward: Number,
    strike: Number,
    time_to_expiry: Number,
    discount_factor: Number,
    volatility: Number,
    option_type: Union[OptionType, str],
) -> float:
    """Return derivative of option value with respect to its forward."""

    return bs_forward_greeks(
        forward, strike, time_to_expiry, discount_factor, volatility, option_type
    ).delta


def bs_forward_gamma(
    forward: Number,
    strike: Number,
    time_to_expiry: Number,
    discount_factor: Number,
    volatility: Number,
    option_type: Union[OptionType, str] = OptionType.CALL,
) -> float:
    """Return second derivative of option value with respect to its forward."""

    return bs_forward_greeks(
        forward, strike, time_to_expiry, discount_factor, volatility, option_type
    ).gamma


def bs_forward_vega(
    forward: Number,
    strike: Number,
    time_to_expiry: Number,
    discount_factor: Number,
    volatility: Number,
    option_type: Union[OptionType, str] = OptionType.CALL,
) -> float:
    """Return forward-convention vega per unit volatility."""

    return bs_forward_greeks(
        forward, strike, time_to_expiry, discount_factor, volatility, option_type
    ).vega


def no_arbitrage_bounds(
    spot: Number,
    strike: Number,
    time_to_expiry: Number,
    rate: Number,
    option_type: Union[OptionType, str],
    dividend_yield: Number = 0.0,
) -> Tuple[float, float]:
    """Return closed lower/upper price bounds for a European option."""

    kind = OptionType.parse(option_type)
    s = _positive("spot", spot)
    k = _positive("strike", strike)
    t = _nonnegative("time_to_expiry", time_to_expiry)
    r = _finite("rate", rate)
    q = _finite("dividend_yield", dividend_yield)
    pv_spot = s * math.exp(-q * t)
    pv_strike = k * math.exp(-r * t)
    if kind is OptionType.CALL:
        return max(pv_spot - pv_strike, 0.0), pv_spot
    return max(pv_strike - pv_spot, 0.0), pv_strike


def forward_no_arbitrage_bounds(
    forward: Number,
    strike: Number,
    discount_factor: Number,
    option_type: Union[OptionType, str],
) -> Tuple[float, float]:
    """Return closed option price bounds given a forward and discount factor."""

    kind = OptionType.parse(option_type)
    f = _positive("forward", forward)
    k = _positive("strike", strike)
    discount = _positive("discount_factor", discount_factor)
    if kind is OptionType.CALL:
        return discount * max(f - k, 0.0), discount * f
    return discount * max(k - f, 0.0), discount * k


def _brentq(
    function: Callable[[float], float],
    lower: float,
    upper: float,
    function_tolerance: float,
    root_tolerance: float,
    max_iterations: int,
) -> float:
    """Find a bracketed root with the safeguarded Brent--Dekker algorithm."""

    x_pre = lower
    x_cur = upper
    f_pre = function(x_pre)
    f_cur = function(x_cur)
    if abs(f_pre) <= function_tolerance:
        return x_pre
    if abs(f_cur) <= function_tolerance:
        return x_cur
    if f_pre * f_cur > 0.0:
        raise ImpliedVolatilityError(
            "Brent solver requires endpoint prices that bracket the target"
        )

    x_block = x_pre
    f_block = f_pre
    step_pre = x_cur - x_pre
    step_cur = step_pre
    for _ in range(max_iterations):
        # Keep x_block on the opposite-sign side of x_cur.
        if f_pre != 0.0 and f_cur != 0.0 and (f_pre > 0.0) != (f_cur > 0.0):
            x_block = x_pre
            f_block = f_pre
            step_pre = x_cur - x_pre
            step_cur = step_pre

        # x_cur is always the best (smallest residual) estimate.
        if abs(f_block) < abs(f_cur):
            old_cur = x_cur
            old_f_cur = f_cur
            x_pre, x_cur, x_block = x_cur, x_block, old_cur
            f_pre, f_cur, f_block = f_cur, f_block, old_f_cur

        bisect_step = 0.5 * (x_block - x_cur)
        local_tolerance = root_tolerance + 2.0 * math.ulp(1.0) * abs(x_cur)
        if abs(f_cur) <= function_tolerance or abs(bisect_step) <= local_tolerance:
            return x_cur

        if abs(step_pre) > local_tolerance and abs(f_cur) < abs(f_pre):
            # Secant for two distinct points; inverse quadratic interpolation
            # when three distinct function values are available.
            if x_pre == x_block:
                trial_step = -f_cur * (x_cur - x_pre) / (f_cur - f_pre)
            else:
                slope_pre = (f_pre - f_cur) / (x_pre - x_cur)
                slope_block = (f_block - f_cur) / (x_block - x_cur)
                trial_step = -f_cur * (
                    f_block * slope_block - f_pre * slope_pre
                ) / (slope_block * slope_pre * (f_block - f_pre))

            # Accept interpolation only when it is safely inside the bracket
            # and materially better than bisection.
            if 2.0 * abs(trial_step) < min(
                abs(step_pre), 3.0 * abs(bisect_step) - local_tolerance
            ):
                step_pre = step_cur
                step_cur = trial_step
            else:
                step_pre = bisect_step
                step_cur = bisect_step
        else:
            step_pre = bisect_step
            step_cur = bisect_step

        x_pre = x_cur
        f_pre = f_cur
        if abs(step_cur) > local_tolerance:
            x_cur += step_cur
        else:
            x_cur += math.copysign(local_tolerance, bisect_step)
        f_cur = function(x_cur)

    raise ImpliedVolatilityError(
        "Brent solver did not converge after {} iterations; final volatility "
        "was {:.12g} with price residual {:.12g}".format(
            max_iterations, x_cur, f_cur
        )
    )


def _validate_solver_settings(
    initial_upper: Number,
    max_volatility: Number,
    price_tolerance: Number,
    volatility_tolerance: Number,
    max_iterations: int,
) -> Tuple[float, float, float, float, int]:
    initial = _positive("initial_upper", initial_upper)
    maximum = _positive("max_volatility", max_volatility)
    if initial > maximum:
        raise PricingError("initial_upper cannot exceed max_volatility")
    price_tol = _positive("price_tolerance", price_tolerance)
    vol_tol = _positive("volatility_tolerance", volatility_tolerance)
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise PricingError("max_iterations must be a positive integer")
    if max_iterations <= 0:
        raise PricingError("max_iterations must be a positive integer")
    return initial, maximum, price_tol, vol_tol, max_iterations


def _check_target_price(
    option_price: Number,
    lower_bound: float,
    upper_bound: float,
    price_tolerance: float,
    time_to_expiry: float,
) -> float:
    target = _nonnegative("option_price", option_price)
    if target < lower_bound - price_tolerance or target > upper_bound + price_tolerance:
        raise NoArbitrageError(
            "option price {:.12g} is outside no-arbitrage bounds "
            "[{:.12g}, {:.12g}]".format(target, lower_bound, upper_bound)
        )
    if time_to_expiry == 0.0:
        raise ImpliedVolatilityError(
            "implied volatility is undefined at expiry; all volatilities have "
            "the same intrinsic value"
        )
    if target <= lower_bound + price_tolerance:
        return lower_bound
    if target >= upper_bound - price_tolerance:
        raise ImpliedVolatilityError(
            "price is at the strict upper no-arbitrage limit, which has no "
            "finite Black--Scholes implied volatility"
        )
    return target


def implied_vol_brent(
    option_price: Number,
    spot: Number,
    strike: Number,
    time_to_expiry: Number,
    rate: Number,
    option_type: Union[OptionType, str],
    dividend_yield: Number = 0.0,
    *,
    initial_upper: Number = 1.0,
    max_volatility: Number = 10.0,
    price_tolerance: Number = 1e-12,
    volatility_tolerance: Number = 1e-12,
    max_iterations: int = 100,
) -> float:
    """Invert a spot-convention option price with bracketed Brent iteration.

    Static no-arbitrage bounds are checked before iteration.  A quote at its
    deterministic lower bound returns ``0.0``; a quote at the strict upper
    bound has no finite solution and raises :class:`ImpliedVolatilityError`.
    ``initial_upper`` is doubled as needed, never beyond ``max_volatility``.
    """

    kind = OptionType.parse(option_type)
    s = _positive("spot", spot)
    k = _positive("strike", strike)
    t = _nonnegative("time_to_expiry", time_to_expiry)
    r = _finite("rate", rate)
    q = _finite("dividend_yield", dividend_yield)
    initial, maximum, price_tol, vol_tol, iterations = _validate_solver_settings(
        initial_upper,
        max_volatility,
        price_tolerance,
        volatility_tolerance,
        max_iterations,
    )
    lower_bound, upper_bound = no_arbitrage_bounds(s, k, t, r, kind, q)
    target = _check_target_price(
        option_price, lower_bound, upper_bound, price_tol, t
    )
    if target == lower_bound:
        return 0.0

    def residual(volatility: float) -> float:
        return bs_price(s, k, t, r, volatility, kind, q) - target

    upper_vol = initial
    upper_residual = residual(upper_vol)
    while upper_residual < 0.0 and upper_vol < maximum:
        upper_vol = min(2.0 * upper_vol, maximum)
        upper_residual = residual(upper_vol)
    if upper_residual < -price_tol:
        raise ImpliedVolatilityError(
            "could not bracket implied volatility at or below max_volatility "
            "{:.12g}; model price is {:.12g} versus target {:.12g}".format(
                maximum, upper_residual + target, target
            )
        )
    return _brentq(residual, 0.0, upper_vol, price_tol, vol_tol, iterations)


def implied_vol_brent_forward(
    option_price: Number,
    forward: Number,
    strike: Number,
    time_to_expiry: Number,
    discount_factor: Number,
    option_type: Union[OptionType, str],
    *,
    initial_upper: Number = 1.0,
    max_volatility: Number = 10.0,
    price_tolerance: Number = 1e-12,
    volatility_tolerance: Number = 1e-12,
    max_iterations: int = 100,
) -> float:
    """Invert a forward-convention (Black--76) option price with Brent."""

    kind = OptionType.parse(option_type)
    f = _positive("forward", forward)
    k = _positive("strike", strike)
    t = _nonnegative("time_to_expiry", time_to_expiry)
    discount = _positive("discount_factor", discount_factor)
    initial, maximum, price_tol, vol_tol, iterations = _validate_solver_settings(
        initial_upper,
        max_volatility,
        price_tolerance,
        volatility_tolerance,
        max_iterations,
    )
    lower_bound, upper_bound = forward_no_arbitrage_bounds(f, k, discount, kind)
    target = _check_target_price(
        option_price, lower_bound, upper_bound, price_tol, t
    )
    if target == lower_bound:
        return 0.0

    def residual(volatility: float) -> float:
        return bs_forward_price(f, k, t, discount, volatility, kind) - target

    upper_vol = initial
    upper_residual = residual(upper_vol)
    while upper_residual < 0.0 and upper_vol < maximum:
        upper_vol = min(2.0 * upper_vol, maximum)
        upper_residual = residual(upper_vol)
    if upper_residual < -price_tol:
        raise ImpliedVolatilityError(
            "could not bracket forward implied volatility at or below "
            "max_volatility {:.12g}".format(maximum)
        )
    return _brentq(residual, 0.0, upper_vol, price_tol, vol_tol, iterations)


def estimate_forward(
    call_price: Number,
    put_price: Number,
    strike: Number,
    time_to_expiry: Number,
    rate: Number = 0.0,
    *,
    discount_factor: Optional[Number] = None,
) -> float:
    """Estimate the forward using European put--call parity.

    With discount factor ``D``, parity is ``C - P = D * (F - K)``, hence
    ``F = K + (C - P) / D``.  Supply either a continuously-compounded ``rate``
    (from which ``D = exp(-rate*T)`` is computed) or ``discount_factor``.
    Mid, bid or ask prices may be used, but the two prices must use the same
    quote convention.
    """

    call = _nonnegative("call_price", call_price)
    put = _nonnegative("put_price", put_price)
    k = _positive("strike", strike)
    t = _nonnegative("time_to_expiry", time_to_expiry)
    if discount_factor is None:
        r = _finite("rate", rate)
        discount = math.exp(-r * t)
    else:
        discount = _positive("discount_factor", discount_factor)
    forward = k + (call - put) / discount
    if not math.isfinite(forward) or forward <= 0.0:
        raise NoArbitrageError(
            "put-call parity implies a non-positive forward {:.12g}; check "
            "that call and put quotes are synchronized".format(forward)
        )
    return forward


# Descriptive compatibility names for callers that prefer expanded spelling.
black_scholes_price = bs_price
black_scholes_delta = bs_delta
black_scholes_gamma = bs_gamma
black_scholes_vega = bs_vega
black_scholes_greeks = bs_greeks
implied_volatility = implied_vol_brent
forward_from_put_call_parity = estimate_forward

