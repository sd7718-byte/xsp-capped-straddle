import pandas as pd
import pytest
from pandas.api.types import is_datetime64_any_dtype

from xsp_straddle.data import (
    DataValidationError,
    asof_forecast,
    calendar_dte,
    validate_forecasts,
    validate_option_quotes,
)


def _quotes():
    rows = []
    for strike in [99.0, 100.0]:
        for right, bid, ask in [("C", 2.0, 2.1), ("P", 1.8, 1.9)]:
            rows.append(
                {
                    "symbol": "XSP",
                    "settlement": "PM",
                    "timestamp": "2026-01-07T20:45:00Z",
                    "expiration": "2026-02-20T21:00:00Z",
                    "strike": strike,
                    "option_type": right,
                    "bid": bid,
                    "ask": ask,
                    "spot": 100.0,
                    "hedge_bid": 99.99,
                    "hedge_ask": 100.01,
                    "rate": 0.04,
                    "dividend_yield": 0.01,
                }
            )
    return pd.DataFrame(rows)


def test_option_quote_validation_normalizes_and_sorts():
    result = validate_option_quotes(_quotes().iloc[::-1])
    assert list(result["option_type"].unique()) == ["C", "P"]
    # Pandas may use nanosecond or microsecond backing storage depending on the
    # installed version. The research invariant is timezone-aware UTC time,
    # not a particular internal resolution.
    assert is_datetime64_any_dtype(result["timestamp"])
    assert str(result["timestamp"].dt.tz) == "UTC"


def test_option_quote_validation_rejects_midpoint_only_and_crossed_inputs():
    with pytest.raises(DataValidationError, match="missing option columns"):
        validate_option_quotes(_quotes().drop(columns="ask"))
    crossed = _quotes()
    crossed.loc[0, "bid"] = 3.0
    with pytest.raises(DataValidationError, match="bid cannot exceed ask"):
        validate_option_quotes(crossed)


def test_option_quote_validation_rejects_am_settled_or_wrong_product():
    am = _quotes()
    am["settlement"] = "AM"
    with pytest.raises(DataValidationError, match="PM-settled"):
        validate_option_quotes(am)
    wrong = _quotes()
    wrong["symbol"] = "SPY"
    with pytest.raises(DataValidationError, match="only XSP"):
        validate_option_quotes(wrong)


def test_option_quote_validation_requires_synchronised_call_put_pairs():
    with pytest.raises(DataValidationError, match="both call and put"):
        validate_option_quotes(_quotes().iloc[1:])


def test_forecast_asof_join_never_uses_future_row():
    forecasts = validate_forecasts(
        pd.DataFrame(
            {
                "timestamp": ["2026-01-01", "2026-01-10"],
                "point_vol": [0.2, 0.9],
                "lower_vol": [0.18, 0.8],
                "upper_vol": [0.22, 1.0],
            }
        )
    )
    row = asof_forecast(forecasts, pd.Timestamp("2026-01-07", tz="UTC"))
    assert row["point_vol"] == pytest.approx(0.2)


def test_calendar_dte_uses_dates():
    assert calendar_dte(pd.Timestamp("2026-01-07T20:45Z"), pd.Timestamp("2026-02-20T21:00Z")) == 44
