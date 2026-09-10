"""Canonical tabular inputs and strict validation for historical research.

The backtester deliberately refuses midpoint-only or unsynchronised data.  Each
option row is a contemporaneous bid/ask observation and carries the underlying
hedge market observed at the same timestamp.
"""

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd


OPTION_COLUMNS = (
    "symbol",
    "settlement",
    "timestamp",
    "expiration",
    "strike",
    "option_type",
    "bid",
    "ask",
    "spot",
    "hedge_bid",
    "hedge_ask",
    "rate",
    "dividend_yield",
)


class DataValidationError(ValueError):
    """Raised when historical inputs are unsafe for an executable backtest."""


def _parse_timestamps(values: pd.Series, name: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    if parsed.isna().any():
        bad = values[parsed.isna()].head(3).tolist()
        raise DataValidationError("{} contains invalid timestamps: {}".format(name, bad))
    return parsed


def validate_option_quotes(
    frame: pd.DataFrame,
    expected_symbol: str = "XSP",
    expected_settlement: str = "PM",
) -> pd.DataFrame:
    """Return a sorted, normalized copy of a synchronized option quote table.

    Required columns are listed in :data:`OPTION_COLUMNS`. ``rate`` and
    ``dividend_yield`` are continuously compounded annual decimal rates.  The
    function requires a single spot/hedge/rate state per timestamp, unique option
    keys, non-crossed markets, and both a call and put for every quoted strike.
    """

    missing = sorted(set(OPTION_COLUMNS).difference(frame.columns))
    if missing:
        raise DataValidationError("missing option columns: {}".format(", ".join(missing)))
    if frame.empty:
        raise DataValidationError("option quote table is empty")
    data = frame.copy()
    data["symbol"] = data["symbol"].astype(str).str.upper().str.strip()
    data["settlement"] = data["settlement"].astype(str).str.upper().str.strip()
    if not (data["symbol"] == expected_symbol.upper()).all():
        raise DataValidationError("option table must contain only {} contracts".format(expected_symbol))
    if not (data["settlement"] == expected_settlement.upper()).all():
        raise DataValidationError(
            "option table must contain only {}-settled contracts".format(expected_settlement)
        )
    data["timestamp"] = _parse_timestamps(data["timestamp"], "timestamp")
    data["expiration"] = _parse_timestamps(data["expiration"], "expiration")
    data["option_type"] = data["option_type"].astype(str).str.upper().str.strip()
    aliases = {"CALL": "C", "PUT": "P"}
    data["option_type"] = data["option_type"].replace(aliases)
    if not data["option_type"].isin(["C", "P"]).all():
        invalid = sorted(data.loc[~data["option_type"].isin(["C", "P"]), "option_type"].unique())
        raise DataValidationError("invalid option_type values: {}".format(invalid))

    numeric = [
        "strike",
        "bid",
        "ask",
        "spot",
        "hedge_bid",
        "hedge_ask",
        "rate",
        "dividend_yield",
    ]
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")
        if not np.isfinite(data[column].to_numpy(dtype=float)).all():
            raise DataValidationError("{} must contain only finite numbers".format(column))
    if (data[["strike", "spot", "hedge_bid", "hedge_ask"]] <= 0.0).any().any():
        raise DataValidationError("strike, spot, and hedge quotes must be positive")
    if (data[["bid", "ask"]] < 0.0).any().any():
        raise DataValidationError("option quotes cannot be negative")
    if (data["bid"] > data["ask"]).any():
        raise DataValidationError("option bid cannot exceed ask")
    if (data["hedge_bid"] > data["hedge_ask"]).any():
        raise DataValidationError("hedge bid cannot exceed ask")
    if (data["expiration"] <= data["timestamp"]).any():
        raise DataValidationError("every expiration must be later than its quote timestamp")

    keys = ["timestamp", "expiration", "strike", "option_type"]
    if data.duplicated(keys).any():
        raise DataValidationError("duplicate option observations for the same synchronized key")

    state_columns = ["spot", "hedge_bid", "hedge_ask", "rate", "dividend_yield"]
    state_counts = data.groupby("timestamp", sort=False)[state_columns].nunique(dropna=False)
    if (state_counts > 1).any().any():
        raise DataValidationError("underlying/rate state is inconsistent within a timestamp")

    right_counts = data.groupby(["timestamp", "expiration", "strike"])["option_type"].nunique()
    if (right_counts != 2).any():
        raise DataValidationError("each timestamp/expiration/strike must contain both call and put")

    return data.sort_values(keys).reset_index(drop=True)


def read_option_quotes_csv(path: Union[str, Path]) -> pd.DataFrame:
    """Read and validate the canonical historical option quote CSV."""

    return validate_option_quotes(pd.read_csv(path))


def validate_forecasts(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize a forecast table consumed by the strategy engine.

    Expected columns are ``timestamp``, ``point_vol``, ``lower_vol`` and
    ``upper_vol``.  Forecast timestamps represent information availability, not
    the end of their prediction horizon.  The engine uses an as-of join.
    """

    required = {"timestamp", "point_vol", "lower_vol", "upper_vol"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataValidationError("missing forecast columns: {}".format(", ".join(missing)))
    if frame.empty:
        raise DataValidationError("forecast table is empty")
    data = frame.copy()
    data["timestamp"] = _parse_timestamps(data["timestamp"], "timestamp")
    for column in ["point_vol", "lower_vol", "upper_vol"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
        if not np.isfinite(data[column].to_numpy(dtype=float)).all() or (data[column] < 0.0).any():
            raise DataValidationError("{} must be finite non-negative decimal volatility".format(column))
    if (data["lower_vol"] > data["point_vol"]).any() or (
        data["point_vol"] > data["upper_vol"]
    ).any():
        raise DataValidationError("forecast bounds must satisfy lower <= point <= upper")
    if data["timestamp"].duplicated().any():
        raise DataValidationError("forecast timestamp must be unique")
    return data.sort_values("timestamp").reset_index(drop=True)


def asof_forecast(forecasts: pd.DataFrame, timestamp: pd.Timestamp) -> pd.Series:
    """Return the last forecast available at or before ``timestamp``."""

    timestamp = pd.to_datetime(timestamp, utc=True)
    eligible = forecasts[forecasts["timestamp"] <= timestamp]
    if eligible.empty:
        raise DataValidationError("no forecast available as of {}".format(timestamp))
    return eligible.iloc[-1]


def calendar_dte(timestamp: pd.Timestamp, expiration: pd.Timestamp) -> int:
    """Calendar days to expiration, using normalized UTC dates."""

    timestamp = pd.to_datetime(timestamp, utc=True)
    expiration = pd.to_datetime(expiration, utc=True)
    return int((expiration.normalize() - timestamp.normalize()).days)
