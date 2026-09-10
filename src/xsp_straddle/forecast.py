"""Leakage-safe HAR forecasts of future close-to-close realized volatility.

The input is a daily close series.  Consequently ``log(close).diff()`` contains
the close-to-close (and therefore overnight) return.  A forecast stamped at
time ``t`` uses features through ``t`` and training targets whose final return
is also known by ``t``; the forecast target itself always starts at ``t + 1``.

All variances are annualized decimal variances and all reported volatilities
are annualized decimals (``0.20`` means 20% volatility).
"""

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252
DEFAULT_TRAINING_WINDOW = 5 * TRADING_DAYS_PER_YEAR
DEFAULT_VARIANCE_FLOOR = 1.0e-12
HAR_FEATURE_COLUMNS = ("v1", "v5", "v22")
_PARAMETER_COLUMNS = ("beta0", "beta1", "beta5", "beta22")


class InsufficientHistoryError(ValueError):
    """Raised when too few completed targets are available to fit HAR."""


@dataclass(frozen=True)
class HARPrediction:
    """A point prediction and empirical interval in annualized volatility."""

    log_variance: float
    variance: float
    point_vol: float
    lower_vol: float
    upper_vol: float


@dataclass(frozen=True)
class HARModel:
    """Fitted log-variance HAR model and its empirical residual bounds."""

    coefficients: np.ndarray
    residuals: np.ndarray
    lower_residual: float
    upper_residual: float
    lower_quantile: float
    upper_quantile: float
    variance_floor: float
    n_obs: int
    fitted_at: Optional[pd.Timestamp] = None
    training_start: Optional[pd.Timestamp] = None
    training_end: Optional[pd.Timestamp] = None
    latest_target_end: Optional[pd.Timestamp] = None

    @property
    def params(self) -> pd.Series:
        """Return named OLS coefficients (intercept, daily, weekly, monthly)."""

        return pd.Series(self.coefficients.copy(), index=_PARAMETER_COLUMNS)

    def predict(
        self,
        features: Union[pd.Series, Mapping[str, float], Sequence[float], np.ndarray],
    ) -> HARPrediction:
        """Predict volatility for one ``v1``/``v5``/``v22`` feature row.

        Empirical quantiles are residuals in *log variance*.  They are added in
        that space and then converted to volatility as
        ``exp(0.5 * (predicted_log_variance + residual_quantile))``.
        """

        values = _coerce_feature_row(features)
        log_features = np.log(np.maximum(values, self.variance_floor))
        design_row = np.concatenate(([1.0], log_features))
        predicted_log_variance = float(design_row @ self.coefficients)
        point_vol = _log_variance_to_volatility(
            predicted_log_variance, self.variance_floor
        )
        lower_vol = _log_variance_to_volatility(
            predicted_log_variance + self.lower_residual, self.variance_floor
        )
        upper_vol = _log_variance_to_volatility(
            predicted_log_variance + self.upper_residual, self.variance_floor
        )
        # Numerical noise in a nearly exact OLS fit must not invert or exclude
        # the point forecast from what is presented as a prediction interval.
        lower_vol = min(lower_vol, point_vol)
        upper_vol = max(upper_vol, point_vol)
        return HARPrediction(
            log_variance=predicted_log_variance,
            variance=point_vol * point_vol,
            point_vol=point_vol,
            lower_vol=lower_vol,
            upper_vol=upper_vol,
        )


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("{} must be a positive integer".format(name))
    value = int(value)
    if value <= 0:
        raise ValueError("{} must be a positive integer".format(name))
    return value


def _validate_variance_floor(variance_floor: float) -> float:
    variance_floor = float(variance_floor)
    if not np.isfinite(variance_floor) or variance_floor <= 0.0:
        raise ValueError("variance_floor must be finite and strictly positive")
    return variance_floor


def _validate_quantiles(lower_quantile: float, upper_quantile: float) -> Tuple[float, float]:
    lower_quantile = float(lower_quantile)
    upper_quantile = float(upper_quantile)
    if not (0.0 <= lower_quantile < upper_quantile <= 1.0):
        raise ValueError("quantiles must satisfy 0 <= lower < upper <= 1")
    return lower_quantile, upper_quantile


def _validate_close(close: pd.Series, require_datetime_index: bool = False) -> pd.Series:
    if not isinstance(close, pd.Series):
        raise TypeError("close must be a pandas Series")
    if close.empty:
        raise ValueError("close must not be empty")
    if require_datetime_index and not isinstance(close.index, pd.DatetimeIndex):
        raise TypeError("rolling forecasts require a pandas DatetimeIndex")
    if not close.index.is_monotonic_increasing or not close.index.is_unique:
        raise ValueError("close index must be strictly increasing and unique")
    try:
        clean = close.astype(float)
    except (TypeError, ValueError) as exc:
        raise TypeError("close values must be numeric") from exc
    values = clean.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("close must contain only finite values")
    if np.any(values <= 0.0):
        raise ValueError("close values must be strictly positive")
    clean.name = close.name
    return clean


def close_to_close_log_returns(close: pd.Series) -> pd.Series:
    """Return ``log(close_t / close_(t-1))``, including each overnight move."""

    close = _validate_close(close)
    result = np.log(close).diff()
    result.name = "log_return"
    return result


def realized_variance_features(
    close: pd.Series,
    annualization: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Compute annualized trailing ``v1``, ``v5`` and ``v22`` from closes.

    For window ``n`` the value stamped at ``t`` is exactly
    ``annualization / n * sum(r[t-i] ** 2, i=0..n-1)``.  The current day's
    close-to-close return is included, so features are intended for use after
    the timestamp's close is known.
    """

    annualization = _positive_int("annualization", annualization)
    returns_squared = close_to_close_log_returns(close).pow(2)
    result = pd.DataFrame(index=close.index)
    for window in (1, 5, 22):
        result["v{}".format(window)] = (
            returns_squared.rolling(window=window, min_periods=window).sum()
            * float(annualization)
            / float(window)
        )
    return result


def future_realized_variance(
    close: pd.Series,
    horizon: int,
    annualization: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Return the annualized variance over the ``horizon`` returns after ``t``.

    The value indexed by ``t`` uses only ``r[t+1]`` through ``r[t+horizon]``.
    It never includes ``r[t]``.  Final rows without a complete future horizon
    are ``NaN``.
    """

    horizon = _positive_int("horizon", horizon)
    annualization = _positive_int("annualization", annualization)
    returns_squared = close_to_close_log_returns(close).pow(2)
    result = (
        returns_squared.rolling(window=horizon, min_periods=horizon).sum().shift(-horizon)
        * float(annualization)
        / float(horizon)
    )
    result.name = "future_variance"
    return result


def har_dataset(
    close: pd.Series,
    horizon: int,
    annualization: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Build auditable HAR features, future target and target completion time.

    ``target_end`` is the timestamp of the last return in ``future_variance``.
    A row may be used for training at forecast timestamp ``s`` only when
    ``target_end <= s``.  Rows are retained with ``NaN`` values so alignment is
    directly inspectable.
    """

    close = _validate_close(close, require_datetime_index=True)
    horizon = _positive_int("horizon", horizon)
    features = realized_variance_features(close, annualization=annualization)
    target = future_realized_variance(close, horizon, annualization=annualization)
    target_end = pd.Series(close.index, index=close.index, name="target_end").shift(-horizon)
    return features.join(target).join(target_end)


# A descriptive alias for callers that prefer a verb to the shorter public name.
build_har_dataset = har_dataset


def _coerce_feature_row(
    features: Union[pd.Series, Mapping[str, float], Sequence[float], np.ndarray]
) -> np.ndarray:
    if isinstance(features, pd.Series):
        values = features.loc[list(HAR_FEATURE_COLUMNS)].to_numpy(dtype=float)
    elif isinstance(features, Mapping):
        values = np.asarray([features[name] for name in HAR_FEATURE_COLUMNS], dtype=float)
    else:
        values = np.asarray(features, dtype=float)
    if values.shape != (3,):
        raise ValueError("features must contain exactly v1, v5 and v22")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("HAR variance features must be finite and non-negative")
    return values


def _log_variance_to_volatility(log_variance: float, variance_floor: float) -> float:
    if not np.isfinite(log_variance):
        raise ValueError("predicted log variance is not finite")
    minimum = float(np.log(variance_floor))
    maximum = float(np.log(np.finfo(float).max))
    return float(np.exp(0.5 * np.clip(log_variance, minimum, maximum)))


def fit_har(
    features: pd.DataFrame,
    target: Optional[pd.Series] = None,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
    variance_floor: float = DEFAULT_VARIANCE_FLOOR,
    min_samples: int = 5,
    fitted_at: Optional[pd.Timestamp] = None,
    target_end: Optional[pd.Series] = None,
) -> HARModel:
    """Fit OLS to log future variance using log ``v1``/``v5``/``v22``.

    ``features`` may either be a frame with the three feature columns and a
    ``future_variance`` column, or ``target`` may be supplied separately.  Zero
    feature/target variances are floored before taking logs.  Missing rows are
    omitted; negative or infinite variances are rejected.  At least
    ``min_samples`` completed rows (and at least five, one more than the four
    OLS parameters) are required.

    This low-level routine assumes the caller has already enforced target
    availability.  Prefer :func:`rolling_har_forecast` for time-series use.
    """

    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")
    missing = [name for name in HAR_FEATURE_COLUMNS if name not in features.columns]
    if missing:
        raise ValueError("missing HAR feature columns: {}".format(", ".join(missing)))
    if target is None:
        if "future_variance" not in features.columns:
            raise ValueError("target is required when future_variance is not a column")
        target = features["future_variance"]
    if not isinstance(target, pd.Series):
        raise TypeError("target must be a pandas Series")
    variance_floor = _validate_variance_floor(variance_floor)
    lower_quantile, upper_quantile = _validate_quantiles(lower_quantile, upper_quantile)
    min_samples = _positive_int("min_samples", min_samples)
    if min_samples < 5:
        raise ValueError("min_samples must be at least five")

    frame = features.loc[:, list(HAR_FEATURE_COLUMNS)].join(target.rename("_target"), how="inner")
    if target_end is not None:
        if not isinstance(target_end, pd.Series):
            raise TypeError("target_end must be a pandas Series")
        frame = frame.join(target_end.rename("_target_end"), how="left")
    numeric = frame.loc[:, list(HAR_FEATURE_COLUMNS) + ["_target"]]
    finite_or_missing = np.isfinite(numeric.to_numpy(dtype=float)) | numeric.isna().to_numpy()
    if not np.all(finite_or_missing):
        raise ValueError("HAR variances must not contain infinity")
    observed = numeric.dropna()
    if (observed.to_numpy(dtype=float) < 0.0).any():
        raise ValueError("HAR variances must be non-negative")
    if len(observed) < min_samples:
        raise InsufficientHistoryError(
            "HAR requires at least {} completed observations; got {}".format(
                min_samples, len(observed)
            )
        )

    x_values = np.maximum(
        observed.loc[:, list(HAR_FEATURE_COLUMNS)].to_numpy(dtype=float), variance_floor
    )
    y_values = np.maximum(observed["_target"].to_numpy(dtype=float), variance_floor)
    design = np.column_stack((np.ones(len(observed)), np.log(x_values)))
    log_target = np.log(y_values)
    coefficients, _, _, _ = np.linalg.lstsq(design, log_target, rcond=None)
    residuals = log_target - design @ coefficients
    if not np.all(np.isfinite(coefficients)) or not np.all(np.isfinite(residuals)):
        raise ValueError("HAR OLS produced non-finite values")
    raw_lower = float(np.quantile(residuals, lower_quantile))
    raw_upper = float(np.quantile(residuals, upper_quantile))
    # With an intercept, zero should lie between ordinary residual quantiles in
    # normal samples.  Include it explicitly for degenerate/roundoff cases so
    # the returned values remain true lower and upper bounds around the point.
    lower_residual = min(raw_lower, 0.0)
    upper_residual = max(raw_upper, 0.0)

    observed_index = observed.index
    training_start = pd.Timestamp(observed_index[0]) if len(observed_index) else None
    training_end = pd.Timestamp(observed_index[-1]) if len(observed_index) else None
    latest_target_end = None
    if target_end is not None:
        aligned_target_end = target_end.reindex(observed_index).dropna()
        if not aligned_target_end.empty:
            latest_target_end = pd.Timestamp(aligned_target_end.max())
    return HARModel(
        coefficients=np.asarray(coefficients, dtype=float),
        residuals=np.asarray(residuals, dtype=float),
        lower_residual=lower_residual,
        upper_residual=upper_residual,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        variance_floor=variance_floor,
        n_obs=len(observed),
        fitted_at=pd.Timestamp(fitted_at) if fitted_at is not None else None,
        training_start=training_start,
        training_end=training_end,
        latest_target_end=latest_target_end,
    )


def _empty_forecast_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    columns = [
        "point_vol",
        "lower_vol",
        "upper_vol",
        "predicted_log_variance",
        "fit_timestamp",
        "model_asof",
        "latest_target_end",
        "n_train",
        "n_available",
        "is_refit",
        "beta0",
        "beta1",
        "beta5",
        "beta22",
    ]
    result = pd.DataFrame(index=index, columns=columns)
    result.index.name = index.name
    return result


def rolling_har_forecast(
    close: pd.Series,
    horizon: int,
    training_window: int = DEFAULT_TRAINING_WINDOW,
    min_train_size: Optional[int] = None,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
    annualization: int = TRADING_DAYS_PER_YEAR,
    variance_floor: float = DEFAULT_VARIANCE_FLOOR,
    forecast_start: Optional[pd.Timestamp] = None,
    forecast_end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Return monthly-refit, leakage-safe HAR volatility forecasts.

    Parameters
    ----------
    close:
        Positive finite daily closes on a strictly increasing, unique
        ``DatetimeIndex``.  Returns are computed close-to-close, so overnight
        moves are included.
    horizon:
        Number of trading-day returns strictly after a feature timestamp in
        each future-variance target.
    training_window:
        Maximum number of completed labeled observations in a fit.  The
        research default is five years, ``5 * 252 == 1260``.
    min_train_size:
        Completed observations required before the first model.  ``None`` (the
        production default) requires the full ``training_window``.  An explicit
        smaller value is provided only for tests and exploratory, non-production
        research and must still be at least five.

    Notes
    -----
    A model is refit at the first forecast timestamp in each calendar month.
    If history is initially insufficient, fitting is retried on later dates
    until the first model can be formed.  A fit at timestamp ``t`` admits only
    labels with ``target_end <= t`` and is then frozen for the rest of that
    calendar month.  Forecast features themselves are still updated daily.

    The returned frame is indexed by timestamps with complete feature rows.
    Before sufficient training history exists, forecast and coefficient columns
    are ``NaN``/``NaT`` while ``n_available`` reports the guard's observed count.
    ``fit_timestamp`` and its alias ``model_asof`` make monthly reuse auditable.
    Volatility columns, not variance columns, are the public point/lower/upper
    outputs.
    """

    close = _validate_close(close, require_datetime_index=True)
    horizon = _positive_int("horizon", horizon)
    training_window = _positive_int("training_window", training_window)
    annualization = _positive_int("annualization", annualization)
    variance_floor = _validate_variance_floor(variance_floor)
    lower_quantile, upper_quantile = _validate_quantiles(lower_quantile, upper_quantile)
    if min_train_size is None:
        min_train_size = training_window
    min_train_size = _positive_int("min_train_size", min_train_size)
    if min_train_size < 5:
        raise ValueError("min_train_size must be at least five")
    if min_train_size > training_window:
        raise ValueError("min_train_size cannot exceed training_window")

    data = har_dataset(close, horizon=horizon, annualization=annualization)
    feature_complete = data.loc[:, list(HAR_FEATURE_COLUMNS)].notna().all(axis=1)
    forecast_index = data.index[feature_complete]
    if forecast_start is not None:
        forecast_index = forecast_index[forecast_index >= pd.Timestamp(forecast_start)]
    if forecast_end is not None:
        forecast_index = forecast_index[forecast_index <= pd.Timestamp(forecast_end)]
    if len(forecast_index) == 0:
        return _empty_forecast_frame(forecast_index)

    training_complete = data.loc[:, list(HAR_FEATURE_COLUMNS) + ["future_variance", "target_end"]]
    training_complete = training_complete.dropna()
    records = []
    model = None  # type: Optional[HARModel]
    model_month = None  # type: Optional[Tuple[int, int]]

    for timestamp in forecast_index:
        current_month = (timestamp.year, timestamp.month)
        available = training_complete.loc[training_complete["target_end"] <= timestamp]
        available = available.tail(training_window)
        n_available = len(available)
        is_refit = False
        if model is None or current_month != model_month:
            if n_available >= min_train_size:
                model = fit_har(
                    available.loc[:, list(HAR_FEATURE_COLUMNS)],
                    target=available["future_variance"],
                    lower_quantile=lower_quantile,
                    upper_quantile=upper_quantile,
                    variance_floor=variance_floor,
                    min_samples=min_train_size,
                    fitted_at=timestamp,
                    target_end=available["target_end"],
                )
                model_month = current_month
                is_refit = True

        if model is None:
            record = {
                "point_vol": np.nan,
                "lower_vol": np.nan,
                "upper_vol": np.nan,
                "predicted_log_variance": np.nan,
                "fit_timestamp": pd.NaT,
                "model_asof": pd.NaT,
                "latest_target_end": pd.NaT,
                "n_train": 0,
                "n_available": n_available,
                "is_refit": False,
                "beta0": np.nan,
                "beta1": np.nan,
                "beta5": np.nan,
                "beta22": np.nan,
            }
        else:
            prediction = model.predict(data.loc[timestamp, list(HAR_FEATURE_COLUMNS)])
            record = {
                "point_vol": prediction.point_vol,
                "lower_vol": prediction.lower_vol,
                "upper_vol": prediction.upper_vol,
                "predicted_log_variance": prediction.log_variance,
                "fit_timestamp": model.fitted_at,
                "model_asof": model.fitted_at,
                "latest_target_end": model.latest_target_end,
                "n_train": model.n_obs,
                "n_available": n_available,
                "is_refit": is_refit,
                "beta0": model.coefficients[0],
                "beta1": model.coefficients[1],
                "beta5": model.coefficients[2],
                "beta22": model.coefficients[3],
            }
        records.append(record)

    result = pd.DataFrame.from_records(records, index=forecast_index)
    result.index.name = close.index.name
    return result


__all__ = [
    "DEFAULT_TRAINING_WINDOW",
    "DEFAULT_VARIANCE_FLOOR",
    "HARModel",
    "HARPrediction",
    "HAR_FEATURE_COLUMNS",
    "InsufficientHistoryError",
    "TRADING_DAYS_PER_YEAR",
    "build_har_dataset",
    "close_to_close_log_returns",
    "fit_har",
    "future_realized_variance",
    "har_dataset",
    "realized_variance_features",
    "rolling_har_forecast",
]
