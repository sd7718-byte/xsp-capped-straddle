import numpy as np
import pandas as pd
import pytest

from xsp_straddle.forecast import (
    DEFAULT_VARIANCE_FLOOR,
    InsufficientHistoryError,
    fit_har,
    future_realized_variance,
    har_dataset,
    realized_variance_features,
    rolling_har_forecast,
)


def _close_from_returns(returns, start="2020-01-02"):
    returns = np.asarray(returns, dtype=float)
    index = pd.bdate_range(start, periods=len(returns) + 1)
    close = 100.0 * np.exp(np.concatenate(([0.0], np.cumsum(returns))))
    return pd.Series(close, index=index, name="close")


def _stochastic_close(n=360, seed=7, start="2019-01-02"):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0002, 0.012, n - 1)
    returns += 0.003 * np.sin(np.arange(n - 1) / 9.0)
    return _close_from_returns(returns, start=start)


def test_trailing_features_and_future_target_have_exact_strict_alignment():
    returns = np.linspace(-0.022, 0.026, 34)
    close = _close_from_returns(returns)
    features = realized_variance_features(close)
    target = future_realized_variance(close, horizon=3)

    # Return number j is stamped at close.index[j + 1].
    timestamp = close.index[25]
    assert features.loc[timestamp, "v1"] == pytest.approx(252.0 * returns[24] ** 2)
    assert features.loc[timestamp, "v5"] == pytest.approx(
        252.0 / 5.0 * np.sum(returns[20:25] ** 2)
    )
    assert features.loc[timestamp, "v22"] == pytest.approx(
        252.0 / 22.0 * np.sum(returns[3:25] ** 2)
    )
    assert target.loc[timestamp] == pytest.approx(
        252.0 / 3.0 * np.sum(returns[25:28] ** 2)
    )
    data = har_dataset(close, horizon=3)
    assert data.loc[timestamp, "target_end"] == close.index[28]
    assert data.loc[timestamp, "target_end"] > timestamp


def test_fit_guard_and_log_variance_to_volatility_interval_conversion():
    rng = np.random.default_rng(41)
    features = pd.DataFrame(
        np.exp(rng.normal(-4.0, 0.55, size=(80, 3))),
        columns=["v1", "v5", "v22"],
    )
    log_target = (
        -0.35
        + 0.20 * np.log(features["v1"])
        + 0.25 * np.log(features["v5"])
        + 0.50 * np.log(features["v22"])
        + rng.normal(0.0, 0.18, len(features))
    )
    target = np.exp(log_target).rename("future_variance")
    with pytest.raises(InsufficientHistoryError):
        fit_har(features.iloc[:4], target.iloc[:4])

    model = fit_har(features, target, lower_quantile=0.10, upper_quantile=0.90)
    prediction = model.predict(features.iloc[-1])
    assert prediction.point_vol == pytest.approx(np.exp(0.5 * prediction.log_variance))
    assert prediction.lower_vol == pytest.approx(
        np.exp(0.5 * (prediction.log_variance + model.lower_residual))
    )
    assert prediction.upper_vol == pytest.approx(
        np.exp(0.5 * (prediction.log_variance + model.upper_residual))
    )
    assert 0.0 < prediction.lower_vol <= prediction.point_vol <= prediction.upper_vol


def test_future_changes_do_not_change_a_historical_forecast():
    close = _stochastic_close(n=380)
    cutoff = close.index[275]
    kwargs = dict(
        horizon=7,
        training_window=110,
        min_train_size=55,
        lower_quantile=0.10,
        upper_quantile=0.90,
    )
    baseline = rolling_har_forecast(close, **kwargs)

    modified = close.copy()
    future_count = int((modified.index > cutoff).sum())
    # Alter every future close, including the first future return and all future
    # training labels.  Nothing stamped at cutoff is permitted to observe it.
    modified.loc[modified.index > cutoff] *= np.exp(
        np.linspace(0.4, -0.3, future_count)
    )
    changed = rolling_har_forecast(modified, **kwargs)

    columns = [
        "point_vol",
        "lower_vol",
        "upper_vol",
        "predicted_log_variance",
        "fit_timestamp",
        "latest_target_end",
        "n_train",
        "beta0",
        "beta1",
        "beta5",
        "beta22",
    ]
    pd.testing.assert_series_equal(baseline.loc[cutoff, columns], changed.loc[cutoff, columns])
    assert baseline.loc[cutoff, "latest_target_end"] <= cutoff


def test_monthly_refits_are_frozen_within_each_calendar_month():
    close = _stochastic_close(n=330, seed=19)
    result = rolling_har_forecast(
        close,
        horizon=5,
        training_window=90,
        min_train_size=45,
        lower_quantile=0.10,
        upper_quantile=0.90,
    ).dropna(subset=["point_vol"])
    assert not result.empty

    periods = result.index.to_period("M")
    for _, month in result.groupby(periods):
        assert month["fit_timestamp"].nunique() == 1
        assert int(month["is_refit"].sum()) <= 1
        for coefficient in ["beta0", "beta1", "beta5", "beta22"]:
            assert month[coefficient].nunique() == 1
    assert result["fit_timestamp"].nunique() >= 3


def test_strict_default_history_guard_and_flat_prices_are_deterministic():
    close = pd.Series(100.0, index=pd.bdate_range("2022-01-03", periods=100))
    strict = rolling_har_forecast(close, horizon=5)
    assert strict["point_vol"].isna().all()
    assert (strict["n_train"] == 0).all()

    exploratory = rolling_har_forecast(
        close,
        horizon=5,
        training_window=35,
        min_train_size=5,
    ).dropna(subset=["point_vol"])
    assert not exploratory.empty
    expected_floor_vol = np.sqrt(DEFAULT_VARIANCE_FLOOR)
    assert np.all(np.isfinite(exploratory[["lower_vol", "point_vol", "upper_vol"]]))
    assert exploratory["point_vol"].to_numpy() == pytest.approx(expected_floor_vol)
    assert np.all(exploratory["lower_vol"] <= exploratory["point_vol"])
    assert np.all(exploratory["point_vol"] <= exploratory["upper_vol"])


@pytest.mark.parametrize("bad_close", [[100.0, 0.0], [100.0, np.nan], [100.0, np.inf]])
def test_invalid_closes_are_rejected(bad_close):
    close = pd.Series(bad_close, index=pd.bdate_range("2024-01-02", periods=2))
    with pytest.raises(ValueError):
        realized_variance_features(close)
