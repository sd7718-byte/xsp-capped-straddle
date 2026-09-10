"""Command-line interface for forecasts, signals, backtests, and diagnostics."""

import argparse
from dataclasses import asdict, is_dataclass
from enum import Enum
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .config import StrategyConfig
from .costs import CostAssumptions
from .data import asof_forecast, read_option_quotes_csv, validate_forecasts
from .diagnostics import (
    BacktestComparison,
    DiagnosticThresholds,
    ReturnAttribution,
    ScenarioResult,
    SensitivityResult,
    TradeContribution,
    evaluate_backtest,
)
from .forecast import rolling_har_forecast
from .signal import analyze_snapshot


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return asdict(value)
    raise TypeError("cannot JSON-encode {}".format(type(value).__name__))


def _emit_json(value: Any, output: Optional[str] = None) -> None:
    text = json.dumps(value, default=_json_default, indent=2, sort_keys=True)
    if output:
        Path(output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def _cmd_forecast(args: argparse.Namespace) -> int:
    source = pd.read_csv(args.closes)
    missing = {args.timestamp_column, args.close_column}.difference(source.columns)
    if missing:
        raise ValueError("close CSV missing columns: {}".format(", ".join(sorted(missing))))
    timestamps = pd.to_datetime(source[args.timestamp_column], errors="raise", utc=True)
    close = pd.Series(
        pd.to_numeric(source[args.close_column], errors="raise").to_numpy(dtype=float),
        index=pd.DatetimeIndex(timestamps),
        name="close",
    ).sort_index()
    result = rolling_har_forecast(
        close,
        horizon=args.horizon,
        training_window=args.training_window,
        min_train_size=args.min_train_size,
        lower_quantile=args.lower_quantile,
        upper_quantile=args.upper_quantile,
    )
    output = result.reset_index()
    output = output.rename(columns={output.columns[0]: "timestamp"})
    output.to_csv(args.output, index=False)
    usable = int(output["point_vol"].notna().sum())
    _emit_json(
        {
            "command": "forecast",
            "output": str(Path(args.output).resolve()),
            "rows": len(output),
            "usable_forecasts": usable,
            "strict_min_train_size": args.min_train_size is None,
        }
    )
    return 0


def _load_forecast_bounds(args: argparse.Namespace, timestamp: pd.Timestamp) -> Dict[str, float]:
    if args.forecasts:
        forecasts = validate_forecasts(pd.read_csv(args.forecasts))
        row = asof_forecast(forecasts, timestamp)
        return {
            "lower": float(row["lower_vol"]),
            "upper": float(row["upper_vol"]),
        }
    if args.lower_vol is None or args.upper_vol is None:
        raise ValueError("supply --forecasts or both --lower-vol and --upper-vol")
    return {"lower": float(args.lower_vol), "upper": float(args.upper_vol)}


def _cmd_signal(args: argparse.Namespace) -> int:
    quotes = read_option_quotes_csv(args.quotes)
    timestamp = (
        pd.to_datetime(args.timestamp, utc=True)
        if args.timestamp
        else quotes["timestamp"].max()
    )
    snapshot = quotes[quotes["timestamp"] == timestamp]
    if snapshot.empty:
        raise ValueError("no quote snapshot exists at {}".format(timestamp))
    bounds = _load_forecast_bounds(args, timestamp)
    decision = analyze_snapshot(
        snapshot,
        pd.to_datetime(args.expiration, utc=True),
        forecast_lower_vol=bounds["lower"],
        forecast_upper_vol=bounds["upper"],
        cost_assumptions=CostAssumptions(
            args.option_commission,
            args.projected_hedge_cost,
            args.other_cost,
        ),
        config=StrategyConfig(),
        approximate_bid_iv=args.approximate_bid_iv,
        approximate_ask_iv=args.approximate_ask_iv,
        prefilter_cost_vol=args.prefilter_cost_vol,
    )
    _emit_json(decision, args.output)
    return 0


def _comparison_from_json(payload: Dict[str, Any]) -> BacktestComparison:
    scenarios = payload["scenarios"]
    return BacktestComparison(
        approximate_iv=ScenarioResult("approximate_iv", scenarios["approximate_iv"]),
        exact_iv=ScenarioResult("exact_iv", scenarios["exact_iv"]),
        midpoint_fills=ScenarioResult("midpoint_fills", scenarios["midpoint_fills"]),
        executable_fills=ScenarioResult("executable_fills", scenarios["executable_fills"]),
        doubled_costs=ScenarioResult("doubled_costs", scenarios["doubled_costs"]),
        trade_contributions=tuple(
            TradeContribution(
                str(item["trade_id"]), str(item["direction"]), float(item["net_profit"])
            )
            for item in payload.get("trade_contributions", [])
        ),
        return_attribution=ReturnAttribution(
            float(payload["return_attribution"]["delta_hedged_return"]),
            float(payload["return_attribution"]["overnight_gap_return"]),
        ),
        dte_sensitivities=tuple(
            SensitivityResult(str(item["setting"]), float(item["net_profit"]))
            for item in payload.get("dte_sensitivities", [])
        ),
        hedge_frequency_sensitivities=tuple(
            SensitivityResult(str(item["setting"]), float(item["net_profit"]))
            for item in payload.get("hedge_frequency_sensitivities", [])
        ),
    )


def _cmd_diagnose(args: argparse.Namespace) -> int:
    payload = json.loads(Path(args.evidence).read_text(encoding="utf-8"))
    report = evaluate_backtest(_comparison_from_json(payload), DiagnosticThresholds())
    _emit_json(report, args.output)
    return 0 if report.accepted else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xsp-straddle",
        description="Research the delta-hedged capped XSP straddle without midpoint assumptions.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    forecast = subparsers.add_parser("forecast", help="fit leakage-safe rolling HAR forecasts")
    forecast.add_argument("--closes", required=True, help="CSV containing timestamp and close")
    forecast.add_argument("--output", required=True, help="forecast CSV to create")
    forecast.add_argument("--timestamp-column", default="timestamp")
    forecast.add_argument("--close-column", default="close")
    forecast.add_argument("--horizon", type=int, required=True)
    forecast.add_argument("--training-window", type=int, default=5 * 252)
    forecast.add_argument(
        "--min-train-size",
        type=int,
        default=None,
        help="omit for the strict full-window default; smaller values are exploratory",
    )
    forecast.add_argument("--lower-quantile", type=float, default=0.05)
    forecast.add_argument("--upper-quantile", type=float, default=0.95)
    forecast.set_defaults(func=_cmd_forecast)

    signal = subparsers.add_parser("signal", help="evaluate one exact package-level signal")
    signal.add_argument("--quotes", required=True)
    signal.add_argument("--expiration", required=True)
    signal.add_argument("--timestamp", help="ISO timestamp; defaults to the latest snapshot")
    bounds = signal.add_mutually_exclusive_group(required=False)
    bounds.add_argument("--forecasts", help="forecast CSV for leakage-safe as-of lookup")
    bounds.add_argument("--lower-vol", type=float)
    signal.add_argument("--upper-vol", type=float)
    signal.add_argument("--option-commission", type=float, required=True)
    signal.add_argument("--projected-hedge-cost", type=float, required=True)
    signal.add_argument("--other-cost", type=float, default=0.0)
    signal.add_argument("--approximate-bid-iv", type=float)
    signal.add_argument("--approximate-ask-iv", type=float)
    signal.add_argument("--prefilter-cost-vol", type=float, default=0.0)
    signal.add_argument("--output")
    signal.set_defaults(func=_cmd_signal)

    diagnose = subparsers.add_parser("diagnose", help="apply all mandatory rejection tests")
    diagnose.add_argument("--evidence", required=True, help="comparison-evidence JSON")
    diagnose.add_argument("--output")
    diagnose.set_defaults(func=_cmd_diagnose)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

