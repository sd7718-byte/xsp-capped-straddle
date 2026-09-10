# XSP Capped-Straddle Research Engine

An auditable Python research implementation of a delta-hedged, capped XSP
straddle. The project separates signal construction from execution assumptions
and subjects candidate results to spread, transaction-cost, hedge-frequency,
and model-approximation stress tests.

This is research software, not investment advice or a live-trading system. It
does not contain brokerage credentials or order-routing code.

## Research design

- Builds exact package-level option signals from executable bid/ask quotes.
- Fits leakage-safe rolling HAR realized-volatility forecasts.
- Prices and filters the capped straddle using configurable cost assumptions.
- Simulates scheduled and threshold-triggered delta hedges.
- Compares approximate versus exact implied volatility, midpoint versus
  executable fills, doubled costs, DTE choices, and hedge frequencies.
- Rejects fragile evidence through explicit diagnostic thresholds.

## Install and test

```sh
git clone https://github.com/sd7718-byte/xsp-capped-straddle.git
cd xsp-capped-straddle
python3 -m pip install -e ".[dev]"
python3 -m pytest
```

The core package is in `src/xsp_straddle`; unit tests are in `tests`.

## Command-line interface

After installation, inspect the available research commands with:

```sh
xsp-straddle --help
xsp-straddle forecast --help
xsp-straddle signal --help
xsp-straddle diagnose --help
```

Inputs must be point-in-time market data. Reported output should not be called
an out-of-sample result unless the underlying dataset, chronology, costs, and
validation split are independently documented.

