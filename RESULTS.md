# Empirical results

## Current evidence status

**No historical out-of-sample return, Sharpe, or P&L is claimed.** The current
evidence covers deterministic pricing, option selection, signal construction,
hedging, costs, stress scenarios, and rejection diagnostics.

## Minimum dataset requirements

- [ ] Timestamped historical XSP option bid/ask quotes for every package leg.
- [ ] Contemporaneous underlying prices and interest/dividend inputs.
- [ ] Corporate-action and trading-calendar normalization.
- [ ] Executable entry, exit, and hedge prices rather than midpoints.
- [ ] Chronological forecast training with no revised-data leakage.
- [ ] Frozen cost, DTE, delta-wing, and hedge-frequency assumptions.
- [ ] Complete trade ledger and reproducible dataset hash.

## Out-of-sample report

| Metric | Estimate | Confidence interval / note |
| --- | ---: | --- |
| Net executable P&L | `PENDING` | `PENDING` |
| Daily Sharpe | `PENDING` | `PENDING` |
| Maximum drawdown | `PENDING` | `PENDING` |
| Trades | `PENDING` | `PENDING` |
| Win rate | `PENDING` | `PENDING` |
| Average bid/ask cost | `PENDING` | `PENDING` |
| Hedge turnover | `PENDING` | `PENDING` |

## Mandatory falsification table

| Scenario | Net P&L | Accepted? |
| --- | ---: | --- |
| Approximate implied volatility | `PENDING` | `PENDING` |
| Exact implied volatility | `PENDING` | `PENDING` |
| Midpoint fills | `PENDING` | `PENDING` |
| Executable bid/ask fills | `PENDING` | `PENDING` |
| Doubled transaction costs | `PENDING` | `PENDING` |
| Nearby entry DTEs | `PENDING` | `PENDING` |
| Nearby hedge frequencies | `PENDING` | `PENDING` |

The strategy must be rejected if profitability exists only at midpoints, is
removed by exact implied volatility or doubled costs, depends on a few
short-volatility winners, or reverses under nearby DTE/hedging choices.

