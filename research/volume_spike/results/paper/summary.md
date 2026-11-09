# Paper test: the path_all CNN with a 2h hold (P1)

Forward paper test of `S5:path_all:d_hold2h` on candles after every study's data end. Plan: `docs/plans/cnn-robustness.md` (P1, rule fixed before any new data was looked at). Code: `research/volume_spike/paper.py`. Ledger: `.cache/paper/ledger/` (gitignored).

- **Rule:** path_all walk-forward CNN, 2026 fold (trained before 2025-07-01, early-stopped on 2025 H2, percentiles of the fold's training reference scores); liquid spikes (vol >= 10x the 24h median, >= 5k USDT qv_base); pct >= 0.98; 5 `CashCapped` slots, highest pct first; enter at the next 5m open; hold 2h; fee 0.1% + slippage 0.05% + latency 0.05% per side (0.4% round trip); $10,000; production engine on 5m candles. Comparison: the same with a 1h hold (`S5:path_all:a_1h`).
- **Window:** signals from 2026-11-02 00:00 UTC to the latest candle, 2026-11-09 00:35 UTC (data end 2026-11-09 00:40).
- **Universe:** 395 coins in the local store (391 listed "all" coins synced daily; 391 with candles to the data end).
- **Updated:** 2026-11-09 00:43 UTC; model `f6e0ef533cef`.

## Result

| Rule | Trades | Open | Win | Mean bp/trade | P&L $ | Equity $ | Return | Max DD |
|---|---|---|---|---|---|---|---|---|
| S5:path_all:d_hold2h (the paper rule) | 5 | 0 | 60% | +28 | +28.27 | 10,028.27 | +0.28% | -0.55% |
| S5:path_all:a_1h (1h) | 5 | 0 | 100% | +109 | +109.33 | 10,109.33 | +1.09% | 0.00% |

## Days

Candidates: liquid spikes scored. Signals: pct >= 0.98. Top: the day's three highest pct.

| Day (UTC) | Candidates | Signals | Top pct | 2h trades | 2h $ | 1h trades | 1h $ |
|---|---|---|---|---|---|---|---|
| 2026-11-02 | 204 | 0 | ZECUSDT 0.882, ONDOUSDT 0.874, NEARUSDT 0.873 | 0 | +0.00 | 0 | +0.00 |
| 2026-11-03 | 212 | 0 | MARSCOINUSDT 0.910, XLMUSDT 0.873, XLMUSDT 0.868 | 0 | +0.00 | 0 | +0.00 |
| 2026-11-04 | 449 | 0 | TRUMPUSDT 0.935, TRUMPUSDT 0.922, TRUMPUSDT 0.918 | 0 | +0.00 | 0 | +0.00 |
| 2026-11-05 | 599 | 28 | DASHUSDT 0.984, XPLUSDT 0.984, ONDOUSDT 0.984 | 5 | +28.27 | 5 | +109.33 |
| 2026-11-06 | 140 | 0 | FILUSDT 0.856, TIAUSDT 0.854, BMTUSDT 0.853 | 0 | +0.00 | 0 | +0.00 |
| 2026-11-07 | 120 | 0 | PUMPUSDT 0.829, PUMPUSDT 0.821, PUMPUSDT 0.805 | 0 | +0.00 | 0 | +0.00 |
| 2026-11-08 | 407 | 0 | XPLUSDT 0.969, HUMAUSDT 0.967, XPLUSDT 0.964 | 0 | +0.00 | 0 | +0.00 |
| 2026-11-09 | 24 | 0 | XPLUSDT 0.926, PUMPUSDT 0.922, ASTERUSDT 0.912 | 0 | +0.00 | 0 | +0.00 |

## Trades: S5:path_all:d_hold2h (the paper rule)

| Symbol | Signal | pct | Entry | Exit | Entry px | Exit px | Net | $ | Status |
|---|---|---|---|---|---|---|---|---|---|
| DASHUSDT | 11-05 15:50 | 0.9838 | 11-05 15:55 | 11-05 17:55 | 43.7837 | 44.1059 | +0.53% | +10.69 | closed |
| XPLUSDT | 11-05 15:50 | 0.9836 | 11-05 15:55 | 11-05 17:55 | 0.0586286 | 0.0580819 | -1.13% | -22.61 | closed |
| ONDOUSDT | 11-05 15:50 | 0.9835 | 11-05 15:55 | 11-05 17:55 | 0.36997 | 0.371129 | +0.11% | +2.26 | closed |
| PUMPUSDT | 11-05 15:50 | 0.9835 | 11-05 15:55 | 11-05 17:55 | 0.00391691 | 0.00392108 | -0.09% | -1.88 | closed |
| UNIUSDT | 11-05 15:50 | 0.9835 | 11-05 15:55 | 11-05 17:55 | 6.71771 | 6.86513 | +1.99% | +39.81 | closed |

Signals not taken: slots_full 23

## Trades: S5:path_all:a_1h (1h)

| Symbol | Signal | pct | Entry | Exit | Entry px | Exit px | Net | $ | Status |
|---|---|---|---|---|---|---|---|---|---|
| DASHUSDT | 11-05 15:50 | 0.9838 | 11-05 15:55 | 11-05 16:55 | 43.7837 | 44.4155 | +1.24% | +24.81 | closed |
| XPLUSDT | 11-05 15:50 | 0.9836 | 11-05 15:55 | 11-05 16:55 | 0.0586286 | 0.0592507 | +0.86% | +17.18 | closed |
| ONDOUSDT | 11-05 15:50 | 0.9835 | 11-05 15:55 | 11-05 16:55 | 0.36997 | 0.374825 | +1.11% | +22.20 | closed |
| PUMPUSDT | 11-05 15:50 | 0.9835 | 11-05 15:55 | 11-05 16:55 | 0.00391691 | 0.00394805 | +0.59% | +11.87 | closed |
| UNIUSDT | 11-05 15:50 | 0.9835 | 11-05 15:55 | 11-05 16:55 | 6.71771 | 6.84315 | +1.66% | +33.28 | closed |

Signals not taken: slots_full 23

## Equity at 00:00 UTC (mark to market)

| Time | 2h hold $ | 1h hold $ |
|---|---|---|
| 2026-11-03 | 10,000.00 | 10,000.00 |
| 2026-11-04 | 10,000.00 | 10,000.00 |
| 2026-11-05 | 10,000.00 | 10,000.00 |
| 2026-11-06 | 10,028.27 | 10,109.33 |
| 2026-11-07 | 10,028.27 | 10,109.33 |
| 2026-11-08 | 10,028.27 | 10,109.33 |
| 2026-11-09 | 10,028.27 | 10,109.33 |
| 2026-11-09 00:35 (latest) | 10,028.27 | 10,109.33 |

## Run

Daily: `uv run --group research python research/volume_spike/paper.py update`.

- sync: 113,781 candles added for 391 symbols
- spikes scored this run: 429 (of 2155 in the window)
- timings: sync 13s, spikes + inputs + scores 3s, books 0s, total 16s
