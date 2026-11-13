# MA rebound: can a CNN pick the good pullbacks to major moving averages?

Unattended study, 2026-11-13. Thesis under test: on Binance spot, a price that comes down to a
major simple moving average often rebounds, and a CNN reading the candles before the touch can
pick the good setups. Plan, protocol and run log: `docs/plans/ma-rebound.md`. Code:
`research/ma_rebound/`. Stage notes: `research/ma_rebound/results/`
([M1](../../research/ma_rebound/results/M1.md),
[M2](../../research/ma_rebound/results/M2.md),
[M3](../../research/ma_rebound/results/M3.md),
[M4](../../research/ma_rebound/results/M4.md),
[M5](../../research/ma_rebound/results/M5.md),
[M6 pick](../../research/ma_rebound/results/M6_pick.md),
[M6 reveal](../../research/ma_rebound/results/M6.md)).

## Summary

1. **The thesis was not confirmed.** No variant met the pre-registered conditions on 2022-24, so
   there was no pick, and the one look at the 2025-26 holdout is description only. On that
   holdout the protocol's finalist (the base CNN, 5 seeds) lost **-36.1% in 2025 and -15.1% in
   2026** (to 2026-11-02), at -23 bp per trade (day-bootstrap 95% CI -65 to +17, 1,996 trades).
2. **Major moving averages are not special.** Setups on the 50/100/200 SMAs earn the same as
   setups on odd lengths (37, 73, 137, 163): the difference is +1 bp per trade on 2020-24 (CI -4
   to +6) and -4 bp on 2025-26 (CI -10 to +3).
3. **Buying every touch does not pay.** Net of costs on 2020-24 it made about 0 bp per trade on
   1h MAs, +49 bp on 4h (CI includes zero) and +366 bp on 1d (CI +133 to +597), with every
   timeframe losing in 2022. On 2025-26 every timeframe was negative, the daily MAs at -167 bp.
   As a 10-slot book it lost every year: -91% / -30% / -33% in 2022-24, -87% / -73% in 2025-26.
4. **The CNN does select, but not by enough.** On 2022-24 it beat a shuffled-score placebo and
   its trades earned more than every candidate's; on 2025-26 its trades still earned 22 bp more
   per trade than the no-model book and its book lost about half as much. A long-only book of
   these setups still lost in 2022, 2025 and 2026.
5. **A gradient-boosted model on hand-made features did not beat its placebo.** What skill there
   is comes from the raw candle windows.
6. **Seed noise is larger than any difference the screen found.** The base CNN's five seeds
   range from -10.6% to +32.7% average yearly return on 2022-24. Seed-averaged scores ranked far
   above their own seeds (breadth model: +97.7% against a seed average of +23.8%), and those
   rows lost 34% to 64% a year on the holdout.
7. **Stops and take profits hurt; a tighter entry is better.** A take profit with a stop under
   the MA was worse than a plain time exit on every timeframe. Buying at the MA beat buying
   further above it; about half of the limit orders never fill.
8. **The only rows positive in both holdout years are daily-MA-only rules** on two of the
   comparison models, with about 60 trades a year. They fail the protocol's trade-count floor,
   were not chosen before the reveal, and one rests on five trades. They are a lead to test on
   new data, not a result.

## Method

- **Data:** 5m and 1m candles of the "all" universe (582 USDT pairs, delisted coins included),
  2020-01-01 to 2026-11-02, plus daily candles before 2020 for the 82 coins already listed, used
  only to complete the long averages. A coin is eligible while its mean daily quote volume over
  the previous 30 days is at least $1M.
- **Moving averages:** SMA 50/100/200 on 1h, 4h and 1d; 20/50/100/200 on 1w; 10/20 on 1M. The
  level `L` is the mean of the last n completed bars' closes, so it is fixed within a bar.
  Placebo MAs: lengths 37, 73, 137, 163.
- **Scale:** each timeframe has a hold limit `T` (1h 8 hours, 4h 32 hours, 1d 8 days, 1w 21
  days, 1M 28 days) and a volatility unit `u = ATR(14 days) x sqrt(T in days)`.
- **Setup (candidate):** the last k bars stayed above `L`; a 5m close at or above `L(1 + 0.45u)`
  arms it; the first 5m close at or below `L(1 + 0.3u)` triggers it. A model sees nothing after
  that close. 2.6M candidates were built (683k limit and 419k touch-market on the study MAs).
- **Entry:** a limit buy at `L(1 + b u)` (b = 0, 0.05 primary, 0.15), re-priced as bars complete,
  valid for `T/2`, filled on 1m candles at the limit price. Comparison: a market buy after the
  first touch.
- **Exits** (the production `resolve_exit`): E1 time exit at `T` (base); E2 at `T/2`; E3 take
  profit `+1u` with a stop `0.5u` below the MA; E4 `+0.5u` with a `0.25u` stop; E5 stop only; E6
  time exit at `2T`.
- **Costs:** fee 0.1% per side, plus 0.1% slippage on market fills. Stress: fills only when the
  price trades 5 bp through the limit, plus 0.1% a side.
- **Book:** $10,000, 10 equal slots of realised equity, one position per coin, a slot taken at
  the fill, one continuous book from 2022-01-01. It reproduces the production engine bit for bit
  on market-entry trades (14,413 trades over 386 coins).
- **Model:** a two-tower 1D CNN (140k parameters) on the last 96 5m candles and the last 128 bars
  of the MA's timeframe, prices as distance from the level in units of u, with BTC's windows and
  a few scalars. Target: the E1 return over u. Walk-forward folds 2022-26: train before Y-1 July,
  early-stop on Y-1 H2, score year Y; a score is its percentile among the fold's training
  scores. Base rule: trade percentile >= 0.80.
- **Pre-registered** (in the plan, pushed before any result):
  - *Qualifies* on 2022-24: at least 100 trades, bp per trade > 0 and book return > 0 in every
    year; bp > 0 under the stress; bp above the placebo's 95th percentile.
  - *Rank*: the mean of the three yearly book returns.
  - *Beats the baseline*: qualifies, is not fragile, and outranks the base CNN by more than its
    seed spread.
  - *Pick*: from 5-seed finalists, committed and pushed before 2025-26 is opened once.
- **Checks:** look-ahead checks by truncation and by an independent direct loop over the raw
  candles (76 checks), input truncation and fold-boundary checks (21), book checks (35).
- **Trials:** 279 ledger rows: 16 no-model benchmarks, 54 scored variants (2 in M4, 27 in M5, 25
  in M6) and their per-seed rows. GPU time: about 5 hours.

## Results

bp = mean net return per trade x 10,000. Book returns are calendar-year returns of the 10-slot
book. 2026 runs to 2026-11-02.

### Without a model: every setup, per trade (limit 0.05u, exit E1)

| MA timeframe | 2020-24 bp (95% CI) | Trades | 2025-26 bp (95% CI) | Trades |
|---|---|---|---|---|
| 1h | -0.4 (-14, +13) | 183,681 | -44 (-66, -23) | 75,071 |
| 4h | +49 (-9, +106) | 47,820 | -70 (-144, +1) | 18,401 |
| 1d | +366 (+133, +597) | 9,189 | -167 (-345, +24) | 3,125 |
| 1w | +197 (-149, +565) | 2,984 | -678 (-1159, -169) | 741 |
| 1M | +237 (-352, +731) | 915 | -288 (-950, +356) | 133 |

- By year, all timeframes pooled: 2020 +45, 2021 +105, 2022 -64, 2023 +31, 2024 +32, 2025 -70,
  2026 -34 bp. The daily MAs made +802 bp in 2021 and -570 in 2022 (25% winners).
- Study MAs minus placebo MAs: +1.0 bp (CI -4 to +6) on 2020-24; -4 bp (CI -10 to +3) on 2025-26.
- Setups beat matched random pullbacks gross on 1h (+27 bp) and 4h (+114 bp) in 2020-24, but the
  controls lack the trend condition, so this does not isolate the MA.

### Base rule on 2022-24 (the selection period)

| | bp per trade 2022 / 2023 / 2024 | Book return 2022 / 2023 / 2024 | Stress bp | Qualifies |
|---|---|---|---|---|
| Every candidate | -50 / -5 / -2 | -91% / -30% / -33% | -41 | no |
| Gradient-boosted benchmark | -74 / +15 / +28 | -74% / +28% / +29% | -34 | no |
| Base CNN, 3 seeds | -52 / +27 / +72 | -61% / +25% / +92% | -11 | no |
| Base CNN, 5 seeds (the finalist) | -61 / +38 / +66 | -65% / +49% / +79% | -9 | no |

- The 3-seed CNN's mean bp beats its placebo's 95th percentile (+15.7 against -4.7); the benchmark's
  does not. Score deciles are monotonic; the validation rank correlation is 0.03 to 0.18.
- Its gains come from 4h and 1d MAs (+64 and +119 bp); 1h trades are most of the book at +2 bp.

### The screen (27 variants, 2022-24)

- **None qualifies; none beats the baseline.** 26 of 27 lose in 2022 (-30% to -82% as a book).
- The exception, daily MAs only, is positive every year (+4% / +7% / +48%) and not fragile, but
  has 29 / 87 / 67 trades a year and a CI of -73 to +664 bp.
- Market breadth, distances to the other MAs and short windows rank +66% to +81% on the
  seed-mean score, but their seeds average +23% to +36% against the base's +18%.
- Not helping: no BTC inputs, a take-profit-probability target, one model per timeframe group,
  thresholds below 0.8, exits E2 / E4 / E5 / E6, 5 slots, weekly + monthly MAs alone.

### The holdout, 2025-26 (opened once, after "there is no pick" was pushed)

| Row | 2025 | 2026 | Pooled bp (95% CI) | Trades |
|---|---|---|---|---|
| Every candidate, base rule | -86.7% | -73.1% | -45 (-66, -22) | 6,718 |
| **Base CNN, 5 seeds (the finalist)** | **-36.1%** | **-15.1%** | **-23 (-65, +17)** | 1,996 |
| Breadth model, base rule (top base-rule row on 2022-24) | -63.7% | -34.3% | -92 (-147, -36) | 1,454 |
| Other-MA model, percentile >= 0.9 (top of 25 rows on 2022-24) | -20.0% | -14.2% | -15 (-124, +132) | 1,013 |
| Gradient-boosted benchmark | +6.1% | -45.8% | -8 | 3,195 |

- All five seeds of the finalist lost in both years (pooled -53% to -82%).
- Of the 54 scored rows, 5 had a positive book in 2025, 6 in 2026 and 2 in both. None of the 16
  no-model books was positive in either year.
- The two rows positive in both years are 1d-only rules: other-MA model +38.5% / +16.2% (113
  trades; 5 trades carry 67% of the P&L) and short-window model +10.6% / +10.6% (120 trades).
- Every scored row lost less than its no-model comparator (54 of 54), but only 33 of 54 had a
  higher bp per trade: the comparators had so little capital left that they could lose little.
- Spearman correlation between the 2022-24 rank and the 2025-26 return: +0.43 over all 54 rows,
  +0.50 over the M5 rows, -0.13 over the 25 M6 rows.
- Deflated Sharpe ratio over 54 trials (expected maximum without edge +1.23): the finalist 0.04
  (2022-24 Sharpe +0.23, 2025-26 Sharpe -0.45); the best row, a 1d-only rule, 0.80.

## Caveats

- **No pick, nothing confirmed.** Every 2025-26 number is descriptive. Choosing the rows that did
  well there would turn the holdout into a selection period.
- **Long only in falling markets.** The loss years are 2022, 2025 and 2026. The study has no
  market-neutral benchmark beyond a per-trade comparison with BTC (the finalist's trades beat BTC
  over the same windows by +51 bp on 2022-24 and +18 bp on 2025-26), so part of every result is
  market direction.
- **Book results depend on capacity.** Setups cluster on market-wide drops, and the frequent 1h
  setups fill the slots; a coin held on one MA blocks its other MAs. `EqualSlots` also skips a
  fill when cash is short of one stake, as the production engine does.
- **Seed means are not seeds.** A percentile of the mean raw score selects differently from any
  single seed; the high 2022-24 ranks of the comparison rows came largely from this.
- **Percentiles are against filled training rows**, so "percentile >= 0.80" selects 10% to 16%
  of candidates.
- **Trading halts are not handled.** 13 coins have a halt of a day or more (token swaps, the LUNA
  halt). 1,346 of 2.6M candidates have level or volatility windows spanning one, and 6 trades
  before 2025 were held across one. None occurs in 2025-26. Not corrected: far below seed noise.
- **Fills are simulated.** Limit fills need the 1m low to reach the limit; the stress run needs
  it to trade 5 bp through. Queue position and partial fills are not modelled.
- **One disclosure.** During M4 a worker's terminal once showed fold 2026's training and
  validation row counts, which count 2025 fills. No 2025-26 outcome was seen before the reveal.
- **2026 is a partial year** (to 2026-11-02).

## Recommendations

- **Do not trade this.** Neither the premise (major MAs give tradeable rebounds) nor the selection
  thesis (a CNN can make them pay) held up under a protocol fixed in advance.
- **Do not port it to the engine.** The follow-up to add limit entries to the simulated broker
  and backtest engine has no strategy to justify it from this study.
- **If the idea is pursued, test these on a fresh protocol and forward data, not on 2020-26 again:**
  - daily MAs only, where the per-trade edge was largest before 2025 and the only two rows
    positive in both holdout years sit (but note the daily baseline itself was -167 bp there);
  - a market-regime gate or market-neutral sizing, since every loss year is a falling market;
  - more seeds per model (the seed spread, 20 to 50 points, exceeds every effect measured here).
- **Reusable from this study:** the event builder with its look-ahead checks, the limit-entry
  book that matches the production engine, and the scorecard with sealing and a ledger.
