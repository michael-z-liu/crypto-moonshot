# Rebound factors: what tells a +30% move apart before it happens?

Unattended study, 2026-11-14. It follows the MA-rebound study
([report](ma-rebound.md)), whose frequency scan found that +30% moves within 30 days are common
on Binance spot but follow a random hour about as often as a touch of a major moving average.
Question here: which factors, known at entry, separate the entries that go on to such a move?
Plan, protocol and run log: `docs/plans/rebound-factors.md`. Code and stage results:
`research/rebound_factors/` and its `results/`
([M1 counts](../../research/rebound_factors/results/m1_counts.md),
[factors](../../research/rebound_factors/results/factors.md),
[M3 single factors](../../research/rebound_factors/results/m3_single.md),
[M4 model and spec](../../research/rebound_factors/results/m4_model.md),
[M5 reveal](../../research/rebound_factors/results/m5_confirm.md)).

## Summary

**Not confirmed.** Two factors found on 2020-23 (low liquidity and a deep 90-day drawdown) still
picked coins that hit +30% more often than random in each of 2024, 2025 and 2026, but the edge
shrank to about a third of its size and is too small to make the trade pay.

| | 2020-23 (found here) | 2024-26 (tested once) |
| --- | --- | --- |
| Base rate, +30% before -10% within 30 days | 25.8% | 20.0% |
| The rule's picks | 33% to 40% by year (walk-forward 2021-23) | 23.3% (random 95th percentile 21.3%) |
| Mean net return per trade, 10% stop | +3.4% to +5.8% by year | +0.2% (95% -1.6% to +2.4%) |
| The same with 0.3% extra cost a side | +4.2% pooled | -0.4% |
| Book, $10,000 in 10 slots, 10% stop | large gains (compounding) | -44.7% (random books: median -78.8%) |

- **"When" matters more than "which", and it could not be pinned down.** The strongest single
  factor is the market: on the best fifth of days by BTC's 30-day return the hit rate is 9.6
  points higher (95% +3.6 to +15.2). But there are only about 1,400 correlated days in 2020-23,
  only 4 of 20 market factors have a spread that excludes zero, and a market gate fitted on
  earlier years almost never opened in 2022.
- **"Which coin" effects are real but small.** Among coins on the same day, the lowest-liquidity
  tenth hit 5.7 points more often and the most beaten-down tenth about 3 points more. In 2024-26
  these were +1.2 and +1.6 points.
- **A model added nothing.** Gradient-boosted trees on all 65 factors scored 0.48 to 0.51 AUC
  walk-forward: they learn the market phase of their training years and carry it into the next.
- **The touch of a moving average is not among the top 20 coin factors**, as the earlier scan
  suggested.
- **The trade's arithmetic is demanding.** With a +30% target and a 10% stop the hit rate has to
  be about 26% to break even after costs. The base rate was above that in 2020, 2021 and nearly
  in 2023, so almost any selection looked profitable there; it was 16% to 24% in 2024-26.

## Method

- **Entries:** every coin at every UTC midnight while its 30-day mean daily volume is at least
  $1M: 592,347 coin-days on 576 coins, 2020-01-31 to 2026-10-03. Entries near a redenomination
  or token swap (nine found by a mechanical price-gap rule) are excluded: 1,934.
- **Outcome:** `hit10`, the high reaches +30% before the low reaches -10%, within 30 days, on 5m
  candles (stop first when one candle touches both). Also scored: a 20% stop and no stop, and the
  net result of the trade (market entry, limit exit at +30%, stop or 30-day exit, 0.1% fee and
  0.1% slippage on market fills).
- **Factors:** 65, all from candles closed before the entry, each checked by recomputing it from
  candles cut at the entry time. 20 market factors (BTC trend and volatility, breadth, dispersion)
  and 45 coin factors (momentum and position, volatility, volume and taker-buy flow, relative
  strength, size, age, and the MA-touch flag).
- **Split:** everything was explored on 2020-01-31 to 2023-12-01. Outcomes from 2024 on were
  written to a sealed folder that no script opened until the chosen rule was committed and
  pushed. The rule was then scored once.
- **Single factors (M3):** hit rate by decile (quintile of days for market factors), within the
  day for coin factors, per year, with intervals from a bootstrap over calendar months.
- **Combination (M4):** walk-forward on 2021, 2022 and 2023, training only on entries whose
  window had ended. Gradient-boosted trees (four feature sets, three seeds) against transparent
  two-factor rules, 18 pick rules, a 10-slot book, a stress run with 0.3% extra cost a side, and
  100 random books drawing the same number of picks on the same days.
- **Frozen rule:** score = mean within-day rank of low liquidity and deep 90-day drawdown; each
  day up to 10 entries in the top 5% of scores; no market gate; 10% stop.
- **Confirmed** was defined in advance as: the picks' `hit10` above the same-day random 95th
  percentile overall and in at least two of three years, and the 10%-stop book positive and above
  the random books' median.

## Results

### Single factors, 2020-23

| Question | Factor | Best group | Lift in `hit10` | 95% interval |
| --- | --- | --- | --- | --- |
| When | BTC 30-day return | top fifth of days | +9.6 pp | +3.6 to +15.2 |
| When | BTC against its 50-day SMA | top fifth | +9.6 pp | +4.5 to +14.7 |
| When | BTC 7-day return | top fifth | +7.6 pp | +2.6 to +13.9 |
| Which | Liquidity (30-day volume) | lowest tenth of the day | +5.7 pp | +4.2 to +7.4 |
| Which | Price against its 200-day SMA | lowest tenth | +3.4 pp | +1.8 to +5.1 |
| Which | 90-day return | lowest tenth | +3.3 pp | +2.0 to +4.9 |
| Which | Drawdown from the 90-day high | deepest tenth | +3.1 pp | +1.6 to +4.7 |
| Which | 30-day against 90-day volume | lowest tenth | +2.9 pp | +1.5 to +4.3 |

All of the top 20 coin factors also raise the no-stop hit rate, so they find rallies rather than
only avoid stop-outs. The 65 factors are far fewer ideas: one linked group alone has 15.

### Walk-forward, 2021-23

| Variant | AUC | `hit10` of the top 5% | Mean `ret_10` of the top 5% |
| --- | --- | --- | --- |
| Every entry (base) | 0.500 | 25.6% | +1.0% |
| Trees, all factors | 0.493 | 21.8% | -0.9% |
| Trees, coin factors | 0.501 | 24.7% | +0.3% |
| Trees, lean set of 12 | 0.508 | 27.1% | +1.0% |
| Rule: BTC gate + low liquidity + drawdown | 0.589 | 38.6% | +6.7% |
| Rule: low liquidity + drawdown, no gate (frozen) | 0.561 | 33.1% | +3.7% |

The gated rules could not reach 100 trades in 2022, so the ungated rule was the only variant
that met the choice rule.

### The reveal, 2024-26

| Year | Base `hit10` | Picks | Random 95th pct | Trades | Mean net return per trade | Book, 10% stop | Random book median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2024 | 24.1% | 26.8% | 25.6% | 412 | +1.2% | +16.2% | -17.2% |
| 2025 | 15.9% | 20.5% | 17.7% | 500 | +0.0% | -29.7% | -70.4% |
| 2026 | 20.1% | 22.6% | 21.9% | 339 | -0.2% | -22.6% | -20.1% |
| All | 20.0% | 23.3% | 21.3% | 1,240 | +0.2% | -44.7% | -78.8% |

| Clause of *Confirmed* | Result |
| --- | --- |
| Picks' `hit10` above the random 95th percentile overall | pass (23.3% against 21.3%) |
| ... and in at least two of three years | pass (3 of 3) |
| 10%-stop book positive | **fail** (-44.7%) |
| 10%-stop book above the random median | pass (-44.7% against -78.8%) |

Also observed, outside the verdict: with $1,000 a trade and no compounding the 10%-stop book
made $2,765 over the period at base costs and lost $4,669 under the stress, with a worst
peak-to-trough of $13,146. Without a stop the picks averaged +2.1% a trade (95% -2.8% to
+6.6%) against a random median of -0.8%, and the book ended flat (+0.1%).

## Caveats

- **The frozen rule was added after a first M4 run had been seen** (the gated rules had too few
  trades in 2022, and in-sample score percentiles were replaced by out-of-fold ones). Only that
  variant met the choice rule, and its factors came from M3 on the same years. The discovery
  figures are therefore optimistic, as the reveal shows.
- **Capacity is small.** The traded coins' median volume is about $1.3M a day, and the stress
  cost alone turns the result negative.
- **Compounded book returns are mechanical.** Hundreds of 30-day trades a year in 10 slots give
  very large or very small figures; the fixed-stake book is the fairer view.
- **Windows overlap.** Entries on neighbouring days share most of their 30-day windows, so the
  intervals come from a month bootstrap, and four discovery years hold few independent market
  phases.
- **Not a clean sheet for 2024-26.** Yearly base rates of that period had been seen in the earlier
  frequency scan, and one 2024 redenomination was seen when building the exclusion rule. No
  factor relationship in 2024-26 was looked at before the reveal. That period is now spent for
  these factors.

## Recommendations

- **Do not trade the rule.** Its edge over random is real but worth about 3 points of hit rate,
  less than the gap between a good and a bad market year.
- **Candle-derived coin factors look exhausted** at this target: 45 of them and a tree model gave
  at most a few points. A better "which coin" answer needs new information: futures funding and
  open interest, listings and other dated events.
- **The market phase is the lever, and it needs a different test.** Four years cannot establish
  a timing rule. A simple, pre-stated gate (for example BTC above its 50-day SMA) should be
  judged on forward data, alongside the exit: the 10% stop turned a hit-rate edge into a loss,
  and the no-stop variant did better in 2024-26, though not significantly.
- **Reusable from this study:** the coin-day panel with sealed labels, the factor table with its
  look-ahead checks, the redenomination filter, and the book with same-day random benchmarks.
