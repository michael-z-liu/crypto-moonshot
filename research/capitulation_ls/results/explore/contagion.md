# Contagion shorts (and the reverse): peers of a lone crash

Scripts: `x4_build.py` (events, correlations, 72h perp paths), `x4_lib.py` (selection, exits, 10-slot
book), `x4_grid.py` (grid), `x4_final.py` (trade files, diag). Data: `$OWN/explore/contagion_*.parquet`.
Everything below is in-sample; no out-of-sample or walk-forward cut was used.

## Hypothesis

When a coin crashes idiosyncratically, the coins most related to it fall next, so shorting the top-k
related peers (that have not fallen yet) on perps pays. Also tested: the reverse (long the peers,
they bounce / are oversold by sympathy) and a market-neutral version (50% short peer, 50% long BTC perp).

## Method

- Events: `feats.parquet`, ret_4h <= -15%, |BTC 4h| <= 1%, perp active, 2021+, first per symbol per 24h:
  646 events (2021: 28, 2022: 39, 2023: 51, 2024: 151, 2025: 202, 2026: 151); 622 had >= 40 eligible peers.
- Relatedness: trailing 30d (720 h, closed hours before the crash) hourly log-return correlation of every
  perp with the crashed coin, in two kinds: raw (kind 0) and BTC-residual (kind 1, same-ecosystem proxy).
  Eligible peers: >= 300 overlapping hours, perp 24h volume >= $1M, not BTC/ETH. Top 30 per kind per event stored.
- Signal 0..1 = sqrt(within-event correlation percentile x crash-size percentile across events).
- Entry at the open of the hour after the crash bar (perp), hold 4/12/24/48/72h, optional take profit 5/10%,
  isolated 1x short liquidation at +100% (or a 10% stop), cost 0.3% round trip, funding paid from the funding
  history (short receives positive funding), stop slippage 0.1%. Neutral version: costs 0.3% on the pair, 50/50,
  close-based path, no tp/stop.
- Book: per-year $10k, 10 equal slots, one position per coin, stake <= 0.5% of the peer's 24h volume,
  priority by signal.

## Configurations tried (all in-sample)

4,680 configs: mode (short / long / neutral) x kind (2) x top-k (3, 5, 10) x peer-not-fallen filter
(none, peer 4h return > -5%, > -2%) x signal threshold (0, .5, .7, .85) x hold (5 values) x tp (none, 5%, 10%)
x stop (none, 10%) (tp/stop only for short/long). Ranked by the mean of per-year average trades 2022-26
(n >= 100 candidates); the top 40 of each mode got a book.

| Mode | configs | share with avg trade > 0 | best avg trade (all candidates) | mean over configs |
| --- | --- | --- | --- | --- |
| short peers | 2,160 | 6.8% | +0.77% | -0.59% |
| long peers (bounce) | 2,160 | 40.9% | +1.67% | -0.02% |
| neutral (short peer + long BTC) | 360 | 0.3% | +0.01% | -0.40% |

Short results get worse with longer holds (24h avg -0.38%, 72h -0.77%),
larger k, the not-fallen filter and a higher signal threshold, i.e. the opposite of what the hypothesis
predicts. Long results are mildly better with larger k and the not-fallen filter, but average about zero.

## Best and baseline configs (10-slot per-year $10k books, in-sample)

Baseline = same mode/kind/hold/tp/stop but k = 10, no threshold, no filter, random priority.

| Config | book trades | avg trade | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | worst | max DD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| short best: resid, k3, thr .5, 72h, tp 10% | 1167 | +0.21% | -12% | +27% | +7% | -6% | -40% | +39% | -40% | -61% |
| short baseline | 2628 | -0.35% | -35% | +12% | -29% | -51% | +40% | -48% | -51% | -64% |
| long best: resid, k10, peer >-2%, thr .85, 48h, tp 10% | 1164 | +0.87% | +24% | +14% | 0% | +39% | -18% | +50% | -18% | -25% |
| long baseline | 3021 | -0.10% | +121% | -36% | -5% | 0% | -70% | +2% | -70% | -74% |
| neutral best: resid, k3, thr .85, 72h | 470 | +0.32% | -5% | +4% | +4% | -1% | 0% | +11% | -5% | -17% |
| neutral baseline | 2400 | -0.17% | -33% | +3% | -6% | -25% | +45% | -19% | -33% | -31% |

No-slot-limit average trade per year (all candidates of the config), %: short best 2021 -1.6, 2022 +3.6,
2023 +0.5, 2024 +1.9, 2025 -1.5, 2026 +1.1 (n = 1,403); long best +3.5, +2.4, +0.3, +2.7, +0.7, +0.9 (n = 1,454).
Targets (avg trade >= +10%, every year >= +50%) are missed by a factor of 10 or more for every config.

## Diag scorecard (`$OWN/explore/contagion_trades.parquet`, `x4_scorecards.md`)

| strategy | n | E_o | median | win | tp win | time win | near miss | never went | stopped | tp_prog med | giveback med | worst yr E_o | yrs > 0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| contagion_short | 1167 | +0.002 | +0.019 | 0.58 | 0.33 | 0.26 | 0.10 | 0.31 | 0.00 | 0.65 | 0.65 | -0.018 | 0.50 |
| contagion_long | 1164 | +0.009 | +0.007 | 0.54 | 0.24 | 0.30 | 0.07 | 0.38 | 0.00 | 0.49 | 0.83 | -0.003 | 0.83 |
| contagion_neutral | 470 | +0.003 | +0.004 | 0.56 | 0.00 | 0.56 | 0.00 | 0.44 | 0.00 | 0.12 | 0.69 | -0.022 | 0.67 |
| short baseline | 2628 | -0.003 | +0.008 | 0.54 | 0.28 | 0.27 | 0.10 | 0.35 | 0.00 | 0.57 | 0.80 | -0.018 | 0.33 |
| long baseline | 3021 | -0.001 | -0.001 | 0.49 | 0.23 | 0.27 | 0.08 | 0.43 | 0.00 | 0.46 | 0.98 | -0.014 | 0.50 |
| neutral baseline | 2400 | -0.002 | +0.003 | 0.54 | 0.00 | 0.54 | 0.00 | 0.45 | 0.00 | 0.10 | 0.67 | -0.019 | 0.33 |

Calibration (outcome by signal decile, entered + below-threshold candidates simulated as entered) is flat:
short E_o by decile runs -0.0002, +0.005, +0.010, -0.005, +0.010, -0.002, -0.010, -0.014, +0.001, -0.009
(Spearman signal vs outcome -0.051); long runs between -0.018 and +0.011 (Spearman +0.07); neutral is
negative in every decile (-0.017..-0.006, Spearman +0.064). threshold_reach: short thr 0.50, 73% of candidates
reach it; long/neutral thr 0.85, 23% reach it and about 19% fall within 0.1 below.

Signal lift (24h hold, no tp/stop, all top-30 candidates, avg net bp): short by signal quintile
[-45, -13, +41, -30, -48] (raw) and [-31, -25, +44, -29, -47] (residual): no monotonic trend, the top
quintile is the worst. By crash-size tercile (small to large) short [+17, -20, -54], long [-79, -40, -1]:
peers of bigger crashes bounce more. By correlation rank (0-4 / 5-14 / 15-29) short [0, -19, -25],
long [-61, -41, -32]: the closest peers hold up best, the more-related coins are not the ones that fall.

## What worked / did not

- Nothing meets either target. The best in-sample short book averages +0.2% a trade with -40% in 2025
  and -61% max drawdown; the best long book +0.9% a trade (worst -18%, max DD -25%), within noise of its
  own baseline given 1,400 correlated trades (about 600 events' worth of baskets, one event gives up to 10 trades).
- Contagion does not exist at the horizon tested: peers of a lone crash do not fall more than the average
  perp after costs. The mean short over all 2,160 configs is -0.59% and only 6.8% are positive; the
  neutral version loses to its 0.3% cost every time because the short-minus-BTC spread is about zero.
- The reverse (peers bounce) is not there either: 41% of long configs are positive and the mean is
  -0.02%. Mild tilt: larger k, not-fallen filter and a high threshold help longs slightly (+0.07%).
- The best configs are noise winners from a 4,680-config grid on a few hundred correlated events (28-39
  events in 2021-2022, 150-200 per year since 2024); the per-year book sign flips, and the best short's
  winning years differ from the baseline's.

## Why it fails

1. A lone crash is mostly coin-specific news (unlocks, exploits, delisting talk), and the 30-day
   correlation between alts is a market-beta artifact: raw and BTC-residual correlation rankings give the
   same answer, so there is no ecosystem signal the trailing return correlation can find.
2. Perps are priced efficiently enough that the repricing of related coins happens within the crash hour, before our
   next-hour entry; the remaining drift is below the 0.3% round trip plus funding.
3. Peers of larger crashes bounce rather than follow, the same behaviour that limits sleeve C (see
   `results/why/regime_other.md`).
4. Small-sample regime dependence: the per-year sign pattern (2025 -40% short, +40% for the baseline) shows
   the books ride market-wide days (several events on the same day) rather than the contagion signal.

## Caveats

- Everything is in-sample and the grid selects the best of 4,680 configs; deflate any positive result.
- Event dedupe per symbol is 24h; clustered events (same day) make trades highly dependent, so the
  effective sample is far below n.
- Peer paths are hourly: tp/stop use hourly highs/lows and favor the stop being hit at the first breaching hour.
- Neutral leg uses close-based paths, no liquidation check, equal dollar weights without beta adjustment.
- Peer shortability uses the perp volume gate (>= $1M 24h) and the 0.5% stake cap, not the order book.
- 2021 has only 28 events (few perps then).
- Threshold column of the trade file for k-limited configs = the lowest traded signal (the top-k cutoff is
  not a fixed threshold). Non-entries are the lower-ranked top-30 candidates of the same events.
