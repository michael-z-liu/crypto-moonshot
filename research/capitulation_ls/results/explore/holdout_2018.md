# Holdout 2017-08 .. 2019-12: the long sleeve L on years nobody tuned on

Every earlier study used 2020-01 .. 2026-11. This one runs the pre-registered rule L, unchanged, on
Binance spot data from 2017-08 to 2019-12. No study, model or threshold had seen those years.

**The pre-registered test** (row marked TEST below): a spot long when the day5 CNN scores a
liquid 5m volume spike at pct >= 0.995. Entry at the next 5m open, take profit +20%, time exit
72h, 0.4% round-trip cost, one position per coin. Book: 10 equal slots, a fresh $10k each year,
and a stake of at most 0.5% of the coin's 24h quote volume.

Scripts: `h1_fetch.py` (archive download), `h1_build.py` (spikes, CNN inputs, outcomes),
`h1_fold2022.py` (saves day5's oldest fold), `h1_score.py` (scoring and parity checks),
`h1_eval.py` (trades, book, diagnostics, neighbours, reference). Data and tables are in
`$DATA/cache/capitulation_ls/holdout/` (`eval_tables.md` is the full output).

## Verdict

**L's edge holds out of sample, at about three quarters of its 2022-26 size per trade.
The number of opportunities is much smaller.**

- 41 trades on 18 flush days. Mean **+954 bp** a trade (median +813), 88% winners, 37% hit the
  take profit, none lost more than 10%. Each year is positive: 2017 +1409 bp (7 trades),
  2018 +1471 bp (10), 2019 +606 bp (24).
- Clustered by day: 16 of the 18 flush days are positive and the mean of the day means is
  +1060 bp. A day-bootstrap 95% CI of the trade mean is [+620, +1379] bp, and no resample is
  <= 0. The 2022-26 reference: +1291 bp a trade, CI [+650, +1692] bp, on 25 days.
- 10-slot book: **2017 (Aug-Dec) +10.2%, 2018 +15.6%, 2019 +13.0%**, drawdowns about 1%. The
  book's per-trade mean is +952 bp, against +1064 bp in 2022-26. The book makes less than in
  2022-26 (+15% to +93%) because there are fewer events, not because trades are worse.
- The calibration has the same **hockey stick**. Outcome is about 0 below pct 0.976 and turns up
  only in the extreme tail: -19 / +56 / +126 / +511 / +1072 bp by fixed bin, against
  -34 / +86 / +355 / +742 / +1348 bp in 2022-26.
- The CNN adds value over a naive rule. Buying any liquid spike after a >= 20% 24h fall, with
  L's exit, earns +566 bp in 2017-19 (+623 in 2022-26). Buying every liquid spike earns -37 bp
  (-71).

The weaker part is the take-profit rate: 0.37 against 0.65. In 2019 it is only 0.17, and the
median best move in 2019 trades is +12.6% against about +20% in 2017-18. Trades still win
(time exits average +374 bp), but the bounces are smaller (see Why).

## Data coverage

- **Source:** data.binance.vision monthly 5m zips (sha256-checked), 2017-08 .. 2020-01. 2020-01
  is fetched only so late-2019 trades have their 72h path.
- **Pairs:** 89 USDT pairs with months in range. Delisted pairs are included (BCC, VEN, BCHABC,
  BCHSV, HC, ERD, NPXS, MCO, ...). Leveraged tokens, pegged assets and non-ASCII tickers are left
  out, as in the "all" universe.
- **Fetched:** 894 symbol-months, no failures left. The archive's 2017-18 timestamps are in
  milliseconds; the parser detects the unit.
- **Outage fix:** after the 2018-02-08/09 exchange outage, the archive's candles are offset by
  3m14.789s for about 20 hours (241 per pair). They are floored to the 5m grid, with no
  collisions.
- **Excluded spikes:** 15 spikes right after that outage (2018-02-09) have BTC/ETH volume
  baselines of 0, which makes those input channels non-finite. They are excluded: a NaN score
  would otherwise sort to pct 1.0.

| year | liquid spikes | coins | median qv_base (USDT / 5m) | first | last |
|---|---|---|---|---|---|
| 2017 | 438 | 6 | 13,610 | 2017-08-25 | 2017-12-31 |
| 2018 | 8,002 | 21 | 22,492 | 2018-01-01 | 2018-12-31 |
| 2019 | 27,477 | 58 | 15,082 | 2019-01-01 | 2019-12-31 |

For scale, 2022-26 has about 270k liquid spikes a year on 300-600 coins.

**Same definitions.** `h1_build.py` calls the builders' own functions:
- spikes, liquidity and warm-up: `build_events.features` and `full_grid`, with
  `process`'s filters, then `build_seq2`'s qv_base >= 5,000;
- inputs: `build_seq2.coin_job`, `grid` and `build_market`;
- outcomes: `fwd_job` and `build_fwd72.job`.

Only two things change: the global grid starts in 2017-07, and `common.load` reads the holdout
store.

**Parity check.** The 2020-01-08 .. 28 spikes built by this pipeline match the shared seq2:
- every shared spike is found, and the only holdout-only rows are the shared store's warm-up
  week, 2020-01-01 .. 07;
- qv_base, vol_mult and all 13 channels agree exactly, and so do the scores.

The day5 model, scored through the same `paper.Model` path as `forward_signals.py`, reproduces
the walk-forward day5 pct exactly on 20,000 shared 2026 rows.

## Scoring

The primary scorer is the saved day5 2026 fold, `.cache/forward_model/day5`: trained before
2025-07-01 and early-stopped on 2025 H2. As in `forward_signals.py`, raw is the seed-mean 1h
median forecast. pct = searchsorted(the fold's own sorted reference scores, raw) / n, where the
reference is about 50k of that fold's 2020-01 .. 2025-06 training rows. So holdout pct is
measured against the 2020-25 spike distribution: the 2026 fold applied backwards.

The second scorer is day5's oldest fold, 2022: trained before 2021-07-01 and early-stopped on
2021 H2. `h1_fold2022.py` retrained it and saved it to `.cache/forward_model/day5_2022`. It
reproduces day5's walk-forward 2022 pct on all 296,060 rows (100% equal). Its reference is the
2020-01 .. 2021-06 training rows.

Share of liquid spikes at or above each pct (day5 2026 fold):

| year | >= 0.95 | >= 0.99 | >= 0.995 | >= 0.998 |
|---|---|---|---|---|
| 2017 | 0.224 | 0.101 | 0.050 | 0.018 |
| 2018 | 0.080 | 0.010 | 0.0047 | 0.0021 |
| 2019 | 0.048 | 0.0052 | 0.0017 | 0.0002 |
| 2022-26 (reference) | 0.044 | - | 0.0025 | - |

## L trades by year (day5 2026 fold, every candidate, one position per coin)

| year | n | mean net (bp) | median (bp) | TP rate | win | days | coins |
|---|---|---|---|---|---|---|---|
| 2017 (Aug-Dec) | 7 | +1409 | +1960 | 0.71 | 0.86 | 5 | 4 |
| 2018 | 10 | +1471 | +1960 | 0.60 | 0.90 | 5 | 8 |
| 2019 | 24 | +606 | +342 | 0.17 | 0.88 | 8 | 17 |
| **2017-19** | **41** | **+954** | +813 | 0.37 | 0.88 | 18 | |
| 2022-26 reference | 1,883 | +1291 | | 0.65 | 0.85 | 25 | |

Exits: 15 take profits (+1960 bp each) and 26 time exits (mean +374 bp).

## 10-slot book ($10k fresh each year)

| year | trades | bp / trade | return | max DD | worst concurrent adverse |
|---|---|---|---|---|---|
| 2017 (Aug-Dec) | 7 | +1409 | **+10.2%** | -0.9% | 0.11 |
| 2018 | 10 | +1471 | **+15.6%** | -0.4% | 0.07 |
| 2019 | 22 | +570 | **+13.0%** | -1.2% | 0.04 |
| 2022-26 reference | 25 / 22 / 69 / 22 / 18 | +933 / +1636 / +1052 / +745 / +873 | +23.5% / +39.4% / +93.0% / +16.4% / +15.0% | -10% .. 0% | |

## diag scorecard

| strategy | n | E_o | median_o | win | tp_win | time_win | near_miss | never_went | went_against | stopped | tp_progress_med | stop_progress_med | giveback_med | worst_year_E_o | years_positive |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **L holdout 2017-19 (TEST)** | 41 | +0.095 | +0.081 | 0.88 | 0.37 | 0.51 | 0.05 | 0.07 | 0.00 | 0.00 | 0.78 | 0.04 | 0.40 | +0.061 | 1.00 |
| L 2022-26 (results/diag) | 1883 | +0.129 | +0.196 | 0.85 | 0.65 | 0.20 | 0.05 | 0.09 | 0.01 | 0.00 | 1.00 | 0.06 | 0.15 | +0.107 | 1.00 |

6.0% of the pct >= 0.95 pool reach the threshold. Spearman(signal, outcome) over the pool is
+0.153.

## Calibration by signal decile (pool pct >= 0.95; below-threshold candidates simulated as if entered)

| decile from pct | n | E_o | win | p_tp | tp_progress | entered |
|---|---|---|---|---|---|---|
| 0.9500 | 69 | -0.001 | 0.48 | 0.15 | 0.33 | 0.00 |
| 0.9518 | 68 | -0.001 | 0.44 | 0.12 | 0.30 | 0.00 |
| 0.9541 | 69 | +0.009 | 0.52 | 0.09 | 0.35 | 0.00 |
| 0.9569 | 68 | +0.010 | 0.49 | 0.12 | 0.32 | 0.00 |
| 0.9593 | 68 | -0.000 | 0.41 | 0.10 | 0.30 | 0.00 |
| 0.9626 | 69 | +0.007 | 0.44 | 0.12 | 0.35 | 0.00 |
| 0.9666 | 68 | -0.010 | 0.43 | 0.06 | 0.39 | 0.00 |
| 0.9709 | 69 | +0.018 | 0.54 | 0.20 | 0.40 | 0.00 |
| 0.9772 | 68 | +0.019 | 0.53 | 0.22 | 0.44 | 0.00 |
| 0.9855 | 68 | **+0.085** | 0.85 | 0.32 | 0.74 | 0.60 |

**The hockey stick is present.** The outcome is flat at about 0 for nine deciles and jumps only
in the top one. That is the same shape as 2022-26, where the top decile is +0.132 and the next
+0.059.

Per-spike outcome by fixed pct bin (L's exit, no per-coin lock):

| pct bin | 2017-19 n | 2017-19 mean (bp) | 2017-19 TP rate | 2022-26 n | 2022-26 mean (bp) | 2022-26 TP rate |
|---|---|---|---|---|---|---|
| 0.90-0.95 | 3,388 | -19 | 0.09 | 77,637 | -34 | 0.11 |
| 0.95-0.976 | 1,269 | +56 | 0.10 | 33,510 | +86 | 0.16 |
| 0.976-0.984 | 316 | +126 | 0.12 | 10,001 | +355 | 0.23 |
| 0.984-0.9955 | 367 | +511 | 0.23 | 16,797 | +742 | 0.41 |
| >= 0.9955 | 81 | +1072 | 0.48 | 3,700 | +1348 | 0.73 |

## Neighbours (honest read; only the TEST row is the test)

| rule | n | mean (bp) | TP rate | days | book 2017 | book 2018 | book 2019 | 2022-26 n | 2022-26 bp | 2022-26 days |
|---|---|---|---|---|---|---|---|---|---|---|
| pct>=0.99 TP15 | 72 | +700 | 0.51 | 23 | +1.4% | +25.8% | +26.1% | 3060 | +880 | 67 |
| pct>=0.99 TP20 | 72 | +672 | 0.29 | 23 | +2.9% | +21.0% | +25.8% | 2934 | +996 | 67 |
| pct>=0.99 TP25 | 72 | +679 | 0.18 | 23 | +2.1% | +24.0% | +24.9% | 2875 | +1019 | 67 |
| pct>=0.995 TP15 | 44 | +891 | 0.57 | 18 | +12.3% | +13.4% | +13.7% | 1928 | +1051 | 26 |
| **pct>=0.995 TP20 (TEST)** | **41** | **+954** | 0.37 | 18 | **+10.2%** | **+15.6%** | **+13.0%** | 1883 | +1291 | 25 |
| pct>=0.995 TP25 | 41 | +960 | 0.22 | 18 | +8.3% | +17.8% | +13.8% | 1852 | +1393 | 25 |
| pct>=0.998 TP15 | 11 | +1380 | 0.91 | 3 | +7.4% | +1.5% | +6.4% | 741 | +1388 | 14 |
| pct>=0.998 TP20 | 10 | +1781 | 0.80 | 3 | +7.8% | +2.0% | +8.0% | 706 | +1800 | 14 |
| pct>=0.998 TP25 | 10 | +1769 | 0.50 | 3 | +8.0% | +2.5% | +7.2% | 665 | +2138 | 14 |

Every neighbour is positive, and the order matches 2022-26: a higher threshold gives better
trades but fewer of them. The take profit changes little because few trades reach +20%. In 2018
and 2019, pct 0.99 makes more money in the book (+21..26%) than 0.995, because it trades more
days. This was not pre-registered and should not be read as a better setting.

**The day5 2022 fold as scorer** (the reference is 2020-21 spikes only, so it is stricter on
2017-19):
- pct >= 0.995: 4 trades, all at the take profit, +1960 bp (2017-12-22 LTC, 2018-01-16 BNB,
  2019-09-24 EOS, 2019-12-10 MATIC);
- pct >= 0.99: 21 trades, +933 bp;
- calibration: flat to the 0.97 decile, then +0.09 / +0.08.

Same conclusion: the signal transfers.

## Flush days (day5 2026 fold, pct >= 0.995)

| day | trades | coins | mean (bp) | TPs | coin 24h ret (median) | BTC 24h ret | max pct |
|---|---|---|---|---|---|---|---|
| 2017-11-29 | 1 | NEO | +984 | 0 | -15.3% | -12.3% | 0.9964 |
| 2017-12-18 | 1 | BNB | +1960 | 1 | +1.9% | -9.9% | 0.9964 |
| 2017-12-20 | 1 | NEO | +1960 | 1 | -27.5% | -20.4% | 0.9951 |
| 2017-12-21 | 1 | BCC | -923 | 0 | -21.7% | -11.7% | 0.9968 |
| 2017-12-22 | 3 | BNB, LTC, NEO | +1960 | 3 | -20.0% | -20.5% | 0.9984 |
| 2018-01-08 | 1 | BNB | +1960 | 1 | -28.0% | -20.0% | 0.9959 |
| 2018-01-11 | 2 | BTC, NEO | +1722 | 1 | -15.9% | -12.9% | 0.9972 |
| 2018-01-16 | 3 | BCC, BNB, ETH | +756 | 1 | -26.7% | -15.3% | 0.9989 |
| 2018-08-14 | 3 | ADA, BNB, ICX | +1692 | 2 | -26.7% | -4.8% | 0.9958 |
| 2018-11-20 | 1 | ONT | +1960 | 1 | -24.2% | -14.4% | 0.9970 |
| 2019-04-03 | 1 | EOS | +729 | 0 | -0.1% | +1.1% | 0.9957 |
| 2019-05-30 | 2 | BTT, LINK | +1146 | 1 | -18.1% | -5.2% | 0.9967 |
| 2019-06-04 | 1 | TRX | +353 | 0 | -17.0% | -9.9% | 0.9965 |
| 2019-06-26 | 1 | EOS | +100 | 0 | -12.6% | +6.7% | 0.9961 |
| 2019-07-11 | 3 | BCHABC, ONT, QTUM | +398 | 0 | -24.9% | -12.9% | 0.9971 |
| 2019-07-16 | 1 | EOS | +1960 | 1 | -22.6% | -9.7% | 0.9971 |
| 2019-09-24 | 12 | ADA, BCHABC, BNB, BTC, EOS, ETC, ETH, LINK, LTC, ONT, TRX, XLM | +758 | 2 | -25.9% | -11.0% | 0.9986 |
| 2019-12-10 | 3 | ERD, FET, ONE | -391 | 0 | -17.5% | -2.2% | 0.9975 |

The flush days are the known capitulations of the era: the December 2017 top, January 2018,
the August 2018 alt flush, the November 2018 BCH-fork crash, July 2019 and the 2019-09-24 BTC
breakdown. The same rule picks them out without any tuning.

## Why the holdout is weaker (+954 vs +1291 bp, TP rate 0.37 vs 0.65) and the book smaller

1. **Smaller bounces in 2019, not a broken signal.** In 2019 the median best move during the
   hold is +12.6% (+20.4% in 2017, +20.6% in 2018). Only 17% of 2019 trades reach +20%, but 88%
   still win, through time exits. With TP 15% the 2019 book barely changes (+13.7% against +13.0%). The 2019 flushes
   were mostly majors: the median trade traded about $16M of quote volume in the past 24h (about
   $6M in 2017 and $10M in 2018). Majors bounce less than the small caps that dominate the
   2022-26 trades (334 coins in 2024 alone).
2. **Few coins, so few flush days and few slots used.**
   - Coins on USDT: 6 (2017), 21 (2018), 58 (2019), against hundreds in 2022-26. Most altcoin
     trading was BTC-quoted then, which this USDT-only study cannot see.
   - L traded 7 / 10 / 24 times a year on 5 / 5 / 8 days, against 25 to 69 book trades a year
     on 25 days in 2022-26.
   - The book is in cash between flushes, and its drawdowns are about 1%. Its return is capped by event count (+10..16% a year), not by trade quality
     (+952 bp a trade in the book, against +1064 bp).
3. **The pct scale is shifted by era.**
   - 2017 is unusually volatile against the 2020-25 reference: 5% of its spikes score >= 0.995,
     20x the base rate. Even so, its 7 trades average +1409 bp.
   - 2019 is calm (0.17%), so the 0.995 cut is effectively stricter in 2019.
   - The 2022 fold, with a narrower 2020-21 reference, is stricter still: 4 trades.
   - A percentile against a fixed training reference does not adapt to regime. The threshold
     held anyway, but the event count is regime-dependent.
4. **Failure modes are the same as 2022-26.** No trade is stopped; `never_went` is 7% (9%) and
   `near_miss` 5% (5%). The worst trades are 2017-12-21 BCC (-923 bp, the day after its
   December top) and 2019-12-10 ERD / FET / ONE (-391 bp, small new listings in an
   idiosyncratic sell-off with BTC calm, -2%). The higher giveback (0.40 against 0.15) is the
   time-exit side of point 1.
5. **Regime is not the explanation.**
   - L is positive in a +222% year (2017), a -73% bear (2018) and a +95% recovery (2019).
   - 2018 is in fact the best year: +1471 bp, TP rate 0.60.
   - That matches 2022-26, where 2022 (-64%) gave +1175 bp.

## Caveats

- **Small sample.** 41 trades on 18 days, and 12 of the trades fall on one day (2019-09-24).
  The day-clustered CI excludes 0, but the mean is uncertain to about +/-370 bp.
- **Archive survivorship.** data.binance.vision keeps delisted pairs, and every USDT pair the
  shared universe had trading in 2020-01 is present (parity). Pairs removed before the archive
  existed would be missing, but the pair list matches the known USDT market of the time. Some
  survivorship cannot be ruled out.
- **USDT pairs only.** In 2017-18 most alts traded against BTC, so the holdout universe is
  mostly large caps. It is a different, harder slice than 2022-26.
- **No perps.** USD-M perps barely existed (BTC from 2019-09), so this tests the spot long
  sleeve only. Nothing here speaks to the short sleeves.
- **Models.** The 2026 fold applied backwards is the primary scorer. The 2022 fold, the oldest
  one, is reported as a check. Neither model saw 2017-19, and nothing was tuned on it.
- **Costs and fills.** 0.4% round trip, as in the 2022-26 work. 2017-18 spreads on these pairs
  were probably wider, and 5m candles hide intrabar slippage at the flush.
- **Excluded spikes.** The 15 spikes right after the 2018-02 outage are excluded (non-finite
  inputs).
