# Backtest validation against the research

The backtest engine (`backtest/`, `broker/simulated.py`, `portfolio.py`, `strategy/`) was checked
against four published research results on the shared data folder. The references and
tolerances were fixed in the plan (`docs/plans/backtest-engine.md`, M4) before any run. This page
gives each case's setup, the numbers, whether they pass, and a measured cause for every
difference. The tests are `tests/test_golden.py`.

## Summary

| # | Case | Reference | Tolerance | Result |
|---|---|---|---|---|
| 1 | F2, pool (PR #8) | bp per trade and trades, 2020..2026 | ±15 bp a year, trades within 5% | **passes**; largest gap 2022, -11.6 bp |
| 2 | R2 capitulation pick, all coins (cnn-replication B2) | $ P&L, 2020..2026 | ±5% or ±$150 a year | **passes**; largest gap 2020, -$60 |
| 3 | Case 2 with 1m fills and the `close` rule | total falls by $1,678 | ±25% | **fails as stated** (falls by $3,047); the like-for-like part, $1,575, passes. The other $1,472 is measured and explained (the engine re-runs the book) |
| 4 | Base CNN, walk-forward scores (PR #8) | bp per trade, 2022..2026 | ±20 bp a year | **passes**; largest gap 2024, -6.9 bp |

No engine bug was found. Every difference comes from one of these deliberate behaviours:

- **Cash check (a):** with equal slots, the engine skips an entry it cannot pay for, and the
  research did not check. This explains -11.4 bp of F2's 2022 and -7.1 bp of the CNN's 2024.
- **Look-ahead filter (c):** the research dropped signals whose next 4 hours crossed a data gap.
  This explains 2020 in cases 1 and 2.
- **Cost model (b):** the engine charges costs multiplicatively. This is worth up to about 1 bp
  a year, or $27 a year.
- **Book re-run (e):** case 3 re-runs the book instead of re-pricing a fixed trade list.

In every case, the engine fed the research's own signals with the cash check neutralised (the
**parity** runs) takes exactly the research book's trades, year by year. Each trade's return
then differs from the research's only by the cost model, to within 0.0002 bp. The exceptions are
3 R2 take profits that fill at a gap open above the target (+$0.89 in all).

## How to run

```
uv run pytest -m golden        # add -s for the per-year tables
```

- The golden tests are marked `golden` and deselected by default (`addopts = -m "not golden"`),
  so the normal check stays fast. `-m golden` on the command line wins.
- **Data:** they read `MOONSHOT_DATA_DIR` (default `/Users/mike/code/crypto-moonshot/data`) and
  need:
  - 5m candles for the `all` universe, plus `top200.txt`, `delisted.txt` and `outside_pool.txt`;
  - 1m candles (case 3);
  - the research caches `cache/volume_spike/{events.parquet, seq2_meta.parquet, wf/base.npz}`
    and `cache/volume_spike_all/events.parquet`.

  A test skips, naming the missing path, when any of these is absent. Nothing is written to the
  data folder: signal tables and the converted CNN score file go to the worktree's `.cache/`.
- **Runtime** (this Mac, 7 tests): about 35 s with cached signals, about 60 s from cold (the R2
  signals over 582 symbols take about 26 s, F2 over 152 about 6 s). Case 3 alone, with 1m fills
  for about 10,000 positions, takes 21 s. Peak memory is about 4.4 GB.

## Method

- **References:**
  - F2 comes from PR #8's `m6_test.py`. Its trade list is `research/volume_spike/results/
    trades_F2.csv`, which is the same blob on PR #8 and on `claude/cnn-replication`.
  - R2 comes from cnn-replication's `m18_longshot.py`. That script writes only the per-year
    table, so its book was rebuilt with the script's own `book` logic over the research caches
    (`cache/volume_spike_all/{events,long30}.parquet`). The rebuild reproduces the published
    dollars to the dollar: +845 / +8,651 / -1,082 / +2,967 / +7,584 / +2,479 / +1,337. It also
    gives the trade counts used by the parity test.
  - Case 3's reference is `m21_fillcheck.py`, whose trade CSV is
    `results/universe_all/m21_fillcheck_longshot_r2_tp20_3d.csv`.
  - The CNN comes from `m12_cnn_wf.py portfolio_years`, rerun the same way. It gives
    334.4 / 169.5 / 57.4 / 67.4 / 52.5 bp with 81 / 100 / 197 / 233 / 135 trades.
- **Years:** each case groups by the research's convention. F2 uses the entry year (m6). R2 uses
  the signal candle's year (m18; no trade differs between the two). The CNN uses the signal year,
  one book per year.
- **Parity runs** remove the deliberate differences, one at a time:
  - **Signals:** they use the research's own signals, its event table filtered by the same rule
    (CNN: the same score file).
  - **Cash:** they use `CashCapped(5)` instead of `EqualSlots(5)`: equal slots, with the stake
    capped at the cash left, so the engine takes every entry the research took. The bp per trade
    is an unweighted mean, so the smaller stake does not change it. With a fixed stake (R2) they
    use $1e9 of capital instead.

## Causes, measured

### (a) Cash checks

Equal slots size each entry at realised equity / 5. After a loss, the cash left can be less than
that: with 4 positions open and the fifth just closed at a loss, cash is `E/5 · (1 + r)` but the
stake is `E/5 + E·r/25`. The engine then skips the entry (`skipped["cash"]`); the research never
checked cash.

Changing the capital does not isolate this with equal slots, because the stake scales with it.
So the diagnostic changes the sizing instead (`CashCapped`). It measures:

- **F2:** 258 signals were skipped for cash. Seven research trades were not taken, and two other
  trades took their slots. Per year: -0.4 / -11.4 / +0.8 / +1.6 bp (2021, 2022, 2024, 2025),
  0 elsewhere.
- **CNN:** 384 signals were skipped for cash. Per year: +2.8 / -7.1 / -2.9 / -0.2 bp (2023..2026),
  0 in 2022.
- **R2:** none. $10,000 covers 200 × $50, and a run with $1e9 takes the same trades.

### (b) Costs

The engine buys at `open · (1 + s)` and sells at `p · (1 - s)`, and pays the fee `f` on each
side's notional. Its net return is `(1 + g) · k - 1`, where `g` is the gross return over the raw
entry open and `k = (1 - s)(1 - f) / ((1 + s)(1 + f)) = 0.9970045` (s = 0.05%, f = 0.1%). The
research took `g - 0.003`.

The difference per trade is about +0.04 bp, minus 0.30 bp for every 1% of gross return `g`:

| g | -30% | -10% | 0 | +2.5% | +20% |
|---|---|---|---|---|---|
| engine - research | +9.0 bp | +3.0 bp | +0.04 bp | -0.7 bp | -5.9 bp ($0.030 on $50) |

Measured on the parity runs, this formula explains every matched trade to within 0.00012 bp
(F2), 0.00018 bp (CNN) and $0.000001 (R2 time exits). The residual is the research's float32
forward paths. Per year, costs are worth:

- **F2:** -0.6 to -1.0 bp; +0.1 bp in 2025, a losing year.
- **CNN:** -0.2 to -1.0 bp.
- **R2:** -$70 over 7 years, from -$91 on 3,061 take-profit exits (+$20 on time exits). Per
  year: -2.7 / -26.5 / +2.9 / -9.1 / -23.2 / -7.7 / -4.2.

### (c) Look-ahead filters in the research

`build_events.py` dropped events whose next 48 candles (4 hours) had a missing candle, and events
in a series' last 4 hours. A live strategy cannot know either, so the production rules keep them.
Our signal frames are a strict superset of the research's:

- **F2:** 6 extra signals.
- **R2:** 19 extra signals.

Every extra signal falls on 2020-12-21, between 10:05 and 12:10 UTC, less than 4 hours before
the Binance outage whose missing candles start at 13:50 or 14:10. No extra signal is due to a series end, and none is missing
from our side. Their effect on the books:

- **F2, 2020:** 3 extra trades (NEAR +325, FIL +442, BCH +190 bp), +2.6 bp.
- **R2, 2020:** 14 extra trades (-$60), which kept 1 research trade (-$3) out. Net -$57.

### (d) No flat padding after halts and delistings

In the `all` universe the research appended up to 300 flat candles at the last close after a
series stopped for a day or more (`common.load`). That feeds the features and the forward paths.
The engine does neither:

- **Features:** none of the 49,429 R2 signals has different features, and the research has no
  signal we lack. Measured effect: 0.
- **Exits:** a position whose series ends exits at the last close, reason `series_end` (13 R2
  trades). That is the same price the research used: its 3-day close carries the last close.
  The P&L matches to $0.000001. The engine frees the slot at the delisting rather than 3 days
  after entry, but the parity run's trade set is identical, so this has no measured effect.

### (e) Slot timing and entry-candle rules

The engine and the research agree on the timing:

- entry at the open of t+1;
- a position whose exit is at or before an entry time frees its slot first;
- same-candle priority order;
- one position per coin;
- a missing entry candle skips the signal. One R2 signal (OOKIUSDT, 2024-11-06 02:55, the
  delisting's last candle) has no entry candle in the store. The research had a flat padding
  candle there (d), but its book did not take the signal either.

The parity runs prove it: 891 / 891 F2 trades, 9,968 / 9,968 R2 trades and 746 / 746 CNN trades
are the same.

Exit times differ in how they are written, not in the slot they free. The research marks a
5-minute exit at the end of the exit candle. The engine writes an exit inside a candle at that
candle's close time, and a gap fill at the candle's open time.

The one fill-rule difference on 5m candles: a candle that opens above the take-profit target
fills at the open in the engine, at the target in the research. This affects 3 R2 trades
(+$0.89).

### (f) Per-year simulation (CNN)

`portfolio_years` ran a separate book per year, each starting from scratch. The golden test does
the same (`start`/`end` per year).

A single continuous run would differ. In 2024 it gives +78.2 bp instead of +57.2 bp (both
without the cash check): a TRBUSDT position opened on 2023-12-31 is still open on 2024-01-01,
blocks four TRB entries and shifts the slot chain. Per-year runs are what the reference measured.

### (g) Other findings

- **No engine bug was found.**
- **Exit-plan windows:** the follow-up about capping the exit window at the data end did not
  show up here. Every golden plan has `max_hold`.
- **Case 3's book:** see below. The m21 reference re-priced a fixed trade list, which the engine
  cannot do by design.

## Case 1: F2, pool universe (PR #8)

**Setup:**

- Universe: pool (`top200.txt` minus pegged assets and stocks).
- Rule: F2 (spike, `ret_4h <= -10%`, `qv_base >= 20,000`, `btc_vol_mult >= 6`), deepest
  `ret_4h` first.
- Book: `EqualSlots(5)`, 5 positions at most, 1-hour hold, default costs.
- Data until 2026-11-02.

| Year | Ref trades | Engine trades | Ref bp | Engine bp | Δ bp | Parity trades | Parity bp | Causes of Δ (bp) |
|---|---|---|---|---|---|---|---|---|
| 2020 | 114 | 117 | +222 | +223.6 | +1.6 | 114 | +221.0 | (c) 3 extra trades +2.6; (b) -0.8 |
| 2021 | 243 | 242 | +319 | +318.0 | -1.0 | 243 | +318.4 | (a) -0.4: ZEN (+1,007) and UNI (+115) skipped, HOT (+703) taken; (b) -1.0 |
| 2022 | 126 | 124 | +223 | +211.4 | **-11.6** | 126 | +222.8 | (a) -11.4: COTI (+527) and RUNE (+1,332) skipped; (b) -0.7 |
| 2023 | 74 | 74 | +287 | +286.6 | -0.4 | 74 | +286.6 | (b) -0.9 |
| 2024 | 156 | 155 | +284 | +283.9 | -0.1 | 156 | +283.1 | (a) +0.8: LUNC 20:50 (+170), AAVE (+321) skipped, LUNC 20:55 (+337) taken; (b) -0.9 |
| 2025 | 133 | 132 | -44 | -42.1 | +1.9 | 133 | -43.7 | (a) +1.6: APE (-255) skipped; (b) +0.1 |
| 2026 | 45 | 45 | +187 | +186.7 | -0.3 | 45 | +186.7 | (b) -0.6 |

How to read the table:

- Δ is against the published, rounded reference.
- The (b) column is parity minus the research's unrounded value: 221.8 / 319.4 / 223.5 / 287.5 /
  284.0 / -43.8 / 187.3.

Against the original tolerance:

- Every year is within ±15 bp; the largest gap is 2022 at -11.6, almost all of it the cash check.
- Trade counts are within 5% (largest 2020, +2.6%).
- **Parity:** every year has exactly the research's trades, and the bp is within 1 of the
  rounded reference. The parity test allows 2.
- **Trade list:** against `trades_F2.csv`, the engine takes 884 of the research's 891 trades
  plus 5 others (listed above). The parity run takes all 891 and no others.

## Case 2: Capitulation pick, all coins (cnn-replication B2)

**Setup:**

- Universe: all coins (pool + delisted + outside the pool).
- Rule: R2 (as F2 with `qv_base >= 1,000`), deepest `ret_4h` first, signal `open_time` before
  2026-10-01.
- Book: `FixedStake($50)` from $10,000, 200 positions at most, one per coin.
- Exits: +20% take profit (`touch`) or 3 days, no stop.

| Year | Ref trades | Engine trades | Ref $ | Engine $ | Δ $ | Tolerance | Parity trades | Parity $ | Causes of Δ ($) |
|---|---|---|---|---|---|---|---|---|---|
| 2020 | 711 | 724 | +845 | +785.5 | **-59.5** | ±150 | 711 | +842.9 | (c) -57 (14 extra trades -60, 1 displaced +3); (b) -2.7; (e) +0.5 |
| 2021 | 2,668 | 2,668 | +8,651 | +8,624.0 | -27.0 | ±433 | 2,668 | +8,624.0 | (b) -26.5 |
| 2022 | 1,467 | 1,467 | -1,082 | -1,079.0 | +3.0 | ±150 | 1,467 | -1,079.0 | (b) +2.9 |
| 2023 | 1,013 | 1,013 | +2,967 | +2,958.3 | -8.7 | ±150 | 1,013 | +2,958.3 | (b) -9.1 |
| 2024 | 2,290 | 2,290 | +7,584 | +7,561.0 | -23.0 | ±379 | 2,290 | +7,561.0 | (b) -23.2 |
| 2025 | 1,170 | 1,170 | +2,479 | +2,471.6 | -7.4 | ±150 | 1,170 | +2,471.6 | (b) -7.7 |
| 2026 | 649 | 649 | +1,337 | +1,333.5 | -3.5 | ±150 | 649 | +1,333.5 | (b) -4.2; (e) +0.3 |

The reference trade counts come from the rebuilt book (see Method).

Against the original tolerance:

- Every year passes; the largest gap is 2020, -$60 of ±$150.
- The cash check has no effect: there are no cash skips, and a $1e9 run is identical.
- Padding (d) has no measured effect.
- **Parity:** the trades are identical (9,968 / 9,968), and the dollars are within $27 a year, all
  of it the cost model plus +$0.89 of gap fills. The parity test allows ±$40.

## Case 3: Capitulation pick with 1m fills

**Setup:** as case 2, with `use_fine=True` (exits on 1m candles where they cover a whole 5m
candle) and `tp_fill="close"` (a take profit fills on a candle closing at or above the target, or
opening above it).

**Reference:** `m21_fillcheck.py` took the 5m book's 3,061 take-profit exits and checked each on
1m candles:

- **Filled (2,857 trades):** a 1m close at or above the target before the time limit. The trade
  keeps its take profit.
- **Touched only (204 trades):** the high reached the target but no 1m close did. The trade gets
  its 3-day close instead.

That costs $1,678 over 2020..2026. m21 kept the book's trades fixed. The engine re-runs the book,
so a "touched only" position now holds its slot until its time exit, up to 3 days.

| Signal year | 5m engine $ | 1m engine $ | Drop | Re-pricing of common trades | Book re-run | m21 loss (reference) |
|---|---|---|---|---|---|---|
| 2020 | +785 | +711 | 74 | 74 | 0 | 75 |
| 2021 | +8,624 | +8,209 | 415 | 330 | 85 | 364 |
| 2022 | -1,079 | -1,454 | 375 | 330 | 45 | 331 |
| 2023 | +2,958 | +2,844 | 114 | 71 | 43 | 83 |
| 2024 | +7,561 | +7,234 | 327 | 219 | 108 | 234 |
| 2025 | +2,472 | +803 | 1,669 | 479 | 1,190 | 515 |
| 2026 | +1,334 | +1,261 | 73 | 73 | 0 | 76 |
| **Total** | **+22,655** | **+19,608** | **3,047** | **1,575** | **1,472** | **1,678** |

**Re-pricing of common trades:** the 9,768 trades both books take; P&L on 5m minus P&L on 1m.
It is m21's measurement, done by the engine, at $1,575 (-6% of $1,678). Those trades break down:

- 187 take profits m21 calls "touched only" become time exits (-$1,539.7), and 3 become
  `series_end` exits (-$51.8).
- Every trade m21 calls "filled" (2,717 here) still fills at its take profit.
- The M1 open rule: a 1m candle that opens above the target fills at its open under `close` too,
  where m21 required a close.
  - 4 "touched only" trades fill this way (STRK on 2025-10-10, PORTO, JTO, GNS): +$22.3 against
    m21's exit.
  - 196 filled trades get a 1m open above the target instead of the target itself: +$16.7.
  - In all, about +$39 against m21's conventions.
- On the 2,911 common trades of its own book, m21 measured $1,616.5. The engine gets $1,575.0.
  The $41.5 between them is the open rule (+$39) and the cost model.

**Book re-run:** the 1m book skips more signals for full slots (25,321 against 23,337). So 213
trades of the 5m book (+$1,470) are not taken, and 18 others are (-$2). 126 of the 213 are on
2025-10-10, the liquidation cascade (+$1,186), when "touched only" positions from the same
evening filled all 200 slots.

Against the original tolerance, the total drop of $3,047 is outside $1,678 ±25% (1,258..2,098).
It needed explaining: the like-for-like part, re-pricing the common trades, is $1,575, inside
the tolerance. The rest is the engine re-running the book, which m21 did not do, and is measured
directly.

The test asserts:

- the re-pricing is within ±25% of $1,678, computed in the test from the two runs;
- the book effect equals the P&L of the trades only one book took;
- the 1m book skipped more signals for full slots.

## Case 4: Base CNN from walk-forward scores (PR #8)

**Setup:**

- Scores: `cache/volume_spike/wf/base.npz` and `seq2_meta.parquet`, converted with
  `convert_wf_npz` into `.cache/scores/base.parquet`.
- Rule: `pct >= 0.98`, priority `-score`.
- Book: 1-hour hold, `EqualSlots(5)`, 5 positions at most, default costs.
- One book per signal year, 2022..2026, as `portfolio_years` ran it.

| Year | Ref trades | Engine trades | Ref bp | Engine bp | Δ bp | Parity trades | Parity bp | Causes of Δ (bp) |
|---|---|---|---|---|---|---|---|---|
| 2022 | 81 | 81 | +334 | +333.4 | -0.6 | 81 | +333.4 | (b) -1.0 |
| 2023 | 100 | 98 | +170 | +171.8 | +1.8 | 100 | +169.0 | (a) +2.8: 3 research trades (mean +129) skipped, 1 other (+325) taken; (b) -0.5 |
| 2024 | 197 | 194 | +57 | +50.1 | **-6.9** | 197 | +57.2 | (a) -7.1: 6 research trades (mean +285) skipped, 3 others (mean +54) taken; (b) -0.2 |
| 2025 | 233 | 231 | +67 | +64.2 | -2.8 | 233 | +67.1 | (a) -2.9: 2 research trades (mean +411) skipped; (b) -0.3 |
| 2026 | 135 | 134 | +53 | +52.1 | -0.9 | 135 | +52.3 | (a) -0.2: 1 research trade (+87) skipped; (b) -0.2 |

Against the original tolerance:

- Every year passes; the largest gap is 2024, -6.9 of ±20 bp.
- The signals are the research's own scores, so (c) and (d) do not apply. No signal lacked its
  entry candle.
- **Parity:** the trades are identical (746 / 746), and the bp is within 1 of the rounded
  reference. The parity test allows 2.
