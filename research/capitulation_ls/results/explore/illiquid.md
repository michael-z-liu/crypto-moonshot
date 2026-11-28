# Illiquid spikes: does L work below seq2's liquidity floor, and does it add flush days?

Question: the spike builder keeps only spikes on coins whose median 5m quote volume (`qv_base`)
is >= 5,000 USDT (about 1.4M USDT a day). L earns only in the extreme CNN tail (pct >= 0.995) on
rare flush days, so more qualifying events is the lever. Do the dropped, thinner spikes add
**qualifying events and new flush days**, and does L still work on them after realistic costs?
Bands: **b1k** = 1,000 <= qv_base < 5,000, **b500** = 500 <= qv_base < 1,000 USDT per 5m candle.
Spot coins 2022-01..2026-10, plus the perp-only coins of p1.

Scripts: `q1_build.py` (spikes, inputs and fold scores, streamed), `q1_paths.py` (spot outcome
paths of the pct >= 0.95 pool), `q1_eval.py` (everything below; raw tables in
`illiquid_tables.md`).

## Verdict: fails as a lever (no new flush days), although the per-trade edge carries over

- **The edge carries over to thin coins.** Spot b1k: 538 trades, mean net **+0.124** at 0.4% cost
  (+0.118 at 1.0%, +0.108 at 2.0%), 86% wins. Spot b500: 79 trades, +0.161 (+0.146 at 2%). The
  hockey stick is there in both bands: the fixed-bin outcome is +0.007 / +0.013 / +0.052 /
  **+0.130** (b1k) and +0.002 / +0.035 / +0.040 / **+0.165** (b500), against spot-liquid's
  ~0 / +0.026 / +0.059 / +0.132. Perp-only b1k: 53 trades, +0.130. Perp-only b500: no candidates.
- **It adds almost no new flush days.** Thin-coin L candidates fire on the same days as liquid
  spot L or perp-only L: 17 entry days over both spot bands, of which **15 are already L days**.
  Only **2 new days** in five years (2022-05-25 with 1 trade, one 2024 day with 2 trades). The
  perp-only b1k candidates add **0**. 2026, the year that needs events, gets none.
- **The book does not improve.** On the shared days the 10 slots are already full, so each band
  trade replaces a liquid L or L-perp-only trade (20 swaps over five years). From 2025 on, the
  band trade's stake is cut by the liquidity cap ($2.0k on average against $5.1-5.3k for the
  trade it replaces), so P&L falls even when both trades hit the take profit. LMC+ (worst year
  +62%) becomes +62..64% worst year, with 2023 down by up to 6 points and 2025 down by 18-54 points.
  This holds at all three costs; cost hardly matters because so few band trades get in.
- **Realism is not the problem.** On flush days the thin coins trade heavily: the entry candle's
  quote volume is >= 3,434 USDT on every trade (5th percentile 18k, median 35k-1.2M per year).
  No trade has a tiny or empty entry candle, and no book stake exceeds the entry candle's volume.
  A stricter take profit (the high must reach +22% to fill at +20%) costs 0.4 points per trade
  (+0.127 -> +0.123), the same as for liquid L (+0.129 -> +0.123).

**Recommendation:** do not add the sub-5k spikes to L. They are the same flush days in thinner
coins, and in a 10-slot book they crowd out larger, uncapped trades. Look for new days elsewhere.

## Look-ahead and pipeline

- Each spike in year Y is scored by **the day5 walk-forward fold Y** (`day5_2022` .. `day5_2025`,
  `day5` for 2026). pct comes from that fold's own reference (`p1_lib.FoldModel`, exactly as
  p1_score.py). No sub-5k spike was in any training set. Threshold, take profit, hold and book
  are unchanged from spot L. The only choices made here are the band edges (given in the brief),
  the 1% / 2% cost rows, the trade-through check and the "filler" book row.
- Spikes and inputs come from `p1_lib.spikes_and_coin_channels` on `common.load` candles (p1
  showed this path reproduces seq2's spikes and inputs bit for bit). Its liquidity floor is lowered
  to 500 (a module patch applied at import, so the spawned workers see it too), and only rows with
  qv_base < 5,000 are kept. The spike counts match the events table: b1k 2022 is 490,329 in both,
  and 2023 has 473,095 here against 472,978 there (+117, because seq2's 48-candle forward-gap
  filter is not applied, as in `paper.spike_job`).
- Outcomes use `build_seq2.fwd_job` / `build_fwd72.job` (entry at the next 5m open, TP +20%, 72h)
  through `sim.simulate`, with sim's loaders swapped as h1_eval does for the holdout. One
  position per coin applies within a band. In the books, `e25_combined.book` keeps one position
  per coin across all sleeves.
- **Caveat: the CNN extrapolates.** Its liquidity input (z-scored log qv_base) is below anything
  it was trained on. It scores thin spikes lower: only 1.1% reach pct 0.95, against about 5% of
  liquid spikes. The calibration still rises steeply in the top bin, so the signal keeps its
  meaning.

## Coverage

555 spot coins, 3,654,210 sub-5k spikes (b1k 2,428,519, b500 1,225,691).

| year | band | spikes | coins | q95 | q995 | q995_days |
|---|---|---|---|---|---|---|
| 2022 | b1k | 490329 | 313 | 4153 | 34 | 1 |
| 2023 | b1k | 473095 | 335 | 4733 | 143 | 2 |
| 2024 | b1k | 485303 | 331 | 10701 | 171 | 7 |
| 2025 | b1k | 584957 | 430 | 10231 | 1147 | 3 |
| 2026 | b1k | 394835 | 423 | 3984 | 16 | 4 |
| 2022 | b500 | 244546 | 273 | 925 | 1 | 1 |
| 2023 | b500 | 275610 | 289 | 1164 | 15 | 2 |
| 2024 | b500 | 113379 | 240 | 562 | 1 | 1 |
| 2025 | b500 | 243425 | 370 | 1478 | 269 | 2 |
| 2026 | b500 | 348731 | 401 | 1654 | 2 | 2 |

Perp-only coins (p1's 200, on perp candles, with p1's activity gate): 340,140 sub-5k spikes. b1k:
246 at pct >= 0.995, 233 of them gated, on 1 day in 2025 and 4 days in 2026. b500: none at
pct >= 0.995. Only 12-16% of b500 perp spikes pass the gate anyway.

## L per band (spot, entry next 5m open, TP +20%, 72h, one position per coin)

| band | year | n | coins | days | net @0.4% | @1.0% | @2.0% | win | tp_rate | entry-candle qv (median) |
|---|---|---|---|---|---|---|---|---|---|---|
| b1k | 2022 | 18 | 18 | 1 | +0.049 | +0.043 | +0.033 | 0.56 | 0.56 | 25,554 |
| b1k | 2023 | 108 | 86 | 2 | +0.131 | +0.125 | +0.115 | 0.97 | 0.24 | 178,440 |
| b1k | 2024 | 112 | 91 | 7 | +0.037 | +0.031 | +0.021 | 0.57 | 0.32 | 112,343 |
| b1k | 2025 | 290 | 189 | 2 | +0.159 | +0.153 | +0.143 | 0.96 | 0.76 | 82,519 |
| b1k | 2026 | 10 | 10 | 4 | +0.139 | +0.133 | +0.123 | 0.80 | 0.80 | 145,034 |
| b500 | 2022 | 1 | 1 | 1 | +0.196 | +0.190 | +0.180 | 1.00 | 1.00 | 2,357,191 |
| b500 | 2023 | 14 | 13 | 2 | +0.119 | +0.113 | +0.103 | 1.00 | 0.21 | 83,403 |
| b500 | 2024 | 1 | 1 | 1 | -0.159 | -0.165 | -0.175 | 0.00 | 0.00 | 28,357 |
| b500 | 2025 | 61 | 48 | 2 | +0.179 | +0.173 | +0.163 | 0.98 | 0.80 | 50,409 |
| b500 | 2026 | 2 | 2 | 2 | +0.056 | +0.050 | +0.040 | 0.50 | 0.00 | 27,212 |
| perp-only b1k | 2025 | 44 | 31 | 1 | +0.135 | +0.129 | +0.119 | 0.84 | 0.64 | 313,739 |
| perp-only b1k | 2026 | 9 | 8 | 4 | +0.104 | +0.098 | +0.088 | 0.89 | 0.44 | 249,370 |

All years: b1k n 538, +0.124; b500 n 79, +0.161; both n 597, +0.127 (+0.111 at 2%).
Trades are clustered by day: b1k is 538 trades on 16 days, 233 of them on 2025-10-10. The mean of
day means is +0.103, 88% of days are positive, and the day-bootstrap 95% CI of the trade mean is
[+0.046, +0.155]. Weak days: 2024-04-12 (44 trades, -0.090) and 2022-05-11 (18 trades, +0.049).

Entry-candle realism (all spot band trades): no trade has an entry candle under 500 USDT or with
zero volume. The smallest entry candle is 3,434 USDT and the fewest trades in one is 53; the 5th
percentiles are 18k USDT and 303 trades. Thin coins become liquid on flush days.

Same coin: 23 b1k trades have a liquid spot-L trade on the same coin within 72h (the coin moved
across 5k). These average -0.024; the rest average +0.131. The book's one-position-per-coin rule
keeps only one of each pair.

### Calibration (diag.py, pool pct >= 0.95, below-threshold candidates simulated as if entered)

Fixed bins (same edges as spot L):

| bin | spot liquid | perp-only (p1) | spot b1k | spot b500 | perp-only b1k |
|---|---|---|---|---|---|
| 0.950-0.976 | ~0 | +0.009 | +0.007 (n 6178) | +0.002 (n 1251) | +0.014 (n 493) |
| 0.976-0.984 | +0.026 | +0.002 | +0.013 (n 612) | +0.035 (n 110) | +0.054 (n 52) |
| 0.984-0.9955 | +0.059 | +0.043 | +0.052 (n 608) | +0.040 (n 104) | -0.009 (n 49) |
| >= 0.9955 | +0.132 | +0.137 | **+0.130** (n 430) | **+0.165** (n 73) | **+0.133** (n 51) |

Decile calibration is flat at about +0.01 for deciles 1-9 and jumps in the top decile in each
band (b1k +0.104, b500 +0.105; full tables in `illiquid_tables.md`). Spearman(signal, outcome)
is +0.16 (b1k), +0.17 (b500) and +0.15 (perp-only b1k); p1 had +0.167. Scorecards:

| strategy | n | E_o | win | share_tp_win | share_time_win | share_never_went | worst_year_E_o | years_positive |
|---|---|---|---|---|---|---|---|---|
| L-spot-b1k | 538 | +0.124 | 0.86 | 0.56 | 0.31 | 0.08 | +0.037 | 1.00 |
| L-spot-b500 | 79 | +0.161 | 0.96 | 0.67 | 0.29 | 0.01 | -0.159 (n 1) | 0.80 |
| L-perp-only-b1k | 53 | +0.130 | 0.85 | 0.60 | 0.25 | 0.02 | +0.104 | 1.00 |

Trade-through take profit (the high must reach +22% to fill at +20%), cost 0.4%:

| trades | n | mean_net | tp_rate | mean_net_tt | tp_rate_tt |
|---|---|---|---|---|---|
| spot liquid L (seq2) | 1883 | 0.129 | 0.649 | 0.123 | 0.576 |
| spot b1k | 538 | 0.124 | 0.556 | 0.120 | 0.504 |
| spot b500 | 79 | 0.161 | 0.671 | 0.160 | 0.610 |

## New flush days (UTC days with an L candidate, pct >= 0.995)

| band | year | spot_L_days | perp_only_L_days | band_days | new vs spot L | new vs spot L + perp-only L | trades on new days |
|---|---|---|---|---|---|---|---|
| b1k | 2022 | 5 | 1 | 1 | 0 | 0 | 0 |
| b1k | 2023 | 3 | 2 | 2 | 0 | 0 | 0 |
| b1k | 2024 | 9 | 8 | 7 | 1 | 1 | 2 |
| b1k | 2025 | 5 | 12 | 3 | 0 | 0 | 0 |
| b1k | 2026 | 4 | 20 | 4 | 1 | 0 | 0 |
| b500 | 2022 | 5 | 1 | 1 | 1 | 1 | 1 |
| b500 | 2023 | 3 | 2 | 2 | 0 | 0 | 0 |
| b500 | 2024 | 9 | 8 | 1 | 0 | 0 | 0 |
| b500 | 2025 | 5 | 12 | 2 | 0 | 0 | 0 |
| b500 | 2026 | 4 | 20 | 2 | 0 | 0 | 0 |
| perp-only b1k | 2025-26 | | | 5 | | 0 | 0 |

Both spot bands together: 2 new days in five years (3 trades, all hit the TP). Perp-only coins
added 24 new days (p1), so the contrast is plain.

## Books (e25 `book`, 10 slots, $10k each year, stake <= 0.5% of qv_base x 288)

Band trades at the cost shown; LMC's own trades at their usual costs. Each cell: return
(drawdown, L-sleeve trades).

| book | 2022 | 2023 | 2024 | 2025 | 2026 | min yr |
|---|---|---|---|---|---|---|
| LMC | +62% (dd -10, L23) | +81% (dd -7, L20) | +156% (dd -16, L62) | +625% (dd -17, L23) | +51% (dd -11, L16) | +51% |
| LMC+ | +62% (dd -10, L23) | +81% (dd -7, L20) | +152% (dd -16, L67) | +790% (dd -17, L43) | +111% (dd -10, L46) | +62% |
| LMC+ + b1k @0.4% | +62% | +75% | +149% | +765% | +113% | +62% |
| LMC+ + b500 @0.4% | +64% | +81% | +152% | +741% | +111% | +64% |
| LMC+ + both @0.4% | +64% | +75% | +149% | +736% | +110% | +64% |
| LMC+ + both as filler (prio below liquid L) @0.4% | +64% | +81% | +149% | +772% | +110% | +64% |
| LMC+ + spot b1k + perp-only b1k @0.4% | +62% | +75% | +148% | +765% | +124% | +62% |
| LMC+ + b1k @1.0% | +62% | +75% | +148% | +764% | +113% | +62% |
| LMC+ + b500 @1.0% | +64% | +81% | +152% | +741% | +111% | +64% |
| LMC+ + both @1.0% | +64% | +75% | +148% | +735% | +109% | +64% |
| LMC+ + spot b1k + perp-only b1k @1.0% | +62% | +75% | +147% | +764% | +123% | +62% |
| LMC+ + b1k @2.0% | +62% | +74% | +146% | +763% | +112% | +62% |
| LMC+ + b500 @2.0% | +64% | +81% | +152% | +740% | +111% | +64% |
| LMC+ + both @2.0% | +64% | +74% | +146% | +735% | +109% | +64% |
| LMC+ + spot b1k + perp-only b1k @2.0% | +62% | +74% | +146% | +763% | +122% | +62% |

Drawdowns are unchanged in every row (-10 / -7 / -16 / -17 / -10).

Stakes of the band trades the book takes (0.4% cost; the same at 1% and 2%):

| book | band trades | cap-limited | avg stake | avg slot | band P&L (5 yrs) |
|---|---|---|---|---|---|
| LMC+ + b1k | 19 | 3 | $1,489 | $1,963 | $3,401 |
| LMC+ + b500 | 5 | 5 (all) | $1,073 | $2,838 | $736 |
| LMC+ + both | 20 | 5 | $1,425 | $1,930 | $3,200 |
| LMC+ + both as filler | 11 | 2 | $1,612 | $1,815 | $2,064 |
| LMC+ + spot b1k + perp-only b1k | 22 | 3 | $1,531 | $1,940 | $4,395 |

Every b500 trade in a book is cap-limited (0.5% x qv_base x 288 is $720-1,440). In 2025 all 3 band
trades were capped. No book stake exceeds its entry candle's quote volume.

## Why it failed

1. **Same days, not new days.** L fires on market-wide capitulations. On those days liquid coins
   qualify too, so the thin coins qualify on days L already has: 15 of 17 spot band days and all
   5 perp-only b1k days. The perp-only coins of p1 added new days because they are a different
   *set of coins listed in 2025-26*. They traded on days when the spot universe had no extreme
   candidate. Thin spot coins are the same coins (and the same years) as the liquid universe, one
   liquidity tier down.
2. **The book is slot-bound on those days.** On a flush day LMC+ already has more L candidates
   than its 10 slots. A band trade can only replace one. Over 2022-26, 20 band trades displaced 19
   liquid / perp-only L trades. In 2025 the displaced trades netted +0.19 on $5.1-5.3k stakes, and
   the band trades netted +0.20 on $2.0k capped stakes. That is a loss of about $3.8k P&L, so the
   2025 return drops from +790% to +736%. Lowering the band trades' priority below liquid L
   ("filler") only halves the damage (+772%). They still enter on the same days, minutes earlier.
3. **The CNN rates thin spikes lower.** Only 1.1% of thin spikes reach pct 0.95 (about 5% of
   liquid ones), so the 3.65M extra spikes yield only 617 entries, two thirds of them on four
   days (2025-10-10, 2023-08-17, 2023-06-10, 2025-02-03).

Not the reason: cost or fills. Mean net stays +0.11 at a 2% round trip, entry candles are never
tiny, and a trade-through take profit costs as much as it does for liquid L.

## Rebuild and disk

OWN/q1 holds 335 MB: `spot_spikes.parquet` (246 MB, every sub-5k spot spike with raw / pct),
`perp_spikes.parquet`, `cand.parquet` + `fwd2.npy` / `fwd72.npy` (80 MB, the pool's paths), and
the trades. The 13-channel input tensors (about 27 GB for 3.65M rows) were never written: they
are built and scored in 60k-row buffers. Rebuild from scratch, from `research/capitulation_ls`:
`q1_build.py spot` (about 6 min on 8 workers), `q1_build.py perp` (2 min), `q1_paths.py` (1 min),
`q1_eval.py` (1 min).
