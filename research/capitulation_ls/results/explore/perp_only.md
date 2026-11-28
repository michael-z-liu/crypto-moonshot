# Perp-only coins: more qualifying events for L, and C on the same coins

Question: the long sleeve L earns only in the extreme tail of the CNN signal (pct >= 0.995, about 25
flush days in 2022-26). Do the Binance USDⓈ-M perps that have **no spot market** add qualifying
events, and do L (as a 1x perp long) and the crash short C still work on them?

Scripts: `p1_fetch.py` (symbol list + 5m klines), `p1_folds.py` (day5 walk-forward folds 2023-25),
`p1_lib.py`, `p1_parity.py`, `p1_build.py` (spikes + CNN inputs), `p1_score.py`, `p1_eval.py`
(everything below; raw tables in `perp_only_tables.md`).

## Verdict

- **L transfers to perp-only coins.** 328 trades in 2022-26 (one position per coin, no slot limit),
  mean net **+0.131** after 0.3% cost and funding, 86% wins, 66% hit the +20% take profit, no
  stops. Every year positive (worst year +0.101). On a day basis: 42 entry days, 86% positive,
  t = 5.2 over day means.
- **The hockey stick is there.** In the same fixed bins as spot L, the outcome is +0.009 / +0.002 /
  +0.043 / **+0.137** for pct 0.950-0.976 / 0.976-0.984 / 0.984-0.9955 / >= 0.9955. Spot: ~0 /
  +0.026 / +0.059 / +0.132. The edge again sits only in the top bin.
- **It adds flush days, mostly in 2025-26.** Perp-only coins have L candidates on 43 days. 24 of
  those days have no spot L candidate: 7 in 2025 and 17 in 2026, where spot L had only 5 and 4
  days. Trades on these new days are weaker than on shared days: n 42, mean +0.079, TP rate 52%,
  against n 286, +0.139, 69% on shared days. Before 2024 almost nothing is added, because only 9
  perp-only coins existed then.
- **The LMC book gains in the years that need it.** 10 slots, $10k each year: 2025 goes from +625%
  to +790%, 2026 from +51% to +111%, drawdowns are unchanged (-17%, -10%), and the worst year goes
  from +51% to +62%. Counting only the new days still gives 2025 +715% and 2026 +79%. With a
  doubled cost (0.6%) the result is about the same (2025 +783%, 2026 +109%).
- **C does not transfer (it fails).** 529 trades, mean net -0.010, flat calibration. Added to the
  book, it fills the slots: 2025 drops to +228% (drawdown -37%) and 2026 to +39% (drawdown -63%).
  Do not add C on perp-only coins.

**Recommendation:** add perp-only coins to L, as a 1x perp long with the perp activity gate, cost
0.3% + funding. Keep C on spot-listed coins only.

## Look-ahead and honesty

- Every perp-only spike in year Y is scored by the **day5 walk-forward fold Y**. The fold trains
  before Y-1 July and early-stops on Y-1 H2. The 2022 fold is `h1_fold2022.py`'s, 2023-25 are
  retrained here (`p1_folds.py`), and 2026 is the saved `day5` fold. All five retrained folds
  reproduce day5's stored walk-forward percentiles exactly on the spot rows (2023: 295,393 rows,
  2024: 388,193, 2025: 332,214, all 100.0000% equal). So the pct here is as out of sample as the
  spot L's. No perp-only coin was ever in any training set: the CNN was trained only on spot coins.
- Pipeline parity (`p1_parity.py`): spot candles of SOL, DOGE, LUNC, WIF and BONK, run through the
  p1 pipeline, give exactly the seq2 spikes (no extra and none missing), bit-identical 13-channel
  inputs, identical BTC 4h returns, and day5 pct 100% equal for the 2022 and 2026 folds.
- Differences from spot L, all as the brief asked:
  - The coin channels (OHLC, volume, taker share, trades) and the liquidity filter (qv_base >= 5k
    per 5m) come from the perp's own candles. BTC/ETH/market channels stay spot.
  - Spikes do not use seq2's look-ahead filters (as `paper.spike_job`). This changed nothing on the
    parity coins.
  - The perp gate is `futures_ls.has_perp`: >= 50k USDT traded with a moving price in each of the
    last two closed hours, and >= 1 day of history.
  - Entry is the next 5m perp open, which must print. Funding is paid or received from
    funding.parquet, which covers all 200 coins.
- Selection is unchanged: the threshold, take profit, hold, cost and gate were fixed on spot
  before this test. The only choices made here were which symbols count as perp-only (below), the
  robustness splits, and the 2x-cost row.

## Universe

Perp-only = a USDT perp in the data.binance.vision `futures/um` monthly klines (918 perps,
delisted ones included) for which neither the symbol nor its spot name (1000X -> X) has spot 5m
klines in the shared data folder. Non-crypto underlyings are dropped using exchangeInfo:
`underlyingType` must be COIN with contract type PERPETUAL. That removes about 240 TradFi stock,
ETF, commodity and FX perps listed in 2025-26, plus the INDEX perps (DEFI, BTCDOM, FOOTBALL,
BLUEBIRD). Four stable or FX-like names are also dropped. Result: **200 coins**, 2 of them no longer
in exchangeInfo (BDXN, DOTECO). The brief's example 1000PEPEUSDT is *not* perp-only: PEPEUSDT has
spot, and 1000PEPE is already in the perp5m store. Perps that traded before their spot listing
(perps-first coins) are not included here; that window is a possible further source of events.

## Coverage

199 perp-only coins with liquid spikes in 2022-01..2026-10, 3322 coin-months of perp 5m klines, 332,212 liquid spikes.

| year | spikes | coins | perp_active | q995 | q995_active | q995_days |
|---|---|---|---|---|---|---|
| 2022 | 641 | 1 | 1.000 | 1 | 1 | 1 |
| 2023 | 4482 | 8 | 0.999 | 5 | 5 | 2 |
| 2024 | 18895 | 37 | 0.993 | 158 | 158 | 8 |
| 2025 | 149706 | 166 | 0.980 | 919 | 917 | 12 |
| 2026 | 158488 | 183 | 0.971 | 86 | 86 | 20 |

### Flush days (UTC days with an L candidate, pct >= 0.995)

| year | spot_L_days | perp_only_days | new_days | spot_L_events |
|---|---|---|---|---|
| 2022 | 5 | 1 | 0 | 248 |
| 2023 | 3 | 2 | 0 | 190 |
| 2024 | 9 | 8 | 0 | 1779 |
| 2025 | 5 | 12 | 7 | 1970 |
| 2026 | 4 | 20 | 17 | 37 |

Days with a perp-only L candidate that have no spot L candidate: 2025-01-19, 2025-01-27, 2025-04-07, 2025-08-14, 2025-10-16, 2025-10-17, 2025-11-03, 2026-01-19, 2026-01-30, 2026-01-31, 2026-02-01, 2026-02-05, 2026-02-23, 2026-04-23, 2026-04-29, 2026-05-26, 2026-05-28, 2026-06-24, 2026-06-25, 2026-07-12, 2026-08-11, 2026-08-22, 2026-09-09, 2026-10-30

## L on perp-only coins (perp long, 1x, cost 0.3% + funding)

| year | n | coins | days | mean_net | median_net | win | tp_rate | fund |
|---|---|---|---|---|---|---|---|---|
| 2022 | 1 | 1 | 1 | 0.197 | 0.197 | 1.000 | 1.000 | 0.000 |
| 2023 | 3 | 2 | 2 | 0.178 | 0.197 | 1.000 | 0.667 | -0.000 |
| 2024 | 78 | 31 | 7 | 0.112 | 0.196 | 0.808 | 0.603 | 0.000 |
| 2025 | 185 | 107 | 12 | 0.148 | 0.197 | 0.914 | 0.703 | -0.000 |
| 2026 | 61 | 44 | 20 | 0.101 | 0.196 | 0.738 | 0.623 | 0.001 |

All years: n 328, mean net +0.1312, TP rate 0.66; by entry day (top 10 days by trades):

| day | n | e |
|---|---|---|
| 2025-10-10 | 126 | 0.144 |
| 2025-02-03 | 38 | 0.182 |
| 2024-12-09 | 31 | 0.153 |
| 2026-06-04 | 16 | 0.049 |
| 2026-02-06 | 16 | 0.197 |
| 2024-04-12 | 11 | -0.052 |
| 2024-08-05 | 9 | 0.197 |
| 2024-01-03 | 8 | 0.085 |
| 2024-03-05 | 8 | 0.188 |
| 2024-06-07 | 8 | 0.059 |

Robustness (day = unit):

- 42 entry days; mean of day means +0.1061, median +0.1603, 86% of days positive; t-stat over days 5.2.
- Without the largest day (2025-10-10): n 202, mean net +0.1231.
- Trades on days with no spot L candidate (new_day) vs days shared with spot L:

| new_day | n | days | mean_net | tp_rate |
|---|---|---|---|---|
| False | 286 | 18 | 0.139 | 0.685 |
| True | 42 | 24 | 0.079 | 0.524 |

Spot L for reference (e25 `sleeve_long`, spot, cost 0.4%):

| year | n | mean_net | tp_rate |
|---|---|---|---|
| 2022 | 73 | 0.117 | 0.808 |
| 2023 | 138 | 0.138 | 0.370 |
| 2024 | 1059 | 0.107 | 0.584 |
| 2025 | 581 | 0.169 | 0.807 |
| 2026 | 32 | 0.130 | 0.781 |

### Diag scorecard and calibration (L)

Threshold 0.995: 8.2% of pct >= 0.95 candidates reach it; Spearman(signal, outcome) +0.167.

Calibration by signal decile (equal-count bins over the pct >= 0.95 pool, label = lowest pct):

| bin | n | E_o | win | p_tp | p_stopped | tp_progress | entered |
|---|---|---|---|---|---|---|---|
| 0.950 | 401 | 0.012 | 0.501 | 0.249 | 0.000 | 0.501 | 0.000 |
| 0.953 | 401 | 0.005 | 0.479 | 0.217 | 0.000 | 0.483 | 0.000 |
| 0.955 | 401 | 0.003 | 0.521 | 0.247 | 0.000 | 0.481 | 0.000 |
| 0.959 | 401 | -0.002 | 0.506 | 0.214 | 0.000 | 0.470 | 0.000 |
| 0.962 | 401 | 0.015 | 0.541 | 0.259 | 0.000 | 0.507 | 0.000 |
| 0.966 | 401 | 0.010 | 0.526 | 0.259 | 0.000 | 0.528 | 0.000 |
| 0.970 | 401 | 0.020 | 0.544 | 0.264 | 0.000 | 0.525 | 0.000 |
| 0.975 | 401 | 0.003 | 0.541 | 0.257 | 0.000 | 0.577 | 0.000 |
| 0.981 | 401 | 0.031 | 0.589 | 0.334 | 0.000 | 0.636 | 0.000 |
| 0.992 | 400 | 0.111 | 0.807 | 0.603 | 0.000 | 1.000 | 0.820 |

Same fixed bins as quoted for spot L (spot: ~0 / +0.026 / +0.059 / +0.132):

| bin | n | E_o | win | p_tp |
|---|---|---|---|---|
| 0.9500-0.9760 | 2880 | 0.009 | 0.517 | 0.244 |
| 0.9760-0.9840 | 447 | 0.002 | 0.530 | 0.268 |
| 0.9840-0.9955 | 396 | 0.043 | 0.636 | 0.361 |
| 0.9955-1.0001 | 286 | 0.137 | 0.871 | 0.692 |

## C on perp-only coins (perp short, 1x, 24h, cost 0.3% + funding)

6309 crash candidates (ret_4h <= -15%, BTC calm, active perp, qv_base >= 20k).

| year | n | coins | days | mean_net | median_net | win | tp_rate | fund |
|---|---|---|---|---|---|---|---|---|
| 2024 | 3 | 2 | 3 | 0.124 | 0.158 | 1.000 | 0.000 | 0.011 |
| 2025 | 227 | 91 | 145 | -0.028 | -0.010 | 0.476 | 0.000 | 0.005 |
| 2026 | 299 | 90 | 181 | 0.003 | 0.021 | 0.542 | 0.000 | 0.007 |

All years: n 529, mean net -0.0096; below-threshold candidates (pct > 0.5) as if shorted: n 578, mean net -0.0088. Spearman(signal, outcome) +0.020.

Calibration (signal = 1 - pct; equal-count bins):

| bin | n | E_o | win | p_tp | p_stopped | tp_progress | entered |
|---|---|---|---|---|---|---|---|
| 0.003 | 222 | -0.037 | 0.500 | 0.000 | 0.023 | 0.639 | 0.000 |
| 0.045 | 221 | 0.018 | 0.502 | 0.000 | 0.014 | 0.763 | 0.000 |
| 0.159 | 222 | 0.000 | 0.563 | 0.000 | 0.018 | 0.694 | 0.392 |
| 0.584 | 221 | -0.000 | 0.502 | 0.000 | 0.014 | 0.703 | 1.000 |
| 0.844 | 221 | -0.027 | 0.502 | 0.000 | 0.036 | 0.784 | 1.000 |

### Scorecards (diag.py)

| strategy | n | E_o | median_o | win | share_tp_win | share_time_win | share_near_miss | share_never_went | share_went_against | share_stopped | contrib_stopped | tp_progress_med | stop_progress_med | giveback_med | worst_year_E_o | years_positive |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L-perp-only | 328 | +0.131 | +0.197 | 0.86 | 0.66 | 0.19 | 0.05 | 0.06 | 0.02 | 0.00 | +0.000 | 1.00 | 0.12 | 0.21 | +0.101 | 1.00 |
| C-perp-only | 529 | -0.010 | +0.006 | 0.52 | 0.00 | 0.52 | 0.18 | 0.25 | 0.04 | 0.02 | -0.023 | 0.76 | 0.15 | 0.94 | -0.028 | 0.67 |

## 10-slot LMC book (e25 `book`, $10k each year)

| book | trades | bp | 2022 | 2023 | 2024 | 2025 | 2026 | min yr |
|---|---|---|---|---|---|---|---|---|
| LMC (base) | 430 | 1158 | +62% (dd -10, L23 C11) | +81% (dd -7, L20 C12) | +156% (dd -16, L62 C23) | +625% (dd -17, L23 C68) | +51% (dd -11, L16 C37) | +51% |
| LMC + L-perp-only | 485 | 1145 | +62% (dd -10, L23 C11) | +81% (dd -7, L20 C12) | +152% (dd -16, L67 C23) | +790% (dd -17, L43 C68) | +111% (dd -10, L46 C37) | +62% |
| LMC + C-perp-only | 942 | 463 | +62% (dd -10, L23 C11) | +81% (dd -7, L20 C12) | +166% (dd -16, L62 C26) | +228% (dd -37, L22 C281) | +39% (dd -63, L16 C336) | +39% |
| LMC + both | 987 | 507 | +62% (dd -10, L23 C11) | +81% (dd -7, L20 C12) | +161% (dd -16, L67 C26) | +295% (dd -35, L41 C278) | +117% (dd -59, L44 C332) | +62% |
| LMC + L-perp-only, new days only | 471 | 1124 | +62% (dd -10, L23 C11) | +81% (dd -7, L20 C12) | +156% (dd -16, L62 C23) | +715% (dd -17, L37 C68) | +79% (dd -11, L43 C37) | +62% |
| LMC + L-perp-only, 2x cost | 485 | 1139 | +62% (dd -10, L23 C11) | +81% (dd -7, L20 C12) | +150% (dd -16, L67 C23) | +783% (dd -17, L43 C68) | +109% (dd -10, L46 C37) | +62% |


## Why C failed on perp-only coins

- **No edge:** mean net -0.010 per trade (spot C: +0.044). 52% wins, median +0.006. 25% of trades
  `never_went`, and the median trade gives back 94% of its best move.
- **No signal:** the CNN "no bounce" filter does nothing here. Shorting the crashes the CNN expects
  to bounce (pct > 0.5) gives -0.009, the same as the entries' -0.010. Spearman(signal, outcome) is
  +0.02, and the calibration is flat (-0.037 .. +0.018 by quintile), with the strongest quintile
  the worst (-0.027, 3.6% stopped).
- **Too many candidates, all crowded:** 6,309 crash candidates and 529 entries over 326 days in
  2025-26. Perp-only coins are mostly young, thin listings that fall 15% in 4h almost daily.
- **Funding makes it negative:** the gross price return of the shorts is -0.0005 on average, with a
  mean absolute move of 0.195. Funding (+0.63% per 24h short, since crowded shorts pay) and the
  0.3% cost turn that into -0.010.
- **Book damage:** the C trades take 280-340 slots a year (spot C: 37-68). In 2026 they push the
  book's drawdown from -11% to -63%. They take slots and capital rather than L trades (the L count
  barely changes), and their equity-weighted losses cluster.

## Caveats

- **Concentration.** 126 of the 328 L trades are on 2025-10-10, the October 2025 crash. Without
  that day the mean is still +0.123. The per-day view (t = 5.2 over 42 days) is the honest
  sample size, not 328.
- **2026 thinness.** 2026 perp-only L has 61 trades on 20 days (mean +0.101). That is more days
  than spot L's 4, but each day is small and idiosyncratic (often a single coin).
- **Liquidity.** Perp-only coins are thin. The book caps each stake at 0.5% of the perp's daily
  quote volume (qv_base x 288), as for spot. Slippage on a +20% wick fill was not modelled beyond
  the 0.3% cost; the 2x-cost row shows that cost is not what drives the result.
- **Data end.** Data runs to 2026-10-31 (the archive's monthly files). The fold models are new
  files in `.cache/forward_model/day5_2023..2025` (the master may keep or delete them).
