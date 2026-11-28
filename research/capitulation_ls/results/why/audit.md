# Red-team audit of the LMC headline (2026-11-28)

Scope: the headline LMC book (L + M + C, 10 equal slots, a fresh $10k each year, 2022-26:
+11.6% a trade, worst year +51%, drawdown -17%) and the checks the report quotes for it. I looked
for bugs that make results look better than they are; I did not try to improve the strategy.
Scripts: `a1_audit.py` (rebuilds the three sleeves with their seq2 rows, replays `e25.book`
trade by trade, checked to give the same trade counts as `e25.book`, then measures what the
headline leaves out) and `a1_e45_fix.py` (re-runs E45 without its look-ahead). No existing file
was edited.

## Summary

| # | Item | Verdict | Effect on LMC per trade / worst year |
| --- | --- | --- | --- |
| 1a | CNN percentiles out of sample | OK | none |
| 1b | feats.parquet features, perp gate, funding at entry | OK | none |
| 1c | M: listing price / `since_list` at entry | OK for the headline | none |
| 1d | **E45 "settings chosen on 2022-24" check: M filter measured 48h after entry** | **bug (medium; robustness row only)** | headline unchanged; E45 row 2025 +544% -> **+371%**, 2026 +89% -> **+54%**, +8.6% / +5.5% -> **+7.5% / +3.8%** a trade, 2025 DD -29% -> -34% |
| 2a | Entry at the next 5m open, path stitching 5m to hourly | OK | none |
| 2b | Order of take profit and stop inside a bar | OK (adverse first; L has no stop) | none |
| 2c | Take profit filled at a touch | OK, small | needing 1% through the level: +11.44% (-0.14 pp), worst year unchanged at +51% |
| 2d | Costs; funding sign for shorts | OK (checked by hand on one trade) | none |
| 3a | Stake cap uses volume known at entry | OK (M's cap is 0.5% of *one hour's* perp volume: stricter than intended) | none (conservative) |
| 3b | Drawdown on closed trades only | issue (low) | max DD -16.7% -> **-18.6%** marked to market (2026: -11.4% -> -17.2%); pessimistic bound -26% (2024) |
| 3c | Trades crossing a year end | issue (low) | 8 trades credited to the entry year; about $11k of 2024 stakes still open when the 2025 book starts with a fresh $10k; worst year (2026) not affected |
| 3d | Equal- vs stake-weighted mean | OK | stake-weighted +12.1% (higher) |
| 4 | Survivorship, delisting during a hold | OK | none (LUNA's -100% long is in the book) |
| 5 | Deflated Sharpe "0.90-0.98" | numbers reproduce; interpretation optimistic (medium) | DSR 0.60 / 0.25 (1,000 / 5,000 trials) if trial Sharpe ratios spread 0.75 instead of 0.48 |
| - | Perp symbol collisions (BOBUSDT, CATUSDT) | latent bug (low) | no LMC trade affected |

**Bottom line.** I found no bug that inflates the headline LMC book itself: the walk-forward
scores, features, entries, exits, costs, funding and survivorship all hold up when checked in
the code and with small numeric tests. One look-ahead bug inflates the strongest robustness
argument, the E45 check ("settings chosen on 2022-24, read on 2025-26"). Fixed, it reads +7.5% /
+3.8% a trade and +371% / +54% a year in 2025 / 2026, still above the targets in 2025 and just
above +50% in 2026, with a drawdown of up to -34%. Drawdowns marked to market are a little
deeper than reported (worst -18.6%). The deflated Sharpe ratio is computed as described, but
its trial variance comes from near-copies of the winner. The main reason the +11.6% will not hold
forward is still in-sample choice, which the report states: every LMC setting was picked on
2022-26. The fixed E45 (+3.8% a trade in 2026) and the 2017-19 holdout for L (+9.5%) are the
best guides to the real size.

## Details

### 1. Look-ahead

**1a. CNN percentiles: OK.** `day5` comes from `i10_train.run` in the I10 worktree
(`research/volume_spike/i10_train.py:510-548`, folds from `m12_cnn_wf.folds_of`, `:366-373`).
Fold Y trains on rows before Y-1 July 1, picks the epoch on Y-1 H2 (`fit`, `:418-468`) and
scores year Y. The percentile is `searchsorted` of each test score against the sorted scores of
**the fold's own training rows** (`ref`, `:518`, `:537`), not against other scores from the test
year. A numeric check agrees: the share of rows at pct >= 0.995 varies by year (2022 0.08%,
2023 0.06%, 2024 0.46%, 2025 0.59%, 2026 0.03%); it would be exactly 0.5% if the percentile were
taken within the test year. Normalisation is "frozen" (statistics from rows before 2021-07-01,
before every fold's cut). The model inputs end at the spike candle (`build_seq2.py` header), and
the entry is the next candle's open (`build_seq2.py:126`, `build_fwd72.py:30-31`).

**1b. Features: OK.** `qv_base` is the shifted rolling median of 5m quote volume
(`build_events.py:50`). `ret_4h` / `btc_ret_4h` end at the spike candle's close
(`build_events.py:60,141`). The context features (`r7d`, `qv7_30`, ...) use the last *closed*
hour (`build_feats.py:39`). Funding `rate` is joined backward on settlement time <= spike open
(`build_feats.py:130-133`); the archive's first column is the settlement (calc) time
(`fetch_futures.py:89`). The perp gate `has_perp` uses the two closed hours before the spike's
hour (`futures_ls.py:77-80`). `n_prior_pumps30` counts earlier days only (`build_feats.py:119`).

**1c. M at entry: OK.** `since_list` = perp open at the entry hour / perp open at the listing
hour - 1 (`e27_listing_feats.py:20-23`). The entry hour's open is the entry price, so it is known
when the short is placed. The activity gate uses the previous hour's volume
(`e13_listing_book.py:36`). The cached `e27_listings.parquet` matches a fresh
`e13.trades(72, 30, 0.5, 0.5)` exactly (201 trades, max |net diff| 0).

**1d. E45 look-ahead: bug.** `e45_frozen_2024.py:58-66` builds the weak-listing set from E27's
`since_list` **measured at listing + 72h**, then lets the 2022-24 grid choose the entry delay
from (24, 72). It chooses 24h. That book shorts at listing + 24h only the listings that will be
at or below their listing price 48 hours later. `a1_e45_fix.py` measures `since_list` at each
delay's own entry:

| E45 M sleeve (delay 24h, 30d, TP / stop 50%) | n | mean |
| --- | --- | --- |
| filter measured at 72h (E45) | 132 | +20.1% |
| filter measured at entry (fix) | 120 | +16.3% |

With the honest filter the 2022-24 choice is still delay 24h and 8 slots, and the book reads:

| E45 book | 2022 | 2023 | 2024 | 2025 | 2026 | a trade, all years |
| --- | --- | --- | --- | --- | --- | --- |
| as reported (leaky) | +95% | +182% | +261% | **+544%** (+8.6%) | **+89%** (+5.5%) | - |
| fixed | +106% | +188% | +165% | **+371%** (+7.5%, DD -34%) | **+54%** (+3.8%, DD -26%) | +7.2% |

The report's robustness row ("2025 +544%, 2026 +89% (+8.6% / +5.5% a trade)") should be
replaced by the fixed numbers. The headline LMC uses delay 72h, where the filter is measured at
entry, so the headline is not affected. No other script applies the 72h filter to an earlier
entry (`x5_m_tail.py` enters at or after 72h; E46 / E48 keep M fixed at 72h).

### 2. Execution

**2a. OK.** `sim.paths` (`sim.py:50-58`) joins fwd2 (5m candles e..e+95 after the entry candle
e) and fwd72 hours 9-72 (candles e+96..e+863). The two are contiguous, and both use the same grid
and entry (`build_fwd72.py:22-35`). `t_in` = spike open + 300 s = the next open
(`e25_combined.py:44`).

**2b. OK.** `exits()` resolves a step in the order liquidation, stop / trail, take profit, time
(`sim.py:141-146`), so the adverse extreme counts first. L has no stop, so a take profit
followed by a low in the same hourly bar is correctly a take profit (a resting limit order would
fill). E13 checks the stop first too (`e13_listing_book.py:49`). Among M's 136 candidates, no bar
hits both levels, and none of the 21 stop-outs opened beyond the stop (no gap fill worse than
-50%).

**2c. Fills at a touch: small.** 26 of the 92 take-profit exits in the book come in a bar whose
high is under 20.5%. If the price must trade through the level before the order fills:

| take profit needs a high of at least | a trade | L a trade | worst year |
| --- | --- | --- | --- |
| 20.0% (as built) | +11.58% | +11.68% | +51% |
| 20.2% | +11.53% | +11.51% | +51% |
| 20.5% | +11.56% | +11.61% | +51% |
| 21.0% | +11.44% | +11.26% | +51% |

**2d. OK.** Costs: `net = lev x (ret - cost - stop_slip - fund)` (`sim.py:219`), with 0.4%
spot and 0.3% perp round trip. Funding: `funding_paid` returns side x (sum of rates in
(t_in, t_out]) (`futures_ls.py:98-102`), so a short pays -rate. Checked by hand on the C trade
that paid most (LAYERUSDT: the rates in the window sum to -19.7%, and the trade is charged
+19.7%). C pays +1.5% on average and M +1.3%, all included in `net`.

### 3. Book

**3a. Stake cap: OK.** L and C: `qv_base x 288 x 0.005` (`e25_combined.py:47-49`), from a
shifted median. M: `qv` is the **previous hour's** perp volume (`e13_listing_book.py:55`), but
`e25.book` applies `qv x 0.005` (`e25_combined.py:198`), while E13's own book uses `qv x 24 x
0.005` (`e13_listing_book.py:78`). So M's cap is 24 times stricter than intended. It is
conservative and bound for only 1 of 135 M trades at $10k.

**3b. Drawdown: low.** `book()` updates the peak and drawdown only when a closed trade is
booked, and closes positions in list order, not exit order (`e25_combined.py:177-185`).
`a1_audit.py` marks every open position at each hourly close (step closes on the 5m / hourly /
perp paths, minus the round-trip cost):

| year | reported DD | closed-trade DD (exit order) | mark-to-market DD | bound (every position at its step extreme) |
| --- | --- | --- | --- | --- |
| 2022 | -9.7% | -10.5% | -11.3% | -11.6% |
| 2023 | -6.9% | -6.7% | -8.0% | -8.0% |
| 2024 | -15.8% | -15.8% | -16.7% | -26.1% |
| 2025 | -16.7% | -16.7% | **-18.6%** | -18.9% |
| 2026 | -11.4% | -11.4% | **-17.2%** | -18.5% |

Sizing on closed-trade equity (`eq`) is a choice, not a bug: stakes never exceed cash.

**3c. Year end: low.** Trades are grouped by entry year, and open trades are credited at the end
of their entry year's book (`e25_combined.py:211-214`). Eight trades cross a year end (3 from
2023, 5 M shorts from 2024 with about $11k of stakes and +$3.8k P&L). The 2025 book still starts
with a fresh $10k, so for a few weeks the two years together use more than one account's
capital. Neither 2026 (the worst year) nor the per-trade mean is affected. Separately,
`dsr.daily_returns` (`dsr.py:22-27`) drops these trades' P&L, because their exit day falls
outside the entry year. That lowers the stated Sharpe ratio a little.

**3d. Mean per trade: OK.** The headline is the equal-weighted mean of book nets (430 trades,
+11.58%). Weighted by stake it is +12.09% (L +11.6%, M +22.2%, C +3.1%).

### 4. Survivorship: OK

The "all" universe adds delisted coins to today's pool (`volume_spike/common.py:83-85`). After a
series ends, `load` pads flat candles at the last close, and fwd72 holds the last close
(`common.py:95-98`, `build_fwd72.py:4-6`), so a long into a delisting exits at the last traded
price. Six book longs end flat: five LUNA longs on 2022-05-11/12 (four take profits, then one at
**-100%**, booked) and KEY in 2024-12 (-12.6%). The perp klines and funding come from the
archive with delisted symbols (`fetch_futures.py:3`). Perp 5m paths were fetched only for coins
with a +20% 4h pump at some time (`fetch_futures.py:149-156`), but all 922 C candidates have a
path, so this does not filter C.

### 5. Selection and the deflated Sharpe ratio

The stated numbers reproduce exactly: realised daily returns, N = 1,000 -> DSR 0.976 and
N = 5,000 -> 0.901 (Sharpe ratio 2.16 annualised, skew 11.8). Two remarks:

- `e43_lmc_stress.py:48` hard-codes the trial variance as 0.48 annualised, taken from E39's 68
  **LQM** books (`e39_dsr.py`). A family of 108 LMC books (threshold x take profit x slots x
  sleeve set) gives 0.49, so the number carries over. Marked to market, the daily returns have a
  higher Sharpe ratio (2.53, skew 4.7) and DSR 0.997 / 0.985. The realised-P&L method is not
  flattering here.
- The weakness is the variance itself. 0.48 is the spread among near-copies of the winning
  books (mean Sharpe 1.96), not across all the trials this search and its predecessors ran,
  which include failed families (leveraged longs, perp pump shorts, listing variants, explorers)
  and the I10 network variants behind `day5`. The DSR depends heavily on it:

| trial Sharpe sd (annualised) | N=100 | N=1,000 | N=5,000 | N=20,000 |
| --- | --- | --- | --- | --- |
| 0.48 (as used) | 1.000 | 0.997 | 0.985 | 0.955 |
| 0.75 | 0.964 | 0.598 | 0.250 | 0.081 |
| 1.00 | 0.498 | 0.019 | 0.000 | 0.000 |

  (MTM daily returns.) The plan itself notes that the robustness study's pick had DSR 0.97 and
  failed. I would quote the DSR as "0.90-0.98 if the trials' Sharpe ratios spread like the LMC
  family; well below 0.9 if they spread more widely", and put more weight on the fixed E45 and
  the 2017-19 holdout.

### Latent bug: perp symbol collisions (low, no effect on LMC)

`spot_name` strips the 1000 / 1M / 1000000 prefixes (`futures_ls.py:15-19`). Two pairs of
different perps then map to the same spot name: BOBUSDT + 1000000BOBUSDT, and CATUSDT +
1000CATUSDT. `e13.perp`, `perp_active` and `funding_cum` group by spot name, so they merge two
price series (with different units) and add up two funding histories. The spot coin 1000CATUSDT
can never find its perp. No LMC, M or C trade involves these symbols today, but any future
listing or crash trade on them would be wrong.
