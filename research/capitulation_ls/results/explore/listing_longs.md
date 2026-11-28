# Longs around new Binance spot listings (the other side of M)

Scripts: `x16_paths.py` (dense 5m spot paths of the 397 e12 listings, 35 days from the first candle),
`x16_listing_longs.py` (the pre-declared grid, picks, scorecards, calibration, books; grid in its
docstring), `x16_posthoc.py` (checks chosen after seeing the grid, labelled post-hoc). Raw output:
`listing_longs_raw.md` (every one of the 1456 configurations, per year), `listing_longs_posthoc.md`,
`x16_grid_{a,b}.csv`, `x16_calibration_{a,b}.csv`. In-sample exploration: choose on 2021-24, read 2025-26.

**Verdict: fails.** No long trade around new spot listings meets the target. Buying listings
unconditionally loses in every one of the 128 listing-day configurations (-1.3% to -14.8% a trade).
The two pre-declared picks both look good in 2021-24 and both turn negative in 2025-26:
* A pick (enter after the first hour if the first-hour return is above +15%, hold 24 h, TP 50%, stop 30%):
  +15.0% in 2021-24 -> -2.6% in 2025-26, n = 30 in six years.
* B pick (long listings still above their listing price at +72 h, hold 30 d, no exits):
  +32.3% -> -19.8%. Its +11.4% mean has a -16.8% median, and three trades (GMT 2022 +815%, C98 2021
  +334%, TIA 2023 +256%) make 136% of its total P&L. Without them it averages -4.2%.

For the strong listings that hurt M, the long is the exact mirror of M's short. On M's 58 strong
coins the long and short net returns correlate at -0.99. The long earns -2.2% a trade against
the short's +0.3%. There is no long edge in the squeeze: strong listings have fat tails both ways
around a falling median. The pair book (C) makes M's worst year worse (+16% -> -4%).

## Setup

* Listings: `e12_listings.parquet`, 397 spot listings 2021-01 .. 2026-09 (123 in 2021, 41, 53, 61, 97, 22).
  First 5m candle = listing. No tokenised stocks or ETFs are in the list. It contains a few renames and
  migrations (BTTC, LUNC, RENDER, POL, S, A, ...). M uses the same list, so they stay in. Dropping them
  changes the picks by +1 point.
* Listing price = close of the first 5m spot candle (the first candle's open is a placeholder,
  e.g. 0G opened at 1.0 and closed at ~4.4). M measures "strong" against the perp's first-hour open
  instead, so B's "above at 72 h" set (91 listings) is wider than M's strong complement (58, e27). C uses M's
  own definition.
* Long spot, 0.4% round trip, entry at the open of the 5m bar after the signal bar, exits on 5m bars
  with the adverse side first (`x7_lib.simulate`). Books: `e25_combined.book`, 10 slots, $10k each year,
  w = 1, one position per coin, stake <= 0.5% of the trailing 24 h spot volume up to the signal bar.
  For a 5m entry that is the first candle's volume alone. The new sleeve has the lowest priority (0.5).
  LMC+ = L + M + C + L on perp-only coins (`p2_lib.lmc`, `lperp`), 2022+.
* Pick rule (declared): best mean of per-year mean net in 2021-24, with >= 20 trades and trades in >= 3
  of the 4 years. A-subsets count only if their split is known at entry: r1h for entries >= 1 h, qv_1d
  for the 24 h entry.

### Configurations tried (all listed in `listing_longs_raw.md`)

* A (1024 rows): entry after the first 5m candle / 1 h / 4 h / 24 h x hold 4 h / 24 h / 72 h / 7 d x
  TP 20 / 50 / 100% / none x stop none / 30%, each on 8 subsets: all; first-hour return r1h <= -15%,
  -15..0, 0..+15%, > +15%; first-day quote volume < $100M, $100M-400M, > $400M.
* B (432 rows): check at 24 / 48 / 72 h x above the listing price by > 0 (all), 0-20%, 20-50%, > 50% x
  hold 24 h / 72 h / 7 d / 30 d x TP 50 / 100% / none x stop 30 / 50% / none.
* C: M + a spot long on M's 58 strong listings (e27 since_list > 0) at M's entry time, with (i) the mirror
  exit (30 d, TP 50%, stop 50%) and (ii) the B pick's exit (30 d, no exits).
* Post-hoc (`x16_posthoc.py`): coin bootstrap and concentration of the picks; first-hour volume
  terciles for the 1 h entry (24 configs); the long vs short on M's strong coins.

## A. Listing-day longs

Buy every listing, no exits (mean / median net per trade, all 397):

| entry after | 4 h | 24 h | 72 h | 7 d |
|---|---|---|---|---|
| first 5m candle | -5.1% / -6.1% | -6.3% / -8.9% | -9.5% / -16.3% | -12.9% / -21.8% |
| 1 h | -2.0% / -2.9% | -3.6% / -6.7% | -6.7% / -13.3% | -9.9% / -17.8% |
| 4 h | -1.9% / -2.5% | -3.2% / -5.9% | -5.6% / -12.2% | -9.2% / -16.4% |
| 24 h | -2.1% / -2.3% | -3.7% / -6.6% | -5.1% / -9.6% | -8.4% / -15.7% |

Every "all listings" row of the grid (128, with any TP/stop) is negative in 2021-24 (best -0.8%).
Of the 608 pickable rows, none has a mean >= +10%. 105 are positive in 2021-24 and 9 in both periods,
all of them small subsets. The rows that look strong (5m entry on the r1h > 15% listings: +36..+44% a
trade) use the first hour's return, which is only known after the entry. Entered once that return is
known (1 h), the same 30 listings make +6.6% (24 h, no exits): the move is over within the first hour.

Splits at entry. First-hour return, enter at 1 h, hold 24 h, no exits: r1h <= -15% -8.8% (64), -15..0
-3.5% (212), 0..+15% -3.7% (91), > +15% +6.6% (30). Only the top bin is positive, and it fades by
72 h (7 d: -11.1%). First-day volume at the 24 h entry: < $100M -2.3%, 100-400M -5.5%, > $400M -10.2%
(72 h hold). The busiest listings fall the most after day 1.

**A pick: entry 1 h, r1h > +15%, hold 24 h, TP 50%, stop 30%.**

| | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | 2021-24 | 2025-26 |
|---|---|---|---|---|---|---|---|---|
| mean net (n) | +31.0% (7) | +25.7% (4) | -6.2% (4) | +9.4% (5) | -14.9% (8) | +9.6% (2) | +15.0% (20) | -2.6% (10) |

Mean of all 30 trades: +8.1% [coin bootstrap 95%: -3.4%, +20.0%], median -5.1%. In 2021-24 alone the
interval is +3.5..+30.9%, but that is the period it was picked in, out of 608 rows. The top 3 trades (all 2021
TP hits) make 62% of its P&L.

## B. Strong-listing longs (above the listing price at 24 / 48 / 72 h)

Averaged over the 36 exit/hold configs of each cell (mean of per-year means):

| check | above > 0 | 0-20% | 20-50% | > 50% |
|---|---|---|---|---|
| 24 h: 2021-24 / 2025-26 | -2.2% / -13.1% | +0.9% / -5.6% | -9.4% / -22.2% | -10.3% / -21.4% |
| 48 h | +1.2% / -11.8% | +3.7% / -6.7% | -0.6% / -11.0% | +1.0% / -27.2% |
| 72 h | +2.1% / -6.2% | +3.5% / +6.8% | -3.1% / -5.0% | +1.9% / -31.7% |

The further above the listing price, the worse, which is the opposite of a squeeze edge. Only 2 of 432
rows have a positive median trade. 23 rows reach +10% in 2021-24, and they average -16.4% in 2025-26
(none of them reach +10% there). The top decile goes from +13.6% to -15.0%.

**B pick: check 72 h, above > 0, hold 30 d, no TP, no stop.**

| | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | 2021-24 | 2025-26 |
|---|---|---|---|---|---|---|---|---|
| mean net (n) | +11.4% (24) | +84.7% (8) | +26.6% (12) | +6.6% (23) | -5.1% (15) | -34.4% (9) | +32.3% (67) | -19.8% (24) |

All 91 trades: mean +11.4% [-8.5%, +37.1%], median -16.8%, win rate 34%. Clipped to +-100% (diag's
E_o) it is -0.049. The one cell positive in both periods in all 12 of its exit configs (72 h, above
0-20%, 30 d: about +19% / +2%, n = 40) has the same shape: median -15%, and three trades make 169% of its P&L.

### Scorecards (diag)

| strategy | n | E_o | median_o | win | tp_win | time_win | near_miss | never_went | went_against | stopped | contrib_stopped | tp_prog | giveback | worst yr E_o | yrs + |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A pick | 30 | +0.081 | -0.051 | 0.43 | 0.37 | 0.07 | 0.03 | 0.07 | 0.27 | 0.20 | -0.061 | 0.48 | 1.00 | -0.149 | 4/6 |
| B pick | 91 | -0.049 | -0.168 | 0.34 | 0.00 | 0.34 | 0.40 | 0.18 | 0.09 | 0.00 | 0.000 | 1.00 | 1.00 | -0.344 | 2/6 |
| C long, mirror exit | 58 | -0.022 | -0.186 | 0.45 | 0.38 | 0.07 | 0.00 | 0.05 | 0.19 | 0.31 | -0.156 | 0.48 | 1.00 | -0.380 | 3/6 |
| buy all at 1 h, 7 d | 397 | -0.118 | -0.178 | 0.24 | - | 0.24 | 0.41 | 0.27 | 0.09 | 0.00 | 0.000 | 0.83 | 1.00 | -0.202 | 0/6 |

### Calibration by the split variable

* A pick, signal = rank of r1h (pool = all 397 at the same entry and exits): deciles 1-9 range from
  -0.065 to +0.016, the top decile +0.064 (33% TP, 21% stopped). Spearman +0.12. There is a step at the top but no slope.
* The same config by first-day volume (ex post): top decile +0.168, the rest -0.08..+0.03. This is a
  mechanical effect: a listing that rallies on day 1 trades more. The volume that is known at the
  1 h entry (first-hour volume, post-hoc) points the other way: top tercile -3.5% vs low -0.8% (24 h,
  TP 50%, stop 30%), and -19.1% in 2025-26.
* B pick, signal = rank of the price vs the listing price at 72 h (pool = all 397 longed at 72 h for
  30 d): every decile is negative (E_o -0.32..-0.002). The entered top three are -0.006, -0.108,
  -0.084. Spearman +0.14 comes from the deepest fallers being worse, not from the strong ones being good.

## Books (10 slots, $10k each year)

| book | trades | bp/trade | worst year | worst DD | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|---|
| LMC+ (2022+) | 485 | +1145 | +62% | -17% | - | +62% | +81% | +152% | +790% | +111% |
| A pick alone | 30 | +806 | -12% | -16% | +23% | +11% | -3% | +4% | -12% | +2% |
| LMC+ + A pick | 508 | +1099 | +77% | -17% | - | +79% | +77% | +163% | +691% | +115% |
| B pick alone | 91 | +1137 | -28% | -28% | +13% | +58% | +25% | +13% | -12% | -28% |
| LMC+ + B pick | 520 | +1176 | +63% | -19% | - | +157% | +132% | +142% | +732% | +63% |
| M alone (2022+) | 135 | +1896 | +16% | -16% | - | +16% | +28% | +19% | +349% | +18% |
| M + C long (mirror exit) | 189 | +1283 | -4% | -21% | - | +11% | +33% | +27% | +361% | -4% |
| LMC+ + C long (mirror exit) | 534 | +1025 | +56% | -16% | - | +56% | +88% | +158% | +872% | +76% |
| M + C long (30 d, no exits) | 184 | +1263 | -10% | -22% | - | +10% | +70% | +32% | +234% | -10% |
| LMC+ + C long (30 d, no exits) | 507 | +1006 | +54% | -16% | - | +54% | +139% | +161% | +553% | +63% |

The B pick's lift to LMC+ in 2022-23 (+62% -> +157%) is one trade (GMT, +815%). In the read years it
costs LMC+ 58 points in 2025 and 48 points in 2026. The A pick raises LMC+'s worst year from +62% to +77%
(2022 +17 points, 2024 +11), but costs 99 points in 2025 and runs 30 trades in six years. It is
not significant (bootstrap interval includes 0) and is negative in 2025-26. Neither addition is recommended.

## C. Pairing M with a long on its strong complement

M's 58 strong listings, long on spot at M's own entry time, 30 d, TP 50%, stop 50% (post-hoc split by
how the long ended):

| long exit | n | long mean | M short mean, same coins | pair (2 stakes) |
|---|---|---|---|---|
| TP +50% (the squeezes) | 22 | +49.6% | -50.8% | -0.6% |
| stop -50% | 18 | -50.4% | +46.9% | -1.8% |
| 30 d time | 18 | -17.2% | +16.3% | -0.4% |
| all | 58 | -2.2% | +0.3% | -0.9% |

The long catches exactly the 22 squeezes that stop M out and gives it all back on the 18 coins that
fall 50%. The pair nets to the cost. Adding the long next to M (it does not replace M's skipped strong
trades, which M already avoids) lowers M's worst year from +16% to -4% / -10%, and LMC+'s from +62% to
+56% / +54%.

## Why it failed

1. Listings fall from the first candle. The first 5m close is near the top: 97% of the listings that end
   at or below their listing price at 72 h made their 72-hour high in the first day. The price then drifts
   down steadily (median -18% by day 3, -24% by day 7; `results/why/listings.md`: hype and volume fade,
   and, speculatively, airdrop/launchpool supply). A
   long entered on listing day is on the side of the M edge. Every entry delay and hold loses
   unconditionally, and longer holds lose more (-2% at 4 h, -10% at 7 d).
2. Strong listings are not squeezes in progress. They are pumps already fading. At +72 h a listing
   > 50% above its price has already peaked: forward median -10% after 1 d, -17% after 3 d, -27% after
   30 d. 74% fall 30% below the entry within 30 d, while 30% also touch +50%. The 0-20% group is
   closest to flat (30 d median -15%, mean +11% from a few doublers). The "squeeze risk" that
   hurts M is two-sided volatility, not positive drift, so the long side inherits a negative median
   and a lottery-ticket right tail.
3. The in-sample winners are right-tail draws. GMT (STEPN) 2022, C98 2021 and TIA 2023 rose 2.5-8x in
   their first month and account for more than the whole P&L of the B pick. The 2025-26 listings, which
   mostly had perps from day 1 (97% in 2025), had no such run (B pick -5%, -34%). Selection across 432
   B rows turns +13.6% in-sample into -15.0% out of period. The A pick rests on 2021 (+31%, three TP hits).
4. Small-n momentum on day 1 is the only positive structure (r1h > +15% then hold <= 24 h), about 5 a
   year. It has no slope below the top decile and reverses by 72 h. Even if it were real it would add
   about 5 trades a year at +8%, far below +10% a trade and +50% a year.

Leads not taken (cheap to test but unlikely to change the verdict): a short-only use of the
first-hour volume tercile (high first-hour volume -> -19% a trade over 24 h in 2025-26 as a long) as an
extra M-style filter; the GMT/TIA-type month-1 runners are the only large long payoffs, and nothing
in the data known at entry ranks them (B calibration flat to negative).
