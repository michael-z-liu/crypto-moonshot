# Why short crowding (low funding, low ls_acc) lifts candidate longs but not the 10-slot book

Script: `w12_crowding_longs.py` (spot paths, day5 pct >= 0.995, TP 20%, 72h, cost 0.4%; unconstrained candidates n=1883
from 25 flush days; books via `simulate_yearly`, 10 slots, fresh $10k per year; base reproduces n=162, +1064 bp, years
+26/+40/+103/+16/+15%). `rate` = last funding rate, `ls_acc` = account long/short ratio, `oi_ch4h` = 4h open-interest
change; 5% (rate) to 13% (ls_acc) are missing (kept in the filter variants). Evidence tables are below the summary.

## Summary of causes

1. **Crowding is mostly a day property, and the candidate-level edge is a between-day edge.** 55% of the variance of
   `rate`, 46% of `oi_ch4h` and 31% of `ls_acc` is between days (24 days with funding data). On most flush days the
   within-day sd of the rate is 1-4 bp (funding sits near its 1 bp baseline for nearly every coin), so inside a day the
   coins are nearly tied. The raw Spearman of rate vs net is -0.20 (ls_acc -0.24), but the day-demeaned (within-day) IC is
   +0.06 for rate, -0.05 for ls_acc, +0.08 for oi_ch4h: no coin-level signal, and the rate sign is even wrong. Within-day
   terciles of the demeaned rate give low/mid/high +1224/+1160/+1488 bp (high best, TP 62/57/74%). Across days the
   rank correlation of day-mean crowding and day-mean net is -0.25 (rate) and -0.36 (ls_acc) on 16 days: about 16
   independent observations, not 1,883.
2. **A priority rule cannot use a day-level property.** It only orders candidates that share an entry bar, and every
   candidate on a flush day faces the same day. Priority acts within a day, where the IC is about zero. The slots are
   full on every large day (13 days with >10 candidates hold 1852 of 1883 candidates; booked trades enter a mean 10 min after the
   episode starts, 81% in the first 15 min), so what is chosen is which 10 of 60-335 same-day candidates, not which days.
3. **The base book is already the more-crowded half, via the days it can reach, and low-rate priority swaps good trades
   for worse ones.** Taken trades: rate -2.6 bp vs +1.5 bp for untaken (mean; medians +0.5 vs +1.0), taken ls_acc 2.81
   vs 2.73; same-day (12 days with both) taken -0.87 vs untaken +1.24 bp. The gap is driven by 2022-05-11 and
   2022-11-08 (mean rates -5 and -18 bp), the only days with strongly negative funding. Prio-low-rate keeps 76 of 162
   trades, drops 86 (+1401 bp unconstrained) and adds 84 (+1105 bp); the replaced trades have lower day5 pct (0.9962 vs
   0.9979, so the primary signal is given up) and the same timing (mean 10.5 vs 10.2 min after the episode start,
   81% in the first 15 min in both). It shifts coins, not entries to later candles. 73 coins appear in both books.
4. **The unconstrained "crowding effect" is reweighted by flush-day size.** Large days (2024-2025) contribute 100-335
   candidates each, and the funding terciles sort days more than coins. The candidates it favours are not the 10 the book
   can take, so the book can only inherit the within-day part, which is nil or negative here.
5. **The hard filters mainly delete days and slots.** rate < 1e-4 keeps 900 of 1883 candidates (+1460 bp, TP 74%) yet
   the book falls to +967 bp / 146 trades: 2023 drops 22 to 16 trades and +40% to +27%, 2022 +26% to +16%, 2024 +103% to
   +81%. A tighter filter keeps the days' best ten only if the survivors on a day still number >= 10; where fewer than
   10 survive, slots stay empty (same mechanism as `early_slot_cap.md`: 82% of P&L is in the first 30 min and a slot
   freed at minute 15 is mostly not refilled). ls_acc < 1.5 raised the bp (+1106) only through 2025 (+16% to +48%, one
   day: 2025-10-10 has the lowest mean ls_acc, 1.80, and is already a slot-allocation effect, see `cascade_wait.md`); it
   lost in 2022/2023/2024 (+26/+40/+103% to +16/+31/+61%).
6. **Effect size inside a day is small relative to ties and noise.** The rate is a clamped regime variable (~1 bp)
   for most days, so there is little dispersion to rank on; ls_acc has 31% day variance but a within-day IC of -0.05.
   

## What would have to be true for crowding to help

- Large within-day dispersion of crowding together with a within-day IC of roughly -0.1 or stronger, so that the 10 taken
  slots can be chosen on it. Observed: within-day sd 1-4 bp, IC +0.06.
- Or candidates that outnumber slots across days, so crowding picks the days: here flush days almost always have >10
  candidates, so slots are never the day filter. Skipping days needs a day-level rule (day-mean rate < threshold)
  and that rule has ~16 observations behind it and would cut flush days from the book (speculation: which ones).
- Or crowding that is cross-sectionally informative on the 2022 type days (waterfalls); the two days with strong
  negative funding (2022-05-11, 2022-11-08) net +1432 and +436 bp, i.e. no better than average.

Not tested: a day-level gate (skip flush days) or a coin-count-weighted stake; no rescue attempted.

## A. Day vs coin

| measure | n | days | share of variance between days | within-day std / total std | mean #cands/day | within-day Spearman IC vs net (day-demeaned ranks) | IC raw |
|---|---|---|---|---|---|---|---|
| rate | 1746 | 24 | 55% | 0.67 | 73 | +0.057 | -0.199 |
| ls_acc | 1682 | 24 | 31% | 0.83 | 70 | -0.045 | -0.244 |
| oi_ch4h | 1707 | 24 | 46% | 0.74 | 71 | +0.081 | +0.184 |

Day-level (days with >=5 cands: 16): Spearman(day mean rate, day mean net) = -0.25; Spearman(day mean ls_acc, day mean net) = -0.36.

Within-day terciles of the day-demeaned rate (low = more crowded shorts than same-day peers):

| tercile | n | net bp | TP share |
|---|---|---|---|
| low | 582 | +1224 | 62% |
| mid | 582 | +1160 | 57% |
| high | 582 | +1488 | 74% |

## B. Taken vs untaken candidates (base book)

| group | n | net bp | rate bp/8h mean | rate median | ls_acc mean | oi_ch4h mean | TP share | mean min after episode start |
|---|---|---|---|---|---|---|---|---|
| taken | 155 | +1053 | -2.60 | +0.50 | 2.81 | +0.024 | 61% | 10 |
| untaken | 1728 | +1312 | +1.51 | +1.00 | 2.73 | -0.021 | 65% | 17 |

Same-day pairs (12 days with both taken and untaken): mean rate taken -0.87 vs untaken +1.24 bp; ls_acc 2.92 vs 2.65.
Candidate days 25; days with >10 candidates 13 (hold 1852 of 1883 candidates); days with <=10 candidates 12 (all taken unless slots busy: taken 31 of 31).

Crowding by entry time inside the episode (all candidates):

| window | n | rate bp/8h | ls_acc | net bp |
|---|---|---|---|---|
| first 15 min | 1026 | +1.75 | 2.82 | +1420 |
| later | 857 | +0.54 | 2.64 | +1135 |

Per flush day (all candidates; rate_sd = within-day std):

| day | cands | taken | mean rate bp | within-day sd bp | ls_acc | net bp |
|---|---|---|---|---|---|---|
| 2022-05-11 | 60 | 12 | -5.29 | 18.71 | 2.27 | +1432 |
| 2022-05-12 | 2 | 2 | -75.00 | 0.00 | 4.25 | -4020 |
| 2022-05-13 | 1 | 1 | -75.00 | 0.00 | 0.81 | +1960 |
| 2022-06-08 | 1 | 1 | +48.61 | 0.00 | 0.60 | +1960 |
| 2022-11-08 | 9 | 9 | -18.29 | 35.00 | 2.88 | +436 |
| 2023-06-10 | 58 | 10 | +0.86 | 0.60 | 2.82 | +1049 |
| 2023-08-17 | 79 | 10 | -0.46 | 4.34 | 2.63 | +1619 |
| 2023-12-11 | 1 | 1 | +1.86 | 0.00 | 3.87 | +1145 |
| 2024-01-03 | 185 | 10 | +3.25 | 1.73 | 3.54 | +937 |
| 2024-03-05 | 170 | 10 | +9.21 | 3.63 | 3.45 | +1775 |
| 2024-04-12 | 213 | 11 | +1.03 | 3.56 | 3.47 | -787 |
| 2024-04-13 | 25 | 0 | +0.72 | 0.59 | 2.43 | +1817 |
| 2024-06-07 | 60 | 10 | +1.41 | 1.05 | 3.40 | +371 |
| 2024-06-18 | 6 | 6 | +0.31 | 0.73 | 2.56 | +1231 |
| 2024-08-05 | 132 | 10 | -0.26 | 1.64 | 2.09 | +1893 |
| 2024-12-03 | 1 | 1 | +nan | 0.00 | nan | +1960 |
| 2024-12-09 | 267 | 11 | +0.82 | 0.76 | 3.09 | +1869 |
| 2025-02-03 | 244 | 10 | -0.30 | 2.80 | 2.31 | +1767 |
| 2025-09-22 | 1 | 1 | -0.91 | 0.00 | 2.81 | -34 |
| 2025-10-10 | 335 | 10 | +0.29 | 1.70 | 1.80 | +1635 |
| 2025-11-21 | 1 | 1 | -0.48 | 0.00 | 1.40 | +499 |
| 2026-02-06 | 24 | 10 | -0.83 | 1.60 | 1.33 | +1898 |
| 2026-06-04 | 6 | 6 | +0.41 | 0.11 | 1.14 | -674 |
| 2026-06-05 | 1 | 1 | +0.67 | 0.00 | 0.69 | +1960 |
| 2026-10-13 | 1 | 1 | -7.88 | 0.00 | 0.54 | -1810 |

## C. Book variants (10 slots, fresh $10k per year, simulate_yearly)

| variant | trades | bp | years |
|---|---|---|---|
| base (prio = day5 pct) | 162 | +1064 | 2021:   - | 2022:   26t   +972bp    +26% | 2023:   22t  +1655bp    +40% | 2024:   74t  +1061bp   +103% | 2025:   22t   +745bp    +16% | 2026:   18t   +873bp    +15% | maxDD -14% |
| prio low rate | 160 | +929 | 2021:   - | 2022:   26t   +856bp    +22% | 2023:   22t  +1347bp    +32% | 2024:   71t   +856bp    +74% | 2025:   23t   +885bp    +20% | 2026:   18t   +873bp    +15% | maxDD -16% |
| prio low ls_acc | 158 | +928 | 2021:   - | 2022:   26t   +424bp    +11% | 2023:   22t  +1383bp    +33% | 2024:   70t  +1032bp    +95% | 2025:   22t   +851bp    +19% | 2026:   18t   +789bp    +14% | maxDD -15% |
| filter rate<1e-4 | 146 | +967 | 2021:   - | 2022:   22t   +793bp    +16% | 2023:   16t  +1608bp    +27% | 2024:   68t   +943bp    +81% | 2025:   22t   +830bp    +18% | 2026:   18t   +873bp    +15% | maxDD -12% |
| filter ls_acc<1.5 | 139 | +1094 | 2021:   - | 2022:   21t   +841bp    +16% | 2023:   18t  +1605bp    +31% | 2024:   55t   +966bp    +61% | 2025:   27t  +1412bp    +42% | 2026:   18t   +789bp    +14% | maxDD -12% |

## D. What a priority rule changes

| book | n | mean off (min) | share first 15 min | mean rate bp | mean ls_acc | uncon. net bp of taken | n distinct days | mean stake $ |
|---|---|---|---|---|---|---|---|---|
| base (prio = day5 pct) | 155 | 10.2 | 81% | -2.60 | 2.81 | +1053 | 24 | 1205 |
| prio low rate | 155 | 10.5 | 81% | -4.72 | 2.68 | +901 | 24 | 1142 |
| prio low ls_acc | 154 | 10.1 | 81% | -2.37 | 1.83 | +901 | 24 | 1177 |
| filter rate<1e-4 | 143 | 9.7 | 80% | -5.62 | 2.73 | +947 | 22 | 1151 |

Prio-low-rate vs base: 76 common trades, 86 dropped, 84 added. Dropped (unconstrained) +1401 bp, added +1105 bp; same-coin overlap: 73 coins in both.
Dropped: mean off 8.8 min, day5 pct 0.9979; added: mean off 9.4 min, pct 0.9962.
Entry timestamps: 68 distinct for 1883 candidates; timestamps with >1 candidate: 40 (covering 1855 candidates). Priority only orders candidates sharing an entry bar.

## E. diag scorecard rows

| strategy | n | E_o | median_o | win | share_tp_win | share_time_win | share_near_miss | share_never_went | share_went_against | share_stopped | contrib_stopped | tp_progress_med | stop_progress_med | giveback_med | worst_year_E_o | years_positive |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| book: base (prio = day5 pct) | 162 | +0.106 | +0.196 | 0.81 | 0.60 | 0.20 | 0.07 | 0.10 | 0.02 | 0.01 | -0.006 | 1.00 | 0.08 | 0.16 | +0.074 | 1.00 |
| book: filter rate<1e-4 | 146 | +0.097 | +0.196 | 0.76 | 0.59 | 0.17 | 0.10 | 0.11 | 0.02 | 0.01 | -0.007 | 1.00 | 0.06 | 0.19 | +0.079 | 1.00 |
| book: prio low rate | 160 | +0.093 | +0.196 | 0.77 | 0.54 | 0.23 | 0.09 | 0.11 | 0.02 | 0.01 | -0.006 | 1.00 | 0.08 | 0.20 | +0.086 | 1.00 |
| all candidates (unconstrained) | 1883 | +0.129 | +0.196 | 0.85 | 0.65 | 0.20 | 0.05 | 0.09 | 0.01 | 0.00 | -0.001 | 1.00 | 0.06 | 0.15 | +0.107 | 1.00 |
| candidates rate<1e-4 (unconstrained) | 900 | +0.146 | +0.196 | 0.89 | 0.74 | 0.15 | 0.05 | 0.05 | 0.01 | 0.00 | -0.001 | 1.00 | 0.07 | 0.12 | +0.106 | 1.00 |
