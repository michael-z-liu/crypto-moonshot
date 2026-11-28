# Why a "wait W minutes into the cascade" rule fails on capitulation longs

Scripts: `w6_a.py` (episodes, replication of the W sweep), `w6_b.py` (1m equal-weight index low per episode),
`w6_c.py` (episode table, book trades by episode), `w6_d.py` (signal outcome vs entry time relative to the low, W sweep
without 2025-10-10). Rule everywhere: day5 pct >= 0.995, 2022-2026, TP +20%, exit 72h, 10 slots, cost 0.4%, spot
paths, a fresh $10k book per year (`simulate_yearly`). Episode = q>=0.995 signals chained with gaps < 2h; "off" = minutes
from the episode's first signal bar to a signal's bar; entry is 5 minutes later. "Index" = equal-weight mean of 1m closes
of up to 80 coins of the episode, relative to the close before the first signal; its minimum is "the low". The W=0..60
results of the failed test were reproduced exactly (n=162/131/89/108/83/51). "Speculation" is labelled.

## Summary of causes

1. **There are only 28 episodes in five years and half of them are over before W.** 4224 signals form 28 episodes
   (6/3/10/5/4 per year). 16 of 28 last under 20 minutes (median length 12.5 min, 75th percentile 85 min). The signal
   count is dominated by a few days: 2024 has 1779 signals and 2025 has 1970, 2023 has 190 and 2026 has 37.
2. **Skipping years is mechanical.** All 2023 episodes last 15 minutes or less (the two real ones: 15 and 5 min), so
   W >= 20 leaves 0 of 190 signals in 2023 (W=10: 10). In 2026, 37 signals, only 3 survive W=10 and 2 survive W=20.
   Where trades vanish there is nothing to evaluate, the year drops out of the book.
3. **The W=20 gain is 2025-10-10 and nothing else.** On that day W=20 trades 53 times at +1868 bp (W=0 trades 9 times at
   -10 bp). Remove that episode from the candidates and the sweep is W=0 +1127 bp, W=5 +1185, W=10 +1062, W=20 +870,
   W=30 +583 bp: waiting makes it worse, monotonically from W=5. The book's per-year swings (2025 +16% to +190%) are
   slot arithmetic on one day.
4. **The market low is early for most episodes, so "wait until after the low" is mostly "wait until after the
   rebound has started".** Median time from episode start to the index low is 18 min (8 of 28 within 5 min,
   15 within 20, 22 of 28 within 120). Episodes where the first-20-minute entries are before the low are 7 of 16
   big (>= 20 signals) ones, so 9 of 16 have the low inside the window and early entries are already after it.
5. **Where early entries lose, late entries lose as well.** The only losing episodes are slow waterfalls (2022-05-12
   21:15, 2022-11-08 FTX): early -4002 / -7001 bp, late -2267 / -420 bp. The signals keep firing through the whole
   fall, so no fixed W gets past the low.
6. **Entering before the low is not itself a loss.** Per signal (unconstrained, 1x), entries up to 10 minutes before the
   index low made +1649 bp and entries more than 30 min before it +1158 bp, against +1233 bp for 0-10 minutes after.
   2025-10-10 is a slot effect (cause 3 of `leverage.md`), not a timing effect: the first ten slots went to seven
   off=0 signals and two off=15 signals, all in the first deep minutes, mean MAE about 70% for the first seven.

## Evidence

### A. Episodes per year (28 episodes, 4224 signals)

| year | episodes | signals | median duration (min) | max duration | single-signal | >= 20 min | >= 60 min |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2022 | 6 | 248 | 192.5 | 370 | 1 | 4 | 4 |
| 2023 | 3 | 190 | 5.0 | 15 | 1 | 0 | 0 |
| 2024 | 10 | 1779 | 17.5 | 95 | 1 | 5 | 3 |
| 2025 | 5 | 1970 | 0.0 | 205 | 3 | 2 | 1 |
| 2026 | 4 | 37 | 5.0 | 25 | 1 | 1 | 0 |

Duration (first to last signal) over all 28: quartiles 0 / 12.5 / 85 min, 90th percentile 205 min, median 31.5
signals per episode. Episodes that still have a signal at off >= W: W=10 16, W=15 14, W=20 12, W=30 10, W=60 8 (of 28).
Signals left: 4224 -> 2908 -> 2623 -> 1964 -> 1493 -> 938.

Signals surviving by year:

| year | W=0 | W=10 | W=20 | W=30 |
| --- | --- | --- | --- | --- |
| 2022 | 248 | 240 | 235 | 232 |
| 2023 | 190 | 10 | 0 | 0 |
| 2024 | 1779 | 1059 | 672 | 619 |
| 2025 | 1970 | 1596 | 1055 | 642 |
| 2026 | 37 | 3 | 2 | 0 |

### B. The 16 big episodes (>= 20 signals): when is the low

Columns: day (UTC start), signals, coins, last signal off (min), index low (min from start), index low (% vs the close
before start), mean unconstrained net (bp) of signals with off < 20 / >= 20 (null = none).

| start | sigs | coins | dur | low at | low % | early bp | late bp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2022-05-11 10:45 | 131 | 57 | 265 | 642 | -34 | +1960 (1) | +1592 |
| 2022-05-12 08:40 | 51 | 1 | 370 | 374 | -97 | +1960 (1) | +1480 |
| 2022-05-12 21:15 | 37 | 2 | 280 | 287 | -91 | -4002 (4) | -2267 |
| 2022-11-08 18:05 (FTX) | 26 | 9 | 120 | 86 | -36 | -7001 (4) | -420 |
| 2023-06-10 04:20 | 109 | 57 | 15 | 7 | -20 | +1075 | none |
| 2023-08-17 21:40 | 80 | 79 | 5 | 4 | -16 | +1623 | none |
| 2024-01-03 12:05 | 290 | 184 | 10 | 5 | -18 | +790 | none |
| 2024-03-05 19:50 | 186 | 170 | 45 | 6 | -14 | +1769 | +1960 (1) |
| 2024-04-12 17:25 | 528 | 210 | 95 | 70 | -20 | +1960 (1) | -791 |
| 2024-04-13 20:05 | 239 | 112 | 85 | 84 | -14 | +1840 | +1773 |
| 2024-06-07 18:00 | 92 | 60 | 10 | 14 | -15 | +399 | none |
| 2024-08-05 06:15 | 154 | 131 | 25 | 10 | -7 | +1890 | +1960 (1) |
| 2024-12-09 20:45 | 279 | 266 | 95 | 19 | -17 | +1864 | +1737 |
| 2025-02-03 01:50 | 810 | 244 | 35 | 17 | -18 | +1864 | +1960 |
| 2025-10-10 20:55 | 1157 | 203 | 205 | 26 | -65 | +1162 | +1929 |
| 2026-02-06 00:10 | 25 | 24 | 5 | 9 | -10 | +1900 | none |

(2022-05-12 08:40 and 21:15 are 1 and 2 coins, i.e. the LUNA/UST-type serial crashes, where the "index" is one or two
coins. Smaller episodes, mostly single coins, are in `w6_c.py` output.)
Time to the low over the 28 episodes: 25th percentile 5 min, median 18, 75th 86; low within 30 / 60 / 120 min of the
start in 61% / 64% / 79%. For single-coin episodes the "low" is a coin drifting for hours and means little. The last signal
falls within about 45 minutes of the low in 13 of the 16 big episodes (median about 12 min after it; counted from the
table above); the exceptions are 2022-05-11 (low 6 hours after the last signal), 2024-12-09 (+81) and 2025-10-10 (+184).

### C. Signal outcome vs entry time relative to the index low (16 big episodes, unconstrained 1x, TP20/72h, 0.4%)

| entry relative to low | signals | episodes | mean net bp | share reaching TP | mean MAE |
| --- | --- | --- | --- | --- | --- |
| > 30 min before | 388 | 6 | +1158 | 87% | 17% |
| 10-30 min before | 225 | 6 | +1510 | 80% | 17% |
| 0-10 min before | 782 | 12 | +1649 | 77% | 26% |
| 0-10 min after | 2171 | 14 | +1233 | 66% | 8% |
| 10-30 min after | 490 | 7 | +1052 | 69% | 12% |
| > 30 min after | 138 | 4 | +1752 | 88% | 4% |

Without 2025-10-10 the 10-30 min-after bucket is 173 signals at -608 bp (the FTX/May-2022/April-2024 tails), the rest
barely move. Weighting each episode equally the "before" buckets a and b are -443 / -446 bp, driven by six episodes in which
a few early signals sit in the slow 2022 waterfalls; the later buckets are +1370 to +1837 bp but have 4-14 episodes.
Pairing early (< 20 min) with late (>= 20 min) signals inside the same episode (12 episodes have both): late is better
in 6 of 12; mean difference +471 bp, median +35 bp; without 2025-10-10 (11 episodes) mean +444 bp, median 0, better in 5.
The mean is the 2022 waterfall episodes (early -4002 / -7001 bp), not a general pattern.

### D. Book results: W sweep, with and without the 2025-10-10 episode

Year cells: trades, book return of a fresh $10k.

| W | n | bp | 2022 | 2023 | 2024 | 2025 | 2026 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 162 | 1064 | 26t +26% | 22t +40% | 74t +103% | 22t +16% | 18t +15% |
| 5 | 144 | 1246 | 25t +23% | 11t +15% | 63t +79% | 33t +66% | 12t +20% |
| 10 | 131 | 1184 | 25t +23% | 9t +11% | 58t +61% | 38t +81% | 1t +2% |
| 20 | 108 | 1359 | 24t +22% | none | 20t +2% | 63t +190% | 1t +2% |
| 30 | 83 | 1188 | 24t +22% | none | 13t -9% | 46t +116% | none |
| **without the 2025-10-10 episode** | | | | | | | |
| 0 | 153 | 1127 | 26t +26% | 22t +40% | 74t +103% | 13t +16% | 18t +15% |
| 5 | 121 | 1185 | 25t +23% | 11t +15% | 63t +79% | 10t +20% | 12t +20% |
| 10 | 103 | 1062 | 25t +23% | 9t +11% | 58t +61% | 10t +20% | 1t +2% |
| 20 | 55 | 870 | 24t +22% | none | 20t +2% | 10t +20% | 1t +2% |
| 30 | 43 | 583 | 24t +22% | none | 13t -9% | 6t +12% | none |

Trades by episode in the book. W=0 is spread out: 27 episodes used, the top two give 26% of the net sum, the biggest
episode has 13 trades. W=10: 15 episodes, top two 45%. W=20: 11 episodes, 2025-10-10 has 53 of 108 trades and
the top two episodes (10-10 and 2022-05-11) give 83% of the net sum. W=30: 9 episodes, 40 of 83 trades on 10-10,
top two 98%. W=30's 2022-05-12 08:40 entry pair is the one loss (-4020 bp, 2 trades).
The 2025 year at W=20: 10 trades in 2025-02-03 (+1960) and 53 on 2025-10-10 (+1868). At W=0 the 2025 book has 22 trades:
10 on 02-03 (+1633), 9 on 10-10 (-10), 3 singles (-34, -322, +499).

### E. Why W=0 on 2025-10-10 loses (trade level)

The nine W=0 book trades on that day were: seven at off=0 (net -3.1%, -1.9%, -8.5%, +19.6%, -18.9%, -5.8%, -7.1%; MAE
55-86%) and two at off=15 (+10.4%, +14.5%; MAE 79%, 93%). The ten slots are full from then on for up to 72 hours (or until
a TP), so the 1000+ later signals, mostly after the low, could not be taken. W=20 frees the slots for those
(53 trades, +1868 bp). This is slot allocation, not information in the wait.

## What blocks a cascade-timing rule

- **No sample.** 28 episodes (16 big ones) in five years, 12 with both early and late signals. A rule that conditions on the
  within-episode clock is estimated on a dozen independent events, one of which (2025-10-10) holds 53 of the 108 W=20 trades.
- **The clock is unobservable and inconsistent.** An episode's start is known (first signal), its end only 2 hours later
  by the gap definition; the low's position ranges from 4 minutes to 10 hours (median 18 min). A fixed W is either
  below the low in the fast crashes (2023, 2024: lows at 4-14 minutes, episode over by 5-15 minutes: W kills them)
  or inside the fall in the slow ones (2022-05, FTX: the low is 86-640 min away and late entries lose as well).
- **Waiting costs candidates where the edge is.** The fast episodes (2023, 2024 January/June, 2026) are the ones that are over
  in 5-15 minutes; W removes exactly these (all 2023). Per signal they were +399 to +1623 bp. What W keeps are
  long, deep episodes (10-10, 2022, 2024-04) with more variable outcomes (-2267 to +1960 bp).
- **The selection of slots, not the entry time, is the real lever.** 10-10 is the only episode where the first ten slots
  were taken at depth; elsewhere early and late entries are statistically the same (median late-early difference 0 bp
  without 10-10).
- **Per-trade bp improves because the denominator shrinks.** W=20 raises bp from 1064 to 1359 by dropping 54
  trades (n 162 to 108); the gain is in the added 10-10 trades (without them W=20 is +870 bp).

## What would have to be true

1. The within-episode clock would need to predict the low: the low would have to come at a stable offset (say 15-30 minutes)
   in most episodes. Observed spread: 4 to 642 minutes, median 18, with 36% of episodes more than an hour out.
2. Episodes would have to last long enough that a late signal exists: now 16 of 28 are under 20 minutes (all of 2023).
   A cascade rule would need many more episodes per year, or a lower threshold (for instance q >= 0.99), which changes the
   sleeve rather than rescues it (not tested here).
3. Early entries would have to lose and late entries win in more than one episode: pairs show late better in 6 of 12
   (5 of 11 without 10-10) with a median difference of 0 bp. Only 10-10 and two 2022 waterfalls carry the mean.
4. Slots would have to be the binding constraint in the early minutes in several episodes, not only on 10-10. A
   slot allocator that does not fill all slots in the first bar (for example, a cap of 3 slots in the first 15 minutes of an
   episode) is the natural test (speculation: not run here; it attacks the one failure that exists, the slot burst, without
   discarding fast episodes). The 2022 waterfalls would still lose.
5. Speculation: a mid-fall observable (breadth of the market-wide drop, rate of new signals per minute falling) might locate
   the low better than a clock. The last signal sits within about 45 minutes of the index low in 13 of 16 big episodes, but it is
   only known in hindsight, and nothing here tests a rule on the signal rate.
