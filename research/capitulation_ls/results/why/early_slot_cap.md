# Why capping early-episode long trades leaves slots empty and costs P&L

Script: `w7_early_cap.py` (own copy of the `e25.book` loop that logs each booked trade; reproduces the uncapped long book
n=156, +1051 bp, years +24/+39/+93/+16/+15%, and the capped (15 min, 3) n=102, +954 bp, +9/+12/+32/+53/-0%).
Candidates = `sleeve_long()` (day5 pct >= 0.995, TP 20%, 72h): 1883 candidate trades. Episode = gap >= 2h between entry
times (31 episodes; w6 used signal bars, 28). `off` = minutes from the episode's first entry. 10 slots, fresh $10k per year.

## Summary of causes

1. **The long sleeve is almost entirely early-episode.** Of 1883 candidates, 54% enter in the first 15 min (60% of summed
   net) and 82% in the first 30 min (98% of summed net). In the booked 156 trades the first 15 min hold 100% of 2023 and
   2026 P&L, 88% of 2024, 82% of 2025 and 45% of 2022. In 2023 and 2026 no candidate exists after minute 15 at all.
2. **Freed slots are almost never refilled.** Capping at 3 displaces 76 of the 156 booked trades and brings in 22
   replacements, all in two episodes (2025-02-03: 7, 2025-10-10: 15). 11 of the 13 episodes with a displacement have zero
   candidates after minute 15 (cap 5: 10 of 12; 30 min, cap 3: 13 of 14). A freed slot there stays empty (or is filled by
   another sleeve in the LMC book). The cap deletes trades, it does not reallocate them.
3. **The displaced trades are good ones.** Displaced: 76 trades at +1409 bp (cap 15/3), 51 at +1443 bp (15/5), 85 at +1451 bp
   (30/3). 2022, 2023, 2024 and 2026 displaced trades average +1340 to +1960 bp, close to the 1960 bp TP-minus-cost cap
   (+20% TP less 0.4% cost). The first ones in an episode are the highest-priority signals, which is why they are ranked first.
4. **2025 improves only through two episodes.** Displaced 2025 trades: 12 at +943 bp (cap 15/3), replaced by 22 at +1839 bp
   (booked sum of net +40463 vs +11317 bp points). 2025-10-10 alone: displaced 5 trades at -203 bp, replaced by 15 at
   +1880 bp, from 327 later candidates on 195 coins. That is the slot-allocation effect from `cascade_wait.md`, now
   arrived at through a cap. Every other year is displacement without replacement: sums of displaced net 2022 +13720 bp
   points, 2023 +24829, 2024 +41545, 2026 +15708, replacements 0.
5. **Replacements are better only where the early trades were a waterfall.** Mean replacement +1770..+1839 bp vs displaced
   +1409..+1451 bp overall, but that gap is the 2025-10-10 gap (early -203/-361 bp, late about +1880 bp). In the other
   episodes with no late candidates the comparison is empty.
6. **A 30-minute window makes the 2024 loss worse.** Uncapped 2024 booked: first 30 min +1464 bp (n=59), later +-1375 bp
   (n=10, -1932$): the late trades in 2024 are poor (candidates after minute 30 in 2024: +-676 bp mean, n=222), so a
   later slot is not a safe one. 2024 return drops +93% to +17% with the 30-min cap vs +33% with the 15-min cap.

## Evidence

### A. Share of long-sleeve trades and P&L in the first W minutes (unconstrained candidates)

| year | n | first 15 min n | share of sum(net) | bp early | bp late (n late) | first 30 min share of sum(net) |
| --- | --- | --- | --- | --- | --- | --- |
| 2022 | 73 | 65 (89%) | 87% | +1151 | +1366 (8) | 87% |
| 2023 | 138 | 138 (100%) | 100% | +1376 | none | 100% |
| 2024 | 1059 | 567 (54%) | 69% | +1371 | +724 (492) | 113% (late -676 bp, n=222) |
| 2025 | 581 | 224 (39%) | 38% | +1668 | +1697 (357) | 81% |
| 2026 | 32 | 32 (100%) | 100% | +1302 | none | 100% |
| all | 1883 | 1026 (54%) | 60% | +1420 | +1135 (857) | 98% |

Booked (10 slots), first-15-min share of P&L in dollars: 2022 45% (n 17 of 25), 2023 100% (22/22), 2024 88% (49/69),
2025 82% (20/22), 2026 100% (18/18). Booked mean bp early vs late (15 min): 2022 +729/+1366, 2024 +1386/+234, 2025 +694/+1247.

### B. Capped vs uncapped

| case | n | bp | years | displaced (n, bp) | replacements (n, bp) | episodes displaced / refilled |
| --- | --- | --- | --- | --- | --- | --- |
| uncapped | 156 | +1051 | +24/+39/+93/+16/+15% | | | |
| 15 min, 3 slots | 102 | +954 | +9/+12/+32/+53/-0% | 76, +1409 | 22, +1839 | 13 / 2 |
| 15 min, 5 slots | 119 | +968 | +13/+20/+47/+38/+7% | 51, +1443 | 14, +1770 | 12 / 2 |
| 30 min, 3 slots | 98 | +918 | +9/+12/+17/+63/-0% | 85, +1451 | 27, +1825 | 14 / 1 |

Replacements all fall in 2025 (22 / 14 / 27); in 2022, 2023, 2024 and 2026 replacement count is 0 in all three cases.
Episodes where displaced trades were not replaced include 2022-05-11, 2023-06-10, 2023-08-17, 2024-01-03, 2024-03-05,
2024-08-05, 2026-02-06 (displaced mean +1600 to +1960 bp each).

## What blocks it, and what would have to be true

- **Blocks:** the sleeve is a few short bursts (median episode under 20 min) whose later signals are rare. Of 31 episodes
  only 3 have candidates after minute 15 in meaningful number (2025-02-03: 30, 2025-10-10: 327, plus 2022/2024 episodes
  with 8 and 492 total). The reserve is therefore empty in most episodes, and the cap trades a +1400 bp trade for idle cash.
- **What would have to be true** for a cap to pay: (a) most episodes keep producing signals after the cap window, (b) those
  later signals are at least as good as the displaced ones, and (c) early trades are worse than late ones on average. Here
  (a) holds in 2 of 13 displaced episodes, (b) holds only in 2025 (+1839 vs +943; in 2024 post-30-min entries are negative),
  (c) holds only on the waterfall day 2025-10-10. One episode carries the whole gain, the same single-day dependence seen in
  `cascade_wait.md`, so a cap tuned on it would be fit to that day.
- **Speculation (untested):** a cap conditional on signal count growing fast (a waterfall in progress) might avoid the
  idle-slot cost, but with 3 slow-waterfall episodes (2022-05-12, 2022-11-08, 2025-10-10) it cannot be validated.
