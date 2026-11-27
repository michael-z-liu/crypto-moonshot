# Why the "other" ideas failed (funding extremes, market-level capitulation, calm threshold, majors, book tweaks)

Scripts: `w5_funding.py`, `w5_calm.py`, `w5_market.py`, `w5_cnn_majors.py` (needs `/tmp/w5_episodes.parquet`,
`/tmp/w5_btc*.npy` written by `w5_market.py`), `w5_squeeze.py`, `w5_squeeze2.py`, `w5_fwdscores.py`.
All numbers are net of the stated costs unless "price" or "gross" is written. Speculation is labelled.

## Summary of causes

1. **Extreme funding does not mean-revert because it is a price-trend signal, not a mispricing, and the
   carry roughly pays for the trend.** After a print >= +0.10% the next print is still >= +0.10% 58% of the
   time (46% three prints later, 36% nine prints later). A short collects only +76 bp of funding over 72 h,
   while the coin's price is right-skewed: mean price pnl of the short is -215 bp (median +12 bp), 14% of
   trades see +30% against, 1.9% see +100%. Longs after -0.30% receive +315 bp funding but the coin keeps falling
   (median price pnl -524 bp). Net: about zero, sign decided by a few tail coins and by the year's mix.
2. **Moderate market drawdowns are neither bounces nor continuations on average; the capitulation edge lives in
   the extreme-flush days only.** 49 BTC 3-day drops <= -8% (2022+): basket (20 majors ex BTC) +0.4/+1.4/+2.0% after
   3/7/14 days, 57% positive. Deeper drops tilt to continuation (-12%: +41% positive at 7 d; -15%: 25%), n small.
   Only 5 of 47 scored episodes contain a real CNN flush (>= 10 rows with pct >= 0.995); the 2026-11-14 slide was a
   mid-size flush (825 spikes, max pct 0.990, 2 rows >= 0.98), below the 0.995 bar by construction.
3. **The BTC-calm threshold "sensitivity" is mostly sample noise from 25 events plus 3 tail losses.** Calm <= 1%:
   152 trades, +420 bp (se 188). 1-2%: 25 trades, -108 bp (se 753), including a -100% (SRM during the FTX collapse),
   -52% and -50%. Without the -100% trade the 1-2% band is +305 bp. The bootstrap difference is +513 bp with a 90%
   interval of -723..+1771 bp. There is a real effect further out: 2-5% band loses (-678 bp, n=49) and > 3% is
   market crashes that bounce (spike share 0.92, -954 bp).
4. **Majors bounce no less than alts; the smaller number is a rule artefact.** With the same rule (r4 <= -10%,
   TP 10%, 24 h) ungated majors make +490 bp vs +216 bp for alts. With the CNN gate day5 >= 0.995 and TP20/72h
   majors make +1539 bp (n=129) vs alts +1272 bp (n=1754). The "+10% alts" figure comes from selecting the extreme
   flush days; majors have too few such rows (and the E29 majors sleeve had a TP 10% cap).
5. **Squeezes cluster because the loss event is a coin-specific or exchange-specific jump (thin coins, delisting
   notices, listing/unlock catalysts) that is nearly independent of BTC, plus a handful of date clusters (2025-09
   delisting, 2025-10-11 flash crash night).** Corr(net, BTC fwd 72 h) = -0.08. Loss mass is concentrated: the worst 5
   / 10 / 20 days (of 224) carry 28 / 42 / 68% of the loss mass. Weekly clustering of -30% trades is only mildly above
   independence (6 weeks with >= 2 vs 4.9 expected). A cap on the pump slots does not help because the cap is not the
   driver; the per-trade tail is (a -100% trade costs 1/10 of equity per slot regardless of how many others are open).
6. **Sleeve weights and weekday filters** only move the worst year because each sleeve has a different bad year, and
   weekday effects are statistically nothing (|t| <= 1.3 in every sleeve).

## Evidence

### 1. Extreme funding (`w5_funding.py`; perp 1h data, 2021+, one entry per coin per 3 days, 24/72 h)

Persistence (prints counted per coin, all intervals):

| condition | prints | coins | still beyond threshold after 1 / 3 / 9 / 27 prints | mean next rate after 1 / 9 prints |
| --- | --- | --- | --- | --- |
| rate >= +0.10% | 21,722 | 679 | 0.58 / 0.46 / 0.36 / 0.24 | +13.9 / +9.2 bp |
| rate >= +0.30% | 2,133 | 291 | 0.41 / 0.23 / 0.17 / 0.08 | +28.2 / +14.3 bp |
| rate <= -0.30% | 15,562 | 662 | 0.60 / 0.42 / 0.23 | n/a |
| rate <= -1.00% | 2,848 | 366 | 0.46 / 0.25 / 0.11 | n/a |

After 9 prints (about 3 days) only 29% of the >= +0.10% cases are back to <= +0.01%, 54% are still >= +0.05%. The
extremes are not rare-coin noise: 679 coins, top-10 coins hold 10% of prints. They are regime-dependent: 12,873 of
21,722 (59%) of the >= +0.10% prints fall in 2021; 2022-23 had 215 in total; 2024 1,662; 2025 2,437; 2026 4,535.
Once-in-a-regime euphoria (2021, Mar-2024, 2026 alt seasons) produces the sample, so the "year" is nearly the
experiment unit.

Price after the print, 72 h (perp, entry next hour open, funding counted separately):

| trade | n | price pnl mean / median | p5 / p95 | adverse >= +30% | funding over 72 h | net mean | net by year |
| --- | --- | --- | --- | --- | --- | --- | --- |
| short after >= +0.10% | 5,642 | -215 / +12 bp | -3405 / +2398 | 14.1% (1.9% >= +100%) | +76 bp received | -182 bp (2021) .. -352 (2025) | 2022 +25 (n=37), 2023 -86, 2024 -183, 2025 -352, 2026 -51 |
| short after >= +0.30% | 870 | -393 / -50 bp | -4255 / +2790 | 21.4% (2.9%) | +133 bp received | -319 bp | 2021 -404, 2025 -580, 2026 -5; 2022-24 n = 20 |
| long after <= -0.30% | 3,927 | -272 / -524 bp | -3045 / +2874 | 10.0% | +315 bp received | +22 bp | 2021 -833, 2022 +165, 2023 +17, 2024 +10, 2025 +189, 2026 -11 |
| long after <= -1.00% | 1,074 | -442 / -833 bp | -4328 / +4498 | 17.8% | +658 bp received | +203 bp | 2023 -76, 2024 +336, 2025 +566, 2026 -103 |

Reading: positive-funding shorts have a small positive median (the median coin does revert slightly) but a
short-side tail that costs 2-4x the funding collected. Negative-funding longs are the mirror: the crowded side is
right on price (median -5.2% in 72 h), the funding paid to longs (+142 bp median, +315 mean) is roughly the
compensation, and the net is within noise of zero (+22 bp, flips sign by year). The coin's move into the print
matters: shorts after a coin ran up > 10% in 24 h lost -368 bp (n=1,041) vs -123 bp when flat (n=4,294): extreme
funding on a runner is momentum, not exhaustion.

By BTC 7-day regime (short after >= +0.10%, net): BTC 7d > +5%: -274 bp (n=2,645); flat: -37 bp (n=2,538); < -5%:
-296 bp (n=459). Median is positive only in the flat regime (+126 bp). Squeezes (strong BTC trend either way)
are what turns the median-positive trade into a negative mean.

Dying coins: 102 shorts (1.8%) within 14 days of the perp's delisting made +763 bp mean (median +1115 bp); this is
survivorship in reverse and is not tradeable ahead of time. Excluding them the >= +0.10% short is -186 bp (n=5,540).

Concentration in time: the >= +0.30% shorts have 37% of trades on 10 of 250 trade days (2021-01..04); the
>= +0.10% sample has 14% on its 10 busiest days (2024-03-02/05: 275 trades).

### 2. Market-level capitulation and the moderate-drop slide (`w5_market.py`, `w5_cnn_majors.py`)

BTC 3-day drop episodes (first hour crossing, 7-day lock, 2022+), forward returns from that hour; "basket" = equal-weight
hourly-rebalanced 19 majors ex BTC (no costs):

| drop | episodes | BTC fwd 3d / 7d / 14d mean | basket fwd 3d / 7d / 14d mean | share basket 7d > 0 |
| --- | --- | --- | --- | --- |
| <= -8% | 49 | -0.5 / +0.2 / +1.2% | +0.4 / +1.4 / +2.0% | 0.57 |
| <= -12% | 17 | -1.6 / -0.3 / +1.9% | -0.4 / -0.2 / +3.0% | 0.41 |
| <= -15% | 8 | -0.3 / -2.1 / +0.3% | +0.1 / -1.4 / +4.3% | 0.25 |
| <= -20% | 4 | +2.9 / +1.2 / +1.4% | +8.2 / +8.9 / +12.5% | 0.75 |

Over all 1,449 overlapping hours with a 3d drop <= -8%, BTC's mean 72 h forward return is +0.02% and in 40% of cases
BTC falls a further 5 points within 72 h. That is momentum with no bounce premium, and round-trip costs (0.4%) are
the same size as the mean forward return. The -20% group (4 episodes) is the only
winner and is not independent evidence.

Slide vs flush. The 5m-flush signal (CNN pct >= 0.995) fires on only 26 days in 2022-26 (13 days with >= 30 rows;
94% of the rows are on the top 11 days). Inside the 47 scored -8% episodes: 41 contain no signal row, 5 contain
>= 10, 1 contains 1-9. Basket 7d forward: no-signal episodes +1.3% (n=41), flush episodes +1.5% (n=5): the market
drop itself carries no information; the flush has too few cases to differ. Sharper hourly drops are more favourable
only weakly: worst 1-hour BTC move <= -3% (n=19): basket 7d +3.0%, BTC +3.0%; otherwise (n=30): basket +0.3%, BTC -1.6%.
Difference about 2.7 points vs a standard error near 2.5, not significant.

Case: 2026-11-14 slide (candles to 2026-11-27; model scores `forward_signals_day5.parquet`). BTC 3-day: -5.2% (11-12),
-7.4% (11-13), -13.3% (11-14 close), then bounce to -8.5% (11-15), stall, a second leg -9.2% (11-20, worst 1 h -3.7%).
Across the 20 majors, 11-14 was a real 5m flush: 18 of 20 majors printed a >= 15x volume down bar, 63 such bars (daily
p99 over 2022+ is 49), basket -10.2% in the day. It still scored below the entry bar: 825 spikes, max pct 0.990,
2 rows >= 0.98, 95 rows >= 0.95, deepest r4 -22.9%. The next days: basket +4.1% on 11-15, then the next 9 days compound to
about -4% (BTC 64,150 -> 59,832, -6.7%) with a second flush-like day on 11-20 (pct max 0.963) and 11-23 (15 rows >=
0.98, max 0.987). So the signal was a mid-size flush that did not bounce enough: it was a slide with intraday flushes,
not an isolated 5m capitulation. (My interpretation: bounce size scales with how one-sided and brief the flush is; the CNN
percentile ranks spikes within the day, so a day whose spikes are large but broad gives no outlier.) Historical
day-level flush: 13 days with >= 30 rows >= 0.995 is the whole alts long sleeve. The -8% BTC filter would have used
49 episodes; 41 had no such days and mean basket 7d of +1.3%, with costs of 0.4% that is no edge.

Spike intensity across drops: peak-hour number of majors with a >= 15x down bar has median 2, p90 14, p99 20 on all
days; BTC 3-day-drop episodes sit at 17-20 coins in a typical case (the market-wide bar volume alone does not separate
slides and flushes). Daily total down-spike bars: p50 4, p90 21, p99 49; 11-14 had 63.

### 3. BTC-calm threshold for crash shorts (`w5_calm.py`, `w5_squeeze2.py`)

Candidates: perp, qv_base >= 20k, r4 <= -15%, day5 <= 0.5, 2022+, 24 h short, isolated 1x, funding paid.

| |BTC 4h| band | n | days | mean / median bp | win | BTC 4h mean | mkt 4h mean | spike share | delisted <30 d | young (<90 d) | p5 / p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0-0.5% | 101 | 97 | +246 / +327 | 0.60 | -0.05% | -0.3% | 0.11 | 3% | 10% | -3039 / +3218 |
| 0.5-1.0% | 51 | 49 | +764 / +560 | 0.61 | +0.07% | -0.6% | 0.15 | 6% | 22% | -3072 / +4329 |
| 1.0-1.5% | 16 | 15 | +1395 / +750 | 0.56 | +0.07% | -2.0% | 0.17 | 0% | 6% | -2806 / +6562 |
| 1.5-2.0% | 9 | 9 | -2779 / -1361 | 0.22 | -0.56% | -2.4% | 0.12 | 11% | 11% | -10000 / +2214 |
| 2.0-3.0% | 8 | 8 | +438 / +733 | 0.75 | -1.17% | -2.4% | 0.15 | 0% | 0% | -3934 / +3183 |
| > 3% | 65 | 5 | -954 / -596 | 0.25 | -4.4% | -32.9% | 0.92 | 3% | 14% | -3260 / +467 |

Summary of what enters at each width: <= 1%: coin-specific crashes (market 4h about -0.5%, spike share 0.11-0.15, 152
trades on 146 days). 1-2%: 25 trades, a bit more market-driven (market 4h -2%, share of days with market 4h <= -2%:
0.36 vs 0.12), and a few coin news cases. 2-3%: 8 trades. > 3%: five market crash days with 65 trades at spike share
0.92: bounces take the shorts (BTC and the alts rebound together), so that band loses.
The 1-2% weakness is made of three trades: SRM 2022-11-14 (-100%, FTX collapse, BTC -1.8%), FLM 2025-10-29 (-52%;
FLM was delisted), XVG 2024-12-02 (-50%, BTC +1.9%); six trades in the band exceed -28%. With them out the band is +305 bp.
Calm <= 2% total: +345 bp (n=177, +524 without the -100% trades), 2-5%: -678 bp (n=49); the "2% is much weaker"
sentence is a statement about 25 extra trades with se 753 bp. The delisted-share explanation does not hold: 0-11% in
every band, no pattern. Coin-leg hedged with 1x BTC over the hold: calm <= 1% +433 bp, 1-2% -200 bp, so BTC
direction after entry (mean -0.93% in the 1-2% band vs +0.13% in the calm band) explains part of the 1-2% band's gap;
speculation: the 1-2% band contains more days where the market is already moving down together and keeps bouncing
and falling in the next 24 h, i.e. regime noise rather than a clean coin-specific vs market-driven split.

By year (calm <= 1%): 2022 +1633 (n=11), 2023 +515 (12), 2024 +451 (22), 2025 +240 (70), 2026 +351 (37). The edge per
trade decays as n grows (2025 is 70 trades at +240 bp), consistent with a small selection effect early.

### 4. Majors vs alts (`w5_cnn_majors.py`)

Crash rows (r4 <= -8%, spot paths, 2022+), forward 24 h:

| group | r4 bucket | n | coins | days | median qv/day | mean 24 h close | mean max high 24 h | TP10 hit | close 72 h |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| majors | -10..-8% | 1,562 | 20 | 150 | $30M | +2.1% | +6.5% | 0.19 | +2.2% |
| majors | -15..-10% | 1,657 | 20 | 77 | $35M | +3.3% | +8.2% | 0.26 | +3.9% |
| majors | -25..-15% | 900 | 19 | 28 | $45M | +6.6% | +13.4% | 0.60 | +9.7% |
| majors | < -25% | 186 | 15 | 7 | $27M | +18.6% | +32.9% | 0.97 | +34.9% |
| alts | -10..-8% | 22,478 | 539 | 1,334 | $4M | +0.6% | +8.3% | 0.26 | +0.15% |
| alts | -15..-10% | 24,124 | 537 | 1,187 | $4.5M | +1.2% | +10.0% | 0.34 | +0.7% |
| alts | -25..-15% | 15,834 | 504 | 698 | $4.9M | +3.2% | +14.6% | 0.53 | +3.6% |
| alts | < -25% | 7,156 | 378 | 255 | $4.3M | +8.6% | +40.4% | 0.77 | +16.9% |

Bounce size is a function of crash depth, not of the group. Majors' crashes are shallower in the tail (p10 of r4: -22.8%
vs -29.9% for alts), and they are 5.5% of all crash rows (2,743 of 49,857), so the majors sleeve gets few deep events
and fewer days (28 days at -15..-25%). Beta: BTC 4h mean during majors' crashes -4% vs -2.5..-3.3% for alts: majors'
crashes are market-driven (and therefore keep falling together, as in section 2), alts' crashes are more idiosyncratic.

Same rule, different groups (r4 <= -10%, TP 10%, 24 h, 0.3% cost, no slot limit): majors n=423 +490 bp (every year +450..+570),
alts n=9,106 +216 bp (2022 +73, 2026 +65), alts with >= $20M/day n=1,582 +248 bp. Gated by CNN day5 (TP20/72h, spot, 0.4%
cost): q 0.98 majors n=356 +428 bp vs alts n=4,992 +468 bp; q 0.995 majors n=129 +1539 bp vs alts n=1,754 +1272 bp.
So the "majors +4.6% vs alts +10%" comparison compares different rules. The honest statement: majors' bounces are the
same size per unit of depth; they are capped by TP 10% and by having shallow crashes and few flush days, and the
extreme-flush days that pay most are the same 11-13 days for both groups.

### 5. Squeezes, pump-short book, weights, weekday (`w5_squeeze.py`, `w5_squeeze2.py`)

Pump shorts (r4 >= 40%, perp, 72 h, isolated 1x, un-regime-filtered P; 295 trades, +247 bp mean, +991 bp median, 224 days):
39 trades (13.2%) end <= -30%, on 36 days, summing -29.1 stakes vs +36.4 from the other 256 trades; 22 hit the -100% liquidation
(2022: 2 of 35, 2023: 1 of 27, 2024: 2 of 45, 2025: 9 of 119, 2026: 8 of 69). Mean bp by year: 328, 716, 212, 14, 447.

Dates of the 15 worst: UNFI 2022-06-07 (x2) and 2024-10-23, CYBER 2023-08-29, XVG 2024-12-02, VOXEL 2025-04-19, ALPACA 2025-04-24
and 04-29, SPK 2025-07-21, BAKE 2025-09-10, HIFI 2025-09-12 (x2), GIGGLE 2025-11-04, TNSR 2025-11-20, NOM 2026-03-30.
BTC context is scattered: BTC 7 d before entry from -11% to +10%, BTC 72 h after from -6.8% to +7.7%; corr(net, BTC fwd 72h) = -0.08,
corr(net, market 24h at entry) = +0.13. By BTC 7d regime at entry: BTC 7d < -3%: -554 bp (n=87, 16% of trades <= -30%);
mid: +422 bp (n=145, 14%); BTC 7d > +5%: +949 bp (n=63, 8%). This is where E34's regime filter comes from.

Clustering: worst 5 days carry 28% of the total loss mass (11% of trades), worst 10 days 42% (13%), worst 20 days 68% (18%). The
2025-09-10/12 pair (BAKE and HIFI, both flagged delisted within 30 d: 7 trades in the sample are within 30 days of a perp
delisting, mean -3,213 bp, 8% of the bad trades vs 2.4% of all) produced the book's 2025 low: the 2025 LQMC low point is
2025-09-14 (cumulative -$10.3k on $10k start), with two P trades of -$2.7k and -$2.3k on those days. Excluding
2025-10-11 the sample is unchanged (39 bad trades; +388 bp). On 2025-10-11 itself (flash crash night, 01:20-01:35 UTC)
20 pump rows entered at once: all 20 are shorts of a violent rebound, mean -17%, none liquidated, 1 winner; the regime
filter (BTC 7d) removes all 20. Weekly clustering of -30% trades: 6 weeks with >= 2 vs 4.9 expected for independent
events (lambda 0.31 over 127 weeks): not clustered more than chance at weekly resolution; the clustering is on the loss
mass (a few -100% trades) and on single nights.

Also: a trade after a pump coin had another pump in the prior 30 days: n=67 with 24% of trades <= -30%, mean -717 bp;
trades sharing the day with 2-3 other pump trades have 25% bad rate for 3 and -586 bp at 6+ (n=32).

Book tweaks (LQMC = L + regime P + C + weak-listing M, 10 slots, 2022-26 yearly $10k books; uncapped +966 bp, years
+66/+108/+186/+572/+98%, worst DD -36% in 2025):

| variant | bp | min year | worst DD |
| --- | --- | --- | --- |
| P cap 1 / 2 / 3 slots | +982 / +943 / +943 | +47 / +53 / +50% | -34 / -36 / -35% |
| L weight 1.5 / 0.5 | +946 / +966 | +59 / +54% | -36 / -36% |
| P weight 0.5 | +966 | +60% | -24% |
| N (listing) 0.5 | +966 | +55% | -33% (2025 +259%) |
| C weight 1.5 / 0.5 | +960 / +966 | +72 / +58% | -36 / -36% |

The 2025 low point is P (delisting squeezes) and N (a -$2.8k listing short on 2025-11-22); reducing a sleeve that is
fine in the bad year costs the other years, so "all weights lower the worst year" (min year is 2022-2026 depending on the
sleeve: weights of L lower 2022, N lower 2025, P lower 2026). Caps do not help since the damage is a single -100% trade
(2025 worst trades: P -2.8k, N -2.8k, P -2.7k, P -2.3k, P -2.2k).

Weekday (weekend entry vs weekday, pooled over 2022-26): L 83 trades +1281 bp vs 1,800 +1291 bp (t -0.13); Q (regime
pump) 65 +791 vs 139 +580 (t +0.40); C 46 +404 vs 115 +449 (t -0.10); M 56 +2394 vs 80 +1586 (t +1.34). Within years the sign
flips (C: weekends +1005 bp in 2025, -713 bp in 2026; Q: +1717 vs +448 in 2023, +384 vs +1333 in 2026). It is noise with n of 3-30
per cell.

## What blocks it

- Funding: the carry is small (8-10 bp per 8 h at the extremes) vs 72 h price dispersion of +-30% (p5/p95 about -3400/+2400 bp);
  extremes come in two or three regime windows (2021, 2024-03, 2026) so effective independent evidence is about 6 year-blocks, and
  the right-tail (+100% in 2-3% of trades) is not insurable on a 1x isolated perp. The negative-funding longs would need a
  filter for "price still falling" that nobody has (median price pnl -5.2%).
- Market-level capitulation: only 4 events at -20%; at -8..-15% the average basket move over 3-14 days is within +-3% and 0.4% round trip
  costs plus 40% continuation probability remove it. No lead beyond -20% exists in 4.5 years of data.
- Calm threshold: 25 events in the 1-2% band; the claim cannot be tested again with the present data (each year adds
  ~5 events). Any threshold between 0.5% and 2% is inside the noise.
- Majors: few crash days (28 days at -15..-25% in 5 years); the pay is in the same 11-13 flush days as the alts, so no diversifying
  extra evidence.
- Pump shorts: -100% losses (22 of 295) are single-coin events (delisting notices, hype on thin perps) that no feature in
  E16/E17 predicted (AUC 0.42-0.63).

## What would have to be true

- Funding mean-reverts only if extreme prints came from forced positioning (liquidations, one-off dislocations) rather than trend. The data say 58%
  persistence and positive carry correlation with price trend, so a tradable version needs a trigger that separates forced from trending crowding
  (e.g. open interest drop plus funding, or perp premium vs spot basis); E17 found OI/LS ratios add nothing for pump squeezes, so I would not expect it
  (speculation).
- Market-level capitulation would work if a flush (not a slide) could be recognised at the daily level and occurred more than 4 times in 5 years; the
  26 CNN signal days plus 453 high-volume flush days for majors (BTC fwd 3d/7d +0.3/+0.5%) say a pure volume flush on majors is not enough.
- A BTC-calm threshold would be robust if the 1-2% band had a stable sign across years; it needs more than 25 events, or a measure of "coin-specific
  news" (listings, exploit, delisting notice) not available in price data.
- Majors would match alts if the CNN gate applied to them gave more than ~129 trades in five years, i.e. if they flushed more often; unlikely.
- Squeeze clustering would be removable if delisting notices (BAKE/HIFI/FLM style) were in the data as a flag; with them excluded the pump sleeve loses
  its worst tail (7 trades within 30 days of a perp delisting, mean -3,213 bp, 3 of the 39 bad trades, including the BAKE/HIFI pair behind the 2025 low point). That is the one actionable lead
  here and it is a data feed, not a model.
