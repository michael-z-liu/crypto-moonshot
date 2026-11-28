# Why the listing-short variants failed (diagnosis, not rescue)

Scripts (new, read-only on existing code): `w4_listings_a.py` (groups, weak/strong, cadence), `w4_listings_b.py`
(hourly paths, volume, young decay, BTC), `w4_listings_c.py` (delay bootstrap / permutation), `w4_listings_d.py`
(pre-listing run-up, monthly cadence). Cache written: `$OWN/w4_spot_trades.parquet`. All numbers are per trade, net of the
0.3% cost and funding unless stated; sleeve = short at spot-listing + 72h, hold 30d, TP/SL 50%, perp 1h candles (201 trades,
143 weak = price <= listing price, 58 strong).

## Summary of causes

1. The edge is a property of the Binance SPOT listing event, not of "new coin". Perp-only listings (no spot listing)
   are smaller, calmer and drift ~0 (median 30d price change -6% vs -22%; mean +2.4% vs -9.7%). Shorting them averages a
   cost-sized +0.4% (weak) / -3.6% (not weak). Average perp-first clock on spot-listed coins is also weaker (+11% weak) than
   the spot clock (+20%), i.e. the clock that matters is the spot listing time.
2. "Weak" works mostly by cutting the squeeze tail, not by momentum in the median. Weak: 15% of trades hit the 50%
   stop; strong: 38%. Strong's median trade is still +16%; its mean is +0.3% only because 22 of 58 trades lose 50%. The
   since-listing bins are non-monotonic (0..15%: -6%; >40%: +18%, n=14), so "momentum" is weak evidence; the robust fact
   is that listings still above their price at +72h are the ones with live squeezes.
3. The front-loaded decay is a hype-fade/volume-fade process: price falls steadily from hour 0 with no bounce, and 35% of the
   30-day median decline happens in the first 3 days. Alt beta aside, decay after week 4 is about cost-sized.
4. 72h is NOT a structural low or rebound. The average path has no bounce around 48-96h; the "peak" is a +3..+4 pp bump
   in a curve that falls monotonically with delay. A coin-level paired bootstrap cannot separate 72h from 24h or 48h
   (P(72h > 24h) 0.84, > 48h 0.97 for weak) but separates it from 96h/120h. The sharp peak is partly composition (which coins
   count as weak at entry) plus selection among 5-13 tried delays. Expect ~+16-17% weak, ~+12-13% all, not +20%.
5. 2025's 73 weak listings are a perp-coverage artifact, mostly. Share of spot listings that have a perp at listing: 6% (2021),
   15%, 32%, 82% (2024), 97% (2025). Listings with a perp: 7, 6, 17, 50, 94, 22. The weak share in 2025 (78%) is only
   modestly above 2022-24 (60-71%).
6. Why perp-only sleeves and young-coin shorts fail is the same: no listing catalyst, so only generic alt bleed
   (+0.5..+1%/week, below noise and funding).

## Evidence

### A. Spot listings vs perp-only (weak 72h/30d, E18 clock = perp first candle)

| group | weak? | n | mean net | median | win | median 24h perp qv |
|---|---|---|---|---|---|---|
| perp-only | no | 152 | -3.6% | +3.0% | 54% | 62M |
| perp-only | yes | 222 | +0.4% | +3.8% | 58% | 43M |
| spot-listed coin (perp clock) | no | 131 | -0.9% | +16.8% | 57% | 223M |
| spot-listed coin (perp clock) | yes | 197 | +11.2% | +21.7% | 71% | 101M |
| spot clock (the sleeve) | yes | 143 | +20.0% | +33.3% | 78% | 179M (day-1 qv) |
| spot clock | no | 58 | +0.3% | +16.4% | 62% | 292M |

First 72h of every perp listed 2021+ (>=800h of data), perp clock:

| | n | median day-1 qv | median 72h return | median high 72h | median low 72h | hourly vol | median 30d after 72h | mean 30d |
|---|---|---|---|---|---|---|---|---|
| spot-listed | 328 | 228M | -6.1% | +18% | -17% | 2.9% | -22.3% | -9.7% |
| perp-only | 478 | 43M | -2.4% | +9% | -12% | 2.0% | -6.0% | +2.4% |

Perp-only coins have 1/5 of the volume, half the early range and no fall afterwards. Perp-only earlier-year means are
negative or nil (2021 -17%, 2023 -5%, 2024 -3%, 2025 +3.8%, 2026 -1.7%).
Perp-before-spot does not matter: the 65 coins whose perp existed >3 days before the spot listing (median age 21d) give
+14.3% (same as the 125 whose perp starts at listing, +14.3%). Their perp volume is 2.3x higher in the 24h after the spot
listing than in the 24h before, and the pre-listing run-up (median +12.5% in 24h, +18% in 72h, +25% in 168h) has
correlation +0.26 with the short's net return (not with weakness): a bigger run-up = a bigger fade (public wall-clock
effect, n=65, not tested for significance).
What is missing: no float/FDV, unlock or airdrop-size data in the cache. The explanation that spot listings carry
Binance-user retail demand and airdrop/launchpool supply that sells into the first week is SPECULATION; the volume path is
consistent with it (below) but cannot prove it.

### B. Weak vs strong at +72h

Exit mix (net): weak: TP 53 (37%) +48.2%, time 68 (48%) +21.4%, stop 22 (15%) -52.2%; strong: TP 18 (31%) +46.9%, time 18
(31%) +16.3%, stop 22 (38%) -50.8%. Funding paid is small (by exit type 0.0..2.8% of notional).

Median since-listing price at entry and 30d outcome by bin:

| since_list | n | mean net | median | win |
|---|---|---|---|---|
| <= -50% | 11 | +21.1% | +48.2% | 73% |
| -50..-30% | 32 | +29.5% | +40.2% | 84% |
| -30..-15% | 55 | +14.9% | +29.5% | 76% |
| -15..0% | 45 | +19.2% | +29.4% | 78% |
| 0..15% | 29 | -5.9% | +14.5% | 55% |
| 15..40% | 15 | -6.7% | -2.4% | 47% |
| 40..100% | 9 | +17.8% | +40.5% | 67% |
| > 100% | 5 | +26.3% | +50.3% | 80% |

Forward path from hour 72 (log, rebased), 129 spot listings with a perp at listing and 1000h of history:

| days after entry | weak mean / median | strong mean / median |
|---|---|---|
| 1 | -3.1% / -3.0% | -2.3% / -3.0% |
| 4 | -10.7% / -11.3% | -2.4% / -4.0% |
| 7 | -16.4% / -13.4% | -6.6% / -10.4% |
| 18 | -29.8% / -31.3% | -12.7% / -16.0% |
| 30 | -40.0% / -39.7% | -16.6% / -23.3% |

Strong at 72h: median peak so far +41% (peak hour 30), median now +16%: a pumped coin that has already begun
fading. Weak: median peak +0.0% at hour 0 (hour-of-peak median 0), median now -24.5%: it has never traded above its listing
price. Strong coins have bigger tails both ways (p75 at 30d +16% in log terms), which is why the 50% stop is hit 38%.
Weak coins' 72h perp quote volume is ~half that of strong ones (1.07G vs 2.16G), they are less liquid and fade faster:
perp volume vs day 1 (median): day 2 0.47 vs 0.64, day 7 0.21 vs 0.32, day 31 0.07 vs 0.13. Reading: the weak
coins are the ones where demand dried up quickly (and, speculatively, where listing-day sellers were
supply-heavy); strong ones still have a buyer base (and squeezes).
BTC: net return correlates -0.36 with BTC's 30d forward return (short beta); weak share rises when BTC is down in the 72h
after listing (85% vs 64%; corr(since_list, BTC 72h return) = +0.32). So part of "weak" is just market direction, and
the sleeve is short-market in disguise: tercile of BTC forward 30d >+4.7%: -5.0% a trade (n=67); <-4.9%: +28%.
That is why BTC-trend filters split inconsistently: they use the past, the cause is the contemporaneous 72h-to-30d market.
BTC 30d before listing terciles: +17.4%/+7.0%/+18.6% (no monotone pattern).

### C. Front-loaded decay

Median log price vs listing price: 3d -17.9%, 7d -24.2%, 14d -29.1%, 30d -51.4%, 41d -54.0%. Average drift per 12h: first 12h
-7.1%, 12-72h -1.3..-3.2%, 72-120h -1.0..-1.6% (with a small +0.9% up window at 120-132h), 120-240h -0.2..-2.1%. 74% of listings
are below the listing price at hour 72, 81% at day 42. 89 of 129 coins have their 7-day high in the first 24h (hour 0
bin); the minimum within 30d falls in days 21-30 for 77 of 129.

Young-coin shorts by age week (spot-listed coins with a perp, 7d hold, 0.3% cost, no funding):

| weeks | mean | n |
|---|---|---|
| 1-4 | +3.3% | 520 |
| 5-13 | +0.5% | 1141 |
| 14-26 | +1.0% | 1534 |
| baseline: perps older than 1 year | -0.1% (median +0.9%) | 11737 |

Weak (<= listing price) or not makes no difference after week 4 (+0.5% either way). Week 1-2 +4.3..+5.6% is what the
sleeve already captures with its 30d hold. The remaining weekly edge is ~+1%, i.e. the E44 result (+1%/week).

### D. Waiting for a drop (E36) and the perp-only weak sleeve
Waiting up to 30 days takes coins that did NOT fall in the first 72h (the strong group) and shorts them later when the
price crosses below the listing price; per hour 72 the fall is partly already done (-17% median) and the remaining 30d
decay is smaller (strong -17%, vs weak -40%). That the late entry average (+15.3%) sits between weak (+20%) and strong (+0.3%)
is what the group split above predicts. The perp-only weak sleeve (+5.2% a trade) is the perp-only group above: no
catalyst, +0.4% net in E18's clock, so slot capacity spent on it displaces better trades.

### E. Entry delay: noise or structure? (30d hold, TP/SL 50%, coin-level bootstrap, 5000 draws)

Mean net by delay (n coins; the weak set changes with the delay):

| delay h | all n / mean (se) | weak n / mean (se) | weak median | weak stop share |
|---|---|---|---|---|
| 12 | 185 / +12.9% (3.1) | 123 / +18.9% (3.5) | +35.9% | 15% |
| 24 | 195 / +14.1% (2.9) | 123 / +17.1% (3.6) | +37.1% | 17% |
| 36 | 199 / +13.8% | 138 / +18.9% | +31.4% | 15% |
| 48 | 200 / +12.8% | 146 / +15.7% | +28.6% | 14% |
| 60 | 201 / +12.2% | 148 / +16.9% | +29.6% | 13% |
| 66 | 201 / +13.4% | 145 / +17.7% | +30.4% | 14% |
| 72 | 201 / +14.3% (2.7) | 143 / +20.0% (3.0) | +33.3% | 11% |
| 78 | 201 / +11.5% | 148 / +16.2% | +31.2% | 14% |
| 84 | 202 / +11.7% | 146 / +16.9% | +31.8% | 13% |
| 96 | 202 / +11.7% | 152 / +14.5% | +30.1% | 15% |
| 120 | 205 / +10.4% | 157 / +14.0% | +27.3% | 15% |
| 144 | 206 / +11.6% | 153 / +14.7% | +25.2% | 13% |
| 168 | 210 / +9.3% | 156 / +14.0% | +19.9% | 13% |

Structure is a smooth decline of ~0.4..0.5 pp per extra 12h of delay (earlier is better), with 72h a 3-4 pp bump above
its neighbours (66h +17.7%, 78h +16.2%). On the coins common to all delays (n=107 weak) the means are
24h +22.4%, 48h +22.3%, 72h +20.7%, 96h +16.6%, 120h +14.1%: no peak at 72h at all. Bootstrap, weak set: P(72h is the best of
24/48/72/96/120) = 0.83 (vs 0.15 for 24h); mean(72h)-mean(24h) +2.9 pp [-1.9, +7.8]; -48h +4.3 pp [+0.5, +8.4]; -96h +5.5 pp
[+2.5, +8.6]; -120h +6.0 pp [+2.5, +9.5]. All trades (no weak filter): 72h is best only 52% of draws (24h 44%); differences vs 24h/48h
+0.3/+1.5 pp with CIs spanning 0. Label-permutation null within coin (delay irrelevant): the 72h bump (max minus mean of others
= 4.7 pp) is at the 95.5th percentile of the null (P = 0.045) before correcting for having looked at 5+ delays and many other
parameters: it is weak evidence. Year dispersion (weak, 24h/72h/120h): 2021 +50/+36/+26, 2022 +48/+38/+30, 2023 +32/+18/+15, 2024
+7/+7/+8, 2025 +20/+25/+18, 2026 -5/+12/-3. 72h beats 24h only in 2025 and 2026 (the two years with the most trades);
24h is better in the four years with <=30 trades: the "peak" comes from 2025-26 and tiny samples.
Why the book floor is lower at other delays: the per-trade level is only 1-4 pp lower but the book's worst year is 2024-26 with
~7-12% a trade, so a small shift in the few trades of a bad quarter moves the floor a lot (2024: 30 weak trades, +7% at every delay).

### F. Cadence and 2025

| year | spot listings | with perp at listing | traded | weak | weak share | weak net | strong net (n) |
|---|---|---|---|---|---|---|---|
| 2021 | 123 | 7 | 8 | 7 | 88% | +35.8% | +50.4% (1) |
| 2022 | 41 | 6 | 6 | 4 | 67% | +37.5% | +16.9% (2) |
| 2023 | 53 | 17 | 21 | 15 | 71% | +17.7% | -4.3% (6) |
| 2024 | 61 | 50 | 50 | 30 | 60% | +7.2% | -3.7% (20) |
| 2025 | 97 | 94 | 94 | 73 | 78% | +24.7% | -3.8% (21) |
| 2026 (Jan-Oct) | 22 | 22 | 22 | 14 | 64% | +12.3% | +14.5% (8) |

2025 had the highest listing count since 2021 (97, monthly 5-16, peak 16 in Sep and 11 each in Mar/Oct) and 97% perp coverage.
Quarterly: 2025Q1 26 listings/22 weak (median since-listing -26%), Q4 18/16 (-25%). 2024 had 50 trades but only 60% weak and +7%:
listings during the 2024 bull (BTC +7% pre-30d median) held up better (median since-listing ~0%). 2021-23 spot listings were
mostly untradable on perps (6-32% coverage); on all spot listings the 3d->30d short median was +21% (2021), +26% (2022),
+8% (2023; perp-less listings), 2024 +27% (no perp, n=11) / +15% (perp, n=50), 2025 +34% (perp, n=94) (spot-only series, not a tradable P&L): the effect was there long before the perp era,
but untestable with these tools.
Sample size is the issue: 4..30 weak listings a year in 2021-24; a 100-coin year is a single regime.

## What blocks it
- Small n and one regime: 143 weak trades, 73 of them in 2025; the SE of the mean is 3 pp, the year-to-year range +7..+38%.
- The signal is a short-market bet on listing-day supply: net return correlates -0.36 with BTC's 30d forward return; the
  weak filter is partly "the market fell in the 72h after listing" (corr +0.32). It cannot be known at entry.
- Tail risk is concentrated in strong listings (38% stop-outs, 50% adverse). Filtering them is the whole improvement.
- No data on float, unlock schedule, airdrop size, announcement time or retail holdings; so the mechanism (supply) cannot be tested.
- The perp-only and young-coin ideas lack the catalyst: no listing event, so they only carry alt bleed (+0.5..1%/week).
- 72h is not structural; per-trade results are flat within +-3 pp of 72h, so the book's year-floor claim is overfitted.

## What would have to be true
- Perp-only and young-coin shorts would need a dated supply event (unlock, airdrop claim, launchpool end) with known size; then
  those events, not "age", would be the sleeve. Needs unlock calendars, FDV/float: not in the cache.
- A 72h peak would need a repeatable bounce at ~day 2-4 (an average path rising there); the data show a monotone fall with
  one small up-window at 120-132h (+0.9%, 57% up share, n=129): not it.
- A BTC filter would need to predict the market over the next 30 days; the past 30 days do not (+17/+7/+19%).
- 2025's cadence (97 listings, 94 with perps) would need to persist; 2026 YTD has 22 listings in 10 months (5-6/quarter).
  If Binance's listing pace stays at 2026's, 143 historical weak trades equal ~7 years of data.
