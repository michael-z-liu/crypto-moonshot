# Why leverage fails on capitulation longs

Scripts: `w2_lev_a.py` (book trades, MAE vs return), `w2_lev_b.py` (all candidates, leverage runs, survival
leverage), `w2_lev_c.py` (1m anatomy of 2025-10-10, majors, whole market), `w2_lev_d.py` (cohort, 1m stop fills,
perp paths). Rule everywhere: day5 pct >= 0.995, 2022-2026, TP +20%, exit 72h, 10 slots, cost 0.3%, spot paths unless
noted. All numbers are from these runs. "MAE" = max adverse excursion of the low from the entry (next 5m open)
until exit; "hypothesis" marks speculation.

## Summary of causes

1. **The edge and the tail are the same trades.** The signal picks the coins that crashed hardest *right now*. Those
   are the coins with deep wicks, and a wick that deep is exactly what the +20% TP later collects. At 1x the
   wick costs nothing (a mean +10.7% trade, 81 of 162 book trades never fall more than 8%). With leverage the wick
   is paid first and the rebound is never collected.
2. **One day carries the whole tail.** 2025-10-10 produced 335 of 581 candidates of 2025 (58%), and 159 of the 187
   candidates with MAE >= 33% in the full 2022-26 set. Outside that day the share of candidates with MAE >= 49% is 0.3%.
   Across 2022-26 only 24 distinct days feed the 162 book trades; 17 trades with MAE >= 33% sit on 6 days.
3. **Slot concentration turns "10 independent bets" into one bet.** Candidates arrive in a burst when the whole
   market crashes. The ten slots are filled first-come in time order, so on 10-10 they were taken by the 20:55-21:10 spikes: the entries
   made *before* the bottom (mean MAE 71%, mean 1x net -3.5%), not the later ones (mean MAE 36%).
4. **Compounding asymmetry.** Isolated margin caps each loss at -100% of margin, but a one-day all-slots loss is a
   loss of the account: at 2x, 10 of 10 positions liquidated on one day = -99% of the stake of about 96% of equity.
   A -89% year needs +800% to repair; the 1x year (+17%) never gets the chance to compound.
5. **The wick is real, not an artifact** (checked on 1m candles, spot and perp paths). It is also very short
   (median 3x deeper than the close an hour later), so the loss comes from the path, not from the outcome.
6. **A stop does not rescue it, but not for the reason the 5m simulator shows.** The 5m sim gaps through the stop
   and liquidates first; on 1m data a 25% stop would fill near -25%. It then converts an unavoidable -100% into
   -51% (2x) / -76% (3x) of margin on all stopped positions at once, and forfeits the rebound that 60% of them would
   have taken.

## Evidence

### A. What happened on 2025-10-10 (1m spot candles, 414 USDT pairs with 1m data)

Crash sequence for WLFI (typical): 20:56 first wick (-9% in a minute, 5.7x volume), 21:12-21:20 the capitulation
(0.1488 -> low 0.0715 in 8 minutes, -52% from the 21:00 open of 0.1482), low at 21:20, 21:22-21:25 back to 0.10-0.11, a quiet range all hour.
Market wide (reference = 21:00 open, window 20:59-23:00):

| statistic | value |
| --- | --- |
| symbols with a wick <= -20% / -33% / -50% / -70% / -90% | 95% / 87% / 76% / 42% / 6% |
| median wick low | -65.7% |
| median close at 22:00 vs the 21:00 open | -20.8% |
| median recovery from the low to the 22:00 close | +43 points |
| BTC / ETH / SOL / TRX / BCH wick | -10.7% / -11.3% / -14.3% / -8.1% / -10.0% |
| BNB / NEAR / XRP wick | -28% / -44% / -50% |
| DOGE / LTC / LINK / ADA / ETC | -57% / -57% / -60% / -62% / -59% |
| AVAX / UNI / DOT / APT / FIL | -66% / -71% / -83% / -82% / -85% |
| ATOM | -99.97% (single 1m print of 0.001, 1,582 trades in the bar) |

The ten book trades (entry = next 5m open; 1m detail): low -55% to -93% of the entry price for nine of them (minute of the low 6-25 after the entry, close 60 minutes after entry -9% to -27%).
WLFI is the exception in the 1m table: its spike was at 15:30 and its entry at 15:35, so the 3h 1m window after entry shows only -1.1%; its 5m-path MAE of 51% is the 21:20 wick later in the 72h hold.
Their 72h results: 4 at TP or better (EPIC +19.7%, TST +14.6%, 1000SATS +10.5%), the rest between -19% and -2%.

**Candidate cohort on 10-10 (335 spikes, all 1x, 72h):** mean +16.5% net, 77% hit the TP, median MAE 26%, 47% had
MAE >= 33%. By entry time (5m `open_time`): 7 at 20:55 -> mean MAE 74%, mean net -3.6%; 119 at 21:10 -> MAE 69%, net +12.7%;
75 at 21:15 -> MAE 29%, net +19.6%; entries at 21:20 or later -> MAE <= 15%, net +19.6%. The same day is
the best day of the strategy for anyone who enters after the low and the worst for anyone who enters before it. A
slot book that fills first-come takes the early group.

### B. MAE versus final return (all 4,224 candidates 2022-26; with the 10-slot book in brackets)

| year | candidates | mean 1x net | MAE >= 33% | MAE >= 50% | max MAE |
| --- | --- | --- | --- | --- | --- |
| 2022 | 73 | +11.8% | 10 | 4 | 99.9% (LUNA) |
| 2023 | 138 | +13.9% | 0 | 0 | 18.8% |
| 2024 | 1,059 | +10.8% | 16 | 0 | 38.7% (2024-04-12) |
| 2025 | 581 | +17.0% | 161 | 137 | 99.9% (ATOM, RVN) |
| 2026 | 32 | +13.1% | 0 | 0 | 24.8% |

Of the 187 candidates with MAE >= 33%, 147 finished profitable (79%); in the 10-slot book 5 of 17 did. Final return by MAE bucket in the
book (162 trades): MAE < 10%: n=90, +16.5%, win 96%; 10-20%: n=40, +13.4%; 20-33%: n=15, -6.2%, win 27%; 33-50%:
n=3, -18.1%, win 0%; > 50%: n=14, -9.5%, win 36%. Book MAE percentiles 50/75/90/95/99 = 7.7 / 16.5 / 33.3 / 76.5 / 88.9%.
So the profitable-but-deep trades are almost all 10-10 trades: the wick came first and the TP followed, which is a pattern
that survives only without liquidation.

### C. Why a mean-positive book goes to -90% (10 slots, liquidity cap, isolated)

| lev | 2022 | 2023 | 2024 | 2025 | 2026 | mean net per trade |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | +26% | +40% | +105% | +17% | +15% | +10.7% |
| 1.25 | +9% (4 liq) | +51% | +140% | -45% (5 liq) | +19% | +8.7% |
| 1.5 | +10% (8 liq) | +63% | +178% | -63% (7 liq) | +22% | +9.9% |
| 2 | -30% (15 liq) | +88% | +267% | -89% (10 liq) | +29% | +15.7% |
| 3 | -36% (25 liq) | +143% | +414% (3 liq) | -80% (16 liq) | +42% | +22.1% |

- At 2x the 2025 liquidations are **all 10 on 10-10**; at 3x 16 of 16. In 2025 at 2x the ten early winners (+39% on
  margin, stakes $1,000) lifted equity to about $13.9k; the next ten positions had $1,328 stakes and 10 of 10 lost 99%: -$12.2k on that one day
  against a -$8.9k year total.
- Naive scaling would give +21.4% (2x) and +32.1% (3x) per trade; the realised +15.7% and +22.1% show the "liquidation
  tax" of 5.7 and 10 points of trade mean, concentrated in a handful of trades. The mean stays positive; the
  *distribution over years* is what breaks, because the loss events are the same day for every slot.
- Years without a crash day gain a lot from leverage (2024 +267% at 2x, +414% at 3x). The leverage drag is 100% crash
  days (2022 LUNA/FTT, 2025-10-10).
- The 10 slots offer no diversification: of the 162 book trades, only 24 distinct days; the 17 trades with MAE >= 33% are on 6 days; on
  2025-10-10 the book's concurrent adverse move (first 10 taken, mean MAE) was 73.8% of equity at gross 1.0, i.e. 110.7% at gross 1.5
  (E25's 104% worst_mtm is the same event under its own slot rules). Cross margin can't survive it either: equity gone before the rebound.
- Perp-path check (same rule on 5m perp prices, 474 of 581 candidates in 2025 have perps): same result: 2x 2025 -92% (10 liq), 3x -86% (13 liq).
  Mean MAE on 10-10 candidates 40.0% on perp vs 45.7% on spot for the same rows. Perp last-price wicks are shallower by about 6 points, not by enough.
  (Mark-price liquidation is not in the data; hypothesis below.)

### D. A 25% stop with leverage

- The 5m simulator (stop_slip 1%): 2x 2025 -32% (11 liq anyway), 3x 2025 -80% (15 liq). In 186 trades crossing -25% inside the first 8h, 160
  jump past -33% inside the same 5m bar and 168 came from < 10% adverse the bar before. Within a bar the simulator checks liquidation first,
  so the stop never gets a chance.
- 1m check on the 335-candidate cohort of 10-10 (stop-market, fill = min(level, bar open)): 174 stopped at -25% (206 at -15%);
  mean fill -25.0%, none worse than -35%; median 3 minutes from entry. So the 5m "gap" is mostly bar resolution. But the stop converts
  into: 1x -25% of every stopped position, 2x -51%, 3x -76% on margin, and 60% of them (65% for the -15% stop) would have reached the +20% TP
  within 72h (stopped positions' mean unstopped 1x net +13.9%).
  With ten slots stopped within the same ten minutes the account takes -51% (2x) or -76% (3x) in one day. A tighter stop (15%) is -31% / -46%.
  Real fills in a book that was empty are probably worse than a bar open (hypothesis).

### E. Majors (E29) and the data-artifact question

- 12 of the 19 majors (XRP -50%, ADA, DOGE, AVAX, LINK, LTC, UNI, ETC, DOT, FIL, APT, ATOM) wicked >= 50% from the 21:00 open on 1m
  spot candles; MATIC has no 1m data (delisted). Only BTC, ETH, SOL, TRX, BCH stayed within 8-15%, BNB -28%, NEAR -44%.
- Artifact checks: (i) the wick is in 76% of the 414 pairs, with the low bars carrying 1.3-15x the median 1m volume; (ii) the lows stand
  for 4 to 131 one-minute bars below half the depth, and half of the depth is regained 1 to 30 minutes after the low;
  (iii) the perp last-price paths show the same wicks (about 6 points shallower); (iv) the one clear outlier is ATOM's 0.001 print
  (1582 trades in that 1m bar, 98.7k volume vs 110-157k neighbours): real prints, but a single minute. Conclusion: not a feed glitch.
  Whether a levered perp holder would have been liquidated depends on the perp mark/index price, which we do not have (hypothesis: index-based marks wicked less on majors, so the spot-path liquidation count is an upper bound for majors; the sim cannot tell).
- Consequence for E29: the 5 or 6 liquidations of "majors" are all on 10-10 and are among the 12 above.

### F. What leverage would have survived every year

Per year, 1 / (max MAE + 1% maintenance margin) over all candidates:

| year | max MAE | max survival leverage |
| --- | --- | --- |
| 2022 | 99.9% | 0.99x |
| 2023 | 18.8% | 5.1x |
| 2024 | 38.7% | 2.5x |
| 2025 | 99.9% | 0.99x |
| 2026 | 24.8% | 3.9x |

Over 2022-26: 0.99x, so isolated leverage above 1.0x loses some margin in a worst case. Removing the three crash events
(LUNA May 2022, FTT Nov 2022, 2025-10-10) the max MAE is 40.6% -> 2.4x; the 99th percentile is 34.3% -> 2.9x. Liquidation
shares of candidates outside 10-10: 0.1% at 1.25x, 0.3% at 1.5x and 2x, 2.3% at 3x. Inside 10-10: 10 / 28 / 42 / 48%.

## What blocks leverage

- Not the mean (positive in every leverage run) and not the number of liquidations in normal days (0.3% at 2x). It is a
  **perfectly correlated tail**: one flash crash puts every open slot below the liquidation level inside ten minutes, and the entry rule
  fills the slots at the start of the crash.
- The tail is **inseparable from the signal**: the best trades (deepest capitulation) are the deepest wicks. A filter that removes deep
  wicks removes the edge (the highest-MAE bucket on 10-10 includes the most profitable positions; 77% hit the TP).
- Liquidation is **path-dependent and unrecoverable**; the rebound is paid only to unlevered holders. 79% of candidates with MAE >= 33% ended profitable.
- A stop cannot sit between: ten minutes from -10% to -90% on thin books means a stop fills at 1m opens at best (our estimate -25%) and
  forfeits the rebound; the account-level damage is -25% x leverage across all slots, so a stop works only at 1x (-25% of equity on the worst day, still a -25% day).
- Position count does not help: only 24 event days in 162 trades, and all 10 slots land on the same day.

## What would have to be true

For leverage to be viable one of these would have to hold; none is shown by this data:

1. **Slots fill after the low.** On 10-10 the 21:15+ entries had mean MAE <= 29% (net +19.6%) against 71% for the 20:55 entries. A rule that waits
   (for example for the first 5m bar that closes above the prior low, or a fixed delay) might hold MAE under about 33% (= 2.9x survival at 3x liquidation 32.3%).
   Hypothesis; needs a pre-registered test, and note that E10/E21 saw a one-candle delay cost about 2.7% at 1x.
2. **A market-wide crash circuit-breaker.** More than N candidates in one hour (10-10 had 335 in two hours) means skip leverage, the rest at 1x. Untested here (hypothesis): it would have to be tuned on the three crash days, which is a sample of three.
3. **Position sizing, not leverage:** the book at gross <= 0.5-1.0x. Worst concurrent adverse is 73.8% of equity at gross 1.0 on 10-10; leverage
   survival over the full history needs gross about 1.0x with no safety margin, so under 1.0x is the only level that survives every year at all.
4. **A margin system that does not liquidate on the wick** (perp mark price, portfolio margin with cross collateral from an unrelated asset). We cannot test this
   from spot/perp last-price data; the 6-point perp-vs-spot difference suggests it would not be enough on its own (hypothesis).

Net: the data supports 1x only; the surviving leverage over 2022-26 is 0.99x. Without LUNA, FTT and 10-10 it would be 2.4x, but those three events (two of the five test years) are the ones that
determine the outcome of a levered book, so the planning case is the worst day, not the mean trade.
