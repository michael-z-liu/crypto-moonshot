# Why the pump shorts failed (diagnosis, not a rescue)

Scripts: `w1_pump_shorts.py` (run `0.4 1`, `0.25 0`, `0.15 0`: thresholds on `ret_4h`, 1 = first spike per coin per 24h),
`w1_pump_extra.py`, `w1_pump_ci.py`, `w1_pump_lsacc.py`. All numbers below come from these runs.

Samples: **S40** = perp-active pumps `ret_4h >= 0.4`, `qv_base >= 20k`, first spike per coin per 24h, 2021-26, n = 385
(the E15 sample). **S15** = `ret_4h >= 0.15`, every 5m spike (n = 31,733, the E19 population; about 8 rows per pump
episode). **S15d** = S15 deduplicated (n = 3,929). "net" = 72h hold, isolated 1x (liquidated at +100%), 30 bp cost,
funding paid. "fade" = short return on the perp close at hour 72 (price reverting down = positive).

## Summary of causes

1. **Perps do not squeeze more than spot.** The price doubles within 72h in 6.2% of S40 trades on the perp and 6.0% on spot, and in 23 of the
   24 cases it does both. The "spot +11%/trade vs perp +3%" gap is a waterfall of cost items, not one mechanism (S40, bp per trade):
   spot +1105 -> perp-vs-spot -392 (one frozen/delisted perp, ALPACA, is -134 of it; the three worst events -166; median gap only -29)
   -> +713 -> liquidation at +100% crystallises losses that would partly have retraced -140 -> +573 -> cost -30 -> **funding paid -236**
   -> **+307**. Funding is the only item that is structural for perps; the rest is tail handling.
2. **The strategy is a short-volatility payoff.** The mean is made by a fat middle (median fade +1380 bp at S40 level) and destroyed by a
   thin tail (p5 of the final short return -4440 bp, p99 max adverse +498%). The mean after stops/liquidation is a small difference between two large numbers
   (+14% median fade vs. about -77% average loss on the 6% of doublings), so every cost or tail-handling choice moves it by more than the edge.
3. **A stop cannot tell a squeeze from a retrace.** Price excursion is the same for both for hours; 75% of trades that touch +50% finish better than -50%, 33% finish in profit, and
   17% of trades that double finish below entry. Stops sell exactly the retraces the strategy lives on.
4. **Squeezes are weakly visible, and only through crowding.** There are very few events (24 doublings in S40; 18 with metrics in S15d-25), so most
   "features" are noise (null AUC sd 0.07-0.11). The one real signal is crowded shorts (low long/short account ratio, AUC 0.37, permutation p = 0.0005-0.0045, same sign in every year), and it is
   too weak to trade: it moves the doubling rate from 1.9% to 6.4% and costs 250 bp of funding, nothing like enough separation to avoid -100% outcomes.
5. **The model's ranking is mostly "bigger pump", and bigger pumps are the squeeze-prone ones.** The top decile is 60% `ret_4h >= 0.4` (rest: 6%) and doubles 2.7-3x as often (8.3% vs 2.6%).
   Its clipped-spot target of +1727 bp becomes +229 bp net. On net return the model's rank IC (0.164) equals `ret_4h` alone (0.160).
6. **The regime result is fragile and sample-dependent.** In S15 the realistic mean is -30 bp (2022-24) vs +321 bp (2025-26), driven by the median fade doubling
   (+531 -> +1164 bp) while funding paid rises from about 40 to about 240 bp. But it needs the repeated-spike population: S15d gives -158 vs -38 bp
   (both CIs include or sit below 0), and S40 gives +446 vs +157 bp (the other way round). BTC-7d is a real split only for 2024-26 pumps; it is not a market-beta effect.

## Evidence per question

### Q1. Why do perps look worse than spot (funding, OI, delistings, who is short)?

S40 per-year, bp before costs (`w1_pump_shorts.py 0.4 1`):

| year | n | spot short | perp raw | perp iso (+100% liq.) | median gap perp-spot | funding paid |
| --- | --- | --- | --- | --- | --- | --- |
| 2021 | 89 | +806 | +644 | +457 | -46 | -41 |
| 2022 | 33 | +1274 | +1012 | +867 | -24 | +84 |
| 2023 | 27 | +1541 | +1043 | +1057 | -256 | +339 |
| 2024 | 44 | +832 | +515 | +199 | -42 | +101 |
| 2025 | 119 | +852 | +156 | +391 | -21 | +408 |
| 2026 | 73 | +1811 | +1569 | +927 | -17 | +408 |
| all | 385 | +1105 | +713 | +573 | -29 | +236 |

* Doubling frequency: perp 6.2% (24), spot 6.0% (23); >= +50%: 16.4% on both. 23 of the 24 perp doublings are also spot doublings; only 1 is perp-only.
  The mean gap of -392 bp falls to -226 without the 3 worst events and -77 without the 50 worst.
* ALPACA 2025-04-29 14:15 (the worst event): perp and spot move together until hour 12 (+0.73 vs +0.68). From hour 24 to hour 72 the perp
  close is **flat at +5.32** (no further movement; consistent with a frozen/settled contract, my reading, not verified), while spot peaks and comes back to +0.16 at hour 72.
  The gap (-5.2) is a stale perp price, not a different squeeze. One frozen perp in 385 events costs 134 bp per trade.
* Funding: shorts paid +236 bp per 72h on average (in calm markets a short usually receives funding). It is
  **+682 bp in the doubling events vs +207 bp otherwise**. By entry funding rate: most-negative quintile pays 573 bp, the most positive quintile receives 51 bp.
  Funding is a cost that rises with the squeeze (the short is paying crowded-short funding exactly while it is losing).
* Who is short (S15d, `w1_pump_extra.py`): share of pumps entered at negative funding rises 11% (2021), 44%, 29%, 14% (2024), **48% (2025), 60% (2026)**; funding paid per trade 
  -45, +37, +82, +11, +193, +279 bp. So in 2025-26 the market is paying the shorts' side less / charging it more; the trade is crowded (hypothesis: the idea itself is widely traded).
* Open interest and long/short ratios (S15d with metrics, 1,255 events, 55 doublings; `w1_pump_lsacc.py`): low long/short **account** ratio tercile
  (more short accounts) doubles 6.4% of the time (sq50 19.4%, funding paid 333 bp, net -133 bp), high tercile 1.9% (sq50 10.9%, funding 82 bp, net +231 bp). AUC for doubling 0.368 (`ls_acc`) and 0.363 (`top_ls_acc`),
  permutation p = 0.0045 / 0.0005; by-year AUC for sq50 0.36-0.47 (always < 0.5). OI change 24h 0.62, OI 1h/4h 0.33/0.39 (noisy, unadjusted). `taker_ls` 0.50. This **refines E17**: crowded shorts do mark squeezes, but weakly.

### Q2. Why does a stop destroy value (path shape)?

S40, perp paths, 72h (`w1_pump_shorts.py 0.4 1`):

| touched adverse | share of trades | ends in profit (below entry) | ends better than -th | mean final | median final | median hours to peak |
| --- | --- | --- | --- | --- | --- | --- |
| +10% | 71.2% | 60.9% | 72.6% | +11 bp | +731 | 12 |
| +20% | 46.0% | 46.9% | 71.2% | -858 | -164 | 16 |
| +30% | 29.6% | 39.5% | 73.7% | -1650 | -941 | 19.5 |
| +50% | 16.4% | 33.3% | 74.6% | -2953 | -2403 | 22 |
| +75% | 8.3% | 18.8% | 68.8% | -6103 | -4589 | 21.5 |
| +100% | 6.2% | 16.7% | 75.0% | -7753 | -5701 | 21.5 |

* Max adverse quantiles: p50 +18%, p75 +35%, p90 +68%, p95 +113%, p99 +498%. Correlation(max adverse, final) = -0.64: bigger excursions lose more,
  but the relation is loose enough that a third of the trades that hit +50% end in profit.
* Stop value (stop fills exactly at the threshold, 30 bp cost, funding; hold = +447 bp):

| stop | hit | mean | lost on stopped trades that would have won | gained on stopped trades that would have lost |
| --- | --- | --- | --- | --- |
| 20% | 46.0% | -78 | -906 | +381 |
| 30% | 29.6% | +47 | -657 | +258 |
| 50% | 16.4% | +112 | -470 | +135 |
| 100% | 6.2% | +307 | -137 | -3 |
| 200% | 2.9% | +170 | -241 | -36 |

  The 50% stop loses 335 bp of 447. A stop at the doubling level (which is also the 1x isolated liquidation) gives back 140 bp because of the 17% that retrace.
  In S15 (31.7k rows) the same pattern holds with a smaller scale: hold +76, stop 50% +24, stop 100% +27, stop 20% -33, stop 10% -60.
* Mechanism: a pump short is a bet on reversion over 3 days. Reversion is strong (median fade +1380 bp at S40), so
  most paths oscillate against the position first (46% go +20% against; half of those still end in profit) and the stop converts
  an excursion into a realised loss. The losers that a stop does remove (beyond +100%) are rare (6%), so stops cannot pay for the winners they cut.
  Once the price is +50% against after 8h the continuation risk is real (P(doubling | >= +50% within 8h) = 45% vs 5.7% unconditional, S15d-25) but 59% of those trades still end in profit,
  and 56% of all doublings were already +50% against within 8h, so most squeezes announce themselves early but share the early look of a normal retrace.

### Q3. Why are squeezes unpredictable?

* Few events and a changing base rate: doublings in S40 by year 5, 1, 1, 2, 8, 7 (rate 5.6%, 3.0%, 3.7%, 4.5%, 6.7%, 9.6%). Null AUC sd for a useless feature is 0.07 (18 positives) to 0.31 (1 positive, 2023). Univariate AUCs
  in S40 look "good" (`wick_up` 0.70, `r5m` 0.66, `off_high4h` 0.63) but the same features have AUC 0.57, 0.54, 0.54 in the 8x larger S15 population, i.e. small-sample noise.
* Squeezes start from the pump itself: median hour of the peak adverse move among doublings is 21.5, a quarter peak within 8h; 67% were already +30% against at hour 8. They are continuation of
  the pump, and the pump features that predict reversion (size, wick, volume multiple) are the same ones that predict continuation (AUC `ret_4h` 0.60, `ret_24h` 0.58-0.61 for doubling, mostly > 0.5).
* The usable signal exists only in position data (crowded shorts, above) and in size; E17's walk-forward GBM AUC 0.42-0.63 matches the sizes of these univariate effects.
* A doubling ends in profit 17% of the time (S40), so even a perfect early alarm for "doubles at some point" would throw away profitable trades.
* Prior squeeze in the same coin does not help: after an earlier squeeze 2/20 (10%) vs 22/365 (6%) otherwise (S40, chronological).

### Q4. Why does ranking skill not become realistic profit?

Transformer walk-forward percentile `pq`, S15, 2022-26 (n = 22,676; `w1_pump_shorts.py 0.15 0`, `w1_pump_extra.py`):

| decile (pq) | mean ret_4h | clipped spot | unclipped spot | perp raw | iso | median perp | doubling rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.21 | +46 | -116 | -147 | -207 | +291 | 2.6% |
| 4 | 0.21 | +363 | +311 | +278 | +187 | +700 | 1.7% |
| 8 | 0.32 | +1119 | +1004 | +854 | +719 | +1276 | 3.0% |
| 9 | 0.59 | +1528 | +1305 | +874 | +684 | +1603 | **6.9%** |

* Top decile vs rest, bp per trade: clipped spot +1727 vs +443; unclipped -199 vs -123; perp-vs-spot **-545** vs -57; liquidation cap -247 vs -53;
  cost -30; funding paid **477** vs 93; **net +229 vs +88**. Doubling rate 8.3% vs 2.6%, ret_4h mean 0.69 vs 0.24. The top decile loses 87% of its clipped edge to tails, perp gaps and funding.
* The model orders by size: rank corr(pq, ret_4h) = 0.45; `ret_4h` alone has IC 0.223 on the clipped spot target vs the model's 0.249, and on net 0.160 vs 0.164. So the model adds almost
  nothing over "pick the biggest pump" on the target that is actually paid.
* Within pump-size bins (top 30% vs bottom 30% by pq, net bp / doubling rate): 15-20%: +288 vs -142 (2.6% / 1.6%); 20-30%: +266 vs -198; 30-40%: +429 vs +173; 40-60%: +458 vs +758 (5.3% vs 0%);
  60%+: +453 vs +2682 (11.7% vs 0%, small n). IC on net by bin: +0.115, +0.113, +0.069, -0.033, -0.048. So ranking works for small pumps (mostly a better-than-average 
  short, no squeeze exposure) and **inverts at 40%+**, where the high scores pick the squeeze-prone pumps.
* At S40 (n = 296 scored): top decile net 371-541 bp (transformer/MLP), 220 (GBM) vs rest 605-716 / 764; the top bin's median fade is -341 bp (transformer) because it holds the 17% doubling cluster (ret_4h 0.81).
* Low scores are worst because they pick the pumps that continue (iso -207 / -279 bp in deciles 0-1, S15).

### Q5. Why 2025-26, and why does BTC-7d matter?

S15 (n = 31.7k, E19 population), mean realistic net and median fade (cluster bootstrap by coin-month, 90% CI; `w1_pump_ci.py`):

| period | n | net | 90% CI | median fade | mean fade |
| --- | --- | --- | --- | --- | --- |
| 2021 | 9057 | -160 | -479..+130 | +529 | -149 |
| 2022-24 | 14436 | -30 | -187..+124 | +531 | +93 |
| 2025-26 | 8240 | **+321** | +97..+547 | **+1164** | +696 |

* The 2025-26 effect is the **median 72h fade doubling** (+531 -> +1164 bp; S40: +1278 -> +1703; S15d: +326 -> +847) with unchanged pump size and no matching change in doubling rate
  (S15d: 1.4-3.2% in 2021-25, 7.4% in 2026). Meanwhile funding paid rises (above), so funding paid rises by about 200 bp, roughly a third of the extra fade. Pump composition: young perps (< 60 days) are 12-28%
  in 2021-24, 15% in 2025, 3% in 2026, so a new-listing cohort does not explain it.
* But it is not robust to the unit of analysis. Each pump episode contributes about 8 rows in S15; the dedup version (S15d) is negative or zero in every period (2021 -291, 2022-24 -158 [-252..-56], 2025-26 -38 [-203..+130]),
  and S40 is positive in 2021-24 (+446 [-46..+950]) and smaller in 2025-26 (+157 [-338..+675]). The "2025-26 only" claim holds for the repeated-spike (late in the pump) rows, not for the first spike. I do not know which is right for trading; it shows the claim rests on the
  re-entry rows plus a regime where pumped alts fade more, with CI that does overlap in S40.
* BTC 7d: S40, BTC 7d > -3% (n = 40/42 in each arm for >= 0.4): +1271 vs -754 bp (diff 90% CI +609..+3324); 2024-26 only: +1471 vs -1556 (n = 28/31; CI +1307..+4588); in 2021-23 it is absent or reversed (2021: 789 vs 369, 2022: 708 vs 787).
  S15d, 2024-26: -71 vs -341 (CI +6..+539), all years: -166 vs -141 (CI -184..+142). So it is a 2024-26 effect, and with small samples at large pumps.
* Mechanism candidates, tested: (a) market beta to a bounce after BTC weakness: **rejected**: BTC's own next-72h return is +1.2% in both arms (2024-26, S15d >= 0.25) and BTC trailing 7d vs forward 72h correlation is 0.05.
  (b) weaker fade plus more squeezes: **supported**: in 2024-26 with BTC weak, median fade +614 vs +586 (S15d) is flat but the doubling rate is 4.9% vs 2.7% and sq50 12.3% vs 10.8%; by tercile 2025-26 low-BTC doubles 5.8% vs 4.0% high;
  mean coin 72h return -1.3% vs -7.5% (median -7.4% vs -12.1%) in 2024-26 at >= 0.25. (c, hypothesis, untested here) thin-market state: crowded shorts (negative funding at entry) concentrate in weak-BTC weeks.

## What blocks the idea

* **Symmetry of the payoff.** The return is +1380 bp median, -77% average on the 6% doublings; with liquidation at +100% each doubling costs -100% of stake (plus cost), so
  the edge needs the doubling rate below about 8-10% at constant median fade, and it was 3-10% by year, drifting up in 2026 (S40: 9.6%).
* **Cost that scales with the tail**: funding (+236 bp avg, +682 bp in squeezes, 48-60% negative-funding entries in 2025-26) and a liquidation rule that crystallises retraces (-140 bp).
* **No stop, no filter, no model works**: stops cut winners 3:1, the only squeeze correlate (crowded shorts) is weak (AUC 0.37), model ranking is pump size, and pump size is the squeeze driver.
* **Tiny effective sample.** S40 has 385 events (24 doublings, 1-8 per year); S15 has 3,929 episodes. Every CI on the 2025-26 vs 2022-24 contrast crosses zero except in the repeated-spike S15.
* **Execution tails not in the data**: the one frozen perp (ALPACA) alone is worth 134 bp/trade; liquidity at the pump and delisting risk are not modelled.

## What would have to be true for it to work

1. A cap or hedge on the tail that does not sell retraces: for example a defined-risk instrument (long calls/cap; not available on this venue), or a cross-margin hedge with a +100% trigger that
   reopens the short; the data say the +100% touch has a 17% profit rate afterwards, so a pure exit loses that. (Hypothesis; untested here.)
2. Funding that does not penalise the side: entry only when the funding rate is positive (high ls_acc tercile pays 82 bp vs 333 bp, net +231 vs -133), which cuts out roughly half of 2025-26 pumps.
   This is a descriptive split from the diagnostics, not a validated rule (in-sample, 1,255 events).
3. A squeeze indicator with AUC well above the 0.63 that crowding gives (e.g. liquidation-cluster or order-book depth data not in this dataset), so the avoided trades are mostly the 6% doublings.
4. A larger, independent sample where first-spike realistic net is clearly positive (> 0 at the lower CI bound). Today S15d is negative in every period, S40 is positive but the CI includes 0 in both halves.
5. Capacity below the thin-book limit (pumped coins are small: perp 24h quote volume median 120-270 MUSD; the 0.5% cap in `sim.py` is what bounds size).
