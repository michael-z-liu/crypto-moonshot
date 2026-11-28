# Why no exit rule beats the plain 24h hold on sleeve C

Script: `w10_c_exits.py` (rebuilds C as in `e25_combined.sleeve_crash_short`: perp, ret_4h <= -15%, |BTC 4h| <= 1%, day5 CNN pct <= 0.5, 2022+, isolated 1x, 0.3% cost + funding, yearly books). The
24h baseline reproduces +436 bp / 162 trades; the exit variants reproduce the test (4/8/12h +215/+176/+222, TP10/15 +100/+154, trail5 +68).

Setup fact: the pool is 922 five-minute rows but only 162 entries. These are about 161 events (one position per coin while held); 5 or 10 slots give the same book. Slots never bind, so
"freed slots" here means same-coin re-entry, not new events.

## 1. The average path does not give back; the distribution does

Short-favourable close (%) by hour after entry, 162 trades (the 3 stopped trades keep their hypothetical close, so read the means with that in mind):

| hr | 1 | 2 | 4 | 8 | 12 | 24 | 36 | 48 | 72 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mean all | +2.4 | +2.3 | +3.4 | +3.9 | +4.5 | +4.0 | +5.3 | +4.3 | +2.6 |
| median all | +0.7 | +1.4 | +2.8 | +2.5 | +4.3 | +6.1 | +7.1 | +7.8 | +9.6 |
| mean winners (n=98, 24h net>0) | +6.1 | +6.6 | +9.7 | +11.2 | +13.6 | +19.4 | +20.1 | +18.9 | +18.5 |
| median winners | +2.0 | +3.2 | +6.9 | +8.1 | +10.7 | +14.9 | +14.8 | +16.9 | +20.1 |
| median losers (n=64) | -3.2 | -2.8 | -5.4 | -5.5 | -5.7 | -6.3 | -6.2 | -3.8 | -5.3 |
| mean losers | -3.3 | -4.4 | -6.2 | -7.3 | -9.5 | -19.9 | -17.8 | -18.4 | -22.4 |

- The median path rises almost monotonically to 72h (+9.6%); the mean is flat at +3..5% from 4h on. No average retrace exists to sell. Winners and losers separate within hours (losers are -5% by 4h and
  never recover), so the "giveback" in the scorecard (0.67) is the max of a noisy path versus its end, not a systematic fade. mfe24 has median 13% against a median close of 6%, which is what any
  volatile path produces.
- The payoff is a right-skewed lottery: median trade +3.7% net, 25th pct -4.8%, 95th +43%; the top 10% of trades contribute 109% of total net (top 20%: 167%). Without the 3 stopped trades the mean
  is +633 bp.

## 2. Take profits cap the right tail

Of 162 trades, 100 (62%) touch +10% favourable within 24h (median 1.6h to touch) and 76 (47%) touch +15% (median 2.8h). Of those that touched:

- +10%: 63% close the 24h *above* +10% (median close +15.0%), 24% between 0 and +10%, 13% below 0.
- +15%: 66% close above +15% (median +18.3%), 25% between 0 and +15%, 9% below 0.
- 48 (L=10%) / 44 (L=15%) of the touchers fell back by 5pp or more at some hour after touching; about half of those (22 / 21) then finished above L at 24h. The retrace-and-resume case is roughly half of all retraces.

Per-trade difference, rule minus 24h hold on the same 162 entries (`w10_c_exits.py` decomposition):

| rule | net diff | better on | worse on | cost on trades whose 24h net > +20% |
| --- | --- | --- | --- | --- |
| TP10 | -292 bp | 41 trades, +328 bp | 58, -620 bp | 33 trades, -544 bp |
| TP15 | -210 | 29, +248 | 46, -458 | 33, -442 |
| trail5 | -312 | 39, +317 | 67, -629 | 29, -468 |
| 4h | -233 | 62, +418 | 98, -652 | 30, -361 |
| 8h | -234 | 55, +337 | 105, -570 | 28, -305 |
| 12h | -198 | 61, +292 | 99, -490 | 29, -269 |

The saved fall-backs are real (+250..+420 bp) but are smaller than what the ~30 big winners (24h net +20..+77%) give up, and those winners exit early too: the five worst diffs are -4.5..-6.7
pp of stake each on trades that returned +53..+77% at 24h. The trade that touches +10% at hour 2 is the same trade that goes on to -30% in 24h.

## 3. Freed slots add trades, and they are worse

Re-entry means entering the same coin again after the TP or trail exits inside the same event (median gap about 0 h, i.e. the next candidate row):

| rule | extra trades | their mean | win | event mean (first trade + re-entries) | 24h plain |
| --- | --- | --- | --- | --- | --- |
| TP10 | 100 (in 40 events; 65 in 2025) | +29 bp | 80% | +163 bp | +436 |
| TP15 | 59 | -44 | 71% | +212 | +436 |
| trail5 | 62 | -78 | 71% | +95 | +436 |
| 8h | 5 (all 2025) | -703 | 40% | | |
| 4h | 11 | +387 | 82% | | |

First trades on the original entries earn +144 / +226 / +124 bp (versus +436 at 24h). The re-entries are shallow, already-extended entries (late in a crash, ret_4h from the exit level), earn
about the cost of a round trip, and still carry the stop tail; per event the chain still ends well below the single 24h position. In 2025 (the year of many small trades) they turn the rule into
churn: +83..+89 bp per trade over 114-141 trades vs +363 at 24h.

## 4. Why 2022 is negative for short holds

12 trades, 9 distinct dates, 7 of them in the Terra / LUNA collapse (LUNA x2, ANC x2, SXP on 5/10-5/11, RSR and UNFI on 6/8). By hour (mean fav %): 4h +8.8, 8h +7.0, 12h +6.6, 24h +23.9.

- The Terra legs come in later waves: ANC 5/10 is -0.1% at 8h and -2.1% at 12h, then +43.6% at 24h; ANC 5/11 is +37.5% at 4h, -2.6% at 12h, +39.6% at 24h. Short holds sit in the lull between waves.
- One trade dominates: LUNA 2022-05-11 13:45 hit the 100% adverse stop within 40 min (a +232% 5m spike) and loses -100% of the stake in every variant. At 24h the big Terra winners (+30..+47% net) absorb it:
  12-trade mean +663 bp, or about +1,635 bp without it. At 4h/8h/12h the same loss is not covered (-117 / -604 / -629 bp; about +781 / +250 / +223 bp without it).
- GALA 11/03 (-33% at 24h, never better than +3%) is the other big loser; no exit helps it.
So the negative 2022 short-hold figure is (a) one data-point stop and (b) the loss of the two-wave Terra winners, on n=12.

## 5. What blocks exit-based improvement, and what would have to be true

Blockers:
1. The edge is a continuation tail, and its entry is the signal itself: the trades that reach +10% early are, 63% of the time, the ones that end above it. Exits cannot tell the 30 trades that
   continue from the ones that stall; both look identical in the first hours (path of winners and losers is separable in sign at 4h, but not in size).
2. Mean path is flat after about 4h while the median keeps rising: there is no systematic retrace for a stop or TP to monetise. The 0.67 giveback is a statistic of noisy maxima.
3. Costs/stop tail are concentrated in 3 stopped trades (LUNA, ALPACA, TUTU; -100% each) that come in the first minutes to hours, which no TP/trail touches; only sizing or an entry filter addresses them.
4. Early exit converts the strategy into re-entry churn on the same event, at worse prices and with 0.3% cost each time.
5. n is small (162; 2022-24 = 48 trades) and the top 10% trades are 109% of the total, so any rule is tested on about 15 tail trades.

What would have to be true for an exit to help:
- a signal available at the exit time that separates stalling from continuing trades (e.g. a funding/open-interest or BTC-correlation state at hour 2-4) so that exits apply only to the stalled set. That is a
  conditional-exit model, not a TP/trail level; with 162 trades and 30 tail winners it is not testable without overfitting.
- or a path with a measurable mean retrace (as the L sleeve's TP20 has, 65% hit rate): C's trades hit +20% only about 25-30% of the time with a continuing distribution above.
- or a much larger event set (more years, more venues), so the tail trades number in the hundreds.
Improvement for C has to come from the entry (fewer stalled/LUNA-type trades) or from sizing, not exit timing, consistent with the diag README's lever table which pointed at "exit timing or a better entry
signal" and this note shows the former is exhausted.
