# Why the capitulation-long variants failed

Scripts (all new, run from `research/capitulation_ls/`): `w3_long_tab.py` (per-row table, written to /tmp),
`w3_long_a.py` (variance decomposition, ICs), `w3_long_b.py` (P&L by day, exit-shape diagnostics),
`w3_long_c.py` (exit rules in the slot book, ensemble, calibration), `w3_long_d.py` (crash-event rankers),
`w3_long_e.py` (tp20day vs day5 in the book).
Winner = day5 pct >= 0.995, 2022+, spot, TP +20%, 72h, 10 slots, cost 0.4%: 162 book trades, +1064 bp/trade,
81% win, 60% TP. Raw candidate rows (no slot limit): 4,224 rows on 26 days; `first_per_event` gives 1,828
(the plan quotes 1,883; I did not reproduce that exact count, 1,774 of 1,828 sit on 12 days with >30 candidates).
"o72" below = the TP20/72h outcome net of 0.4%, per row, on spot paths. IC = correlation of a score with an outcome.

## Summary of causes

1. The edge is a regime call, not a coin call. Of o72 variance on crash rows, 56% is explained by which
   day it is (adj. R2 on 1,247 days); of the 1h outcome only 17%. Any model fit to o72 spends its capacity on
   day/regime structure (yearly IC up to 0.57), where only ~26 flush days exist to learn from.
2. tp20day (CNN on the realistic target) is not worse on the flush days (+1162 bp vs day5 +1064 on the days
   both trade). It loses because it also fires on 48 other days where it makes -248 bp. Its gap to day5
   is day selection, not within-day ranking.
3. A 72h TP outcome is a noisy label for the coin-level thing the CNN can see (the first hour of the
   bounce): within-day it has IC 0.06-0.09 vs 0.12-0.13 for day5 on the same rows, and it is ~57% day-level noise.
4. Exits: no exit rule is distinguishable from fixed 20% at the sample size we have (23-24 independent
   days). The peak distribution is heavy-tailed and barely predictable, so every threshold trades a hit rate
   against a runner tail on a flat curve. The book also rewards capital turnover (mean hold 36h with TP20 vs 72h).
5. Bounces continue after +20% because the right tail is large (mean further high +36%, median +12%) and
   median drift after TP is only +1.6%. A fade loses on the skew and on cost, not because every coin keeps running.
6. The edge is a flush-day phenomenon: 85% of book net comes from 11 broad flush days; the 13 other days
   make +646 bp vs +1205 bp.
7. Architectures plateau at one shared within-day IC (0.122-0.129 vs 1h outcome) and averaging five of them
   adds nothing (0.124). The 1h outcome has ~0 coin-day persistent component: what is predictable is limited to
   a ~1.5% share of within-day variance (IC^2), consistent with the information in 8h of candles being nearly exhausted by one model.

## Evidence

### 1. Label noise: 1h vs 72h outcomes (w3_long_a.py)

Rows 2022+ with scores: 1.46M; crash = ret_4h <= -10%: 49,857 rows on 1,247 days.

Variance explained (adjusted R2 of one-way group means), crash rows:

| outcome | sd | year | year-month | day | symbol |
| --- | --- | --- | --- | --- | --- |
| y1h (clipped +-20%) | 0.059 | 0.007 | 0.069 | 0.170 | 0.047 |
| y4h | 0.091 | 0.020 | 0.104 | 0.257 | 0.088 |
| o72 (TP20/72h) | 0.164 | 0.051 | 0.170 | 0.562 | 0.153 |
| close 72h (clipped) | 0.213 | 0.091 | 0.237 | 0.578 | 0.177 |

All 1.46M rows: day R2 is 0.023 (y1h), 0.279 (o72). Rows with day5 >= 0.995: day R2 0.37 (y1h), 0.54 (o72);
year R2 0.02 vs 0.15. Within-day variance share: y1h 81% (crash) / 98% (all); o72 43% / 72%.

Same score, different target (pooled day-demeaned correlation, all rows 2022+):

| score | y1h | y4h | o72 |
| --- | --- | --- | --- |
| day5 | +0.128 | +0.142 | +0.125 |
| tp20day | +0.073 | +0.084 | +0.087 |

On crash rows: day5 vs o72 within-day 0.062, tp20day 0.086; total (across-day) Spearman with o72 is 0.259
(day5) vs 0.331 (tp20day). So tp20day wins on the between-day dimension and loses within-day on the 1h
and 4h outcomes. The 1h outcome is only a partial proxy for the 72h one (day-demeaned corr 0.32 all rows, 0.41
crash rows), which is why training on 1h still gives a usable o72 ranking (0.125 IC): the 1h label is a
less noisy measurement of the thing the first 8h of candles can say. (Interpretation: label noise + regime
contamination; I did not run a label-noise ablation, so the split between the two is not measured.)

### 2. Why realistic-target models learn regimes (w3_long_d.py)

Crash rows 2022+, target = o72 (22,131 rows, 408 days with >= 5 rows). Target day-level variance share 0.57.

| ranker | yearly Spearman 2022..26 | within-day pooled | per-day Spearman (n>=5 / n>=30) | top-1% picks |
| --- | --- | --- | --- | --- |
| E26 GBM | .12 .21 .28 .49 .17 | +0.169 | 0.062 / 0.129 | 222 picks, 14 days, top-3 days 87% |
| transformer 72h TP20 | .15 .24 .44 .57 .25 | +0.186 | 0.117 / 0.235 | 222 picks, 4 days, 100% in 2022+2025 |
| day5 (crash rows) | .07 .34 .24 .31 .15 | +0.069 | 0.131 / 0.213 | 231 picks, 2 days |

Pooled Pearson is dominated by the biggest flush days, per-day Spearman is not; both are shown because they
disagree about the transformer. Score variance is 65% day-level for GBM/transformer (day5 58%). Top-1% pick lift
decomposes (bp over the crash base +409): GBM +286 day-part + 539 within-day; transformer +909 + 325;
day5 +1243 + 309. The better the yearly IC, the more the lift is day-part. Selection is "pick the flush day",
which exists ~2-4 times a year; with 5 test years that is not enough to also learn coin ranking.

### 3. Why tp20day is half of day5 in the book (w3_long_e.py)

| | raw rows (days) | raw mean o72 | book trades (days) | book bp | yearly |
| --- | --- | --- | --- | --- | --- |
| day5 q.995 | 4,224 (26) | +1311 | 162 (24) | +1064 | 26/40/103/16/15% |
| tp20day q.995 | 6,532 (75) | +1505 | 199 (72) | +517 | 10/47/48/13/-6% |
| day5 q.98 | 25,530 (240) | +763 | 747 | +247 | |
| tp20day q.98 | 21,675 (285) | +727 | 818 | -26 | |

Raw mean o72 is higher for tp20day. In the book it trades day5's 24 days at +1162 bp (108 trades; 49 are the
same rows as day5) and 48 other days at -248 bp (91 trades). The loss is days where there was no broad
flush. The slot book also fills in time order (priority only breaks ties at the same entry time), so the
unconstrained mean overstates the book for either score: +1311 raw vs +1064 for day5.

### 4. Ensembles, seeds, 2h head, architectures (w3_long_c.py, w3_long_a.py)

Within-day IC on 1.46M rows, y1h / y4h / o72: day5 .128/.142/.125, day10b .129/.144/.126, transformer
.123/.138/.122, TCN .129/.142/.126, GRU .127/.139/.126; average-rank ensemble of the five .124/.140/.126.
Score cross-correlation within day on crash rows: day5-day10b 0.99, day5-TCN 0.79, day5-transformer 0.65,
day5-tp20day 0.41. Different networks are not copies, yet their ensemble has the same IC: the extra
variation is noise, not extra signal.

Forecast calibration (day5 path, pool day5 >= 0.98, 25,530 rows): predicted 1h median sd 4.9 vs realised
sd 7.5; q10/q90 coverage 16%/6% (1h) and 16%/10% (8h): the forecast is roughly calibrated in width and biased
somewhat high in level (+3.7 vs +2.2 predicted vs realised at 1h). Over all rows the predicted-median sd is 1.2
vs realised sd 3.4: IC 0.13 means ~1.6% of within-day variance is predictable. Same-coin same-day
persistence of the day-demeaned 1h outcome (ICC among 5,958 coin-days with >= 2 spikes) is -0.03: no stable
coin-day component beyond the spike's own shape. (Speculation: that, plus one shared feature set, explains the plateau;
a ceiling number is not identified by this data.)
The in-pool ICs (inside day5 >= 0.98) of other nets are meaningless as a comparison (range restricted by day5).

### 5. Exits (w3_long_b.py, w3_long_c.py)

Peak (max high within 72h) of the 4,224 raw candidate rows: quantiles 10/25/50/75/90 = 0.09/0.18/0.28/0.46/0.99.
P(peak >= 10/15/20/25/30/50/100%) = 0.89/0.80/0.72/0.59/0.45/0.23/0.10. Rows that hit 20% have median peak 0.35,
90th pct 1.21. Median time to hit 20% is 9h; 37% within 1h, 83% within 24h. Corr(day5 pct, peak) 0.36
(0.31 among hits): the runner tail is only weakly predictable.

Unconstrained mean per row (cost 0.4%), no slot limit: TP5 +414, TP10 +731, TP20 +1311, TP30 +1576, TP50 +1881,
TP75 +2202, no TP 72h +2489 bp. Longer is better unconstrained, because capital is not scarce and mean
reward lives in the right tail (mean 72h close +25%, median +16%). In the 10-slot book it reverses:

| rule (book, per-year fresh $10k) | trades | bp | yearly % | mean hold h |
| --- | --- | --- | --- | --- |
| TP20/72h | 162 | +1064 | 26/40/103/16/15 | 36 |
| TP30/72h | 154 | +1202 | 18/38/125/17/24 | 52 |
| TP50/72h | 149 | +1121 | 6/42/120/9/25 | 68 |
| TP10/72h | 203 | +650 | 14/29/69/32/6 | 14 |
| no TP, 72h | 148 | +973 | -12/38/117/9/25 | 72 |
| no TP, 24h | 160 | +1167 | 27/32/143/1/31 | 24 |
| trail arm .2 / .1, 72h | 153 | +1029 | 16/41/88/13/19 | 50 |
| trail arm .3 / .1, 72h | 149 | +1077 | 6/39/116/10/21 | 62 |

Day-cluster bootstrap of per-trade difference vs TP20 (about 23 days): trail .3/.1 +16 bp [-275, +294]; trail
.2/.1 -34 [-179, +118]; no TP -88 [-479, +272]; TP30 +141 [-32, +310]. So "fixed 20% beats trailing" is not
shown; the data say it is indistinguishable, and TP30 is nominally higher. What is real: nothing beats it by
a margin the data can see, and the tail loss is the same (p5 net -1787 bp, worst -100%, LUNA, for every rule).

After the +20% hit (rows that hit, 71.5% of 4,224 = about 3,020 rows): close vs the 1.2 level at
+1h / +4h / +24h / 72h = mean +730/+862/+612/+1374 bp, median +67/+170/+159/+161 bp, P(up) .58/.62/.58/.56.
Further max high after TP: median +12%, mean +36%, P(+10% more) 57%, P(+25% more) 32%. Worst low after TP:
median -8%, P(< -10%) 39%, P(< -20%) 15%. Fading at the TP level: -902 bp (4h), -652 (24h), -1414 (72h).
So continuation is a skew effect (mean >> median), and the typical coin is flat to slightly up; a fade loses on the right tail and on 0.4% cost.

Reversal frequency (best close reached the arm level, then to 72h): arm .2: 67% of rows arm; mean best +58%,
mean 72h close +39%, giving back 18.5 points on average; 40% finish below 20%, 5% below 0. Trailing exits cut
the winners at the first 10% pullback, which on hourly bars of volatile coins is frequent (trail exits fire on 57% of rows
at arm .2). Unconstrained, trail .3/.1 gets +1771 bp vs TP20 +1311, but that is not reproduced in the book after slot blocking.

### 6. Flush days (w3_long_b.py)

Book trades by day (24 trading days, 162 trades): top 1 day (2022-05-11) = 15% of summed per-trade net, top 3 = 38%,
top 5 = 61%, top 11 = 107% (the rest net negative). Days with >30 candidates: 11 days, 121 trades, +1205 bp,
85% of net; other 13 days, 41 trades, +646 bp, 15%. Mean net by day (days with >= 5 trades, n=15): min -1072,
median +1511, max +1960; 3 negative. 98 TP trades make exactly +1960 bp each; 64 time exits make -309 bp
each. That makes the book roughly "did the flush bounce 20% (60% of the time) or not."

Crash-row outcome by breadth of the day (crash rows, 2022+):

| crash rows that day | days | rows | o72 bp | hit20 | y1h bp | BTC 4h % |
| --- | --- | --- | --- | --- | --- | --- |
| 1-2 | 339 | 484 | +80 | 0.32 | +92 | -0.2 |
| 3-9 | 389 | 2,071 | -152 | 0.27 | +60 | -0.4 |
| 10-29 | 322 | 5,564 | -236 | 0.25 | +0 | -0.5 |
| 30+ | 197 | 41,738 | +430 | 0.36 | +102 | -3.7 |

4,217 of the 4,224 day5 >= 0.995 rows fall on 30+ days: day5 at this threshold is a broad-flush detector with
within-day ranking. Mid-size days (3-29) lose, not a smooth scale.

## What blocks improvement

- Sample: ~24 independent days drive the book; every comparison in section 5 has a CI of +-250 bp per trade, wider than any exit effect.
- The within-day signal is small (IC ~0.13, ~1.6% variance) and shared across five architectures: there is no
  unexploited architecture-specific information to extract from the 8h candles.
- Realistic labels are 56% day-level; learning them needs many more independent flush days than exist.
- Slot book fills in time order and the book mean lies below the unconstrained mean; improving ranking inside a day matters
  less than which spikes arrive early on a flush day (speculation: not tested directly).
- Heavy tail: 10% of candidates run >= +98% and are only weakly predicted (corr 0.31-0.36).

## What would have to be true

- For realistic-target models: many more broad-flush days (or a way to hold the day effect fixed in the loss, e.g. a
  day-demeaned o72 target with day-level gating done separately) so that within-day IC on o72 exceeds day5's 0.125.
- For exits: >= 100 independent flush days to resolve 100-200 bp differences, or a runner predictor with corr well
  above 0.36 to justify holding past +20%.
- For fades: post-TP median drift would have to be negative (it is +1.6%) or fees under ~0.1%.
- For ensembles/architectures: new information (order book, funding/OI at entry, cross-coin context), since the
  candle-only networks agree at 0.65-0.99 and share one IC.
