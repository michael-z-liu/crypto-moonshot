# X9: entry features for crash shorts C (candidates n=161, 144 days, mean +436 bp; 2022-24 n=47, 2025-26 n=114)

Mean net (bp) by feature tercile (low/mid/high), within each period; IC = Spearman rank IC
with net over all candidates; IC_day = within-day IC (days with >= 3 candidates).

| feature | missing | 22-24 low/mid/high | 25-26 low/mid/high | IC | IC_day |
|---|---|---|---|---|---|
| ret_4h | 0% | +116/+574/+812 | +653/+66/+503 | -0.001 | +nan |
| ret_24h | 0% | -45/+712/+837 | +229/+860/+165 | +0.058 | +nan |
| r1h | 0% | +430/+448/+612 | -268/+650/+879 | +0.099 | +nan |
| r15m | 0% | +434/+189/+883 | +74/+610/+568 | +0.077 | +nan |
| vol_mult | 0% | +906/+820/-293 | +268/+770/+213 | -0.120 | +nan |
| off_low4h | 0% | +581/+287/+623 | +99/+780/+377 | +0.060 | +nan |
| off_high24h | 0% | +420/+433/+638 | +339/+3/+887 | +0.126 | +nan |
| dlo30d | 0% | +177/+268/+1073 | +372/+651/+222 | +0.049 | +nan |
| dhi30d | 0% | +522/-190/+1194 | +641/+143/+440 | +0.012 | +nan |
| r7d | 0% | +51/+394/+1073 | +680/+148/+395 | +0.076 | +nan |
| r30d | 0% | +138/+474/+895 | +474/+202/+547 | +0.078 | +nan |
| vol30d | 0% | +653/+941/-152 | +113/+964/+178 | +0.015 | +nan |
| qv7_30 | 0% | +802/+205/+474 | +50/+594/+607 | +0.130 | +nan |
| listed_days | 0% | +576/+788/+94 | +180/+380/+682 | +0.035 | +nan |
| taker1h | 0% | +144/+796/+545 | +167/+419/+658 | +0.090 | +nan |
| wick_dn | 0% | +712/+719/+21 | +645/+139/+440 | -0.050 | +nan |
| n_prior_pumps30 | 0% | +317/+nan/+1010 | +168/+807/+618 | +0.092 | +nan |
| rate | 1% | +435/+647/+382 | +473/+161/+1338 | -0.020 | +nan |
| rate24 | 1% | +435/+1208/-107 | +405/-24/+868 | -0.033 | +nan |
| mkt_r4h | 0% | +150/+655/+688 | -65/+238/+1073 | +0.085 | +nan |
| mkt_r24h | 0% | +512/+435/+538 | +261/+514/+469 | +0.042 | +nan |
| btc_r24h | 0% | +838/-220/+889 | +172/+327/+742 | +0.058 | +nan |
| spike_share | 0% | +481/+544/+455 | +715/+73/+432 | +0.041 | +nan |
| oi_ch1h | 0% | +401/+967/+89 | -275/+993/+553 | +0.002 | +nan |
| oi_ch4h | 0% | +855/+734/-147 | +156/+675/+420 | -0.023 | +nan |
| oi_ch24h | 0% | +1011/+733/-312 | +679/+529/+25 | -0.087 | +nan |
| oi_to_vol | 0% | +665/+283/+537 | +29/+262/+953 | +0.019 | +nan |
| ls_acc | 0% | +1087/+886/-556 | +846/-32/+400 | -0.077 | +nan |
| top_ls_pos | 0% | +710/+216/+687 | +1139/+744/-657 | -0.129 | +nan |
| taker_ls_1h | 0% | +697/+450/+324 | +118/+793/+344 | -0.017 | +nan |
| cnn | 0% | +397/+1007/+50 | +13/+580/+659 | +0.011 | +nan |

## Pre-declared gates (C candidates, then the LMC book with C gated)

| gate | n | E_o | win | share_stopped | giveback_med | 22-24 bp | 25-26 bp | LMC bp | LMC trades | LMC min yr | LMC years |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 161 | +0.044 | 0.60 | 0.02 | 0.67 | +494 | +412 | +1158 | 430 | +51 | +62/+81/+156/+625/+51 |
| off_low4h <= 3% (no bounce yet) | 74 | +0.037 | 0.58 | 0.01 | 0.67 | +524 | +295 | +1294 | 351 | +29 | +44/+80/+148/+562/+29 |
| r30d <= 0 (already weak) | 48 | +0.032 | 0.54 | 0.02 | 0.80 | -104 | +498 | +1381 | 327 | +27 | +55/+71/+128/+602/+27 |
| dhi30d <= -40% (deep under 30d high) | 64 | +0.067 | 0.61 | 0.03 | 0.65 | +588 | +692 | +1409 | 335 | +51 | +64/+73/+129/+565/+51 |
| rate >= 0 (shorts not paid) | 80 | +0.040 | 0.56 | 0.01 | 0.70 | +396 | +396 | +1274 | 356 | +28 | +48/+73/+140/+602/+28 |
| ls_acc >= 1 (shorts not crowded) | 92 | +0.020 | 0.62 | 0.02 | 0.71 | +198 | +199 | +1228 | 368 | +47 | +59/+77/+133/+464/+47 |
| oi_ch4h >= 0 (OI rising into crash) | 91 | +0.066 | 0.66 | 0.02 | 0.66 | +240 | +837 | +1350 | 363 | +45 | +57/+82/+128/+706/+45 |
| vol_mult top half | 81 | +0.036 | 0.57 | 0.01 | 0.79 | +95 | +472 | +1294 | 356 | +41 | +41/+78/+151/+529/+50 |
| listed_days >= 365 | 99 | +0.053 | 0.62 | 0.03 | 0.59 | +291 | +710 | +1293 | 378 | +50 | +52/+80/+153/+634/+50 |

IC_day is empty: the 161 candidates fall on 144 days, so almost no day has three candidates and
C is effectively one trade per day (unlike L, where the day decides everything).

## Verdict: no entry feature improves C (fails)

- **No stable feature.** Of 31 features, none has the same tercile ordering in 2022-24 (n=47)
  and 2025-26 (n=114) with a meaningful gap; the largest rank ICs are +-0.13 (qv7_30, top_ls_pos,
  off_high24h, vol_mult), within noise for n=161 (1/sqrt(n) = 0.08). Crowding (ls_acc,
  top_ls_pos) looks strong in one period and flips in the other.
- **Gates raise LMC bp per trade only by removing C trades.** C's +4.4% mean is below L and M,
  so any gate lifts the book's mean per trade (+1158 -> +1228..+1409 bp) while cutting yearly
  returns (2025 +625% -> +464..+706%, worst year +51% -> +27..+51%). Only "dhi30d <= -40%" and
  "listed_days >= 365" keep the worst year at +50%, and both lower 2023-24. They were
  pre-declared, but with n=47 in 2022-24 none is distinguishable from chance.

## Why it failed

1. **Too few independent events.** 161 trades in five years, 47 before 2025. A real effect of
   +300 bp between terciles has a standard error of ~+-450 bp per tercile per period.
2. **C's outcome is a few large winners.** As in `why/c_exits.md`, the top 10% of trades carry
   more than the whole net (giveback 0.67, win 60%). An entry feature has to predict which
   trades keep falling the most (the ~15 largest continuations), and none of the price, volume,
   funding, OI or crowding features does (stopped share 1-3% everywhere, so the tail is not the
   issue: the body is near zero).
3. **The CNN's role is exhausted.** C already requires CNN pct <= 0.5; within that, cnn terciles
   flip (+397/+1007/+50 vs +13/+580/+659).

C stays as pre-registered: a small diversifier that adds trades and years, not an edge to tune.
