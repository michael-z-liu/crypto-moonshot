# X12: perp order-book depth at entry vs the four sleeves (2023-26, in-sample)

Data: Binance USD-M `bookDepth` daily archives (data.binance.vision), ~30-second snapshots of the
cumulative book notional within ±1..5% of mid. `x12_fetch.py` downloaded the entry day and the day
before for every 2023+ candidate's perp: 4,030 symbol-days, 3,900 parsed, 130 not in the archive.
It kept the last snapshot of each 5-minute bucket (1.12M rows, 80 MB in
`OWN/bookdepth/depth5m.parquet`; the zips were deleted). Features use the last snapshot at or before
`t_in`, no older than 15 min, so there is no look-ahead. Features, hypotheses, rules and gates were
declared in the docstring of `x12_book_depth.py` before any outcome was read. `x12_followup.py`
holds the post-hoc checks, labelled as such.

## Verdict: not confirmed. Depth does not beat the existing rules, for L or for the shorts.

- **L, within-day choice:** no pre-declared rule clears the bar. The R1 tie-break (thinnest bids
  relative to volume first) takes L from +1218 to +1192 bp in LMC and drops LMCF's worst year
  from +103% to +98%. R2 (imbalance) is worse: L +1093 bp, min year +86%. The skip rules R3/R4
  raise L bp per trade (+1297/+1287 in LMC) but lower the book: LMC min year 51 -> 46 (R3), and
  LMCF min year 103 -> 96/94. The reason: within a flush bar **the CNN signal is already a better
  ranker** (within-day IC +0.23 in 2023-24, +0.29 in 2025-26, positive on 10 of 12 days). The
  best depth feature, dep5_qv, reaches -0.12, and only -0.07 once the CNN rank is removed. The
  base book already orders same-bar ties by CNN (`prio = 3 + pct`), so a depth-first order throws
  away information. A CNN + depth composite (post hoc) is a wash: L +1236 vs +1218 bp, LMC
  years +80/+152/+636/+51 vs +81/+156/+616/+51.
- **L, between days:** a raw effect exists but fades. Thin books relative to volume bounced more
  in 2023-24: dep5_qv terciles +1492/+1264/+610 bp, raw IC -0.29. In 2025-26 the spread shrinks
  to +1770/+1630/+1567. Most of the raw IC comes from differences between days, which a slot
  book cannot use.
- **F/M/C, tails:** the thin-ask hypothesis holds only for F, and too few trades to use. All 7
  F stop-outs had ask2_qv < 0.027 and small absolute ask depth (<= $145k within 1% above mid).
  F stopped share by ask1 tercile is 14/2/0%, and it is flat by coin size (qv terciles 5/7/5%),
  so this is not just "small coins". But the 0.027 cut drops 76 of 128 F trades (59%), and the
  dropped group still averages +696 bp. For M the sign is reversed: the gate removes M's best
  trades (dropped +3623 bp vs kept +1530) while the stop-outs stay (18% stopped among kept vs 5%
  among dropped). C has 2 stops in total, and depth tells its trades apart neither in net nor in
  stops.
- **Books:** every short gate lowers the worst year. G1 (ask2_qv) LMC min 51 -> 26, LMCF
  103 -> 31; G4 (dep1_qv) 51 -> 32 and 103 -> 37. G2 (imb1) lowers it slightly (49, 83). G3
  (book halved in 24h) raises LMC's worst year to +61, but it does so by moving return out of
  the other years: 2025 falls from +616% to +423%, and LMCF ends at 101 vs 103. Applying the gate to F only (post hoc) still cuts the LMCF worst year 103 -> 57
  (worst DD -32% -> -17%). The coin-size control (F only, qv gate) is 97.

## Main numbers (10 slots, $10k a year, 2023-26)

| book | rule | trades | bp/trade | L bp | min year % | worst DD % | 2023/24/25/26 % |
|---|---|---|---|---|---|---|---|
| LMC | base | 392 | +1133 | +1218 | +51 | -17 | +81/+156/+616/+51 |
| LMC | R1 L tie: low bid2_qv first | 391 | +1125 | +1192 | +51 | -17 | +76/+145/+637/+51 |
| LMC | R3 L skip: thick bids | 377 | +1152 | +1297 | +46 | -17 | +79/+162/+575/+46 |
| LMC | R4 L skip: bid-heavy | 379 | +1150 | +1287 | +55 | -17 | +73/+151/+596/+55 |
| LMC | G1 shorts: ask2_qv >= 0.027 | 281 | +1184 | +1224 | +26 | -16 | +63/+158/+256/+26 |
| LMC | P0 L tie: CNN + depth (post hoc) | 391 | +1139 | +1236 | +51 | -17 | +80/+152/+636/+51 |
| LMCF | base | 488 | +1075 | +1234 | +103 | -32 | +104/+145/+812/+103 |
| LMCF | R1 L tie: low bid2_qv first | 486 | +1069 | +1193 | +98 | -32 | +98/+135/+827/+103 |
| LMCF | R4 L skip: bid-heavy | 478 | +1088 | +1343 | +94 | -32 | +94/+147/+787/+108 |
| LMCF | G1 shorts: ask2_qv >= 0.027 | 322 | +1196 | +1234 | +31 | -14 | +77/+194/+347/+31 |
| LMCF | P3 F only: ask2_qv >= 0.027 (post hoc) | 430 | +1146 | +1228 | +57 | -17 | +97/+192/+760/+57 |

Only two rules raise LMC's worst year. R4 does it by a little (+55 vs +51) and costs 2023 and
2025 (+73 vs +81, +596 vs +616). G3 (+61) does it by giving up a third of 2025. In LMCF both
lower the worst year (R4 94, G3 101 vs 103). An in-sample effect that changes sign between the
two books is not a rule.

## Why it failed

1. **The bar was the CNN, not "the first 10".** Within a flush, nearly all of a day's L
   candidates arrive on one or two 5-minute bars (2024-01-03: 183 of 185 on the first bar), so the
   book's choice is the same-bar order, and the base order is the CNN's. The CNN's within-day IC
   is +0.23..+0.29; depth's is -0.02..-0.12, and -0.02..-0.09 once the CNN is partialled out.
   Even with look-ahead, the 10 lowest-ask2_qv candidates per day beat the CNN's first 10 by only
   +175 bp (+1409 vs +1234). A composite gains +34 bp in that offline test and +18 bp of L
   in the book.
2. **The L depth effect is mostly between days, and it fades.** Raw IC dep5_qv -0.29, but
   within-day -0.12. Its tercile spread is 882 bp in 2023-24 and 203 bp in 2025-26. Depth/qv
   is strongly tied to coin size within a day (r = -0.59 with log qv), although the
   size-partialled within-day IC stays at -0.107.
3. **The short tail is too rare to gate.** F has 7 stop-outs and M has 21 among the candidates
   with depth. The thin-ask F signal has a positive predictive value of 7/76 = 9%, and the other
   91% of the gated trades earn +696 bp on average. Within M, stops do lean towards thin asks
   (stopped 20/19/9% by M's own ask2_qv tercile). But the low tercile's net is unstable:
   +481 bp in 2023-24 and +2730 bp in 2025-26. The pooled cut reaches only M's thinnest 16%,
   which holds M's best listings (+3623 bp, 5% stopped).
4. **Book path dependence magnifies small gates.** Dropping F trades in 2026 freed slots for
   worse trades: LMCF 2026 went from +103% to +57% under P3, although the F trades it removed
   averaged less than the ones it kept.

## Open leads (not tested here)

- M stop tail vs `imb5` (stopped 27/12/9% by tercile, IC +0.19) and `ch1` (book thinning in the
  last 24h: stopped 23/19/7%, IC +0.28). Both are post hoc and have about 21 stops; they would
  need a pre-declared test on new listings.
- A depth-aware stake size for F (smaller stake when ask1 < $150k), instead of a skip. This
  would cap the 7 stop-outs without freeing slots.

Disk: `OWN/bookdepth/` 80 MB (`depth5m.parquet`, `x12_feats.parquet`, `fetch_log.csv`); `raw/` is empty.

---

# Appendix A: pre-declared tables (written by x12_book_depth.py to book_depth_tables.md)

See the docstring of `x12_book_depth.py` for the pre-declared features, rules and gates.

## Coverage (candidates with a depth snapshot <= 15 min before entry)

| sleeve | candidates | with perp | with depth | with 24h-earlier depth | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|
| L | 1810 | 1770 | 1708 | 1703 | 132/138 | 983/1059 | 561/581 | 32/32 |
| F | 128 | 128 | 128 | 128 | 11/11 | 35/35 | 57/57 | 25/25 |
| M | 132 | 132 | 129 | 128 | 15/15 | 30/30 | 70/73 | 14/14 |
| C | 149 | 149 | 148 | 146 | 12/12 | 22/23 | 76/76 | 38/38 |

Mean net of candidates with vs without depth (bp):

- L: with +1308 (n=1708), without +1073 (n=102)
- F: with +849 (n=128), without +0 (n=0)
- M: with +1829 (n=129), without +3355 (n=3)
- C: with +411 (n=148), without +1381 (n=1)

## Sleeve L: feature terciles (n=1708; mean net bp; terciles within each period)

| feature | 23-24 bp low/mid/high | 25-26 bp low/mid/high | IC | IC_day (days) |
|---|---|---|---|---|
| bid1 | +1316/+1066/+985 | +1667/+1648/+1657 | -0.121 | -0.001 (14) |
| bid2 | +1284/+1048/+1035 | +1653/+1569/+1745 | -0.105 | +0.004 (14) |
| bid5 | +1206/+1131/+1030 | +1619/+1621/+1727 | -0.106 | +0.003 (14) |
| ask1 | +1278/+1144/+944 | +1676/+1661/+1635 | -0.036 | +0.001 (14) |
| ask2 | +1351/+1108/+908 | +1666/+1641/+1668 | -0.082 | -0.008 (14) |
| ask5 | +1393/+1099/+874 | +1687/+1651/+1636 | -0.133 | -0.038 (14) |
| imb1 | +1070/+1096/+1201 | +1637/+1738/+1597 | -0.034 | -0.004 (14) |
| imb5 | +767/+1210/+1392 | +1638/+1671/+1659 | +0.070 | +0.028 (14) |
| dep1_qv | +1439/+1302/+625 | +1749/+1646/+1580 | -0.176 | -0.074 (14) |
| dep5_qv | +1492/+1264/+610 | +1770/+1630/+1567 | -0.289 | -0.121 (14) |
| bid2_qv | +1440/+1184/+741 | +1648/+1695/+1625 | -0.241 | -0.054 (14) |
| ask2_qv | +1469/+1245/+651 | +1657/+1727/+1591 | -0.203 | -0.076 (14) |
| ch1 | +1220/+1038/+1109 | +1769/+1715/+1488 | +0.031 | -0.057 (14) |
| ch5 | +1359/+946/+1062 | +1814/+1653/+1499 | -0.150 | -0.132 (14) |

## Sleeve F: feature terciles (n=128; mean net bp; terciles within each period)

| feature | 23-24 bp low/mid/high | 25-26 bp low/mid/high | IC | IC_day (days) | stopped % low/mid/high | squeeze % low/mid/high |
|---|---|---|---|---|---|---|
| bid1 | -251/+1015/+504 | +723/+1243/+1334 | -0.088 | +nan (0) | 14/2/0 | 35/23/24 |
| bid2 | -14/+52/+1214 | -111/+2316/+1127 | -0.123 | +nan (0) | 14/2/0 | 37/21/24 |
| bid5 | +46/+421/+782 | +643/+1814/+847 | -0.142 | +nan (0) | 14/2/0 | 37/21/24 |
| ask1 | +59/+886/+302 | +816/+2008/+473 | -0.259 | +nan (0) | 14/2/0 | 33/26/24 |
| ask2 | +54/+951/+243 | +649/+2051/+603 | -0.223 | +nan (0) | 14/2/0 | 28/30/24 |
| ask5 | +97/+500/+648 | +1083/+1207/+997 | -0.232 | +nan (0) | 12/5/0 | 28/26/29 |
| imb1 | +864/-112/+442 | +859/+223/+2213 | +0.123 | +nan (0) | 5/9/2 | 28/30/24 |
| imb5 | +1167/-826/+833 | +386/+2173/+754 | -0.011 | +nan (0) | 7/5/5 | 33/35/14 |
| dep1_qv | -197/+551/+911 | +2289/-105/+1060 | -0.147 | +nan (0) | 5/12/0 | 35/30/17 |
| dep5_qv | -673/+878/+1091 | +2016/-346/+1583 | -0.045 | +nan (0) | 9/5/2 | 33/33/17 |
| bid2_qv | +44/+234/+971 | +1032/+992/+1265 | +0.020 | +nan (0) | 12/2/2 | 37/28/17 |
| ask2_qv | -915/+1070/+1158 | +1659/+699/+907 | -0.101 | +nan (0) | 7/9/0 | 33/33/17 |
| ch1 | +333/+861/+36 | +830/+1187/+1279 | +0.137 | +nan (0) | 2/9/5 | 19/26/38 |
| ch5 | +152/+731/+358 | +850/+1064/+1398 | +0.140 | +nan (0) | 5/5/7 | 23/26/33 |

## Sleeve M: feature terciles (n=129; mean net bp; terciles within each period)

| feature | 23-24 bp low/mid/high | 25-26 bp low/mid/high | IC | IC_day (days) | stopped % low/mid/high | squeeze % low/mid/high |
|---|---|---|---|---|---|---|
| bid1 | +1342/-56/+1831 | +2149/+1668/+2872 | +0.092 | +nan (0) | 18/17/14 | 34/24/35 |
| bid2 | +1675/-566/+1952 | +2113/+1358/+3207 | +0.072 | +nan (0) | 16/19/14 | 34/26/33 |
| bid5 | +168/+238/+2808 | +1768/+2090/+2858 | +0.088 | +nan (0) | 18/14/16 | 30/29/35 |
| ask1 | +1501/-100/+1703 | +1777/+2103/+2836 | +0.076 | +nan (0) | 16/17/16 | 34/24/35 |
| ask2 | +1725/+461/+940 | +1219/+2703/+2837 | +0.029 | +nan (0) | 14/19/16 | 32/29/33 |
| ask5 | +2045/-225/+1239 | +1745/+2437/+2548 | -0.040 | +nan (0) | 14/12/23 | 25/26/42 |
| imb1 | +281/+1822/+1210 | +2992/+2044/+1636 | -0.060 | +nan (0) | 23/12/14 | 32/24/37 |
| imb5 | -80/+961/+2399 | +1577/+2886/+2289 | +0.192 | +nan (0) | 27/12/9 | 39/31/23 |
| dep1_qv | +1130/+280/+1743 | +2883/+1084/+2673 | -0.016 | +nan (0) | 16/26/7 | 34/38/21 |
| dep5_qv | +1006/+421/+1744 | +3102/+976/+2551 | -0.090 | +nan (0) | 16/24/9 | 39/31/23 |
| bid2_qv | +936/+468/+1775 | +2521/+1530/+2619 | -0.035 | +nan (0) | 18/24/7 | 39/31/23 |
| ask2_qv | +481/+1451/+1342 | +2730/+1353/+2572 | -0.083 | +nan (0) | 20/19/9 | 36/36/21 |
| ch1 | +1271/+651/+1319 | +1086/+1880/+3768 | +0.279 | +nan (0) | 23/19/7 | 35/33/26 |
| ch5 | +513/+1647/+1064 | +2210/+1884/+2599 | +0.045 | +nan (0) | 26/12/12 | 33/30/31 |

## Sleeve C: feature terciles (n=148; mean net bp; terciles within each period)

| feature | 23-24 bp low/mid/high | 25-26 bp low/mid/high | IC | IC_day (days) | stopped % low/mid/high | squeeze % low/mid/high |
|---|---|---|---|---|---|---|
| bid1 | +333/+612/+286 | +437/+104/+688 | +0.034 | +nan (0) | 4/0/0 | 22/10/2 |
| bid2 | +641/+83/+480 | +209/+275/+787 | +0.041 | +nan (0) | 4/0/0 | 22/10/2 |
| bid5 | -199/+724/+755 | +236/+34/+1007 | +0.083 | +nan (0) | 4/0/0 | 20/12/2 |
| ask1 | +203/+1093/-52 | +217/+580/+449 | +0.023 | +nan (0) | 4/0/0 | 20/10/4 |
| ask2 | +793/-392/+788 | +9/+932/+321 | +0.008 | +nan (0) | 4/0/0 | 18/10/6 |
| ask5 | +292/+452/+491 | +256/+543/+445 | +0.014 | +nan (0) | 4/0/0 | 20/8/6 |
| imb1 | +696/-81/+584 | +517/-5/+712 | -0.011 | +nan (0) | 0/2/2 | 10/14/10 |
| imb5 | +61/+473/+723 | +846/-197/+618 | +0.041 | +nan (0) | 0/2/2 | 16/12/6 |
| dep1_qv | +380/+391/+457 | +508/+452/+276 | -0.087 | +nan (0) | 4/0/0 | 20/10/4 |
| dep5_qv | +482/+461/+274 | +782/+34/+445 | -0.076 | +nan (0) | 4/0/0 | 16/15/4 |
| bid2_qv | +623/+58/+524 | +372/+549/+338 | -0.064 | +nan (0) | 4/0/0 | 20/10/4 |
| ask2_qv | +380/+574/+274 | +450/+772/+24 | -0.107 | +nan (0) | 4/0/0 | 18/8/8 |
| ch1 | +327/+202/+866 | +973/+103/+141 | -0.090 | +nan (0) | 2/2/0 | 12/10/12 |
| ch5 | -142/+1105/+557 | +908/-12/+364 | -0.038 | +nan (0) | 2/2/0 | 12/14/8 |

## Pooled shorts F+M+C (n=405)

| feature | 23-24 bp low/mid/high | 25-26 bp low/mid/high | IC | IC_day (days) | stopped % low/mid/high | squeeze % low/mid/high |
|---|---|---|---|---|---|---|
| bid1 | +541/+557/+845 | +428/+1433/+1625 | +0.053 | +0.070 (20) | 10/4/8 | 26/21/21 |
| bid2 | +602/+228/+1121 | +671/+1196/+1634 | +0.034 | +0.041 (20) | 9/7/7 | 25/22/21 |
| bid5 | +174/+369/+1415 | +690/+1183/+1628 | +0.004 | +0.157 (20) | 10/8/4 | 27/22/19 |
| ask1 | +549/+546/+850 | +819/+1282/+1380 | -0.013 | +0.174 (20) | 10/5/7 | 26/21/21 |
| ask2 | +508/+716/+716 | +844/+1237/+1400 | -0.038 | +0.095 (20) | 10/5/7 | 26/19/24 |
| ask5 | +836/+213/+897 | +837/+1402/+1243 | -0.085 | +0.061 (20) | 9/7/6 | 22/22/24 |
| imb1 | +373/+765/+805 | +1342/+720/+1414 | +0.035 | -0.071 (20) | 6/11/5 | 20/28/21 |
| imb5 | +169/+288/+1503 | +894/+1146/+1459 | +0.083 | -0.094 (20) | 7/10/4 | 29/26/13 |
| dep1_qv | +180/+657/+1113 | +1262/+814/+1400 | +0.066 | -0.048 (20) | 6/6/10 | 24/23/21 |
| dep5_qv | +140/+1000/+802 | +1107/+814/+1571 | +0.066 | -0.050 (20) | 7/7/8 | 24/26/18 |
| bid2_qv | +249/+624/+1077 | +967/+1041/+1489 | +0.107 | -0.035 (20) | 6/7/10 | 25/24/19 |
| ask2_qv | -69/+920/+1098 | +1101/+1115/+1262 | +0.034 | -0.069 (20) | 4/10/8 | 21/28/19 |
| ch1 | +157/+1080/+774 | +1036/+728/+1716 | +0.089 | -0.026 (20) | 7/12/3 | 19/26/23 |
| ch5 | +324/+1127/+557 | +1459/+1022/+1010 | -0.050 | -0.159 (20) | 10/8/4 | 24/23/22 |

## L within-day look-ahead diagnostic (days with >= 20 L candidates with depth)

| feature | days | first-10 bp | 10 lowest bp | 10 highest bp | all bp |
|---|---|---|---|---|---|
| bid2_qv | 11 | +1234 | +1374 | +1244 | +1311 |
| imb1 | 11 | +1234 | +1319 | +1312 | +1312 |
| dep1_qv | 11 | +1234 | +1360 | +1207 | +1313 |
| ask2_qv | 11 | +1234 | +1409 | +1222 | +1313 |
| ch5 | 11 | +1234 | +1300 | +1056 | +1311 |
| bid2 | 11 | +1234 | +1294 | +1285 | +1311 |

## Short gates (applied to F, M, C at once; missing depth passes)

| gate | sleeve | n kept | n dropped | E net kept bp | E net dropped bp | stopped % kept | stopped % dropped | squeeze % kept | squeeze % dropped |
|---|---|---|---|---|---|---|---|---|---|
| G1 ask2_qv >= 0.0270 | F | 52 | 76 | +1071 | +696 | +0 | +9 | +19 | +33 |
| G1 ask2_qv >= 0.0270 | M | 111 | 21 | +1530 | +3623 | +18 | +5 | +30 | +33 |
| G1 ask2_qv >= 0.0270 | C | 50 | 99 | +266 | +495 | +0 | +2 | +8 | +13 |
| G2 imb1 <= +0.25 | F | 114 | 14 | +810 | +1162 | +5 | +7 | +27 | +29 |
| G2 imb1 <= +0.25 | M | 127 | 5 | +1787 | +3795 | +17 | +0 | +30 | +40 |
| G2 imb1 <= +0.25 | C | 129 | 20 | +410 | +471 | +1 | +5 | +12 | +10 |
| G3 ch5 >= log(0.5) | F | 118 | 10 | +909 | +130 | +6 | +0 | +28 | +20 |
| G3 ch5 >= log(0.5) | M | 129 | 3 | +1881 | +1125 | +16 | +33 | +30 | +33 |
| G3 ch5 >= log(0.5) | C | 135 | 14 | +287 | +1677 | +1 | +7 | +12 | +7 |
| G4 dep1_qv >= 0.0268 | F | 49 | 79 | +1124 | +677 | +0 | +9 | +18 | +33 |
| G4 dep1_qv >= 0.0268 | M | 119 | 13 | +1670 | +3637 | +17 | +8 | +30 | +31 |
| G4 dep1_qv >= 0.0268 | C | 49 | 100 | +492 | +382 | +0 | +2 | +4 | +15 |

## 10-slot books, $10k a year, 2023-26 (book() from e25)

| book | rule | trades | bp/trade | L trades | L bp | min year % | worst DD % | years 23/24/25/26 % |
|---|---|---|---|---|---|---|---|---|
| LMC | base | 392 | +1133 | 121 | +1218 | +51 | -17 | +81/+156/+616/+51 |
| LMC | R1 tie bid2_qv asc | 391 | +1125 | 120 | +1192 | +51 | -17 | +76/+145/+637/+51 |
| LMC | R2 tie imb1 asc | 390 | +1095 | 119 | +1093 | +49 | -17 | +65/+139/+628/+49 |
| LMC | R3 skip bid2_qv top tercile | 377 | +1152 | 106 | +1297 | +46 | -17 | +79/+162/+575/+46 |
| LMC | R4 skip imb1 top tercile | 379 | +1150 | 108 | +1287 | +55 | -17 | +73/+151/+596/+55 |
| LMC | G1 ask2_qv >= 0.0270 | 281 | +1184 | 122 | +1224 | +26 | -16 | +63/+158/+256/+26 |
| LMC | G2 imb1 <= +0.25 | 371 | +1143 | 122 | +1224 | +49 | -16 | +75/+153/+546/+49 |
| LMC | G3 ch5 >= log(0.5) | 375 | +1113 | 121 | +1218 | +61 | -19 | +78/+168/+423/+61 |
| LMC | G4 dep1_qv >= 0.0268 | 286 | +1289 | 121 | +1218 | +32 | -16 | +63/+154/+373/+32 |
| LMCF | base | 488 | +1075 | 114 | +1234 | +103 | -32 | +104/+145/+812/+103 |
| LMCF | R1 tie bid2_qv asc | 486 | +1069 | 113 | +1193 | +98 | -32 | +98/+135/+827/+103 |
| LMCF | R2 tie imb1 asc | 486 | +1040 | 112 | +1085 | +86 | -32 | +86/+128/+816/+100 |
| LMCF | R3 skip bid2_qv top tercile | 474 | +1089 | 101 | +1305 | +96 | -32 | +102/+146/+783/+96 |
| LMCF | R4 skip imb1 top tercile | 478 | +1088 | 103 | +1343 | +94 | -32 | +94/+147/+787/+108 |
| LMCF | G1 ask2_qv >= 0.0270 | 322 | +1196 | 118 | +1234 | +31 | -14 | +77/+194/+347/+31 |
| LMCF | G2 imb1 <= +0.25 | 459 | +1100 | 116 | +1227 | +83 | -32 | +94/+164/+762/+83 |
| LMCF | G3 ch5 >= log(0.5) | 468 | +1065 | 114 | +1234 | +101 | -32 | +101/+143/+617/+115 |
| LMCF | G4 dep1_qv >= 0.0268 | 325 | +1302 | 118 | +1234 | +37 | -14 | +77/+187/+510/+37 |

# Appendix B: follow-up (written by x12_followup.py to book_depth_followup.md)

## Follow-up (post hoc)

### L within-day IC by period and controls

| feature | IC_day 23-24 (days) | IC_day 25-26 (days) | days IC<0 / IC>0 (>= 10 cands) |
|---|---|---|---|
| dep5_qv | -0.109 (10) | -0.151 (4) | 9 / 3 |
| dep1_qv | -0.075 (10) | -0.072 (4) | 7 / 5 |
| bid2_qv | -0.057 (10) | -0.048 (4) | 7 / 5 |
| ask2_qv | -0.084 (10) | -0.061 (4) | 9 / 3 |
| ch5 | -0.091 (10) | -0.226 (4) | 9 / 3 |
| bid5 | +0.006 (10) | -0.003 (4) | 6 / 6 |
| lqv | +0.054 (10) | +0.025 (4) | 5 / 7 |
| cnn | +0.228 (10) | +0.288 (4) | 2 / 10 |

Within-day correlation of log dep5_qv with log qv: -0.59; within-day IC of dep5_qv with log qv partialled out: -0.107 (14 days).

| feature | within-day IC after removing the CNN rank (days) | top-10 by CNN: mean net bp | top-10 by CNN+depth composite |
|---|---|---|---|
| dep5_qv | -0.070 (14) | +1472 | +1506 |
| ch5 | -0.087 (14) | +1472 | +1531 |
| ask2_qv | -0.042 (14) | +1472 | +1498 |
| bid2_qv | -0.019 (14) | +1472 | +1497 |

### L trades taken in the LMC book: base vs R1 (L pnl in $ of the yearly $10k book)

| year | base n | R1 n | base L pnl $ | R1 L pnl $ |
|---|---|---|---|---|
| 2023 | 20 | 19 | +3904 | +3490 |
| 2024 | 62 | 62 | +10031 | +9127 |
| 2025 | 23 | 23 | +9304 | +9932 |
| 2026 | 16 | 16 | +1068 | +1068 |

F stopped % by qv tercile low/mid/high: 5/7/5.

F stop-outs (ask1 = log10 USDT within 1% above mid):

| symbol | year | net | ask1 | ask2_qv | qv $M | imb1 |
|---|---|---|---|---|---|---|
| UNFIUSDT | 2024 | -1.00 | 4.41 | 0.0017 | 28.5 | +0.42 |
| XVGUSDT | 2024 | -1.00 | 4.87 | 0.0203 | 7.0 | +0.03 |
| ALPACAUSDT | 2025 | -1.00 | 4.35 | 0.0004 | 74.6 | -0.01 |
| BAKEUSDT | 2025 | -1.00 | 4.69 | 0.0237 | 5.8 | -0.01 |
| HIFIUSDT | 2025 | -1.00 | 4.85 | 0.0143 | 8.7 | -0.16 |
| HIFIUSDT | 2025 | -1.00 | 5.16 | 0.0181 | 16.8 | -0.17 |
| BANKUSDT | 2026 | -1.00 | 4.86 | 0.0170 | 10.3 | +0.00 |

### Post-hoc book rows (10 slots, 2023-26)

| book | rule | trades | bp/trade | L trades | L bp | min year % | worst DD % | years 23/24/25/26 % |
|---|---|---|---|---|---|---|---|---|
| LMC | base | 392 | +1133 | 121 | +1218 | +51 | -17 | +81/+156/+616/+51 |
| LMC | P0 L tie CNN rank + 0.5 x low-dep5_qv rank | 391 | +1139 | 120 | +1236 | +51 | -17 | +80/+152/+636/+51 |
| LMC | P1 L tie dep5_qv asc | 391 | +1121 | 120 | +1179 | +51 | -17 | +74/+146/+637/+51 |
| LMC | P2 L tie qv desc (control) | 390 | +1104 | 119 | +1124 | +51 | -17 | +72/+138/+620/+51 |
| LMCF | base | 488 | +1075 | 114 | +1234 | +103 | -32 | +104/+145/+812/+103 |
| LMCF | P0 L tie CNN rank + 0.5 x low-dep5_qv rank | 487 | +1077 | 113 | +1244 | +103 | -32 | +103/+140/+825/+103 |
| LMCF | P1 L tie dep5_qv asc | 486 | +1066 | 113 | +1180 | +96 | -32 | +96/+135/+827/+103 |
| LMCF | P2 L tie qv desc (control) | 485 | +1054 | 112 | +1129 | +93 | -32 | +93/+130/+806/+103 |
| LMCF | P3 F only: ask2_qv >= 0.0270 | 430 | +1146 | 117 | +1228 | +57 | -17 | +97/+192/+760/+57 |
| LMCF | P4 F only: ask1 >= 10^5.13 (2023-24 F 33rd pct) | 450 | +1088 | 117 | +1228 | +58 | -20 | +87/+176/+794/+58 |
| LMCF | P5 F only: qv >= 8.3M (control) | 462 | +1084 | 115 | +1220 | +97 | -25 | +97/+146/+722/+97 |
