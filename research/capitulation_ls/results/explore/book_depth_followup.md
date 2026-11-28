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
