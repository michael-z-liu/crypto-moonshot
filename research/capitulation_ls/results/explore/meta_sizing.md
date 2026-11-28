# X10: book-level sizing rules (10 slots, $10k a year, in-sample)

| book | rule | trades | bp/trade | min year % | worst DD % | years 22..26 % | mean w |
|---|---|---|---|---|---|---|---|
| LMC | base | 430 | +1158 | +51 | -17 | +62/+81/+156/+625/+51 | 1.00 |
| LMC | momentum | 419 | +1138 | +51 | -21 | +58/+84/+142/+806/+51 | 1.17 |
| LMC | inverse | 429 | +1157 | +32 | -11 | +60/+58/+100/+206/+32 | 0.65 |
| LMC | loss_brake | 430 | +1158 | +50 | -16 | +62/+81/+154/+511/+50 | 0.99 |
| LMC | book_brake | 430 | +1158 | +45 | -16 | +64/+81/+149/+441/+45 | 0.98 |
| LMCF | base | 539 | +1112 | +101 | -32 | +101/+104/+145/+812/+103 | 1.00 |
| LMCF | momentum | 523 | +1091 | +103 | -39 | +103/+111/+115/+950/+114 | 1.17 |
| LMCF | inverse | 538 | +1111 | +57 | -17 | +84/+70/+108/+308/+57 | 0.65 |
| LMCF | loss_brake | 539 | +1112 | +101 | -24 | +101/+104/+138/+761/+101 | 0.99 |
| LMCF | book_brake | 539 | +1112 | +77 | -26 | +100/+104/+111/+582/+77 | 0.88 |

## Loss-brake neighbourhood (same-sleeve trade closed <= thr in the last N days)

| book | thr | days | weight | bp/trade | min year % | worst DD % | years 22..26 % | braked trades |
|---|---|---|---|---|---|---|---|---|
| LMC | -20% | 3 | 0.0 | +1175 | +56 | -16 | +62/+81/+156/+600/+56 | 8 |
| LMC | -20% | 3 | 0.5 | +1158 | +53 | -16 | +62/+81/+156/+613/+53 | 8 |
| LMC | -20% | 7 | 0.0 | +1091 | +49 | -16 | +62/+81/+149/+376/+49 | 33 |
| LMC | -20% | 7 | 0.5 | +1158 | +50 | -16 | +62/+81/+154/+494/+50 | 33 |
| LMC | -20% | 14 | 0.0 | +1151 | +44 | -14 | +63/+80/+129/+394/+44 | 69 |
| LMC | -20% | 14 | 0.5 | +1158 | +48 | -11 | +63/+81/+149/+505/+48 | 69 |
| LMC | -30% | 3 | 0.0 | +1179 | +56 | -16 | +62/+81/+156/+616/+56 | 7 |
| LMC | -30% | 3 | 0.5 | +1158 | +53 | -16 | +62/+81/+156/+620/+53 | 7 |
| LMC | -30% | 7 | 0.0 | +1103 | +49 | -16 | +62/+81/+149/+403/+49 | 31 |
| LMC | -30% | 7 | 0.5 | +1158 | +50 | -16 | +62/+81/+154/+511/+50 | 31 |
| LMC | -30% | 14 | 0.0 | +1151 | +44 | -14 | +63/+80/+129/+421/+44 | 63 |
| LMC | -30% | 14 | 0.5 | +1158 | +48 | -11 | +63/+81/+149/+520/+48 | 63 |
| LMC | -50% | 3 | 0.0 | +1164 | +51 | -16 | +62/+81/+156/+618/+51 | 3 |
| LMC | -50% | 3 | 0.5 | +1158 | +51 | -16 | +62/+81/+156/+621/+51 | 3 |
| LMC | -50% | 7 | 0.0 | +1126 | +44 | -16 | +62/+81/+170/+489/+44 | 14 |
| LMC | -50% | 7 | 0.5 | +1158 | +47 | -16 | +62/+81/+163/+552/+47 | 14 |
| LMC | -50% | 14 | 0.0 | +1142 | +41 | -14 | +64/+81/+184/+444/+41 | 29 |
| LMC | -50% | 14 | 0.5 | +1158 | +46 | -14 | +63/+81/+170/+530/+46 | 29 |
| LMCF | -20% | 3 | 0.0 | +1156 | +101 | -20 | +101/+104/+139/+920/+109 | 16 |
| LMCF | -20% | 3 | 0.5 | +1112 | +101 | -24 | +101/+104/+142/+863/+106 | 16 |
| LMCF | -20% | 7 | 0.0 | +1082 | +97 | -17 | +101/+104/+123/+606/+97 | 49 |
| LMCF | -20% | 7 | 0.5 | +1112 | +100 | -24 | +101/+104/+138/+693/+100 | 49 |
| LMCF | -20% | 14 | 0.0 | +1091 | +90 | -16 | +103/+103/+104/+493/+90 | 92 |
| LMCF | -20% | 14 | 0.5 | +1112 | +97 | -22 | +102/+103/+132/+664/+97 | 92 |
| LMCF | -30% | 3 | 0.0 | +1168 | +101 | -20 | +101/+104/+139/+1000/+109 | 13 |
| LMCF | -30% | 3 | 0.5 | +1112 | +101 | -24 | +101/+104/+142/+906/+106 | 13 |
| LMCF | -30% | 7 | 0.0 | +1108 | +100 | -17 | +101/+104/+123/+722/+100 | 42 |
| LMCF | -30% | 7 | 0.5 | +1112 | +101 | -24 | +101/+104/+138/+761/+101 | 42 |
| LMCF | -30% | 14 | 0.0 | +1110 | +93 | -16 | +103/+103/+104/+588/+93 | 81 |
| LMCF | -30% | 14 | 0.5 | +1112 | +98 | -22 | +102/+103/+132/+728/+98 | 81 |
| LMCF | -50% | 3 | 0.0 | +1159 | +101 | -20 | +101/+104/+142/+1003/+103 | 8 |
| LMCF | -50% | 3 | 0.5 | +1112 | +101 | -24 | +101/+104/+143/+907/+103 | 8 |
| LMCF | -50% | 7 | 0.0 | +1127 | +93 | -19 | +101/+104/+149/+824/+93 | 23 |
| LMCF | -50% | 7 | 0.5 | +1112 | +98 | -24 | +101/+104/+147/+811/+98 | 23 |
| LMCF | -50% | 14 | 0.0 | +1132 | +89 | -19 | +104/+104/+155/+687/+89 | 46 |
| LMCF | -50% | 14 | 0.5 | +1112 | +96 | -22 | +103/+104/+150/+736/+96 | 46 |

(Weight 0.0 rows drop the braked candidates; `book()` would otherwise read a zero weight as 1.)

## Verdict: no book-level sizing rule is an edge (fails; the brake is a one-cluster risk dial)

- **Momentum** only adds exposure (mean weight 1.17): 2025 rises, 2024 falls (+156% -> +142%
  LMC, +145% -> +115% LMCF), drawdown is worse (-17% -> -21%, -32% -> -39%). The inverse rule
  is the same lever pointed down. Sleeve results do not persist enough to size on.
- **Loss brake** (after a same-sleeve trade closes at <= -30%, skip or halve that sleeve's
  entries for a few days) cuts LMCF's drawdown in every one of 18 neighbours (-32% to -16..-24%)
  and keeps the worst year at +89..+101%. LMC changes little (-17% to -11..-16%, worst year
  +41..+56%).

## Why it is not adopted

1. **One cluster carries it.** LMCF's -32% drawdown is in 2025. The 3-day skip removes 4 F
   trades in 2025-26 averaging -46% (half stopped): one squeeze cluster. In 2022-24 the trades
   it skips average +14.5% (F) and the brake costs money. Every neighbour reduces the drawdown
   because every neighbour catches the same cluster.
2. **Longer brakes skip the best listing shorts.** At 7 days the skipped M trades in 2025-26
   average +47% (sleeve mean +23%): a stopped listing short says nothing about the next listing.
3. **Squeezes cluster in time, but rarely.** This agrees with the earlier finding that slot caps
   do not remove the squeeze drawdown. A short brake on F after a stop-out is a reasonable risk
   setting for live trading, but the evidence is about one episode, so it is not counted as an
   improvement.
