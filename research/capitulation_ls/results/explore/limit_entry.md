# X11: limit entries below the signal price for L (candidates n=4224, 2022-26)

| limit | window h | filled | fill % | mean net bp | 22-24 bp | 25-26 bp | TP % | E_o | LMC trades | LMC bp | LMC min yr % | LMC years 22..26 % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0% | 0.0 | 4224 | 100% | +1311 | +833 | +1839 | 71% | +0.131 | 437 | +1142 | +51 | +65/+82/+143/+650/+51 |
| 2% | 1 | 2486 | 59% | +1457 | +950 | +1870 | 79% | +0.146 | 415 | +1208 | +37 | +63/+69/+127/+869/+37 |
| 2% | 4 | 2551 | 60% | +1451 | +957 | +1868 | 79% | +0.145 | 427 | +1205 | +37 | +59/+69/+153/+897/+37 |
| 4% | 1 | 1864 | 44% | +1541 | +856 | +1895 | 85% | +0.154 | 405 | +1181 | +37 | +51/+63/+108/+858/+37 |
| 4% | 4 | 1930 | 46% | +1537 | +900 | +1893 | 85% | +0.154 | 422 | +1164 | +35 | +45/+65/+134/+891/+35 |
| 6% | 4 | 1609 | 38% | +1598 | +822 | +1912 | 88% | +0.160 | 403 | +1241 | +39 | +58/+64/+127/+907/+39 |
| 8% | 4 | 1379 | 33% | +1606 | +527 | +1924 | 90% | +0.161 | 384 | +1257 | +39 | +52/+57/+126/+846/+39 |
| 10% | 8 | 1247 | 30% | +1590 | +405 | +1931 | 91% | +0.159 | 395 | +1204 | +39 | +47/+59/+127/+809/+39 |
| 4% | 8 | 2061 | 49% | +1445 | +775 | +1887 | 82% | +0.145 | 436 | +1158 | +34 | +49/+65/+156/+908/+34 |
| 6% | 8 | 1697 | 40% | +1534 | +753 | +1905 | 86% | +0.153 | 423 | +1242 | +38 | +66/+67/+167/+923/+38 |

Row 1 (0%) is the market entry recomputed here (+1142 bp, years close to the reference LMC
+1158 bp; small differences come from exit bookkeeping). Every row was tried and is listed.

## Verdict: limit entries fail

Each filled trade earns more (+1311 -> +1457..+1606 bp, take-profit rate 71% -> 79-91%), but the
book gets worse where it matters: the worst year falls from +51% to +34..+39% (2026), 2023 falls
from +82% to +57..+69%, and 2022-24 per-trade results fall for limits of 6% or deeper (+833 ->
+405..+822 bp). Only 2025 improves.

## Why it failed (adverse selection; see the split below, market-entry outcomes)

| limit, window | period | filled? | n | net at market entry, bp | TP rate | days |
|---|---|---|---|---|---|---|
| 4%, 4h | 2022-24 | no | 1524 | +870 | 51% | 15 |
| 4%, 4h | 2022-24 | yes | 693 | +753 | 58% | 14 |
| 4%, 4h | 2025-26 | no | 770 | +1912 | 97% | 8 |
| 4%, 4h | 2025-26 | yes | 1237 | +1793 | 89% | 6 |
| 8%, 4h | 2022-24 | no | 1903 | +932 | 53% | 17 |
| 8%, 4h | 2022-24 | yes | 314 | +238 | 55% | 12 |
| 8%, 4h | 2025-26 | no | 942 | +1911 | 97% | 8 |
| 8%, 4h | 2025-26 | yes | 1065 | +1775 | 88% | 4 |

1. **The limit misses the best trades.** Candidates that never trade 4-8% lower were better at
   the market price in both periods (an instant V-bounce). The ones that keep falling are worse,
   strongly so in 2022-24 (+238 bp at market for 8% dips, against +932 bp for the rest).
2. **The better price is real but only pays on the big cascade days.** In 2025-26 nearly
   everything bounces (TP 88-97%), so buying lower simply adds the discount; 2025's book rises.
   In the slower 2022-24 and 2026 flushes, the fills are the coins that kept falling.
3. **Fills come later, and slots fill with them.** On flush days the first 10 fills take the
   slots, which are no longer the strongest signals but the deepest fallers.
