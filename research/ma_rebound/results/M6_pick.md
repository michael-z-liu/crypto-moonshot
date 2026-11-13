# M6 pick record (written before the holdout is opened)

Everything here is 2022-24, the selection period; the book is continuous from 2022-01-01.
2025-01-01 onward is the sealed holdout and no number of it appears here. The design was
fixed in the plan's Decisions entry "M6 design" before M6 ran.

- **Models** (5 seeds each, 0-4, folds 2022-2026, otherwise identical to their registered
  3-seed variant): `base5` (the base CNN), `breadth5`, `otherma5`, `short5`, and `combo5`
  (fine 48, coarse 64, groups coin + btc + breadth + otherma).
- **Rules** on each model's seed-mean scores (row `m6-<model>-<rule>`) and seed by seed
  (`m6-<model>-<rule>-seed<i>` in the ledger): `base` (limit at L(1 + 0.05u), exit E1,
  percentile >= 0.80, 10 slots), `pct0.9`, `tf1h4h` (1h + 4h MAs only), `slots20`, `tf1d`
  (1d MAs only); every non-base rule changes one thing from the base rule.
- **Eligibility:** only `m6-base5-base` can be the pick, and only if it qualifies and is not
  fragile on 2022-24. Every other row is a comparison that can be neither picked nor confirmed.

All cards are in `results/m6_cards.txt`; every row, seed rows included, is in
`results/ledger.csv` (stage M6), with the training run id in its `source`.

## All 25 rows on 2022-24

M6 rows in bold. Under each, where one exists, the M4/M5 row it corresponds to (3 seeds):
the same model under the base rule, or for `base5` the same rule on the 3-seed base CNN.
`combo5` and the non-base rules on the other models have no earlier counterpart.

| row | model / rule | trades 2022 / 23 / 24 | bp per trade 2022 / 23 / 24 | book return 2022 / 23 / 24 | stress bp | placebo p95 bp | fragile | qualifies | rank value |
|---|---|---|---|---|---|---|---|---|---|
| **m6-base5-base** | base CNN / base rule (pct >= 0.80, 10 slots) | 1395 / 1178 / 1091 | -61 / +38 / +66 | -64.6% / +48.5% / +78.8% | -9.2 | -7.2 | yes | no | +20.9% |
| ref: cnn-base (M4, 3 seeds) | for reference, not an M6 row | 1398 / 1119 / 1103 | -52 / +27 / +72 | -60.7% / +25.4% / +92.2% | -10.9 | -4.7 | yes | no | +19.0% |
| **m6-base5-pct0.9** | base CNN / percentile >= 0.9 | 722 / 573 / 505 | -89 / +113 / +178 | -57.0% / +84.4% / +114.2% | +45.3 | -2.8 | yes | no | +47.2% |
| ref: rule-pct0.9 (M5, 3-seed base CNN) | for reference, not an M6 row | 803 / 585 / 592 | -71 / +138 / +122 | -53.7% / +113.8% / +86.1% | +39.6 | -6.2 | yes | no | +48.7% |
| **m6-base5-tf1h4h** | base CNN / MAs: 1h + 4h only | 1601 / 1229 / 1195 | -25 / +55 / +22 | -38.2% / +89.2% / +22.0% | -6.6 | -2.5 | yes | no | +24.3% |
| ref: rule-tf1h4h (M5, 3-seed base CNN) | for reference, not an M6 row | 1661 / 1195 / 1240 | -16 / +55 / +52 | -30.4% / +83.8% / +78.7% | +5.5 | -3.7 | yes | no | +44.0% |
| **m6-base5-slots20** | base CNN / 20 slots | 2212 / 1904 / 1856 | -63 / +63 / +67 | -59.0% / +73.6% / +62.5% | -0.1 | -5.1 | yes | no | +25.7% |
| ref: rule-slots20 (M5, 3-seed base CNN) | for reference, not an M6 row | 2273 / 1820 / 1879 | -50 / +53 / +99 | -54.3% / +52.2% / +118.6% | +9.0 | -3.5 | yes | no | +38.9% |
| **m6-base5-tf1d** | base CNN / MAs: 1d only | 26 / 93 / 57 | +27 / +145 / +700 | +0.6% / +12.9% / +37.7% | +276.8 | +59.7 | yes | no | +17.1% |
| ref: rule-tf1d (M5, 3-seed base CNN) | for reference, not an M6 row | 29 / 87 / 67 | +155 / +104 / +686 | +4.3% / +6.6% / +48.3% | +267.9 | -78.8 | no | no | +19.7% |
| **m6-breadth5-base** | + market breadth / base rule (pct >= 0.80, 10 slots) | 742 / 1240 / 1174 | -80 / +120 / +55 | -48.6% / +284.7% / +56.9% | +9.6 | -5.0 | yes | no | +97.7% |
| ref: cnn-breadth (M5, 3 seeds) | for reference, not an M6 row | 772 / 1255 / 1149 | -109 / +115 / +58 | -63.3% / +254.7% / +50.1% | -4.6 | -12.7 | yes | no | +80.5% |
| **m6-breadth5-pct0.9** | + market breadth / percentile >= 0.9 | 396 / 723 / 666 | -37 / +129 / +104 | -16.3% / +143.5% / +83.7% | +43.5 | +10.1 | yes | no | +70.3% |
| **m6-breadth5-tf1h4h** | + market breadth / MAs: 1h + 4h only | 776 / 1320 / 1295 | -34 / +66 / +43 | -26.4% / +130.7% / +62.1% | -1.1 | -7.0 | yes | no | +55.5% |
| **m6-breadth5-slots20** | + market breadth / 20 slots | 1209 / 2102 / 2154 | -59 / +115 / +78 | -33.3% / +206.5% / +96.2% | +21.9 | -2.6 | yes | no | +89.8% |
| **m6-breadth5-tf1d** | + market breadth / MAs: 1d only | 12 / 95 / 100 | -838 / +650 / +393 | -9.8% / +76.0% / +33.1% | +38.8 | -94.9 | yes | no | +33.1% |
| **m6-otherma5-base** | + distances to the other MAs / base rule (pct >= 0.80, 10 slots) | 1125 / 1257 / 1292 | -92 / +68 / +64 | -70.5% / +127.2% / +87.6% | -9.8 | -7.1 | yes | no | +48.1% |
| ref: cnn-otherma (M5, 3 seeds) | for reference, not an M6 row | 1127 / 1420 / 1207 | -85 / +67 / +78 | -67.6% / +150.9% / +121.7% | -3.6 | -9.4 | yes | no | +68.3% |
| **m6-otherma5-pct0.9** | + distances to the other MAs / percentile >= 0.9 | 613 / 665 / 761 | -116 / +113 / +216 | -58.8% / +105.9% / +319.1% | +45.0 | -2.4 | yes | no | +122.1% |
| **m6-otherma5-tf1h4h** | + distances to the other MAs / MAs: 1h + 4h only | 1287 / 1300 / 1549 | -36 / +72 / +53 | -42.0% / +145.2% / +111.8% | +3.8 | -2.7 | yes | no | +71.7% |
| **m6-otherma5-slots20** | + distances to the other MAs / 20 slots | 1860 / 2034 / 2309 | -94 / +98 / +93 | -66.0% / +158.7% / +146.7% | +7.8 | -5.8 | yes | no | +79.8% |
| **m6-otherma5-tf1d** | + distances to the other MAs / MAs: 1d only | 5 / 78 / 86 | +1038 / +598 / +722 | +5.1% / +53.7% / +68.3% | +781.4 | +388.5 | no | no | +42.4% |
| **m6-short5-base** | windows 48 x 5m and 64 bars / base rule (pct >= 0.80, 10 slots) | 1374 / 1355 / 1070 | -74 / +85 / +49 | -71.9% / +207.2% / +48.1% | -2.8 | -4.4 | yes | no | +61.1% |
| ref: cnn-short (M5, 3 seeds) | for reference, not an M6 row | 1440 / 1381 / 1023 | -74 / +75 / +89 | -73.7% / +178.2% / +92.9% | +5.1 | -5.9 | yes | no | +65.8% |
| **m6-short5-pct0.9** | windows 48 x 5m and 64 bars / percentile >= 0.9 | 848 / 683 / 509 | -74 / +141 / +182 | -54.0% / +163.2% / +117.2% | +59.5 | +10.8 | yes | no | +75.5% |
| **m6-short5-tf1h4h** | windows 48 x 5m and 64 bars / MAs: 1h + 4h only | 1645 / 1432 / 1255 | -40 / +70 / +22 | -53.5% / +159.1% / +25.7% | -7.9 | -3.5 | yes | no | +43.8% |
| **m6-short5-slots20** | windows 48 x 5m and 64 bars / 20 slots | 2293 / 2150 / 1832 | -53 / +104 / +79 | -52.7% / +197.4% / +77.8% | +18.8 | +0.9 | yes | no | +74.2% |
| **m6-short5-tf1d** | windows 48 x 5m and 64 bars / MAs: 1d only | 56 / 58 / 60 | -149 / +630 / +1836 | -8.9% / +43.0% / +167.1% | +743.4 | +71.5 | yes | no | +67.1% |
| **m6-combo5-base** | short windows + breadth + other-MA distances / base rule (pct >= 0.80, 10 slots) | 850 / 1405 / 1325 | -113 / +83 / +57 | -69.7% / +208.0% / +69.4% | -13.8 | -8.8 | yes | no | +69.2% |
| **m6-combo5-pct0.9** | short windows + breadth + other-MA distances / percentile >= 0.9 | 449 / 818 / 752 | -142 / +156 / +133 | -52.5% / +255.5% / +131.9% | +25.2 | -11.8 | yes | no | +111.7% |
| **m6-combo5-tf1h4h** | short windows + breadth + other-MA distances / MAs: 1h + 4h only | 985 / 1529 / 1833 | -44 / +51 / +37 | -39.4% / +107.4% / +80.2% | -9.1 | -3.6 | yes | no | +49.4% |
| **m6-combo5-slots20** | short windows + breadth + other-MA distances / 20 slots | 1420 / 2391 / 2545 | -83 / +99 / +67 | -50.5% / +205.1% / +95.4% | +4.0 | -3.0 | yes | no | +83.3% |
| **m6-combo5-tf1d** | short windows + breadth + other-MA distances / MAs: 1d only | 28 / 75 / 106 | -539 / +814 / +139 | -14.9% / +84.3% / +5.5% | +133.4 | +85.9 | yes | no | +25.0% |

Qualifies: at least 100 trades, mean bp > 0 and book return > 0 in each year, stress bp > 0,
and mean bp above the placebo's 95th percentile. Fragile: some year's mean bp turns negative
without that year's best trade. Rank value: the mean of the three yearly book returns.

## Seeds

Each model under the base rule, seed by seed. *Mean row inside seed range*: whether the rank
value of the 5-seed-mean score lies between the lowest and the highest per-seed rank value.

| model | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | seed mean of ranks | spread (max - min) | 5-seed-mean row | mean row inside seed range | seeds qualifying | seeds fragile |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base5 | +29.4% | +9.5% | +15.0% | -10.6% | +32.7% | +15.2% | 43.2 pp | +20.9% | yes | 0 of 5 | 5 of 5 |
| breadth5 | +26.2% | +7.1% | +34.3% | +23.7% | +27.6% | +23.8% | 27.2 pp | +97.7% | no | 0 of 5 | 5 of 5 |
| otherma5 | +25.7% | +1.5% | +48.0% | +13.5% | +15.7% | +20.9% | 46.5 pp | +48.1% | no | 0 of 5 | 5 of 5 |
| short5 | +39.8% | +22.0% | +47.0% | +18.0% | +64.6% | +38.3% | 46.6 pp | +61.1% | yes | 0 of 5 | 5 of 5 |
| combo5 | +4.8% | +46.6% | +16.8% | +32.3% | -3.8% | +19.4% | 50.4 pp | +69.2% | no | 0 of 5 | 5 of 5 |

Every row, seed by seed:

| row | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | spread | seeds qualifying | seeds fragile |
|---|---|---|---|---|---|---|---|---|
| m6-base5-base | +29.4% | +9.5% | +15.0% | -10.6% | +32.7% | 43.2 pp | 0 of 5 | 5 of 5 |
| m6-base5-pct0.9 | +5.5% | +12.6% | +21.8% | +15.9% | +70.1% | 64.6 pp | 0 of 5 | 5 of 5 |
| m6-base5-tf1h4h | +28.8% | +20.8% | +24.7% | -9.7% | +6.7% | 38.5 pp | 0 of 5 | 5 of 5 |
| m6-base5-slots20 | +35.8% | +12.9% | +19.2% | +11.6% | +44.3% | 32.7 pp | 0 of 5 | 5 of 5 |
| m6-base5-tf1d | +19.1% | +11.7% | +3.9% | +22.9% | +14.8% | 19.0 pp | 0 of 5 | 5 of 5 |
| m6-breadth5-base | +26.2% | +7.1% | +34.3% | +23.7% | +27.6% | 27.2 pp | 0 of 5 | 5 of 5 |
| m6-breadth5-pct0.9 | +13.3% | +61.2% | +77.6% | +15.9% | +81.5% | 68.2 pp | 0 of 5 | 5 of 5 |
| m6-breadth5-tf1h4h | +10.1% | +18.5% | +35.4% | +26.8% | +14.0% | 25.3 pp | 0 of 5 | 5 of 5 |
| m6-breadth5-slots20 | +29.9% | +21.7% | +68.0% | +34.2% | +59.6% | 46.3 pp | 0 of 5 | 5 of 5 |
| m6-breadth5-tf1d | +23.0% | +8.6% | +9.3% | +2.2% | +17.1% | 20.8 pp | 0 of 5 | 5 of 5 |
| m6-otherma5-base | +25.7% | +1.5% | +48.0% | +13.5% | +15.7% | 46.5 pp | 0 of 5 | 5 of 5 |
| m6-otherma5-pct0.9 | +76.3% | +2.4% | +56.8% | +20.5% | +40.1% | 73.9 pp | 0 of 5 | 5 of 5 |
| m6-otherma5-tf1h4h | +51.8% | +8.1% | +18.8% | +10.2% | +17.8% | 43.7 pp | 0 of 5 | 5 of 5 |
| m6-otherma5-slots20 | +49.2% | +10.0% | +43.1% | +5.7% | +37.3% | 43.5 pp | 0 of 5 | 5 of 5 |
| m6-otherma5-tf1d | +42.5% | +19.0% | +29.6% | +9.9% | +59.2% | 49.3 pp | 0 of 5 | 5 of 5 |
| m6-short5-base | +39.8% | +22.0% | +47.0% | +18.0% | +64.6% | 46.6 pp | 0 of 5 | 5 of 5 |
| m6-short5-pct0.9 | +45.4% | +61.4% | +70.3% | +51.3% | +66.5% | 24.9 pp | 0 of 5 | 5 of 5 |
| m6-short5-tf1h4h | +20.9% | -7.2% | +15.9% | +18.3% | +32.8% | 40.0 pp | 0 of 5 | 5 of 5 |
| m6-short5-slots20 | +33.2% | +31.3% | +59.7% | +38.1% | +65.6% | 34.4 pp | 0 of 5 | 5 of 5 |
| m6-short5-tf1d | +19.6% | +47.1% | +57.0% | +32.6% | +40.4% | 37.5 pp | 0 of 5 | 5 of 5 |
| m6-combo5-base | +4.8% | +46.6% | +16.8% | +32.3% | -3.8% | 50.4 pp | 0 of 5 | 5 of 5 |
| m6-combo5-pct0.9 | +92.5% | +26.5% | +91.7% | +57.6% | +34.7% | 66.1 pp | 0 of 5 | 5 of 5 |
| m6-combo5-tf1h4h | -7.1% | +17.5% | -6.4% | +23.9% | +18.4% | 31.0 pp | 0 of 5 | 5 of 5 |
| m6-combo5-slots20 | +16.3% | +61.6% | +54.2% | +51.1% | +33.4% | 45.3 pp | 0 of 5 | 5 of 5 |
| m6-combo5-tf1d | +15.9% | +27.7% | +37.7% | +29.4% | +11.5% | 26.1 pp | 0 of 5 | 5 of 5 |

## Pick

**There is no pick.** The only eligible row, `m6-base5-base`, can be the pick only if it
qualifies and is not fragile on 2022-24; it does not qualify and it is fragile. By the plan,
nothing can be called confirmed at the reveal. The other 24 rows are comparisons and
cannot be picked whatever their numbers.

- Verdict criteria (2022-24): trades 1395/1178/1091 ok; bp -61/+38/+66 FAIL; return -64.6%/+48.5%/+78.8% FAIL; stress -9.2 bp FAIL; mean +14.6 vs placebo p95 -7.2 bp ok.
- Qualifies: no.
- Fragile: yes (mean bp per year without that year's best trade: -64 / +35 / +58).

Comparison rows that qualify: none; that qualify and are not
fragile: none. They are recorded for the reveal's rank correlation
and deflated Sharpe ratio only.

For the reveal's "top-ranked finalist" (used only when there is no pick), the 2022-24
rank values give: the protocol's only finalist is `base5`, and `m6-base5-base` is its
base-rule row; the top-ranked of the five models' base-rule rows is `m6-breadth5-base` (+97.7%); the top-ranked of all 25 rows is `m6-otherma5-pct0.9` (+122.1%). Which of these the reveal reports is the master's call, fixed before
the holdout is opened.

Rows not yet scored: none.

## What will be opened

From the plan's Decisions entry "M6 design" (*Reveal content*), opened once for every ledger row:

- per row: 2025, 2026 and 2025-26 trades, bp per trade, book return and drawdown;
- the pick's (or, with no pick, the top-ranked finalist's) day-bootstrap CI and its comparison
  with *every candidate*;
- the Spearman correlation between the 2022-24 rank value and the 2025-26 book return over
  non-seed rows;
- a deflated Sharpe ratio over all non-seed ledger rows;
- and, as context, M2's per-timeframe and study-vs-placebo tables on 2025-26.

**Confirmed** (plan, "M6 pick and reveal") applies to the pick only: on 2025-26 the
day-bootstrap 95% CI of the mean net bp per trade is above 0, the book return is positive, and
it beats *every candidate* under the same rule. With no pick, nothing can be called confirmed.
