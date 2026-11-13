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

Qualifies: at least 100 trades, mean bp > 0 and book return > 0 in each year, stress bp > 0,
and mean bp above the placebo's 95th percentile. Fragile: some year's mean bp turns negative
without that year's best trade. Rank value: the mean of the three yearly book returns.

## Seeds

Each model under the base rule, seed by seed. *Mean row inside seed range*: whether the rank
value of the 5-seed-mean score lies between the lowest and the highest per-seed rank value.

| model | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | seed mean of ranks | spread (max - min) | 5-seed-mean row | mean row inside seed range | seeds qualifying | seeds fragile |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base5 | +29.4% | +9.5% | +15.0% | -10.6% | +32.7% | +15.2% | 43.2 pp | +20.9% | yes | 0 of 5 | 5 of 5 |

Every row, seed by seed:

| row | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | spread | seeds qualifying | seeds fragile |
|---|---|---|---|---|---|---|---|---|
| m6-base5-base | +29.4% | +9.5% | +15.0% | -10.6% | +32.7% | 43.2 pp | 0 of 5 | 5 of 5 |
| m6-base5-pct0.9 | +5.5% | +12.6% | +21.8% | +15.9% | +70.1% | 64.6 pp | 0 of 5 | 5 of 5 |
| m6-base5-tf1h4h | +28.8% | +20.8% | +24.7% | -9.7% | +6.7% | 38.5 pp | 0 of 5 | 5 of 5 |
| m6-base5-slots20 | +35.8% | +12.9% | +19.2% | +11.6% | +44.3% | 32.7 pp | 0 of 5 | 5 of 5 |
| m6-base5-tf1d | +19.1% | +11.7% | +3.9% | +22.9% | +14.8% | 19.0 pp | 0 of 5 | 5 of 5 |

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

Rows not yet scored: m6-breadth5-base, m6-breadth5-pct0.9, m6-breadth5-tf1h4h, m6-breadth5-slots20, m6-breadth5-tf1d, m6-otherma5-base, m6-otherma5-pct0.9, m6-otherma5-tf1h4h, m6-otherma5-slots20, m6-otherma5-tf1d, m6-short5-base, m6-short5-pct0.9, m6-short5-tf1h4h, m6-short5-slots20, m6-short5-tf1d, m6-combo5-base, m6-combo5-pct0.9, m6-combo5-tf1h4h, m6-combo5-slots20, m6-combo5-tf1d.

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
