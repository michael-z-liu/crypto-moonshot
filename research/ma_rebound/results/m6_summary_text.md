## Outcome

**There was no pick, so nothing is confirmed.** The protocol's only finalist, the base CNN with 5
seeds under the base rule (`m6-base5-base`), had already failed on 2022-24 (it lost in 2022 and was
fragile), and `results/M6_pick.md` recorded "There is no pick" before 2025-26 was opened. This
page reports what the holdout (fills from 2025-01-01 to the data end, 2026-11-02) showed for every
ledger row, as the plan fixed in advance. It is description, not confirmation.

- **The finalist lost money in both holdout years.** `m6-base5-base`: 1,109 trades in 2025 at
  -32 bp per trade and -36.1% book return; 887 trades in 2026 (to 2026-11-02) at -12 bp and -15.1%.
  Pooled 2025-26: 1,996 trades, -23 bp per trade (day-bootstrap 95% CI -65 to +17), book -45.8%,
  max drawdown -59.3%, stress -49 bp. All five of its seeds also lost in both years (pooled -53% to
  -82%).
- **It lost less than buying every candidate.** *Every candidate* under the same rule: 6,718
  trades, -45 bp (CI -66 to -22), -86.7% in 2025 and -73.1% in 2026 (pooled -96.4%). The model's
  trades earned 22 bp more per trade than the no-model book and 18 bp more than BTC over the same
  windows. Selection still works in the direction it did on 2022-24, but not by enough to make the
  long-only book positive.
- **The two comparison rows named for their 2022-24 rank did not hold up.** `m6-breadth5-base`
  (first among base-rule rows, +97.7% on 2022-24): -92 bp (CI -147 to -36), -63.7% and -34.3%.
  `m6-otherma5-pct0.9` (first of all 25 rows, +122.1%): -15 bp (CI -124 to +132), -20.0% and -14.2%.
- **Few rows were positive.** Of the 54 non-seed scored rows, 5 had a positive book in 2025, 6 in
  2026 and 2 in both. Those 2 are 1d-MA-only rules on M6 models (`m6-otherma5-tf1d`, 113 trades,
  +38.5% / +16.2%; `m6-short5-tf1d`, 120 trades, +10.6% / +10.6%). Both fail the protocol's
  100-trades-a-year floor, and the first one's gain rests on a few trades: 5 of its 113 trades
  carry 67% of its 2025-26 P&L, the largest a +133% trade in ASRUSDT. None of the 16 *every
  candidate* books was positive in either year (-42% to -95% in 2025).
- **Every scored row beat its no-model comparator on book return** (54 of 54 pooled), but only
  33 of 54 beat it on bp per trade. The comparators lose so heavily that a smaller book loses less.
- **The 2022-24 ranking carried some information across the whole screen, none within M6.**
  Spearman correlation of the 2022-24 rank value with the 2025-26 summary return: +0.43 over all
  54 rows, +0.50 over the 27 M5 rows, -0.13 over the 25 M6 rows (with pooled bp per trade: +0.27,
  +0.12, +0.06).
- **The deflated Sharpe ratio does not separate any row from luck.** Over 54 trials the expected
  maximum annual Sharpe ratio without edge is +1.23. The finalist's 2022-24 Sharpe is +0.23 (DSR
  0.04); its 2025-26 Sharpe is -0.45. The highest 2022-24 Sharpe, +1.70 (`m6-short5-tf1d`, a 1d-only
  rule of about 60 trades a year), gives a DSR of 0.80, below the usual 0.95.
- **Without a model the premise failed again on 2025-26.** Buying every limit fill at L(1 + 0.05u)
  with exit E1 averaged -70 bp per trade in 2025 (CI -111 to -30) and -34 bp in 2026 (CI -56 to
  -12), and every timeframe was negative. The daily MAs, the one positive timeframe of M2's
  2020-24 table (+366 bp), averaged -167 bp (CI -345 to +24, 3,125 trades). Study MAs again did not
  beat placebo MAs: -4 bp per trade (CI -10 to +3).

## Caveats

- **No pick, nothing confirmed.** Every 2025-26 number here is descriptive. The rows that did well
  in 2025-26 were not chosen before the reveal and cannot be picked now: choosing them after seeing
  2025-26 would make 2025-26 a selection period.
- **The two named comparison rows were selected on 2022-24**, for their 2022-24 rank, so their
  2022-24 numbers were biased upward. Their decline is partly what that selection predicts.
- **Seed-mean rows ranked far above their own seeds on 2022-24** (M5 and M6a). On 2025-26 the seed
  tables also select differently: in 2025 and 2026 base5's single seeds pass 8.5% to 23% of the
  candidates at percentile >= 0.80, its seed mean 15% and 6%, so the seed rows trade more (2,496 to
  3,119 trades against 1,996).
- **2026 is a partial year**: it ends at the data end, 2026-11-02 (about 10 months). Positions open
  at the data end close at the coin's last 5m close (3 of the finalist's trades).
- **Comparator books started 2025 near zero.** The no-model books lost most of their capital in
  2022-24 (`every-L0.05-E2` began 2025 with $49 of $10,000). Their 2025-26 returns are still
  percentages of the equity on 2025-01-01, but with that equity a slot's stake is a few dollars.
- **Trading halts** (plan, M4 review): a series is treated as continuous across a halt of a day or
  more. No halt occurs in 2025-26, but delistings do; a delisting closes the position at the last
  close (for example HARDUSDT, -93% in April 2025).
- **Fold 2026's row counts were once printed** during M4's development (training and validation
  row counts that count 2025 fills); no outcome of 2025-26 was seen before this reveal.
- **The M2 context tables leave out labels still open at the data end**, as M2 leaves out
  unfinished labels; this drops some of the latest 2026 trades, more for long exits.
