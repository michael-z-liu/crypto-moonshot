# Issue #10: testing the improvement ideas for the capitulation CNN

Unattended study, 2026-11-10, of the ideas in issue #10 for the walk-forward `path_all` CNN
(`docs/research/cnn-robustness.md`). Ideas 1-6 and 8 were tested; idea 7 (futures data) is a
follow-up. Plan and run log: `docs/plans/cnn-improvements.md`. Code: `research/volume_spike/i10_*.py`.
Stage notes: `research/volume_spike/results/issue10/` ([M1](../../research/volume_spike/results/issue10/M1.md),
[M2](../../research/volume_spike/results/issue10/M2.md),
[M3](../../research/volume_spike/results/issue10/M3.md),
[M4](../../research/volume_spike/results/issue10/M4.md),
[M5 pick](../../research/volume_spike/results/issue10/M5_pick.md),
[M5](../../research/volume_spike/results/issue10/M5.md)).

## Summary

1. **The pre-registered pick did not hold up.** `M5:volday5:b_h1_2h` (volatility scaling + equal
   weight per day, 5 seeds, 1h-median ranking, 2h hold) made +358 bp per trade on 2022-24. It made
   **+64 bp over 2025-26** (205 trades, +21%, max drawdown -27%). Its day-bootstrap 95% CI is
   [-77, +225] and the deflated Sharpe ratio over all 157 trials is 0.67. It was not confirmed. The
   runner-up (volatility scaling, 2h hold) made +125 bp (CI [+10, +249], DSR 0.83).
2. **Equal weight per UTC day (idea 2) is the most robust change out of sample.**
   - `day` (3 seeds, 1h hold) made +90 bp over 2025-26 against base3's +46, with +76% and a -13%
     drawdown.
   - Every rule on `day5` is positive over 2025-26 in all five seeds, except `esc_2h` (four).
   - Its 2025-10-10 losses stayed at -2,000 to -4,000 bp summed, about base3's, against -9,000 to
     -11,000 for the 1h volatility-scaled rules.
3. **The 2h hold is again the most reliable rule change.** In 2025 it beat the 1h hold for both
   heads, averaged over the M5 score sets: h1 head +80 vs +41 bp, h2 head +62 vs +28. On base3 it
   gives +150 bp over 2025-26 against +46.
4. **Volatility scaling (idea 6) was the largest 2022-24 gain but did not carry.**
   - At the 1h hold it went from +224 bp on 2022-24 to +19 over 2025-26.
   - With a 2h hold it made +125.
5. **What failed, already on 2022-24 or out of sample:**
   - a target relative to the universe (idea 5), which failed both periods (+41 / -7);
   - early stopping or loss weighting on the tail (idea 1);
   - fixed epochs;
   - recency weighting, a rolling 3-year window and normalisation per fold (idea 8);
   - the risk-adjusted head (idea 4);
   - cascade caps and daily loss limits (idea 3).
6. **No change reduced seed noise.** Seed spreads at q 0.98 were 27 to 197 bp, as wide as or wider
   than base3's. The average-of-seeds score often disagreed with the per-seed results.
7. **Selection had coarse skill but no fine skill.**
   - Over all 42 #10 rows, the Spearman correlation between the 2022-24 mean and 2025 bp is
     +0.62: what failed in-sample mostly failed out of sample.
   - Among the 18 finalists it is +0.04: ranking good candidates on 2022-24 picked noise.
8. **Bottom line.** The ranking edge survives on the survivorship-free universe. Day weighting with
   a 2h hold is a cheap, better default than base3. No #10 idea made the strategy decisively more
   profitable out of sample. All 2025-26 statements are on a holdout already seen in PR #8 and the
   robustness study, so only forward data can confirm any of them.

## Method

- **Universe and scores:** universe "all" (582 USDT pairs incl. delisted). Walk-forward CNN folds:
  2022-26, trained before Y-1 July, early-stopped on Y-1 H2. Scores are fold percentiles.
- **Book:** the production engine as one continuous book from 2022-01-01 to 2026-11-02, $10,000.
  - pct >= 0.98, highest first, `CashCapped(5)`, enter at the next 5m open.
  - 1h hold unless stated.
  - Fee 0.1% + slippage 0.05% + latency 0.05% per side (0.4% round trip), S5's realistic setup.
  - Stresses: 0.6% round trip, one candle late. Placebo: 20 within-day shuffles.
- **Harness:** `i10_train.py` reproduces `path_all.npz` bit for bit (seeds 0-1, 1.46M rows).
- **Baseline `base3`:** path_all with 3 seeds. 2022-24 +120.9 bp; fragile (2022 without its best
  trade turns negative). Per seed +110 / +103 / +129, so the **noise band** is 26.8 bp.
- **Pre-registered (before any #10 result):**
  - **Qualify** as in the robustness study.
  - **Beat the baseline:** qualify, not fragile, and seed-mean 2022-24 mean > +147.7.
  - **Combination rule for M5:**
    - the top two training variants, retrained with 5 seeds, plus their combination;
    - rules (a) h1 1h, (b) h1 2h, (c) the best M2 rule, (d) its other hold;
    - pick the top qualified, non-fragile run.
  - **Reveal:** 2025-26 opened once for every row, after the pick was committed (`06050ac`, pushed).
- **Comparison rows:** the escalating threshold (`esc`) was added to M5 as a comparison that could
  not be picked. The decision was made after M2 showed it was the only M2 rule up in every seed, and
  before any M3 result or 2025-26 number.
- **Trials:** 42 ledger rows (M1 1, M2 14, M3 4, M4 5, M5 18), plus 162 per-seed rows that are not
  counted as trials. GPU time: about 6.7 h.

## Results

2022-24 is the seed-mean mean bp per trade. 2025-26 bp is pooled over both years' trades (2026 runs
to 11-02). "10-10" is the summed net bp on 2025-10-10 (trade count in brackets).

### Training variants (h1 head, 1h hold)

| Variant | Idea | 2022-24 | Beats base3 | 2025 | 2026 | **2025-26** (n) | Return 25-26 | Max DD | 10-10 |
|---|---|---|---|---|---|---|---|---|---|
| base3 | - | +121 | - | +9 | +191 | **+46** (314) | +19% | -31% | -3268 (27) |
| es_top | 1 | +153 | no (fragile) | +9 | +93 | **+28** | | | |
| tail | 1 | +137 | no | +65 | +156 | **+85** | | | |
| day | 2 | +179 | **yes** | +87 | +95 | **+90** (369) | +76% | -13% | -3191 (21) |
| fixed | 2 | +139 | no (fragile) | +6 | +82 | **+25** | | | |
| xs_univ | 5 | +41 | no (fails late) | -7 | -8 | **-7** (1878) | -35% | | |
| volscale | 6 | +224 | **yes** | +9 | +51 | **+19** (328) | -3% | -25% | -10677 (26) |
| recency | 8 | +163 | no (fragile) | +5 | +177 | **+48** | | | |
| roll3y | 8 | +122 | no | -9 | +82 | **+12** | | | |
| foldnorm | 8 | +171 | **yes** | +6 | +210 | **+44** | | | |

### Rules on base3 scores (idea 3: cascade awareness; idea 4: hold-horizon heads)

| Rule | 2022-24 | Beats base rule | **2025-26** | 10-10 |
|---|---|---|---|---|
| base (h1, 1h) | +121 | - | **+46** | -3268 (27) |
| h1, 2h hold | +211 | yes | **+150** | +14160 (17) |
| h2 (2h median), 2h hold | +248 | yes | **+113** | +14160 (17) |
| h2, 1h hold | +174 | yes | **+28** | -3211 (27) |
| h2 risk-adjusted, 1h / 2h | +57 / +123 | no | **+2 / +34** | |
| day cap 5 | +173 | yes | **+103** | +2291 (5) |
| day cap 10 | +95 | no | **+28** | -5924 (10) |
| cascade cap 5 / 10 | +71 / +138 | no | **+41 / +46** | -12058 (12) / -8664 (17) |
| daily loss limit 3% / 6% | +99 / +93 | no | **-1 / +15** | -12058 (12) |
| cascade age <= 60 / 180 min | +129 / +145 | no | **+57 / +44** | -8664 (17) / -3268 (27) |
| escalating threshold (esc) | +166 | yes | **+60** | -3268 (27) |

### M5 (5 seeds; per-seed 2025-26 average in brackets)

| Score set | (a) h1 1h | (b) h1 2h | (c) h2 2h | (d) h2 1h | esc (n/e) | esc 2h (n/e) |
|---|---|---|---|---|---|---|
| 2022-24: volscale5 | +224 | +334 | +270 | +157 (fragile) | +295 | +387 |
| 2022-24: day5 | +180 | +247 | +243 | +168 | +215 | +243 |
| 2022-24: volday5 | +255 | **+358 (pick)** | +311 | +251 | +326 | +386 |
| 2025-26: volscale5 | +60 | +125 (+40) | +120 (+2) | +68 (-17) | +85 | +133 |
| 2025-26: day5 | +98 | +88 | +96 | +85 | **+131** | +96 |
| 2025-26: volday5 | +49 | **+64 (+37)** | +11 (+40) | +8 (+32) | +95 | +63 |

- The pick's 2025 was +42 bp on 144 trades and its 2026 +115 on 61. It lost -10,060 bp summed on
  2025-10-10 (11 trades), but all five seeds were positive over 2025-26.
- `day5:esc` (not eligible) had the best out-of-sample book: +131 bp, +104%, max drawdown -9%.

### Deflated Sharpe ratio

N = 157 trials, the robustness study's 115 plus the 42 #10 rows, all rebuilt from 2022-24 daily
mark-to-market returns. The expected maximum Sharpe with no edge is 1.27.

| | 2022-24 Sharpe (rank of 157) | Skew / kurtosis | DSR | DSR, normal returns | 2025-26 Sharpe |
|---|---|---|---|---|---|
| Pick | 1.47 (68th) | 10.1 / 243 | 0.674 | 0.635 | 0.55 |
| Runner-up | 1.68 (52nd) | 9.6 / 208 | 0.830 | 0.762 | 0.47 |

For comparison, the robustness study's pick had DSR 0.967 at N = 115.

## Caveats

- **2025-26 is a spent holdout.** It was seen in PR #8 and the robustness study, and ideas 3 and 4
  were motivated by 2025. Every out-of-sample statement here says what didn't fail, not what will
  work.
- **Seed noise is large.**
  - Seed means differ by up to 197 bp on 2022-24.
  - A seed-mean book's 2025-26 can differ from its seeds' average by 100 bp (volscale5:c +120
    against +2).
  - Five seeds are not enough to rank variants that differ by less than about 50 bp.
- **Look-ahead in the cascade rules is excluded by construction.** Cascades and cascade age use only
  earlier candidates. The engine's group cap counts only entries actually taken. The daily loss
  limit is realised-only, so it cannot stop entries made together in one candle, which is why it
  did nothing on 2025-10-10.
- **The universe offset is not suspect.** The offset for `xs_univ` passed hand, own-coin and
  look-ahead checks, so its failure is not a data bug. It did early-stop on correlation with the raw
  1h return, which is a possible handicap.
- **Execution is still untested.** The S2 finding (one candle late costs about 60 bp) holds for these
  rules too.

## Recommendations

- **Don't adopt the pick.** Its out-of-sample CI includes zero and its DSR is 0.67.
- **Use day weighting with a 2h hold as the new default.** It is the one training change that held
  in every seed out of sample, kept the cascade losses at base3's level, and costs nothing. Add `day5` with a 2h hold
  (and `day5:esc` as a second arm) to the P1 forward paper test next to `path_all` 2h. These choices
  have seen 2025-26, so the paper test is the test.
- **Drop the dead ends:**
  - the universe-relative target;
  - top-quantile early stopping;
  - tail weighting;
  - the risk-adjusted head;
  - cascade caps and realised-only loss limits;
  - rolling windows.
- **Next:**
  - idea 7 (funding, open interest and basis from Binance futures), the most likely source of new
    information;
  - a day cap or escalating threshold tested forward;
  - measuring real fills at the spike-candle close.
