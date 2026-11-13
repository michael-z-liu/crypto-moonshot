# CNN robustness study

Unattended study, 2026-11-07, of what makes the walk-forward capitulation CNN of PR #8 more
robust first and more profitable second. Plan and run log: `docs/plans/cnn-robustness.md`.
Code: `research/volume_spike/` (`scorecard.py`, `s0_baseline.py` .. `s5_confirm.py`). Stage
notes and tables: `research/volume_spike/results/robustness/` ([S0](../../research/volume_spike/results/robustness/S0.md),
[S1](../../research/volume_spike/results/robustness/S1.md),
[S2](../../research/volume_spike/results/robustness/S2.md),
[S3](../../research/volume_spike/results/robustness/S3.md),
[S4](../../research/volume_spike/results/robustness/S4.md),
[S5](../../research/volume_spike/results/robustness/S5.md),
[S5 pick](../../research/volume_spike/results/robustness/S5_pick.md)).

## Summary

1. **On 2022-24, several changes looked like large, robust gains.** Against path_all's +152 bp
   per trade, they were:
   - a BTC flush gate (BTC volume >= 6x: +291; BTC 4h <= -2%: +222);
   - a 2h hold (+221);
   - 10 slots at a 99th-percentile threshold (+291);
   - a breadth gate (+222);
   - cross-coin day context as CNN inputs (S4 ctx: +188, with half the drawdown).

   Combined with each other, all 24 S5 runs qualified, none was fragile, and the means reached
   +462 bp.
2. **Most of it did not carry into 2025.** The pre-registered pick, `S5:path_all5:e_s10q99` (5-seed
   path_all, 10 slots, pct >= 0.99, 1h hold, 0.4% round trip incl. latency), had +462 bp on
   2022-24.
   - It lost **-58 bp per trade in 2025** (-12%) and made +559 in 2026 (36 trades). That is
     **+80 bp per trade over 2025-26** (161 trades, +7% in 22 months, max drawdown -16%).
   - The day-bootstrap 95% CI is [-136, +384] and the out-of-sample Sharpe is 0.33.
   - It traded on only 22 days. One day, 2025-10-10, cost it -12,832 bp summed over 31 trades.
   - The runner-up (ctx + BTC volume gate + 2h hold) made +32 / +276, so +79 bp over 2025-26.
   - The plain S0 path_all baseline made +94 bp over the same months.
3. **The 2022-24 ranking did not predict 2025.** Over the 24 runs, the Spearman correlation
   between the 2022-24 mean and the 2025 bp is -0.20. Only 19 of 24 runs were positive in 2025,
   14 at 0.6% cost and 10 one candle late. Every run is positive over 2025-26 because 2026 (to
   2026-11-02) is strong for all of them.
   - The BTC gates, the main 2022-24 finding, lost most of their edge in 2025, as F2 did.
   - The 2h-hold rules held up best (+121 and +142 bp on average in 2025).
4. **Deflated Sharpe ratio** over all 115 trials: the pick's 2022-24 Sharpe (1.83) beats the
   expected best of 115 no-edge trials (1.25) with probability 0.97, or 0.84 with normal returns.
   The 2022-24 edge was probably not pure selection. But it is much smaller now, and the
   out-of-sample CI includes zero.
5. **Things that did not help, even on 2022-24:**
   - stops, trailing stops and take profits;
   - score-weighted sizing;
   - pausing on the strategy's own recent results;
   - liquidity and tick filters;
   - a GBM stacker;
   - a net-of-cost asymmetric target;
   - a rank-within-day loss;
   - 6-monthly retraining;
   - 7 days of hourly bars or the last hour of 1m candles as inputs.

   Five seeds and ensembles moved results only by seed noise.
6. **Bottom line.** The CNN still ranks coins within crash days on the survivorship-free
   universe: every S5 run made money over 2025-26, and 23 of 24 still did at 0.6%. But the edge
   is about +50 to +150 bp per trade, near zero in 2025 for many rules, and carried by a few
   crash days. None of this study's robustness gains was confirmed out of sample.

## Method

- **Universe "all":** 582 USDT pairs, delisted coins and coins outside the top-200 pool included
  (PR #8's survivorship-free universe). The pool is not used.
- **Scores:** walk-forward CNN percentiles. Fold Y is trained on data before Y-1 July,
  early-stopped on Y-1 H2 and scored on Y (2022 .. 2026); a row's score is its percentile among
  the fold's training scores. 2 seeds unless stated.
- **Default rule:** buy spike candles with pct >= 0.98 at the next 5m open, highest score first,
  5 equal slots, 1h hold, fee 0.1% + slippage 0.05% per side (0.3% round trip).
- **Scorecard (`scorecard.py`):** the production engine (`crypto_moonshot.backtest`) on 5m
  candles, as one continuous book from 2022-01-01 to the data end 2026-11-02, $10,000. Per entry
  year: trades, bp per trade, win rate, $ and % return, max drawdown. Stresses: a 0.6% round
  trip (slippage 0.2% per side; a rule's latency is kept) and entering one candle late. Placebo:
  20 books with the scores shuffled within each UTC day. Day-bootstrap 95% CI of the 2022-24 mean
  bp; the mean without the 5 best days; "fragile" if some 2022-24 year turns negative without
  that year's single best trade.
- **Selection rule (fixed before any result):** a variant qualifies if on 2022-24 its bp per
  trade is > 0 every year, its mean is > 0 at 0.6% cost and one candle late, and the mean beats
  the placebo's 95th percentile. Qualified variants rank by the mean of the three yearly bp, the
  worst year breaking ties.
- **Ledger and sealing:** every variant tried is a row in `results/robustness/ledger.csv`
  (115 rows: S0 4, S1 46, S2 28, S3 9, S4 4, S5 24). Every card also computes 2025-26 but writes it
  only to the gitignored `sealed/` folder; it was opened once, in S5, after the pick was committed
  (commit `730eb0a`). Base, market and path_all on "all" had already been seen for 2025-26 in
  PR #8.

## S0: baselines

Three CNNs retrained on "all" (2 seeds); their scores equal PR #8's. ens3 is the mean of their
percentiles. The engine reproduces the research simulator's bp to within 1 bp per year when fed
the same entries; the remaining gaps are the engine's cash check (it skips an entry it cannot pay
for; one skipped LUNA trade moves base's 2022 by -74 bp) and the continuous book.

| Variant | 2022 | 2023 | 2024 | Mean bp | 0.6% | Late | w/o 5 days | Fragile |
|---|---|---|---|---|---|---|---|---|
| path_all | +162 | +219 | +75 | **+152** | +121 | +89 | +85 | no |
| ens3 | +55 | +294 | +99 | **+149** | +119 | +88 | +99 | yes (LUNA) |
| market | +84 | +183 | +24 | **+97** | +67 | +60 | +45 | no |
| base | +18 | +174 | +72 | **+88** | +58 | +43 | +42 | yes (LUNA) |

All four qualify. The placebo loses about 80 bp per trade, so criterion 4 never binds; the
binding criteria are every year, 0.6% and late. The weak spot is 2024 one candle late (+15 to
+28 bp).

## S1: exits and sizing (46 trials, no training)

| path_all rule | Mean bp | Worst year | Late | w/o 5 days | Trades/yr | Note |
|---|---|---|---|---|---|---|
| baseline 1h | +152 | +75 | +89 | +85 | 240 | |
| hold 2h | +221 | +92 | +126 | +108 | 195 | the clearest exit gain |
| hold 4h | +235 | +54 | +157 | +74 | 161 | weak 2024 |
| TP +5% / 2h, no stop | +146 | +70 | +84 | +129 | 255 | lower mean, less tail-dependent |
| trailing 3% / 5% | +54 / +78 | | | | | cuts the right tail |
| TP +5% + stop -10% | +60 | -23 | | | | fails |
| CashCapped | +149 | +72 | +86 | +85 | 245 | |
| 10 slots, q 0.99 | +291 | +213 | +126 | +211 | 137 | best region |
| q 0.97 (any slots) | +24 to +53 | | | | | poor |

Holding longer helps, stops and trailing stops hurt, score-weighted sizing adds nothing, and more
slots at a higher threshold is the best region. ens3 behaves the same (10 slots x q 0.99: +378).

## S2: regime and execution (28 trials)

| path_all rule | Mean bp | Worst year | Late | w/o 5 days | Trades/yr |
|---|---|---|---|---|---|
| baseline | +152 | +75 | +89 | +85 | 240 |
| BTC volume >= 6x | +291 | +225 | +184 | +164 | 139 |
| BTC 4h <= -2% | +222 | +142 | +131 | +165 | 153 |
| BTC 4h <= -4% | +300 | +159 | +182 | +81 | 80 |
| spike share >= p75 | +222 | +109 | +142 | +146 | 188 |
| own last-20 trades >= 0 | +40 | -134 | -35 | -4 | 161 |
| latency 0.05% per side | +142 | +65 | +79 | +75 | 240 |
| two candles late | +95 | +7 | +72 | +39 | 240 |

BTC flush gates help most and in every year; breadth gates moderately; liquidity and tick
filters do nothing reliable; the own-performance pause hurts. Latency of 0.05-0.1% per side
costs 10-20 bp. On 1m candles, entering inside the spike candle (t+4m, t+3m) is 90-190 bp worse
than the engine's entry at its close, in every year.

## S3: training and ensembles (9 trials)

| Variant | Mean bp | Worst year | Late | w/o 5 days | Fragile |
|---|---|---|---|---|---|
| path_all (S0) | +152 | +75 | +89 | +85 | no |
| path_all5 (5 seeds) | +154 | +57 | +88 | +78 | no |
| path_all_h6 (6-monthly) | +149 | +53 | +68 | +86 | yes |
| ens_s3 (8 CNNs) | +151 | +89 | +61 | +118 | no |
| ens_top | +151 | +59 | +73 | +91 | no |
| allch_xs (rank-in-day loss) | +116 | +65 | +56 | +66 | no |
| allch_y1h | +113 | +48 | +77 | +45 | no |
| allch_cost (asymmetric) | +82 | +2 | +40 | +30 | yes |
| GBM stacker / scalars (2023-24) | +43 / +56 | | | | fail (no 2022) |

Nothing beats path_all beyond seed noise (about 10-20 bp a year). Ensembles buy evenness, not
edge; the stacker and the cost-aware target are worse.

## S4: inputs (4 trials)

| Variant | 2022 | 2023 | 2024 | Mean bp | 0.6% | Late | w/o 5 days | CI | Fragile |
|---|---|---|---|---|---|---|---|---|---|
| path_all (S0) | +162 | +219 | +75 | +152 | +121 | +89 | +85 | [+69, +236] | no |
| **ctx** (day context + coin scalars) | +268 | +218 | +79 | **+188** | +159 | +111 | +112 | [+97, +280] | no |
| long7d (7 days of 1h bars) | +159 | +229 | +32 | +140 | +110 | +66 | +81 | [+52, +221] | no |
| m1last (last hour of 1m) | +90 | +194 | +72 | +119 | +88 | +68 | +50 | [+6, +221] | yes |
| inputs_all | +153 | +185 | +48 | +129 | +98 | +74 | +58 | [+12, +227] | no |

Cross-coin day context (BTC 4h return and volume, how many coins are crashing, the coin's rank
among them, spike share, tick size, age, liquidity) is the only input that helps, and nearly all
of its gain is 2022. Longer or finer price history fits the average better but picks a worse top
2%.

## S5: combine and confirm (24 trials)

The combination rule was written into the plan before any S3/S4 result and before any 2025-26
number. It used three score sets: path_all (the reference) and the two best qualified,
non-fragile S3/S4 score sets, S4 ctx (+188) and S3 path_all5 (+154). Each got eight rules:

- (a) the default 1h;
- (b) BTC volume >= 6x;
- (c) BTC 4h <= -2%;
- (d) a 2h hold;
- (e) 10 slots at q 0.99;
- (f) b with a 2h hold;
- (g) b with a +5% take profit and a 2h time exit, no stop;
- (h) spike share >= p75 with a 2h hold.

Every run used `CashCapped` equal slots and latency 0.05% per side (fee 0.1% + slippage 0.05%),
plus the usual stresses (latency kept in both).

**The pick** was committed (`S5_pick.md`, commit 730eb0a) before 2025-26 was opened. All 24 runs
qualified and none was fragile, so the rule reduced to the highest 2022-24 mean.

- **Pick:** `S5:path_all5:e_s10q99`, +579 / +572 / +236 bp, mean +462, worst year +236, 0.6%
  +431, late +268, without the 5 best days +272, CI [+251, +692], 127 trades a year.
- **Runner-up:** `S5:ctx:f_btcvol6_hold2h`, mean +394.08. It was 0.17 bp ahead of
  `S5:ctx:e_s10q99` (+393.91), a tie in practice that the rule settles by the mean.

The full 2022-24 table is in [S5_pick.md](../../research/volume_spike/results/robustness/S5_pick.md).

### The reveal (2025-26, opened once, every row)

The table shows bp per trade net of costs; the 2025-26 bp is pooled over both years' trades.
Returns and drawdowns are of the continuous book; 2026 runs to 2026-11-02. The S0 baselines use
their S0 rule (`EqualSlots(5)`, no latency). The 2025-26 numbers for base, market and path_all
were already known from PR #8.

| Run | Mean bp 22-24 | 2025 bp (n) | 2026 bp (n) | **2025-26 bp** (n) | 0.6% 25-26 | Late 25-26 | $ P&L 25-26 | Return 2025 / 2026 / 25-26 | Max DD 25-26 |
|---|---|---|---|---|---|---|---|---|---|
| base | +88 | +9 (374) | +21 (165) | **+12** (539) | -18 | +27 | -2014 | -10% / +4% / -6% | -36% |
| market | +97 | -22 (374) | +122 (62) | **-2** (436) | -31 | -26 | -2549 | -22% / +15% / -10% | -25% |
| path_all | +152 | +63 (304) | +210 (80) | **+94** (384) | +64 | +90 | +33270 | +31% / +37% / +79% | -27% |
| ens3 | +149 | -20 (150) | +367 (42) | **+65** (192) | +35 | +10 | +4808 | -13% / +34% / +17% | -26% |
| path_all:a_1h | +138 | +53 (305) | +200 (80) | **+84** (385) | +53 | +80 | +19787 | +23% / +35% / +66% | -28% |
| path_all:b_btcvol6 | +278 | +7 (160) | +315 (49) | **+79** (209) | +49 | +43 | +22160 | -4% / +34% / +29% | -20% |
| path_all:c_btc4h-2 | +210 | +40 (173) | +334 (49) | **+105** (222) | +75 | +78 | +23527 | +8% / +37% / +47% | -15% |
| path_all:d_hold2h | +210 | +63 (244) | +190 (65) | **+90** (309) | +60 | +44 | +3392 | -12% / +24% / +8% | -34% |
| path_all:e_s10q99 | +384 | -55 (179) | +674 (46) | **+94** (225) | +64 | +14 | +4625 | -14% / +35% / +15% | -25% |
| path_all:f_btcvol6_hold2h | +367 | -36 (135) | +512 (45) | **+101** (180) | +70 | +69 | +26864 | -15% / +54% / +31% | -28% |
| path_all:g_btcvol6_tp5_2h | +197 | -44 (155) | +181 (48) | **+9** (203) | -21 | +4 | -1499 | -18% / +19% / -3% | -32% |
| path_all:h_share75_hold2h | +290 | +69 (196) | +258 (60) | **+113** (256) | +83 | +49 | +12036 | -13% / +32% / +14% | -31% |
| ctx:a_1h | +176 | +10 (299) | +215 (55) | **+42** (354) | +12 | +51 | +7996 | -4% / +26% / +21% | -22% |
| ctx:b_btcvol6 | +324 | +16 (180) | +215 (39) | **+51** (219) | +21 | +31 | +11471 | -2% / +17% / +15% | -23% |
| ctx:c_btc4h-2 | +200 | +32 (181) | +195 (38) | **+61** (219) | +30 | +61 | +8667 | +4% / +15% / +19% | -22% |
| ctx:d_hold2h | +225 | +135 (253) | +174 (51) | **+142** (304) | +111 | +82 | +14181 | +20% / +18% / +42% | -15% |
| ctx:e_s10q99 | +394 | +88 (129) | +911 (25) | **+221** (154) | +191 | +154 | +8784 | +7% / +24% / +33% | -13% |
| ctx:f_btcvol6_hold2h (runner-up) | +394 | +32 (162) | +276 (39) | **+79** (201) | +49 | +73 | +17561 | +3% / +23% / +26% | -25% |
| ctx:g_btcvol6_tp5_2h | +187 | +19 (173) | +135 (43) | **+42** (216) | +12 | +26 | +4661 | -1% / +12% / +11% | -26% |
| ctx:h_share75_hold2h | +268 | +184 (217) | +205 (48) | **+188** (265) | +157 | +106 | +32130 | +36% / +20% / +64% | -17% |
| path_all5:a_1h | +145 | +57 (231) | +144 (67) | **+77** (298) | +47 | +41 | +12190 | +19% / +20% / +43% | -16% |
| path_all5:b_btcvol6 | +283 | +32 (126) | +138 (43) | **+59** (169) | +29 | +2 | +9721 | +1% / +12% / +13% | -18% |
| path_all5:c_btc4h-2 | +185 | +33 (141) | +183 (48) | **+71** (189) | +41 | +12 | +8027 | +2% / +18% / +21% | -18% |
| path_all5:d_hold2h | +212 | +164 (188) | +94 (56) | **+148** (244) | +118 | +39 | +7923 | +14% / +10% / +25% | -20% |
| **path_all5:e_s10q99 (pick)** | +462 | -58 (125) | +559 (36) | **+80** (161) | +50 | +49 | +2402 | -12% / +21% / +7% | -16% |
| path_all5:f_btcvol6_hold2h | +363 | +8 (107) | +185 (42) | **+58** (149) | +28 | -7 | +7373 | -5% / +16% / +10% | -24% |
| path_all5:g_btcvol6_tp5_2h | +191 | -6 (124) | +186 (45) | **+45** (169) | +15 | +35 | +3812 | -8% / +18% / +8% | -28% |
| path_all5:h_share75_hold2h | +271 | +174 (162) | +111 (55) | **+158** (217) | +128 | +33 | +12291 | +9% / +12% / +21% | -21% |

- **The pick:** 2025 -58 bp (125 trades on 15 days), 2026 +559 (36 trades on 7 days).
  - 2025-10-10 alone was 31 trades summing to -12,832 bp; without it 2025 is +59.
  - 2026-02-06 and 2026-06-04 carry 2026.
- **The runner-up:** 2025 +32 (+109 without 2025-10-10), 2026 +276.
- **By rule, averaged over the three score sets, 2025 bp:**

  | Rule | Average 2025 bp |
  |---|---|
  | h share75 + hold 2h | +142 |
  | d hold 2h | +121 |
  | a 1h | +40 |
  | c BTC 4h | +35 |
  | b BTC volume | +18 |
  | f BTC volume + hold 2h | +1 |
  | e 10 slots q 0.99 | -9 |
  | g BTC volume + take profit | -10 |

  The rules with the highest 2022-24 means (e, f, b) did worst in 2025.
- **By score set, 2025-26 average:** ctx +103, path_all5 +87, path_all +84.

### 1m fills

The pick and the runner-up were re-run with exits filled on 1m candles (`use_fine`, take profit
`touch`; neither rule has a take profit, so there is no `close` variant). Both give exactly the
5m numbers in every year (pick: +579 / +572 / +236 / -58 / +559; runner-up: +625 / +404 / +153 /
+32 / +276). This follows from the rules: a time exit fills at the same close on either
resolution. The 1m fill model would only matter for take-profit or stop exits.

### Deflated Sharpe ratio

- **Trials:** N = 115, every ledger row S0-S5. Every trial's 2022-24 book was rebuilt without
  placebos, and all 115 match their ledger rows.
- **Returns:** daily mark-to-market returns on 1,096 UTC midnights, annualised with sqrt(365).
- **Trials' annualised Sharpe:** mean 1.49, max 2.67, variance 0.235. The expected maximum of
  115 no-edge trials is 1.25.
- **Pick:** Sharpe 1.83 (32nd of 115 by Sharpe), skew 11.3, kurtosis 167.
- **DSR = 0.967.** With normal returns it is 0.841; the pick's positive skew from a few crash days
  flatters the first figure.
- **Out of sample 2025-26:** Sharpe 0.33 (PSR against 0: 0.66). The bp per trade day-bootstrap
  95% CI is [-136, +384].
- **Runner-up:** DSR 0.848 (0.710 with normal returns), 2025-26 Sharpe 0.63, CI [-85, +232].

## Caveats

- **ctx's gain is mostly 2022:** +268 vs +162 bp, while 2023-24 equal path_all. In 2025 ctx at
  the default rule made +10 bp (path_all at the same S5 rule: +53).
- **The gates overlap ctx's inputs.** BTC 4h return, BTC volume multiple and spike share are
  ctx inputs, so ctx combined with a BTC or breadth gate is not two independent gains.
- **The BTC gates select the market-wide flushes that F2 used, and those decayed in 2025.**
  Gate b was +7 / +16 / +32 bp in 2025 for the three score sets, against +278 to +324 on
  2022-24.
- **Seed noise is about 10-20 bp a year at q 0.98, and larger at q 0.99.** path_all and
  path_all5 at rule e differ by 23 to 146 bp a year on 2022-24 (+513 vs +579, +426 vs +572,
  +213 vs +236).
- **Earlier knowledge of 2025-26.** The 2025-26 numbers of base, market and path_all on "all"
  were known from PR #8 when this study started. path_all as the reference, and the two
  path_all derivatives in S5, were chosen with that knowledge.
- **Few days decide everything.** 2025-26 is 22 months. The pick traded on 15 days in 2025 and
  the runner-up on 41, so one crash day (2025-10-10) can decide a year. 2026 is strong for every
  rule, which lifts every 2025-26 total.
- **The trials are not independent.** N = 115 counts variants of one strategy, which makes the
  DSR's no-edge benchmark conservative. The skew and kurtosis come from a handful of days.
- **The 1m-fill check was empty for these two rules** (time exits only), and S2 showed that
  entering inside the spike candle is worse. Live fills remain untested.

## Recommendations

- **Do not adopt the pick.** Its 2025 was negative, it trades on very few days, and one crash day
  with 10 slots refilling cost it a year.
- **If anything is paper-traded, keep it simple.** Use path_all or ctx at pct >= 0.98, 5
  cash-capped slots, a 2h hold, no gate. The 2h hold is the one change that was better than 1h
  in 2022-24 and also in 2025 for all three score sets (+63 / +135 / +164 bp vs +53 / +10 / +57).
  This choice has now seen 2025-26, so the paper trade is the test, not a confirmation. Use the
  spot testnet, compare with the 1h default, and log fills against the 5m close the backtest
  assumes.
- **What to try next:**
  - a loss limit per crash day (the 2025-10-10 loss came from refilling slots through one
    cascade; the S1 day caps were mixed on 2022-24);
  - a pre-registered test on the data after 2026-11-02 rather than more 2022-24 tuning;
  - measuring real entry slippage on the spike-candle close, the largest unknown now that
    latency costs 10-20 bp;
  - order book and funding data (out of scope here), if the ranking edge is to be rebuilt
    rather than filtered.
