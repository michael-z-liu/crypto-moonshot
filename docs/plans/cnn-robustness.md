# CNN robustness study: deeper and wider on the capitulation CNN

- **Status:** building <!-- building | blocked | review | done -->
- **Branch:** `claude/cnn-robustness` (from `claude/cnn-replication`, with `feat/backtest-engine` merged in)
- **PR:** none (research branch; report in the session, open a PR only if asked)
- **Updated:** 2026-11-07T14:45Z

## Goal

PR #8's walk-forward CNN ranks coins within a market-wide crash day better than any rule, but the
edge is shrinking and is weakest on the survivorship-free universe (+20 bp per trade in 2025;
negative at 0.6% cost or one candle late). The user asked for a deeper and wider investigation
(5-10 hours, unattended) of anything that makes the CNN strategy more robust first and more
profitable second: inputs, training, ensembles, exits, sizing, regime gating and execution.

## Out of scope

New data sources (order books, funding, open interest, aggTrades at scale); placing orders;
changing `main` or PR #9; tuning on 2025-26.

## Ground rules

- Universe "all" (582 coins incl. delisted and outside the pool) decides; the pool is not used.
- Shared data and caches are READ ONLY. `research/volume_spike/cache_all/` in this worktree is a
  real directory whose input files are symlinks into
  `/Users/mike/code/crypto-moonshot/data/cache/volume_spike_all/` (events, fwd, fwd2, fwd72,
  long30, market, seq2, seq2_meta, m1). NEVER re-run a builder that would rewrite one of those
  (np.save / write_parquet through a symlink writes into the shared folder). New files (wf scores,
  new datasets, logs) are created locally in `cache_all/` (gitignored). Run scripts with
  `MOONSHOT_UNIVERSE=all`.
- GPU (MPS) runs one training at a time; CPU work runs alongside.
- Trading rule unless a stage changes it: walk-forward percentile scores (fold Y trained before
  Y-1 July, early stop on Y-1 H2, scored on Y), top 2% (pct >= 0.98), 5 equal slots, priority
  highest score, hold 1h, 0.1% + 0.05% per side, evaluated as one continuous book with the
  production engine (`crypto_moonshot.backtest`) on 5m candles.

## Selection rule (fixed before any result was seen)

A variant **qualifies** if, on 2022-24 (each year a walk-forward fold):
1. net bp per trade > 0 in every year;
2. still > 0 on average over 2022-24 at a 0.6% round trip (slippage 0.2% per side);
3. still > 0 on average over 2022-24 entering one candle late;
4. beats the placebo (scores shuffled within each day, 20 draws) in mean bp per trade 2022-24, by
   more than the placebo's 95th percentile.

Qualified variants are ranked by mean bp per trade 2022-24 (average of the three years), worst
year as the tiebreak. 2025-26 numbers are computed but not shown or used until S5 opens them once
for the finalists. Every variant tried is a row in the ledger
(`research/volume_spike/results/robustness/ledger.csv`), so S5 can correct for the number of
trials (deflated Sharpe ratio). Base, market and path_all on "all" were already seen for 2025-26
in PR #8; that is noted, not hidden.

## Milestones

### S0: Baseline and scorecard (done)

- **Delivers:** walk-forward scores on "all" for base, market and path_all (2 seeds, existing
  `m12_cnn_wf.py` configs) in local `cache_all/wf/`; a scorecard module (engine-based) giving per
  year: trades, bp per trade, win rate, $ and % return, max DD; stress: 0.6% cost, one candle late;
  placebo (20 within-day shuffles); bootstrap-by-day 95% CI; without the 5 best days; the
  qualification verdict; 2025-26 masked unless `reveal=True`. Ledger started with the baselines.
- **Done when:** baselines are in the ledger with verdicts; the scorecard reproduces the research
  simulator's 2022-24 bp for base within the M4-documented differences; check command passes.
- **Worker:** opus
- **Commits:** 4f13faa, 3ad4aa9, 3018355, 375136d, ecfc29c, ab1b478, ee37c4c

### S1: Exits and sizing (done)

- **Delivers:** on S0 scores (no training): hold 30m/1h/2h/4h; +3/+5/+8% take profit without a
  stop (the post-hoc B1 finding); trailing stops; a -10% safety stop; score-weighted sizing;
  slots x threshold frontier; caps per crash day. Each row in the ledger.
- **Worker:** sonnet
- **Commits:** a15524c, 8e90855, 39d3e9e

### S2: Regime and execution (done)

- **Delivers:** gates (BTC flush depth, market breadth / share of coins spiking, the strategy's
  own trailing hit rate), liquidity and tick-size filters, latency cost, entry delay, entry timing
  inside the spike candle on 1m candles for CNN signals. Each row in the ledger.
- **Worker:** sonnet
- **Commits:** ff4118a, 14ec347

### S3: Training and ensembles (done)

- **Delivers:** rank-within-crash-day loss, net-of-cost / downside-aware targets, 6-monthly
  retraining, 5 seeds, gradient-boosted stacker on CNN outputs + scalars, ensembles of the
  baseline and new variants. Each in the ledger.
- **Worker:** opus
- **Commits:** 2c336e6, c91ae75, 1a9110c, 3ca481f, 68203d0, 2f8e46b, 64f9729

### S4: Inputs (done)

- **Delivers:** cross-coin day context (the coin's rank among the day's crashing coins, number of
  coins crashing, BTC/market depth) as scalars; liquidity, tick and age scalars; longer context
  (3-7 days of hourly candles); 1m candles for the last hour. Cut in reverse order if time runs
  short, and record the cut.
- **Worker:** opus
- **Commits:** 8f97af8, d110db8, f327413, b05826e, 8b51b3e, b137130, 14fe609

### S5: Combine and confirm (in progress)

- **Delivers:** a combination of qualified winners by a rule written here before it runs;
  finalists through the engine with 1m fills, latency and cash limits; 2025-26 opened once;
  deflated Sharpe over the ledger; `docs/research/cnn-robustness.md`; a journal section in the
  user's claude.ai journal (https://claude.ai/artifact/EwkgNPxbXeNexdbwG2eQGf).
- **Combination rule (written 2026-11-07T10:40Z, after S1/S2 results were seen on 2022-24, before
  any S3/S4 result and before any 2025-26 number):**
  - Sizing for every S5 run: `CashCapped` equal slots (stake = min(equity / slots, cash)). The
    strict cash check made S0/S1 results hinge on single skipped LUNA entries.
  - Score sets (3): path_all (reference) and the top 2 qualified, non-fragile score sets by the
    ranking key at the default rule among S3 and S4 variants (if fewer qualify, take what does;
    ens3 fills a free place).
  - Rules (8), each on every score set: (a) default 1h; (b) BTC volume >= 6x gate; (c) BTC 4h <=
    -2% gate; (d) hold 2h; (e) 10 slots, q 0.99; (f) BTC volume >= 6x + hold 2h; (g) BTC volume
    >= 6x + TP +5% / 2h, no stop; (h) spike share >= p75 + hold 2h.
  - Each of the 24 runs uses latency 0.05% per side (S2's estimate of real execution) and the
    usual stresses; all go in the ledger.
  - Pick: the top-ranked qualified, non-fragile combination on 2022-24 (Selection rule ranking).
    Runner-up and the default-rule path_all are kept as comparisons.
  - Reveal: 2025-26 is opened once for ALL 24 S5 runs and the S0 baselines (no hiding of
    losers); the pick's 2025-26 is the headline. The pick and runner-up are also run with 1m
    fills. A deflated Sharpe ratio over the full ledger trial count is reported for the pick.
- **Worker:** opus (report: master)
- **Commits:**

## Decisions

- 2026-11-07 Objective: robust edge first, profit as tiebreak; universe "all"; all directions;
  unattended, 5-10 hours. (user)
- 2026-11-07 Merged `feat/backtest-engine` into this research branch so final checks use the
  production engine; `uv.lock` regenerated with `uv lock`.
- 2026-11-07 Continuous book (not per-year books) for every evaluation: realistic, and the same
  for every variant.
- 2026-11-07 (S0) Scorecard = production engine, continuous book 2022-01-01..2026-11-02 (data end
  fixed as in the golden tests); 2025-26 sealed in gitignored `results/robustness/sealed/`.
- 2026-11-07 (S0) A "fragile" flag is reported with every variant (some 2022-24 year turns
  negative without that year's single best trade). Added after S0 showed base and ens3 pass
  criterion 1 only through single LUNA trades; it is reported, not a qualification criterion,
  so the pre-registered rule stays as written.
- 2026-11-07 S1, S2 (CPU) run alongside S3 (GPU); each commits only its own files and writes the
  ledger under a file lock.

## Research log

- 09:00Z S0: base/market/path_all retrained on "all" (2 seeds, 16-21 min each); scores equal
  PR #8's. 2022-24 bp per trade (engine): path_all +162/+219/+75 (mean +152), ens3 +55/+294/+99
  (+149), market +84/+183/+24 (+97), base +18/+174/+72 (+88); all four qualify. Base 2022 is
  -74 bp vs the research simulator because the engine's cash check skips one LUNA entry
  (+36,437 bp) and the slot chain then differs. Weakest: 2024 one candle late (+15..+28 bp).
- 10:05Z S2 (28 trials; results/robustness/S2.md). 2022-24 mean bp per trade, path_all / ens3
  (baseline +152 / +149): BTC volume >= 6x +291 / +275 (every year above the baseline mean, removes
  ens3's fragility); BTC 4h <= -2% +222 / +227; BTC 4h <= -4% +300 / +356 but 66-80 trades a year
  and +81 / +93 without the 5 best days; spike share >= p75 +222 / +204; own-performance pause
  hurts path_all (+40, fails). Liquidity / tick filters: no reliable gain. Latency 0.05-0.1% per
  side costs 10-20 bp; two candles late leaves +7 / +6 bp in 2024. Entry inside the spike candle
  on 1m (diagnostic): t+4m -90 bp, t+3m -190 bp vs the engine's t+5m entry, every year: wait for
  the close. Note: the gates select the same market-wide flushes that F2 used, which decayed in
  2025; only the S5 reveal can tell.
- Known scorecard gap: the 0.6% stress replaced a rule's costs, dropping latency; fixed in
  20fca48 (affects only S2's two latency rows' 0.6% column).
- 10:40Z S1 (46 trials; results/robustness/S1.md), 2022-24 mean bp, path_all / ens3 (baseline
  +152 / +149): hold 2h +221 / +202, 4h +235 / +253 (weak 2024, low ex-5-days); TP +5% no stop
  1h +109 / +144, TP +5% 2h +146 / +179 (less dependent on best trades, ens3 no longer fragile);
  trailing stops and TP + -10% stop hurt; -10% stop alone neutral; CashCapped +149 / +217 (the
  ens3 gain is one LUNA entry); score-weighted sizing adds nothing; 10 slots x q 0.99 +291 / +378
  on 105-137 trades a year (late +126 / +247); q 0.97 poor; day caps mixed.
- 10:40Z S5 combination rule written into the plan (see S5), before S3/S4 results.
- 12:17Z Usage limit paused both workers from ~10:50Z to 12:00Z. S3's GPU chain finished
  meanwhile (allch_y1h, allch_xs, allch_cost, path_all5, path_all_h6 trained; stack_gbm and
  gbm_scalars built). Workers resumed; S4 GPU budget moved to end by 15:30Z (cut inputs_all,
  then m1last, if needed).
- 12:30Z S3 (9 trials; results/robustness/S3.md), 2022-24 mean bp (path_all +152): path_all5
  +154 (2024 late +6), ens_top (path_all5 + path_all + path_all_h6) +151, ens_s3 (all CNNs) +151
  (best worst year +89, best ex-5-days +118, CI low +86, but late +61), path_all_h6 +149
  (fragile), allch_xs +116, allch_y1h +114, allch_cost +82 (fragile); the GBM stacker (+43 on
  2023-24) and scalars-only GBM (+56) are worse and unscored for 2022. Nothing beats path_all
  beyond seed noise (~10-20 bp a year); training changes buy robustness (ensembles), not edge.
- 14:45Z S4 (4 trials; results/robustness/S4.md), 2022-24 mean bp: ctx (cross-coin day context +
  coin scalars) +188 (+268/+218/+79; 0.6% +159, late +111, ex-5-days +112, CI [+97, +280]; DD
  about half of path_all's), not fragile; nearly all the gain is 2022. long7d +140, inputs_all
  +129, m1last +119 (fragile): longer or finer history doesn't help. ctx overlaps the S2 BTC gates.
- 14:45Z S5 score sets by the fixed rule: path_all (reference), S4:ctx (+188), S3:path_all5 (+154).
  Ledger: 91 trials before S5.

## Blocked on

## Follow-ups
