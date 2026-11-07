# CNN robustness study: deeper and wider on the capitulation CNN

- **Status:** building <!-- building | blocked | review | done -->
- **Branch:** `claude/cnn-robustness` (from `claude/cnn-replication`, with `feat/backtest-engine` merged in)
- **PR:** none (research branch; report in the session, open a PR only if asked)
- **Updated:** 2026-11-07T10:05Z

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

### S1: Exits and sizing (in progress)

- **Delivers:** on S0 scores (no training): hold 30m/1h/2h/4h; +3/+5/+8% take profit without a
  stop (the post-hoc B1 finding); trailing stops; a -10% safety stop; score-weighted sizing;
  slots x threshold frontier; caps per crash day. Each row in the ledger.
- **Worker:** sonnet
- **Commits:**

### S2: Regime and execution (done)

- **Delivers:** gates (BTC flush depth, market breadth / share of coins spiking, the strategy's
  own trailing hit rate), liquidity and tick-size filters, latency cost, entry delay, entry timing
  inside the spike candle on 1m candles for CNN signals. Each row in the ledger.
- **Worker:** sonnet
- **Commits:** ff4118a, 14ec347

### S3: Training and ensembles (in progress)

- **Delivers:** rank-within-crash-day loss, net-of-cost / downside-aware targets, 6-monthly
  retraining, 5 seeds, gradient-boosted stacker on CNN outputs + scalars, ensembles of the
  baseline and new variants. Each in the ledger.
- **Worker:** opus
- **Commits:**

### S4: Inputs (in progress)

- **Delivers:** cross-coin day context (the coin's rank among the day's crashing coins, number of
  coins crashing, BTC/market depth) as scalars; liquidity, tick and age scalars; longer context
  (3-7 days of hourly candles); 1m candles for the last hour. Cut in reverse order if time runs
  short, and record the cut.
- **Worker:** opus
- **Commits:**

### S5: Combine and confirm (todo)

- **Delivers:** a combination of qualified winners by a rule written here before it runs;
  finalists through the engine with 1m fills, latency and cash limits; 2025-26 opened once;
  deflated Sharpe over the ledger; `docs/research/cnn-robustness.md`; a journal section in the
  user's claude.ai journal (https://claude.ai/artifact/EwkgNPxbXeNexdbwG2eQGf).
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
- Known scorecard gap: the 0.6% stress replaces a rule's costs, dropping latency; fix before S5.

## Blocked on

## Follow-ups
