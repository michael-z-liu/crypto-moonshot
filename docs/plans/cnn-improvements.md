# CNN improvements: testing the ideas of issue #10

- **Status:** building <!-- building | blocked | review | done -->
- **Branch:** `feat/cnn-improvements` (from `claude/cnn-robustness`; the PR is stacked on it)
- **PR:** <!-- link, once opened -->
- **Updated:** 2026-11-11T00:40Z

## Goal

Issue #10 reviews the walk-forward `path_all` capitulation CNN (`research/volume_spike/m12_cnn_wf.py`,
study `docs/research/cnn-robustness.md`) and ranks eight improvement ideas. The main weakness: the
model is trained and early-stopped on all ~1.85M spike rows, but only earns on the top 1-2% of
scores during a handful of crash days. Test ideas 1-6 and 8, report what holds up, and open a PR
stacked on `claude/cnn-robustness`.

## Out of scope

- Idea 7 (funding rate, open interest, perp basis from Binance futures): a new data source, a
  separate follow-up project.
- Placing orders; the P1 paper test running daily in the `claude/cnn-robustness` worktree.
- Changing `main`, PR #8, PR #9 or `claude/cnn-robustness`.
- Rewriting any shared cache (see Ground rules).

## Ground rules

- Universe "all" (582 USDT pairs incl. delisted), `MOONSHOT_UNIVERSE=all`, run from
  `research/volume_spike/` with `uv run --group research python ...`.
- `research/volume_spike/cache_all/` in this worktree is a local (gitignored) directory. Its input
  files are symlinks into the shared, READ-ONLY `/Users/mike/code/crypto-moonshot/data/cache/volume_spike_all/`
  (events, fwd, fwd2, fwd72, long30, market, seq2, seq2_meta, m1). Existing walk-forward scores
  are symlinked read-only from the robustness worktree
  (`.claude/worktrees/bridge-cse_015bssARehNUJUAkMqq7L4zo/research/volume_spike/cache_all/wf/`).
  NEVER write through a symlink (np.save / write_parquet onto an existing linked path). New files
  are created as new local files in `cache_all/` (e.g. `cache_all/wf_i10/`).
- Market data: `/Users/mike/code/crypto-moonshot/data` (default `DATA_DIR`), read only.
- GPU (MPS): one training at a time. CPU scoring may run alongside.
- Data end for every book: 2026-11-02 (exclusive), as in the robustness study.

## Evaluation protocol (fixed 2026-11-10T12:28Z, before any #10 result)

**Book.** The robustness scorecard (`scorecard.py`, production engine, 5m candles, one continuous
book 2022-01-01..2026-11-02, $10,000), with S5's realistic setup: `CashCapped(5)` equal slots,
pct >= 0.98, highest score first, enter at the next 5m open, 1h hold unless the variant says
otherwise, fee 0.1% + slippage 0.05% + latency 0.05% per side (0.4% round trip). Stresses: 0.6%
round trip (latency kept) and one candle late. Placebo: 20 within-day shuffles.

**Ledger.** A separate ledger and sealed folder under `research/volume_spike/results/issue10/`
(sealed folder gitignored). Every variant tried is a row.

**Baseline.** `base3`: path_all retrained with the M1 harness, 3 seeds, scored with the rule
above (the counterpart of `S5:path_all:a_1h`, +138 bp on 2022-24).

**Qualifies** (on 2022-24, the seed-mean score): bp per trade > 0 every year; mean > 0 at 0.6%;
mean > 0 one candle late; mean above the placebo's 95th percentile. Ranked by the mean of the
three yearly bp, worst year breaking ties. The "fragile" flag is reported.

**Seed noise.** Every trained variant has >= 3 seeds and is also scored seed by seed (each seed's
own percentiles). The noise band is the spread (max - min) of base3's per-seed 2022-24 mean bp.
A variant **beats the baseline** only if it qualifies, is not fragile, and its seed-mean 2022-24
mean bp exceeds base3's by more than that spread. No-training rules (M2) are compared on base3's
seed-mean scores and must also beat the base rule by more than the spread.

**Sealing.** 2025-26 is computed but written only to the sealed folder until M5. Nobody (master
or worker) opens it before the M5 pick is committed.

**Combination rule (M5).**
- Training variants: the top two from M3/M4 that beat the baseline (if fewer, take what does; if
  none, base3 alone), each retrained with 5 seeds; plus one variant combining both changes when
  two exist and the changes are compatible.
- Rules, on each: (a) the 1h default; (b) a 2h hold; (c) the best M2 rule that beats the base
  rule (if none, the M2 rule with the highest 2022-24 mean that qualifies); (d) rule c with a 2h
  hold (if c already holds 2h, rule c with the 1h hold instead).
- Pick: the top-ranked qualified, non-fragile run on 2022-24. It is committed before 2025-26 is
  opened.
- Reveal: 2025-26 is opened once for every #10 ledger row and base3. Report the pick's 2025-26,
  the 2025-10-10 loss of every row, 1h vs 2h, the Spearman correlation between the 2022-24 mean and
  2025 bp, and a deflated Sharpe ratio over all trials (the robustness study's 115 plus the #10
  rows). 2025-26 is a spent holdout: it was seen in PR #8 and the robustness study, and ideas 3
  and 4 were inspired by 2025; the report says so.

## Milestones

### M0: Setup and reproduction (done)

- **Delivers:** branch and plan; `cache_all/` links; a reproduction of `S5:path_all:a_1h` from the
  linked `path_all.npz` with the scorecard.
- **Done when:** the reproduced 2022-24 yearly bp equal the robustness ledger's row; check command
  passes.
- **Worker:** master
- **Commits:** (plan only; the links are local and gitignored)

### M1: Training harness and the base3 baseline (done)

- **Delivers:** a harness (extending `m12_cnn_wf.py` or a new module registering into it) with
  options for: early stopping on correlation (current), on the mean target of the top 2% of
  validation scores, or a fixed epoch count; loss weights (none, equal per UTC day, tail, recency
  half-life); training window (expanding or rolling N years); normalisation statistics frozen
  pre-2021-07 (current) or per fold from the fold's training rows; volatility scaling of target
  and price channels; target relative to the universe's forward return (hook only; data in M4);
  extra score heads for path models (1h median, 2h median, 2h median / (q90 - q10)), each with
  fold percentiles and per-seed percentiles saved. An `issue10` scorecard config (own ledger and
  sealed folder, per-seed scoring helper). base3 trained (path_all, 3 seeds).
- **Done when:** with default options the harness reproduces `path_all.npz` (seeds 0-1 give the
  same pct on every row, or a documented tolerance if MPS is non-deterministic); base3 is in the
  ledger with per-seed rows and the noise band; check command passes.
- **Worker:** opus
- **Commits:** 07491a0, 488c6b0, bd56910, 373c29b, c92b178

### M2: Cascade caps and score heads, no training (done)

- **Delivers:** on base3 scores (CPU): idea 3, caps per cascade and per day, a daily loss limit, a
  ranking penalty for entries already taken in the same event and for time since the flush began;
  idea 4, ranking on the 2h median with a 2h hold and on the risk-adjusted head. Each a ledger row,
  scored also per seed; a results note.
- **Done when:** every rule is in the ledger with verdict and per-seed spread; note written;
  check command passes.
- **Worker:** sonnet
- **Commits:** 66a75a8, 4af3289, a474115

### M3: Objective and sampling, ideas 1-2 (in progress)

- **Delivers:** trained and scored (3 seeds): `es_top` (early stop on top-2% validation return),
  `tail` (tail-weighted loss + es_top), `day` (equal weight per UTC day), `fixed` (fixed epochs,
  no early stopping). Results note.
- **Done when:** four variants in the ledger with per-seed rows; note written; check passes.
- **Worker:** sonnet
- **Commits:**

### M4: Target, scaling and stationarity, ideas 5, 6, 8 (in progress)

- **Delivers:** a builder for the universe's forward return path per signal time (new local file);
  trained and scored (3 seeds): `xs_univ` (target minus the universe's forward return over the same
  window), `volscale` (target and price channels scaled by trailing volatility, converted back for
  ranking), `recency` (recency-weighted loss), `roll3y` (rolling 3-year window), `foldnorm`
  (normalisation per fold). Cut in the order foldnorm, roll3y, recency if GPU time runs short.
  Results note.
- **Done when:** variants in the ledger with per-seed rows; builder checked for look-ahead;
  note written; check passes.
- **Worker:** opus
- **Commits:**

### M5: Combine, reveal and report (todo)

- **Delivers:** the combination rule above (5 seeds), the pick committed before the reveal, the
  reveal, the DSR, `docs/research/cnn-issue10.md`, the PR (refers to #10, does not close it).
- **Done when:** report written with every row's 2022-24 and 2025-26 numbers; check passes; PR open.
- **Worker:** opus (report and PR: master)
- **Commits:**

## Decisions

- 2026-11-10 Evaluation: pre-register on 2022-24, reveal 2025-26 once for every variant, report
  both, labelled as a spent holdout. (user)
- 2026-11-10 Idea 7 (futures data) is a follow-up project. (user)
- 2026-11-10 GPU budget up to ~10 hours. (user)
- 2026-11-10 PR stacked on `claude/cnn-robustness`. (user)
- 2026-11-10 (M1) Harness is `i10_train.py` (m12 untouched); scoring `i10_score.py`; scorecard got
  `configure(out)` so the robustness ledger is never written. Top-2% selection metric uses the hold
  target clipped to +-20%; weights none/day/tail (1 + 3 x crash mask)/recency; volscale divides
  price channels 0-3 and the target by the trailing 96-candle std of 5m log returns (floor 0.001).
- 2026-11-10 (M1) base3's seed-mean (3 seeds) is +120.9 bp and fragile (2022 without its best
  trade -19.6), below the 2-seed path_all's +138.4; noise band 26.8 bp, so "beats the baseline"
  needs > +147.7 bp, qualified and not fragile. Kept as pre-registered.
- 2026-11-10 (M1) The base3 run printed two 2025 H2 *validation* top-2% means (fold 2026's early
  stopping half-year), a return-like number, not a book result. Now logged only to the gitignored
  log CSV for validation periods reaching 2025; fold-2026 rows of `wf_i10/*_log.csv` are not read
  before M5. Recorded as a minor breach of the seal; it can't have influenced any choice.
- 2026-11-10 M2 (CPU) and M3 (GPU) run at once in the same worktree; each commits only its own
  paths. M2 owns `i10_score.py`; M3 owns the variant registry in `i10_train.py`.
- 2026-11-11 (M2) The engine gained `max_entries_per_group` and `group_tiers` (signal columns
  `group`, `score`; default off, tested) for the cascade caps and the escalating threshold; a
  `RiskCheck` cannot see a signal's cascade or score. The daily loss limit is the existing
  `DailyLossLimit` (realised P&L only).
- 2026-11-11 (M2 -> M5) By the pre-registered rule, M5 rule (c) is `h2_2h` (2h-median head, 2h
  hold: highest qualifying 2022-24 mean that beats the base rule, +248.1) and rule (d) is its 1h
  variant `h2_1h`; rule (b) is `h1_2h`. Note: per seed, `h2_2h` (avg +140.0) is below `h1_2h`
  (+154.2) and only `esc` (escalating threshold per cascade; per-seed avg +147.7 vs base +114.0,
  up in every seed) and the 2h hold improve every seed. So M5 also runs `esc` and `esc` + 2h hold
  on each score set as comparison rows that are NOT eligible for the pick (decided now, before any
  M3 result and before any 2025-26 number).

## Research log

- 12:35Z M0: `cache_all/` linked (shared inputs, robustness `wf/` scores, `s4_ctx.parquet`). The
  scorecard on the linked `path_all.npz` with S5's `a_1h` rule gives 2022 / 2023 / 2024 +147.13 /
  +206.11 / +62.08 bp on 279 / 138 / 317 trades, mean +138.44: identical to the robustness ledger's
  `S5:path_all:a_1h` row.
- M1: base3 reproduces path_all (seeds 0-1 bit-exact on 1,456,656 rows). 2022-24 seed-mean
  +106 / +208 / +48, mean +120.9 (0.6% +90.6, late +60.4, fragile); per seed +110.1 / +102.6 /
  +129.4; noise band 26.8 bp. 29.7 min for 5 folds x 3 seeds.
- M2 (14 rules, base3 scores, 2022-24 seed-mean mean bp vs base +120.9): h2_2h +248.1, h1_2h
  +211.2, h2_1h +173.8, daycap5 +172.9, esc +166.4 beat the base rule; age180 +145.0, casc10
  +137.7, age60 +129.1, h2ra_2h +122.5, dll3 +98.7, daycap10 +94.7, dll6 +92.8, h2ra_1h +57.3
  don't; casc5 fails (2022 -107). Per-seed averages disagree with several seed-mean rows (daycap5
  106.4, h2_1h 115.5, h2_2h 140.0 vs base 114.0); esc (147.7) and the 2h hold (154.2) gain in
  every seed. The risk-adjusted head is clearly worse. Caps and loss limits mostly hurt.

## Blocked on

## Follow-ups

- Idea 7: funding rate, open interest and perp basis from Binance futures as model inputs.
