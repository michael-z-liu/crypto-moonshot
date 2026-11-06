# Replicate the PR #8 CNN results on a new computer

- **Status:** building <!-- building | blocked | review | done -->
- **Branch:** `claude/cnn-replication` (from PR #8's `claude/high-volume-trading-strategy-0c5d1e`)
- **PR:** none yet (report to the user in the session; open a PR only if asked)
- **Updated:** 2026-11-06T08:20Z

## Goal

Before exploring the CNN further on this computer (Apple M4, 24 GB), re-download the data into a
folder that every session and worktree can share, and replicate a few of PR #8's results closely
enough to trust the setup. Report back when they match.

## Ground rules

- Shared data folder: `MOONSHOT_DATA_DIR=/Users/mike/code/crypto-moonshot/data` (the main
  checkout's gitignored `./data`), now also the default in `research/volume_spike/common.py`.
- Shared research caches: in each worktree, `research/volume_spike/cache` is a symlink (gitignored)
  to `/Users/mike/code/crypto-moonshot/data/cache/volume_spike`, so the event table and sequence
  datasets are built once per computer. Create it in a new worktree with
  `ln -s /Users/mike/code/crypto-moonshot/data/cache/volume_spike research/volume_spike/cache`.
  Two sessions must not rebuild the same cache file at the same time; walk-forward runs write
  one `cache/wf/<variant>.npz` each, so use distinct variant names in parallel sessions.
- The old pool list (`top200.txt`) was never committed. It was rebuilt from today's exchange info
  minus `results/archive_groups.json` (198 symbols, 152 after pegged/stocks, the same count as
  PR #8), and saved as `research/volume_spike/results/pool_symbols.txt` and in the data folder.
- Data range for the replication: 5m candles 2020-01-01 to 2026-11-02 (`--end 2026-11-02`), the
  range the original run had. Later sessions may extend it; that changes 2026 numbers a little.
- Pool universe only (no delisted archive) for the replication.
- Re-running the scripts overwrites committed tables in `results/`. Compare against `git show
  HEAD:<file>`, keep the new numbers in `results/replication/`, then restore the originals.
- MPS and seeds: exact matches are not expected for the CNNs (different machine, MPS
  nondeterminism); "similar" means the same sign and rough size per year.

## Milestones

### R1: Data (done)

- `moonshot data sync` of the 198 pool symbols, 5m, 2020-01-01..2026-11-02 into the shared folder.
  Log: `/Users/mike/code/crypto-moonshot/data/logs/sync-5m-pool.log`. Re-run the same command to
  resume (finished months are kept).
- **Done when:** the sync exits 0 and `moonshot data status` shows all 198 series.

### R2: Rule sanity check (done)

- `build_events.py`, then `m6_test.py`. **Reference** (`results/m6_test_output.txt`), F2 bp per
  trade by year 2020..2026: +222 / +319 / +223 / +287 / +284 / −44 / +187 (114/243/126/74/156/133/45
  trades). Should match almost exactly if the data matches.

### R3: M7 CNN (fixed split) (done)

- `build_seq.py`, `m11_cnn.py train`, `report`, `portfolio`. **Reference**
  (`results/m11_cnn_portfolio.csv`), coin+BTC CNN, 5 slots: valid 2024 +218 bp (118 trades), test
  2025–26 +114 bp (254 trades; 2025 +120, 2026 +106).

### R4: Walk-forward CNNs (done)

- `build_seq2.py` (+ `fwd`), `m12_cnn_wf.py run base path_all`, then `eval`. **Reference**
  (`results/m12_variants.csv`, 5 slots, top 2%, 1h), bp per trade 2022..2026:
  - base: +334 / +170 / +57 / +67 / +53
  - path_all: +278 / +206 / +107 / +65 / +101
  - market (optional): +257 / +184 / +89 / +88 / +175

### R5: Report to the user (done)

- Table of reference vs replicated numbers, and anything that differs.

### B1: Bracket CNN, +5% / -2% (in progress, asked 2026-11-06)

User ask: a shorter time frame, many small high-conviction trades, each with about +5% upside and a
-2% stop, using the CNN. Journal (claude.ai doc, the user's page for this line of work):
https://claude.ai/artifact/EwkgNPxbXeNexdbwG2eQGf. Update it as results come in.

**Pre-registered protocol (written before any bracket model was trained):**
- Data: the walk-forward dataset (`seq2`, 632k liquid spikes, 24h x 13 channels), folds as M9: for
  test year Y in 2022..2026 train before Y-1 July, early-stop on Y-1 July-Dec, score Y; 2 seeds.
- Label: enter at the next candle's open; take profit +5%, stop -2%, time stop H; if both are
  touched in one 5m candle the stop counts first; 0.3% round trip. Target = that net return (%).
- Base rates (no model, all spikes): H=1h TP-first 5.5%, stop 24.8%, -29 bp; deep flushes
  (coin -8% in 4h, BTC volume >= 4x) TP-first 20.3%, stop 47.0%, +23 bp. Break-even win rate
  (TP vs stop only) 33%.
- Variants (`m17_bracket.py`): br1h (COIN+BTC, 8h window, H=1h, MSE) = primary; br1h_w2h (2h
  window); br1h_mkt (+ market channels); br1h_bce (classify TP-first); br2h (H=2h); br30m (H=30m).
  Comparison without retraining: base and path_all scores traded with the bracket.
- Default rule: top 2% of the fold's training scores, 5 slots, bracket exit. Selection: only
  2022-24 decides (mean bp per trade, every year >= 0); 2025-26 is reported, not used to choose.
  The equal-weight ensemble of all br* variants is the no-selection alternative.
- Volume vs conviction frontier: thresholds top 0.5/1/2/5/10% x 5/10/20 slots, per year.
- Stress: 0.6% cost, extra 0.2% slippage on stop fills, entry one candle late, placebo
  (scores shuffled within each day).

## Resume

If the session stops (usage limit), the heartbeat (cron in the session) or the user says
"resume the CNN replication": read this file, check the log below and the processes still
running (`pgrep -fl "moonshot|volume_spike"`), and continue from the first milestone not done.
Heavy outputs are in `research/volume_spike/cache/` (gitignored), so finished stages need not be
re-run.

## Research log

- 2026-11-06 06:10Z Reviewed PR #8. Installed uv (brew), `uv sync --group research` (torch 2.15,
  MPS available). Rebuilt the pool list (198). Started the R1 sync (~65M candles, ETA ~30 min).
- 06:53Z R1 done: 64.8M candles, 198 series, 2.0 GB, 0 rate-limit hits, 44 min.
- 06:56Z R2 done: build_events 2,258,333 rows / 1,972,584 spikes (PR 8: ~2M). m6_test: F1, F2, F3
  identical to PR 8 in every year (trades and bp); only the 2019 reference rule differs by 2-18
  trades of ~13k a year. Output in `results/replication/`.
- 06:58Z R3: build_seq (1,972,584 x 8 x 96, no NaN); `m11_cnn.py train` started, log
  `cache/logs/m11_train.log`.
- 07:05Z R4: build_seq2 632,082 x 13 x 288 (PR 8: 632k), fwd built. `m12_cnn_wf.py run base
  path_all` started next to the M7 training (both on MPS), log `cache/logs/m12_run.log`; outputs
  `cache/wf/base.npz`, `cache/wf/path_all.npz`. Not yet run: `eval` (overwrites
  results/m12_variants.csv; restore it afterwards).
- 07:20Z R4 done: base (~9 min) and path_all (~10 min) re-trained from scratch on MPS. `eval`
  reproduces PR 8 exactly: base +334/+170/+57/+67/+53, path_all +278/+206/+107/+65/+101 bp per
  trade 2022..2026, same trade counts and drawdowns (training is deterministic with fixed seeds,
  the locked torch and the same M4 chip). Table in `results/replication/m12_variants.csv`;
  committed results/m12_variants.csv restored.
- 07:30Z R3 done: `m11_cnn.py train` (2 variants x 3 seeds x 8 epochs, 24 min next to R4) and
  `portfolio`. results/m11_cnn_portfolio.csv is byte-identical to PR 8: coin+BTC CNN valid 2024
  +218 bp (118 trades), test 2025-26 +114 bp (254 trades; 2025 +120, 2026 +106), DD -11%; boosted
  trees test +10 bp; F2 +15 bp. Training tables differ only in the `secs` column and float
  rounding at the 15th digit. Copies in `results/replication/`.
- 07:30Z R5: replication complete; reported to the user. Heartbeat deleted. Caches for further CNN
  work are ready in the shared cache folder (events, seq, seq2, fwd2, wf/base, wf/path_all).
