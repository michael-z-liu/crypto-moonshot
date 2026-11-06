# Replicate the PR #8 CNN results on a new computer

- **Status:** building <!-- building | blocked | review | done -->
- **Branch:** `claude/cnn-replication` (from PR #8's `claude/high-volume-trading-strategy-0c5d1e`)
- **PR:** none yet (report to the user in the session; open a PR only if asked)
- **Updated:** 2026-11-06T06:58Z

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

### R3: M7 CNN (fixed split) (in progress)

- `build_seq.py`, `m11_cnn.py train`, `report`, `portfolio`. **Reference**
  (`results/m11_cnn_portfolio.csv`), coin+BTC CNN, 5 slots: valid 2024 +218 bp (118 trades), test
  2025–26 +114 bp (254 trades; 2025 +120, 2026 +106).

### R4: Walk-forward CNNs

- `build_seq2.py` (+ `fwd`), `m12_cnn_wf.py run base path_all`, then `eval`. **Reference**
  (`results/m12_variants.csv`, 5 slots, top 2%, 1h), bp per trade 2022..2026:
  - base: +334 / +170 / +57 / +67 / +53
  - path_all: +278 / +206 / +107 / +65 / +101
  - market (optional): +257 / +184 / +89 / +88 / +175

### R5: Report to the user

- Table of reference vs replicated numbers, and anything that differs.

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
