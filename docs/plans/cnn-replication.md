# Replicate the PR #8 CNN results on a new computer

- **Status:** building <!-- building | blocked | review | done -->
- **Branch:** `claude/cnn-replication` (from PR #8's `claude/high-volume-trading-strategy-0c5d1e`)
- **PR:** none yet (report to the user in the session; open a PR only if asked)
- **Updated:** 2026-11-06T14:17Z

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

### B1: Bracket CNN, +5% / -2% (done, asked 2026-11-06)

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

### B2: Long shots, $50 a trade, no stop (done, asked 2026-11-06)

User ask: fund $10,000, buy many $50 positions with no stop, aim for +30-50% per winning trade;
explore a few variations. Up to 30 days holding agreed; delisted coins downloaded first (user OK).

**Pre-registered protocol (written before any long-horizon result was seen):**
- Universe: "all" = pool + 191 delisted + 239 listed outside the pool (archive, imported into the
  shared data folder); pegged coins and tokenised stocks excluded. Cache: `cache_all/` (symlink
  into the shared folder).
- Outcomes (`build_long30.py`): for every event row (spikes + random bars), entry at the next 5m
  open; on the gap-free 5m grid (missing candles carry the last close; after a series ends, the
  last close holds), the first time the high reaches +20/+30/+50/+100%, and the close and lowest
  low after 1, 3, 7 and 30 days. Entries after 2026-10-01 are dropped (30 days must fit).
- Exits: take profit TP in {20, 30, 50, 100%} or the time limit H in {1, 3, 7, 30 days},
  whichever comes first; no stop. Reference: no TP, hold H. Cost 0.3% round trip.
- Book: $10,000, $50 per trade, at most 200 open positions, one per coin; signals in time order,
  same-candle ties by priority; signals that find no room are skipped. P&L summed (no compounding).
  Report by entry year: trades, hit rate, mean net % per trade, P&L in $ and % of $10k, worst
  realised drawdown in $.
- Entries: R0 random 5m bars (baseline); R1 every spike (volume >= 10x its 24h median) with
  >= 1k USDT median per candle; R2 capitulation: R1 + coin -10% in 4h + BTC volume >= 6x; R3 deep
  crash: R1 + coin -30% in 24h; R4 breakout: R1 on an up candle at a 7-day high. CNN: liquid spikes
  (>= 5k USDT, the `seq2` dataset on "all"), all 13 channels, 24h window, classifying "+30% within
  7 days", walk-forward as M9 (2 seeds), traded at the top 1/2/5% of scores.
- Selection: 2022-24 decides (rules also shown for 2020-21); 2025-26 reported, not used to choose.
  Every result is shown next to R0 random with the same exit.

### B3: Quiet coins before a spike (done, asked 2026-11-06)

User ask: go back to the original idea. Enter while a coin is quiet (little activity, little price
movement) and predict an imminent upward spike, instead of entering after a crash. One full round:
rules first, then a CNN, then a fill check; unattended overnight run authorised, judgment calls
delegated, a detailed report in the journal (claude.ai doc) expected.

**Pre-registered protocol (written before any quiet-coin result was seen):**
- Snapshots (`build_quiet.py`, cache_all/quiet.parquet): every coin in the "all" universe, every
  4 hours (the candle closing at 00:00, 04:00, ... UTC), from 30 days after listing; entry at the
  next 5m open. Features use candles up to the snapshot only.
- Quietness, each relative to the coin's own past (ratio to its median over the previous 180 days
  of snapshots, at least 30 days of history): 1-day and 7-day realised volatility of 5m returns,
  1-day and 7-day high-low range, 1-day and 7-day quote volume. Plus: share of unchanged 5m
  closes in the last day, days since the last 10x volume spike, 1/7/30-day returns, distance from
  the 30-day high, taker-buy share (1 day minus 7 days), trade count and average trade size
  relative to 30 days, price tick size, coin age, BTC 1-day/7-day return and volatility ratio.
- Quiet scores: Q1 (1 day) = geometric mean of the 1-day volatility, range and volume ratios;
  Q7 (7 days) = the same over 7 days. Deciles cut on 2020-2021 snapshots only, applied to all years.
- Outcomes (same method as build_long30): first time the high reaches +10/+20/+30/+50/+100% and
  the low reaches -10/-20/-30/-50% within 30 days; close and low/high after 1, 3, 7, 30 days.
  A touch counts. Only snapshots with entry before 2026-10-01 and >= 1k USDT median per 5m candle
  over the last day are used.
- Step 1 (no model): by Q1 and Q7 decile and by year, P(+20% within 3 days), P(-20% within 3 days),
  P(+20% before -20%), mean 3-day and 7-day return; compared with all snapshots. The idea passes
  if the quietest 10-20% show more up than down moves than ordinary snapshots in 2022-24, checked
  in 2025-26.
- Step 2 (rules, $50 book as B2: $10,000, $50 a trade, <= 200 open, one per coin, no stop, TP
  +20/30/50/100% or 1/3/7/30 days, 0.3% cost): Q1 decile 1, Q7 decile 1, both, and "quiet + early
  buying" (Q7 decile 1-2 with taker-buy share up and trade count up). Chosen on 2022-24 only.
- Step 3 (CNN on quiet snapshots, Q7 deciles 1-3): input 7 days of 1h candles (coin OHLC, volume,
  taker share, trades, avg trade size; BTC close and volume) plus the last 24h of 5m candles, plus
  the scalar features; target "+20% within 3 days" (classification); walk-forward as M9 (train
  before Y-1 July, early-stop on Y-1 July-Dec, score Y), 2 seeds; traded at the top 1/2/5% with
  the same book. Compared with the quiet rules and with random quiet snapshots.
- Step 4: 1m candles around the chosen strategy's take-profit hits (data.binance.vision daily
  1m files for those days) to check the +20% target would have filled (price at or above the
  target for at least 1 minute, not one tick).
- Every result also without 2025-10-10/11. Report: journal section "Study 3".
**Round 2, pre-registered before any result (the judgment call: round 1 tested quiet without the
spike, PR 8 tested the spike without the quiet; the 2019 idea is the two together):**
- Events: 10x volume spikes in the "all" events table (cache_all/events.parquet, >= 1k USDT median
  per 5m candle), joined as-of to the same coin's latest quiet snapshot at least 5 minutes before
  the spike candle (so the spike is not inside the quiet measure; at most 4 hours old). Entry at
  the next 5m open (as everywhere), outcomes from cache_all/long30.parquet.
- Rules (`m22_quiet_spike.py`): S1 quiet before (Q7 decile <= 2) + up candle; S2 = S1 + taker-buy
  share > 0.6 + run-up in the previous hour <= 2% (the 2019 rule inside a quiet spell); S3 = S1 +
  first spike for >= 3 days (days_since_spike at the snapshot); S4 quiet before, any candle;
  baselines: B-busy (Q7 decile >= 5) + up candle, and every up-candle spike.
- Same $50 book and exit grid as B2 (TP 20/30/50/100% or 1/3/7/30 days, no stop), chosen on 2022-24.
  Also the mean 1h and 4h return after entry (from fwd.npy) to see how much of the move is already
  over at entry. Secondary: the B2 long-shot CNN scores (cache_all/wf/longshot.npz) restricted to
  quiet-context spikes.


### B4: A trigger in the first minute of a spike (in progress, asked 2026-11-07)

User ask: react in the first minute of a spike, using 1-minute (or trade-by-trade) data; get the
data and validate the idea. Report in the journal (new section "Study 4").

**Pre-registered protocol (written before any 1-minute result was seen):**
- Data (`fetch_1m.py`): Binance monthly 1m kline archives (data.binance.vision, checksum-verified)
  for every coin in the "all" universe, every month it has 5m data, 2020-01..2026-10, written with
  the repo's KlineStore into the shared data folder (`klines/spot/interval=1m`). Months without an
  archive file are listed, not filled.
- Part A, speed on the known edge: the B2 pick (R2 capitulation, $50, +20% or 3 days, no stop).
  For each signal's 5m spike candle (5 one-minute candles), the 1m trigger is the first minute
  whose volume is >= 10x the median 1m volume of the previous 24 hours (if none, the last minute).
  Entry at the next 1m open; the trade then runs on 1m candles (+20% first touch, else the close
  3 days after entry). Compared with the 5m entry (open after the 5m candle), same signals, same
  book. Passes if the 1m entry earns more in 2022-24 and in 2025-26.
- Part B, the quiet idea at 1-minute speed: 1m spikes (volume >= 20x the 24h 1m median, an up
  minute, >= 5k USDT traded in the minute) on coins that were quiet before (latest 4-hourly
  snapshot that closed before the minute, Q7 decile <= 2). Entry at the next 1m open; exits: take
  profit +5/+10/+20% or 15 minutes / 1 hour / 4 hours / 1 day, no stop; the $50 book (200 slots,
  one per coin). Compared with the same events entered 5 minutes later (what 5m data allows) and
  with 1m spikes on busy coins (Q7 decile >= 5). Chosen on 2022-24, reported for 2025-26; also at
  0.6% costs (spreads are wider in a spike) and without 2025-10-10/11.
- Part C, trade-level check on a sample: for 200 random trades of the best Part A or B rule,
  Binance daily aggTrades files; the first trade price after the trigger minute closes vs the 1m
  open assumed (slippage), and how many trades print in that first minute.

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
- 08:50Z B1 interim (br1h, br1h_w2h done): top 2%, 5 slots, bracket: br1h +61/+36/+31/-1/-31 bp
  per trade 2022..2026 (150-412 trades a year); br1h_w2h +30/+10/+21/-2/+8. PR 8 base CNN traded
  with the bracket instead of a 1h hold: +45/+47/+47/+27/-2 (vs +334/+170/+57/+67/+53 with the 1h
  hold). Why: the stop is hit inside the first 5m candle on 25-28% of top-2% signals (median
  first-candle range 2.5-2.9%); same-candle TP/SL ambiguity is only 4-5%. A -2% stop sits inside
  normal noise at capitulation moments.
- 09:40Z B1 results (pre-registered). Top 2%, 5 slots, +5%/-2% bracket, bp per trade 2022..2026:
  br1h +61/+36/+31/-1/-31, br1h_w2h +30/+10/+21/-2/+8, br1h_mkt +30/+50/+5/+16/+114, br1h_bce
  +49/+49/+2/-31/-62, br2h +47/+23/+66/+3/-1, br30m +64/+26/+43/+20/-38, ensemble +79/+10/+56/-1/
  +93; base+bracket +45/+47/+47/+27/-2. Selection rule picks br2h: 2025-26 flat. Looser thresholds
  (top 5-10%, 10-20 slots: 1,000-8,000 trades a year) lose 0.1-0.25% per trade. Stress (br2h):
  0.6% cost negative in 2023/25/26; one candle late -26 bp in 2025; placebo -31 to -71 bp. Verdict:
  the +5%/-2% bracket fails; the CNN ranks (placebo far worse) but the -2% stop is inside noise.
- 09:55Z POST-HOC (`m17_shapes.py`, results/m17_shapes.csv, m17_tp_only.csv): same scores, other
  exits. Removing the stop and keeping the +5% target with a 1h time exit: bracket ensemble top 2%,
  5 slots +153/+172/+185/+111/+157 bp; top 5%, 5 slots +133/+113/+66/+120/+41 bp (106-177 trades
  a year, returns +58/+26/+24/+34/+4%); top 10%, 20 slots +87/+82/+33/+81/+29 bp (309-841 trades).
  Positive every year at 0.6% cost; one candle late -31 bp in 2025; placebo -50 to -156 bp
  (2026 +75 on 22 trades). Worst trade -35% (no stop). Needs a pre-registered confirmation.
- Simulator note: `SimResult.stats()` total_x / cagr divide by the equity at the end of the first
  day, so the first day's P&L is left out (per-trade numbers unaffected). New code reports the
  sum of stake x return instead; `portfolio.py` left unchanged so PR 8 numbers stay comparable.
- 10:15Z B1 done; journal filled (results chart, volume table, conclusions). Proposed next steps,
  waiting for the user: (1) pre-registered confirmation of a +5% take profit / 1h / no tight stop
  (+ a -8..-10% safety stop) with a CNN trained on that target, tested on the archive coins outside
  the pool; (2) 1m candles around the trades (entry speed, delayed stops); (3) paper trading.
- 11:30Z B2 setup: archive fetch running (`fetch_archive.py fetch` then `publish`, log
  `research/volume_spike/cache/logs/archive_fetch.log`; ~140 months a minute, ~2h; re-running
  `fetch` skips months already stored). `build_long30.py` checked against brute force (600 events,
  0 mismatches). `m18_longshot.py rules|cnn_run|cnn_eval` written; a pool-only dry run (not the
  study) showed the 2025-10-10 cascade dominates crash rules (SOLV +1,877% in a day; 55% of R3's
  2025 trades), so every result now also reports P&L without 2025-10-10/11.
  Next, in order: publish -> `MOONSHOT_UNIVERSE=all` build_events -> build_long30 -> m18 rules ->
  build_seq2 (+ fwd) -> m18 cnn_run -> cnn_eval -> journal section b2.
- 11:55Z Archive fetched and imported: 18,846 months, 0 problems; delisted.txt 191, outside_pool.txt
  239 in the shared data folder (klines now 6.9 GB). B2 pipeline started on "all", log
  `research/volume_spike/cache_all/logs/b2_pipeline.log` (events -> long30 -> rules -> seq2).
- 10:12Z (clock check) Times in the entries from 08:20Z on were estimates and ran ahead of the real
  clock; the order of the entries is right. From here on, times come from `date -u`.
- 10:16Z B2 rules on "all" (11.05M event rows, 4.54M spikes; results/universe_all/m18_rules.csv).
  % of $10k a year, 2022-24 / 2025-26 (ex 2025-10-10/11): R0 random +42 / -38 at best (TP20 30d);
  R1 every spike and R4 breakout lose in 2025-26 at every exit; R2 capitulation positive with
  1-7 day limits; R3 deep crash 2025-26 is the cascade (ex-cascade +2..+10). Pre-registered pick
  (best 2022-24 P&L among R1-R4): R2, TP +20%, 3 days: $ per year 2020..2026 +845 / +8,651 /
  -1,082 / +2,967 / +7,584 / +2,479 (ex-cascade +879) / +1,337; 1,000-2,300 trades a year, mean
  +4% per trade, hit rate 17-39%, worst trade -90% (LUNA), worst realised drawdown -$2,375.
  +30-50% targets are reached on only 3-20% of trades; the profit is the bounce.
  `Data(mmap=True)` added to m12 (seq2 on "all" is 13.9 GB, 1.85M rows).
- 10:51Z B2 CNN (`m18_longshot.py cnn_run`, 33 min; results/universe_all/m18_cnn.csv): base rate
  of "+30% within 7 days" 15.5% of liquid spikes; validation top-2% hit 22-42%. Pre-registered CNN
  pick (best 2022-24): top 5%, TP +50%, 30 days: $ per year 2022..2026 +1,099 / +361 / +1,404 /
  +5,850 (ex-cascade -1,445) / -420; hits +50% on 25-46% of trades; worst trades -87..-95%.
  66-77% of its top 1-2% picks in 2025 were bought on 2025-10-10. Verdict: the CNN does not beat
  the capitulation rule for long shots. R2 pick book: median 1 open position, mean 9.4, none open
  42% of hours, 200 (cap) on crash days; mean hold 2.4 days.
- 11:00Z B2 done; journal section "Study 2" written (chart, target x hold table, CNN table,
  conclusions). Waiting for the user.
- 12:42Z B3 step 1 (`m19_quiet.py step1`, results/universe_all/m19_step1.csv): 4.53M snapshots,
  582 coins (3.34M used: liquid, entry before 2026-10-01). FAILS the pre-registered bar. 2022-24,
  all vs Q1 decile 1 vs Q7 decile 1: +20% in 3d 9.1 / 4.4 / 5.3%; -20% in 3d 5.3 / 3.4 / 4.2%;
  +20% before -20% 45 / 38 / 40%; mean 7d return +0.2 / -1.7 / -1.0%. Quiet stays quiet and drifts
  down. Q7 decile 1 beats the average in 2021, 2025, 2026 (2025-26: 45 vs 35% up-first), is worse
  in 2023; cascade-free numbers barely differ.
- 12:42Z B3 step 2 (`m19_quiet.py rules`, m19_rules.csv): every quiet rule loses in 2022-24 at
  every exit (mean trade -0.5..-1.5%) except QD (quiet week + early buying) TP 50-100% / 30d,
  +3.6..+3.8% of $10k a year; that pre-registered pick loses 28% a year in 2025-26. QB/QC with 7d
  limits make +7..+10% a year in 2025-26 but lose in 2022-24. Verdict: quietness alone, or with
  simple early-buying filters, does not predict up spikes. Going on to step 3 (CNN) as planned:
  the user's idea is that a specific pattern within quiet coins comes before a spike.
- 12:44Z B3 step 3 started: `build_quiet_seq.py` 1,147,641 quiet snapshots (Q7 d1-3),
  qseq_h 3.9 GB + qseq_m 4.0 GB; `m20_quiet_cnn.py run` then `gbm` (comparison) then `eval`, log
  `research/volume_spike/cache_all/logs/m20.log`. Outputs cache_all/wf/quiet_cnn.npz, quiet_gbm.npz,
  results/universe_all/m20_quiet_cls.csv and m20_quiet_book.csv. Judgment call: the 1h branch is
  normalised by the 7-day window's own median (no longer history needed).
- 13:09Z B3 step 4 (`m21_fillcheck.py longshot`, results/universe_all/m21_fillcheck_longshot_r2_tp20_3d.csv):
  judgment call: the quiet rules failed, so the fill check runs first on the only surviving
  strategy, the B2 pick (R2 capitulation, +20% or 3 days); 10,368 of 10,378 symbol-days of 1m
  data downloaded (cache_all/m1). 3,061 take-profit exits: 2,857 (93%) filled (a 1m close at or
  above the target, in the same minute as the first touch), 204 touched only (fall back to the
  3-day exit: +3.3% instead of +19.7%). Filled share by year 89-97%, 2025 83%. Cost ,678 over
  2020-26 (2022-26 total +$13,285 -> +$12,046, ~+$2,540 a year). Entry: 1m open equals the 5m
  open on 3,057 of 3,061 trades.
- 13:16Z B3 step 3 (m20_quiet_cls.csv, m20_quiet_book.csv): base rate of +20% in 3 days among
  quiet snapshots 6.7%. CNN AUC 0.64/0.62/0.57/0.59/0.56 (2022..26), top-2% hit 19/8/5/11/11% vs
  base 7.4/3.3/4.4/6.3/5.6%. GBM (scalars only) AUC 0.59-0.64, top-2% 9-15%. Pre-registered CNN
  pick (best 2022-24: top 5%, TP 20%, 7 days): $ +259 / +16 / +59 / -375 / -111; placebo
  +587 / +85 / +80 / -1,822 / -513 (CNN beats placebo only in 2025-26; both lose). GBM best
  (TP 20%, 30d) +1,324 / +384 / +577 / +298 / -136, its placebo better in 2022-24. Top picks are
  young (median age 245 vs 850 days), deep below the 30-day high (-36..-45% vs -23%), least quiet
  of the quiet (rv1_rel ~1.0 vs 0.8); taker share barely differs. Verdict: round 1 fails.
- 13:19Z B3 round 2 (`m22_quiet_spike.py`, m22_quiet_spike.csv, m22_quiet_spike_context.csv):
  4.23M spikes with a quiet measure. Best 2022-24 / same setting 2025-26, $ a year: S1 -739 / -3,840;
  S2 (2019 rule in a quiet spell) -390 / -3,762; S3 (first spike in 3 days, ~60 trades a year)
  -35 / -9; S4 -735 / -3,873; busy + up candle +1,507 / -6,019; every up-candle spike +1,514 /
  -6,035. Mean return 1h / 4h after entry, 2022-24: quiet -0.12 / -0.18%, busy -0.25 / -0.41%;
  median spike candle out of quiet +0.4%. Secondary (B2 CNN on quiet-context spikes): 4-47 trades
  a year, +368 / -4 / +66 / +345 (ex-cascade +43) / -51 -> noise. Verdict: study 3 fails in every
  form tested.
- 13:19Z B3 done. Journal: "Where we stand after three studies" summary at the top, Study 3
  section (charts, tables, conclusions), journal rows. Waiting for the user.
- 14:42Z B4 data: `fetch_1m.py` 25,711 symbol-months, 0 problems, 1,111,441,287 1m candles,
  25 GB in the shared data folder (klines/spot/interval=1m), 24 min. Analysis `m23_minute.py`
  speed -> quiet -> report running, log research/volume_spike/cache_all/logs/m23.log. Tested on
  AAVE: the "m5" entry equals the 5m backtest entry exactly.
- 14:49Z B4 part A (m23_speed.csv): 49,409 R2 signals. The 1m trigger fires in the spike
  candle's first minute (mean minute 0.22; 99.7% before the last minute); its entry is a median
  0.23% above the 5m entry; mean 1h return +2.5% vs +3.5%. $ a year 2020..2026, 5m entry
  (replayed on 1m candles) +833 / +8,650 / -1,082 / +2,967 / +7,584 / +2,479 / +1,336 (matches B2);
  1m trigger -76 / +5,203 / -2,561 / -758 / +1,220 / +341 / +295. FAILS: in a crash, reacting
  faster buys before the bottom; the end of the 5m crash candle is the better moment.
