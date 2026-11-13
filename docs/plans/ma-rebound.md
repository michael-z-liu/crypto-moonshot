# MA rebound: can a CNN pick the good pullbacks to major moving averages?

- **Status:** building <!-- building | blocked | review | done -->
- **Branch:** `worktree-bridge-cse_015bssAVNSvc6d3Ch9Adf7UE` (created by the desktop app; kept)
- **PR:** <!-- link, once opened -->
- **Updated:** 2026-11-13T03:40Z
- **Last report to the user:** 2026-11-13T03:40Z (M1 accepted)

## Goal

The user's thesis: on spot, a price that comes down to a major simple moving average (hourly up
to monthly) often rebounds, and a CNN reading the candles before the touch can pick the good
setups, raising the odds. This project is a research study that tests the premise without a model
first (does buying every touch pay?), then tests whether a walk-forward CNN adds to it, with the
protocol below fixed before any result. Entry is a limit buy resting slightly above the MA.

## Out of scope

- Limit orders in the backtest engine, a strategy in `src/`, paper or live trading (follow-ups).
- EMAs and other MA types, futures data (funding, open interest), short trades.
- Changing the candle store or its format. The study only reads `klines/`.
- Any rerun of the volume-spike studies or their caches.

## Ground rules

- Code in `research/ma_rebound/`, run from there with `uv run --group research python ...`.
  Small result tables and notes are committed under `research/ma_rebound/results/`.
- Market data: `/Users/mike/code/crypto-moonshot/data` (`MOONSHOT_DATA_DIR`), read only: 5m and 1m
  candles for the "all" universe (582 USDT pairs incl. delisted, minus pegged assets and tokenised
  stocks, as `research/volume_spike/common.py` defines it).
- Cache: `research/ma_rebound/cache` is a gitignored symlink to
  `/Users/mike/code/crypto-moonshot/data/cache/ma_rebound/` so a later session or worktree can
  reuse it. Never write into `data/cache/volume_spike*`.
- Data end for everything: 2026-11-02 00:00 UTC (exclusive), as in the earlier studies.
- GPU (MPS): one training at a time. Check for another session's training before starting one
  (`ps`), and wait rather than share.
- **Holdout:** 2025-01-01 .. 2026-11-02 (by fill time). Until the M6 pick is committed and pushed,
  no outcome of a 2025-26 candidate (label, return, win rate, book) may be printed, plotted,
  committed or read by anyone, master or worker. Counts of 2025-26 candidates are allowed. Scripts
  compute 2025-26 results only into `research/ma_rebound/results/sealed/` (gitignored).

## Protocol (fixed 2026-11-13T03:05Z, before any result)

**Moving averages.** 15 SMAs: 50/100/200 on 1h, 4h and 1d; 20/50/100/200 on 1w; 10/20 on 1M. Bars
are aggregated from the 5m candles on UTC boundaries (4h from 00:00, weeks from Monday, calendar
months). Daily candles before 2020 are downloaded for coins listed earlier, only to complete the
1d/1w/1M averages. The level `L` at any time is the SMA of the last n *completed* bars' closes, so
it is constant within a bar and known in advance (no look-ahead). Placebo MAs for M2: lengths 37,
73, 137 and 163 on the same timeframes.

**Scale per timeframe.** Hold limit `T`: 1h 8 hours, 4h 32 hours, 1d 8 days, 1w 21 days, 1M 28
days. Volatility unit `u = atr_d x sqrt(T in days)`, where `atr_d` is the ATR(14) of completed
daily bars as a fraction of the last daily close. All distances below are fractions of price.

**Universe.** Every "all" coin while its mean daily quote volume over the previous 30 complete UTC
days is at least $1,000,000. Majors (BTC, ETH and the 20 highest 30-day volumes at the time) are
reported as a separate cut.

**Candidate (the setup).** Per coin and MA, on 5m closes:
- *Above*: each of the last k completed bars of the MA's timeframe had its low above the SMA level
  in force during that bar (k = 12 for 1h and 4h, 10 for 1d, 4 for 1w, 2 for 1M), and no 5m low of
  the forming bar has reached `L`.
- *Armed* by a 5m close at or above `L(1 + 0.45u)` while *above* holds.
- *Trigger*: the first 5m close at or below `L(1 + 0.3u)` while armed and *above*. The trigger
  disarms. Everything a model sees ends at this close.

**Entry.** At the trigger close a limit buy is placed at `L(1 + b)`, re-priced whenever a bar
completes and `L` changes, valid for `T/2`. Buffers `b`: 0, 0.05u (primary), 0.15u. It fills in the
first 1m candle (5m where 1m is missing) whose low is at or below the limit, at the limit price. If
the price is already at or below the limit when the order is placed, it fills at the next 1m open
as a taker. Unfilled orders expire: no trade. Comparison entry (*touch-market*): a market buy at
the open after the first 5m candle whose low reaches `L`, scored at that candle's close.

**Exits** (through the production `resolve_exit`, stop first when a candle touches both):
E1 time exit at `T`; E2 time exit at `T/2`; E3 take profit `+1u` from entry, stop at `L(1 - 0.5u)`
(L at fill), time `T`; E4 take profit `+0.5u`, stop at `L(1 - 0.25u)`, time `T`; E5 stop at
`L(1 - 0.5u)` only, time `T`; E6 time exit at `2T` (at most 28 days). Base rule: E1.

**Costs.** Fee 0.1% per side. Limit fills (entry, take profit): no slippage. Market fills (taker
entry, time exit, stop): 0.1% slippage and latency. Stress: fills only when the price trades 5 bp
through the limit, plus 0.1% extra cost per side.

**Book.** $10,000, 10 equal slots of realised equity, one position per coin, a slot is taken at
the fill (no free slot or cash: the fill is skipped), ordered by fill time then score. One
continuous book from 2022-01-01 to the data end; yearly numbers by fill year.

**Model.** Walk-forward: fold Y (2022..2026) trains on candidates whose trigger and label window
end before Y-1 July, early-stops on Y-1 July..December (label windows ending before Y), and
scores year Y. A score is its percentile among the fold's training scores. Base rule: trade
candidates with percentile >= 0.80. Every trained variant has at least 3 seeds and is scored on
the seed mean and seed by seed. Benchmarks: *every candidate* (no model) and a gradient-boosted
model on hand-made features with the same folds.

**Scorecard** (per ledger row, on 2022-24): per year trades, fill rate, mean net bp per trade,
win rate, book return and max drawdown; the stress run; a placebo (scores shuffled within UTC
month x timeframe, 20 draws); a day-bootstrap 95% CI of the mean bp; the mean without the 5 best
days; *fragile* if a year turns negative without its single best trade; results per timeframe.

**Qualifies** (2022-24): at least 100 trades each year; mean net bp per trade > 0 each year; book
return > 0 each year; mean bp > 0 under the stress; mean bp above the placebo's 95th percentile.
**Rank**: mean of the three yearly book returns, the worst year breaking ties.
**Beats the baseline**: qualifies, is not fragile, and its rank value exceeds the base CNN's by
more than the base CNN's seed spread (max - min of the per-seed rank values). The CNN "adds
value" only if it also beats *every candidate* and the gradient-boosted benchmark by that margin.

**Ledger.** Every variant tried is a row in `results/ledger.csv`, keyed by name and config hash.

**M5 screen.** One change at a time from the base. Training (3 seeds each): inputs (coin only;
+ BTC; + BTC and market breadth; + distances to the other MAs); window length (short, long);
target (return at `T` in units of u; path quantiles; bracket win probability); one pooled model
against one model per timeframe group (1h+4h, 1d, 1w+1M); touch-market candidates. Rules on the
base scores: thresholds 0.5, 0.7, 0.9, 0.95; buffers 0 and 0.15u; exits E2..E6; MA subsets
(intraday only, daily only, weekly+monthly only, the 200s only, majors only); 5 and 20 slots.

**M6 pick and reveal.** The top two training variants that beat the baseline (fewer if fewer do;
the base CNN if none) are retrained with 5 seeds, plus their combination when compatible. On each:
the base rule, and the two best-ranked M5 rules that beat the base rule (if none, the base rule
alone). Pick: the top-ranked qualified, non-fragile run. It is committed and pushed, then 2025-26
is opened once for every ledger row. **Confirmed** means, for the pick on 2025-26: the
day-bootstrap 95% CI of the mean net bp per trade is above 0, the book return is positive, and it
beats *every candidate* under the same rule. Also reported: the Spearman correlation between
2022-24 rank and 2025-26 across rows, and a deflated Sharpe ratio over all ledger rows.

## Reporting to the user

The user asked to be told that things are moving, at most every few hours unless needed.
- Send a `PushNotification` when a milestone is accepted, on any stop condition, and when the PR
  is ready.
- Otherwise, on a heartbeat, if `Last report to the user` is more than 4 hours old, send one
  one-line status (milestone in progress, what is running) and update that field.
- Never report a 2025-26 outcome before the reveal.

## Milestones

### M0: Setup, protocol and pre-2020 daily candles (done)

- **Delivers:** this plan with the fixed protocol; `research/ma_rebound/` with `common.py` (paths,
  universe, constants of the protocol), the cache link, `.gitignore` entries; `fetch_daily.py`
  that downloads pre-2020 1d candles (public Binance market-data host, no key) into the cache.
- **Done when:** the protocol is committed and pushed before any result; pre-2020 daily candles
  exist for every "all" coin listed before 2020; the check command passes.
- **Worker:** master
- **Commits:** 7f0d575 (plan and protocol), ff64627

### M1: MA grid, candidates, limit fills and labels (done)

- **Delivers:** `build_bars.py` (1h/4h/1d/1w/1M bars and the 15 + placebo SMA levels, `atr_d`,
  liquidity), `build_candidates.py` (candidates per the protocol, both entry families),
  `build_fills.py` (limit fills per buffer on 1m candles), `build_labels.py` (exits E1..E6 through
  `resolve_exit`, forward path), and `check_events.py` (look-ahead and invariant checks).
- **Done when:** truncating the data at a random time leaves every earlier candidate, level and
  fill decision unchanged; recomputing 20 random candidates (2024 or earlier) by a direct loop
  over the raw candles gives the same level, trigger, fill and exits; counts per MA and year are
  in `results/M1.md` (no 2025-26 outcomes); the check command passes.
- **Worker:** opus
- **Commits:** 6b1c81e, acd8552, de853cf, 8bc16f1

### M2: Baseline: does buying every MA touch pay? (in progress)

- **Delivers:** `m2_baseline.py` and `results/M2.md`: per MA, buffer and exit on 2020-24, the fill
  rate, mean net bp, win rate and day-bootstrap CI of every candidate; the same for placebo MAs
  and for matched random entries (same coin and month, same distance fallen, no MA nearby); the
  majors cut; a go/no-go note on the premise.
- **Done when:** the tables exist for 2020-24 only, with CIs; the note states plainly whether
  major MAs beat placebo MAs and random entries; the check command passes.
- **Worker:** sonnet
- **Commits:**

### M3: Book simulator and scorecard for limit entries (todo)

- **Delivers:** `book.py` (the protocol's book on top of `resolve_exit`, `Costs` and the
  production metrics) and `scorecard.py` (the scorecard, stress, placebo, bootstrap, fragile flag,
  ledger, sealing), with the *every candidate* rows in the ledger.
- **Done when:** fed market-entry trades, the book equals the production engine's trade list and
  equity to within rounding; 2025-26 goes only to the sealed folder; the *every candidate* rows
  (base rule and E1..E6) are in the ledger; the check command passes.
- **Worker:** opus
- **Commits:**

### M4: Walk-forward CNN, base model and gradient-boosted benchmark (todo)

- **Delivers:** `build_inputs.py` (model inputs per candidate, memory-mapped), `train.py` (the
  walk-forward harness with the M5 options), the base CNN with 3 seeds, `gbm.py` (benchmark).
- **Done when:** scores for 2022-26 exist for the base CNN (seed mean and per seed) and the GBM;
  their base-rule rows and the seed spread are in the ledger; the folds are purged as the protocol
  says (checked by a test on fold boundaries); the check command passes.
- **Worker:** opus
- **Commits:**

### M5: Variant screen (todo)

- **Delivers:** every M5 training variant and rule as a ledger row, `results/M5.md` with the
  verdicts.
- **Done when:** each listed variant has a row judged by the fixed rule on 2022-24, with per-seed
  rows for trained variants; the check command passes.
- **Worker:** sonnet
- **Commits:**

### M6: Finalists, pick and reveal (todo)

- **Delivers:** finalists with 5 seeds, `results/M6_pick.md` committed and pushed before the
  reveal, then `results/M6.md` with 2025-26 for every row, the bootstrap CI, the rank correlation
  and the deflated Sharpe ratio.
- **Done when:** the pick's commit precedes the reveal commit on the pushed branch; the reveal
  table covers every ledger row; the check command passes.
- **Worker:** opus
- **Commits:**

### M7: Report and hand-off (todo)

- **Delivers:** `docs/research/ma-rebound.md`, an updated `research/README.md`, the PR.
- **Done when:** the report states what held up and what did not, with caveats; the check command
  passes; CI is green.
- **Worker:** master
- **Commits:**

## Decisions

- 2026-11-13: research study only, all 582 coins with a liquidity floor, holds scaled to the MA,
  thorough study (user's answers during scoping).
- 2026-11-13: the user accepts a multi-day run; wants resume after usage limits and a progress
  note at most every few hours (see Reporting).
- 2026-11-13: SMA levels use completed bars only. A level that moves with the live price cannot be
  a resting limit price, and the difference is small for n >= 50.
- 2026-11-13: variants are ranked by yearly book return, not bp per trade, because holds range
  from 8 hours to 28 days and capital tied up for weeks must count.
- 2026-11-13: 2025-26 is used as the holdout. It was opened by the volume-spike studies, but for
  different events; no MA-touch outcome on it has been seen.

- 2026-11-13 (M0): 82 coins were listed before 2020; all their pre-2020 daily candles came from
  the public API (it still serves delisted pairs' daily history), none from the archive.

- 2026-11-13 (M1): accepted. 2.62M candidates (study: 683k limit, 419k touch-market), 7.97M fill
  rows, 23.9M labels; the full build takes about 5 minutes. All 76 checks pass on the master's
  own run; BTC's daily SMA200 candidates match known history (2021-05-19, 2023-03-10, 2023-08-17,
  2024-07-04). The 19 settled definitions are in `research/ma_rebound/results/M1.md`.
- 2026-11-13 (M1): a candidate's `year` is its trigger year; the holdout is by fill time, so
  every later step filters on `fill_t` / `entry_t` < 2025-01-01.
- 2026-11-13 (M1): one MA can trigger several times before it is touched, and those orders fill
  in the same candle at the same price. That is one resting order refreshed, so per-trade
  statistics count fills of the same coin, MA, buffer and fill candle once (the earliest
  trigger). The book's one-position-per-coin rule already does this.
- 2026-11-13 (M1): the 1M placebo lengths 137 and 163 cannot exist in the data and 1M 73 has 14
  candidates; weekly and monthly MAs are thin (1w 200: 104 candidates in 2020-24).
- 2026-11-13 (M1): the worker's four commits carried a model name in the co-author trailer; the
  messages were rewritten before the first push (no pushed history changed).

## Blocked on

## Follow-ups

- Limit-order entries in `SimulatedBroker` and the backtest engine, and the rule as a strategy in
  `src/` with a paper test, if the pick is confirmed.
- EMAs, VWAP and other levels; futures data as model inputs.
