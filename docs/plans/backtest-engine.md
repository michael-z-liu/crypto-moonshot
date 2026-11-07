# Backtest engine, validated against the research

- **Status:** building <!-- building | blocked | review | done -->
- **Branch:** `feat/backtest-engine`
- **PR:** <!-- link, once opened -->
- **Updated:** 2026-11-07T02:35Z

## Goal

Build the production backtest engine from `docs/architecture.md` (backtest engine,
`SimulatedBroker`, position sizing, risk checks, metrics, `moonshot backtest`) so research results
can be reproduced and validated by production code. Proven by golden tests that reproduce PR #8's
F2 rule, cnn-replication's B2 capitulation pick (5m and 1m fills) and PR #8's walk-forward base CNN
on the shared data folder, with tolerances fixed before running and every difference explained.

## Out of scope

- Live and paper engines, `BinanceBroker`, live-only risk checks (stale-candle guard, kill switch).
- Training models: CNN strategies read precomputed scores.
- Merging or changing the research branches; walk-forward rule selection.
- Short selling; tick or aggTrades data.

## Requirements (from the request)

1. Timing: signal on the close of candle t, entry at the open of t+1; configurable entry delay.
2. Orders: market entry, take-profit limit, stop, time exit, trailing stop. A candle touching both
   stop and target fills the stop first unless 1m candles resolve the order; a gap through a level
   fills at the open.
3. Multi-resolution: signals from 5m candles, fills on 1m candles when available.
4. Costs: fee + slippage per side (default 0.1% + 0.05%), optional latency cost.
5. Portfolio: fixed stake ($50 of $10,000) and equal-slot compounding (equity / slots); max open
   positions; one position per coin; priority order for signals in the same candle.
6. A series that ends closes the position at its last traded price.
7. Signals from rules or from precomputed model scores (walk-forward CNN per-row files).
8. Outputs: trade list, mark-to-market equity curve, yearly and overall metrics; total return
   includes the first day's P&L.
9. Golden tests with stated tolerances (M4).
10. Update `docs/architecture.md` where the design changes. No orders, no network, no API keys.

Inputs (read only): research branches `origin/claude/high-volume-trading-strategy-0c5d1e` (PR #8)
and `origin/claude/cnn-replication`, read with `git show origin/<branch>:<file>`; data in
`MOONSHOT_DATA_DIR=/Users/mike/code/crypto-moonshot/data` (5m and 1m candles, `top200.txt`,
`delisted.txt`, `outside_pool.txt`; research caches in `cache/volume_spike/`). Never write there;
own caches stay inside the worktree (gitignored).

## Milestones

### M1: Orders and SimulatedBroker (done)

- **Delivers:** order and position models; `SimulatedBroker` that fills market entries, take-profit
  limits, stops, trailing stops and time exits against candles; stop-first when one candle touches
  both, unless finer (1m) candles resolve the order; gaps through a level fill at the open;
  take-profit fill rule `touch` (default) or `close` (a candle close at or above the target);
  fee + slippage per side and optional latency cost; a series that ends closes at its last traded
  price.
- **Done when:** unit tests on synthetic candles cover each rule; full check passes.
- **Worker:** opus
- **Commits:** d487c8c, 5ccedc2, 129a23a, 36a942d

### M2: Engine, sizing, risk checks and outputs (done)

- **Delivers:** backtest engine (signal at close of t, entry at open of t+1 plus configurable
  delay); sizing: fixed stake and equal-slot compounding; risk checks: max open positions, one
  position per coin, same-candle priority, max exposure, daily loss limit; outputs: trade list,
  mark-to-market equity curve, yearly and overall metrics, total return including the first day.
- **Done when:** synthetic end-to-end tests cover each item; full check passes.
- **Worker:** opus
- **Commits:** 5d44b5c, ccb5c2b, 2144ca9, 6a4846d, fe26230, fd65027, 1628124, 2b4ae5b

### M3: Data and strategies (done)

- **Delivers:** candle loading for a universe (5m; 1m only for open-position windows; gaps and
  ended series); strategy interface returning signals plus an exit plan; F2 and capitulation (R2)
  rules with features ported from `build_events.py`; model-score signals from a Parquet file, and
  a converter from the research cache's `wf/*.npz` files.
- **Done when:** unit tests, incl. feature parity with the research formulas on fixtures; full
  check passes.
- **Worker:** opus (raised from sonnet: exact parity with the research features is subtle)
- **Commits:** 6b03721, 35dc163, fc13908, 4946023, af25e03, db24156, 401494a, 1d82296

### M4: Golden validation (in progress)

- **Delivers:** golden tests marked `golden` (skipped by default and when data is missing; run with
  `uv run pytest -m golden`) and `docs/backtest-validation.md` explaining every difference.
  - F2 (pool, 5 slots, 1h hold), net bp per trade 2020..2026: +222 / +319 / +223 / +287 / +284 /
    -44 / +187. Tolerance ±15 bp per year, trade counts within 5%.
  - Capitulation pick (all coins, $50, +20% or 3 days, 200 slots), $ per year 2020..2026: +845 /
    +8,651 / -1,082 / +2,967 / +7,584 / +2,479 / +1,337. Tolerance ±5% or ±$150 per year,
    whichever is larger.
  - Same with 1m fills and the `close` take-profit rule: total falls by $1,678 ± 25%.
  - Base CNN from walk-forward scores (5 slots, top 2%, 1h), bp per trade 2022..2026: +334 / +170 /
    +57 / +67 / +53. Tolerance ±20 bp per year.
- **Done when:** golden tests pass within tolerance, or a difference is explained by a documented,
  deliberate behaviour difference (not tuned away); full check passes.
- **Worker:** opus
- **Commits:**

### M5: CLI and docs (todo)

- **Delivers:** `moonshot backtest f2|capitulation|scores` with the main options, writing results
  under `results/`; `docs/architecture.md` updated (order-level exits alongside targets, etc.).
- **Done when:** CLI tests; docs updated; full check passes.
- **Worker:** sonnet
- **Commits:**

## Decisions

- 2026-11-07 Take-profit fill rule configurable, default `touch`; golden 1m test uses `close`
  (that is how cnn-replication's m21 fill check measured the -$1,678). (user)
- 2026-11-07 Golden tests opt-in via a `golden` pytest marker. (user)
- 2026-11-07 Include a base-CNN golden test from precomputed scores. (user)
- 2026-11-07 Include a minimal `moonshot backtest` CLI. (user)
- 2026-11-07 Costs are per fill and multiplicative (fee on each side's notional, slippage on
  price, latency on market orders only); research subtracted 0.3% additively. Explained in M4.
- 2026-11-07 (M1) Order, Fill, ExitPlan, Costs live in `broker/base.py`; `resolve_exit` is a pure
  vectorised resolver over a mixed 5m/1m fill path. Exits inside a candle happen at its close
  time, gap fills at its open time; time exit at entry + max_hold at the last close before it.
- 2026-11-07 (M1) Fixed and trailing stop together: the higher level applies (research picked the
  fixed stop on ties). No golden setup uses both.
- 2026-11-07 (M1) `close` take-profit rule: a candle opening at or above the target fills at the
  open, as with `touch` (m21 required a close); may show in the 1m golden comparison.

- 2026-11-07 (M2) The engine places every order through `SimulatedBroker` (~62k positions in
  3.3 s, in memory). Signals ordered by entry time, priority, symbol; exits at or before an entry
  settle first. Equal-slot stake = realised equity / slots (as PR #8).
- 2026-11-07 (M2) Entries are skipped when cash is short of the stake; the research books never
  checked cash. Kept (realistic); M4 measures its effect on the golden numbers.
- 2026-11-07 (M2) Daily loss limit uses realised P&L; equity curve marks open positions at the last
  close at or before each mark; total return is from initial capital.
- 2026-11-07 (M3) Research dropped events whose next 4h crossed a data gap or the series' last 4h
  (look-ahead). The production rules do not; M4 explains the difference.
- 2026-11-07 (M3) Strategies produce signals + an exit plan (`strategy/`); features are the
  research's polars expressions verbatim. All 8,383 research F2 events and 14,053 R2 events are
  reproduced with identical features. Signal frames cached in the worktree's `.cache/`.
- 2026-11-07 (M3) No flat padding after halts/delistings (research "all" mode padded up to 300
  candles, which also fed features); fills are equivalent, features after long halts are not.
- 2026-11-07 (M3) Smoke run, F2 bp 2020..26: +224/+318/+211/+287/+284/-42/+187 (ref +222/+319/
  +223/+287/+284/-44/+187); R2 $: +785/+8,624/-1,079/+2,958/+7,561/+2,472/+1,334; base CNN bp
  +333/+172/+50/+64/+52 (ref +334/+170/+57/+67/+53).

## Blocked on

## Follow-ups

- Cap the engine's exit window at the source's data end, so a plan without `max_hold` cannot raise
  for positions still open at the data end (M5 if small).
