# Capitulation long/short: search for a much more profitable version

- **Status:** building
- **Branch:** `feat/capitulation-long-short`
- **PR:**
- **Started:** 2026-11-27T13:46Z. Keep searching until at least **2026-11-28T09:46Z** (20 h)
  before hand-off; budget about 25 h (five usage windows).
- **Updated:** 2026-11-27T14:05Z

## Goal

Find what has to change in the capitulation strategy (buy liquid coins on 5m volume spikes during
market-wide flushes, ranked by a walk-forward CNN) to reach an **average trade of +10% or more**
and **+50% or more a year**, for a $10k-50k account (no market impact). Extend it to **both
directions**: long after capitulation dumps, short (USD-M perpetuals) after blow-off rallies. Make
the forecaster tell coins apart instead of predicting the average. This is a wide, adaptive search:
architectures, signals (Binance data only), training windows, exits (take profit / stop grids),
sizing and leverage. Keep trying new ideas, however unlikely, for at least 20 hours.

## Out of scope

- Live, testnet or paper trading code; futures/margin account mechanics; anything on mainnet.
- Data from outside Binance (data.binance.vision and the public Binance APIs only).

## Ground rules (fixed before any result)

- **Universe:** "all" (582 USDT pairs incl. delisted), as in the robustness and #10 studies.
- **Costs:** spot 0.1% fee + 0.05% slippage + 0.05% latency per side (0.4% round trip). Perps:
  0.05% taker + 0.05% slippage + 0.05% latency per side (0.3% round trip) plus funding paid or
  received over the hold. Stress at 2x cost and one candle late.
- **Leverage:** the research decides (user, 2026-11-27). Every result is reported unlevered, and
  the final candidates also at 1x-5x on perps with the liquidation price checked on the path
  (a position whose adverse move reaches 1/leverage minus maintenance margin is liquidated).
- **Periods:** development on the walk-forward years 2021-2024. 2025-26 (to 2026-11-02) has been
  seen by three studies: it is used as a gate each idea family must pass, read once per family and
  logged. Data after 2026-11-02 is unseen: it is read once, for the final pick only.
- **Ledger:** every trial is one row in `research/capitulation_ls/results/ledger.csv` (id, idea,
  2021-24 per-year bp, 2025-26 bp, trades, return, drawdown). The final pick gets a deflated
  Sharpe ratio over all trials.
- **Disk:** at most 20 GB of new data (user, 2026-11-27). Check free space before each download.

## Milestones

### M0: Harness, baseline and scorecard (in progress)

- **Delivers:** `research/capitulation_ls/` with a fast event-level simulator (long and short,
  TP/SL/time exits on 5m paths, slots, leverage), the scorecard (bp per trade, yearly return,
  drawdown, within-day fidelity: rank correlation of forecast and outcome among one day's
  candidates), the ledger, and the baseline (path_all and day5 with a 2h hold) reproduced.
- **Done when:** the baseline matches the #10 study's 2025-26 bp within noise; the check command
  passes.
- **Worker:** master

### M1: Binance futures data (todo)

- **Delivers:** USD-M perp 5m klines, funding rates, premium index and metrics (open interest,
  long/short ratios) for every coin with a perp, from data.binance.vision; aligned features per
  event; synced spot candles to the latest day.
- **Done when:** coverage table per coin and year; features join the event table without
  look-ahead.
- **Worker:** master (or sonnet for the downloader)

### M2: Short side (todo)

- **Delivers:** pump/blow-off event study, rules and models for perp shorts, with funding.
- **Done when:** short-side scorecard for rules and the best model, ledgered.
- **Worker:** master

### M3: Per-coin forecast fidelity (todo)

- **Delivers:** models that separate coins: distribution heads, in-event ranking losses,
  event-only training, coin embeddings, transformers / TCN / GRU / state-space, GBM, stacks;
  training windows (expanding, rolling, recency-weighted).
- **Done when:** fidelity and portfolio scorecards for each, ledgered.
- **Worker:** master

### M4: Big-trade hunting (todo)

- **Delivers:** longer holds (hours to days), TP/SL grids, forecast-driven exits, rarer and
  deeper events, concentration, Kelly sizing, leverage, combined long/short book.
- **Done when:** the best books per target (10% per trade, 50% a year) are found and ledgered.
- **Worker:** master

### M5: Open idea loop (todo)

- **Delivers:** whatever the earlier milestones suggest, until the 20-hour mark: liquidation
  cascades (open-interest drops), extreme funding, listings, order book depth, regime switching,
  time of day, spot-perp basis, and more.
- **Worker:** master

### M6: Final pick, forward read-out, report, PR (todo)

- **Delivers:** a pick written down and committed before the unseen data is read; the read-out;
  `docs/research/capitulation-long-short.md`; PR.
- **Worker:** master

## Decisions

- 2026-11-27: Leverage is decided by the research; shorts are modelled on USD-M perps; up to 20 GB
  of new data (user answers at scoping).
- 2026-11-27: The user asked for a wide search (newer architectures, other techniques beyond CNNs,
  other signals, training windows, TP/SL grids) and to adapt to findings: widen the scope where a
  discovery points, keep going and keep trying. Milestones M2-M5 are a menu, not a fixed order.
- 2026-11-27: Work lives in a new `research/capitulation_ls/` that reads the existing volume-spike
  caches (`research/volume_spike/cache_all`, symlinked to the shared data folder) instead of
  rebuilding them.

## Run log

- 2026-11-27T14:05Z: plan written; caches found (seq2 1.85M liquid spikes 2020-01..2026-11-01,
  8h and 72h forward paths, i10 walk-forward scores in the AUhKx worktree).

## Blocked on

## Follow-ups
