# Capitulation long/short: search for a much more profitable version

- **Status:** building
- **Branch:** `feat/capitulation-long-short`
- **PR:**
- **Started:** 2026-11-27T13:46Z. Keep searching until at least **2026-11-28T09:46Z** (20 h)
  before hand-off; budget about 25 h (five usage windows).
- **Updated:** 2026-11-27T14:22Z

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

### M0: Harness, baseline and scorecard (done)

- **Delivers:** `research/capitulation_ls/` with a fast event-level simulator (long and short,
  TP/SL/time exits on 5m paths, slots, leverage), the scorecard (bp per trade, yearly return,
  drawdown, within-day fidelity: rank correlation of forecast and outcome among one day's
  candidates), the ledger, and the baseline (path_all and day5 with a 2h hold) reproduced.
- **Done when:** the baseline matches the #10 study's 2025-26 bp within noise; the check command
  passes.
- **Worker:** master

### M1: Binance futures data (in progress)

- **Delivers:** USD-M perp 5m klines, funding rates, premium index and metrics (open interest,
  long/short ratios) for every coin with a perp, from data.binance.vision; aligned features per
  event; synced spot candles to the latest day.
- **Done when:** coverage table per coin and year; features join the event table without
  look-ahead.
- **Worker:** master (or sonnet for the downloader)

### M2: Short side (in progress)

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
- ~13:52Z E01-E03: event landscape. Longer holds change the picture: one entry per event, shorts
  after +40% 4h pumps average +5% at 24h and +11% at 72h (median +12.5% at 24h), positive every
  year 2021-26; longs on the top 0.5% day5 scores average +11% at 24h. Simulator reproduces the
  #10 study's day5 1h result (+98 bp over 2025-26).
- ~13:56Z M1: perp funding (2.85M rows, 919 symbols) and 1h perp klines (16.5M rows) downloaded;
  spot 5m synced to 2026-11-27 (unseen forward window from 2026-11-02).
- ~13:58Z E05/E07: pump-short grid and walk-forward (per-year $10k books, perp-gated, funding):
  WF pick +86/+134/+157/+236/+551% in 2022-26, +327 bp/trade, worst DD -55%. Hindsight best
  (thr 0.4, hold 72h, 5 slots) >= +66% every year at +10%/trade, DD -29%.
- ~14:00Z E08: GBM walk-forward ranking of pumps: IC 0.13-0.31 every year; top 20% book +820
  bp/trade, positive every year, DD -53%. Crash longs with GBM: unstable, mostly losing.
- ~14:02Z E09: perp vs spot over the same hours: median gap 0.2% but delisting squeezes (ALPACA
  +584% perp vs +28% spot) and frozen perps break the spot proxy. Perp gate now requires traded
  volume in the last two hours; shorts will be re-simulated on 5m perp paths (download running).
- ~14:05Z E10: short entry timing: immediate entry is best; each delay costs edge (5m late
  -25 bp, 1h late -200 bp) but stays profitable even 4h late. Rule r4>=0.4 on the active-perp
  gate: +1020 bp/trade, +62..+400% a year 2022-26, DD -29%.
- ~14:06Z nn_seq/E11: tabular MLP IC 0.15-0.28 (= GBM); GBM+MLP blend IC 0.17-0.34, within-day
  IC +0.30 (the models do separate coins on the same day). As a veto on the rule set they add
  almost nothing (+1020 -> +1050 bp).
- ~14:07Z E12/E13: new listings fall: short 1-3 days after spot listing on the perp, hold 7-30
  days: +10..+15% a trade, positive most years (few trades 2021-23). Second short component.
- ~14:15Z E14-E16 (correction): on perp paths the pump short is weaker (r4>=0.4, 72h: +979 bp
  spot -> +656 bp perp), and the earlier books floored losses at -100% while keeping positions
  that went +100% against them. 6.5% of pump shorts see a +100% move within 72h (ALPACA's
  delisting squeeze: -516%). Realistic tails: isolated 1x (liquidated at +100%) or cross margin
  (losses beyond the stake). Then r4>=0.4, 10 slots, 72h: +260 to +310 bp a trade, worst year
  about -6%..+7%; cross margin with 5 slots loses a whole year to one squeeze. Stops at 50% hurt.
- ~14:20Z E17: squeezes are not predictable (AUC 0.42-0.63 by year, metrics add nothing).
  Realistic pump-short reference: iso1x, 10 slots, 72h: +313 bp, +7..+26% a year, DD -33%.
  Transformer ranker (3 seeds) IC 0.33/0.18/0.30/0.24 for 2022-25: better than GBM/MLP.

## Idea backlog

- Pump shorts on perp paths (5m, to 7 days); leverage 1-3x with liquidation on the perp path.
- Open interest / long-short ratios at entry as squeeze predictors (metrics download running).
- Transformer and TCN rankers (training), seed ensembles, a ranker trained only on the rule set.
- Listing shorts: combine with pump shorts in one book; listing pumps inside the first week.
- Long side: flush-day longs on perps (lower fees), CNN top 0.5% held 24h, a BTC-crash gate.
- Combined long/short book; capital split; Kelly sizing; drawdown control.
- Forward read-out on 2026-11-02..27 (spot synced; perp prices and funding via daily archive/API).
- Crazy ones: pyramiding into squeezes, basket shorts of the day's pumps, shorting extreme
  positive funding, weekday/hour effects, pump-then-dump chains (second pump within days).

## Blocked on

## Follow-ups
