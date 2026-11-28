# Capitulation long/short search

Scripts of the long/short capitulation search (plan and run log:
`docs/plans/capitulation-long-short.md`; report: `docs/research/capitulation-long-short.md`).
They read the volume-spike caches of universe "all" through `research/volume_spike/cache_all`
(a link into the shared data folder) and write their own data to
`$MOONSHOT_DATA_DIR/cache/capitulation_ls/`. Nothing here places orders.

```bash
uv sync --group research
cd research/capitulation_ls
uv run --group research python e05_short_grid.py   # any script: run from this folder
```

## Building blocks

| File | What it does |
| --- | --- |
| `common_ls.py` | loaders: seq2 meta, 8h (fwd2) and 72h (fwd72) forward paths, walk-forward CNN scores, `first_per_event` |
| `sim.py` | event simulator: long/short, take profit / stop / trailing / time exits on 5m-then-hourly paths (spot) or 5m-then-hourly perp paths to 7 days, liquidation, funding, delayed entry; slot portfolio with a liquidity cap; `simulate_yearly` = a fresh $10k book per year |
| `wf.py` | walk-forward selection of rule configurations (earlier years only) |
| `dsr.py` | deflated Sharpe ratio from a book's daily returns |
| `fetch_futures.py` | data.binance.vision: USD-M funding, 1h perp klines, 5m perp klines for pumped coins, futures metrics on event days |
| `futures_ls.py` | perp symbol mapping (1000PEPE -> PEPE), perp activity gate (traded volume in the last two hours), funding paid over a hold |
| `build_feats.py` | feature table of big-move events (sequence, 7-90 day context, funding, perp flag, targets) |
| `build_metric_feats.py` | open interest / long-short ratio features at entry |
| `build_perp_paths.py` | perp price paths of every event with an active perp |
| `build_tp20_offset.py` | target offset so the i10 harness learns the TP20/72h outcome (`tp20day`) |
| `nn_seq.py` | walk-forward transformer / TCN / MLP rankers on the 24h sequence + features |
| `e25_combined.py` | sleeves (L, P, P', N, N2/M, C) and the shared-slot book with a worst-concurrent-adverse check |
| `forward_signals.py`, `save_day_model.py`, `lmc_forward.py` | forward scoring of new spikes, the saved day5 2026 fold, and the pre-registered LMC paper read-out |

## Experiments (`eNN_*.py`)

| Script | Question |
| --- | --- |
| e01, e02 | how large are moves after each event type, long and short, 1h-72h? |
| e03, e04 | simulator check against the #10 study; audit of the pump-short result |
| e05, e07 | pump-short grid; walk-forward selection (spot proxy and perp-gated with funding) |
| e06 | which entry features predict reversal (pumps) or bounce (crashes)? |
| e08, e11 | walk-forward GBM / model evaluation on big-move events |
| e09, e15 | perp vs spot prices after pumps; where the gap comes from (delisting squeezes) |
| e10 | short entry timing |
| e12, e13, e18, e27, e28, e36, e44 | listing shorts (spot listings, all perps, weak-listing filter, triggers, young-coin decay) |
| e14, e16, e17, e19 | pump shorts on perp paths; honest tails; squeeze prediction; realistic-target ranking |
| e20-e24, e29-e33, e38 | capitulation longs: holds, take profits, robustness, score sets, ensembles, majors with leverage, bounce fades, trailing stops, pyramiding, adaptive take profit |
| e26, e30 | rankers trained on the realistic long outcome |
| e34, e37 | regime filters (BTC trend) for pump shorts |
| e35, e39, e43 | stress tests and deflated Sharpe ratio of the combined books |
| e40 | does the CNN tell coins' volatility apart within a day? |
| e42 | idiosyncratic crash shorts |
| e45 | settings frozen on 2022-24, read on 2025-26 |
| e46 | share of random configurations that meet both targets |
| e47 | extreme funding as a signal |
| e48 | block bootstrap of the books |
| e49 | late perp listings |
| e50 | network architectures with day-weighted training |
| e51 | large-cap weekly reversal |
| d01, d02 | numerical diagnostics (`diag.py`): sleeve scorecards and calibration; failed variants (`results/diag/`) |
| w1-w12 | "why it failed" investigations, one note each in `results/why/` (w1 pump shorts, w2 leverage, w3 long variants, w4 listings, w5 regime/other, w6 cascade wait, w7 early slot cap, w8 funding-filter limits, w9 large-cap reversal, w10 crash-short exits, w11 squeeze longs, w12 crowding and longs) |
| x1-x14 | explorers, one note each in `results/explore/` (x1 perp premium, x2 quiet breakouts, x3 lag catch-up, x4 contagion, x5 listing-short tail, x6 pump-short tail, x7 second pumps and listing months 2-3, x8 calendar and baskets, x9 crash-short entry features, x10 book-level sizing, x11 limit entries for longs, x12 perp order-book depth: `x12_fetch.py` data, `x12_book_depth.py` pre-declared tests, `x12_followup.py` post hoc; x13 L on perps vs spot, x14 LMC+ without the best flush days) |
| h1 | holdout: the long sleeve on 2017-08..2019-12 spot data (`results/explore/holdout_2018.md`) |
| p1 | perp-only coins (USD-M perps without spot): L as a perp long and C, day5 walk-forward folds 2022-26 (`results/explore/perp_only.md`) |
