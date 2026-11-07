# Architecture

> **This is a living document, not a fixed spec.** It records the design as first agreed on
> 2026-10-31 and has been revised as the data layer, strategies, simulated broker and backtest
> engine were built (live trading is still design only). Needs will change and the design should change with
> them. If a requirement shifts, or building something shows part of this design is wrong or
> awkward, change the design and update this document in the same PR. Each decision below gives
> its reason; when the reason stops holding, the decision is open. The binding rules for this
> repo are in `CLAUDE.md`, not here.

## Goals

1. Download Binance candles (klines) once and cache them locally, fetching only what is missing.
2. Backtest trading algorithms against the cached history.
3. Trade autonomously in real time: watch live candles, evaluate the algorithm, and enter or exit
   when its conditions are met.

## Current scope

| Decision        | Choice                                                         |
| --------------- | -------------------------------------------------------------- |
| Exchange        | Binance spot                                                   |
| Direction       | Long only: each symbol is either held or not, priced in USDT   |
| Symbols         | Several, traded as one portfolio                               |
| Data            | Candles only (OHLCV)                                           |
| Resolution      | 5m to start; any Binance interval is supported                 |
| History         | From 2020-01-01, or a symbol's listing date if later           |
| Decision timing | On closed candles only                                         |

## Overview

```
Binance REST ──► data.sync ──► data.store (Parquet)
                                  │                    │ recent history at startup
                                  ▼                    ▼
                          backtest.engine         live.engine ◄── Binance WebSocket
                                  │                    │          (all symbols, one connection)
                                  └────────┬───────────┘
                                           ▼
                 Strategy ──► Position sizing ──► Risk checks
                 (entry signals + ExitPlan,         │
                  or targets: what to hold)         │
                                           ┌────────┴────────┐
                                           ▼                 ▼
                                    SimulatedBroker     BinanceBroker
                                  (backtest, paper)   (testnet, mainnet)
                                  entry + bracket exit (take profit, stop,
                                  trailing stop, time exit) as orders
```

The backtester and the live trader share everything from the strategy to the broker interface.
They differ only in where candles come from and which broker fills orders. Exits are orders at
the broker (a bracket attached to the entry), not later decisions of the strategy; the
alternative style, where the strategy states what it wants to hold and the engine trades the
difference, is described under Strategy and not built.

## Components

### Data store

- Parquet files, queried with DuckDB or Polars.
- Layout: `data/klines/spot/interval=5m/symbol=BTCUSDT/2026-10.parquet`, one file per symbol per
  month. The Hive-style folder names let DuckDB expose symbol and interval as columns.
- Columns: open time (UTC), open, high, low, close, volume, quote volume, trade count, taker buy
  base and quote volume. A candle is identified by symbol, interval and open time.
- Closed candles only. Prices are floats for analysis and become exact decimals only when an order
  is placed.
- Size: about 720k rows per symbol for 5m candles since 2020.
- The data root is configurable (`MOONSHOT_DATA_DIR` or `--data-dir`), defaults to `./data/`, and
  is gitignored.
- `docs/data-format.md` is the exact contract for readers: columns and dtypes, invariants, the
  per-series `_meta.json` (listing date and known exchange gaps), and the schema version stored in
  each file's Parquet metadata. Changing the format means bumping the schema version.

### Data sync

- Works out which candles are missing, then fetches only those ranges from `GET /api/v3/klines`,
  1,000 candles per request (about 720 requests per symbol for the first download). Requests go
  to `data-api.binance.vision`, Binance's public market-data host, so no API key is needed.
- `data/binance.py` keeps one client per run and paces it to stay under half of Binance's
  6,000-per-minute IP weight limit, using the `X-MBX-USED-WEIGHT-1M` header. A 429 pauses every
  request for the full `Retry-After`; after three 429s in one run (`--max-rate-limit-hits`) the
  run aborts. A 418 (IP ban) aborts at once. 5xx and network errors are retried with backoff.
- `data/sync.py` plans each (symbol, interval) series, then runs one unit of work per month that
  has missing candles. Units of every series share one concurrency limit (default 8) and run
  series by series in the order given, so each series finishes (and is usable) as early as
  possible; only the tail of one series overlaps the start of the next.
- A month file is rewritten only when it gained candles: normally just the current month. Writes
  go to a temporary file that is then renamed into place. A month file that fails the format
  checks counts as missing and is refetched and overwritten.
- Binance has some real gaps from exchange outages. Sync records them in `_meta.json` so it
  doesn't keep refetching, but never a hole that ends within two candles (or five minutes) of now:
  that may just be publishing latency, so the next run tries again.
- One failing series doesn't stop the others. A rate-limit abort or IP ban stops the whole run,
  and the result says so. Progress is reported as events to a sink; sync itself never prints.
- Sync and import are the only code that writes candles.

### Data CLI and progress

- `moonshot data sync` drives sync and renders its events: live rich bars (one row per series
  plus a total) when stderr is a terminal, otherwise plain timestamped lines with a snapshot
  every few seconds, so an agent can follow a run. `--json` prints a machine-readable summary on
  stdout. Exit codes: 0 ok, 1 a series failed, 2 usage error, 3 aborted on rate limits, 130
  interrupted.
- `moonshot data status` reports coverage per series from the files alone: range, missing
  candles, recorded gaps and invalid files.
- `moonshot data import <folder>` copies candles from another data root so they need not be
  downloaded again. Each month file must pass the same structural checks sync applies to its own
  files (layout, schema version, columns, UTC, alignment, duplicates, closed candles); anything
  else is skipped and left for the next sync to download. There is deliberately no conversion
  from older schema versions: downloading again is cheap. Overlapping months are merged with the
  existing candles winning, and the source is never modified.

### Strategy

A strategy decides *when to enter*; it never sizes positions or places orders. There are two
styles; only the first is built.

**Entry signals plus an exit plan** (`strategy/`). The strategy produces entry signals from
closed candles and an `ExitPlan` (take profit, stop, trailing stop, time exit) that the broker
attaches to every entry as a bracket, so exits are orders at the broker, not later decisions:

- **Signals** (`strategy/base.py`, `Signals`): a polars frame with `symbol`, `open_time` (the
  signal candle *t*; entry is at the open of *t+1*), `priority` (lower first when signals compete
  for a slot in one candle) and any extra columns, carried into the trade list.
- **Rule strategies** (`SignalRule`) compute in three vectorised, backward-looking steps, so a
  universe can be computed in parallel: `candidates(candles)` turns one symbol's whole series
  into the rows it might trade, with their features; `context(series)` builds cross-symbol
  context by open time from `context_symbols` (e.g. BTC); `select(candidates)` takes every
  symbol's candidates with the context joined on and returns the signals.
- `strategy/compute.py` runs a rule over a universe: one process per symbol at a time (spawned,
  8 by default), reading each whole series from the store up to an optional data end
  (`until`), then caches the signal frame under a cache folder (`.cache/signals/`, gitignored),
  keyed by the rule's parameters, the symbols, the data end and a signature of the month files
  read. The pool universe (152 coins) takes ~6 s, `all` (582) ~20 s on 10 cores.
- **Features** (`strategy/features.py`) are ported from the volume-spike research with the same
  formulas and windows, on a gap-free grid where a missing candle is a row of nulls; a parity
  test checks them against a direct loop over the research formulas.
- **Rules**: `strategy/capitulation.py`, one parameterised rule (volume spike after a 4-hour
  fall while BTC spikes) with the research's F2 and R2 picks as presets.
- **Model scores** (`strategy/scores.py`): `ScoreStrategy` reads precomputed per-candle scores
  (format in `docs/data-format.md`, "Score files") and enters where `score >= threshold`, with
  `priority = -score`. `convert_wf_npz` turns the research's walk-forward CNN outputs into
  that format. Training models is out of scope.
- **Universes** (`data/universe.py`): `pool` (the data root's `top200.txt` minus pegged assets
  and tokenised stocks) and `all` (plus `delisted.txt` and `outside_pool.txt`).

**Targets** (not built yet): the strategy says only which symbols it wants to hold (LONG or
FLAT) at the newest candle, and the engine trades the difference:

```python
class Strategy(Protocol):
    symbols: list[str]
    interval: Interval
    warmup: int  # candles of history needed before the first decision

    # Add indicator and signal columns.
    def compute(self, candles: pl.DataFrame) -> pl.DataFrame: ...

    # Return LONG or FLAT for each symbol at the newest candle.
    def decide(self, latest: pl.DataFrame, positions: Positions) -> dict[str, Target]: ...
```

In both styles `compute` is vectorised and may only look backwards. Backtests run it once over
the full history; the live engine will run it over a rolling window, and a parity test will
check both produce the same signals.

### Position sizing

Turns "hold this symbol" into a USDT amount (`portfolio.py`). The engines keep a `Book`: cash,
the cost basis of each open position (entry fee included) and the realised P&L of the current
UTC day. Sizing and risk decisions use *realised* equity, cash plus open cost basis: open
positions are not marked to market for them, as in the research simulators, so a decision never
depends on prices after the entries already made.

- `FixedStake(usdt)`: the same amount per entry; no compounding, P&L simply adds up.
- `EqualSlots(slots)`: realised equity / slots per entry (compounding).

The stake is the USDT spent including the entry fee. An entry whose stake exceeds the cash is
skipped.

### Risk checks

Pre-trade checks in `risk.py`, shared by the backtest and live engines. Each is a small object
with a `reason` and `allows(book, symbol, stake)`; an engine runs a list of them and skips an
entry at the first that refuses (`first_rejection`). They read only the book, never prices.

- `MaxPositions`: open positions below the limit.
- `OnePerSymbol`: no second position in a symbol (the brokers hold one position per symbol, so
  the backtest always applies it).
- `MaxExposure`: open cost basis, the new stake included, at most a fraction of realised equity.
- `DailyLossLimit`: no new entries for the rest of the UTC day once the day's *realised* P&L has
  fallen below `-limit x` the realised equity at the start of the day. A position that is down
  but still open does not count until it closes; once hit, the halt lasts until midnight UTC
  even if later exits recover the loss.

Still to come with the live engine: maximum order size, no trading on a symbol whose latest
candle is more than about two intervals old, and a kill switch (`moonshot live stop`).

### Brokers

- One interface (`broker/base.py`): place order, cancel order, balances, open orders. The
  order models live there too: `Order`, `Fill`, `ExitPlan` (take profit, stop, trailing stop,
  time exit) and `Costs` (fee, slippage, latency per side).
- A buy can carry an `ExitPlan`; the broker turns it into a bracket sell whose first exit to
  fill cancels the rest (OCO). Exits are orders at the broker, not decisions of the strategy.
- `SimulatedBroker` (`broker/simulated.py`) fills a market buy at the open of the first candle at
  or after the order time, with fees (0.1% by default), slippage (0.05%) and an optional latency
  cost. Its core is `resolve_exit`, a pure function that simulates a position's whole exit over
  its candle window in one vectorised numpy pass (stop first when a candle touches both stop and
  target, gaps fill at the open, 1m candles replace a coarser candle they fully cover, a series
  that ends closes at its last close). The module docstring is the exact rule set. Candles are
  passed as numpy arrays (`broker/candles.py`), not frames, because a backtest resolves tens of
  thousands of positions.
- `BinanceBroker` rounds price and quantity to each symbol's trading rules (tick size, step size,
  minimum notional) from a cached `exchangeInfo`, and sets a client order ID on every order so a
  retry can't place it twice.

### Backtest engine

`backtest/engine.py` is event-driven over *signals*, not a bar-by-bar loop over every candle:
long-only strategies with bracket exits only need candles around each position, and the 1m
history (about 25 GB) cannot be loaded whole.

- **Signals in:** a polars frame with `symbol`, `open_time` (the signal candle *t*, whose close
  the signal uses) and `priority` (lower first), plus any extra columns, which are carried into
  the trade list. One `ExitPlan` per run, in `BacktestConfig`.
- **Candles in:** a `CandleSource` (`backtest/source.py`) returns a symbol's `Candles` window at an
  interval for a time span, so a Parquet-backed source can read lazily, one window per position
  (at the signal interval, and at 1m with `use_fine`). A window that reaches the end of an ended
  series must say so (`Candles.ended`). `InMemoryCandleSource` serves tests.
  `ParquetCandleSource` reads the store lazily: only the month files a window touches, kept as
  numpy arrays in an LRU cache bounded by memory (2 GB by default). An optional `until` hides
  candles at or after a data end, so a run can reproduce a fixed data range. A window that
  reaches past a series' last candle is `ended` when that candle closes before the data end (a
  delisting); a series still trading at the data end is not. With no month file at an interval
  (often 1m) the window is `None` and fills use the signal candles. About 50k R2 positions over
  582 coins with 1m fills run in ~30 s.
- **Timing:** entry is a market buy at the open of candle *t+1* (plus `entry_delay` candles). If
  that exact candle is missing (a hole, or the series ended) the signal is skipped.
- **Event order:** signals are processed by entry time, then priority, then symbol. Before each
  entry, every position whose exit time is at or before the entry time is closed, so its cash and
  slot are free (as in the research books).
- **Orders through the broker:** each entry is placed with a `SimulatedBroker` and carries the
  exit plan; the broker resolves the bracket exit over the position's window with
  `resolve_exit` and applies it when the engine settles to that time. The window starts at
  `max_hold` (or `BacktestConfig.window` without one) and doubles until the exit is known. The
  engine reads the resolved exit (`SimulatedBroker.pending_exit`) only to size windows and to mark
  the position, never to decide. About 60k positions from 250k signals run in a few seconds.
- **Outputs** (`BacktestResult`): the trade list; a mark-to-market equity curve at every
  `mark_every` boundary (midnight UTC by default) from the start of the first entry's day to the
  end of the last exit's day, valuing open positions at the last close at or before each point
  (a hole or an ended series carries the last close); counts of skipped signals by reason; and
  metrics (`backtest/metrics.py`). Trade metrics (trades, mean bp, win rate, P&L, worst trade,
  average hold) are grouped by entry year, as the research reports; equity metrics (total return
  from the initial capital, so the first day's P&L counts, CAGR, max drawdown, Sharpe on daily
  returns annualised with 365, time in market, calendar-year returns) come from the curve.
- **Data end:** the engine asks the source for its data end (`CandleSource.data_end`) and never
  loads a position's window past it; a window that reaches it counts as ended. A position still
  open there (an exit plan without `max_hold`, or a time exit beyond the data) closes at the last
  close with reason `series_end`, the same reason as a delisting; its exit time is the data end.
- **CLI** (`backtest_cli.py`): `moonshot backtest f2|capitulation|scores` run the engine with the
  presets' defaults (F2: pool, 5 equal slots, 1 hour; capitulation: all coins, $50 stakes, +20%
  or 3 days, signals before `R2_LAST_ENTRY`; scores: top 2%, 5 slots, 1 hour) and options for
  the data end, signal range, sizing (`--slots` or `--stake`), exits (`--hold`, `--take-profit`,
  `--stop`, `--trailing`, `--tp-fill`), costs, 1m fills (`--fine`) and entry delay. A run writes
  `trades.parquet` and `.csv`, `equity.csv`, `yearly.csv` and `metrics.json` (metrics, skipped
  signals, the full configuration and the data end used) into `results/<strategy>-<UTC
  timestamp>/` (gitignored) and prints the overall metrics and the yearly table; `--json` prints
  the `metrics.json` document instead. `moonshot backtest convert-scores NPZ META OUT` turns a
  research walk-forward `.npz` into a score file. Signal caches stay in `.cache/` of the working
  directory, never in the data folder.
- **Validation:** golden tests (`tests/test_golden.py`, opt-in with `uv run pytest -m golden`)
  reproduce the research's F2, capitulation (5m and 1m fills) and walk-forward CNN results on the
  real data; [`backtest-validation.md`](backtest-validation.md) gives the numbers and explains
  every difference (cash checks, the research's look-ahead filters, the cost model, re-running
  the book on 1m fills).

### Live engine

- On startup: run sync to bring stored candles up to date, load each symbol's warm-up window, and
  check balances and open orders against the Binance account.
- One combined WebSocket connection carries every symbol's kline stream. Updates for a candle that
  is still forming are ignored; each closed candle is appended to its symbol's window. When every
  symbol has closed the same timestamp, or after a short timeout, the strategy is evaluated once
  for all of them.
- Binance drops connections every 24 hours. After any reconnect, missed candles are fetched over
  REST before anything is evaluated.
- Orders, fills and positions are saved in a SQLite file so a restart can resume safely.

## Run modes

| Mode                           | Candles             | Order fills                           |
| ------------------------------ | ------------------- | ------------------------------------- |
| `backtest` (built)             | Stored history      | `SimulatedBroker`                     |
| `paper`                        | Live mainnet stream | `SimulatedBroker`                     |
| `testnet` (default for orders) | Live mainnet stream | Binance spot testnet                  |
| `live`                         | Live mainnet stream | Binance mainnet; requires `--mainnet` |

## Package layout and CLI

```
src/crypto_moonshot/
├── cli.py              # Typer app; data_cli.py holds the `data` commands, backtest_cli.py `backtest`
├── cli_output.py       # summary, status and backtest tables (rich in a terminal, plain text otherwise)
├── progress.py         # sync progress renderers: bars and lines
├── config.py           # API keys from env/.env; per-run settings (symbols, params, limits) from a file
├── models.py           # Interval, kline schema, Target
├── data/               # binance.py (REST), store.py, sync.py, importer.py, status.py, universe.py, stream.py (WebSocket)
├── strategy/           # base.py, features.py, compute.py, capitulation.py, scores.py
├── portfolio.py        # the book and position sizing
├── risk.py
├── broker/             # base.py (orders, ExitPlan, Costs), candles.py, simulated.py (built); binance.py
├── backtest/           # engine.py, metrics.py, source.py (candle windows; Parquet-backed and in-memory)
└── live/               # engine.py, state.py (SQLite)
```

Commands:

- `moonshot data sync`: download missing candles
- `moonshot data status`: show coverage and gaps
- `moonshot data import <folder>`: reuse candles downloaded into another data folder
- `moonshot backtest f2|capitulation|scores`: backtest a strategy and write `results/`
- `moonshot backtest convert-scores NPZ META OUT`: make a score file from a research CNN output
- `moonshot live <strategy> --mode paper|testnet|live`

Dependencies so far: polars, numpy (the broker and engine work on arrays), httpx, pydantic-settings,
rich, typer. Expected later: duckdb, websockets.

## Key decisions and why

- **Parquet files rather than a database server.** The data is small (a few hundred MB for ten
  symbols at 5m) and backtests read whole columns, which columnar files do well. DuckDB gives SQL
  over the files with nothing to run. SQLite is slower for full-column scans, a single DuckDB
  database file allows only one writing process at a time, and Postgres or TimescaleDB would be a
  server to maintain for data that fits in memory. Revisit if the data grows by orders of magnitude
  (for example tick data) or several processes need to write candles at once.
- **Decide on closed candles only.** A closed candle is final, so the live engine sees exactly the
  data a backtest sees, and 5m history is enough to test against. If the algorithm ever needs to
  react while a candle is still forming, the live engine becomes tick-driven and backtests need
  finer-grained data.
- **One strategy for backtest and live.** Strategies that do well in backtests and lose live often
  calculate their signal differently on the two paths. Sharing the code removes that cause.
- **Exits are orders at the broker.** A strategy returns entries plus an `ExitPlan`; take profit,
  stop, trailing stop and time exit are bracket orders. Live, the exchange can then protect a
  position while the bot is down, and the backtest resolves exits with the same order semantics
  (stop first on a candle touching both, gaps fill at the open) instead of the strategy
  re-deciding each bar.
- **The backtest walks signals, not bars.** With bracket exits only the candles around each
  position matter, so the engine loads one window per position (1m only where fills need it)
  and 25 GB of 1m history never has to fit in memory. A strategy that must act on every bar
  (the targets style) would need a bar loop; that is the cost of adding it.
- **Only sync and import write candles.** Few writers, all using atomic renames, mean the live
  trader and a backtest can read while data is being updated. Running two writers on the same
  data folder at once is not supported.
- **SQLite for live state.** Orders and fills are many small writes that must survive a crash,
  which suits a transactional database rather than Parquet.
- **Mainnet candles even when orders go to testnet.** The spot testnet is its own thin market and
  its prices drift from the real one, so signals computed from testnet candles would be misleading.

## Open questions

- **Cross-symbol signals in the live engine.** Rule strategies take cross-symbol context
  (`SignalRule.context`, e.g. BTC's volume for the capitulation rules); the live engine will need
  the context symbols' windows too.
- **Position sizing scheme.** Equal weight is a placeholder until a strategy needs something else.

## Build order

1. Models and config
2. Data layer: REST client, store, sync, `data sync` and `data status`
3. Strategy interface and the capitulation rules (done; the moving-average crossover was
   dropped in favour of the research strategies)
4. Backtest engine, metrics, `moonshot backtest` and golden validation (done)
5. Live engine in paper mode
6. Binance broker on testnet, with risk checks
7. Mainnet with small position sizes
