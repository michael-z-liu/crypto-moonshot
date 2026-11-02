# Architecture

> **This is a living document, not a fixed spec.** It records the design as first agreed on
> 2026-10-31, before any of it was built. Needs will change and the design should change with
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
                        Strategy → Position sizing → Risk checks
                                           │
                                ┌──────────┴──────────┐
                                ▼                     ▼
                         SimulatedBroker        BinanceBroker
                       (backtest, paper)     (testnet, mainnet)
```

The backtester and the live trader share everything from the strategy to the broker interface.
They differ only in where candles come from and which broker fills orders.

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
  has missing candles. Units of every series share one concurrency limit (default 8) and are
  interleaved across series, so all of them make progress.
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

- The strategy says only what it wants to hold. It doesn't size positions or place orders.
- `compute` is vectorised and may only look backwards. Backtests run it once over the full history;
  the live engine runs it over a rolling window. A parity test checks both produce the same signals.

### Position sizing

Turns "hold this symbol" into a USDT amount, for example equal weights with a cap on open
positions. With several symbols, this is where capital is split between them.

### Risk checks

- Maximum order size, maximum total exposure, maximum open positions.
- Daily loss limit that halts trading.
- No trading on a symbol whose latest candle is more than about two intervals old.
- Kill switch: `moonshot live stop`.

### Brokers

- One interface: place order, cancel order, balances, open orders.
- `SimulatedBroker` fills at the next candle's open, with fees (0.1% by default) and slippage.
- `BinanceBroker` rounds price and quantity to each symbol's trading rules (tick size, step size,
  minimum notional) from a cached `exchangeInfo`, and sets a client order ID on every order so a
  retry can't place it twice.

### Backtest engine

- Lines up candles across symbols by open time. Signals use the close of candle *t*; orders fill at
  the open of candle *t+1*.
- Tracks USDT cash and holdings.
- Outputs an equity curve, a trade list, and metrics (return, CAGR, max drawdown, Sharpe, win rate,
  time in market) under `results/`.

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
| `backtest`                     | Stored history      | `SimulatedBroker`                     |
| `paper`                        | Live mainnet stream | `SimulatedBroker`                     |
| `testnet` (default for orders) | Live mainnet stream | Binance spot testnet                  |
| `live`                         | Live mainnet stream | Binance mainnet; requires `--mainnet` |

## Package layout and CLI

```
src/crypto_moonshot/
├── cli.py              # Typer app; data_cli.py holds the `data` commands
├── cli_output.py       # summary and status tables (rich in a terminal, plain text otherwise)
├── progress.py         # sync progress renderers: bars and lines
├── config.py           # API keys from env/.env; per-run settings (symbols, params, limits) from a file
├── models.py           # Interval, Candle, Order, Fill, Position, Target
├── data/               # binance.py (REST), store.py, sync.py, importer.py, status.py, stream.py (WebSocket)
├── strategy/           # base.py + algorithms
├── portfolio.py        # position sizing
├── risk.py
├── broker/             # base.py, simulated.py, binance.py
├── backtest/           # engine.py, metrics.py
└── live/               # engine.py, state.py (SQLite)
```

Commands:

- `moonshot data sync`: download missing candles
- `moonshot data status`: show coverage and gaps
- `moonshot data import <folder>`: reuse candles downloaded into another data folder
- `moonshot backtest <strategy>`
- `moonshot live <strategy> --mode paper|testnet|live`

Dependencies so far: polars, httpx, pydantic-settings, rich. Expected later: duckdb, websockets.

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
- **Only sync and import write candles.** Few writers, all using atomic renames, mean the live
  trader and a backtest can read while data is being updated. Running two writers on the same
  data folder at once is not supported.
- **SQLite for live state.** Orders and fills are many small writes that must survive a crash,
  which suits a transactional database rather than Parquet.
- **Mainnet candles even when orders go to testnet.** The spot testnet is its own thin market and
  its prices drift from the real one, so signals computed from testnet candles would be misleading.

## Open questions

- **Cross-symbol signals.** The strategy receives all symbols together, so signals that use other
  symbols (for example BTC's trend as a filter) are possible, but no requirement for them exists yet.
- **Position sizing scheme.** Equal weight is a placeholder until a strategy needs something else.

## Build order

1. Models and config
2. Data layer: REST client, store, sync, `data sync` and `data status`
3. Strategy interface, plus a simple moving-average crossover strategy to test the plumbing
4. Backtest engine and metrics
5. Live engine in paper mode
6. Binance broker on testnet, with risk checks
7. Mainnet with small position sizes
