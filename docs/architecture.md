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
- The data root is configurable, defaults to `./data/`, and is gitignored.

### Data sync

- Works out which candles are missing, then fetches only those ranges from `GET /api/v3/klines`,
  1,000 candles per request (about 720 requests per symbol for the first download).
- Rewrites only the current month's file; finished months never change. Writes go to a temporary
  file that is then renamed into place.
- Binance has some real gaps from exchange outages. Sync records them so it doesn't keep refetching.
- Sync is the only code that writes candles.

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
├── cli.py              # Typer commands
├── config.py           # API keys from env/.env; per-run settings (symbols, params, limits) from a file
├── models.py           # Interval, Candle, Order, Fill, Position, Target
├── data/               # binance.py (REST), store.py, sync.py, stream.py (WebSocket)
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
- `moonshot backtest <strategy>`
- `moonshot live <strategy> --mode paper|testnet|live`

Expected dependencies: polars, duckdb, httpx, websockets, pydantic-settings.

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
- **Only sync writes candles.** A single writer means the live trader and a backtest can run at the
  same time without contending for files.
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
