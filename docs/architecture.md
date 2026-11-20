# Architecture

> **This is a living document, not a fixed spec.** It records the design as first agreed on
> 2026-10-31 and has been revised as the data layer, strategies, simulated broker and backtest
> engine were built, and again on 2026-11-20 when the trading platform was designed (P1 builds its
> paper mode). Needs will change and the design should change with them. If a requirement shifts, or building something shows part of this design is wrong or
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
Binance REST ──► data.sync ──► data.store (Parquet) ──────── warm-up ────────┐
                                  │                                           ▼
                                  │                Binance WebSocket ──► live.feed ──► ticks
                                  ▼                (closed candles)            │   (one per closed
                          backtest.engine                                      │    candle time)
                     (signals in, one window          in-process plugins ◄─────┤
                      per position)                                            │
                                  │                external providers  ◄─ API ─┘ candles, ticks
                                  │                (own process/machine, ── API ─► signal batches
                                  │                 e.g. a torch CNN)          │
                                  │                                            ▼
                                  │                live.engine: per-strategy books ► sizing ►
                                  │                risk checks ► account checks ► orders
                                  │                (state.sqlite for recovery)  │
                                  ▼                                             ▼
                          SimulatedBroker                    PaperBroker (replay, paper)
                          (whole exit window                 BinanceBroker (testnet, mainnet)
                           at once)
                     every entry carries a bracket exit (take profit, stop, trailing stop,
                     time exit), resolved with the same rules everywhere
```

The backtester and the trading platform share the strategy code, the order models, position
sizing, risk checks and the exit rules. They differ in where candles come from, how a strategy's
signals reach the engine and which broker fills orders.

Strategies are *plugins*: built outside the platform and loaded from a module or file path named
in a run config, or run as *external providers*, separate processes that get candles from the
trading daemon and post signals back over HTTP. Both speak one signal contract, so a strategy is
written once and can be backtested, replayed, paper traded and run in or out of process without
changing the platform. The daemon itself does no heavy computation: anything that needs a model
or heavy libraries runs as an external provider.

Exits are orders at the broker (a bracket attached to the entry), not later decisions of the
strategy; a strategy may also close a position it holds with an exit signal. The alternative
style, where the strategy states what it wants to hold and the engine trades the difference, is
not built.

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

**Plugins for trading** (see Trading platform, Strategy plugins). For the live side a strategy
is a plugin implementing `live.plugin.Strategy`: whenever a candle closes it gets each symbol's
recent candles and returns the signals for that candle. A `SignalRule` such as the capitulation
presets becomes one through the `RuleStrategy` adapter, which runs `candidates`, `context` and
`select` over rolling windows; a parity test checks it gives the same signals as
`compute_signals` over the whole history. The "targets" style (the strategy says only what it
wants to hold and the engine trades the difference) is not built; enter and exit signals can
express it.

`candidates` and `on_candles` are vectorised and may only look backwards, so a rolling window at
least the rule's warm-up long gives the same features as the whole series.

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

The trading platform runs these per strategy book and adds account-wide checks on top: maximum
order size, maximum total exposure, no entries on a symbol whose latest candle is more than
`stale_after` intervals old, and the pause and flatten controls (see Trading platform, Engine).

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
- `PaperBroker` (`broker/paper.py`) fills orders against candles as they arrive, calling the
  same `resolve_exit` on the candles seen so far (see Trading platform, Paper broker).
- `BinanceBroker` (P3) rounds price and quantity to each symbol's trading rules (tick size, step
  size, minimum notional) from a cached `exchangeInfo`, and sets a deterministic client order ID
  on every order so a retry can't place it twice. Brackets become OCO order lists (take-profit
  limit plus stop-loss, trailing stops via `trailingDelta`); the engine places time exits.

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
  the `metrics.json` document instead. `moonshot backtest plugin SOURCE` runs any strategy
  plugin (its batch `backtest_signals`, or `on_candles` over rolling windows) and `moonshot
  backtest signals FILE` a signal file produced elsewhere, with the same sizing, exit, cost and
  output options. `moonshot backtest convert-scores NPZ META OUT` turns a
  research walk-forward `.npz` into a score file. Signal caches stay in `.cache/` of the working
  directory, never in the data folder.
- **Validation:** golden tests (`tests/test_golden.py`, opt-in with `uv run pytest -m golden`)
  reproduce the research's F2, capitulation (5m and 1m fills) and walk-forward CNN results on the
  real data; [`backtest-validation.md`](backtest-validation.md) gives the numbers and explains
  every difference (cash checks, the research's look-ahead filters, the cost model, re-running
  the book on 1m fills).

### Trading platform

The live side is one long-running **daemon** process (`moonshot live run`), built for Docker on
an always-on host. It owns the market data feed, the in-process plugins, the engine (books,
sizing, risk, orders), the broker, the state file and the HTTP API. **External providers** are
separate processes, anywhere on the network, that run strategies the daemon should not host
(heavy libraries, models, GPUs). Everything in the daemon runs on one asyncio event loop with an
injected clock: wall-clock UTC live, simulated time in replay.

Status: P1 (`docs/plans/live-paper-engine.md`) builds paper mode and replay; the web UI,
controls UI and Docker image are P2; the Binance broker is P3 (see Build order).

#### Run config

A TOML file (`live/config.py`, validated with pydantic) names the mode, the account, the
account-wide limits and each strategy instance. Secrets never go in it: the API token and
(later) exchange keys come from the environment.

```toml
mode = "paper"                     # paper | testnet | live (live also needs --mainnet)
state = "state/paper.sqlite"

[account]
initial_cash = 10000               # paper and replay only; testnet/live read the balance
max_order_usdt = 500
max_total_exposure = 1.0           # open cost basis / account realised equity
stale_after = 2                    # intervals: no entries on a symbol whose data is older

[engine]
tick_timeout = "10s"               # wait this long after a close for every symbol's candle
signal_deadline = "30s"            # after a candle closes, wait this long for signal batches

[[strategies]]
name = "r2"                        # instance name: unique, used in the state and the API
source = "crypto_moonshot.strategy.capitulation:R2"   # module:attr or path/to/file.py:attr
params = {}                        # passed to the plugin factory, if it is callable
symbols = "all"                    # a universe name or a list
capital = 2000                     # this strategy's book
sizing = { stake = 50 }            # or { slots = 5 } or { weighted_slots = 5 }
risk = { max_positions = 200, daily_loss_limit = 0.1 }
exit = { take_profit = 0.2, max_hold = "3d" }         # optional: overrides the plugin's plan

[[strategies]]
name = "cnn"
source = "external"                # an external provider posts this strategy's batches
interval = "5m"
symbols = "pool"
warmup = 2016
capital = 3000
sizing = { slots = 5 }
exit = { max_hold = "2h" }
```

#### Strategy plugins

`live/plugin.py` defines what a plugin is and how one is loaded:

```python
class Strategy(Protocol):
    name: str  # set from the config's instance name
    interval: Interval
    warmup: int  # candles of history each window holds (at least)
    context_symbols: tuple[str, ...]  # extra symbols whose windows it reads (e.g. BTCUSDT)
    exit: ExitPlan | None  # default bracket for its entries

    def on_candles(self, windows: Mapping[str, pl.DataFrame], time: datetime) -> pl.DataFrame:
        """Signals for the candle that opened at `time` (just closed), from each symbol's
        window: closed candles in `KLINE_SCHEMA`, oldest first, ending at `time` (a symbol
        whose candle at `time` is missing ends earlier); holes are possible. Returns a frame
        with `symbol` and `priority`, optional `action` ("enter", the default, or "exit"),
        `weight`, exit plan columns overriding the default, and any extra columns."""
```

- **Loading.** A source is `package.module:attr` or `path/to/file.py:attr`, imported at start-up,
  so a strategy can live in its own repository or a folder mounted into the container and needs
  no change to the platform. If `attr` is callable (a class or factory) it is called with the
  config's `params`. A `SignalRule` is wrapped in `RuleStrategy` automatically. The result is
  checked against the protocol before the daemon starts.
- **Dependencies.** In-process plugins may only use what the daemon has installed (polars,
  numpy). A plugin that needs more runs as an external provider, which has its own environment.
- **Backtesting.** A plugin may also offer `backtest_signals(root, symbols, until) -> Signals`,
  a fast batch computation over history (`RuleStrategy` uses `compute_signals`); otherwise
  `moonshot backtest plugin` runs `on_candles` over rolling windows, which is slower but gives
  exactly the live signals. `moonshot backtest signals FILE` backtests a signal file produced
  anywhere (the columns of `Signals.frame`).
- **Runtime.** The engine calls in-process plugins in a worker thread with the tick's deadline, so
  a slow plugin misses its batch like a late provider instead of stalling the loop.

#### Signal contract (v1)

`live/contract.py` (pydantic). Every strategy, in process or external, delivers exactly one
**batch** per closed candle of its interval, even when it has no signals, so the engine knows
when everyone has reported:

```json
{
  "contract": 1,
  "strategy": "cnn",
  "interval": "5m",
  "candle_time": "2026-11-20T13:40:00Z",
  "signals": [
    {"symbol": "SOLUSDT", "action": "enter", "priority": -0.991, "weight": null,
     "exit": {"take_profit": 0.05, "max_hold": "2h"}, "info": {"score": 0.991}}
  ]
}
```

- `candle_time` is the open time of the signal candle *t*; entries are placed for the open of
  *t+1*, as in the backtest. `priority` orders signals competing for a slot (lower first).
  `weight` feeds weighted sizing. `exit` overrides the strategy's exit plan for that entry.
  `info` (scalars) is stored with the signal and carried into the trade list.
- An `exit` action closes the strategy's position in the symbol at market (cancelling its
  bracket); one for a symbol the strategy does not hold is ignored and recorded.
- A batch is identified by `(strategy, candle_time)`; a second batch for the same candle is
  refused, so a provider may retry safely.
- **Deadline.** The engine processes a candle's batches when every strategy of that interval has
  reported, or `signal_deadline` after the candle closed, whichever is first. A batch that arrives
  later is *late*: its entries are rejected as expired (an entry well after the open of *t+1* no
  longer matches the backtest), its exits are still carried out. How late every batch was is
  recorded, so provider latency is visible.
- Changing the contract means bumping `contract`; the daemon refuses versions it does not know.

#### Market data feed

`data/stream.py` (Binance WebSocket client) and `live/feed.py` (ticks and windows).

- WebSocket connections to Binance's market-data stream host (`data-stream.binance.vision`, no
  key) carry the kline streams of every (symbol, interval) any strategy needs, context symbols
  included. Subscriptions are sent in chunks within Binance's message-rate limit, and another
  connection is opened when one would exceed the stream limit per connection (1024). Updates of a
  candle still forming are ignored.
- **Ticks.** Closed candles are grouped by (interval, open time). A tick is emitted when every
  subscribed symbol has closed that candle, or `tick_timeout` after the close; symbols missing
  then are listed in the tick and filled in later by backfill (a late candle never re-opens a
  tick).
- **Reconnects.** Binance drops connections every 24 hours and at other times. The client
  reconnects with backoff; after any reconnect, or a gap noticed in a symbol's open times, the
  missed candles are fetched over REST (`data/binance.py`) before the next tick is emitted.
- **Windows.** The feed keeps each symbol's last `max(warmup)` closed candles in memory. At
  start-up they are seeded from the Parquet store where it has them, and the rest (or all of it,
  in a bare container) from REST. The daemon never writes the store.
- **Staleness.** A symbol whose latest candle is more than `stale_after` intervals old is stale:
  no entries in it, and the status says so.

#### Engine

`live/engine.py`. For each tick (interval, candle *T*) of a live or replay feed:

1. **Broker first.** The broker is advanced to the close of *T* (`PaperBroker.on_candles`):
   entries placed for the open of *T* fill, and every bracket checks candle *T*. Their fills are
   recorded and the books updated, so exits at or before the next entry time free cash and slots
   before new entries, as in the backtest.
2. **Signals.** In-process plugins for that interval run on the windows; external providers get
   the tick over the API. The engine collects batches until all have reported or the deadline.
3. **Exits** from exit signals are placed (market sells for the open of *T+1*).
4. **Entries**, across strategies, in order of priority then symbol (each strategy's book is
   separate, so the order matters only for account limits and symbols two strategies want).
   Each passes, in order: pause, the account's one-strategy-per-symbol rule, staleness, the
   strategy's sizing (on its book's realised equity) and risk checks (`risk.py`, unchanged),
   cash in its book, then the account checks (`max_order_usdt`, `max_total_exposure`). The first
   refusal is recorded as the signal's reason; the rest become market buys for the open of
   *T+1*, carrying the exit plan. The stake is reserved in the book at once, as the backtest
   spends it at entry.
5. **Time exits** are part of the bracket for the paper broker; for an exchange broker (P3) the
   engine places them, because Binance has no order that expires after a holding time.

- **Books.** Each strategy instance has its own `portfolio.Book` with its allocated capital, so
  strategies cannot starve each other and each can be judged alone. Account checks read the sum.
  Only one strategy may hold a symbol at a time (the brokers hold one position per symbol).
- **Controls.** `pause` stops new entries (exits keep working); `resume` lifts it; `flatten`
  sells every position at market and pauses. Controls are stored in the state file, so they
  survive a restart.
- **Entry candle missing.** If candle *T+1* never arrives for a symbol with a pending buy (a
  hole), the buy is cancelled and its stake released. The backtest skips such a signal before
  sizing, so a live run can differ from a backtest only when another entry in the same candle was
  refused for a slot the cancelled one held.

#### Paper broker

`broker/paper.py`. Fills against candles as they arrive, with exactly the backtest's rules:

- A market order placed for time *t* fills at the open of the first candle at or after *t*, once
  that candle has closed, with `Costs` as in `SimulatedBroker`. The fill time is the candle's
  open time.
- Each open bracket is re-evaluated on every new candle by calling `resolve_exit` on the
  position's candles so far (from the entry candle, `ended=False`); the first exit it returns is
  applied. Because it is the same function, a position fed candle by candle exits exactly where
  `SimulatedBroker` would; a parity test checks this over random paths and exit plans.
- Brackets are evaluated on candles of the position's strategy interval (no 1m fills live).
- A paper account's cash, positions and brackets are rebuilt from the state file on restart.

#### State and recovery

`live/state.py`: one SQLite file per run (WAL mode, schema version in a `meta` table, migrations
applied at start-up), written only by the daemon. Tables: `meta`, `strategies` (config and
capital), `ticks` (last processed candle per interval), `batches` (strategy, candle time,
received time, lateness, status), `signals` (each signal, its outcome and reason), `orders`
(client order ID, strategy, symbol, side, amounts, exit plan, status), `fills`, `positions`
(entry and exit details, P&L), `controls` (paused, flatten requested), `events` (time, level,
kind, message: the alert feed the UI shows) and `providers` (last batch seen).

- **Intent before action.** An order row is written with status `pending` before the broker is
  called and updated after, in one transaction per tick step. Client order IDs are deterministic
  (strategy, symbol, candle time, side; hashed to Binance's 36 characters), so a retry or a
  restart can never place an order twice.
- **Books are derived, not stored.** On start-up each book is rebuilt from `positions` and
  `fills` (capital, plus closed P&L, minus open cost; today's P&L and entries from today's rows),
  so there is no second copy that can drift.
- **Restart.** The daemon loads open positions and pending orders, re-seeds the windows and
  backfills candles since the last processed tick, and lets the broker catch up: the paper broker
  re-runs `resolve_exit` from each entry, so exits that happened while it was down are applied at
  their real times, as the exchange would have done. Ticks missed while down are recorded as
  missed; their signals are not replayed (they would be stale). An exchange broker (P3)
  reconciles open orders and balances with Binance instead.
- **Crash safety** is tested by killing a replay at arbitrary points and restarting it: the end
  state must be identical to an uninterrupted run.

#### Replay

`moonshot live replay --config FILE --start --end` runs the same engine over stored candles: the
feed reads ticks from the Parquet store, the clock jumps to each close, and the deadline waits for
every strategy's batch (external providers included) instead of wall-clock time. It writes the
state file plus a trade list, equity curve and metrics in the backtest's format. Replay is how
the platform is tested without keys or live markets: its trades must match `backtest.engine` on
the same signals and settings, and a replay interrupted and restarted must match one that was
not.

#### API

`live/api.py`, served by the daemon (FastAPI on uvicorn, in the same event loop), versioned under
`/api/v1`. Every endpoint except `health` needs `Authorization: Bearer <MOONSHOT_API_TOKEN>`;
the token comes from the environment and is never logged. The daemon binds to localhost unless
configured otherwise (on the LAN host, in Docker).

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | alive, feed lag, last tick |
| `GET /status` | mode, controls, each strategy's book, positions, open orders, stale symbols, providers |
| `GET /candles?interval=&symbols=&limit=` | windows for a provider's warm-up (Arrow IPC, or JSON) |
| `WS /ticks?interval=` | each tick as it is emitted, with its closed candles |
| `POST /signals` | a signal batch (the contract above) |
| `POST /control/{pause,resume,flatten}` | controls |
| `GET /events?since=` | the event feed |

#### External providers

`crypto_moonshot/provider.py` is a small SDK: `run_provider(strategy, url, token)` fetches the
warm-up windows, follows the tick stream, keeps its own windows, calls the *same*
`Strategy.on_candles` and posts the batch. `moonshot provider run SOURCE --url --params` runs a
plugin this way, so the same plugin file runs in process or out of process unchanged. A provider
may be written in any language: it only needs the endpoints and the contract above.

## Run modes

| Mode                           | Candles             | Order fills                           |
| ------------------------------ | ------------------- | ------------------------------------- |
| `backtest` (built)             | Stored history      | `SimulatedBroker`                     |
| `replay`                       | Stored history      | `PaperBroker`, simulated clock        |
| `paper`                        | Live mainnet stream | `PaperBroker`                         |
| `testnet` (default for orders) | Live mainnet stream | Binance spot testnet                  |
| `live`                         | Live mainnet stream | Binance mainnet; requires `--mainnet` |

`replay` and `paper` need no API keys: candles come from the public market-data hosts.

## Package layout and CLI

```
src/crypto_moonshot/
├── cli.py              # Typer app; data_cli.py `data`, backtest_cli.py `backtest`, live_cli.py `live` and `provider`
├── cli_output.py       # summary, status and backtest tables (rich in a terminal, plain text otherwise)
├── progress.py         # sync progress renderers: bars and lines
├── config.py           # settings from MOONSHOT_* environment variables (data dir, API token, later keys)
├── models.py           # Interval, kline schema
├── data/               # binance.py (REST), store.py, sync.py, importer.py, status.py, universe.py, stream.py (WebSocket)
├── strategy/           # base.py, features.py, compute.py, capitulation.py, scores.py
├── portfolio.py        # the book and position sizing
├── risk.py
├── broker/             # base.py (orders, ExitPlan, Costs), candles.py, simulated.py, paper.py; binance.py (P3)
├── backtest/           # engine.py, metrics.py, source.py (candle windows; Parquet-backed and in-memory)
├── live/               # config.py (TOML run config), contract.py (signal batches), plugin.py (Strategy,
│                       # loader, RuleStrategy), feed.py (ticks, windows; live and replay), state.py (SQLite),
│                       # engine.py, api.py, daemon.py (wires them together)
└── provider.py         # SDK for external signal providers
```

Commands:

- `moonshot data sync`: download missing candles
- `moonshot data status`: show coverage and gaps
- `moonshot data import <folder>`: reuse candles downloaded into another data folder
- `moonshot backtest f2|capitulation|scores`: backtest a strategy and write `results/`
- `moonshot backtest plugin SOURCE`: backtest any strategy plugin
- `moonshot backtest signals FILE`: backtest a signal file produced anywhere
- `moonshot backtest convert-scores NPZ META OUT`: make a score file from a research CNN output
- `moonshot live run --config FILE`: run the daemon (`mode` from the config; P1 accepts `paper`)
- `moonshot live replay --config FILE --start --end`: run the engine over stored candles
- `moonshot live status|pause|resume|flatten`: talk to a running daemon (`status` reads the state
  file when the daemon is down)
- `moonshot provider run SOURCE --url URL`: run a plugin as an external provider

Dependencies so far: polars, numpy (the broker and engine work on arrays), httpx, pydantic-settings,
rich, typer; the trading platform adds websockets, fastapi and uvicorn. Expected later: duckdb.

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
- **Strategies are plugins, not platform code.** Strategy research moves faster than the trading
  platform and needs different libraries. Loading strategies from a path named in a config, or
  taking their signals from another process, lets new strategies trade without a platform
  release, and keeps a bug or a heavy model in a strategy from taking down the trader.
- **A light daemon; heavy compute out of process.** The daemon must react to every candle close
  within seconds and never stall; a torch model scoring hundreds of symbols can take longer, use
  a GPU and pull in large dependencies. External providers isolate that, and can run on another
  machine. The daemon is the single market-data hub, so the engine and every provider see the
  same candles.
- **One batch per strategy per candle, with a deadline.** Waiting for every strategy's batch
  (empty or not) lets the engine rank all of a candle's signals at once, as the backtest does,
  without waiting forever for a dead provider. Entries that arrive late are refused because a
  fill well after the open of *t+1* would no longer be the strategy that was backtested.
- **Per-strategy books.** Separate capital per strategy keeps one strategy from using up another's
  cash and slots, and makes each strategy's live results comparable with its own backtest.
- **The paper broker reuses `resolve_exit`.** Calling the backtest's exit function on the
  candles seen so far gives identical exits by construction, rather than a second implementation
  that has to be kept in step.
- **Replay before live.** Running the live engine over stored candles tests the whole path
  (feed, plugins, batches, books, broker, state, recovery) against the backtest without keys or
  waiting for markets.

## Open questions

- **Position sizing scheme.** Equal weight is a placeholder until a strategy needs something else.
- **Capital between strategies.** Books are fixed allocations; rebalancing capital between
  strategies (or sharing it) is open until there are several strategies worth running.
- **Fine fills live.** The paper broker fills brackets on the strategy's interval; 1m fills (as
  `--fine` backtests use) would need the feed to subscribe 1m streams for held symbols.

## Build order

1. Models and config
2. Data layer: REST client, store, sync, `data sync` and `data status`
3. Strategy interface and the capitulation rules (done; the moving-average crossover was
   dropped in favour of the research strategies)
4. Backtest engine, metrics, `moonshot backtest` and golden validation (done)
5. Trading platform in paper mode: plugins, external providers, feed, paper broker, state,
   engine, replay, API (P1)
6. Operating it: web dashboard, controls, alert feed, Docker image for the Proxmox host,
   runbook (P2)
7. Binance broker on testnet: signed REST, exchange-filter rounding, OCO brackets, user data
   stream, reconciliation; built against mocks until testnet keys exist (P3)
8. Mainnet with small position sizes (P4)
