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
`risk.standard_checks` builds the list in the one order both engines apply it.

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
- The live engine trades through the async `ExecutionBroker` (`broker/execution.py`), which the
  paper broker satisfies behind `PaperExecution` and exchange brokers implement directly (see
  Trading platform, Exchange broker).
- `BinanceBroker` (`broker/binance.py`, P3) trades on the exchange: market entries, brackets as
  OCO order lists, fills from the user data stream; the engine runs its time exits. See Trading
  platform, Exchange broker.

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

Status: P1 (`docs/plans/live-paper-engine.md`) built paper mode, replay, the API, the provider SDK
and the CLI; P2 (`docs/plans/live-operations.md`) adds health rules and alerts, watchdogs, backups
and the dashboard's endpoints (built), then the web UI and the Docker image; the Binance broker is
P3 (see Build order). Strategy authors: see [strategy-plugins.md](strategy-plugins.md).

#### Run config

A TOML file (`live/config.py`, validated with pydantic) names the mode, the account, the
account-wide limits and each strategy instance. Secrets never go in it: the API token and
(later) exchange keys come from the environment.

```toml
mode = "paper"                     # paper | testnet | live (only paper runs in P1)
state = "state/paper.sqlite"       # relative to the config file's folder

[account]
initial_cash = 10000               # paper and replay only; testnet/live read the balance
max_order_usdt = 500               # an entry's stake above this is refused
max_total_exposure = 1.0           # open cost basis / realised equity, both summed over the books
stale_after = 2                    # intervals: no entries on a symbol whose data is older
costs = { fee = 0.001, slippage = 0.0005, latency = 0.0 }   # the simulated exchange's

[engine]
tick_timeout = "10s"               # wait this long after a close for every symbol's candle
signal_deadline = "30s"            # after a candle closes, wait this long for signal batches
entry_grace = "5m"                 # cancel a pending buy whose candle is this overdue
replay_timeout = "1h"              # replay: longest real wait for a candle's batches

[health]                           # all optional; see Operations
health_every = "15s"               # how often the health rules are evaluated
feed_grace = "60s"                 # a feed connection down this long raises feed_disconnected
no_ticks_intervals = 2             # no_ticks: this many intervals (+ tick_timeout) without a tick
provider_misses = 3                # provider_missing / plugin_failing: this many batches in a row
backfill_window = "1h"             # backfill_failed: a seed or backfill request failed within this
stale_fraction = 0.2               # stale_symbols: more than this share of an interval's symbols
startup_grace = "2m"               # ... checked only this long after start (the feed is seeding)
min_free_gb = 1.0                  # disk_low: less free disk than this where the state file lives
# watchdog_after = "15m"           # exit 3 after this long without a tick (default 10 x the
                                   # shortest interval, at least 15 minutes)
loop_blocked_after = "60s"         # exit 3 at once when the event loop did not run for this long
backup_every = "24h"               # automatic state backups: at start, then this often
backup_keep = 14                   # ... keeping the newest this many

[[strategies]]
name = "r2"                        # instance name: unique, used in the state and the API
source = "crypto_moonshot.strategy.capitulation:R2"   # module:attr or path/to/file.py:attr
params = {}                        # passed to the plugin factory, if it is callable
symbols = "all"                    # a universe name or a list
capital = 2000                     # this strategy's book
sizing = { stake = 50 }            # or { slots = 5 } or { weighted_slots = 5 }
risk = { max_positions = 200, daily_loss_limit = 0.1 }   # also max_exposure, max_entries_per_day
exit = { take_profit = 0.2, max_hold = "3d" }         # optional: overrides the plugin's plan

[[strategies]]
name = "cnn"
source = "external"                # an external provider posts this strategy's batches
interval = "5m"
symbols = "pool"
warmup = 2016
context_symbols = ["BTCUSDT"]      # extra symbols the provider's plugin reads; the daemon subscribes them
capital = 3000
sizing = { slots = 5 }
exit = { max_hold = "2h" }
```

- **Validation.** Every error names its field (`strategies[1] (cnn).warmup: ...`). Names are
  unique; exactly one sizing key; an external strategy needs `interval` and `warmup`, takes
  no `params` and may list `context_symbols` (an in-process plugin declares its own, so the
  field is refused there); the strategies' capital may not exceed `initial_cash`; durations are strings
  (`models.parse_duration`).
- **Health.** `[health]` (`HealthConfig`) holds every threshold of the health rules, the
  watchdogs and the automatic backups; omitted keys keep the defaults above.
- **Mode.** `testnet` and `live` parse, but `RunConfig.require_runnable` (and the engine)
  refuse them with a clear message: P1 places no exchange orders.
- **Costs** are account-wide: the paper broker is one simulated exchange with one fee schedule.
- **In-process plugins.** `interval`, if given, must match the plugin's; `warmup` replaces the
  plugin's window length. `exit` overrides fields of the plugin's plan (a plan without any exit
  opens no bracket). `max_positions` has no default, even with slot sizing.
- **Universes** (`pool`, `all`) resolve against the data root's symbol lists at the strategy's
  interval (`StrategyConfig.resolve_symbols`).

#### Strategy plugins

`live/plugin.py` defines what a plugin is and how one is loaded:

```python
class Strategy(Protocol):
    name: str  # set from the config's instance name
    interval: Interval
    warmup: int  # candles of history each window holds (at least)
    context_symbols: tuple[str, ...]  # extra symbols whose windows it reads (e.g. BTCUSDT)
    exit: ExitPlan | None  # default bracket for its entries

    def on_candles(
        self,
        windows: Mapping[str, pl.DataFrame],
        time: datetime,
        context: Mapping[str, pl.DataFrame],
    ) -> pl.DataFrame:
        """Signals for the candle that opened at `time` (just closed). `windows` holds the
        traded symbols' windows, `context` the windows of `context_symbols` (a symbol may be in
        both). A window is closed candles in `KLINE_SCHEMA`, oldest first, ending at `time` (a
        symbol whose candle at `time` is missing ends earlier); holes are possible. Returns a
        frame with `symbol` and `priority`, optional `action` ("enter", the default, or
        "exit"), `weight`, exit plan columns overriding the default, and any extra columns.
        Only signals for the candle `time` are allowed, so there is no `open_time` column."""
```

- **Loading.** A source is `package.module:attr` or `path/to/file.py:attr`, imported at start-up,
  so a strategy can live in its own repository or a folder mounted into the container and needs
  no change to the platform. If `attr` is callable (a class or factory) it is called with the
  config's `params`. A `SignalRule` is wrapped in `RuleStrategy` automatically. The result is
  checked against the protocol before the daemon starts; a source that cannot be loaded or does
  not satisfy it is a `PluginError` naming the source.
- **Output frames.** `normalize_signal_frame` checks every `on_candles` result (required columns
  and types, actions, weights > 0, exit overrides valid, one signal per symbol and action) and
  gives it canonical columns; backtests and the engine both use it. A null in an exit-override
  column keeps the strategy's value for that field.
- **Rule strategies.** `RuleStrategy` adapts a `SignalRule`: `on_candles` runs `candidates` on
  each traded window, joins `context` on `open_time` and applies `select`, keeping the signals of
  candle `time`; `backtest_signals` is `compute_signals`. Its default warm-up is `MIN_HISTORY`
  (8 days, 2304 candles of 5m): enough for the 7-day features and for the length check of
  `features.spikes`, so a window's signals equal the full-history ones (a parity test checks
  this on a store with holes). One edge case differs: a symbol's first `MIN_HISTORY` stored
  candles. The full-series computation applies the length check to the whole series, so it
  signals from the symbol's 7th day once the series ever reaches 8 days, while a window (or a
  live run) cannot know the future length and gives nothing until it holds 8 days. Backtests
  through `backtest_signals` may therefore show signals in a new listing's 8th day that live
  would not take; the parity test excludes exactly those.
- **Dependencies.** In-process plugins may only use what the daemon has installed (polars,
  numpy). A plugin that needs more runs as an external provider, which has its own environment.
- **Backtesting.** A plugin may also offer `backtest_signals(root, symbols, until) -> Signals`,
  a fast batch computation over history (`RuleStrategy` uses `compute_signals`); otherwise
  `moonshot backtest plugin` runs `on_candles` over rolling windows, which is slower but gives
  exactly the live signals (`rolling_signals`: windows of `warmup` candles sliced from the stored
  series, a symbol left out of a tick when its latest candle is over `stale_after` intervals
  old, ticks where no symbol has a candle skipped; only "enter" signals are kept).
  `moonshot backtest signals FILE --interval 5m` backtests a signal file produced anywhere (the
  columns of `Signals.frame`, exit plan from the command's options). Both commands take the
  other `backtest` options; `plugin` uses the plugin's `exit` unless the options override it.
- **Runtime.** The engine reads a plugin's windows on the event-loop thread and calls
  `on_candles` in a worker thread with the tick's deadline, so a slow plugin misses its batch
  like a late provider instead of stalling the loop. A plugin still computing an earlier candle
  misses the next one too (a thread cannot be killed); one that raises or returns a frame that
  breaks the contract misses its batch with an error event. A plugin whose traded symbols have
  no candle in a tick gets an empty batch without being called, and symbols whose latest candle
  is more than `stale_after` intervals old are left out of its windows, both as in
  `rolling_signals`. A plugin with `needs_windows = False` gets empty windows (the replay's
  `PrecomputedStrategy`).

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
  longer matches the backtest), its exits are still carried out (market sells for the next
  candle open after it arrived). How late every batch was is recorded, so provider latency is
  visible. A batch that missed the deadline is recorded as `missed`; when it turns up, that
  record becomes `late` (it is not a duplicate). A batch may arrive before the engine has seen
  its tick, once the candle has closed by the engine's clock; it joins that candle's
  decisions.
- **Exit overrides and info.** `exit` is a partial plan: a field left out keeps the strategy's
  value; durations are strings (`90m`, `2h`, `3d`, `1w`). `info` takes scalars only; the frame
  helpers (`frame_to_batch`, `batch_to_frame`) turn extra columns into `info` and back.
  `candle_time` must be a timezone-aware UTC time aligned to the interval, and a batch may hold
  one `enter` and one `exit` per symbol at most.
- Changing the contract means bumping `contract`; the daemon refuses versions it does not know.

#### Market data feed

`data/stream.py` (Binance WebSocket client) and `live/feed.py` (ticks and windows).

- WebSocket connections to Binance's market-data stream host (`data-stream.binance.vision`, no
  key; `MOONSHOT_STREAM_URL` overrides the base) carry the kline streams of every (symbol,
  interval) any strategy needs, context symbols included. `KlineStream` subscribes with
  `SUBSCRIBE` messages of 200 streams, at most 4 a second (Binance allows 5), and opens another
  connection when one would exceed 1024 streams. Only closed candles (`k.x`) are yielded; forming
  updates and acks are dropped, subscription errors are reported. Any disconnect or error
  reconnects with exponential backoff and jitter and re-subscribes; an optional silence timeout
  treats a quiet connection as dead (off by default, as an hourly stream is quiet for an hour;
  ping/pong already detects dead sockets).
- **`Feed` interface** (`live/feed.py`), what the engine uses: `ticks()` (async iterator of
  `Tick(interval, open_time, candles, missing, emitted_at)`), `windows(interval, symbols, time,
  warmup)`, `last_candle`, `stale_symbols(interval, now, stale_after)`, `health()` and
  `aclose()`. `LiveFeed` implements it over the WebSocket; `ReplayFeed` (`live/replay.py`) implements it
  over stored candles with simulated time. Candles in ticks are `Candle` named tuples (the
  `KLINE_SCHEMA` columns); windows are frames.
- **Ticks.** Closed candles are grouped by (interval, open time). A tick is emitted when every
  subscribed symbol has closed that candle, or `tick_timeout` after the close; symbols missing
  then are listed in the tick and filled in later (a late candle joins its window and never
  re-opens a tick). Ticks of an interval come in open-time order and only after running
  backfills of that interval. A candle is held back until its tick is emitted, so a window seen
  during tick T ends at T for every symbol that has it.
- **Reconnects.** Binance drops connections every 24 hours and at other times. The client
  reconnects with backoff; after any reconnect, or a gap noticed in a symbol's open times, the
  missed candles are fetched over REST (`data/binance.py`) before the next tick of that interval
  is emitted. A range REST cannot fill (an exchange outage) stays a hole and is not retried.
- **Windows.** The feed keeps each symbol's last `max(warmup) + 32` closed candles in memory (a
  frame, with new candles appended in O(1) and concatenated when read). At start-up they are
  seeded from the Parquet store where it has them (only the month files the window touches),
  and the rest (or all of it, in a bare container) from REST. The daemon never writes the store.
- **Events.** The feed reports connects, disconnects, seeding, backfills, subscription errors
  and changes of the stale set to an optional sink callback, and never prints.
- **Staleness.** A symbol whose latest candle is more than `stale_after` intervals old is stale:
  no entries in it, and the status says so.

#### Engine

`live/engine.py` (the module docstring is the exact rule set). One asyncio component with an
injected clock, driven by any `Feed`; it owns the broker (an `ExecutionBroker`, see Exchange
broker: the paper broker behind `PaperExecution` unless another is given) and the books, and
writes everything to the state file, from the event-loop thread only and never inside a
transaction across an `await`. For each tick (interval *I*, candle *T*, entry time *E = T + I*):

1. **Broker first.** Orders left `unknown` (a submit that got no answer) are resolved with
   `query`. A broker that `needs_candles` (paper) gets them for every series it needs on *I* (an
   open position from its entry candle, a pending order from its time): the tick's candle when
   the previous one was handed over, else the feed's window over the last `HISTORY_SLACK`
   candles, which picks up late candles that filled a hole. Then `events(E, interval=I)`: for
   the paper broker, entries placed for the open of *T* fill and every bracket checks candle
   *T*; an exchange broker returns what it has heard since the last call. Every event is
   recorded in one transaction (fills, positions, bracket order rows, cancellations,
   rejections), so exits at or before *E* free cash and slots before new entries, as in the
   backtest. For a `needs_candles` broker, pending buys whose entry candle has not closed
   `entry_grace` after it should have are cancelled (`no_candle`). Each interval advances only
   its own series, so a 1h bracket is never decided at a 5m tick before the 1h candle has been
   handed over. For a broker that does not manage time exits, the engine runs them here (step
   4).
2. **Signals.** Tick listeners run first (the API broadcasts the tick to external providers there).
   In-process plugins for *I* run on the windows (see Strategy plugins, Runtime); external
   batches arrive through `Engine.submit_batch`. Every batch is recorded when it arrives. The
   engine waits until every strategy of *I* has reported (or its plugin failed), or until
   `signal_deadline` after the close by the clock; a tick emitted after the deadline is decided
   at once. In replay it waits for every batch, up to `replay_timeout` of real time, so a replay
   with an external strategy and no provider waits that long per candle.
3. **Decisions** (one transaction, ending with the tick marker). Missing batches are recorded
   as `missed`. The controls are read: a requested flatten sells every open position for *E*,
   cancels pending entries and pauses. **Exits** from exit signals become market sells for *E*
   (ignored, with a reason, when the strategy does not hold the symbol, its entry is still
   pending or a sell is already pending). Then **entries**, across strategies, in order of
   signal priority, then strategy name, then symbol (the tie-break; each strategy's book is
   separate, so the order matters only for account limits and symbols two strategies want).
   Each passes, in order, and the first refusal is recorded as the signal's reason: `paused`;
   `unknown_symbol` (not one of the strategy's symbols); `symbol_held` (one pending or open
   position per symbol across strategies); `stale` (`feed.stale_symbols`); sizing on the
   strategy's book (realised equity; `weighted_slots` reads the signal's `weight`); the book's
   risk checks (`risk.first_rejection`, unchanged); `cash` in the book; `max_order_usdt`;
   `max_total_exposure`. An accepted entry is a market buy for *E* with a deterministic client
   order ID, carrying the strategy's exit plan with the signal's partial override. Entries are
   placed one at a time in that order, so one the broker rejects frees its slot and cash for
   the next. Refusals (read back from the state file, so a resumed step counts them all) are
   summed into one event per tick, written with the tick marker in the step's last
   transaction.
4. **Time exits** are part of the bracket for the paper broker (`manages_time_exits`). For a
   broker without them (an exchange: Binance has no order that expires after a holding time)
   the engine sells, in the broker step of the first tick of the position's interval at or
   after `entry time + max_hold`: it cancels the bracket, and only once the cancel is confirmed
   (a fill that raced the cancel wins, and nothing is sold) places a market sell with reason
   `time` (client ID `time_exit_id(entry ID, E)`, so a failed attempt is retried at the next
   tick).

- **Intent, then action, then outcome.** Every order (entry, exit signal, flatten, time exit)
  is written first: its row as `new` (with its deterministic client order ID) and, for an
  entry, its pending position (stake reserved, `placed_at = E`) and its signal's outcome, in one
  transaction; then `broker.submit` is awaited; then the result is recorded in a second
  transaction: accepted (`pending`, plus any fill that came with it), rejected (the order
  `rejected`; an entry's position cancelled with `broker_<reason>`, its stake released, its
  signal `rejected` with `broker_<reason>`) or unknown (the order `unknown`, resolved with
  `query` at the next broker step; meanwhile the symbol stays held). A whole-position sell
  cancels the position's bracket first when the broker's own sell does not
  (`sell_cancels_bracket`; the paper broker's does), and sells only once the cancel is
  confirmed. So the decision step is several transactions (`Step.BEFORE_BROKER_CALL` lies
  between an intent and `submit`, `Step.AFTER_BROKER_CALL` between `submit` and its outcome);
  a decision step resumed after a crash skips signals that already have an outcome and gives
  each book exactly the numbers the interrupted step had (derived without the tick's placed
  entries, which are then reopened one by one in decision order), so a resumed paper run is
  bit-identical to an uninterrupted one.

- **Books.** Each strategy instance has its own `portfolio.Book` with its allocated capital, so
  strategies cannot starve each other and each can be judged alone. Account checks read the
  sum. Books are kept in memory for the decisions and re-derived from the state file
  (`StateStore.derive_book`) after every step that changed it and whenever the UTC day changes:
  they always equal `derive_book` (a test checks it after every tick), and a restarted engine
  computes bit-for-bit the same stakes as one that never stopped.
- **Controls.** `pause` stops new entries (exits keep working); `resume` lifts it; `flatten`
  sells every position at market at the next tick, cancels pending entries and pauses.
  Controls are stored in the state file, so they survive a restart; `Engine.pause`, `resume`
  and `flatten` set them.
- **Events.** Feed events (`Engine.on_feed_event` is the feed's sink), missed and late batches,
  plugin errors and timeouts, refusals (one event per tick), cancelled entries, flattens,
  restarts and missed ticks go to the state file's event feed.
- **Errors.** A plugin never crashes the engine. Anything else (a broken state file, a bug)
  propagates out of `Engine.run`: the daemon stops and recovers from the state file on
  restart. `Engine.stop` abandons a tick still collecting batches; it is decided after the
  restart (its recorded batches are kept).
- **For the API.** `submit_batch(batch, received_at) -> BatchOutcome` (`on_time`, `late`,
  `duplicate`, or `refused` with a reason), `add_tick_listener`, `reconciliation` (the latest
  report) and `unprotected_positions()` (for the health rules), `status()` (mode, controls,
  books, positions, open orders, stale symbols, last ticks, providers, health; `to_json()`),
  `pause`/`resume`/`flatten`, `on_feed_event`, `record_event`, `collecting(interval)` (the candle
  being collected), `running`, `run(feed)` and `stop()`. `build_specs`,
  `feed_subscriptions` (traded, context and held symbols) and `feed_warmups` give what the
  daemon needs to build a `LiveFeed`.

##### Where a replay can differ from the backtest

The fills are the same (the paper broker calls `resolve_exit`); a few decisions can differ
because the backtest knows some things earlier than a live run can. Each is tested
(`tests/test_replay.py`), and the replay = backtest test uses a fixture where none occurs.

- **A gap exit at the next entry time.** The backtest frees cash, a slot and the symbol from an
  exit at exactly the next entry time *T+1* (a stop or take profit gapped through at the open
  of *T+1*) before taking entries at *T+1*; live learns of that exit only when *T+1* closes, so
  the slot is not free and the symbol is still held for entries at *T+1*. Time exits at exactly
  *T+1* are known in time (at the close of *T*) and do not diverge.
- **A missing entry candle.** The backtest skips such a signal before sizing; live reserves the
  stake until the hole is detected (when a later candle closes, or after `entry_grace`), so
  another entry in the same candle (or the next) may be refused for the slot or cash the
  cancelled one held.
- **`RuleStrategy` on a new listing's days 7-8.** With `on_candles` the replay gives no signals
  until a symbol has `MIN_HISTORY` candles; a backtest through `compute_signals` signals from
  day 7 (see Strategy plugins, Rule strategies).
- **Refusal reasons** (not trades) can differ: live checks the account's one position per
  symbol before the book's risk checks, so a signal the backtest counts as `slots_full` may be
  recorded as `symbol_held`.

#### Paper broker

`broker/paper.py`. Fills against closed candles as they arrive, with exactly the backtest's
rules (the module docstring is the full rule set). The engine reaches it through
`PaperExecution` (`broker/execution.py`: `submit` is `place_order`, `events` is `advance`,
`cancel` is `cancel_order`). The engine hands it candles
(`add_candles(symbol, candles)`, any order, duplicates ignored) and moves it forward
(`advance(until)`); it never reads a clock, and a decision uses only candles that have *closed*
by `until`, so candles handed over early never leak into it.

- **Market buy** for time *t*: the stake is reserved at placement (a buy above the free cash is
  rejected, `OrderRejected` with a `RejectReason`). It fills at the open of candle *t* once that
  candle has closed, with `Costs` as in `SimulatedBroker` and fill time *t*. If the first
  closed candle at or after *t* opens later, the buy is cancelled with `no_entry_candle` and its
  cash released, as the backtest skips the signal.
- **Market sell** (the whole position, or part of one without a bracket): fills at the open of
  the first candle at or after its time, once that candle has closed. Selling the whole
  position cancels its bracket. If the bracket exits first (or at the same time: a gap at the
  open of the sell's candle), the sell is cancelled with `position_closed`.
- **Brackets.** On every `advance` each open bracket calls `resolve_exit` on the position's
  closed candles from the entry candle (`ended=False`) and applies the exit only once it is
  certain. A take profit or stop found in a prefix is the first hit of the full window too (the
  scan goes forward, the trailing peak looks back), so it is applied as soon as its candle has
  closed; a gap fill at a candle's open is applied at that candle's close, with the open as its
  fill time. **A time exit is certain only once every candle that opens before the deadline has
  closed** (`until >= entry + ceil(max_hold / interval) x interval`): `resolve_exit` returns a
  time exit for any unended window without a hit, even one that stops short of the deadline,
  which the backtest never sees because it always loads the window through `max_hold`.
  `series_end` never happens live (a series never ends; a delisted symbol's position stays open
  until the engine sells it).
- **Events.** `advance` returns `EntryFilled` (fill and the new position, with `entry_ref` and
  the bracket ID), `ExitFilled` (fill, position, what remains after a partial sell) and
  `OrderCancelled` (order, reason) in time order across symbols, crediting cash in that order.
  An event's time is when it happened, at or before `until`.
- **IDs.** Orders keep the ID they are given: the engine passes deterministic client order IDs
  (`broker.base.client_order_id(strategy, symbol, candle_time, side)`, 36 characters in
  Binance's charset). A bracket's ID is derived from its buy's (`child_order_id`).
- **Intervals.** Each order fills on the candles of its strategy's interval (`place_order(order,
  interval=...)`, default the broker's); candles are kept per (symbol, interval) only while that
  series has a position or a pending order, from the entry candle or order time on.
  `advance(until, interval=...)` decides only that interval's series. No 1m fills live.
- **Recovery.** `snapshot()` / `PaperBroker.restore()` save and rebuild cash, positions (with
  brackets) and pending orders; the state store builds the same snapshot from its tables
  (`StateStore.broker_snapshot`). After a restore, `required_history()` lists each position whose
  entry candle is missing; once its candles from the entry on are added, `advance(now)` applies
  the exits that happened while the daemon was down, at their real times.
- **Parity.** A test feeds random paths (gaps, holes, candles touching both levels, trailing
  stops, close fills, time exits inside candles, no exit yet) one candle at a time, in bursts,
  out of order, ahead of time and with advances inside candles: the fills equal
  `SimulatedBroker`'s with the whole window, and each exit is applied at the first advance that
  can know it.
- **Known differences from a backtest.** The fills are the same; what can differ is the engine's
  timing around them (see Engine, "Where a replay can differ from the backtest"). And a candle
  that arrives after the advance that needed it is too late: the broker has decided without
  it.

#### State and recovery

`live/state.py`: one SQLite file per run (stdlib `sqlite3`), written only by the daemon. WAL
journal, `synchronous=FULL`, foreign keys on; the schema version lives in `meta` and migrations
are applied at open in one transaction (a file from a newer build is refused). Datetimes are
stored as fixed-width ISO-8601 UTC text and always read back as aware UTC datetimes.

Tables: `meta`; `strategies` (name, config JSON, capital); `ticks` (last processed candle per
interval); `batches` (strategy, interval, candle time, received time, lateness, status
`on_time`/`late`/`missed`); `signals` (each signal, its outcome `placed`/`rejected`/`expired`/
`ignored` and reason); `orders` (client order ID, strategy, symbol, interval, side, type,
amounts, exit plan, time, status `new`/`pending`/`unknown`/`filled`/`cancelled`/`rejected`,
reason, signal);
`fills`; `positions` (strategy, symbol, interval, status `pending`/`open`/`closed`/`cancelled`,
placed time, entry order, entry and exit details, bracket ID, proceeds, P&L, signal info);
`controls` (paused, flatten requested); `events` (time, level, kind, message, data: the alert
feed the UI shows; ids only grow); `providers` (last batch, last seen) and, from schema version 2,
`alerts` (the *active* alerts: key, rule, level, detail, fired time, acknowledged time; a row
lives from firing to resolution, so an acknowledgement ends with it). Version 2 also indexes
`events (kind, id)` and `batches (candle_time)` for the dashboard's queries. Version 3 adds the
order states `new` (the intent is committed, the broker has not answered) and `unknown` (the
broker could not tell whether it placed the order); SQLite cannot change a CHECK constraint, so
it rebuilds the `orders` table (rowids kept). Migrations run with foreign keys off and check them
before they commit. A unique index allows one pending or open position per symbol across
strategies.

- **Read-only open.** `StateStore.open_readonly` (`moonshot live status --config` with the daemon
  down) opens an existing file without creating, migrating or writing anything, and refuses one
  of another schema version (so a version-1 file must be opened by the daemon, which migrates it,
  first).
- **One writer.** The daemon, a replay and `live restore` hold an exclusive lock on
  `<state>.lock` (see Operations); backups and read-only opens do not need it.
- **Typed API.** Records are dataclasses (`OrderRecord`, `PositionRecord`, `BatchRecord`, ...);
  `transaction()` makes one tick step atomic and nests as savepoints; every write outside one
  runs in its own. Statuses only move forward (an order goes `new` -> `pending` or `unknown` ->
  `filled`/`cancelled`/`rejected`, each final; a position goes pending -> open -> closed, or
  pending -> cancelled); a write that would move one back raises. "Open orders"
  (`pending_orders`) are the working ones: new, pending or unknown.
- **Intent before action.** An order row is written as `new` and committed before the broker is
  called, and its outcome is recorded in a second transaction (see Engine). Client order IDs are
  deterministic (strategy, symbol, candle time, side; hashed to Binance's 36 characters) and are
  the primary key, so a retry or a restart can never place an order twice. A bracket is an order
  row too (type `bracket`, written `pending` when the broker reports it), so its exit fill has an
  order.
- **Books are derived, not stored.** `derive_book(strategy, capital, now)` rebuilds a
  `portfolio.Book` from positions alone: cash = capital + closed P&L - cost of open and pending
  positions (a pending entry's stake is reserved); the day's starting equity, P&L, worst running
  P&L (exit order) and entries from today's rows. A cancelled entry never counts, as the
  backtest never counts a skipped signal; entries count on the UTC day of `placed_at`, which the
  engine sets to the entry time (the open of *T+1*), where the backtest's book counts them. A
  test checks it equals a `Book` fed the same entries and closes in time order.
- **Restart.** The engine checks the config against the stored strategies (a changed capital,
  a changed `initial_cash` or a removed strategy with open positions is refused with a clear
  message), loads open positions and pending orders (`broker_snapshot`), rebuilds the broker
  (`PaperBroker.restore`) and derives the books. On the first tick of each interval it hands
  the broker each position's candles from its entry on (and each pending order's from its
  time): from the feed's windows when they reach back far enough, else from the `history`
  fetcher (a `KlineFetcher`: the REST client live, the Parquet store in replay). That tick's
  `advance` then applies exits that happened while it was down at their real times, as the
  exchange would have done. Ticks missed while down are recorded as one `ticks_missed` event;
  their signals are not replayed (they would be stale). Then, before the first tick, every order
  a crash left `new` (or a lost answer `unknown`) is resolved with `broker.query`: one the broker
  holds is accepted (the paper broker holds every one: its snapshot includes `new` orders, and
  it is deterministic, so restoring an order is placing it); one it never placed is rejected
  `not_placed` (an entry's position is cancelled, so only a later tick can enter the symbol
  again), except a sell, which finishes the exit it was meant for (bracket cancelled first, same
  client ID: it never reached the broker). Nothing is re-submitted blind. Then `reconcile`
  compares the broker with the state file (the paper broker trivially agrees; see Exchange
  broker).
- **Crash safety.** An exception inside a transaction rolls it back; a process killed mid-
  transaction (tested with `os._exit` and `kill -9`) leaves the last committed state on reopen.
  Each tick is three steps: the broker step (one transaction), the batches (each recorded in
  its own transaction as it arrives) and the decisions (one transaction that ends with the
  tick marker). A tick whose marker was not committed is processed again after a restart: its
  broker step finds nothing new (the restored broker already holds its effects), recorded
  batches are reused instead of recomputed, and the decisions are made again from the same
  state; the decision step is several transactions (see Engine), and a resumed one reuses the
  signal outcomes already written and rebuilds the books exactly. `tests/test_recovery.py`
  interrupts a replay at every step boundary (after the broker step, after the signals, between
  an order's intent and the broker call, between the call and its outcome, before the tick
  marker, after it), several times in a row and with a real `os._exit` in a subprocess: every
  table, the book, the trades and the equity curve equal an uninterrupted run.
  `tests/test_engine_exchange.py` does the same on an exchange-style broker (a crash after an
  intent, after a submit, during a time exit): the order is queried, never placed twice, and the
  final state has the same rows as an uninterrupted run.

#### Replay

`live/replay.py`. `moonshot live replay --config FILE --start --end` runs the same engine
over stored candles (`run_replay`, or `prepare_replay` to wire the API before running, as
`--serve` does):

- **`ReplayFeed`** implements `Feed` over the Parquet store for `[start, end)`: one tick for
  every candle time at which at least one subscribed symbol has a candle (several intervals in
  order of close time, then interval length), windows sliced from series read once (from twice
  the warm-up plus `HISTORY_SLACK` before `start`, rounded down to a month start), staleness by
  the live rule. Its clock is the close time of the tick being processed, so every time the
  engine writes is deterministic.
- **Deadline.** Replay waits for every strategy's batch (external providers included) instead
  of wall-clock time, up to `replay_timeout`.
- **Lock.** `prepare_replay` takes the state file's lock before anything else and `ReplayRun.close`
  releases it, so a replay can never run beside a daemon (or another replay) on one state file.
- **Resume.** A state file with processed ticks resumes after them; candles older than the
  feed's windows come from the store (`StoreHistory`).
- **Fast path.** With `signals="backtest"` a plugin that has `backtest_signals` is run once over
  the range and its signals are served per tick (`PrecomputedStrategy`) instead of calling
  `on_candles` on every candle; the default `on_candles` path gives exactly the live signals.
- **Outputs.** The state file plus `trades.parquet`/`.csv` (the closed positions in the
  backtest's columns, then `strategy` and the signals' info), `equity.csv` (the backtest's
  curve code, from `initial_cash`; positions still open at the end are valued at their last
  close and are not in the trade list), `yearly.csv` and `metrics.json` (overall metrics, trade
  metrics per strategy, signal outcomes, open positions, the config).

Replay is how the platform is tested without keys or live markets: its trades match
`backtest.engine` on the same signals and settings (except the divergences listed under
Engine), and a replay interrupted and restarted matches one that was not.

#### API

`live/api.py` (`create_app(engine, feed, token)`), served by the daemon (FastAPI on uvicorn, in
the same event loop, so handlers run on the thread that owns the state file), versioned under
`/api/v1`. Every endpoint except `health` needs `Authorization: Bearer <MOONSHOT_API_TOKEN>`,
the WebSocket included. The token is a `SecretStr` in `Settings`, compared with
`hmac.compare_digest`, never put in a URL (the WebSocket reads the header only; a missing or
wrong one gets a plain HTTP 401 before the handshake) and never logged or echoed: uvicorn's access
log is off and its logging configuration is not used, so nothing about a request is logged. The
daemon refuses to start without a token. It binds to `127.0.0.1` unless `--host` says otherwise
(on the LAN host, in Docker); there is no TLS, so anything beyond a private network needs a
reverse proxy or VPN. The interactive docs are off; `GET /openapi.json` (with the token) serves
the schema.

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | no auth: alive, mode, running, feed connected / connections / lag, last tick per interval |
| `GET /status` | `Engine.status().to_json()`: mode, controls, each strategy's book, positions, open orders, stale symbols, providers |
| `GET /strategies/{name}` | what a provider needs: interval, symbols, warm-up, context symbols, `stale_after`, exit plan, contract version, `external` |
| `GET /candles?interval=&symbols=&limit=&format=arrow\|json` | the latest closed candles per symbol (default Arrow IPC stream with a `symbol` column; JSON `{"columns", "windows"}`); a provider's warm-up. 422 for an interval nothing trades or a symbol nobody subscribes |
| `WS /ticks?interval=` | one message per tick of that interval (below) |
| `POST /signals` | a signal batch (the contract above) |
| `POST /control/{pause,resume,flatten}` | controls; the reply holds the resulting controls |
| `GET /events?since=&limit=` | events with an id above `since`, oldest first (without `since`, the latest `limit`); `last_id` is the cursor |
| `GET /trades?strategy=&limit=` | closed positions, newest exit first (below) |
| `GET /equity?strategy=&limit=` | realised equity points and the mark-to-market per strategy (below) |
| `GET /signals?strategy=&limit=` | recent signals with outcome and reason, newest first (below) |
| `GET /providers` | per strategy, the last 24 h of batches (below) |
| `GET /alerts?history=` | active alerts and the alert history (below) |
| `POST /alerts/{key}/ack` | acknowledge an active alert (below) |

- **Dashboard endpoints** (all JSON, token required; field names are stable: the dashboard
  reads them). An unknown `strategy` is a 404 `{"detail": ...}`, a `limit` outside its range a
  422. Times are ISO-8601 UTC.
  - `GET /trades` (`limit` 1-1000, default 100): `{"strategy", "limit", "count", "trades": [{"id",
    "strategy", "symbol", "interval", "entry_time", "entry_price", "exit_time", "exit_price",
    "exit_reason", "quantity", "cost", "proceeds", "pnl", "return_fraction", "hold_seconds",
    "signal_info"}]}`. `cost` includes the entry fee, `proceeds` is after the exit fee, so `pnl =
    proceeds - cost` and `return_fraction = pnl / cost`.
  - `GET /equity` (`limit` 1-10000, default 2000 points per strategy): `{"time", "limit",
    "strategies": {name: {"capital", "realised_pnl", "realised_equity", "unrealised_pnl",
    "equity", "open_positions", "unpriced_positions", "points": [{"time", "equity"}]}}}`.
    `points` is the realised equity (capital plus the cumulative P&L in exit order) after each
    of the latest `limit` closed positions, oldest first. `unrealised_pnl` values the strategy's
    *open* positions at the feed's last close (quantity x close - cost, before the exit fee;
    pending entries are not counted; a position whose symbol has no candle is valued at its
    entry price and counted in `unpriced_positions`); `equity = realised_equity +
    unrealised_pnl`.
  - `GET /signals` (`limit` 1-1000, default 100): `{"strategy", "limit", "count", "signals":
    [{"id", "strategy", "interval", "candle_time", "batch_status", "symbol", "action",
    "priority", "weight", "outcome", "reason", "info"}]}`; `outcome` is `placed`, `rejected`,
    `expired`, `ignored` or `null` while undecided, `reason` says why (`slots_full`, `late`, ...).
  - `GET /providers`: `{"window_hours": 24, "since", "providers": [{"strategy", "external",
    "interval", "on_time", "late", "missed", "median_lateness_seconds",
    "worst_lateness_seconds", "last_batch_time", "last_seen"}]}`, one row per strategy
    (in-process ones count their `missed` batches too, i.e. plugin failures); the counts are of
    batches whose candle is within the window; `last_batch_time` is the candle of the latest
    received batch, `last_seen` the provider's last heartbeat (`null` for in-process ones).
  - `GET /alerts` (`history` 0-500, default 50): `{"active": [{"key", "rule", "level",
    "detail", "fired_at", "acknowledged", "acknowledged_at"}], "unacknowledged": n, "history":
    [{"id", "time", "level", "key", "rule", "state": "firing" | "resolved", "detail"}]}`;
    `active` oldest first, `history` (the `alert` events) newest first.
  - `POST /alerts/{key}/ack`: 200 `{"status": "acknowledged", "alert": {...}}` (acknowledging
    twice keeps the first time), 404 for a key that is not active (unknown, or resolved). It
    writes an `alert_acknowledged` info event.
- **`POST /signals`** calls `Engine.submit_batch`: 202 `{"status": "on_time" | "late",
  "batch_id", "lateness_seconds"}`, 409 `duplicate` (retrying is safe), 422 `refused` (with the
  reason: unknown or in-process strategy, another interval, a candle that has not closed) or
  `invalid` (the body breaks the contract; `errors` lists each field). 401 without the token.
- **Tick stream.** The API registers a tick listener with the engine. Each tick becomes one JSON
  message `{"type": "tick", "interval", "candle_time", "columns", "candles": {symbol: [values in
  `columns` order]}, "missing"}` with every symbol of the interval that closed the candle (traded
  and context symbols), sent to the clients of that interval. A client first gets
  `{"type": "hello", "interval", "last_tick", "collecting", "contract"}`: `collecting` is the
  candle whose batches the engine is waiting for right now, so a provider that connects (or
  reconnects) after that tick was sent answers it from `/candles`. A finished replay sends
  `{"type": "end", "reason"}` and closes the sockets.
- **The broadcast never blocks the engine.** The listener only does `put_nowait` into one bounded
  queue (64 ticks) per client; a client whose queue is full is dropped (closed with code 1013)
  and a `ws_client_dropped` warning event is written. Sending happens in per-client tasks.

#### External providers

`provider.py` is a small SDK: `run_provider(strategy, url=, token=, name=)` (async; `run_provider_sync`
for a script, `Provider` for tests) reads `GET /strategies/{name}`, opens `WS /ticks`, fetches the
warm-up windows (`/candles`), keeps its own windows (each tick's candles appended, bounded to
`warmup` plus the feed's slack), calls the *same* `Strategy.on_candles` and posts the batch.
`moonshot provider run SOURCE --name NAME --url URL [--param k=v]` loads a plugin with
`load_strategy` and runs it this way, so the same plugin file runs in process or out of process
unchanged. A provider may be written in any language: it only needs the endpoints and the
contract above (see [strategy-plugins.md](strategy-plugins.md)).

- **Same inputs.** A provider builds its plugin's windows the way `Engine._plugin_inputs` does
  (traded symbols' windows ending at the tick's candle, bounded to `warmup`, symbols whose latest
  candle is more than `stale_after` intervals old left out, context windows with an empty frame
  for a missing symbol), and posts an empty batch without calling the plugin when no traded
  symbol closed. A test compares the two on a feed with a stale symbol and a missing context
  candle. The plugin's own `exit` and `warmup` are not used: the daemon's config decides.
- **Failure handling.** The plugin runs in a worker thread, so pings keep working during a long
  model call. A dropped connection (or a daemon that is not up yet) is retried with backoff,
  re-fetching the windows each time and answering a pending candle (`hello.collecting`). A wrong
  token, an unknown strategy, an in-process strategy, an interval mismatch or a context symbol
  the daemon does not subscribe is fatal with a message saying what to change; so is a plugin
  that raises (unless `--keep-going`, which skips that candle and lets the daemon record it
  missed). Refused batches are logged as errors and counted.
- **Known difference.** A candle that reaches the daemon late for an earlier tick joins the
  daemon's windows (for in-process plugins) but not the provider's until it reconnects.
- **Dependencies.** The provider process needs this package installed (the SDK, polars and the
  contract) next to the plugin's own libraries; the daemon needs none of them.

#### Daemon

`live/daemon.py` wires everything into one process (`moonshot live run`, paper only). `run_daemon`
checks the token (`MOONSHOT_API_TOKEN`) and the mode first, loads the strategies (`build_specs`),
opens the state file, builds the `Engine`, the `BinanceClient` and the `LiveFeed`
(`feed_subscriptions` and `feed_warmups`, `tick_timeout`, `stale_after`, `on_event =
engine.on_feed_event`, the stream URL from `MOONSHOT_STREAM_URL`), and serves the API on uvicorn in
the same event loop (a bound socket, so a taken port fails first; uvicorn's own signal handling
is replaced by the daemon's).

- **A long-running REST client.** `BinanceClient` aborts for good after `max_rate_limit_hits` HTTP
  429s (right for a one-off `data sync`); the daemon builds it with a limit of a million, so
  backfill keeps waiting out each `Retry-After` instead. An HTTP 418 (IP ban) still makes the
  client refuse requests: backfills then fail and are reported as events until a restart.
- **Shutdown.** SIGINT and SIGTERM (or the `stop` event) stop the engine after the tick in
  progress (a tick still collecting batches is abandoned and decided again after the restart; an
  engine that has not stopped after 30 s is cancelled), close the feed, shut the API down (open sockets are closed), then close the state file. An
  engine failure does the same and exits with an error. `daemon_started` and `daemon_stopped`
  events bracket a run. While it runs the daemon also holds the state file's lock, runs the
  health monitor, the stall watchdog and the backup scheduler as tasks, and the blocked-loop
  watchdog (see Operations).
- **Replay serving.** `serve_replay(prepare_replay(...), settings, ...)` is `moonshot live replay
  --serve`: the same API over a replay's engine and feed, with the health monitor running (the
  rules about wall-clock time are off in simulated time). It waits for a tick client for every
  external strategy (`--provider-timeout`), runs the replay, tells the providers it is over and
  stops the server. An interrupt leaves the state file resumable.

#### Operations

How the daemon is run unattended (P2, `docs/plans/live-operations.md`): on the user's Proxmox
host in Docker, watched from a browser on the LAN. Alerts appear in the dashboard only (no push
channel). The runbook (`docs/runbook.md`) is the operator's guide; this section is the design.

- **Health rules** (`live/health.py`). `HealthMonitor.check()` evaluates a fixed set of rules
  against the engine, the feed and the state file at the engine's clock time; a task in the daemon
  (and in a served replay) calls it every `health_every` (default 15 s), and tests call it
  directly with a fake clock. Each rule yields findings with a key (e.g. `feed_disconnected`, or
  `provider_missing:cnn`). A key that fires and is not active becomes an active alert (a row in
  the state file's `alerts` table) and writes one `alert` event (level `warning` or `error`, data
  `{rule, key, state: "firing", detail}`); an active key that stops firing is deleted and writes
  one with `state: "resolved"` (level `info`). A condition that lasts therefore writes two
  events, not one per check, and active alerts live in the state file, so a restart keeps them
  (and their acknowledgement) and resolves the ones that ended while it was down. A rule that
  raises is logged and leaves its alerts as they are. Rules:

  | Rule | Fires when | Level |
  | --- | --- | --- |
  | `feed_disconnected` | a feed connection has been down longer than `feed_grace` (60 s), as seen by the monitor | error |
  | `no_ticks` | no tick of an interval was processed for 2 intervals plus `tick_timeout` (counted from the start when none was yet) | error |
  | `provider_missing:<strategy>` | an external strategy's last 3 batches were all missed or late | warning |
  | `plugin_failing:<strategy>` | an in-process plugin gave no signals (raised, timed out or broke the contract) for its last 3 candles | warning |
  | `daily_loss_limit:<strategy>` | the strategy's `DailyLossLimit` refuses entries today (book derived at the clock's day) | warning |
  | `backfill_failed` | a `seed_failed` or `backfill_failed` event is within the last hour | warning |
  | `stale_symbols:<interval>` | more than 20% of an interval's traded symbols are stale (not before `startup_grace`, 2 min, while the feed seeds) | warning |
  | `backup_failed` | the last automatic backup failed | error |
  | `disk_low` | less than 1 GB free where the state file lives | warning |
  | `orphan_order` | the latest reconciliation found open orders on the exchange that the state file does not know | warning |
  | `balance_mismatch` | the latest reconciliation found balances that do not cover the open positions (the engine paused entries) | error |
  | `unprotected_position:<symbol>` | an open position's exit plan needs a bracket the broker does not hold, and no sell is working | error |

  Thresholds are the `[health]` settings of the run config. Rule inputs are public accessors:
  `Engine.tick_times()` / `last_tick_at`, `Engine.reconciliation` / `unprotected_positions()`,
  `Feed.health()` / `stale_symbols()`, the store's
  `recent_batch_statuses`, `derive_book`, `count_events` and `alerts`, and injected callables for
  the disk (`disk_free`) and the backup scheduler's last error. **In a served replay** the clock
  is simulated, so `feed_disconnected`, `no_ticks` and the start-up grace of `stale_symbols` are
  off; the other rules run. An alert is *active* while its rule fires. Acknowledging it
  (`POST /alerts/{key}/ack`, stored in the state file) keeps it listed but out of the
  `unacknowledged` count (the dashboard's badge) until it resolves and fires again, which clears
  the acknowledgement; it never silences the rule.
- **Watchdog** (`live/watchdog.py`). The daemon exits with code 3 (so Docker's restart policy
  starts it again, and restart recovery takes over) in two cases.
  *Stalled engine* (`StallWatchdog`, checked with the health rules): no tick was processed for
  `watchdog_after` (default 10 x the shortest interval, at least 15 minutes) while the feed
  reports messages (one within `feed_grace`). The daemon writes a `watchdog` error event, stops
  gracefully (an engine that is stuck is cancelled after 30 s) and raises `DaemonError` with
  `code = 3`. A feed that is down or silent never triggers it: that is `feed_disconnected`'s job.
  *Blocked loop* (`LoopWatchdog`): a loop task refreshes a timestamp every second; a thread
  notices when it is more than `loop_blocked_after` (60 s) old, logs a critical line and calls
  `os._exit(3)`, since a blocked loop cannot shut down cleanly (the state file is crash safe).
  Thresholds, clocks and the exit function are arguments, so tests inject them. Not in replay:
  a replay legitimately waits for providers.
- **Single instance** (`live/lock.py`). The daemon, a replay (also a served one) and `live
  restore` hold an exclusive, non-blocking `fcntl.flock` on `<state>.lock` for as long as they use
  the state file, so two of them can never write one file. The lock goes away with the process;
  the lock file stays. A second start fails at once (exit 1) naming the holder's pid. Backups and
  read-only opens need no lock.
- **Backups** (`live/backup.py`). SQLite's online backup API copies the state file (read-only
  connection, a worker thread) while the daemon runs: WAL mode means the copy is one committed
  snapshot (a transaction in progress is not in it) and the daemon's writes are not blocked. The
  copy is switched to the rollback journal (one self-contained file), made under a `.partial`
  name and renamed to `backups/<stem>-<UTC timestamp>.sqlite` next to the state file only after
  `PRAGMA integrity_check` passes on it, so a name in `backups/` is always a verified backup. The
  daemon backs up at start and every `backup_every` (default 24 h) and keeps the newest
  `backup_keep` (default 14); a failure writes a `backup_failed` event and raises the
  `backup_failed` alert until a backup works again. `moonshot live backup --config FILE [--out DIR]`
  makes one on demand (daemon running or not; it never deletes). `moonshot live restore BACKUP
  --config FILE` checks the backup (integrity; a schema version not newer than the build's, an
  older one is migrated at the next open), takes the lock (so it refuses while a daemon or replay
  runs), moves the current file and its `-wal` to `<state>.before-restore-<UTC timestamp>` and
  puts the backup in place.
- **Logging** (`logs.py`). `--log-format text|json` and `--log-level` on `live run` and `live
  replay`. Text (default) is `time level logger: message` on stderr; `json` is one object per line
  on stdout with `time` (ISO-8601 UTC), `level`, `logger`, `message`, any `extra=` fields and
  `exception`, for Docker. The API token never reaches a log line (the code does not log it, uvicorn's
  access log is off); as a second line of defence the formatter replaces the token with `***`.
- **More API for the dashboard.** The endpoints are in the API table above (`/trades`,
  `/equity`, `/signals`, `/providers`, `/alerts`, `/alerts/{key}/ack`); the queries behind them are
  bounded (limits are capped, and the indexes added in schema version 2 cover the lookups).

- **Dashboard** (`live/web/`, shipped in the package; served by `live/webui.py`). `GET /` is the
  page and `GET /assets/<file>` its files; they are listed when the app is created, so a request
  path is only a lookup key, and every other path under `/` is a plain 404 (never the API). No
  build step, no npm, nothing fetched from the internet: plain HTML, CSS and ES modules
  (`app.js` shell, `pages.js`, `chart.js`, `api.js`, `dom.js`, `format.js`), hand-written SVG. The
  static files carry no data and need no token; they come with `Content-Security-Policy:
  default-src 'self'; script-src 'self'; style-src 'self'; ...` (no inline script or style: the
  code sets styles through the CSSOM only), `X-Content-Type-Options: nosniff`, `Referrer-Policy:
  no-referrer` and `Cache-Control: no-cache` with an ETag (a cheap 304; an upgrade is never
  half-loaded). All data comes from the API with `Authorization: Bearer`; the token is kept in
  `sessionStorage` only (never a URL or `localStorage`), a 401 goes back to the login with a
  message and Log out clears it. Strings from the API are inserted with `textContent` only.
  Pages (hash routes): overview (mode, running, feed and lag, last tick, controls state, the
  alert banner, one row per strategy from `/status`, `/equity` and `/providers`), strategy detail
  (equity chart, open positions, recent trades and signals), positions and open orders (marks from
  `/candles?limit=1`, best effort), trades (strategy filter), signals and providers (lateness
  table), alerts and events (acknowledge; level filter). `/status` and `/alerts` are polled with
  the current page's data every 5 s (3, 5, 10 or 30 s in the header), not while the tab is hidden;
  a failed poll shows "Connection lost. Last updated ...". The alert banner shows on every page and
  is red for an unacknowledged error, amber for a warning. Pause and resume ask in a native
  `<dialog>`; flatten needs the word `flatten` typed; buttons are disabled while a request runs.
  Times are in the browser's zone with the UTC time in a tooltip. P&L is never colour-only: a sign
  (a real minus), an arrow and a hidden "gain"/"loss" word. The equity chart is one blue line
  (realised equity per closed trade), a hollow ring for the current mark-to-market equity, a
  capital baseline, a crosshair with tooltip (pointer and arrow keys) and a table view; light and
  dark follow `prefers-color-scheme`. Browser tests (`tests/test_dashboard_browser.py`, marker
  `browser`, off by default; CI runs them in their own job) drive Chromium through Playwright
  against a served API with seeded data.
- **Docker** (`Dockerfile`, `.dockerignore`, `compose.yaml`, `deploy/config/paper.toml`,
  `moonshot.env.example`). The image is multi-stage: the build stage copies the pinned `uv`
  binary, runs `uv sync --locked --no-dev` (no dev or research group) and installs the package
  non-editable into `/app/.venv` with bytecode compiled; the runtime stage is `python:3.13-slim`
  pinned by digest, with only the venv copied in, a fixed non-root user (uid/gid 10001, no home
  folder), `PYTHONDONTWRITEBYTECODE=1`, `MOONSHOT_DATA_DIR=/data`, `EXPOSE 8765`, a `HEALTHCHECK`
  that fetches `/api/v1/health` with the standard library, `STOPSIGNAL SIGTERM`, entrypoint
  `moonshot` and the default command `live run --config /config/paper.toml --host 0.0.0.0
  --log-format json`. Volumes: `/config` (ro), `/state` (rw: the state file, its lock and
  `backups/`; the only place the daemon writes), `/data` (ro, optional candle store) and
  `/strategies` (ro, plugin files). Run configs use absolute container paths (`state =
  "/state/paper.sqlite"`, `source = "/strategies/x.py:X"`), because a relative plugin source is
  relative to the working directory (`/app`) and a relative `state` to the config's folder
  (`/config`, read-only). The daemon needs no home folder and no writable working directory
  (`tests/test_deploy.py` runs the CLI that way), so `compose.yaml` runs the root filesystem
  read-only with a tmpfs `/tmp`, `cap_drop: [ALL]` and `no-new-privileges`. The one service has
  `restart: unless-stopped` (every exit that is not a `docker stop` restarts: 3 from the
  watchdog, 1, and 2 for a bad config, which shows as a restart loop), `init: true` (tini
  forwards SIGTERM; the daemon's handler stops gracefully and exits 0), a 60 s stop grace period,
  the token from `moonshot.env` (not committed), port 8765, and `json-file` logs with rotation.
  The `docker` CI job validates the compose file, builds the image, runs a container against an
  unreachable Binance (no external network), and checks `/api/v1/health`, 401 without and 200 with
  the token, the dashboard, the non-root uid, and a graceful `docker stop` (exit 0). The
  operator's guide is `docs/runbook.md`.

#### Exchange broker

P3 (`docs/plans/testnet-broker.md`): trading on the Binance spot **testnet**, built and tested
against a fake exchange; mainnet stays refused until P4. The API facts it relies on are in
[binance-spot-api.md](binance-spot-api.md).

- **One execution interface** (`broker/execution.py`, built in M3). The engine talks to an async
  `ExecutionBroker` (the module docstring is the contract):
  - `submit(order, *, interval) -> SubmitResult`: `accepted` (`exchange_id`, and in `events`
    anything that already happened, e.g. a market order's fill), `rejected` (`reason`, a short
    code) or `unknown` (no answer: the outcome must be queried);
  - `cancel(order) -> CancelResult`: `cancelled`, `not_open` (the order is final or was never
    placed; what happened to it is in the result's `events` or was reported before) or
    `unknown`;
  - `query(order) -> QueryResult`: `not_found` (never placed), `open`, `filled`, `cancelled`,
    `rejected` or `unknown`, with the order's unreported events;
  - `events(until, *, interval) -> list[BrokerEvent]`: in time order, everything not reported
    yet: `EntryFilled`, `ExitFilled`, `OrderCancelled` (the paper broker's, shared; `CancelReason`
    gains `cancelled` and `expired`), `OrderRejection` (accepted, then rejected), `BracketFailed`
    and `BracketPlaced` (an unprotected position, then protected again), and `Reconciliation`
    (a report from a reconciliation the broker ran itself, after a reconnect). Each change of a
    known order is reported once; the engine ignores a duplicate. `until` bounds a simulated
    broker; an exchange broker returns everything it has received;
  - `reconcile(orders, positions, *, now) -> Reconciliation`: at start-up, after the engine
    resolved its `new` and `unknown` orders, with the working orders and open positions of the
    state file; it brings them up to date (the changes come from the next `events`) and reports
    `orphans` and balance `mismatches` (computed after those updates). It places and cancels
    nothing.
  - Three properties tell the engine what the broker needs: `needs_candles` (the paper broker
    simulates fills from candles: `candle_needs(interval)`, `add_candles`), `manages_time_exits`
    (the paper broker's bracket includes the time exit; on Binance the engine runs it) and
    `sell_cancels_bracket` (the paper broker's whole-position sell cancels the bracket; on
    Binance the OCO holds the balance, so the engine cancels it first). `cancel` and `query`
    take the engine's `Order`, because Binance needs the symbol and a bracket is an order list.
  `PaperExecution` puts `PaperBroker` behind it (and is restored from the state file instead of
  reconciling), so paper mode, replay and their parity and recovery tests did not change.
  `tests/execution_fakes.py` has `FakeExchangeBroker`, an exchange-style fake the engine tests
  use (asynchronous fills, rejections, unknown outcomes, bracket races, reconcile reports).
- **Intent, then action, then outcome** (built, see Engine). The engine commits the order row
  (`new`, with its deterministic client order ID) before calling `submit`, and records the
  outcome in a second transaction. A crash between the two leaves a `new` row that the next
  start resolves by querying the order by its client ID: filled, open or cancelled (accepted,
  its events recorded), rejected, or never placed: an entry is then rejected `not_placed` (its
  signal is stale; only a later tick may enter the symbol), a sell is placed (it finishes an
  exit that was provably intended). An `unknown` submit result is resolved the same way at the
  next broker step. Nothing is ever retried blind.
- **Entries.** A market buy by `quoteOrderQty` (the stake in USDT, fee included, as in the book).
  The fill's average price becomes the position's reference price: exit levels are relative to
  where the testnet actually filled, because testnet prices drift from the mainnet candles that
  drive the signals. The base quantity bought minus a commission charged in the base asset, rounded
  down to the step size, is what the bracket sells; the rest is *dust*, recorded on the position
  and never sold by the broker. Cost = USDT spent; proceeds = USDT received minus USDT
  commission; a commission in another asset (BNB) is recorded, not converted.
- **Brackets.** Take profit and stop become one OCO sell list: `LIMIT_MAKER` above at
  `ref x (1 + take_profit)` and `STOP_LOSS` below at `ref x (1 - stop)` (a market order once
  triggered), prices rounded to the tick size (take profit up, stop down, so neither is tighter
  than planned). A trailing stop is the `STOP_LOSS` leg with `trailingDelta = trailing x 10,000`
  (basis points) instead of a stop price. Only one of stop and trailing can be live on Binance: a
  plan with both is accepted when `stop >= trailing` (the trailing level is never below the fixed
  one, so the trailing leg alone is equivalent) and refused at start-up otherwise. A plan with
  only a take profit or only a stop places that single order; `tp_fill = "close"` cannot be
  mirrored (a resting limit fills on touch) and is refused in exchange modes. IDs: the list's
  `listClientOrderId` and its legs' IDs are `child_order_id`s of the entry, so a restart finds
  them.
- **Levels already crossed.** If the price is already beyond the take profit when the bracket is
  placed (the `LIMIT_MAKER` would match at once) or below the stop, the broker sells at market at
  once with the corresponding reason, as the backtest fills a gap at the open.
- **Time exits** (engine, built in M3, for brokers that do not manage them): at the first tick
  at or after `entry time + max_hold`, cancel the position's order list, then sell its quantity
  at market, reason `time`. A fill that races the cancel wins; the engine checks before selling
  (the sell's intent is then cancelled `position_closed`). Exit signals and flatten sell the same
  way. A cancel that gets no answer leaves the position unsold (the sell is rejected
  `bracket_open`); a time exit retries at the next tick.
- **Unprotected positions.** If a bracket cannot be placed (a filter such as
  `PERCENT_PRICE_BY_SIDE`, a balance or rate-limit error), the broker reports `EntryFilled`
  without a bracket ID and `BracketFailed`; the engine writes a `bracket_failed` event and the
  `unprotected_position:<symbol>` error alert fires (any open position whose plan has a take
  profit, stop or trailing stop and no working bracket or sell) until the broker, retrying every
  tick, reports `BracketPlaced` (a new client ID for every bracket that reached the exchange).
  Flatten still works.
- **Fills.** The user data stream on a WebSocket API connection
  (`userDataStream.subscribe.signature`): `executionReport` events become fills and cancellations,
  matched to positions by client order ID (`c`, or `C` for cancels); `listStatus` closes an OCO.
  While the stream is down (and after every reconnect, at least once a minute regardless) the
  broker polls open orders over REST, so a missed event only delays a fill, never loses it.
- **Reconciliation.** At start-up and after a reconnect: every local pending or open order is
  queried by client ID and brought up to date; open orders on the exchange with our ID prefixes
  but unknown locally are reported (`orphan_order` event and alert) and left alone; each open
  position's base balance (free plus locked) must cover its quantity, otherwise a
  `balance_mismatch` error event and alert, and the engine pauses entries (resuming is the
  operator's call). Reconciliation never places or cancels orders on its own; the engine
  finishes an exit it can prove was intended (a never-placed sell, a time exit past its
  deadline). The engine runs it at start-up (M3); the broker runs it after reconnects and
  reports through `events` (M4).
- **Account checks.** In exchange modes the account's free USDT at start must cover the sum of
  the strategies' capital (otherwise start-up refuses), books still come from the state file, and
  `initial_cash` is ignored. Symbols not trading on the exchange are dropped from strategies with
  a warning event; entries below a symbol's minimum notional are refused (`min_notional`).
- **Rate limits.** The signed client (`exchange/client.py`) paces requests by the weight and
  order-count headers (and by the orders it sent itself), backs off on 429 for `Retry-After` and
  refuses everything after 418 (`Banned`; the broker raises an `exchange_banned` error alert).
  It never retries an order-placing or cancelling request: a timeout or 5xx raises
  `OrderStatusUnknown` and the broker queries by client order ID. Read-only requests retry on
  5xx and network errors, and a `-1021` triggers one server-time resync and retry. Symbol
  filters come from `ExchangeRules` (refreshed hourly or after a filter rejection), with all
  prices and quantities as `Decimal` rounded to the tick and step. The fake testnet REST server
  for tests is `tests/exchange_fakes.py`.
- **Keys** (`MOONSHOT_BINANCE_TESTNET_API_KEY`, `MOONSHOT_BINANCE_TESTNET_API_SECRET`, from the
  environment or `.env`) are `SecretStr`s, required only in testnet mode, sent only in the
  `X-MBX-APIKEY` header and signatures, and never logged, echoed or stored. Mainnet keys will use
  `MOONSHOT_BINANCE_API_KEY` / `_SECRET` (P4), so the two cannot be mixed up.
- **Known differences from paper.** Fills happen at the testnet's prices, not the mainnet candles;
  a stop fills at market after it triggers (slippage beyond the level); take profits fill on touch;
  dust stays behind; time exits happen at the first tick after the deadline rather than exactly at
  it.

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
├── cli.py              # Typer app (`python -m crypto_moonshot` too); data_cli.py `data`, backtest_cli.py `backtest`, live_cli.py `live` and `provider`
├── cli_output.py       # summary, status and backtest tables (rich in a terminal, plain text otherwise)
├── progress.py         # sync progress renderers: bars and lines
├── config.py           # settings from MOONSHOT_* environment variables (data dir, API token, later keys)
├── models.py           # Interval, kline schema
├── data/               # binance.py (REST), store.py, sync.py, importer.py, status.py, universe.py, stream.py (WebSocket)
├── strategy/           # base.py, features.py, compute.py, capitulation.py, scores.py
├── portfolio.py        # the book and position sizing
├── risk.py
├── exchange/           # client.py (SignedClient: signing, time offset, pacing), pacing.py, rules.py (exchangeInfo filters, Decimal rounding), errors.py (typed Binance errors), models.py (parsed responses) (P3)
├── broker/             # base.py (orders, ExitPlan, Costs), candles.py, simulated.py, paper.py, execution.py (ExecutionBroker, PaperExecution); binance.py (P3)
├── backtest/           # engine.py, metrics.py, results.py (result files), source.py (candle windows)
├── live/               # config.py (TOML run config), contract.py (signal batches), plugin.py (Strategy,
│                       # loader, RuleStrategy), feed.py (ticks, windows), state.py (SQLite), engine.py,
│                       # replay.py (replay feed and runs), api.py (HTTP + WebSocket), daemon.py (wires them together)
│                       # health.py (health rules, alerts), watchdog.py (stall and blocked-loop exits), lock.py (state
│                       # file lock), backup.py (backups, restore), web/ (dashboard, P2)
├── logs.py             # text / JSON logging for the long-running commands
└── provider.py         # SDK for external signal providers

examples/               # paper.toml (a run config) and strategies/ma_cross.py (an example plugin)
Dockerfile, compose.yaml # the daemon's image and a compose file for the Proxmox host (P2)
```

Commands:

- `moonshot data sync`: download missing candles
- `moonshot data status`: show coverage and gaps
- `moonshot data import <folder>`: reuse candles downloaded into another data folder
- `moonshot backtest f2|capitulation|scores`: backtest a strategy and write `results/`
- `moonshot backtest plugin SOURCE`: backtest any strategy plugin
- `moonshot backtest signals FILE`: backtest a signal file produced anywhere
- `moonshot backtest convert-scores NPZ META OUT`: make a score file from a research CNN output
- `moonshot live run --config FILE [--host --port --data-dir --log-format text|json --log-level]`:
  run the daemon (`mode` from the config; P1 accepts `paper`; needs `MOONSHOT_API_TOKEN`)
- `moonshot live replay --config FILE --start --end [--out --state --signals rolling|backtest
  --serve --host --port --json --log-format --log-level]`: run the engine over stored candles and print the metrics like
  `moonshot backtest` (`--serve` keeps the API up for external providers)
- `moonshot live status [--url] [--config] [--json]`, `live pause|resume|flatten [--url]`
  (`flatten` asks to confirm unless `--yes`): talk to a running daemon; `status --config` reads
  the state file (read-only) when no daemon answers
- `moonshot live backup --config FILE [--out DIR] [--json]`: an online, verified backup of the state
  file (daemon running or not); `moonshot live restore BACKUP --config FILE`: replace the state file
  with a backup (daemon stopped; the old file is kept as `<state>.before-restore-<time>`)
- `moonshot provider run SOURCE --name NAME --url URL [--param k=v] [--keep-going]`: run a plugin as
  an external provider (token from `MOONSHOT_API_TOKEN`)

Exit codes of the `live` and `provider` commands: 0 ok, 1 runtime failure (the daemon stopped, a
refused token, an unreachable daemon, a state file in use by another process, a failed backup or
restore), 2 usage or configuration error (no token, a bad config, a plugin that does not load,
`testnet`/`live` mode), 3 the daemon's watchdog stopped it, 130 interrupted. Errors go to stderr.

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
