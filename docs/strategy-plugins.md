# Writing a strategy plugin

A strategy is a small Python object the trading daemon calls once for every closed candle. It
reads the recent candles of its symbols and answers with the entries (and exits) it wants. The
same object is backtested, replayed over stored candles and paper traded, so what a backtest
finds is what the daemon would have seen. It can run inside the daemon, or in a separate
process (or machine) when it needs heavy libraries such as torch.

Everything here is paper trading: no order reaches an exchange.

## The protocol

A plugin is any object with these attributes (`crypto_moonshot.live.plugin.Strategy`):

| Attribute | Meaning |
| --- | --- |
| `name` | The instance name. The platform sets it from the run config; give any default. |
| `interval` | The candle interval it trades, an `Interval` (`Interval.H1`, `Interval.M5`, ...). |
| `warmup` | Candles of history each window holds (at least). Windows are `warmup` rows at most. |
| `context_symbols` | A tuple of extra symbols whose candles it reads, e.g. `("BTCUSDT",)`. |
| `exit` | The default bracket for its entries: an `ExitPlan` or `None`. |
| `on_candles(windows, time, context)` | Called for every closed candle; returns the signals. |

```python
from collections.abc import Mapping
from datetime import datetime, timedelta

import polars as pl

from crypto_moonshot.broker.base import ExitPlan
from crypto_moonshot.models import Interval


class MaCross:
    def __init__(self, fast: int = 12, slow: int = 48) -> None:  # params from the config
        self.name = "ma_cross"
        self.interval = Interval.H1
        self.fast, self.slow = fast, slow
        self.warmup = slow + 2
        self.context_symbols: tuple[str, ...] = ()
        self.exit = ExitPlan(take_profit=0.05, stop=0.03, max_hold=timedelta(hours=48))

    def on_candles(
        self,
        windows: Mapping[str, pl.DataFrame],
        time: datetime,
        context: Mapping[str, pl.DataFrame],
    ) -> pl.DataFrame: ...  # see examples/strategies/ma_cross.py for the whole file
```

[`examples/strategies/ma_cross.py`](../examples/strategies/ma_cross.py) is a complete, readable
example (a moving-average cross, polars only). Copy it to start.

### What `on_candles` receives

- `time`: the open time of the candle that just closed (timezone-aware UTC).
- `windows`: the traded symbols' windows, `{symbol: frame}`. A frame holds closed candles with
  the columns `open_time, open, high, low, close, volume, quote_volume, trade_count,
  taker_buy_base_volume, taker_buy_quote_volume` (see [data-format.md](data-format.md)), oldest
  first, at most `warmup` rows, ending at `time`. Holes are possible (a missing candle is not
  filled in), so test `window["open_time"][-1] == time` when you need the candle of `time`.
- `context`: the same for `context_symbols` (a symbol may be in both). A context symbol with no
  candles gives an empty frame.
- A symbol whose latest candle is more than `stale_after` intervals old (the run config's
  `account.stale_after`, default 2) is left out of `windows`: the data is not trustworthy.
- If none of your traded symbols closed this candle, the platform does not call you and records
  an empty batch.

Warm-up: the first calls after a start see fewer than `warmup` rows for a young symbol. Check
`window.height` before computing something that needs history. The daemon seeds every window
with `warmup` candles from the data folder and Binance's REST API at start-up, so a restart
does not leave a gap.

`on_candles` runs in a worker thread (a slow plugin cannot stall the daemon, but one still
computing misses its deadline and the candle after), so it may be slow but must not share
mutable state without care. It must not do I/O that blocks for long.

### What it returns

A polars frame, one row per signal; an empty frame (`pl.DataFrame()`) means no signals.

| Column | Required | Meaning |
| --- | --- | --- |
| `symbol` | yes | A traded symbol (String). Symbols you do not trade are refused. |
| `priority` | yes | A number, lower first. Signals compete for slots in this order. |
| `action` | no | `"enter"` (default) or `"exit"` (sell the strategy's position in the symbol). |
| `weight` | no | A number > 0, used by `weighted_slots` sizing. |
| `take_profit`, `stop`, `trailing` | no | Fractions overriding the default `exit` for this entry. |
| `max_hold`, `tp_fill` | no | A duration (`timedelta` or `"90m"`) and `"touch"`/`"close"`. |
| anything else | no | Scalars (str, number, bool) kept with the signal and carried into the trade list. |

A null in an exit column keeps the strategy's value for that field. One `enter` and one `exit`
per symbol at most. There is no `open_time` column: signals are for the candle `time`, and the
entry is placed for the open of the next candle, exactly as in the backtest. A frame that breaks
these rules is an error that names the problem (`SignalFrameError`).

## Loading a plugin

A source is `package.module:attr` or `path/to/file.py:attr`.

- A **file path** needs no installation: `source = "strategies/my_model.py:Model"`. The file is
  imported once at start-up; relative paths are relative to the directory you start the daemon
  (or the command) from.
- If `attr` is a class or factory it is called with the config's `params` (`params = { fast =
  12 }` in the run config, `--param fast=12` on the command line; values are JSON when they
  parse as JSON, else text). An instance takes no params.
- A `SignalRule` (the research rules) is wrapped automatically.
- Anything wrong (a missing file, a failing constructor, a missing attribute) is a clear error
  naming the source, before anything runs.

An optional `backtest_signals(root, symbols, until)` method may give a fast batch computation
over history; without it a backtest calls `on_candles` over rolling windows (slower, but exactly
the live signals).

## Testing it

1. **Backtest on stored candles**, no daemon involved:

   ```bash
   uv run moonshot backtest plugin examples/strategies/ma_cross.py:MaCross \
       --symbols BTCUSDT,ETHUSDT,SOLUSDT --param fast=12 --param slow=48 --slots 3
   ```

   It runs `on_candles` over rolling windows (the live signals) and prints the same metrics as
   the other backtests. `--hold`, `--take-profit`, `--stop` and the other options override the
   plugin's `exit`.

2. **Replay the whole engine** over stored candles with a simulated clock: books, sizing,
   account checks, the paper broker and the state file, exactly as live.

   ```bash
   uv run moonshot live replay --config examples/paper.toml \
       --start 2024-01-01 --end 2024-06-01 --state /tmp/replay.sqlite
   ```

   It prints the metrics and writes `trades.parquet`, `equity.csv` and `metrics.json` under
   `results/replay-<timestamp>/`. Use `--state` (or a separate config) so a replay never shares
   a state file with a live run. `--signals backtest` uses `backtest_signals` when the plugin has
   one (faster on big universes).

3. **Paper trade** it with `moonshot live run` once the replay looks right.

## In process or as an external provider

**In process** (the default): the config's strategy has `source = "path/file.py:Class"` and the
daemon loads and calls it. It can only use what the daemon's environment has installed (polars,
numpy, ...).

**External provider**: the plugin runs in its own process, anywhere that can reach the daemon's
HTTP port, with its own environment (torch, a GPU, a model file). The daemon's config lists it
as external, with everything the daemon needs to follow the right candles:

```toml
[[strategies]]
name = "cnn"
source = "external"
interval = "5m"
warmup = 2016
symbols = "pool"                 # what it trades
context_symbols = ["BTCUSDT"]    # extra symbols the plugin reads (its context_symbols)
capital = 3000
sizing = { slots = 5 }
exit = { max_hold = "2h" }       # the plugin's default exit plan: set it here
```

and the provider process runs the **same plugin file, unchanged**:

```bash
export MOONSHOT_API_TOKEN=<your-token>
uv run moonshot provider run strategies/cnn.py:Model --name cnn \
    --url http://daemon-host:8765 --param threshold=0.98
```

`--name` is the strategy's name in the daemon's config. The provider needs this package
installed (it imports the SDK, polars and the contract) next to your own libraries. Or call the
SDK from your own script:

```python
from crypto_moonshot.provider import run_provider_sync

run_provider_sync(Model(), url="http://daemon-host:8765", token=token, name="cnn")
```

What the provider does: reads the strategy's settings from the daemon, fetches the warm-up
windows, follows the tick stream, keeps its own windows, calls your `on_candles` with exactly the
inputs an in-process plugin gets (same stale-symbol rule, same bounded windows), and posts one
batch per candle, empty ones included. It reconnects with backoff when the daemon goes away and
re-fetches its windows; it stops (exit code 1) for a wrong token, an unknown strategy, an
interval or context-symbol mismatch, or a plugin that raises (`--keep-going` logs the failure
and skips that candle instead; the daemon records a missed batch). A batch the daemon refuses is
logged as an error. When a replay served with `--serve` ends, the provider exits 0.

Differences from running in process, worth knowing:

- **Latency counts.** The daemon waits `engine.signal_deadline` (default 30 s) after a candle
  closes for every strategy's batch. A batch that arrives later is *late*: its entries expire
  (an entry well after the open of the next candle no longer matches the backtest), its exits
  still run. The lateness of every batch is recorded.
- A candle that reaches the daemon late for an earlier tick joins the daemon's windows but not
  the provider's until it reconnects.
- The plugin's `exit` and `warmup` come from the daemon's config, not from the plugin object.
- In a replay, the daemon waits for the provider before the first candle and for every batch
  (up to `engine.replay_timeout`), so a replay with an external strategy needs `--serve` and a
  provider.

## The HTTP API, for providers in any language

A provider only needs HTTP and WebSocket. Everything is under `/api/v1` and, except `/health`,
needs the header `Authorization: Bearer <your-token>` (the token is the daemon's
`MOONSHOT_API_TOKEN`; never put it in a URL).

| Endpoint | Purpose |
| --- | --- |
| `GET /strategies/{name}` | `interval`, `symbols`, `warmup`, `context_symbols`, `stale_after`, `exit`, `contract`, `external` |
| `GET /candles?interval=1h&symbols=A,B&limit=500&format=arrow\|json` | the latest closed candles per symbol |
| `WS /ticks?interval=1h` | one message per closed candle of the interval |
| `POST /signals` | one batch (below) |
| `GET /status`, `GET /events?since=&limit=`, `GET /health` | monitoring |

**Loop.** Open `WS /ticks` first, then fetch `/candles` (so no tick is missed between the two),
and keep your own windows. For each `tick` message: append its candles, compute, and `POST` a
batch for `candle_time`: always one, even with no signals.

**Tick stream.** First message: `{"type": "hello", "interval": "1h", "last_tick": ..., "collecting":
"<candle time>" | null, "contract": 1}`. `collecting` is the candle the daemon is waiting on right
now, if its tick was sent before you connected: build it from `/candles` and answer it. Then one
message per candle:

```json
{"type": "tick", "interval": "1h", "candle_time": "2026-11-20T13:00:00+00:00",
 "columns": ["open_time", "open", "high", "low", "close", "volume", "quote_volume",
             "trade_count", "taker_buy_base_volume", "taker_buy_quote_volume"],
 "candles": {"BTCUSDT": ["2026-11-20T13:00:00+00:00", 91000.0, 91200.0, 90800.0, 91100.0,
                         12.5, 1137500.0, 4210, 6.1, 555000.0]},
 "missing": ["XYZUSDT"]}
```

`candles` holds the symbols that closed this candle; `missing` the subscribed ones that did not.
A replay ends with `{"type": "end", "reason": "replay finished"}`. A client that falls too far
behind (64 ticks) is disconnected with close code 1013: reconnect and re-fetch the windows.

**Candles.** `format=arrow` (default) returns an Arrow IPC stream (`application/vnd.apache.arrow.stream`)
with a `symbol` column plus the candle columns; `format=json` returns `{"columns": [...],
"windows": {"BTCUSDT": [[...row...], ...]}, "time": ...}`. `limit` is per symbol (at most 5000).
Fetch `warmup` rows (or a few more).

**A batch** (the signal contract, version 1):

```json
{"contract": 1, "strategy": "cnn", "interval": "5m", "candle_time": "2026-11-20T13:40:00Z",
 "signals": [{"symbol": "SOLUSDT", "action": "enter", "priority": -0.991, "weight": null,
              "exit": {"take_profit": 0.05, "max_hold": "2h"}, "info": {"score": 0.991}}]}
```

`candle_time` is the open time of the candle the signal is for (timezone-aware, aligned to the
interval); the entry is placed for the open of the next one. `priority` orders competing
signals (lower first); `exit` is a partial override of the strategy's exit plan (durations are
`90m`, `2h`, `3d`, `1w`); `info` takes scalars only. `action: "exit"` sells the strategy's
position in that symbol at market.

**Replies to `POST /signals`:**

| Status | Meaning |
| --- | --- |
| `202` | `{"status": "on_time" \| "late", "batch_id", "lateness_seconds"}`: accepted; `late` means its entries expire |
| `409` | `{"status": "duplicate"}`: a batch for this strategy and candle already exists (retrying is safe) |
| `422` | `{"status": "invalid" \| "refused", "reason"}`: the body breaks the contract, or the daemon refuses it (unknown strategy, a strategy that runs in process, another interval, a candle that has not closed) |
| `401` | missing or wrong token |

```bash
curl -s -H "Authorization: Bearer $MOONSHOT_API_TOKEN" \
     -H 'Content-Type: application/json' \
     -d '{"contract":1,"strategy":"cnn","interval":"5m","candle_time":"2026-11-20T13:40:00Z","signals":[]}' \
     http://127.0.0.1:8765/api/v1/signals
```

`GET /api/v1/openapi.json` (with the token) describes the HTTP endpoints. The daemon binds to
`127.0.0.1` by default; to reach it from another machine start it with `--host 0.0.0.0` on a
network you trust (the token is the only protection; there is no TLS: put it behind a reverse
proxy or a VPN if the network is not private).
