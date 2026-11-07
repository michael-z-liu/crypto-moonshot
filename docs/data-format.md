# Stored data format

This is the contract between the code that writes market data (`moonshot data sync` and
`moonshot data import`) and everything that reads it (backtests, visualisation, notebooks). It is
schema version **1**. Readers should depend only on what is written here, not on the writer's
internals. A change to anything below bumps the schema version and updates this file in the same
PR.

## Where the data lives

The data root is `$MOONSHOT_DATA_DIR`, default `./data` relative to the working directory. It is
gitignored and never committed.

```
<data root>/
└── klines/
    └── spot/
        └── interval=<interval>/          e.g. interval=1h
            └── symbol=<SYMBOL>/          e.g. symbol=BTCUSDT
                ├── _meta.json            listing date and known gaps (see below)
                ├── 2020-01.parquet       candles whose open_time is in January 2020
                ├── 2020-02.parquet
                └── …
```

- `<interval>` is a Binance spot interval: `1s 1m 3m 5m 15m 30m 1h 2h 4h 6h 8h 12h 1d 3d 1w 1M`
  (`crypto_moonshot.models.Interval`). Note `1m` is one minute and `1M` one month.
- `<SYMBOL>` is the Binance symbol in upper case, base asset then quote asset, e.g. `ETHUSDT`.
- One Parquet file per symbol, interval and calendar month (UTC), named `YYYY-MM.parquet`. A candle
  belongs to the month of its `open_time` (a `1w` candle opening on 29 January is in the January
  file).
- The folder names are Hive-style, so readers that support Hive partitioning (polars, DuckDB)
  expose `interval` and `symbol` as string columns.

### Symbol lists and delisted coins

A data root may hold plain-text symbol lists (one symbol per line) next to `klines/`:

- `top200.txt`: the pairs fetched with `moonshot data sync` (today's top USDT pairs by volume).
- `delisted.txt`: USDT pairs Binance has delisted (status `BREAK`), e.g. `MATICUSDT`, `WAVESUSDT`.
- `outside_pool.txt`: USDT pairs still listed but not in `top200.txt`, including coins that
  collapsed and still trade (`LUNAUSDT` from 2020-08, `FTTUSDT`).

`moonshot data sync` cannot fetch delisted pairs (the live API has no history for them). The
second and third lists come from Binance's public archive, data.binance.vision, via
`research/volume_spike/fetch_archive.py` (`groups`, then `fetch`, then `publish`, which runs
`moonshot data import` into the data root and writes the lists). They hold 5m candles from 2020-01
to 2026-10, stored in the same format as everything else. A delisted series simply ends at the
delisting; the archive has its own gaps (`moonshot data status` shows them).

## Candle columns (schema v1)

| Column                   | Type                       | Meaning                                                  |
| ------------------------ | -------------------------- | -------------------------------------------------------- |
| `open_time`              | `Datetime("us", "UTC")`    | Candle open time, timezone-aware UTC                     |
| `open`                   | `Float64`                  | First trade price, in the quote asset (USDT for `…USDT`) |
| `high`                   | `Float64`                  | Highest trade price                                      |
| `low`                    | `Float64`                  | Lowest trade price                                       |
| `close`                  | `Float64`                  | Last trade price                                         |
| `volume`                 | `Float64`                  | Traded volume in the base asset (BTC for `BTCUSDT`)      |
| `quote_volume`           | `Float64`                  | Traded volume in the quote asset                         |
| `trade_count`            | `Int64`                    | Number of trades                                         |
| `taker_buy_base_volume`  | `Float64`                  | Base-asset volume bought by takers                       |
| `taker_buy_quote_volume` | `Float64`                  | Quote-asset volume bought by takers                      |

Columns appear in this order. The same definition is in code as
`crypto_moonshot.models.KLINE_SCHEMA`, with `SCHEMA_VERSION`.

**Invariants** that hold for every file:

- Rows are sorted by `open_time`, and `open_time` is unique within a series.
- Only closed candles are stored. A candle closes at the next interval's open time; there is no
  `close_time` column (it is `open_time + interval`).
- Every `open_time` is aligned to the interval: a multiple of the interval length since the Unix
  epoch, except `1w` (Mondays 00:00 UTC) and `1M` (the 1st of the month 00:00 UTC).
- Each file carries Parquet key-value metadata `moonshot.schema_version = "1"`.
- A missing candle means Binance has none (an exchange outage, recorded in `_meta.json`) or it has
  not been downloaded yet. Readers must not assume a series has no holes.

## `_meta.json`

One per series (symbol and interval):

```json
{
  "schema_version": 1,
  "listing": "2020-08-11T06:00:00+00:00",
  "gaps": [
    ["2020-02-19T12:00:00+00:00", "2020-02-19T17:00:00+00:00"]
  ]
}
```

- `listing`: the first open time Binance has for this symbol and interval, or `null` if not
  looked up yet. Data never starts earlier than this.
- `gaps`: half-open `[start, end)` ranges of open times where Binance has no candles. Sorted and
  non-overlapping. Useful for shading outages on a chart instead of drawing them as flat lines.
- Times are ISO-8601 strings with a `+00:00` offset.

## Writes and concurrent readers

- Only `moonshot data sync` and `moonshot data import` write under `klines/`. Everything else is
  read-only.
- Finished months never change. The current month's file is rewritten as new candles close.
- Every write goes to a temporary file in the same folder, then is atomically renamed into place.
  So a reader sees either the old file or the new one, never half a file. Temporary files start
  with `.` and do not end in `.parquet`, so `*.parquet` globs skip them.

## Reading it

polars, one series:

```python
import polars as pl

btc = pl.scan_parquet("data/klines/spot/interval=1h/symbol=BTCUSDT/*.parquet").collect()
```

polars, every series as one table with `interval` and `symbol` columns:

```python
candles = pl.scan_parquet("data/klines/spot/**/*.parquet", hive_partitioning=True)
daily_btc = candles.filter(pl.col("symbol") == "BTCUSDT", pl.col("interval") == "1d").collect()
```

DuckDB:

```sql
SELECT symbol, interval, open_time, close
FROM read_parquet('data/klines/spot/*/*/*.parquet', hive_partitioning = true)
WHERE symbol = 'ETHUSDT' AND interval = '1h'
ORDER BY open_time;
```

## Score files

Precomputed model scores, read by `crypto_moonshot.strategy.scores` (`ScoreStrategy`), are a
separate format, not part of the kline store and not versioned with it. One Parquet file, one
row per scored candle:

| Column      | Type                    | Meaning                                                   |
| ----------- | ----------------------- | --------------------------------------------------------- |
| `symbol`    | `String`                | Binance symbol, e.g. `BTCUSDT`                            |
| `open_time` | `Datetime("us", "UTC")` | Open time of the signal candle, scored on its close       |
| `score`     | `Float64`               | Higher is better; null or NaN means not scored            |
| any other   | any                     | Carried into the signals and the backtest's trade list    |

`(symbol, open_time)` is unique. Other time units or zones are converted to µs UTC on reading.
`convert_wf_npz` writes this format from the volume-spike research's walk-forward outputs
(`cache/volume_spike/wf/<variant>.npz` with `seq2_meta.parquet`): `score` is the npz's `pct`,
with the meta's `event` and the npz's other per-row arrays (such as `raw`) as extra columns.
