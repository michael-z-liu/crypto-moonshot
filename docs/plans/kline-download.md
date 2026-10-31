# Kline download

- **Status:** building
- **Branch:** `claude/kline-download-functionality-cc0cd7`
- **PR:**
- **Updated:** 2026-10-31T11:49Z

## Goal

A simple `moonshot data` CLI, usable by the user or by an agent, that downloads Binance spot klines
into the local Parquet store, fetching only what is missing. Downloads run in parallel across
symbols and months, with clear per-download progress (rich bars in a terminal, plain lines and
`--json` for agents). Data from another folder can be imported after structural checks, so it
doesn't have to be downloaded again. Step 2 of the build order in `docs/architecture.md`, plus the
parts of step 1 it needs.

## Out of scope

- Futures, WebSocket streaming, trades/aggTrades
- Bulk archives from `data.binance.vision`
- Symbol lists from a config file; an "all USDT pairs" option
- DuckDB query layer
- Upgrading data from older schema versions: files that fail import checks are skipped and
  re-downloaded by the next sync
- API keys and `.env` loading (not needed for public market data; comes with the broker work)

## Design notes

Shared by all milestones. Workers may refine the details; record changes under *Decisions*.

**Layout.** `models.py` (Interval), `config.py` (Settings), `data/binance.py` (REST client),
`data/store.py`, `data/sync.py`, `data/importer.py`, `progress.py` (renderers), `cli.py`.

**Interval.** `StrEnum` of every Binance spot interval: 1s 1m 3m 5m 15m 30m 1h 2h 4h 6h 8h 12h
1d 3d 1w 1M. Provides `floor(dt)` and `next(dt)` / candle open-time alignment for each (1w opens on
Monday 00:00 UTC; 1M on the 1st of the month; others are multiples of their length since the
epoch), and a way to count/enumerate expected open times in a range.

**Config.** `pydantic-settings`, env prefix `MOONSHOT_`, environment variables only (no `.env` in
this PR): `MOONSHOT_DATA_DIR` (default `./data`), `MOONSHOT_BINANCE_BASE_URL` (default
`https://data-api.binance.vision`, Binance's public market-data-only host).

**Schema v1** (one Parquet file per symbol, interval and month):
`open_time` Datetime(us, UTC), `open`, `high`, `low`, `close`, `volume`, `quote_volume` Float64,
`trade_count` Int64, `taker_buy_base_volume`, `taker_buy_quote_volume` Float64. Sorted by
`open_time`, unique, closed candles only. File key-value metadata `moonshot.schema_version = "1"`.
Path: `<data_dir>/klines/spot/interval=<i>/symbol=<SYM>/<YYYY-MM>.parquet`. Writes go to a temp
file in the same directory, then `os.replace`.

**Per-series metadata.** `<series dir>/_meta.json`: schema version, the first open time Binance
has for the symbol (listing), and known gaps as `[start, end)` open-time ranges where Binance
returned no candles. Sync never refetches the pre-listing range or recorded gaps.

**REST client.** async `httpx`. `GET /api/v3/klines` with `startTime`, `endTime`, `limit=1000`,
paging until done; weight 2 per request. Listing date = first candle from `startTime=0, limit=1`.
Only closed candles are returned to callers (`open_time + interval <= now`).

**Rate-limit safety** (the user asked for this explicitly, to avoid an IP ban):
- Track `X-MBX-USED-WEIGHT-1M` from every response; keep usage under 50% of the 6,000/minute IP
  limit, pausing until the next minute when the budget would be exceeded.
- 429: wait the full `Retry-After`, then retry; count it. After `max_rate_limit_hits` (default 3)
  in one run, abort the whole run.
- 418 (IP ban): abort the whole run immediately, never retry.
- 5xx and network errors: retry with exponential backoff, a few times. Other 4xx (for example an
  unknown symbol, code -1121): fail that download only, no retry.

**Sync.** For each (symbol, interval): desired range = `[max(start, listing), min(end, last
closed open time + 1))`; missing = desired minus stored candles minus recorded gaps, grouped by
month. Each month with anything missing is one unit of work: fetch the span, merge with the
existing file, write atomically, record any gaps found inside fetched spans. Units run concurrently
across symbols and months under a global concurrency limit (default 8) and the shared rate
limiter. A failing (symbol, interval) does not stop the others; a rate-limit abort or ban cancels
everything. Ctrl-C leaves completed months intact. Progress is reported as events to a sink
protocol, independent of rendering.

**CLI.**
`moonshot data sync SYMBOL... [-i 5m]... [--start 2020-01-01] [--end now] [--concurrency 8]
[--max-rate-limit-hits 3] [--data-dir PATH] [--json] [-v|-q]`,
`moonshot data status [SYMBOL...] [-i ...] [--data-dir] [--json]`,
`moonshot data import SOURCE_DATA_DIR [--dry-run] [--data-dir] [--json]`.
In a terminal: a rich progress row per (symbol, interval) with candles done/total, months, rate,
ETA, status, plus an overall row and a summary table. Without a TTY: a plain progress line per
active download every few seconds, plus a line per finish/failure/rate-limit hit, on stderr.
`--json` prints a machine-readable summary to stdout. Exit codes: 0 ok, 1 some downloads failed,
2 usage error, 3 aborted on rate limits or ban.

**Import.** Each `*.parquet` under the source's `klines/` tree is imported only if all checks pass,
otherwise skipped with its reason: path matches the layout with a valid interval, symbol and month;
schema version equals the current one; column names and dtypes match; open times are UTC, inside
the file's month and aligned to the interval; no duplicates; closed candles only. Overlapping months
are merged, existing candles win. Source `_meta.json` gaps/listing are merged in if valid. The
source is never modified.

## Milestones

### M1: Models, config and REST client (in progress)

- **Delivers:** `Interval`, `Settings`, dependencies (httpx, polars, pydantic-settings, rich), and
  the async Binance klines client with paging, closed-candle filtering, listing-date lookup and
  the rate-limit safety rules above.
- **Done when:** tests with mocked HTTP (`httpx.MockTransport`, no network) cover interval
  alignment for fixed, weekly and monthly intervals; paging across several pages; dropping the
  still-open candle; listing-date lookup; pausing near the weight budget; 429 retry with
  Retry-After and abort after N hits; immediate abort on 418; 5xx retry; unknown-symbol error.
  All datetimes UTC. Check command passes.
- **Worker:** sonnet
- **Commits:**

### M2: Parquet store and gaps (todo)

- **Delivers:** `data/store.py`: read/write monthly files with schema v1 and version metadata,
  atomic writes, merge with de-duplication, stored open times / coverage for a series, `_meta.json`
  read/write (listing, gaps), and missing-range calculation grouped by month.
- **Done when:** tests cover write/read round trip including metadata and dtypes; atomic write
  (no partial file left on failure); idempotent merge; missing ranges with and without gaps and
  listing date; month grouping across year boundaries. Check command passes.
- **Worker:** sonnet
- **Commits:**

### M3: Parallel sync engine (todo)

- **Delivers:** `data/sync.py`: plans missing months per (symbol, interval), runs them
  concurrently under a global limit and the shared rate limiter, writes each month on completion,
  records gaps and listing date, emits progress events, isolates per-series failures, propagates
  rate-limit aborts/bans, and returns a per-series result summary.
- **Done when:** tests with a fake client show: units run concurrently (max in-flight > 1, never
  above the limit); a second run makes zero requests; only months with missing candles (including
  the current month) are fetched; gaps are recorded and not refetched; pre-listing range not
  requested; one failing symbol leaves the others complete; abort cancels all and keeps finished
  months; events arrive in a sensible order. Check command passes.
- **Worker:** opus
- **Commits:**

### M4: CLI and progress display (todo)

- **Delivers:** `moonshot data sync` and `moonshot data status` with rich progress (TTY), plain
  periodic lines (no TTY), `--json`, `-v`/`-q`, `--concurrency`, `--max-rate-limit-hits`,
  `--data-dir`, and the exit codes above.
- **Done when:** CliRunner tests (fake client injected, no network) cover plain non-TTY output,
  the JSON summary format, status table and JSON, bad interval / bad date / bad symbol handling,
  exit codes for failure and abort. Check command passes.
- **Worker:** sonnet
- **Commits:**

### M5: Import from another data folder (todo)

- **Delivers:** `data/importer.py` and `moonshot data import` with the checks above, `--dry-run`
  and `--json`.
- **Done when:** tests cover a valid folder; each failing check skipping the file with its reason;
  merge with existing months (existing wins); dry run writes nothing; source untouched; JSON
  summary. Check command passes.
- **Worker:** sonnet
- **Commits:**

### M6: Docs (todo)

- **Delivers:** `docs/architecture.md` updated (endpoint, rate-limit safety, concurrency, progress,
  `_meta.json`, schema version and import), README usage for `moonshot data`.
- **Done when:** docs match the code. Check command passes.
- **Worker:** master
- **Commits:**

### M7: Real run, BTCUSDT 1h from 2020 (todo)

- **Delivers:** `moonshot data sync BTCUSDT -i 1h --start 2020-01-01 --concurrency 4` against the
  live public API (about 60 requests).
- **Done when:** row count equals closed hours since 2020-01-01 minus recorded gaps; no
  duplicates; open times strictly increasing UTC; a second run fetches only new candles;
  `data status` matches the files. **Stop** (blocked) on any 418, a second 429 across M7–M8, or
  used weight above 50% of the limit.
- **Worker:** master
- **Commits:**

### M8: Real run, top 5 in parallel at 1h, plus import check (todo)

- **Delivers:** `moonshot data sync BTCUSDT ETHUSDT BNBUSDT SOLUSDT XRPUSDT -i 1h --start
  2020-01-01 --concurrency 4` (about 300 requests); then `data import ./data` into an empty scratch
  data dir.
- **Done when:** M7's data checks pass for every symbol; SOLUSDT starts at its listing date; the
  output shows parallel progress; the imported copy's `data status` matches the original. Same stop
  rules as M7.
- **Worker:** master
- **Commits:**

## Decisions

- 2026-10-31: REST from `data-api.binance.vision` (public market data only), not bulk archives.
- 2026-10-31: The user approved real downloads of public market data from Binance for M7 and M8
  (no API keys, read-only, into the gitignored `./data/`).
- 2026-10-31: Top 5 = BTC, ETH, BNB, SOL, XRP against USDT (largest non-stablecoin pairs).
- 2026-10-31: Import has no schema upgrades; failing files are skipped (user request).
- 2026-10-31: Config reads environment variables only in this PR; `.env` and keys come later.

## Blocked on

## Follow-ups
