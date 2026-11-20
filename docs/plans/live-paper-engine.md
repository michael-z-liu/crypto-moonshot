# Strategy-agnostic live engine in paper mode

- **Status:** review
- **Branch:** `worktree-bridge-cse_015bssAUbtXue4wkarka78Pb`
- **PR:** https://github.com/michael-z-liu/crypto-moonshot/pull/18
- **Updated:** 2026-11-20T18:10Z

## Goal

Build the trading side of the platform alongside the strategy research. P1 is a trading daemon
that knows nothing about specific strategies: it runs any number of them in paper mode on live
Binance candles, survives restarts without losing or duplicating positions, and can be replayed
over stored history to prove it matches the backtest. Strategies are built outside the
platform and plug in without rebuilding it. Heavy computation (e.g. a torch CNN) runs in a
separate provider process, not in the trader.

## Out of scope

- Web UI, alerts (UI only, no push), Docker image (P2).
- Any Binance order placement, testnet or mainnet (P3, P4). No API keys are used.
- Model training; shorts; the "targets" strategy style (expressible as enter/exit signals).
- Writing live candles into the Parquet store (only sync and import write candles).

## Milestones

### M1: Architecture revision (done)

- **Delivers:** `docs/architecture.md` rewritten for the live side: daemon / provider split, the
  signal contract (v1), plugin loading, per-strategy books and account checks, the SQLite state
  file, recovery, replay, the HTTP/WebSocket API, the revised build order and package layout.
- **Done when:** the doc is a complete spec for M2-M6; check command passes.
- **Worker:** master
- **Commits:** 841ecfc

### M2: Plugin contract (done)

- **Delivers:** the live strategy protocol and loader (`module:attr` or `path/file.py:attr`, with
  params), the adapter for existing `SignalRule`s (capitulation), `moonshot backtest plugin
  <spec>` and `moonshot backtest signals <file>`.
- **Done when:** tests show a plugin loads from a temporary file path; running the capitulation
  adapter candle by candle over a rolling window gives the same signals as `compute_signals`;
  both CLI commands produce results on fake data.
- **Worker:** sonnet
- **Commits:** a2bbed8, fed9191, 54011ba, 99fde41, 8754178, ef9b290, 3bddad3, 35f2474

### M3: Market data feed (done)

- **Delivers:** combined Binance kline WebSocket (closed candles only), a tick assembled per
  candle time with a timeout, reconnect with backoff (incl. the 24-hour drop), REST backfill of
  missed candles, rolling windows seeded from the store or REST, stale-symbol detection.
- **Done when:** tests against a local fake WebSocket server and fake REST cover disconnects,
  gaps, out-of-order and late symbols.
- **Worker:** sonnet
- **Commits:** 15644ec, b492336, a0d4652, bff02cc

### M4: Paper broker and state store (done)

- **Delivers:** an incremental paper broker that triggers brackets candle by candle with exactly
  the backtest's rules; a SQLite state file (signals, orders, fills, positions, events; schema
  version, WAL, transactional writes).
- **Done when:** parity test: candles fed one at a time give fills identical to
  `SimulatedBroker`; state round-trip and interrupted-write recovery tests.
- **Worker:** opus
- **Commits:** ff4940e, 6ebfbc7, 809fe5a, 9d71637, ba6b81f, 90b9dda

### M5: Live engine and replay (done)

- **Delivers:** tick loop, per-tick signal batches with deadline and expiry, per-strategy books,
  account checks (total exposure, max order size, stale data, pause), pause/flatten, time exits,
  restart recovery, replay over stored candles with a simulated clock.
- **Done when:** replaying the capitulation preset over a stored period gives the same trades as
  the backtest engine; killing a replay partway and restarting gives an identical result.
- **Worker:** opus
- **Commits:** f6b9c76, 2db1218, df56714, f179db3, fdc2523, 0e1922b, f8fa7b7, 0688c1c, 80b5002, 8c15316, 2b52010

### M6: Daemon API, provider SDK and CLI (done)

- **Delivers:** HTTP + WebSocket API (signals in; candles, ticks, status, health out) with token
  auth; a provider SDK running the same plugin class out of process; an example external
  provider; TOML run config; `moonshot live run|replay|status|pause|resume|flatten`, paper only.
- **Done when:** end-to-end test: the daemon replays while an external provider in a separate
  process trades over HTTP, matching the in-process result; unauthenticated requests are refused.
- **Worker:** sonnet
- **Commits:** 5396f68, b8ca957, cf405e8, ab724a2, 0811257, a5900cd, cb5d51b, 6e284e4, 3f303fb, 0333f37, c0a6b6f, 1b0d705 (master)

## Decisions

- 2026-11-20: Split into projects P1 (this), P2 operate (UI, controls, Docker for Proxmox,
  runbook), P3 testnet broker against mocks, P4 mainnet small sizes. User approved.
- 2026-11-20: Deployment target is Docker on the user's Proxmox host, UI on the LAN (user).
- 2026-11-20: Strategies plug in both in-process and as external processes over HTTP, one
  contract (user).
- 2026-11-20: No testnet keys for now; P3 is built against mocks only (user).
- 2026-11-20: Alerts appear in the UI only, no push channel (user).
- 2026-11-20: Signals arrive in one batch per strategy per closed candle (empty batches too);
  entries expire after the open of t+1 plus a grace period; lateness is recorded.
- 2026-11-20: Each strategy has its own book (capital, sizing, risk); account-wide checks sit on
  top; only one strategy holds a symbol at a time.
- 2026-11-20: Strategies may emit exit signals as well as relying on bracket exits.
- 2026-11-20: Architecture (M1): the daemon is the single market-data hub; the paper broker calls
  `resolve_exit` on the candles seen so far (parity by construction); books are derived from the
  state file, not stored; client order IDs are deterministic; API is FastAPI under `/api/v1`
  with a bearer token from `MOONSHOT_API_TOKEN`; deps added later: websockets, fastapi, uvicorn.
- 2026-11-20: `on_candles` takes traded symbols' windows and context windows separately (M2), so
  `RuleStrategy` knows which symbols to compute candidates for.
- 2026-11-20 (M2): `RuleStrategy` warm-up defaults to `MIN_HISTORY` (8 days, 2304 5m candles).
  Known edge: the full-history computation signals on a symbol's candles 2016-2303 (days 7-8
  after listing) once the series ever reaches 8 days; a live window cannot, so those rare
  signals appear in backtests only. Documented; excluded from the parity test only.
- 2026-11-20 (M2): `on_candles` frames have no `open_time`; exit-override columns are partial
  (null keeps the strategy's value); backtests of plugins keep only entries and apply the single
  config exit plan (per-signal overrides apply live only).
- 2026-11-20 (M3): `Feed` protocol: `ticks()`, `windows()`, `last_candle()`, `stale_symbols()`,
  `health()`, `aclose()`; `Tick.candles` are named tuples. Not every candle time gets a tick
  (a fully-down interval or an empty backfill); windows hold `max(warmup) + 32` candles; a
  failed backfill leaves a hole and is reported, not retried. `silence_timeout` off by default.
- 2026-11-20 (M3, for M5): the REST client aborts for good after 3 rate-limit hits (a sync-run
  rule); the daemon must configure or recreate it so backfill keeps working.
- 2026-11-20 (M4): `resolve_exit` returns a TIME exit on any window not marked ended, so the
  paper broker applies a TIME exit only once advanced past the candle-aligned deadline; stop and
  take-profit exits as soon as their candle closes. `resolve_exit` unchanged. Parity proven over
  random paths and delivery orders.
- 2026-11-20 (M4): a pending buy with no candle at all stays pending (the engine cancels it after
  a grace period); a hole at the entry candle cancels it (`no_entry_candle`). Cancelled entries
  never count in books. One pending/open position per symbol across strategies (unique index).
- 2026-11-20 (M4): known backtest optimism, not changed (golden tests pin it): the backtest frees
  cash, slot and symbol from an exit at exactly the next entry time, e.g. a stop gapped through at
  the open of t+1; live learns of it only when t+1 closes. Time exits at that instant are known
  in time. Replay parity must document and test this divergence.
- 2026-11-20 (M5): replay equals the backtest exactly with fixed stakes and to 1e-9 with equal
  slots (books are re-derived from SQL after every change so a resumed run is bit-identical);
  each documented divergence has its own test; golden backtest tests still pass on real data.
- 2026-11-20 (M5): costs are account-wide (one simulated exchange); refusal order is paused,
  unknown_symbol, symbol_held, stale, sizing, risk checks, cash, max_order_usdt,
  max_total_exposure; ties across strategies by priority, strategy name, symbol. New engine
  settings `entry_grace` (5m) and `replay_timeout` (1h). A tick emitted after the signal deadline
  (e.g. after a long backfill) is decided at once, so in-process plugins miss that candle.
- 2026-11-20 (M5): the paper broker's `place_order` runs inside the decisions transaction (it is
  rebuilt from the state file); P3's exchange broker must commit the intent first.
- 2026-11-20 (M6): the API token (`MOONSHOT_API_TOKEN`) is required to start the daemon and is
  accepted in the `Authorization` header only (never a query string); uvicorn access logs off.
  The WebSocket hello carries the candle being collected so a provider can join mid-tick; a
  served replay sends `end`. A plugin exception stops a provider unless `--keep-going`.
  Providers take `warmup`, `exit` and `stale_after` from the daemon's `/strategies/{name}`.
- 2026-11-20 (M6, master): `API_PREFIX` moved to `live/contract.py` so providers don't need
  fastapi/uvicorn; live state files gitignored.
- 2026-11-20 (hand checks): F2 replay over Q1 2024 on the real data folder gives the same 26
  trades as `moonshot backtest f2` (prices identical, P&L within 5e-13; 26,208 candles in 40 s).
  A real `moonshot live run` on public Binance data (1m example plugin) seeded, connected,
  processed 5 consecutive on-time ticks and stopped gracefully on SIGTERM; token never logged.
- 2026-11-20 (hand-off): CI exposed two test-only issues (a provider-counter race and CLI error
  panels wrapping at 80 columns); fixed in tests, CI green.

## Blocked on

## Follow-ups

- From the build: an HTTP 418 (IP ban) still disables the daemon's REST client until restart;
  providers miss late candles until they reconnect; no TLS on the API (LAN + token only); the
  suite now takes ~50 s (the e2e test is 8 s); several intervals in one run and partial sells are
  untested; a backtest-only optimism (exit gapped at the next entry's open) is documented, not
  changed.

- P2: web dashboard (positions, orders, P&L, signals, provider/stream health, alert feed),
  pause/resume/flatten controls, Docker image and compose for Proxmox, runbook.
- P3: Binance testnet broker: signed REST, exchange-filter rounding, OCO brackets with trailing
  stops, user data stream, reconciliation; mocks only until testnet keys exist.
- P4: mainnet with small sizes (user holds keys and the go-ahead).
