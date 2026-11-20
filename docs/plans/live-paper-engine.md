# Strategy-agnostic live engine in paper mode

- **Status:** building
- **Branch:** `worktree-bridge-cse_015bssAUbtXue4wkarka78Pb`
- **PR:**
- **Updated:** 2026-11-20T14:45Z

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

### M3: Market data feed (in progress)

- **Delivers:** combined Binance kline WebSocket (closed candles only), a tick assembled per
  candle time with a timeout, reconnect with backoff (incl. the 24-hour drop), REST backfill of
  missed candles, rolling windows seeded from the store or REST, stale-symbol detection.
- **Done when:** tests against a local fake WebSocket server and fake REST cover disconnects,
  gaps, out-of-order and late symbols.
- **Worker:** sonnet
- **Commits:**

### M4: Paper broker and state store (todo)

- **Delivers:** an incremental paper broker that triggers brackets candle by candle with exactly
  the backtest's rules; a SQLite state file (signals, orders, fills, positions, events; schema
  version, WAL, transactional writes).
- **Done when:** parity test: candles fed one at a time give fills identical to
  `SimulatedBroker`; state round-trip and interrupted-write recovery tests.
- **Worker:** opus
- **Commits:**

### M5: Live engine and replay (todo)

- **Delivers:** tick loop, per-tick signal batches with deadline and expiry, per-strategy books,
  account checks (total exposure, max order size, stale data, pause), pause/flatten, time exits,
  restart recovery, replay over stored candles with a simulated clock.
- **Done when:** replaying the capitulation preset over a stored period gives the same trades as
  the backtest engine; killing a replay partway and restarting gives an identical result.
- **Worker:** opus
- **Commits:**

### M6: Daemon API, provider SDK and CLI (todo)

- **Delivers:** HTTP + WebSocket API (signals in; candles, ticks, status, health out) with token
  auth; a provider SDK running the same plugin class out of process; an example external
  provider; TOML run config; `moonshot live run|replay|status|pause|resume|flatten`, paper only.
- **Done when:** end-to-end test: the daemon replays while an external provider in a separate
  process trades over HTTP, matching the in-process result; unauthenticated requests are refused.
- **Worker:** sonnet
- **Commits:**

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

## Blocked on

## Follow-ups

- P2: web dashboard (positions, orders, P&L, signals, provider/stream health, alert feed),
  pause/resume/flatten controls, Docker image and compose for Proxmox, runbook.
- P3: Binance testnet broker: signed REST, exchange-filter rounding, OCO brackets with trailing
  stops, user data stream, reconciliation; mocks only until testnet keys exist.
- P4: mainnet with small sizes (user holds keys and the go-ahead).
