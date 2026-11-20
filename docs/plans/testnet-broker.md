# Binance testnet broker (built against mocks)

- **Status:** building
- **Branch:** `feat/testnet-broker` (stacked on PR #19, `feat/live-operations`)
- **PR:**
- **Updated:** 2026-11-21T01:45Z

## Goal

P3 of the trading platform (P1 PR #18 paper engine, P2 PR #19 operations). Let the daemon trade
on the Binance spot testnet: a signed REST client, an exchange broker that places market entries
with OCO bracket exits, follows fills over the WebSocket API user data stream, runs time exits
and reconciles with the exchange after a restart. The user has no testnet keys yet, so everything
is built and tested against a fake exchange; a real testnet run needs the user's keys and an
explicit go-ahead.

## Out of scope

- Any request to the real testnet or mainnet, and reading `.env` or real keys (stop condition).
- `mode = "live"` (mainnet) stays refused (P4).
- Push alerts, TLS (as P2).

## Milestones

### M1: Exchange broker design (todo)

- **Delivers:** `docs/architecture.md` "Exchange broker" design (execution layer shared by paper
  and exchange brokers, order lifecycle, brackets as OCO, time exits, fees and dust, user data
  stream, reconciliation, testnet mode); `docs/binance-spot-api.md`, the API reference the
  workers build against, with sources.
- **Done when:** the docs are the spec for M2-M5; check command passes.
- **Worker:** master
- **Commits:**

### M2: Signed Binance client (todo)

- **Delivers:** signed REST client for the spot testnet (HMAC-SHA256, timestamp and server-time
  offset, recvWindow, `X-MBX-APIKEY`), request weight and order-count pacing, exchange rules
  (`exchangeInfo` filters) with exact `Decimal` rounding, typed errors mapped from Binance codes,
  a fake testnet REST server for tests.
- **Done when:** tests against the fake server cover signing (known vectors), time sync, -1021
  retry, rate limits, filter rounding and rejection, each error mapping, and that keys never
  appear in logs or reprs.
- **Worker:** sonnet
- **Commits:**

### M3: Execution layer refactor (todo)

- **Delivers:** an async execution interface both brokers satisfy (submit, cancel, events,
  reconcile; whether the broker needs candles and manages time exits); the engine commits each
  order's intent before calling the broker and records the result after; engine-run time exits
  for brokers that do not manage them; the paper broker behind an adapter.
- **Done when:** every P1/P2 paper, replay-parity, recovery and API test passes unchanged; new
  tests drive the engine with a fake exchange-style broker (async fills, rejections, timeouts,
  engine-run time exits) and crash recovery between intent and acknowledgement.
- **Worker:** opus
- **Commits:**

### M4: BinanceBroker (todo)

- **Delivers:** market entries by quote amount, brackets as OCO (take-profit `LIMIT_MAKER` plus
  `STOP_LOSS`, trailing via `trailingDelta`), single-leg exits, immediate-trigger fallbacks,
  fees and dust, the WebSocket API user data stream (`userDataStream.subscribe.signature`) with
  REST polling as fallback, reconciliation at start and after reconnects; a fake exchange
  simulator (REST plus WebSocket API, a price path, matching, OCO, trailing, events, faults).
- **Done when:** simulator tests cover entry fills, OCO take profit / stop / trailing, single
  legs, gap-through levels at placement, partial fills, rejections (filters, balance), -1007
  timeouts, WS disconnects with missed events, a restart mid-order, orphaned and unknown orders,
  balance mismatches.
- **Worker:** opus
- **Commits:**

### M5: Testnet mode (todo)

- **Delivers:** `mode = "testnet"` runnable: keys from `MOONSHOT_BINANCE_TESTNET_API_KEY` /
  `_SECRET` (environment or `.env`, never logged), testnet URLs configurable, start-up checks
  (keys, symbols listed on testnet, balances cover allocated capital), status/dashboard showing
  exchange state, `.env.example`, runbook section for the first real testnet run.
- **Done when:** end to end: the daemon in testnet mode against the fake exchange (driven by a
  replayed candle path) trades consistently with paper on the same candles; testnet refuses to
  start without keys; mainnet still refused; keys never in logs, events, status or errors.
- **Worker:** sonnet
- **Commits:**

## Decisions

- 2026-11-21: Binance retired the spot REST `listenKey` user data stream (removal 2026-02-20); use
  the WebSocket API `userDataStream.subscribe.signature` (HMAC-signed, events wrapped as
  `{subscriptionId, event}`), documented for `wss://ws-api.testnet.binance.vision/ws-api/v3`.
- 2026-11-21: Testnet keys env names `MOONSHOT_BINANCE_TESTNET_API_KEY` / `_SECRET`; mainnet keys
  keep `MOONSHOT_BINANCE_API_KEY` / `_SECRET` (P4), so the two can never be confused.
- 2026-11-21: If the user provides testnet keys, add an M6: a small real testnet run (stop and ask
  for the go-ahead first).

## Blocked on

## Follow-ups
