# Binance testnet broker (built against mocks)

- **Status:** building
- **Branch:** `feat/testnet-broker` (stacked on PR #19, `feat/live-operations`)
- **PR:**
- **Updated:** 2026-11-21T03:35Z

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

### M1: Exchange broker design (done)

- **Delivers:** `docs/architecture.md` "Exchange broker" design (execution layer shared by paper
  and exchange brokers, order lifecycle, brackets as OCO, time exits, fees and dust, user data
  stream, reconciliation, testnet mode); `docs/binance-spot-api.md`, the API reference the
  workers build against, with sources.
- **Done when:** the docs are the spec for M2-M5; check command passes.
- **Worker:** master
- **Commits:** 0e66658

### M2: Signed Binance client (done)

- **Delivers:** signed REST client for the spot testnet (HMAC-SHA256, timestamp and server-time
  offset, recvWindow, `X-MBX-APIKEY`), request weight and order-count pacing, exchange rules
  (`exchangeInfo` filters) with exact `Decimal` rounding, typed errors mapped from Binance codes,
  a fake testnet REST server for tests.
- **Done when:** tests against the fake server cover signing (known vectors), time sync, -1021
  retry, rate limits, filter rounding and rejection, each error mapping, and that keys never
  appear in logs or reprs.
- **Worker:** sonnet
- **Commits:** 1eba174, 28d30cd, 55a6f93, 2cbfa5c

### M3: Execution layer refactor (done)

- **Delivers:** an async execution interface both brokers satisfy (submit, cancel, events,
  reconcile; whether the broker needs candles and manages time exits); the engine commits each
  order's intent before calling the broker and records the result after; engine-run time exits
  for brokers that do not manage them; the paper broker behind an adapter.
- **Done when:** every P1/P2 paper, replay-parity, recovery and API test passes unchanged; new
  tests drive the engine with a fake exchange-style broker (async fills, rejections, timeouts,
  engine-run time exits) and crash recovery between intent and acknowledgement.
- **Worker:** opus
- **Commits:** edccfce, 7df02e6, 5590e40, 0331ec4, 659e603, 6ee614b, a507820

### M4: BinanceBroker (in progress)

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
- 2026-11-21 (M1): exit levels are relative to the testnet fill price (testnet prices drift from
  the mainnet candles); stop and trailing cannot both be live on Binance, so a plan with both is
  accepted only when `stop >= trailing` (trailing alone is then equivalent); `tp_fill = "close"`
  refused in exchange modes; dust stays on the position; unprotected positions alert and retry;
  REST polling backs up the user data stream; reconciliation never trades on its own except to
  finish a provable time exit. New modules: `exchange/` (signed client, rules, errors),
  `broker/execution.py`, `broker/binance.py`.
- 2026-11-21 (M2): `exchange/` = `client.py` (`SignedClient`), `rules.py`, `errors.py`, `models.py`,
  `pacing.py`. Order POST/DELETE never retried (raise `OrderStatusUnknown`; query by client ID).
  httpx logs signed URLs at INFO, so the client filters any record containing `signature=`.
  Unverified until a real testnet run: request weights, exact -2010 message wording, order-count
  header behaviour.
- 2026-11-21 (M3): `ExecutionBroker` (submit/cancel/query/events/reconcile; flags `needs_candles`,
  `manages_time_exits`, `sell_cancels_bracket`); `PaperExecution` adapter; state schema v3 order
  states `new` -> `pending`|`unknown` -> final. Unknown outcomes are queried at the next broker
  step; a never-placed entry is rejected (`not_placed`), a never-placed sell is re-submitted under
  the same ID. A reconcile mismatch pauses entries until resumed by hand. New alerts:
  `orphan_order`, `balance_mismatch`, `unprotected_position:<symbol>`. All paper/replay/recovery
  expectations unchanged; the real-data F2 replay still equals the backtest exactly.
- 2026-11-21 (M3 review): the engine reads broker events only in each tick's broker step, so an
  exchange fill could reach the state file up to one interval late; M4 adds a drain between ticks.
- 2026-11-21: If the user provides testnet keys, add an M6: a small real testnet run (stop and ask
  for the go-ahead first).

## Blocked on

## Follow-ups
