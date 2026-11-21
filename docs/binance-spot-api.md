# Binance spot API: what the exchange broker relies on

A working reference for `broker/binance.py` and its tests, read from Binance's official docs on
2026-11-21. It records only what the exchange broker uses. Binance changes its API; when this page
and the official docs disagree, the docs win and this page is updated. Points marked
*unverified* were not stated on the pages read and must be confirmed against the real testnet
before relying on them in mainnet trading.

Sources:
[testnet general info](https://developers.binance.com/en/docs/products/spot/testnet/general-info),
[testnet trading endpoints](https://developers.binance.com/docs/binance-spot-api-docs/testnet/rest-api/trading-endpoints),
[account endpoints](https://developers.binance.com/docs/binance-spot-api-docs/rest-api/account-endpoints),
[WebSocket API user data stream requests](https://developers.binance.com/docs/binance-spot-api-docs/websocket-api/user-Data-Stream-requests),
[testnet user data stream events](https://developers.binance.com/en/docs/products/spot/testnet/user-data-stream),
[testnet errors](https://developers.binance.com/en/docs/products/spot/testnet/errors),
[deprecated listenKey endpoints](https://developers.binance.com/docs/binance-spot-api-docs/rest-api/user-data-stream-endpoints-deprecated).

## Hosts (spot testnet)

| Use | URL |
| --- | --- |
| REST | `https://testnet.binance.vision/api` (also `https://api1.testnet.binance.vision/api`) |
| WebSocket API (orders, user data stream) | `wss://ws-api.testnet.binance.vision/ws-api/v3` (also `:9443`) |
| Market streams | `wss://stream.testnet.binance.vision/stream` (not used: candles come from mainnet public data) |

- Only `/api` exists on testnet (no `/sapi`). Funds are virtual. The testnet is reset about once a
  month without notice; API keys are kept, balances and orders are not.
- Filters, IP limits and order rate limits are "generally the same" as production; testnet lists
  far fewer symbols than mainnet, and its prices drift from mainnet's.
- WebSocket connections last at most 24 hours; the server pings every 20 s and drops a client that
  does not answer within a minute. A connection attempt costs 2 weight.

## Signing (REST)

- Header `X-MBX-APIKEY: <key>`. Parameters `timestamp` (ms, required) and `recvWindow` (default
  5000 ms, max 60000), then `signature` = hex HMAC-SHA256 of the exact query string (plus the body
  if any) with the secret key.
- Accepted when `timestamp < serverTime + 1000` and `serverTime - timestamp <= recvWindow`; else
  `-1021 INVALID_TIMESTAMP`. Keep a server-time offset from `GET /api/v3/time`.
- WebSocket API signed requests: all params except `signature` sorted by name, joined
  `name=value&...`, HMAC-SHA256 hex. `apiKey` is a param and part of the signed string (the user
  data stream page's example signs `apiKey=...&filter=...&timestamp=...`).

## Orders

`POST /api/v3/order` (weight 1, 1 unfilled order):

- `MARKET` buy by `quoteOrderQty` (USDT to spend); the executed base quantity follows `LOT_SIZE`.
  `MARKET` sell by `quantity`.
- `newClientOrderId`: unique among open orders; reusable only once the previous order with it is
  filled (or expired). Our IDs are 36 characters from `client_order_id` / `child_order_id`.
- `newOrderRespType` `ACK` / `RESULT` / `FULL`; `MARKET` defaults to `FULL`, whose `fills[]` hold
  `price`, `qty`, `commission`, `commissionAsset`, `tradeId`. Average price = `cummulativeQuoteQty
  / executedQty`.
- `LIMIT_MAKER` (`quantity`, `price`): rejected if it would trade at once ("Order would immediately
  match and take.").
- `STOP_LOSS` (`quantity`, `stopPrice` and/or `trailingDelta`): a market order once triggered. A
  `STOP_LOSS SELL` triggers below the market. `trailingDelta` is in basis points (BIPS); its range
  comes from the symbol's `TRAILING_DELTA` filter.

`POST /api/v3/orderList/oco` (weight 1, 2 unfilled orders):

- `symbol`, `side`, `quantity` (both legs), `aboveType`, `belowType`, `listClientOrderId`,
  `aboveClientOrderId`, `belowClientOrderId`, `newOrderRespType`, `timestamp`, `recvWindow`.
- `aboveType` in `STOP_LOSS_LIMIT, STOP_LOSS, LIMIT_MAKER, TAKE_PROFIT, TAKE_PROFIT_LIMIT`;
  `belowType` in `STOP_LOSS, STOP_LOSS_LIMIT, TAKE_PROFIT, TAKE_PROFIT_LIMIT`.
- Our sell bracket: `aboveType=LIMIT_MAKER` + `abovePrice` (take profit), `belowType=STOP_LOSS` +
  `belowStopPrice` and/or `belowTrailingDelta` (stop / trailing stop).
- SELL price rule: `abovePrice` (LIMIT_MAKER) > last traded price > `belowStopPrice`; violating it
  rejects the list.
- One leg filling cancels the other. Cancelling either leg (`DELETE /api/v3/order`) cancels the
  whole list; `DELETE /api/v3/orderList` cancels by `orderListId` or `listClientOrderId`.
- *Unverified:* whether `belowTrailingDelta` combined with `belowStopPrice` acts as an activation
  price (the general trailing-stop FAQ suggests so); we never send both.

Queries: `GET /api/v3/order` (by `origClientOrderId`, weight 4), `GET /api/v3/openOrders` (weight 6
per symbol), `GET /api/v3/orderList` (by `origClientOrderId`, which is the list's
`listClientOrderId`), `GET /api/v3/openOrderList`, `GET /api/v3/account` (weight 20;
`balances[]{asset, free, locked}`), `GET /api/v3/myTrades` (by `orderId`, weight 5),
`GET /api/v3/ticker/price` (public, weight 2: the last price; the broker reads it only to tell
which level a refused bracket crossed).
`GET /api/v3/exchangeInfo` gives each symbol's status and filters (`PRICE_FILTER` tick size,
`LOT_SIZE` / `MARKET_LOT_SIZE` step and bounds, `MIN_NOTIONAL` / `NOTIONAL`, `PERCENT_PRICE_BY_SIDE`,
`TRAILING_DELTA`, `MAX_NUM_ORDERS`).

## Rate limits

- REST responses carry `X-MBX-USED-WEIGHT-1M` and, on successful orders, `X-MBX-ORDER-COUNT-10S` /
  `-1D`. Order limits are per account (documented examples: 50 per 10 s, 160,000 per day); weight
  is per IP (6,000 per minute). HTTP 429 means back off for `Retry-After`; HTTP 418 means banned.
- WebSocket API responses carry a `rateLimits` array with the same limits.

- A client that logs requests must not print signed URLs: `httpx` logs the full URL (signature
  included) at INFO, so `exchange/client.py` installs a log filter that drops those lines.
- `exchangeInfo` filter values of `0` mean "no limit" (`stepSize` of `MARKET_LOT_SIZE` is
  usually `0.00000000`: market orders then use `LOT_SIZE`'s step). Older symbols carry
  `MIN_NOTIONAL` (`applyToMarket`) where newer ones carry `NOTIONAL` (`applyMinToMarket`).
- Verified by test vector: the signature of the docs' example query (secret
  `NhqPtm...Nj0j`) is `c8db5682...bd6b71`; see `tests/test_exchange_client.py`.

## Errors the broker handles

| Code | Meaning | Broker action |
| --- | --- | --- |
| -1021 | timestamp outside recvWindow | resync server time, retry once |
| -1022 / -2014 / -2015 | bad signature / key format invalid / bad key, IP or permissions | fatal: stop placing orders, error alert |
| -1013 | rejected before the matching engine; filter failures ("Filter failure: LOT_SIZE", ...) | reject the order with the filter's name |
| -2010 | new order rejected (insufficient balance, duplicate order, would match and take) | map by message: `insufficient_balance`, `duplicate` (query by client ID), `would_take` |
| -2021 | stop order would trigger immediately (stop price already crossed) | place a market order instead (`WouldTrigger`) |
| -2011 / -2013 | cancel rejected / no such order | query the order: it filled or was already cancelled |
| -1007 / -1006 | timeout / unexpected response, status unknown (HTTP 503 / 409) | query by client order ID before any retry |
| -1015 / -1003 | too many orders / requests | back off |

## User data stream (WebSocket API)

The REST `listenKey` endpoints are deprecated (removal announced for 2026-02-20). Subscribe on a
WebSocket API connection instead:

- `userDataStream.subscribe.signature`, params `apiKey`, `timestamp`, `recvWindow` (optional),
  `signature` (HMAC over the sorted params), optional `filter`. Reply `{"id", "status": 200,
  "result": {"subscriptionId": 0}}`. Not closed by `session.logout`.
  (`userDataStream.subscribe` needs a session logged on with an Ed25519 key; not used.)
- `userDataStream.unsubscribe` (`subscriptionId` optional: none closes all);
  `session.subscriptions` lists them.
- Events arrive one per text frame as `{"subscriptionId": n, "event": {...}}`:
  - `executionReport`: `s` symbol, `c` client order ID, `C` original client order ID (on cancels),
    `S` side, `o` type, `q` quantity, `p` price, `P` stop price, `x` execution type (`NEW`,
    `TRADE`, `CANCELED`, `REJECTED`, `EXPIRED`, `TRADE_PREVENTION`), `X` order status, `r` reject
    reason, `i` order ID, `g` order list ID, `l` last executed quantity, `z` cumulative quantity,
    `L` last price, `n` commission, `N` commission asset, `T` transaction time, `t` trade ID, `Z`
    cumulative quote quantity, `Y` last quote quantity.
  - `listStatus`: `s`, `g`, `c` (`OCO`), `l` list status type, `L` list order status, `r`, `C`
    list client order ID, `T`, `O[]{s, i, c}`.
  - `outboundAccountPosition`: `u`, `B[]{a, f, l}` (asset, free, locked) after balance changes.
  - `balanceUpdate`, `externalLockUpdate`, `eventStreamTerminated` (the subscription ended:
    resubscribe), and `serverShutdown` (no `subscriptionId`: reconnect).
- Events can be missed while disconnected: after every reconnect the broker reconciles over REST.
- Requests on the connection are `{"id", "method", "params"}` frames; the answer carries the
  same `id` with `"status": 200, "result": ...` or an `"error": {"code", "msg"}` (codes as in
  REST). `exchange/ws_api.py` implements this.

## Observed on the real spot testnet (2026-11-21)

A short run (P3 M6: `moonshot exchange check`, then `live run` with 10 USDT test stakes on
BTCUSDT and ETHUSDT, a restart, time exits and a flatten) confirmed:

- `userDataStream.subscribe.signature` with an HMAC key works on
  `wss://ws-api.testnet.binance.vision/ws-api/v3` (subscription id 0), with the stream's clock
  corrected by the REST server-time offset (offsets seen: 16-148 ms).
- Account and rules: a fresh testnet account held 10,000 USDT, 1 BTC and 1 ETH (and more assets);
  BTCUSDT tick 0.01, step 0.00001, min notional 5; ETHUSDT tick 0.01, step 0.0001, min notional 5.
- A market buy with `quoteOrderQty=10` bought `floor(10 / price / step) x step` (0.00017 BTC for
  9.58 USDT): the executed quantity is rounded down to the lot step, so less than the quote amount
  is spent. The book records the USDT actually spent.
- No commission was charged on these testnet trades (no base-asset deduction; no dust); the
  account's commission rates on the testnet may be zero, so real fee handling is still only
  covered by the simulator.
- An OCO SELL with `aboveType=LIMIT_MAKER` and `belowType=STOP_LOSS` is accepted (two open orders
  locking the quantity), and so is a single `STOP_LOSS` sell with only `trailingDelta` (one open
  order). Cancelling the order list and the single order, then selling at market, worked for both
  the engine's time exits and a flatten.
- Restarting the daemon mid-position reconciled cleanly (no orphans, no mismatches) and kept the
  brackets; the testnet USDT balance changed by exactly the recorded P&L.
- Request weight peaked at 194 per minute over a session with start-up, reconcile, polling and
  four orders.

- A run with 0.1% levels saw the real triggers: the OCO's `LIMIT_MAKER` take profit filled 23 s
  after entry at exactly `ref x 1.001` rounded up to the tick (the stop leg was cancelled by the
  exchange), and the trailing `STOP_LOSS` (10 bips, no stop price) followed the price up and
  triggered about 3 minutes later. Both fills arrived over the user data stream and closed the
  positions with reasons `take_profit` and `trailing_stop`.

Still not observed (no rejection was provoked and no fill was partial): the `-2010` and
OCO-relationship wording, the order-count headers, partial fills, and a fixed `STOP_LOSS` leg
triggering. They remain covered by the simulator only.

## Assumptions to check on the real testnet

The broker is built so that none of these is load-bearing for correctness (each has a fallback),
but each is *unverified* until a real testnet run (M6 in the plan):

- The order of a REST answer and the stream's reports for the same order is not fixed: the
  broker accepts either and counts each change once (cumulative `z`/`Z`, trade IDs).
- An OCO leg's first partial fill expires the other leg (and `listStatus` is `ALL_DONE` only
  once no leg can fill). The simulator does this; the broker closes a position only when a leg
  is `FILLED` or every leg is final either way.
- The OCO price-relationship rejection is `-1013` "The relationship of the prices for the
  orders is not correct." (an unnamed filter failure); the broker then reads the last price to
  tell which level was crossed, so the code or wording may differ without harm.
- A `STOP_LOSS` sell with only `trailingDelta` starts trailing at once from the price at
  placement (no activation price). Placement is confirmed on the testnet; the trigger was not
  observed.
- `quoteOrderQty` is accepted with 8 decimals for USDT pairs (the broker rounds down to 8).
  Confirmed for `10` on the testnet; the executed quantity is rounded down to the lot step.
- A market buy's commission is charged in the base asset unless BNB pays fees; the broker
  handles base, USDT and any other asset.
- The `userDataStream.subscribe.signature` timestamp must be within `recvWindow` of the server
  clock like a REST request: the stream must be given a clock corrected by the REST client's
  server-time offset when the local clock is off.
