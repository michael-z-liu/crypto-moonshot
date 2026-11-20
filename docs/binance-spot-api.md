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
per symbol), `GET /api/v3/orderList`, `GET /api/v3/openOrderList`, `GET /api/v3/account` (weight 20;
`balances[]{asset, free, locked}`), `GET /api/v3/myTrades` (by `orderId`, weight 5).
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
