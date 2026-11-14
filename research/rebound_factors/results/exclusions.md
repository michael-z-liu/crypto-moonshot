# Excluded entries: price-series breaks

A break is a pair of consecutive 5m candles whose open is more than 3 times, or less
than a third of, the previous close (a redenomination or token swap after a trading
halt). An entry is excluded when a break of its coin lies within 200 days before `D`
or inside its 30-day window. Built by `build_exclusions.py`.

| coin | reopened (UTC) | open / last close | halt (hours) |
| --- | --- | --- | --- |
| BNXUSDT | 2023-02-22 08:00 | 0.01 | 149 |
| BTCSTUSDT | 2021-03-19 07:00 | 0.0999984 | 96 |
| COCOSUSDT | 2021-01-23 02:00 | 1000 | 96 |
| DREPUSDT | 2021-04-02 04:00 | 100.008 | 96 |
| LUNAUSDT | 2022-05-31 06:00 | 20000 | 437 |
| QUICKUSDT | 2023-07-21 08:00 | 0.001 | 101 |
| STRAXUSDT | 2024-03-28 08:00 | 0.1 | 197 |
| SUNUSDT | 2021-06-18 04:00 | 0.000999927 | 96 |
| VIDTUSDT | 2022-11-09 08:00 | 0.0930036 | 221 |

| split | entries excluded | of |
| --- | --- | --- |
| confirm | 249 | 299711 |
| discovery | 1661 | 282923 |
| gap | 24 | 9498 |
| unused | 0 | 215 |
