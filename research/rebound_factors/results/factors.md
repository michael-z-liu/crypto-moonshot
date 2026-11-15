# Rebound factors: the factor dictionary (M2)

Built by `build_factors.py` into `cache/factors.parquet`, one row per panel entry
(`symbol`, `d_us`), every year. Every factor uses only 5m candles with open time before
`D`, aggregated to UTC-day bars; the last complete day is `D - 1` and its close is `P`.
Notation: `c[k]` is the daily close `k` days before the last complete day (so
`c[0] = P`), `h`/`l` daily high/low, `qv` quote volume, `tb` taker-buy quote volume,
`tc` trade count, `lr` the daily log return; `max h 30` is the highest daily high of
the 30 days `D-30 .. D-1`. A day without candles is a flat bar at the last close with
zero volume. A factor is null when its lookback reaches before the coin's first 5m
candle (the 5m data start on 2020-01-01, so long lookbacks are null early in 2020).

*Market* factors are the same for every coin on a day; *coin* factors differ by coin.
Breadth factors use the coins in the panel on that day. Within-day ranks of the coin
factors are computed in M3.

| # | name | family | kind | formula | lookback |
| --- | --- | --- | --- | --- | --- |
| 1 | `m_btc_ret1` | market regime | market | BTC c[0] / c[1] - 1 | 1 d |
| 2 | `m_btc_ret7` | market regime | market | BTC c[0] / c[7] - 1 | 7 d |
| 3 | `m_btc_ret30` | market regime | market | BTC c[0] / c[30] - 1 | 30 d |
| 4 | `m_btc_ret90` | market regime | market | BTC c[0] / c[90] - 1 | 90 d |
| 5 | `m_btc_sma50` | market regime | market | BTC c[0] / mean(c, 50 d) - 1 | 50 d |
| 6 | `m_btc_sma200` | market regime | market | BTC c[0] / mean(c, 200 d) - 1 | 200 d |
| 7 | `m_btc_dd90` | market regime | market | BTC c[0] / max h 90 - 1 | 90 d |
| 8 | `m_btc_vol7` | market regime | market | BTC std of lr over 7 d (daily, not annualised) | 8 d |
| 9 | `m_btc_vol30` | market regime | market | BTC std of lr over 30 d | 31 d |
| 10 | `m_btc_vol_30_180` | market regime | market | BTC std lr 30 d / std lr 180 d | 181 d |
| 11 | `m_ethbtc_ret30` | market regime | market | (ETH c[0] / c[30]) / (BTC c[0] / c[30]) - 1 | 30 d |
| 12 | `m_breadth20` | market regime | market | share of the day's panel coins with c_sma20 > 0 | 20 d |
| 13 | `m_breadth50` | market regime | market | share of the day's panel coins with c_sma50 > 0 | 50 d |
| 14 | `m_breadth200` | market regime | market | share of the day's panel coins with c_sma200 > 0 | 200 d |
| 15 | `m_med_ret7` | market regime | market | median c_ret7 over the day's panel coins | 7 d |
| 16 | `m_disp_ret7` | market regime | market | std of log(1 + c_ret7) over the day's panel coins | 7 d |
| 17 | `m_med_ret30` | market regime | market | median c_ret30 over the day's panel coins | 30 d |
| 18 | `m_disp_ret30` | market regime | market | std of log(1 + c_ret30) over the day's panel coins | 30 d |
| 19 | `m_moon30` | market regime | market | share of the day's panel coins whose max h 30 >= 1.3 c[30] (last ended window) | 31 d |
| 20 | `m_qv_7_90` | market regime | market | all universe coins' summed qv: mean 7 d / mean 90 d | 90 d |
| 21 | `c_ret1` | momentum and position | coin | c[0] / c[1] - 1 | 1 d |
| 22 | `c_ret3` | momentum and position | coin | c[0] / c[3] - 1 | 3 d |
| 23 | `c_ret7` | momentum and position | coin | c[0] / c[7] - 1 | 7 d |
| 24 | `c_ret14` | momentum and position | coin | c[0] / c[14] - 1 | 14 d |
| 25 | `c_ret30` | momentum and position | coin | c[0] / c[30] - 1 | 30 d |
| 26 | `c_ret90` | momentum and position | coin | c[0] / c[90] - 1 | 90 d |
| 27 | `c_dd30` | momentum and position | coin | c[0] / max h 30 - 1 (drawdown from the high) | 30 d |
| 28 | `c_dd90` | momentum and position | coin | c[0] / max h 90 - 1 | 90 d |
| 29 | `c_ru30` | momentum and position | coin | c[0] / min l 30 - 1 (run-up from the low) | 30 d |
| 30 | `c_ru90` | momentum and position | coin | c[0] / min l 90 - 1 | 90 d |
| 31 | `c_pos30` | momentum and position | coin | (c[0] - min l 30) / (max h 30 - min l 30) | 30 d |
| 32 | `c_sma20` | momentum and position | coin | c[0] / mean(c, 20 d) - 1 | 20 d |
| 33 | `c_sma50` | momentum and position | coin | c[0] / mean(c, 50 d) - 1 | 50 d |
| 34 | `c_sma100` | momentum and position | coin | c[0] / mean(c, 100 d) - 1 | 100 d |
| 35 | `c_sma200` | momentum and position | coin | c[0] / mean(c, 200 d) - 1 | 200 d |
| 36 | `c_sma50_slope` | momentum and position | coin | SMA50 today / SMA50 10 days earlier - 1 | 60 d |
| 37 | `c_up10` | momentum and position | coin | number of days with c[k] > c[k+1] among the last 10 | 11 d |
| 38 | `c_ret1_atr` | momentum and position | coin | (c[0] - c[1]) / ATR14 | 15 d |
| 39 | `c_atr14` | volatility and compression | coin | mean true range over 14 d / c[0] (TR = max(h-l, |h-c[1]|, |l-c[1]|)) | 15 d |
| 40 | `c_vol7` | volatility and compression | coin | std of lr over 7 d | 8 d |
| 41 | `c_vol30` | volatility and compression | coin | std of lr over 30 d | 31 d |
| 42 | `c_vol_7_30` | volatility and compression | coin | c_vol7 / c_vol30 | 31 d |
| 43 | `c_range_7_30` | volatility and compression | coin | (max h 7 - min l 7) / (max h 30 - min l 30) (compression) | 30 d |
| 44 | `c_habit` | volatility and compression | coin | share of past entries b (D-365 .. D-30) whose 30-day window [b, b+30) ended by D with max h >= 1.3 c(b-1); null under 120 windows | 366 d |
| 45 | `c_maxgain30` | volatility and compression | coin | largest c[k] / c[k+1] - 1 over the last 30 days | 31 d |
| 46 | `c_qv_1_30` | volume and flow | coin | qv[0] / mean qv of the 30 days before it | 31 d |
| 47 | `c_qv_7_30` | volume and flow | coin | mean qv 7 d / mean qv 30 d | 30 d |
| 48 | `c_qv_30_90` | volume and flow | coin | mean qv 30 d / mean qv 90 d | 90 d |
| 49 | `c_qv_z` | volume and flow | coin | z-score of log(1+qv[0]) against the 30 days before it | 31 d |
| 50 | `c_tb1` | volume and flow | coin | tb[0] / qv[0] (taker-buy share of the last day) | 1 d |
| 51 | `c_tb7` | volume and flow | coin | sum tb 7 d / sum qv 7 d | 7 d |
| 52 | `c_tb_chg` | volume and flow | coin | c_tb7 - (sum tb 30 d / sum qv 30 d) | 30 d |
| 53 | `c_tc_7_30` | volume and flow | coin | mean trade count 7 d / mean trade count 30 d | 30 d |
| 54 | `c_ats_7_30` | volume and flow | coin | average trade size (sum qv / sum tc) 7 d over 30 d | 30 d |
| 55 | `c_upvol14` | volume and flow | coin | sum qv on up days (c[k] > c[k+1]) / sum qv, 14 d | 15 d |
| 56 | `c_rs7` | relative strength, size, age | coin | c_ret7 - m_btc_ret7 | 7 d |
| 57 | `c_rs30` | relative strength, size, age | coin | c_ret30 - m_btc_ret30 | 30 d |
| 58 | `c_rs90` | relative strength, size, age | coin | c_ret90 - m_btc_ret90 | 90 d |
| 59 | `c_beta60` | relative strength, size, age | coin | OLS slope of the coin's lr on BTC's lr over 60 d | 61 d |
| 60 | `c_resid30` | relative strength, size, age | coin | log(c[0]/c[30]) - c_beta60 * BTC log(c[0]/c[30]) | 61 d |
| 61 | `c_log_liq` | relative strength, size, age | coin | log(panel liq) (mean daily qv of the 30 days before D) | 30 d |
| 62 | `c_liq_rank` | relative strength, size, age | coin | rank of liq among the day's panel coins, (rank - 1) / (n - 1) | 30 d |
| 63 | `c_log_age` | relative strength, size, age | coin | log(panel days_listed) | all |
| 64 | `c_touch_ma` | reference flag | coin | 1 if l[0] <= SMA n (closes to D-2) for some n in 50/100/200, with l > SMA n on each of the 10 days before; null without SMA50 | 211 d |
| 65 | `c_below_200d` | reference flag | coin | 1 if c[0] < mean(c, 200 d), else 0 | 200 d |

## Count by family

| family | kind | factors |
| --- | --- | --- |
| market regime | market | 20 |
| momentum and position | coin | 18 |
| reference flag | coin | 2 |
| relative strength, size, age | coin | 8 |
| volatility and compression | coin | 7 |
| volume and flow | coin | 10 |
| **all** | | 65 |
