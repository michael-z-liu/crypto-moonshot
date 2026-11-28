# Post-delisting perps: L on perps that kept trading after their spot delisting

Question: the spot backtest stops at a coin's spot delisting, and p1 left out every coin that has
spot klines. Do the USD-M perps that kept trading **after** their coin's spot delisting add L
events, and does L (1x perp long) still work there?

Script: `p3_postdelist.py` (coin list, 5m fetch, spikes, fold scoring, L trades, book). Data:
OWN/futures/perp5m_postdelist/ (18 files, 24 MB), OWN/p3/.

## Verdict

- **Too small to matter; it does not add to LMC+.** 18 perps traded at least 3 days after
  their spot delisting, 3,534 coin-days in 2022-26. They give 11,958 liquid spikes, but only
  **15 L signals (pct >= 0.995 with an active perp) on 6 coins and 4 days**, and 8 trades with
  one position per coin.
- **The trades are fine, but there are too few to judge:** n 8, mean net **+0.096**, TP rate 0.50, 2024 +0.168 (n 5, all
  on 2024-12-09), 2025 -0.023 (n 3). Without the FLM trade (entered 1.4 h before the perp
  stopped trading, so probably not possible in practice, see below): n 7, mean +0.120.
- **No new flush days.** All 4 signal days (2024-12-09, 2025-09-22, 2025-10-10, 2025-11-21) are
  already spot-L or perp-only-L days. So these trades only compete for slots on days that already
  have enough candidates. On 2024-12-09 the LMC+ book already has nearly 300 L candidates in the hours after 21:00.
- **Book (10 slots, $10k each year):** LMC+ 2024 +152% -> +153%, 2025 +790% -> +784%; other
  years and drawdowns are unchanged; 485 -> 487 trades. That is a wash.
- **Recommendation:** do not add a post-delisting rule. It is harmless but adds nothing. If a
  live scanner covers every active USD-M perp anyway, these coins are already included without a
  special case.

## Why it adds nothing

1. **Few real windows.** `data/delisted.txt` lists 191 spot symbols, and 107 of them have a USD-M
   perp. In perp_1h, 98 of those perps print candles at least 3 days past the spot delisting. But
   80 of the 98 only print flat zero-volume candles: the archive keeps writing these after a perp
   is delisted. Binance usually delists the perp at or before the spot delisting. Only 18 perps
   really traded on. 4 of them are long-lived (XMR, LIT, AERGO, MDT: 64% of the coin-days), and
   most of the rest lasted 9-230 days.
2. **Tail events are rare in such a small sample.** L needs pct >= 0.995: roughly 25 market-wide
   flush days in 2022-26. 3.5k coin-days of mostly dying, low-interest coins catch only 4 of those
   days.
3. **The days are not new.** A capitulation flush is market-wide. On the days it fires here,
   spot and perp-only coins already fill the 10 slots, so a few more coins barely change the
   book.

## Method and honesty

- Coin list: every perp in perp_1h.parquet whose spot name has spot 5m klines in the shared store
  and whose last **active** hour (quote volume > 0 and high > low) is >= 3 days after the last spot
  5m candle. Every coin found is in delisted.txt.
- 5m klines are fetched from the month before the spot delisting up to the perp's last active
  month + 1. The lookback covers the 7-day spike features, the warm-up and the 288-bar CNN
  sequence. Spikes count only from the first 5m candle after the last spot candle up to
  2026-11-01. Flat zero-volume candles after the last traded candle are dropped.
- Spikes, coin channels, the liquidity filter (qv_base >= 5k), BTC/ETH/market channels and
  scoring are the p1 / p2_prespot pipeline, unchanged (`p2_lib.score_coins`). Every spike in year
  Y is scored by the day5 walk-forward fold Y (2022-25: `day5_<Y>`, 2026: `day5`). The coin
  channels come from the perp's own candles, as in p1. None of these perps' post-delisting candles
  was in any training set.
- L is unchanged: pct >= 0.995, perp activity gate (`futures_ls.has_perp`), entry at the next 5m
  perp open, TP +20%, 72h, 1x, cost 0.3% + funding (funding.parquet covers these perps), one
  position per coin (`p2_lib.sim_perp`).
- **Perp delisting mid-trade:** the 72h path runs past the perp's last traded candle for **1
  trade** (FLMUSDT 2025-11-21, entered 07:35, perp's last candle 08:55). It was still open then,
  so it is settled at the last traded price (net -0.071; `perp_paths` forward-fills the close). In
  practice Binance moves a perp to reduce-only before it delists, so this entry was probably not
  possible. HIFI (entered 11 days before its perp's end) and all other trades finished within the
  perp's life.
- Caveat: the book check matches perp trades to spot L by symbol, but a spot L position open at
  the spot delisting is truncated by the spot backtest. This does not arise here (no
  post-delisting signal falls within 72h of a spot delisting).

## Raw output (`p3_postdelist.py`)

### Coverage

18 perps traded at least 3 days after their coin's last spot 5m candle (all 18 in data/delisted.txt); 3,534 coin-days in 2022-01..2026-10:

| perp | spot_last | perp_last_traded | window_days | days_in_2022_26 |
|---|---|---|---|---|
| HNTUSDT | 2022-10-14 02:55:00+00:00 | 2023-03-20 09:00:00+00:00 | 157.3 | 157.2 |
| XMRUSDT | 2024-02-20 02:55:00+00:00 | 2026-10-31 23:55:00+00:00 | 984.8 | 984.8 |
| ANTUSDT | 2024-02-20 02:55:00+00:00 | 2024-04-01 08:55:00+00:00 | 41.2 | 41.2 |
| XEMUSDT | 2024-06-17 02:55:00+00:00 | 2024-12-09 09:00:00+00:00 | 175.3 | 175.2 |
| OMGUSDT | 2024-06-17 02:55:00+00:00 | 2025-01-31 09:00:00+00:00 | 228.2 | 228.2 |
| BONDUSDT | 2024-07-22 02:55:00+00:00 | 2024-12-16 08:55:00+00:00 | 147.2 | 147.2 |
| REEFUSDT | 2024-08-26 02:55:00+00:00 | 2025-01-22 09:00:00+00:00 | 149.3 | 149.2 |
| LOOMUSDT | 2024-08-26 02:55:00+00:00 | 2024-12-09 08:55:00+00:00 | 105.2 | 105.2 |
| LITUSDT | 2025-02-10 02:55:00+00:00 | 2026-10-31 23:55:00+00:00 | 628.8 | 628.8 |
| AERGOUSDT | 2025-03-28 02:55:00+00:00 | 2026-06-30 23:55:00+00:00 | 459.8 | 459.8 |
| LEVERUSDT | 2025-07-04 02:55:00+00:00 | 2025-09-03 08:55:00+00:00 | 61.2 | 61.2 |
| ALPHAUSDT | 2025-07-04 02:55:00+00:00 | 2025-09-23 09:00:00+00:00 | 81.3 | 81.2 |
| BSWUSDT | 2025-07-04 02:55:00+00:00 | 2025-09-15 08:55:00+00:00 | 73.2 | 73.2 |
| BAKEUSDT | 2025-09-17 02:55:00+00:00 | 2025-10-03 08:55:00+00:00 | 16.2 | 16.2 |
| HIFIUSDT | 2025-09-17 02:55:00+00:00 | 2025-10-03 09:00:00+00:00 | 16.3 | 16.2 |
| FLMUSDT | 2025-11-12 02:55:00+00:00 | 2025-11-21 08:55:00+00:00 | 9.2 | 9.2 |
| PERPUSDT | 2025-11-12 02:55:00+00:00 | 2025-11-21 09:00:00+00:00 | 9.2 | 9.2 |
| MDTUSDT | 2026-04-23 02:55:00+00:00 | 2026-10-31 23:45:00+00:00 | 191.8 | 191.8 |

11,958 liquid perp spikes after the spot delisting (scored by the walk-forward fold of their year):

| year | spikes | coins | perp_active | q995 | q995_active | q995_days |
|---|---|---|---|---|---|---|
| 2022 | 319 | 1 | 1.000 | 0 | 0 | 0 |
| 2023 | 328 | 1 | 1.000 | 0 | 0 | 0 |
| 2024 | 5304 | 7 | 0.987 | 10 | 10 | 1 |
| 2025 | 4087 | 12 | 0.987 | 5 | 5 | 3 |
| 2026 | 1920 | 3 | 0.980 | 0 | 0 | 0 |

### L trades

L signals (pct >= 0.995 with an active perp): 15 spikes on 6 coins and 4 days. Trades (perp long, 1x, TP 20%, 72h, cost 0.3% + funding, one per coin):

| year | n | coins | days | mean_net | tp_rate |
|---|---|---|---|---|---|
| 2024 | 5 | 4 | 1 | 0.168 | 0.800 |
| 2025 | 3 | 3 | 3 | -0.023 | 0.000 |

All: n 8, mean net +0.0962, TP rate 0.50, 4 days. The 72h path runs past the perp's last traded candle for 1 trades; 1 were still open then and are settled at the last traded price. 0 of the 4 signal days are not spot-L or perp-only-L days (-); trades on them: n 0, mean net +nan.

| symbol | open_time | pct | ret_4h | net | exit_kind | settled_at_end |
|---|---|---|---|---|---|---|
| REEFUSDT | 2024-12-09 21:00:00+00:00 | 0.999 | -0.327 | 0.197 | tp | False |
| OMGUSDT | 2024-12-09 21:00:00+00:00 | 0.999 | -0.264 | 0.197 | tp | False |
| BONDUSDT | 2024-12-09 21:00:00+00:00 | 0.999 | -0.260 | 0.197 | tp | False |
| XMRUSDT | 2024-12-09 21:00:00+00:00 | 0.998 | -0.201 | 0.197 | tp | False |
| REEFUSDT | 2024-12-09 22:10:00+00:00 | 0.996 | -0.241 | 0.051 | time | False |
| HIFIUSDT | 2025-09-22 06:00:00+00:00 | 0.995 | -0.112 | -0.126 | time | False |
| XMRUSDT | 2025-10-10 21:15:00+00:00 | 0.995 | -0.159 | 0.129 | time | False |
| FLMUSDT | 2025-11-21 07:30:00+00:00 | 0.995 | -0.152 | -0.071 | time | True |

### Book (10 slots, $10k each year)

| book | trades | bp | 2022 | 2023 | 2024 | 2025 | 2026 | min yr |
|---|---|---|---|---|---|---|---|---|
| LMC+ | 485 | 1145 | +62% (dd -10) | +81% (dd -7) | +152% (dd -16) | +790% (dd -17) | +111% (dd -10) | +62% |
| LMC+ + post-delisting L | 487 | 1140 | +62% (dd -10) | +81% (dd -7) | +153% (dd -16) | +784% (dd -17) | +111% (dd -10) | +62% |

