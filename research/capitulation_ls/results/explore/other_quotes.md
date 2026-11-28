# Other quote currencies 2020-26: L on coins that had no USDT spot pair

L makes money on flush days, so it is limited by how many of those days it can trade. Adding
coins has helped whenever it added days: the perp-only coins (p1) in 2025-26, and the 2017-19
BTC pairs (h2: 26 new days out of 38). The 2022-26 backtest used only USDT spot pairs plus the
perp-only perps. This study repeats h2 for 2020-01 .. 2026-10. It uses spot pairs quoted in
BUSD, BTC, FDUSD, USDC, TUSD, ETH or BNB, and only coin-times with no USDT spot pair.

**The rule is unchanged.** The spike builder is h1/h2's, the conversion is `h2_fetch.convert`,
and L is the same: pct >= 0.995, entry at the next 5m open, TP +20%, 72h time exit, one position
per coin. Each year is scored with its own walk-forward fold: 2022-25 by `day5_<Y>`, 2026 by
`day5`. **2020-21 have no walk-forward fold, so they are scored with the 2022 fold applied
backwards.** That fold was trained on 2020-01 .. 2021-06 USDT rows, so these years are in its
training period (on other coins), and they are reported apart. Costs are 0.4%, and 1.2% with
the conversion legs.

Scripts:
- `h3_fetch.py`: listing, plan, download and conversion;
- `h3_build.py`: spikes, coin-time filters, CNN inputs, outcomes;
- `h3_score.py`: fold scoring, plus the study's USDT spikes of 2020-21 scored with the 2022 fold;
- `h3_eval.py`: trades, new days, calibration, books.

Data and full tables are in `$DATA/cache/capitulation_ls/h3/` (`eval_tables.md`).

## Verdict: fails. It adds almost no flush days, and none from 2024 on

- **2022-26 (walk-forward): 2 trades on 1 day, both +1960 bp.** The day is 2022-11-08 (FTX:
  USTC/BUSD, PHB/BUSD), which spot USDT L already trades. **0 new flush days** against spot USDT
  L, against perp-only L, and against both.
- **2020-21 (2022 fold applied backwards): 51 trades on 10 days.** Mean +1706 bp at 0.4% and
  +1626 bp at 1.2%, TP rate 0.94. Day-bootstrap CI at 1.2%: [+1148, +1880] bp. But **only 2 of
  the 10 days are new** against spot USDT L scored the same way:
  - 2020-12-28: COVER/BUSD, the Cover Protocol mint exploit, 2 trades, mean -1425 bp;
  - 2021-05-21: EZ/BTC, 1 trade, +1960 bp.
- **LMC+ book 2022-26 (10 slots, $10k each year), with the h3 trades added:**
  - **2022: +62.1% -> +66.0% at 0.4%, +65.8% at 1.2%.** That comes from the two FTX-day trades
    filling idle slots.
  - 2023-26 are unchanged: +81.3%, +151.8%, +789.6%, +111.2%.
  - The worst year is still 2022 (+65.8% at 1.2%), and the max drawdown is unchanged (-16.8%).
- **The 2020-21 L-only book gets slightly worse** (USDT L with the backward fold):
  - 2020: +17.1% -> +15.5% at 1.2%, with drawdown -2.4% -> -4.9% because of COVER;
  - 2021: +329.9% -> +327.7%.

The per-trade edge on these coins is as strong as on USDT pairs, and calibration in 2020-21 is
a hockey stick. But almost every trade lands on a day that the USDT universe already trades.

## Coverage

- **Listing.** data.binance.vision has 1,936 plain pairs in these quotes with 5m months in
  2019-12 .. 2026-10. Stablecoin, fiat, wrapped and leveraged bases are excluded.
- **USDT months.** They come from the study's own USDT stores (shared folder + volume-spike
  archive store). An archive-only USDT pair blocks its whole months; only YBUSDT, which the
  study's stock filter had dropped.
- **Plan and fetch.** The plan has 1,283 pairs and 5,256 symbol-months: months with no USDT pair,
  their neighbours, and pre-listing months. 5,236 months were stored. The 8 failures are all
  `BNBUSD` (BNB/USD, mis-parsed as BN/BUSD; no data after conversion). `USDTUSD` (USDT/USD) was
  also mis-parsed and stored, then excluded from the build.
- **Spikes.**
  - 75,957 liquid spikes in total;
  - 61,536 while the coin had no USDT pair (spike before the USDT pair's first candle or after
    its last);
  - 0 removed for a live perp-only perp. No p1 coin has a non-USDT spot pair while its perp
    trades.
- **Conversion** uses BTCUSDT, ETHUSDT, BNBUSDT, BUSDUSDT, FDUSDUSDT, USDCUSDT and TUSDUSDT
  (shared folder, plus 131 fetched months: BUSDUSDT, TUSDUSDT and every quote's 2019-12).
- **The CNN's market channels** come from the study's own `market.parquet`, the inputs the folds
  were trained on (as p1).

Pairs with stored months, per quote and year. From 2024 on these months are only listing months
of coins that got a USDT pair at the same time:

| quote | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|
| BTC | 178 | 165 | 62 | 40 | 39 | 8 | 0 |
| BUSD | 86 | 126 | 65 | 28 | 0 | 0 | 0 |
| BNB | 105 | 61 | 14 | 12 | 26 | 64 | 3 |
| ETH | 97 | 39 | 23 | 7 | 1 | 2 | 0 |
| USDC | 1 | 0 | 1 | 0 | 19 | 90 | 23 |
| FDUSD | 0 | 0 | 0 | 10 | 50 | 62 | 1 |
| TUSD | 1 | 1 | 1 | 15 | 2 | 0 | 0 |

Kept liquid spikes (coin without a USDT pair), per quote and year. There are none after
2023-11-15:

| quote | 2020 | 2021 | 2022 | 2023 | 2024-26 |
|---|---|---|---|---|---|
| BTC | 3,553 | 21,134 | 2,033 | 810 | 0 |
| BUSD | 441 | 15,964 | 7,204 | 6,282 | 0 |
| ETH | 66 | 2,540 | 57 | 5 | 0 |
| BNB | 8 | 1,410 | 11 | 0 | 0 |
| TUSD | 0 | 18 | 0 | 0 | 0 |
| **all** (coins) | 4,068 (95) | 41,066 (95) | 9,305 (45) | 7,097 (29) | 0 |

Share of spikes at pct >= 0.995 by year: 0.15% / 0.28% / 0.04% / 0.00%, against 0.25% in the
USDT data of 2022-26.

## L trades by year

| year | fold | n | mean 0.4% (bp) | mean 1.2% (bp) | TP rate | win | flush days | new vs spot USDT L | new vs perp-only L | new vs both | coins |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2020 | 2022, backwards | 3 | -297 | -377 | 0.67 | 0.67 | 2 | 1 | 2 | 1 | 2 |
| 2021 | 2022, backwards | 48 | +1831 | +1751 | 0.96 | 0.98 | 8 | 1 | 8 | 1 | 29 |
| 2022 | day5_2022 | 2 | +1960 | +1880 | 1.00 | 1.00 | 1 | 0 | 0 | 0 | 2 |
| 2023 | day5_2023 | 0 | | | | | 0 | 0 | 0 | 0 | |
| 2024-26 | | 0 (no spikes) | | | | | 0 | 0 | 0 | 0 | |

- **Reference days.** Spot USDT L traded 3 / 10 flush days in 2020 / 2021 (2022 fold backwards),
  5 in 2022. Perp-only L has no 2020-21 data.
- **2021 trades** are by quote BTC 33, BUSD 9, ETH 6. They are old alts that still traded only
  against BTC or BUSD (BQX, AGI/AGIX, XVG, LOOM, STEEM, GAS, BCD, AERGO, PROS, FRONT, ...), on
  the market flushes 2021-02-22/23, 04-18, 04-23, 05-13, 05-19 (19 trades) and 09-07 (10).
- **Exits, 2020-21:** 48 take profits and 3 time exits (mean -2362 bp, two of them COVER).
- **Spot USDT L, same backward fold:** 2020 n 26, +1372 bp; 2021 n 651, +1730 bp. The h3 trades
  are no better per trade.
- **Naive baselines** (L's exit, one position per coin):

  | baseline | 2020-21 | 2022-26 |
  |---|---|---|
  | every liquid spike | -15 bp | -10 bp |
  | spike after a >= 20% 24h fall | +638 bp | -201 bp |

## Calibration (0.4%, per spike, fixed pct bins)

| pct bin | 2020-21 n | 2020-21 mean (bp) | TP | 2022-26 n | 2022-26 mean (bp) | TP | USDT 2022-26 (h2 table) |
|---|---|---|---|---|---|---|---|
| 0.90-0.95 | 2,270 | +52 | 0.35 | 649 | -2 | 0.32 | -34 |
| 0.95-0.976 | 1,141 | +356 | 0.48 | 278 | +67 | 0.38 | +86 |
| 0.976-0.984 | 340 | +951 | 0.65 | 52 | -35 | 0.35 | +355 |
| 0.984-0.9955 | 368 | +1166 | 0.72 | 24 | +429 | 0.33 | +742 |
| 0.9955+ | 111 | +1794 | 0.95 | 4 | +1960 | 1.00 | +1348 |

- **2020-21: a hockey stick, but with a raised handle** (2022 fold backwards, so partly in its
  training period).
  - Deciles of the pct >= 0.95 pool: +0.02 .. +0.07 for the first seven, then +0.12 / +0.11 /
    +0.16.
  - Spearman(signal, outcome) +0.24.
  - Neighbours are monotone: pct >= 0.99 gives +1429 bp (106 trades, 16 days); 0.995 gives
    +1706; 0.998 gives +1960.
- **2022-26: too few high-score spikes to tell.** The 0.9-0.976 handle is flat near zero, as on
  USDT pairs. Spearman over the pool is -0.07.

## 10-slot books

LMC+ = `p2_lib.lmc()` + `to_book(lperp())`. The h3 trades are added as L rows (prio 3 + pct)
with the stake capped at 0.5% of the pair's 24h quote volume in USDT. `e25_combined.book`, $10k
fresh each year.

| book | trades | bp | 2022 | 2023 | 2024 | 2025 | 2026 | worst yr | max dd |
|---|---|---|---|---|---|---|---|---|---|
| LMC+ (reference) | 485 | 1145 | +62.1% | +81.3% | +151.8% | +789.6% | +111.2% | +62.1% | -16.8% |
| LMC+ + h3 L, 0.4% | 485 | 1151 | +66.0% | +81.3% | +151.8% | +789.6% | +111.2% | +66.0% | -16.8% |
| LMC+ + h3 L, 1.2% | 485 | 1151 | +65.8% | +81.3% | +151.8% | +789.6% | +111.2% | +65.8% | -16.8% |
| L spot + L perp | 218 | 1059 | +25.9% | +39.6% | +86.4% | +58.4% | +65.0% | +25.9% | -14.6% |
| L spot + L perp + h3 L, 1.2% | 219 | 1062 | +25.7% | +39.6% | +86.5% | +61.5% | +65.0% | +25.7% | -14.6% |

- The LMC+ reference matches `perp_only.md` (+62 / +81 / +152 / +790 / +111%).
- In "L spot + L perp + h3 L", the 2025 change comes from slot reshuffling: there are no h3
  trades in 2025.

2020-21, L-only books (2022 fold backwards):

| book | 2020 | 2021 |
|---|---|---|
| USDT L | +17.1% (dd -2.4) | +329.9% (dd -2.4) |
| USDT L + h3 L, 0.4% | +15.7% (dd -4.8) | +330.7% (dd -2.4) |
| USDT L + h3 L, 1.2% | +15.5% (dd -4.9) | +327.7% (dd -2.4) |

## Why it failed

1. **The extra coins flush on the same days.** L's days are market-wide capitulations.
   - In 2017-19 the USDT universe was tiny (6 / 21 / 58 pairs), so most flush days had no
     USDT coin at all, and the BTC pairs added days.
   - From 2020 the USDT universe has hundreds of coins. Every market flush already has USDT
     candidates, and the no-USDT coins (old alts on BTC/BUSD) bounce on those same days. 8 of
     the 10 flush days in 2020-21 and the only one in 2022 are already traded.
   - The perp-only coins worked for a different reason: they are a distinct population (new
     micro-cap listings) with their own idiosyncratic flushes.
2. **The no-USDT coin-times disappear.**
   - In 2022-23, Binance added USDT pairs to the last BTC/BUSD-only alts (AMB, GAS, LOOM, AST,
     ARK, GFT, ...), and the BUSD pairs were delisted by 2024-03.
   - From 2024 every new coin lists with a USDT pair from its first candle: the lead time is 0
     days in 357 of the 2024-26 pairs checked.
   - The last kept spike is on 2023-11-15. FDUSD, USDC and the 2024-26 BNB/BTC pairs add nothing.
3. **What is left scores low.** In 2022-23 there are 16k no-USDT spikes, but only 4 reach
   pct >= 0.995 (0 in 2023). They are mostly BUSD-only coins in quiet markets.
4. **The one new day with size is a loss.** 2020-12-28 (COVER exploit) is idiosyncratic, and L
   loses there (-48% on the second entry).

## Caveats

- **2020-21 are not out of sample in time** (2022 fold backwards). The point there is the day
  count, not the per-trade mean.
- **Months were fetched selectively.**
  - A pair's series has month gaps, padded with `common.load`'s 300 flat candles. The
    detector's one-week warmup applies only at the start of a series, not after a gap.
  - As in h2, the liquidity filter (qv_base >= 5,000 USDT) removes baseline artefacts after a
    gap, but a few spikes right after a gap may carry a short baseline.
- **Converted high/low** use the quote's close (h2's approximation). For stablecoin quotes the
  factor is ~1.
- **Spikes end on 2026-10-28** (the fetched months end with 2026-10). This does not matter,
  since there are none after 2023.
