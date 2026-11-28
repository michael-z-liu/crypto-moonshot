# Holdout 2017-08 .. 2019-12 on BTC-quoted pairs: the long sleeve L where the coins really traded

The USDT holdout (h1, `holdout_2018.md`) found L's edge out of sample, but on a small sample:
41 trades on 18 days. Only 6 / 21 / 58 USDT pairs existed in 2017 / 2018 / 2019, and most alts
then traded only against BTC. This study repeats the test on those BTC pairs (XXXBTC). Their
candles are converted to USDT terms, and only coin-times with no USDT pair are used, so no coin is
counted twice.

**The rule is unchanged.** Same spike builder, same saved day5 2026 fold (and the 2022 fold as a
check), and the same L: pct >= 0.995, entry at the next 5m open, take profit +20%, time exit 72h,
one position per coin. Book: 10 slots, a fresh $10k each year, stake <= 0.5% of 24h quote volume.

Scripts:
- `h2_fetch.py`: download and conversion;
- `h2_build.py`: spikes, CNN inputs, outcomes;
- `h2_score.py`: scoring;
- `h2_eval.py`: trades, books, calibration, combined holdout;
- `h2_parity.py`: does the conversion change outcomes?;
- `h2_robust.py`: post hoc robustness.

Data and full tables are in `$DATA/cache/capitulation_ls/holdout/btc/` (`eval_tables.md`,
`parity.txt`, `robust.txt`).

## Verdict

**L holds on the BTC pairs too, at full size. The evidence now covers 44 flush days instead of
18. The edge is real out of sample, but in this holdout it is dominated by the 2017-12 .. 2018-01
altcoin top and crash.**

- **140 trades on 38 flush days. Mean +1574 bp at 0.4% cost** (median +1960), 91% winners, 84%
  hit the take profit. In the wide-cost scenario it is **+1494 bp**: 1.0% round trip on the alt/BTC
  pair plus 0.1% each way on BTCUSDT, 1.2% in all. 26 of the 38 days are not in the USDT holdout.
- Day-bootstrap 95% CI of the trade mean: [+1223, +1779] bp at 0.4% and [+1143, +1699] bp at 1.2%.
  33 of 38 days are positive.
- **Combined holdout** (USDT + BTC pairs): 181 trades on 44 days. Mean +1433 bp at 0.4%, CI
  [+1114, +1660]. With USDT trades at 1.0% and BTC-pair trades at 1.2%: +1358 bp, CI
  [+1041, +1583]. No resample is <= 0.
- **Hockey stick again, and steeper.** By fixed pct bin, per spike: -80 / +128 / +297 / +762 /
  +1660 bp. Spearman(signal, outcome) over the pct >= 0.95 pool is +0.35 (+0.15 on USDT pairs).
- **The CNN adds value over a naive dip rule.** Buying any liquid spike after a >= 20% 24h fall,
  with L's exit: +701 bp. Buying every liquid spike: -93 bp.
- **The conversion is not the source of the result.** On 731 USDT-pair spikes whose coin also
  had a BTC pair, the converted path and the native USDT path give the same mean net outcome
  (+421 vs +421 bp). For the 90 L-grade spikes it is +1088 vs +1048 bp (converted 40 bp
  optimistic), and the take profit agrees in 96% of cases.
- **The weak point is concentration.** 104 of the 140 trades fall in 2017-12 .. 2018-01, and
  half are on two days: 2018-01-11 (22 trades) and 2018-01-16 (28).
  - Without those two days: +1476 bp, CI [+1063, +1768].
  - Without both months: 36 trades on 21 days, +856 bp at 1.2%, CI [+220, +1461] bp. That is
    still positive and close to the USDT holdout's +954 bp.
  - 2019 is thin: 6 trades.

So the out-of-sample per-trade edge looks solid, and the hit rate and size match 2022-26
(+1291 bp, TP 0.65) or exceed them. The yearly book numbers below are inflated by a one-off
mania regime and are not a forecast.

## Data coverage

- **Source:** data.binance.vision monthly 5m zips, sha256-checked. Zips were deleted after
  conversion, and the raw folders were removed.
- **Pairs:** the archive has 191 plain `XXXBTC` pairs with months in 2017-08 .. 2020-01.
  Stablecoin, wrapped and gold bases are excluded. 40 pairs are dropped because their coin had a
  USDT pair in every month (e.g. ETH, ATOM, MATIC, DOGE). That leaves **151 BTC pairs and 3,077
  symbol-months**, with no fetch failures. Months fetched: those without a USDT pair, plus the
  next month for exit paths. Delisted pairs are in the archive and included (HSR, ICN, MOD, SUB,
  BCD, SALT, TNB, ...).
- **De-duplication:** a spike is kept only if its open time is before the coin's first USDT candle
  or after its last. This removes 1,083 of 54,053 liquid spikes.
- **Conversion** at the same 5m open time with BTCUSDT from h1's store:
  - open = open_btc x BTCUSDT open, close = close_btc x BTCUSDT close;
  - **high / low = high_btc / low_btc x BTCUSDT close** (an approximation: the two intrabar
    extremes need not coincide), widened to cover open and close;
  - quote volume = BTC quote volume x BTCUSDT close; base volume and trade count unchanged;
  - values kept at float32 precision so the store compresses.
- **Inputs:** BTC and ETH channels come from BTCUSDT and ETHUSDT. Market channels 11-12 are built
  over the combined universe (h1's 89 USDT pairs plus the 151 converted pairs). Scoring them
  instead with h1's USDT-only market series changes little: raw correlation 0.965, L agreement
  99.8%, L +1555 bp on 105 trades.
- 21 rows with non-finite inputs (the 2018-02-09 outage) get pct NaN, as in h1.

| year | liquid spikes | coins | median qv_base (USDT / 5m) | USDT holdout (h1) |
|---|---|---|---|---|
| 2017 (from 08-24) | 5,007 | 86 | 9,620 | 438 spikes, 6 coins |
| 2018 | 38,896 | 143 | 9,753 | 8,002, 21 |
| 2019 | 9,067 | 112 | 7,758 | 27,477, 58 |

Share of spikes at pct >= 0.995: 2.5% / 0.38% / 0.07% by year, against 0.25% in 2022-26. The 2026
fold sees 2017 as extreme, as in h1.

## L trades by year (day5 2026 fold)

| year | n | mean net 0.4% (bp) | mean net 1.0% | mean net 1.2% | median | TP rate | win | days | new days vs USDT holdout | coins |
|---|---|---|---|---|---|---|---|---|---|---|
| 2017 (Aug-Dec) | 48 | +1404 | +1344 | +1324 | +1960 | 0.83 | 0.85 | 19 | 14 | 31 |
| 2018 | 86 | +1718 | +1658 | +1638 | +1960 | 0.87 | 0.95 | 14 | 9 | 57 |
| 2019 | 6 | +871 | +811 | +791 | +609 | 0.33 | 0.83 | 5 | 3 | 4 |
| **2017-19** | **140** | **+1574** | +1514 | **+1494** | +1960 | 0.84 | 0.91 | 38 | 26 | |
| USDT holdout (h1) | 41 | +954 | | | +813 | 0.37 | 0.88 | 18 | | |
| 2022-26 reference | 1,883 | +1291 | | | | 0.65 | 0.85 | 25 | | |

- Exits: 117 take profits (+1960 bp each) and 23 time exits (mean -391 bp).
- Unlike h1 (no loss above 10%), there are real losers. The worst are on 2017-09-04, the China ICO
  ban: HSR -33%, WTC -33%, NEO -11%.

**Second scorer, day5_2022 fold** (older reference, stricter in practice): 45 trades on 8 days,
+1723 bp at 0.4% (+1643 at 1.2%), TP 0.93, CI [+893, +1960]. It has no 2019 trades.

## Calibration (day5, 0.4%)

Per spike, by fixed pct bin (no per-coin lock):

| pct bin | n | mean net (bp) | TP rate | USDT holdout (h1) | 2022-26 |
|---|---|---|---|---|---|
| 0.90-0.95 | 2,335 | -80 | 0.23 | -19 | -34 |
| 0.95-0.976 | 1,192 | +128 | 0.31 | +56 | +86 |
| 0.976-0.984 | 431 | +297 | 0.35 | +126 | +355 |
| 0.984-0.9955 | 554 | +762 | 0.53 | +511 | +742 |
| 0.9955+ | 248 | +1660 | 0.87 | +1072 | +1348 |

By decile of the pct >= 0.95 pool, mean outcome: -0.026, +0.002, -0.013, +0.000, +0.028, +0.016,
+0.058, +0.052, **+0.137, +0.157**. It is flat near zero, rises gently, then jumps in the top two
deciles: **a hockey stick**, with a less flat handle than on USDT pairs. Neighbours are monotone:
- pct >= 0.99: +1207 bp (274 trades);
- 0.995: +1574;
- 0.998: +1668;
- TP 15 / 25%: +1276 / +1724.

## 10-slot book ($10k fresh each year)

| year | BTC pairs 0.4% | BTC pairs 1.2% | combined 0.4% | combined (USDT 1.0% / BTC 1.2%) | combined trades | max DD (combined 0.4%) |
|---|---|---|---|---|---|---|
| 2017 (Aug-Dec) | +86.6% | +80.1% | **+87.2%** | **+80.6%** | 49 | -8.2% |
| 2018 | +130.6% | +121.6% | **+146.2%** | **+135.9%** | 61 | -1.7% |
| 2019 | +5.3% | +4.8% | **+19.0%** | **+16.9%** | 28 | -1.2% |

- USDT pairs alone (h1): +10.2% / +15.6% / +13.0%.
- The 2017-18 books come almost entirely from 2017-12 .. 2018-01. Without those two months, the
  BTC-pair book at 1.2% makes +6.6% (2017) and +20.5% (2018).

## Robustness (post hoc, day5, `robust.txt`)

| check | n | mean net | note |
|---|---|---|---|
| as tested, 0.4% | 140 | +1574 | |
| highs lowered 1% of price, 1.2% cost | 139 | +1429 | TP rate 0.80 |
| highs lowered 2% of price, 1.2% cost | 139 | +1359 | TP rate 0.76 |
| without 2018-01-11 and 01-16, 1.2% | 90 | +1396 | CI (0.4%) [+1063, +1768] |
| without 2017-12 .. 2018-01, 1.2% | 36 | +856 | CI [+220, +1461], P(<= 0) 0.004 |
| qv_base 5-10k USDT / 5m, 1.2% | 53 | +1677 | |
| qv_base 10-30k, 1.2% | 61 | +1398 | |
| qv_base >= 30k, 1.2% | 26 | +1344 | |
| 1.2% + one satoshi tick | 140 | +1461 | CI [+1099, +1673] |
| entries with tick > 0.2% of price, 1.2% + tick | 15 | +793 | coarse-tick coins are worse |

## Caveats

- **Conversion approximation.** Highs and lows use the BTC-pair extreme times the BTCUSDT close,
  so 5m extremes carry BTC's intrabar move as noise.
  - In overlap months, converted closes differ from the native USDT pair by about 0.1% (median),
    with p99 0.4-0.8% (more for BNB in 2017-11, when its USDT pair had just listed).
  - On outcomes the error is small and slightly optimistic: +40 bp on L-grade spikes, TP
    agreement 96%.
  - The haircut test shows that lowering every high by 2% still leaves +1359 bp.
- **Trading in BTC terms.** The test assumes the alt is bought with BTC that was itself bought with
  USDT at entry and sold at exit. The 1.2% scenario charges that leg. It ignores the BTCUSDT bid-ask
  at the 5m open (minor) and assumes both legs fill at the next 5m open.
- **Survivorship.** The archive includes delisted pairs (HSR, ICN, MOD, SUB, SALT, ...).
  - Pairs absent from the archive cannot be checked.
  - The archive starts in mid-2017, so coins that died before then are missing.
  - The pair list comes from what the archive keeps today.
- **Liquidity and spreads.** Median trade-day quote volume is 1.5-6M USDT, so a $1k slot is
  0.02-0.07% of a day's volume, and the book caps stakes at 0.5%. Real 2017-18 BTC-pair books were
  thin, though:
  - a take profit at exactly +20% on a 5m high may not fill in full;
  - coins priced at a few satoshis had ticks of 1-12% of price.
  - The 1.2% and tick scenarios cover a spread, not market impact in a crash.
- **Regime.** 2017-12 .. 2018-01 was a one-off: a peak altcoin mania followed by violent crashes
  with fast rebounds. It holds 74% of the trades. The rest of the holdout (21 days) is positive
  but much smaller.
- **Scorer reference.** pct is measured against the 2020-25 spike distribution. The 2026 fold
  puts 2.5% of 2017 spikes above 0.995, ten times the 2022-26 share.
