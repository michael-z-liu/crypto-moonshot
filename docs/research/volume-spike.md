# Volume-spike strategy research

Overnight exploration, 2026-11-02, of the user's 2019 idea: *a sudden spike in 5-minute volume
precedes a price jump, so buying the spike and taking a small profit wins more often than it
loses.* Code and result tables are in `research/volume_spike/`; the run log is in
`docs/plans/volume-spike-research.md`.

## Summary

1. **The 2019 rule loses money, in every year, at every holding period.** Buying 5m volume
   spikes on up bars, mostly bought by takers, with no prior run-up, and exiting at +1% / −5%
   wins 52–67% of trades but loses about 0.3% per trade after a 0.3% round-trip cost (79k trades,
   2020–2026). A high win rate with a small target and a wide stop is exactly the trap: a few
   stops wipe out many small wins.
2. **A volume spike on its own predicts nothing.** Across 2M spikes, the price drift over the
   next 1–4 hours is within about ±5 bp of random bars, far below the 30 bp it costs to trade.
3. **What the spike *does* tell you depends on what happened before it.**
   - After a sharp **pump**, a spike marks exhaustion: −50 to −90 bp over the next 4 hours.
   - After a sharp **crash**, a spike marks capitulation: the price bounces. The deeper the crash,
     the bigger the bounce, and the bounce is much bigger than after a crash with no spike.
4. **The capitulation bounce is a real, large edge in 2020–2024, but only during market-wide
   flushes.** It works when BTC is also crashing (BTC volume spike, or BTC down 2%+ in 4h); a coin
   crashing alone (hack, unlock, delisting) does not bounce.
5. **It decayed badly in 2025.** The untouched 2025–2026 test period is roughly break-even for the
   moderate versions and only modestly positive for the most extreme ones. 2026 so far looks
   better than 2025, but there is too little data to call a recovery.
6. **Honest headline:** a walk-forward version (parameters re-chosen every six months from the
   previous two years only) made about +20% a year from 2022 to 2026 with a −16% worst drawdown,
   trading on about 40 crash days a year, and was flat in 2025. That is promising enough to
   paper-trade, not to fund.

![Deciles](../../research/volume_spike/results/deciles.png)

## Method

- **Data:** Binance spot 5m candles, 2020-01 to 2026-11, for the 198 USDT pairs downloaded with
  `moonshot data sync`. Stablecoins, fiat, gold, wrapped assets and tokenised stocks are dropped,
  leaving 152 coins. Each coin's first week is skipped.
- **Spike:** a closed 5m candle whose volume is at least 10× the median 5m volume of the previous
  24 hours (about the top 3% of candles; 4× turned out to be 10% of all candles).
- **No look-ahead:** every feature uses candles up to and including the spike candle. The trade
  enters at the **next candle's open**. Stops are checked before targets inside a candle.
- **Costs:** 0.1% fee + 0.05% slippage per side, 0.3% round trip; stressed at 0.6% and 1.0%.
- **Splits:** train 2020–2023, validation 2024, test 2025-01 to 2026-10. The finalists were
  written into the plan (commit `7d26050`) before the test period was looked at.
- **Portfolio simulation:** 5 equal slots, one position per coin, compounding; simultaneous
  signals are taken deepest-crash first. Because crash signals cluster on a few days, results are
  also reported **per crash day** (an equal-weight basket of that day's signals), which is less
  sensitive to which coins happen to grab the slots.

## Findings in detail

### The original rule and the raw spike (train 2020–2023)

| Entry                                   | Exit                | Trades  | Net per trade | Win rate |
| --------------------------------------- | ------------------- | ------- | ------------- | -------- |
| Every spike                             | +1% / −5% / 4h      | 775k    | −31 bp        | 60%      |
| Every spike                             | best of 125 exits   | 775k    | −22 bp        | 42%      |
| Random 5m bar                           | best of 125 exits   | 118k    | −24 bp        | 42%      |
| 2019 rule (up bar, taker buys > 60%)    | +1% / −5% / 4h      | 42k     | −30 bp        | 57%      |

Spikes on up bars, the core of the 2019 idea, drift *down* afterwards (−12 bp over 4h, gross).

### Capitulation

Gross 4h return after a spike, by the coin's prior 24h return (train): the bottom decile is
+47 bp versus +15 bp for random bars after the same drop; the top decile is −86 bp. Inside the
crash subset, bounces are bigger when the drop is fast (last 1–4h), the spike candle has a huge
range, the coin is at its 24h low, BTC is crashing too, and the coin is liquid. Extreme taker-buy
shares (either way) are worse than middling ones.

The big wins are real moves, not bad ticks: 2020-03-12, 2021-05-19, 2021-12-04, 2022-05/11,
2024-08-05 and similar flushes, where liquid coins fell 20–40% and bounced. Dropping the 20 busiest
crash days still leaves +2.4% per trade.

### Finalists (5 slots, 0.3% cost)

| Rule | Train 2020–23 | Valid 2024 | **Test 2025–26** |
| ---- | ------------- | ---------- | ---------------- |
| **F1** spike, coin −8% in 4h, ≥ 5k USDT/bar, BTC volume ≥ 6×, hold 1h | +148 bp/trade, 5.2/wk, CAGR 114%, DD −15% | +140 bp, CAGR 112%, DD −7% | **−3 bp, CAGR −5%, DD −21%** |
| **F2** spike, coin −10% in 4h, ≥ 20k USDT/bar, BTC volume ≥ 6×, hold 1h | +273 bp, 2.7/wk, CAGR 110%, DD −10% | +284 bp, CAGR 105%, DD −7% | **+15 bp, CAGR 0%, DD −22%** |
| **F3** spike, coin −15% in 4h, ≥ 5k USDT/bar, hold 4h | +366 bp, 1.6/wk, CAGR 81%, DD −11% | +566 bp, CAGR 103%, DD −6% | **+42 bp, CAGR +5%, DD −24%** |

![Equity](../../research/volume_spike/results/equity.png)

Per crash day, which is the steadier measure, F2 made +185 to +360 bp in each year 2020–2024 with
79–91% of crash days positive; +50 bp (2025) and +78 bp (2026) with about 60% positive.

![Per year](../../research/volume_spike/results/yearly.png)

### Robustness checks (2020–2024)

- Positive in every calendar year 2020–2024, for all three finalists.
- Survives a 1.0% round-trip cost (F3 with a 4h hold: +405 bp → +335 bp per trade).
- Works on coins listed before 2020 and on newer ones (checked for F3 and a broader variant),
  so it is not only new-listing chaos.
- **Speed matters:** entering one candle (5 minutes) later loses about a third of the edge.
- Stops, targets and trailing stops did not beat a plain time exit; 1h holds were best.
- Limit orders below the spike close do not help: they fill only on the trades that keep falling
  and miss the immediate bounces (adverse selection).
- Majors only (BTC, ETH, SOL, BNB, XRP and 10 others) during flushes: +120 to +350 bp per trade
  each year 2020–2024 and 2026, negative in 2025.

### What happened in 2025

- Only the most extreme flushes kept working (coin −15% or more in 4h, or BTC −5% or more).
  Moderate dips kept dipping, especially in coins listed less than a year earlier.
- Market-regime filters (equal-weight alt index 7/30/90-day trend) and volatility-scaled crash
  depth did not fix it.
- Results inside 2025 swing from −0.5% to +3% per trade with small changes such as the number of
  slots: the year is decided by which handful of coins get bought on a few crash days.
- 2025-10-10 (the liquidation cascade with 80–99% wicks on some Binance alts) is not what drives
  the result either way; excluding it moves the test numbers by 5–20 bp per trade.

### Walk-forward (the number to believe)

Every six months, the rule with the best t-statistic over the previous 24 months (grid: drop 8–20%,
liquidity 5k–50k, BTC volume filter 0–10×, hold 1h or 4h) is traded for the next six months, as an
equal-weight basket per crash day with half of equity committed per day:

| Period            | Crash days | Net per crash day | Days positive |
| ----------------- | ---------- | ----------------- | ------------- |
| 2022              | 34         | +276 bp           | 79%           |
| 2023              | 27         | +107 bp           | 74%           |
| 2024              | 56         | +132 bp           | 77%           |
| 2025              | 74         | +15 bp            | 53%           |
| 2026 (to Oct)     | 23         | −99 bp            | 57%           |
| **2022–2026**     | **214**    | **+87 bp**        | **66%**       |

Equity ×2.44 over 4.8 years (CAGR about +20%), worst drawdown −16%.

## Caveats

- **Survivorship bias.** The universe is today's top 199 coins. Coins that crashed and never came
  back (LUNA, FTT and many 2021 alts) are missing, which flatters any dip-buying rule. The real
  edge is smaller than shown, by an unknown amount.
- **Crash-time execution.** The backtest buys at the next candle's open plus 0.05% slippage. During
  a flush, spreads widen, order books thin out and Binance has throttled or frozen before
  (2021-05-19). The cost stress (1%) helps, but real fills on the worst days could be much worse.
- **Few independent bets.** Hundreds of trades, but only 20–70 crash days a year; one bad week
  can decide a year.
- **Multiple testing.** Several hundred rule variants were tried on 2020–2024 before the
  finalists were fixed. The 2025–2026 test, the only clean check, is the weak part.
- **Decision timing.** Signals use closed 5m candles. A live bot must act within seconds of the
  close, because a one-candle delay costs about a third of the edge.

## Recommendation and next steps

1. **Do not trade the 2019 rule.** The data rejects it decisively.
2. **Paper-trade the capitulation idea, don't fund it yet.** If anything is built, build F2 (or the
   walk-forward version) in `paper` mode on the live stream for a few months: it would show real
   fill prices on flush days, which is the biggest unknown.
3. **Get faster data for flush days.** 1m candles (or trades) around past flushes would show how
   much of the bounce happens inside the 5m spike candle and whether a faster trigger recovers the
   2025 edge.
4. **Measure crash-time slippage.** Record order-book depth and spread for liquid coins on the next
   few flush days before sizing anything.
5. **Fix survivorship before trusting any dip-buying result:** add delisted symbols to the data
   pool (Binance's public data archive, data.binance.vision, may still have them).

## Reproduce

```bash
uv sync --group research
uv run python research/volume_spike/build_events.py   # event table, ~15 s
uv run python research/volume_spike/build_long.py     # 24h paths for crash events
uv run python research/volume_spike/m2_baseline.py    # raw spike vs random, exit grid
uv run python research/volume_spike/m3_features.py spike
uv run python research/volume_spike/m6_test.py        # finalists on all periods, equity.png
uv run python research/volume_spike/m7_walkforward.py t 30
uv run python research/volume_spike/plots.py
```

Set `MOONSHOT_DATA_DIR` if the candles are not in the default location the scripts expect
(`research/volume_spike/common.py`).
