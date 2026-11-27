# Quiet-coin breakouts (x2_quiet_*.py)

## Hypothesis
Coins that were quiet for weeks (low realised volatility and volume) and then break out of their
20-day range on a volume surge trend for days. Long the upside breaks (spot), optionally short the
downside breaks (perp). Builds on B3 (quiet alone, and quiet + 5m spike, did not work).

## Method
- Hourly bars from the 5m candles, 582 coins (universe "all"), 2020-2026. Candidate = hourly close
  above the prior 20-day high (below the low for shorts), up/down candle, hourly quote volume >= 3x
  the prior 21-day median, 21-day median volume >= $2k/h. One candidate per coin, side and 3 days.
  44,746 candidates (2021-26: ~42.5k). Entry at the open of the next hour.
- Quiet = 0.5*log(21d vol / prior 69d vol) + 0.5*log(21d median volume / prior 69d median volume),
  both measured before the breakout bar. Signal (0..1) = sqrt(quiet pct * breakout pct), quiet pct =
  1 - rank of that quiet measure, breakout pct = mean of the ranks of volume surge and breakout size.
  Ranks cut on 2020-23 candidates (in-sample), per side.
- Exits on hourly paths: hold 1/3/7/14 days, TP 20/30/50%/none, stop 15/25%/none (48 configs per
  side); stop wins a same-hour tie. Costs: spot long 0.4%; perp short 0.3% + funding assumed 0.03%
  a day; isolated 1x short liquidated at +100% (net -1). Thresholds tried: 0, .5, .6, .7, .8, .9.
  576 config x threshold cells per side with n >= 150 (results/explore/x2_grid.csv).
- In-sample = 2021-23 (selection); 2024-26 read afterwards. Book: $10k per year, 10 slots, stake =
  equity/10 capped at 0.5% of prior 24h volume, first come first served.
- Not modelled: perp availability for the short coins (all candidates assumed shortable), real
  funding, intra-hour order (hourly highs: TP fills assumed at the limit price).

## Results (mean net per trade; years 2021..2026)
| Config | n | avg | median | per-year mean net | 10-slot book per year (return / trades) |
| --- | --- | --- | --- | --- | --- |
| Long, 14d, TP30, stop25, signal >= 0.5 (in-sample pick) | 8070 | +0.1% | -3.5% | +5, -5, +3, +2, -4, -1 % | +65%/337, -94%, -27%, -64%, -91%, -58% |
| Long baseline: same exits, every breakout (no quiet filter) | 20659 | +1.4% | -1.9% | | +520%/462, -95%, +6%, -54%, -95%, -33% |
| Short, 7d, TP30, signal >= 0.9 (in-sample pick, n small) | 167 | +5.9% | +8.3% | nan, +7, +10, +8, -1, -5 % | 0, +18, +36, +19, -3, -12 % |
| Short baseline: same exits, every breakdown | 22310 | +0.7% | +0.5% | | -65, +172, -33, -15, +140, +52 % |

Signal bins, long (TP30/stop25/14d): mean net by signal 0-.3: +3.1%, .3-.5: +1.5%, .5-.6: +1.2%,
.6-.7: +1.0%, .7-.8: -1.2%, .8-1: -2.7% (median -9.8%). Hit rate of TP30 about 26-33% in every bin.
Quiet component alone (long): 21d volatility ratio < 0.7: +3.4% (n 5.5k) vs >= 1.0: +1.2%; volume ratio < 0.7: +1.0% vs >= 1.0: +2.4%.
Best raw grid cell for longs, 14d no stop style, never above +1.1% in-sample, -0.8% in 2024-26.

## What worked / did not
- Nothing reaches +10% per trade or +50% every year. Upside breakouts, quiet or not, average about
  zero after 0.4% costs: a 30% TP is hit about 27% of the time, the 25% stop about as often.
- The more extreme the "quiet + breakout" signal, the worse the long: top bins lose (median -10%).
  Breakouts on very quiet coins are the thin, pump-like ones and mean-revert (consistent with B3: quiet
  coins drift down). Only mild quiet (volatility ratio < 0.7) is marginally better than loud.
- The baseline's only strong stretch is the 2021 bull market (+520% book), and it fails every
  other year; 2022, 2025, 2026 lose 30-95%. The result is regime, not signal.
- Downside breakdown shorts at signal >= 0.9 look good in-sample (+9.7% avg 2021-23, n 167 over
  3 years) but fade to +2.4% in 2024-26 (2025 -1%, 2026 -5%) and are selected from 576 cells x
  2 sides; with funding assumed and perp availability unchecked this is not a finding.

## Caveats
Hourly bars: TP and stop fills at the limit price, no slippage beyond the flat costs; stop-first
tie rule; short funding is a flat assumption; first-come book ordering; many same-day candidates
(cascade days) make books lumpy. Signal ranks are in-sample (2020-23).

## Why it failed
Breakouts after quiet periods are mostly thin-market pumps that retrace; the quietness measure
anti-selects (higher signal means worse). The long side has no edge net of 0.4% cost, and the
one short corner that looks good is small and not stable out of sample.

Files: x2_quiet_build.py (candidates + paths), x2_quiet_eval.py (signal, grid), x2_quiet_final.py
(books, trade files), results/explore/x2_grid.csv; trades at
`$OWN/explore/quiet_trades.parquet` (4 configs), `quiet_nonentries.parquet` (20k below-threshold long candidates).
