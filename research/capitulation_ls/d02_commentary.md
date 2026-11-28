## Reading each failed variant (baseline first)

Baselines: L_day5 (+0.129, 85% wins, 65% TP, no stops), P_nostop (+0.025), C_calm1 (+0.044), M_e27 (+0.200).
Calibration shapes are the E_o-by-decile lines above; "mode" is the failure mode with the largest drag.

1. **Trade-aligned CNN and other architectures (L_tp20day, L_day_transformer / tcn / gru).** Per trade
   they are as good as day5 or better (E_o +0.155 / +0.121 / +0.124 / +0.114; tp20day wins 82% with only 2%
   `never_went`) and all four calibrations are the same hockey stick (top decile +0.12..+0.16, rest under
   +0.06). So the per-trade scorecard does not show why tp20day lost in the book (+517 vs +1064 bp at q 0.995,
   -26 vs +247 bp at q 0.98); what differs is where the events fall: 937 of its 1,299 entries are in 2025
   (day5: 1,059 in 2024 and 581 in 2025) and 2026 has 4 trades at -0.15, so it fires on different and fewer
   flush days. Dominant failure: not a trade mode but event selection on ~25 days; the architectures add
   nothing over day5. Implication: the signal is capped by the number of flush days, so do not tune the
   model; widen the event set (more coins / data).
2. **Lower threshold (day5 >= 0.99 and >= 0.98).** E_o falls monotonically as the threshold drops: +0.129 ->
   +0.100 (0.99) -> +0.047 (0.98); TP share 65% -> 50% -> 28%, `never_went` 9% -> 13% -> 25%, giveback 0.15 ->
   0.25 -> 0.59, and 2022 turns negative (-0.03) at 0.98. The pool below the threshold is ~0 (+0.008..+0.012,
   54% wins, 39% `never_went`, giveback 0.88). Calibration: hockey stick (flat ~0 below the 0.98 CNN
   signal, +0.026 / +0.059 / +0.132 in the top three deciles). Lowering theta only mixes in the flat part;
   lever: more qualifying events, not a lower threshold.
3. **Pump shorts with a 50% stop (P_sl50) vs none (P_nostop).** The stop makes the dominant failure worse:
   stopped share 7% -> 19%, contrib_stopped -0.075 -> -0.100, stop_progress median 0.19 -> 0.38, and E_o
   +0.025 -> +0.012 (2022 and 2025 at -0.00). Most pumps that go +50% later revert (giveback 0.52 -> 0.60), so
   the stop sells retraces; `never_went` falls 12% -> 5% (the stop converts them into stops). Calibration is
   flat in both (decile E_o -0.025..+0.020, Spearman +0.105 / +0.063, and the Spearman drops with the stop).
   Implication: flat signal, so no exit rule can fix it; tail control must be position sizing or gates (the
   funding gate F), not a tighter stop.
4. **Crash shorts with BTC calm 2% (C_calm2) vs 1% (C_calm1).** E_o +0.044 -> +0.028 (2022-23: +0.04 / +0.01
   instead of +0.07 / +0.05); 61 more entries (198 vs 161) are filler: the added trades have the same ~27-28%
   `never_went` and giveback 0.67. Calibration is not monotonic (a bump in decile 9, the top decile ~0 in both:
   -0.009), Spearman +0.106 / +0.151. Dominant mode: `never_went` plus giveback, not stops (2% stopped).
   The widening admits crashes that are market-driven; the signal does not separate them. Implication: a
   better entry signal or shorter holds; the BTC gate is a real filter but the CNN score is not monotonic.
5. **Leverage on the capitulation longs (lev 2, 3 vs 1x).** Per entry, on the stake, the levered sleeves look
   better, not worse: E_o +0.129 -> +0.178 (2x) -> +0.261 (3x); the cost is the stopped mode: 0% -> 8% -> 12%
   (contrib_stopped -0.081 / -0.122), and the below-threshold pool gets -0.018 / -0.049 from liquidations
   (worst year -0.115 at 3x). Calibration stays a hockey stick (top decile +0.178 / +0.263). The failure is
   therefore not the per-trade expectancy but correlated ruin: 95% of the trades sit on the 11 busiest days,
   25 entry days in all, and the worst day averages -0.40 (1x), -0.52 (2x), -0.62 (3x) per trade, i.e. every
   position in a flush is liquidated together; a per-trade scorecard cannot show that, the book's drawdown can.
   Implication: not an exit or signal problem; leverage needs the day-level (portfolio) view and sizing.
6. **Listing shorts.** On all new perps with e18 `trades(72, 30, 0.5, 0.5)` (E18_all_perps) E_o is +0.023: 28%
   stopped (contrib -0.148, about 6x the whole edge), giveback 0.57, stop_progress median 0.45. Restricting to
   perps already at or below their first price (E18_weak_perps) gives +0.055 and still 24% stopped; the perps
   above it get -0.024 with 34% stopped. The calibration is monotone but shallow (deciles -0.100..+0.108,
   Spearman +0.128), against the spot-listing M_e27: +0.200, 15% stopped, `never_went` 1%, 37% TP. The perp-listing
   pool fails on the stop tail, not on missing moves (median tp_progress 0.59-0.62). Young-coin weekly shorts
   (Y_young_all / weak): +0.011 / +0.010, no TP and no stops to blame (1% liquidated): 28% `never_went`,
   median giveback 0.72-0.74, and the calibration is inverted (Spearman -0.055: the weakest vs listing price
   earn least, deciles +0.017..-0.012). Dominant failure: no signal (flat/inverted) on 5,550 trades; the decay
   is already harvested in the first month (M).
7. **Quiet breakouts.** Long (30d, TP 30% / stop 25%): entries E_o +0.001, below-threshold +0.022 (the filter
   anti-selects), unfiltered baseline +0.014; Spearman -0.062 and the calibration is inverted, decile E_o
   falling from +0.036 to -0.023. Failure mode: stopped 22-24% plus `went_against` 18% (stop_progress median
   0.56-0.59, giveback 1.00 median: the breakout is given back in full); TP is hit in 27-29%, which is not
   enough against a 25% stop. Years positive 0.50. Shorts: the unfiltered 7d baseline is +0.007 (41%
   `never_went`); the filtered Q_short has only 167 trades, +0.059, calibrated flat inside the entries (no
   below-threshold pool, so no verdict). Implication: inverted signal on the long side: find a different signal.

## Ranking of failure causes across all variants

1. **No information in the signal (flat or inverted calibration), 5 variants**: pump shorts (P, both stops), young-coin
   shorts (inverted), quiet breakout longs (inverted), perp listings on all coins, C with a wider BTC band.
   Cost: the whole edge; exits cannot fix it. Lever: change the signal.
2. **Edge only in the extreme tail of a hockey-stick signal (L variants, 5 rows)**: below the threshold the
   outcome is ~0 and 39% `never_went`; lowering theta from 0.995 to 0.98 loses 64% of E_o and the other CNNs
   are the same curve. Lever: more flush events (data / universe), not a lower threshold or a new model.
3. **Stop and liquidation tail on shorts (P_sl50, E18, M, quiet longs)**: stopped 19-28% (contrib -0.10 to -0.15),
   2-6x the edge. A tighter stop worsens it (7% -> 19%). Lever: gates (funding, weak vs listing price),
   sizing, not stops.
4. **Concentrated, correlated risk invisible per trade (leverage, L)**: 25 flush days hold 95% of trades;
   the 3x sleeve has the best per-trade E_o (+0.261) yet loses in the book because positions are liquidated
   together (worst day -0.62). Lever: day-level sizing / drawdown view, not the trade scorecard.
5. **Giveback and `never_went` on weak calm-market shorts (C, young coins)**: 27-29% `never_went`, giveback
   0.67-0.74, E_o +0.01..+0.04. Lever: shorter hold / better entry.
