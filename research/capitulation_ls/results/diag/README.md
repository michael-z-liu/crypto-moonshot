# Numerical diagnostics: how to read them, and what they say

`diag.py` puts every strategy on the same scales (user request 2026-11-27 23:20Z):

| Measure | Range | Meaning |
| --- | --- | --- |
| signal | 0..1 | strength of the entry signal (a percentile within the candidate pool) |
| threshold | 0..1 | the signal a candidate needs to be traded |
| outcome o | -1..+1 | net return on the stake after costs, clipped; -1 = stake lost |
| tp_progress | 0..1 | best move during the hold / take-profit level (no TP: / 20%) |
| stop_progress | 0..1 | worst move during the hold / stop or liquidation level (none: / 100%) |
| giveback | 0..1 | share of the best move given back by the exit |

Failure modes (one per trade): `tp_win`, `time_win`, `near_miss` (lost after reaching half the
take profit), `never_went` (lost without reaching half of it), `went_against` (same, but the price
went half-way to the stop), `stopped` (stop, liquidation or o <= -0.95). `contrib_<mode>` is the
mode's share of trades x its mean outcome; the contributions add up to E[o].

Files: `scorecards.md` (one row per strategy) and `calibration_<strategy>.csv` (outcome by signal
decile over traded and below-threshold candidates, the latter simulated as if entered).
`d01_scorecards.py` rebuilds them for the sleeves L, P, F, C, M.

## What the numbers say (2022-26, every candidate trade, no slot limit)

**L, capitulation longs: a working signal with a sharp threshold.** E[o] +0.129, 85% wins, 65%
of trades reach the +20% take profit, none is stopped; its only failure mode is `never_went` (9%).
The calibration is a hockey stick: outcome is about 0 for CNN signals 0.950-0.976, +0.026 at
0.976-0.984, +0.059 at 0.984-0.9955, and +0.132 above 0.9955 (take-profit hit rate 14% -> 67%).
The edge lives only in the extreme tail of the signal, which is why trades are rare (11.5% of the
pct >= 0.95 pool, about 25 flush days in five years). Improving L means more events with signal
>= 0.995, not better exits.

**M, weak-listing shorts: strong and calibrated, with a stop tail.** E[o] +0.200, 78% wins, 37%
reach the 50% take profit; 15% are stopped (contribution -0.080, about 40% of the edge). Listings
below their listing price earn +0.11..+0.37 a trade; above it -0.06..+0.08 with 30-45% stopped.

**P, pump shorts: no usable signal.** E[o] +0.025. The calibration is flat across all pump sizes
(-0.025..+0.020 by decile) while the stopped share rises from 0-1% to 8% with the signal: a
bigger pump buys reversion and squeeze risk in equal measure. The 7% stopped trades contribute
-0.075, three times the whole edge; the median trade gives back 52% of its best move.

**F, pump shorts with the funding gate: the gate halves the tail.** E[o] +0.095, stopped 5%
(contribution -0.049), giveback 34%. The gate does not change the pump-size calibration; it removes
the crowded-short squeezes.

**C, idiosyncratic crash shorts: weak and fragile.** E[o] +0.044. 27% of trades `never_went`
and the median gives back 67% of its best move; the moves are small (median tp_progress 0.66 of a
20% reference). The strongest "no bounce" decile is worse (stopped 7%), so the signal is not
monotonic.

## How to use this to improve

| Symptom in the numbers | Lever |
| --- | --- |
| Calibration flat (P) | the signal carries no information: find a different signal, do not tune exits |
| Large negative `contrib_stopped` (P, M) | tail control that does not sell retraces (gates like funding, position sizing), not tight stops |
| High `never_went` and giveback (C) | exit timing / shorter holds, or a better entry signal |
| Hockey-stick calibration (L) | more qualifying events (wider universe, more data) rather than a lower threshold |
