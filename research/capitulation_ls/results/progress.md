# Search scoreboard (updated 2026-11-27 22:30 UTC)

Started 13:46 UTC; hand-off not before 2026-11-28 09:46 UTC. 49 experiment scripts, 9 "why it
failed" investigations, 67 commits on `feat/capitulation-long-short`.

Targets: average trade >= +10%, every year >= +50%. Backtest 2022-26, per-year $10k books, 1x.

## Works (in-sample)

| Strategy | Avg trade | Worst year | Max DD | Status |
| --- | --- | --- | --- | --- |
| **LMC** = L + M + C, 10 slots | **+11.6%** | **+51%** (~+30% at neighbouring settings) | -17% | pre-registered pick |
| LMCF = LMC + F, 10 slots | +11.2% | +95% | -32% | found after pick (in-sample) |
| L: capitulation longs (CNN top 0.5%, TP 20%, 72h, spot) | +10.6% | +15% | -14% | sleeve |
| M: weak spot-listing shorts (perp, 30d) | +20.0% | +7% | -24% | sleeve |
| C: idiosyncratic crash shorts (perp, 24h) | +4.4% | +5% | -21% | sleeve |
| F: pump shorts, r4 >= 40%, funding >= 0 (perp, 72h) | +9.5% | +5% | -26% | sleeve (new, 22:50) |

## Tried and did not work (each explained in `results/why/`)

| Area | Ideas | Why (short) |
| --- | --- | --- |
| Pump shorts | plain, stops, delays, models, OI/LS/funding predictors, low CNN score | cost waterfall on perps (gap, liquidation, funding); stops sell retraces; models rank pump size |
| Leverage | 2x/3x alts and majors, stops | one wick day (2025-10-10); survivable leverage 0.99x |
| Long-sleeve variants | trade-aligned target, GBM, transformer/TCN/GRU, 10 seeds, ensembles, exits, top-K, pyramiding, vol sizing, BTC hedge, cascade timing, early-slot cap | day explains 56% of the outcome; one shared coin-level ceiling; ~23 flush days |
| Listing variants | all perps, young coins, triggers, delays, late perps, regime filter, longer holds | edge is the spot-listing event; front-loaded decay; 72h peak mostly noise |
| Regime / other | funding extremes, market basket after BTC drops, majors, sleeve caps/weights, weekday | crowded funding trends; moderate drops keep falling; squeezes coin-specific |

## Forward (unseen) test

2026-11-02..27: 0 trades for LMC (no flush, no lone crash, no new crypto listings). Rerun
`forward_signals.py day5` then `lmc_forward.py day5` after a candle sync.

## How the search is running

Work happens in bursts: on each 20-minute heartbeat, when a background training job finishes,
or when an investigator sub-agent reports. Between those, nothing runs.
