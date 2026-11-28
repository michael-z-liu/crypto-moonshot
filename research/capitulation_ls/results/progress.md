# Search scoreboard (final, 2026-11-28 11:11 UTC)

Started 2026-11-27 13:46 UTC; hand-off not before 2026-11-28 09:46 UTC. 133 commits on
`feat/capitulation-long-short`: 51 experiments, 12 "why it failed" investigations, 17 explorers,
2017-19 holdouts (USDT and BTC pairs), a perp-only extension and an independent audit (`README.md` maps them).

Targets: average trade >= +10%, every year >= +50%. Backtest 2022-26, per-year $10k books, 1x.

## Works

| Strategy | Avg trade | Worst year | Max DD (closed / marked) | Status |
| --- | --- | --- | --- | --- |
| **LMC** = L + M + C, 10 slots | **+11.6%** | **+51%** (~+30% at neighbouring listing delays) | -17% / -19% | pre-registered pick (in-sample) |
| **LMC+** = LMC + L on perp-only coins | +11.5% | +62% (2026 +111%) | -17% | found after the pick; walk-forward scores |
| LMCF = LMC + F | +11.2% | +95% | -32% | found after the pick (in-sample) |
| L: capitulation longs (CNN top 0.5%, TP 20%, 72h, spot) | +10.6% | +15% | -14% | sleeve |
| L on perp-only coins (perp long 1x) | +13.1% | every year > 0 | | sleeve extension |
| M: weak spot-listing shorts (perp, 30d) | +20.0% | +7% | -24% | sleeve |
| C: idiosyncratic crash shorts (perp, 24h) | +4.4% | +5% | -21% | sleeve |
| F: pump shorts, r4 >= 40%, funding >= 0 (perp, 72h) | +9.5% | +5% | -26% | sleeve |

## Out-of-sample evidence

| Test | Result |
| --- | --- |
| L on 2017-08..2019-12 (data no study had seen) | USDT pairs: +9.5% a trade, 41 trades, 18 flush days. With BTC-quoted pairs: 181 trades, 44 flush days, +14.3% (CI +11.1..+16.6%); book +87% / +146% / +19% (mostly the 2017-18 mania) |
| Settings frozen on 2022-24, read on 2025-26 (corrected by the audit) | LMC 2025 +371%, 2026 +54% (+7.5% / +3.8% a trade); LMC+ +498% / +128% |
| Forward 2026-11-02..27 | LMC: 0 trades. Perp-only L: 1 trade (BRUSDT 11-21), +19.7% |
| Audit | no inflating bug in the headline; one look-ahead in the frozen check (fixed); drawdowns 2-6 points deeper marked to market |

## Tried and did not work (each explained in `results/why/` or `results/explore/`)

| Area | Ideas | Why (short) |
| --- | --- | --- |
| Pump shorts | plain, stops, delays, models, OI/LS/funding predictors, low CNN score, second pumps, baskets, perp-only coins | cost waterfall on perps (gap, liquidation, funding); stops sell retraces; perp-only coins squeeze 4x as often |
| Leverage | 2x/3x alts and majors, stops | one wick day (2025-10-10); survivable leverage 0.99x |
| Long-sleeve variants | trade-aligned target, GBM, transformer/TCN/GRU, ensembles, exits, top-K, pyramiding, vol sizing, BTC hedge, cascade timing, early-slot cap, limit entries, crowding, order-book depth, perp venue, less liquid coins, other quote pairs 2020-26, post-delisting perps | the day explains most of the outcome; the CNN already ranks within a day; extra coins help only if they add flush days |
| Listing variants | all perps, young coins, triggers, delays, late perps, regime filter, longer holds, months 2-3, listing-day and strong-listing longs | edge is the spot-listing event; front-loaded decay; first candles are near the top |
| Crash shorts | exits, 31 entry features, perp-only coins | a few large continuations carry it; nothing at entry predicts them |
| Book level | sleeve momentum, loss brake, calendar effects | exposure, one squeeze cluster, chance |
| New families | quiet breakouts, lag catch-up, premium extremes, contagion, large-cap reversal, flush-loser baskets | flat calibration or overlap with L |

## Leads (not confirmed)

- Long pumps on perp-only coins: +4.7% a trade, positive in 69 of 72 settings, but a lottery (median -14%), two years only, and the best version lost on the pre-spot coins.
- Pause pump shorts for a few days after a 30% loss: cut LMCF's drawdown -32% -> -16..-24%, all from one 2025 cluster.
