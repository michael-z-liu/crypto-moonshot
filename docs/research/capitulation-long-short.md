# Capitulation long/short: searching for a much more profitable version

Unattended search, started 2026-11-27, for changes to the capitulation strategy that reach an
**average trade of +10% or more** and **+50% or more a year** for a $10k-50k account, trading
both directions (longs after dumps, shorts on USD-M perpetuals after rallies). Plan and run log:
`docs/plans/capitulation-long-short.md`. Code: `research/capitulation_ls/` (`e01_` .. `e43_`,
`sim.py`, `wf.py`, `dsr.py`, `fetch_futures.py`, `build_*.py`, `nn_seq.py`).

> Status: written mid-search and updated as it continues. The pre-registered forward read-out
> (2026-11-02..27) found no trades: see [Forward read-out](#forward-read-out).

## Summary

1. **Both targets are reached in the 2022-26 backtest by a three-sleeve book, without
   leverage.** "LMC" shares 10 equal slots between:
   - **L, capitulation longs** (spot): the day-weighted walk-forward CNN of the #10 study at its
     top 0.5% (`day5` pct >= 0.995), take profit +20%, time exit 72 hours;
   - **M, weak-listing shorts** (perp, isolated 1x): new Binance spot listings that trade at or
     below their listing price 72 hours after listing; take profit 50%, stop 50%, 30 days;
   - **C, idiosyncratic crash shorts** (perp, isolated 1x): a coin down 15%+ in 4 hours while BTC
     moved less than 1% and the CNN expects no bounce (pct <= 0.5); 24 hours.

   | LMC, 10 slots, $10k each year | 2022 | 2023 | 2024 | 2025 | 2026 (to Nov) |
   | --- | --- | --- | --- | --- | --- |
   | Return | +62% | +81% | +156% | +625% | +51% |
   | Max drawdown | -10% | -7% | -16% | -17% | -11% |

   +11.6% a trade on 430 trades; double costs +11.3%; longs one candle late +10.7%.
2. **Longer holds are the single biggest change.** Prior studies held 1-2 hours for +0.5-1.5% a
   trade. The same CNN signal held up to 72 hours with a +20% take profit makes +10.6% a trade
   (every year positive, drawdown -14%), and +7-11% across all 13 walk-forward score sets.
3. **Shorting pumps is real but squeezes cap it.** On spot prices, shorting +40% pumps looked
   like +10% a trade; on perp prices with honest tails (isolated 1x liquidation at +100%) it is
   +3% a trade, a waterfall of the perp-spot gap, liquidation, funding and cost (see
   [Why it did not work](#why-it-did-not-work)). 6% of pump shorts see the price double within
   72 hours, and squeezes are barely predictable. It only works when BTC has not fallen over the
   past week (+6.5% a trade, every year positive).
4. **Leverage does not survive flash crashes.** 2x on longs, even on majors, is wiped out by the
   2025-10-10 wicks. The targets are met at 1x gross; longs belong on spot.
5. **On the forecaster's per-coin fidelity:** within a day, the CNN's forecast band ranks coins'
   realised 8h moves at IC 0.22 (trailing volatility alone 0.21; the model adds a partial 0.13);
   its 10-90% band holds 73% of outcomes (too narrow; widen 1.2x for 80%). A transformer ranks
   pumps better within a day (IC 0.34 vs 0.25 for trees), but better ranking did not turn into
   better books, because books are decided by tails and by which days trade.
6. **Caveats.** Every sleeve and setting was chosen on 2020-26 data that earlier studies had also
   seen. The deflated Sharpe ratio is 0.90-0.98 (5,000 / 1,000 trials). The long sleeve's
   evidence rests on about 25 flush days; 2025 dominates the book (listing shorts). Only forward
   data can confirm any of this.

## Robustness checks on LMC

| Check | Result |
| --- | --- |
| Double costs | +11.3% a trade, worst year +48% |
| Longs one candle (5 min) late | +10.7% a trade, worst year +42% |
| Entry 1-2 minutes after the assumed open (1m candles) | costs at most ~0.3% a trade |
| Other walk-forward score sets for L (volscale5, path_all5, 10 seeds) | +10.9..+11.7% a trade, worst year +29..+52% |
| Settings chosen on 2022-24 only, read on 2025-26 | 2025 +544%, 2026 +89% (+8.6% / +5.5% a trade) |
| 400 random configurations from plausible ranges | 96% positive every year; median +10.0% a trade, median worst year +41%; both targets in 14% |
| Without each year's best flush day for L | +10.6% a trade, worst year +21% |
| Without each year's top 3 listing shorts | +10.0% a trade, worst year +32% |
| Starting each year with $50k / $250k | worst year +50% / +53% (2025's compounding meets the volume cap) |
| 7-day block bootstrap of daily returns | one-year return p5 +26%, median +129%; P(>= +50%) 86% (68% without 2025) |
| Deflated Sharpe ratio | 0.98 (1,000 trials) / 0.90 (5,000); annualised Sharpe 2.16 |

The per-trade target is typical of the whole region; +50% every year is the binding target and
depends on catching the one or two biggest flush days of a year.

## Forward read-out

The pick was written into the plan before any forward outcome was computed. The day5 2026 fold
was saved (`save_day_model.py`, reproducing day5's 2026 scores exactly) and every liquid spike
from 2026-11-02 to 11-27 was scored (`forward_signals.py day5`). `lmc_forward.py day5`: **no
trades.** The highest long score was 0.990 (no flush), no crash met the crash-short rule, and all
18 new spot listings were tokenised stocks or ETFs without perps. BTC did fall about 15% in three
days around 2026-11-14/15, but as a slow slide without 5m capitulation spikes, so the long sleeve
stayed out by design. The window can neither confirm nor reject LMC; the paper trade going
forward is the test (rerun the two scripts after a sync).

## What changed against the earlier capitulation studies

| Change | Effect |
| --- | --- |
| Hold 72h with a +20% take profit instead of 1-2h | +1% -> +10.6% a trade for the top 0.5% |
| Shorts on perps (funding, liquidation, perp prices) | a second and third sleeve; spot proxies overstate shorts |
| Survivorship-free universe, perp availability check (traded volume in the last 2 hours) | removes frozen and delisted perps |
| Per-year $10k books with a liquidity cap (0.5% of 24h volume) | yearly returns that do not depend on earlier compounding |
| Worst concurrent adverse move check | shows why >1x gross is unsafe |

## Findings by sleeve

### L: capitulation longs (E20-E24, E29-E33, E38)

- Region, not a point: pct 0.995-0.997, take profit 15-30%, hold 24-72h, 10 slots: +9..+16% a
  trade, every year positive. Walk-forward choice of these settings is noisy (one to three
  earlier years), but 112 of 200 configurations are positive in every year.
- In the 10-slot book 81% of the 162 trades win and 60% reach the take profit (85% / 65% over all
  1,883 candidates); the worst are systemic collapses (LUNA -100%, FTT -70%) and the 2024-04-12
  cascade. 1,803 of 1,883 candidate trades fall on 11 broad flush days.
- Did not help: a CNN trained on the realistic TP20/72h target (+5.2% vs +10.6%), a GBM or
  transformer on that target (they learn which years are good, not which coins), ensembles,
  10 seeds instead of 5, the 2h head, per-day top-K selection, a second position per coin,
  fading the bounce after the take profit, leverage. Trailing stops, other take profits and
  forecast-based exits were not better, but with ~23 independent flush days none of the exit
  variants is statistically distinguishable from the +20% take profit (see `results/why/`).

**Network architectures with the same day-weighted training on all 1.85M spikes (E50, 3 seeds
each; CNN 5 seeds):**

| Network | L at pct >= 0.995 | LMC a trade | LMC worst year |
| --- | --- | --- | --- |
| CNN (day5, the pick) | +10.6% | +11.6% | +51% |
| Patch transformer | +10.3% | +11.7% | +53% |
| Dilated TCN | +9.6% | +11.6% | +53% |
| Conv + GRU | +9.1% | +11.0% | +31% |

Newer networks match the CNN at best: the bounce signal looks limited by the data, not the model.

### M: weak-listing shorts (E12, E13, E18, E27, E28, E36)

- New Binance spot listings with a perp fall: shorting 72 hours after listing for 30 days makes
  +14% a trade; restricting to listings at or below their listing price makes +20%, better in
  every year 2021-26.
- Shorting every new perp (perp-only listings included) is weak (+1.5-3.5%) and noisy.

### C: idiosyncratic crash shorts (E42)

- A coin crashing alone (BTC calm) keeps falling when the CNN expects no bounce: +4.4% a trade
  over 24 hours, every year positive. Sensitive to the BTC-calm threshold (2% is weaker).
- A third of its trades are coins Binance later delisted (+8.5% a trade against +2.6%): part of
  the edge is selling after delisting announcements.

### P: pump shorts (E01-E19, E34, E37)

- Realistic reference (r4 >= 40%, perp, isolated 1x, 72h, 10 slots): +3.1% a trade, +7..+26% a
  year. With BTC 7d > -3%: +6.5% a trade. Stops at 50% hurt; delayed entry costs edge; open
  interest, long/short ratios and funding do not predict squeezes.
- A book with P (LQM) also meets both targets (+11.4% a trade, every year >= +53%) but with a
  -36% drawdown against LMC's -17%.

### Other ideas that did not work

Shorting every new perp listing; weekly shorts of young coins; shorts or longs against extreme
funding; open interest, long/short ratios and funding as squeeze predictors; pump shorts picked
by a low CNN score; capping or re-weighting sleeves; BTC-trend filters on listing shorts; longer
listing holds inside the book.

## Why it did not work

Each failure was investigated separately (notes in `research/capitulation_ls/results/why/`,
scripts `w1_`..`w6_`). The common thread: the edges here are event-driven and rest on few
independent days, so any rule that needs fine distinctions inside those days drowns in noise.

**Pump shorts** (`why/pump_shorts.md`). Perps do *not* squeeze more than spot (doublings 6.2% vs
6.0% of trades, 23 of 24 shared). The drop from +11% (spot, losses floored) to +3% (perp, honest
tails) is a cost waterfall: perp-vs-spot gap -3.9 points (a third of it one frozen, delisting perp,
ALPACA), liquidation at +100% -1.4 (17% of doublings later retrace below entry), funding -2.4
(shorts pay heavily in squeezes; entries at negative funding rose from 14% of pumps in 2024 to 60%
in 2026), cost -0.3. Stops sell the retraces the trade lives on: 75% of trades that touch +50%
against end better than -50%. Models mostly rank pump size, which drives both the reversion and the
squeezes, so ranking skill does not survive the tails. Deduplicated to one entry per pump, the
"2025-26 edge" is about zero; the one real squeeze signal (crowded shorts, a low long/short account
ratio) is too weak to trade.

**Leverage** (`why/leverage.md`). The edge and the tail are the same trades: the signal picks the
coins with the deepest wicks, and the wick is what the +20% take profit later collects. At 1x the
wick costs nothing (79% of trades that went 33%+ against finished profitable); with leverage it is
paid first. 2025-10-10 produced 58% of 2025's candidates and 159 of the 187 with a 33%+ adverse
move; the wicks were real (414 pairs, median -66% on 1m, close -21% an hour later; 12 of 19 majors
wicked 50%+). The slots filled with the first spikes, before the low. The largest leverage that
avoided liquidation over 2022-26 was 0.99x.

**Long-sleeve variants** (`why/long_variants.md`). On crash rows the day explains 56% of the
TP20/72h outcome but only 17% of the 1h outcome, so models trained on the realistic target learn
regimes, not coins (yearly IC up to 0.57, within-day 0.06-0.12). The TP20-target CNN is fine on the
CNN's own days (+11.6%) but trades 48 extra losing days. All five day-weighted networks share one
within-day IC (0.12-0.13; their ensemble 0.124), about 1.6% of within-day variance: the 8h of
candles hold no more coin-level information. Exit variants are not distinguishable with ~23
independent flush days.

**Listing variants** (`why/listings.md`). The edge belongs to the Binance *spot* listing: perp-only
coins have a fifth of the volume and barely decline (median 30d -6% vs -22%). The weak filter mostly
cuts the squeeze tail (stop-outs 15% vs 38%) and partly bets on a falling market. Decay is smooth
and front-loaded (median -18% at 3 days, -51% at 30 days), so the 72h entry peak is mostly noise on
that decline. 2025's many weak listings are mostly perp coverage: 97% of 2025 spot listings had a
perp at listing vs 6-32% in 2021-23.

**Regime and other ideas** (`why/regime_other.md`). Extreme funding persists because it follows the
price trend (funding collected is outweighed by price). Moderate BTC drawdowns are neither bounce
nor continuation; only the four -20% events won. The 2026-11-14 slide was a real 5m flush on the
majors but scored just below the entry bar and its bounce faded. The BTC-calm sensitivity of the
crash short rests on 25 trades with three tail losses. Majors do not bounce less (the +4.6% vs +10%
gap compared different rules). Squeezes are coin-specific jumps, not time clusters, so slot caps
cannot cut them; the one lead is a delisting-notice feed.

**Cascade timing** (`why/cascade_wait.md`). Skipping the first minutes of each signal episode
fails because there are only 28 episodes in five years and 16 last under 20 minutes; its apparent
gain is 2025-10-10 alone. The time to the low ranges from 4 to 642 minutes, so there is no clock to
wait for. Capping the slots the first minutes may take fails too (`why/early_slot_cap.md`): the
first 15 minutes of episodes hold 60% of the long sleeve's net, and 11 of 13 capped episodes have
no later signals to fill the freed slots.

## Data added (Binance only, `fetch_futures.py`)

USD-M perpetual funding rates (2.85M rows, 919 symbols), 1h perp klines (16.5M rows), 5m perp
klines for 450 perps with pump events, futures metrics (open interest, top-trader and account
long/short ratios, taker ratio; 6.3M 5m rows on event days), spot 5m candles synced to
2026-11-27. About 3 GB of the 20 GB allowed.

## Caveats

- **The 72h listing entry is a sharp peak.** With the listing short entered at 24/48/96/120h
  instead, LMC makes +9.7..+10.3% a trade with a worst year of +24..+32%; only 72h gives +51%.
  Treat ~+10% a trade and a worst year near +30% as the fair expectation, not +51%.
- **In-sample choices.** Sleeves, filters (weak listing, BTC calm, BTC 7d) and settings were all
  found on 2020-26. 2025-26 had already been opened by three earlier studies.
- **Few independent events.** The long sleeve trades on 5-7 days a year.
- **Execution.** Entries at the next 5m open; one candle late costs the long sleeve ~1% a trade.
  Shorts assume perps are borrowable and the isolated-margin liquidation at +100% fills there.
- **2025 dominates.** LMC's 2025 (+625%) comes mostly from listing shorts in a year of many weak
  listings; 2026 (to November) is +51%.

## Recommendations

1. Paper-trade LMC forward (testnet for the perp shorts) before funding; compare fills with the
   5m opens assumed here.
2. Keep gross exposure at 1x; no per-position leverage.
3. If anything is built first, build the long sleeve (L) on spot and the weak-listing short (M):
   they carry most of the edge with the least tail risk.
