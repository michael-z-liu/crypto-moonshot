# Capitulation long/short: search for a much more profitable version

- **Status:** building
- **Branch:** `feat/capitulation-long-short`
- **PR:**
- **Started:** 2026-11-27T13:46Z. Keep searching until at least **2026-11-28T09:46Z** (20 h)
  before hand-off; budget about 25 h (five usage windows).
- **Updated:** 2026-11-27T14:56Z

## Goal

Find what has to change in the capitulation strategy (buy liquid coins on 5m volume spikes during
market-wide flushes, ranked by a walk-forward CNN) to reach an **average trade of +10% or more**
and **+50% or more a year**, for a $10k-50k account (no market impact). Extend it to **both
directions**: long after capitulation dumps, short (USD-M perpetuals) after blow-off rallies. Make
the forecaster tell coins apart instead of predicting the average. This is a wide, adaptive search:
architectures, signals (Binance data only), training windows, exits (take profit / stop grids),
sizing and leverage. Keep trying new ideas, however unlikely, for at least 20 hours.

## Out of scope

- Live, testnet or paper trading code; futures/margin account mechanics; anything on mainnet.
- Data from outside Binance (data.binance.vision and the public Binance APIs only).

## Ground rules (fixed before any result)

- **Universe:** "all" (582 USDT pairs incl. delisted), as in the robustness and #10 studies.
- **Costs:** spot 0.1% fee + 0.05% slippage + 0.05% latency per side (0.4% round trip). Perps:
  0.05% taker + 0.05% slippage + 0.05% latency per side (0.3% round trip) plus funding paid or
  received over the hold. Stress at 2x cost and one candle late.
- **Leverage:** the research decides (user, 2026-11-27). Every result is reported unlevered, and
  the final candidates also at 1x-5x on perps with the liquidation price checked on the path
  (a position whose adverse move reaches 1/leverage minus maintenance margin is liquidated).
- **Periods:** development on the walk-forward years 2021-2024. 2025-26 (to 2026-11-02) has been
  seen by three studies: it is used as a gate each idea family must pass, read once per family and
  logged. Data after 2026-11-02 is unseen: it is read once, for the final pick only.
- **Ledger:** every trial is one row in `research/capitulation_ls/results/ledger.csv` (id, idea,
  2021-24 per-year bp, 2025-26 bp, trades, return, drawdown). The final pick gets a deflated
  Sharpe ratio over all trials.
- **Disk:** at most 20 GB of new data (user, 2026-11-27). Check free space before each download.

## Milestones

### M0: Harness, baseline and scorecard (done)

- **Delivers:** `research/capitulation_ls/` with a fast event-level simulator (long and short,
  TP/SL/time exits on 5m paths, slots, leverage), the scorecard (bp per trade, yearly return,
  drawdown, within-day fidelity: rank correlation of forecast and outcome among one day's
  candidates), the ledger, and the baseline (path_all and day5 with a 2h hold) reproduced.
- **Done when:** the baseline matches the #10 study's 2025-26 bp within noise; the check command
  passes.
- **Worker:** master

### M1: Binance futures data (in progress)

- **Delivers:** USD-M perp 5m klines, funding rates, premium index and metrics (open interest,
  long/short ratios) for every coin with a perp, from data.binance.vision; aligned features per
  event; synced spot candles to the latest day.
- **Done when:** coverage table per coin and year; features join the event table without
  look-ahead.
- **Worker:** master (or sonnet for the downloader)

### M2: Short side (in progress)

- **Delivers:** pump/blow-off event study, rules and models for perp shorts, with funding.
- **Done when:** short-side scorecard for rules and the best model, ledgered.
- **Worker:** master

### M3: Per-coin forecast fidelity (todo)

- **Delivers:** models that separate coins: distribution heads, in-event ranking losses,
  event-only training, coin embeddings, transformers / TCN / GRU / state-space, GBM, stacks;
  training windows (expanding, rolling, recency-weighted).
- **Done when:** fidelity and portfolio scorecards for each, ledgered.
- **Worker:** master

### M4: Big-trade hunting (todo)

- **Delivers:** longer holds (hours to days), TP/SL grids, forecast-driven exits, rarer and
  deeper events, concentration, Kelly sizing, leverage, combined long/short book.
- **Done when:** the best books per target (10% per trade, 50% a year) are found and ledgered.
- **Worker:** master

### M5: Open idea loop (todo)

- **Delivers:** whatever the earlier milestones suggest, until the 20-hour mark: liquidation
  cascades (open-interest drops), extreme funding, listings, order book depth, regime switching,
  time of day, spot-perp basis, and more.
- **Worker:** master

### M6: Final pick, forward read-out, report, PR (todo)

- **Delivers:** a pick written down and committed before the unseen data is read; the read-out;
  `docs/research/capitulation-long-short.md`; PR.
- **Worker:** master

## Decisions

- 2026-11-27: Leverage is decided by the research; shorts are modelled on USD-M perps; up to 20 GB
  of new data (user answers at scoping).
- 2026-11-27: The user asked for a wide search (newer architectures, other techniques beyond CNNs,
  other signals, training windows, TP/SL grids) and to adapt to findings: widen the scope where a
  discovery points, keep going and keep trying. Milestones M2-M5 are a menu, not a fixed order.
- 2026-11-27: Work lives in a new `research/capitulation_ls/` that reads the existing volume-spike
  caches (`research/volume_spike/cache_all`, symlinked to the shared data folder) instead of
  rebuilding them.

## Run log

Note: the entry times from "~13:52Z" up to the entry marked "(real time ...)" ran ahead of the
clock; those entries all happened between 13:46Z and about 14:50Z. Later times are real.

- 2026-11-27T14:05Z: plan written; caches found (seq2 1.85M liquid spikes 2020-01..2026-11-01,
  8h and 72h forward paths, i10 walk-forward scores in the AUhKx worktree).
- ~13:52Z E01-E03: event landscape. Longer holds change the picture: one entry per event, shorts
  after +40% 4h pumps average +5% at 24h and +11% at 72h (median +12.5% at 24h), positive every
  year 2021-26; longs on the top 0.5% day5 scores average +11% at 24h. Simulator reproduces the
  #10 study's day5 1h result (+98 bp over 2025-26).
- ~13:56Z M1: perp funding (2.85M rows, 919 symbols) and 1h perp klines (16.5M rows) downloaded;
  spot 5m synced to 2026-11-27 (unseen forward window from 2026-11-02).
- ~13:58Z E05/E07: pump-short grid and walk-forward (per-year $10k books, perp-gated, funding):
  WF pick +86/+134/+157/+236/+551% in 2022-26, +327 bp/trade, worst DD -55%. Hindsight best
  (thr 0.4, hold 72h, 5 slots) >= +66% every year at +10%/trade, DD -29%.
- ~14:00Z E08: GBM walk-forward ranking of pumps: IC 0.13-0.31 every year; top 20% book +820
  bp/trade, positive every year, DD -53%. Crash longs with GBM: unstable, mostly losing.
- ~14:02Z E09: perp vs spot over the same hours: median gap 0.2% but delisting squeezes (ALPACA
  +584% perp vs +28% spot) and frozen perps break the spot proxy. Perp gate now requires traded
  volume in the last two hours; shorts will be re-simulated on 5m perp paths (download running).
- ~14:05Z E10: short entry timing: immediate entry is best; each delay costs edge (5m late
  -25 bp, 1h late -200 bp) but stays profitable even 4h late. Rule r4>=0.4 on the active-perp
  gate: +1020 bp/trade, +62..+400% a year 2022-26, DD -29%.
- ~14:06Z nn_seq/E11: tabular MLP IC 0.15-0.28 (= GBM); GBM+MLP blend IC 0.17-0.34, within-day
  IC +0.30 (the models do separate coins on the same day). As a veto on the rule set they add
  almost nothing (+1020 -> +1050 bp).
- ~14:07Z E12/E13: new listings fall: short 1-3 days after spot listing on the perp, hold 7-30
  days: +10..+15% a trade, positive most years (few trades 2021-23). Second short component.
- ~14:15Z E14-E16 (correction): on perp paths the pump short is weaker (r4>=0.4, 72h: +979 bp
  spot -> +656 bp perp), and the earlier books floored losses at -100% while keeping positions
  that went +100% against them. 6.5% of pump shorts see a +100% move within 72h (ALPACA's
  delisting squeeze: -516%). Realistic tails: isolated 1x (liquidated at +100%) or cross margin
  (losses beyond the stake). Then r4>=0.4, 10 slots, 72h: +260 to +310 bp a trade, worst year
  about -6%..+7%; cross margin with 5 slots loses a whole year to one squeeze. Stops at 50% hurt.
- ~14:20Z E17: squeezes are not predictable (AUC 0.42-0.63 by year, metrics add nothing).
  Realistic pump-short reference: iso1x, 10 slots, 72h: +313 bp, +7..+26% a year, DD -33%.
  Transformer ranker (3 seeds) IC 0.33/0.18/0.30/0.24 for 2022-25: better than GBM/MLP.
- ~14:25Z E18: shorting every new perp listing is weak (+1.5..+3.5% a trade); the edge is in new
  Binance *spot* listings that have a perp (E13: +10..+14%).
- ~14:28Z E19: GBM trained on the realistic target (perp short return, -100% at +100% adverse):
  top 10% +449 bp, -1..+74% a year, DD -28%. The pump-short edge is mostly 2025-26.
- ~14:32Z E20-E22 (main long finding): capitulation longs with the day-weighted CNN at
  pct >= 0.995, take profit +20%, time exit 72h, 10 slots: +10.6%/trade, +15..+103% a year
  2022-26 (all positive), DD -14%, ~33 trades a year. Broad region (q 0.995-0.997, TP 15-30%,
  24-72h): +9..+16%/trade. Survives 0.8% cost (+10.2%), perp paths (+10.9%), one candle late
  (+8.0%). Across 13 walk-forward score sets: +7..+11%/trade. Per-position leverage is a trap:
  2x/3x isolated longs are wiped out by the 2025-10-10 wicks (2025 -89%).
- 14:20Z E23/E24: walk-forward choice of the long rule's settings is noisy (day5 WF +260 bp,
  CAGR +36%; path_all5 WF +1744 bp on 62 trades, +30%); 112 of 200 configs positive every year.
  Ensembles of score sets (mean or min percentile) give +11..+18%/trade on fewer trades.
- 14:30Z E25 (combined book): sleeves L (capitulation longs), P (pump shorts, iso1x), N (spot-
  listing shorts) share one slot pool. LN, 15 slots, gross 1.0: +1271 bp/trade, +38/+54/+97/
  +238/+35% in 2022-26, DD -18%. Gross 1.5: min +55% but the worst concurrent adverse move of
  the long sleeve reaches 104% of equity on 2025-10-10, so >1x gross under cross margin can be
  wiped out; longs belong on spot (no liquidation), shorts on isolated 1x perps. LPN 15 slots:
  +877 bp, min +49%, DD -36%. All sleeve settings were chosen in-sample.
- 14:30Z E26: a GBM trained on the realistic long outcome (TP 20%, 72h) ranks crash events
  poorly within a day (IC +0.06) and concentrates its picks in 2022/2025; the day5 CNN stays the
  better within-day ranker for longs.
- 14:35Z E27/E28: listing shorts work best on listings already below their listing price at
  entry (+14.3% -> +20.0% a trade, better in every year). On perp-only listings the same filter
  helps (+2.3% -> +5.5%) but stays noisy.
- 14:37Z E25 again with N2 = weak spot listings ("M"): LM 10 slots +1523 bp, +43/+71/+136/+494/
  +30%, DD -14%; LPM 10 slots +934 bp, every year >= +54%, DD -38%; WPM (L at q 0.99, TP 15%)
  10 slots +836 bp, every year >= +63%, DD -38%.
- 14:40Z E29: capitulation longs on 20 majors: 1x is a steady sleeve (r4<=-10%, TP 10%, 24h:
  +457 bp, +15..+87% a year, DD -10%), but 2x/3x is wiped out in 2025 even on majors (Binance
  wicks on 2025-10-10 liquidated 5-6 positions). Leverage stays off the table for longs.
- 14:42Z E30: transformer on the realistic long target: yearly IC 0.15-0.57 but within-day IC
  0.12 (day5 0.13, blend 0.16); its picks cluster by year (it learns regimes). The day5 rule
  stays the long sleeve (+1193 bp on crash rows).
- 14:44Z E31: fading capitulation bounces after the TP (+15-30%) loses: bounces keep going.
- 14:46Z E32: trailing stops armed at +10-30% (hold 72h/168h) do not beat the fixed TP 20% / 72h
  (best +1209 vs +1088 bp on perp paths, with a weaker 2025).
- 14:50Z E33: a second position in a coin already held (deeper spike) adds nothing.
- 14:52Z E34: regime: pump shorts lose when BTC fell over the last 7 days (squeeze rallies);
  with BTC 7d > -3% they make +647 bp a trade, positive every year 2022-26 (+551/+766/+597/
  +478/+954). Longs work in every BTC regime. Weekday filters are not consistent.
- 14:55Z E25/E35 (candidate): LQM = capitulation longs (day5 >= 0.995, TP 20%, 72h, spot) +
  regime pump shorts (r4 >= 40%, BTC 7d > -3%, perp, isolated 1x, 72h) + weak spot-listing
  shorts (entry +72h if price <= listing price, perp, TP 50%, stop 50%, 30 days), one pool of 10
  equal slots, gross 1.0: +1137 bp a trade, +53/+101/+172/+518/+78% in 2022-26, DD -36%, worst
  concurrent adverse 41% of equity. Stresses: 2x costs +1105 bp (min +51%); one candle late
  +1029 bp (min +26%); both +997 bp (min +24%); other score sets for L +1091..+1133 bp; 15 slots
  +1159 bp, min +42%, DD -25%. In-sample choices throughout: only forward data can confirm.
- 14:55Z: training a day-weighted CNN (i10 harness) whose target is the realistic long outcome
  (TP 20% / 72h) via target_offset (wf_i10/tp20day).
- 15:05Z E36: waiting up to 30 days for a listing to fall below its listing price adds ~40
  trades but lowers the average (+1969 -> +1534 bp); keep the plain weak-listing rule.
- 15:07Z E37/E38: the realistic-target GBM adds little to regime pump shorts (+633 -> +723 bp);
  a take profit set from the CNN's own path forecast is not robust (2022 drops to +6%).
- 15:08Z: longs on perps (0.3% + funding) +1109 bp vs spot +1064 bp: equivalent; spot is safer.
- 15:10Z E39: deflated Sharpe ratio of LQM (10 slots): annualised Sharpe 2.11 on 1,766 days,
  skew 6.3; Sharpe sd across 68 related books 0.48. DSR 0.99 / 0.93 / 0.84 for 100 / 1000 / 4000
  trials. Moderate evidence, not proof (the robustness study's pick had DSR 0.97 and failed).
- 15:15Z E40 (the user's fidelity question): within a day, the day5 forecast's 8h band width
  ranks coins' realised |8h move| at IC 0.224 (trailing 24h vol alone 0.209; the band adds a
  partial 0.131 beyond it; blend 0.232); direction IC within a day 0.104. The 10-90% band holds
  73% of outcomes (too narrow, the "averaging" the user saw); widening it 1.2x gives 80%.
- 15:18Z E41 (concentration): 1,803 of the 1,883 long-sleeve candidate trades fall on 11 broad
  flush days (> 30 signals each); in the LQM book the long sleeve exits on 5-7 days a year, so its
  evidence is ~25 flush days, not hundreds of independent trades. Pump and listing shorts trade
  on 14-55 days a year and spread the book in time; 2025's +518% is mostly listing shorts.
- 15:22Z E42: new sleeve C, idiosyncratic crash shorts: coin r4 <= -15% while |BTC 4h| <= 1%
  and the CNN expects no bounce (day5 <= 0.5), perp, isolated 1x, 24h: +436 bp, every year
  positive (+5..+29%), DD -21% (sensitive to the BTC-calm threshold: 2% is weaker).
- 15:25Z E25: LMC (longs + weak listing shorts + crash shorts, no pump shorts), 10 slots:
  +1158 bp a trade, +62/+81/+156/+625/+51% in 2022-26, DD -17%, worst concurrent adverse 26%.
  8 slots: +1123 bp, min +59%, DD -22%. LQMC 10 slots: +967 bp, min +66%, DD -36%.
- 15:30Z E43: LMC stresses: 2x costs +1125 bp (min +48%); L one candle late +1071 bp (min
  +42%); C variants +1066..+1320 bp (min +37..+49%); other score sets for L +1094..+1141 bp
  (min +29..+35%). DSR 0.98 (1000 trials) / 0.90 (5000), annualised Sharpe 2.16, skew 11.8.
- 15:35Z tp20day (CNN with the realistic TP20/72h target via target_offset, day weights, 3 seeds,
  2 fixed epochs): much worse than day5 (q 0.995 +517 vs +1064 bp; q 0.98 -26 vs +247 bp). The
  long-horizon target is too noisy; the 1h path target learns a cleaner bounce signal that
  carries to 72h. Ensembles (min of day/day5/volday5) do not improve LMC either.
- 15:40Z: report draft `docs/research/capitulation-long-short.md`. Futures liquidation snapshots
  are not in the archive; order-book depth (2023+) and BVOL (2023-06+) are too short for ~25
  flush days, so not pursued.
- 15:45Z E44: weekly young-coin decay shorts (age 7-180 days) are weak (+1%/week, losing years);
  the listing decay is concentrated in the first month, which sleeve M already captures.
- 15:50Z forward_signals.py: 7,382 liquid spikes 2026-11-02..11-27 scored with the paper test's
  path_all 2026 fold (no outcomes computed). Parity with the paper ledger: corr 0.95 (market
  channels differ: shared 582-coin universe vs the paper's 395). No spike reaches pct 0.995 in
  the window: the long sleeve would not have traded (no flush), so the forward read-out can only
  speak to the short sleeves.
- 14:50Z (real time; earlier entries' times ran ahead of the clock) E45, choice on 2022-24 only:
  L q 0.99 / TP 10% / 72h, C r4 <= -10% / calm 1% / CNN <= 0.5 / 72h, M delay 24h / 30 days /
  TP 50% / stop 50%, 8 slots. Read after the choice: 2025 +544% (+857 bp, DD -29%), 2026 +89%
  (+545 bp, DD -26%). The annual target holds; the per-trade target does not (the 2022-24 choice
  prefers more, smaller trades). The weak-listing filter itself was fixed with 2025 in view.
- 14:55Z capacity: LMC from $50k each year: +61/+82/+156/+504/+50% (only 2025's compounding meets
  the 0.5%-of-volume cap); from $250k still >= +53% a year.
- 14:57Z C sleeve anatomy: 49 of 161 trades are coins later delisted (+851 bp, win 67%) vs
  +255 bp for the rest: part of the edge is post-delisting-announcement selling.
- 15:05Z E46 (how wide is the region?): 400 random LMC configurations from plausible ranges:
  96% positive every year; median avg trade +997 bp (10-90%: +815..+1237); median worst year
  +41% (10-90%: +17..+66%); median DD -20%. Both targets at once: 14% of configurations. The
  per-trade target is typical; >= +50% every year is the binding one and needs q >= 0.995,
  TP 25-30% and 6-8 slots.
- 15:12Z 2025 listing shorts are broad-based: 73 weak listings, median +40%, the top 10 carry
  28% of the sleeve's P&L (the rest average +20.7%); 2025 simply had many weak listings.
- 15:15Z forward window (2026-11-02..27, no outcomes read): no L signal (max pct 0.986), no C
  candidate, only 3 pump rows on 2 days, and listing shorts need 30 days to resolve. The window
  cannot confirm or reject anything; a forward paper trade over months is the test.

- 15:00Z E47: trading against extreme funding (short after rate >= +0.1%/+0.3%, long after
  <= -0.3%/-1%) has no consistent edge (means -305..+182 bp, signs flip by year): crowded
  positioning tends to persist.

- 15:02Z E48: 7-day block bootstrap of LMC's daily returns: one-year return p5 +26%, median
  +129%, p75 +211%; P(year < 0) 0.9%, P(year >= +50%) 86%. Without 2025: p5 +9%, median +73%,
  P(>= +50%) 68%. (In-sample returns: this shows dispersion, not a forecast.)

- 15:06Z E49: spot listings whose perp only arrives later add 5 trades in five years: negligible.

- 15:18Z forward read-out (pre-registered LMC, day5 2026 fold saved by save_day_model.py, which
  reproduces day5's 2026 scores exactly): **0 trades** from 2026-11-02 to 11-27. L: highest pct
  0.990 (no flush); C: no candidate; M: all 18 new spot listings were tokenised stocks/ETFs
  without perps. The window neither confirms nor rejects LMC; `lmc_forward.py day5` can be
  rerun as data arrives (results/forward.md).

- 15:25Z weak-listing shorts by BTC 7d / 30d trend at entry: no consistent split (signs flip by
  year on small samples); M stays unfiltered.

- 15:33Z per-day top-K selection for L (top 3-20 per day among pct >= 0.98, on days whose max
  pct >= 0.99 / 0.995): +536..+1049 bp, never better than the global pct >= 0.995 threshold
  (+1064 bp, all years positive); weaker 2022/2026.

- 15:55Z LMC + regime pump shorts with a sleeve cap (at most 2-3 of the slots): +929..+943 bp,
  min year +50..+54%, DD -35..-45%; the cap does not remove the squeeze drawdown (it clusters).
  LMC stays the better balance (+1158 bp, DD -17%).

- 16:35Z fragility: LMC without each year's best long-sleeve flush day: +1056 bp, years +21/+55/
  +128/+522/+32%; without the best 2 days: +1034 bp, min +17%; without each year's top 3 listing
  shorts: +1001 bp, min +32%. The per-trade target survives; >= +50% every year depends on
  catching the one or two biggest flush days a year (an outage on such a day costs the year's
  target).

- 16:50Z day10b (day-weighted CNN, 10 seeds, 112 min): L at pct >= 0.995 +1093 bp (day5 +1064),
  q 0.997 +1359 (day5 +1377); LMC with day10b +1168 bp (day5 +1158), years +62/+81/+164/+618/
  +52%. More seeds change nothing beyond noise; day5 stays the pick.

- 17:08Z pump shorts picked by a low CNN score (day5 <= 0.2-2%, perp, iso1x, with/without the
  BTC 7d gate): +135..+426 bp, DD -29..-56%; the lowest scores are the worst (most extreme
  pumps squeeze hardest). No better than the r4 rule.

- 17:12Z day5's 2h head (pct_h2) or h1/h2 blends for L: +1053..+1075 bp at q 0.995 (h1 +1064),
  nearly the same picks; no change.

- 17:18Z listing-short hold 45-90 days: +2159..+2307 bp a trade (30 days +2000), but in LMC the
  longer slot use lowers the worst year (45d +47%, 60d +42% vs +51%); keep 30 days.

- 17:25Z fill realism for L on 1m candles (all 1,883 candidate trades): one minute after the
  assumed entry (next 5m open) the price is on average 102 bp lower (median 19 bp higher), two
  minutes after 28 bp higher. Entering within 1-2 minutes of the spike close costs at most
  ~0.3%; a full 5m candle late costs ~2.7% (E21).

- 17:35Z sleeve weights (stake multipliers L/M/C): every tilt lowers the worst year (+39..+49%
  vs +51% equal); tilting to M inflates 2025 but raises DD and worst concurrent adverse (38%).
  Equal weights stay. Hour of day: C is weak 12-18 UTC (+54 bp, n=46) vs +419..+973 bp other
  hours, too few trades to filter on; M entry hours are set by Binance's listing times.

## Pre-registered pick for the forward read-out (written 2026-11-27 15:20Z, before any forward outcome)

**LMC, 10 equal slots of equity, gross 1.0, $10k book from 2026-11-02:**

- L: spot long when the day5-variant CNN (2026 fold) scores a liquid 5m spike at pct >= 0.995;
  entry at the next 5m open; take profit +20%; time exit 72h; cost 0.4% round trip.
- M: perp short of a new Binance spot listing 72h after its first spot candle if the perp's open
  then is at or below the listing price and the perp traded >= 50k USDT in the previous hour;
  take profit 50%, stop 50% (liquidation at +100% isolated 1x); 30 days; cost 0.3% + funding.
- C: perp short of a liquid spike with ret_4h <= -15%, |BTC 4h| <= 1% and CNN pct <= 0.5, perp
  active; 24h; isolated 1x; cost 0.3% + funding.
- One position per coin; signals in time order, L before C before M at the same time.

Read-out: `research/capitulation_ls/lmc_forward.py` (signals from the forward scorer; outcomes
on spot 5m for L and perp klines for C/M). Success for the paper trade: mean net per closed
trade > 0 and no closed year below 0. Until the day5 2026 model is saved, the forward scorer
uses the paper test's path_all 2026 fold (a stated deviation). Search results after this point
are in-sample only.

## Next steps (for a resume)

1. Done: forward read-out (0 trades). To refresh later: sync spot candles, then
   `forward_signals.py day5` and `lmc_forward.py day5`.
2. Done: day10b = day5 within noise.
3. Queue: (a) done (fill realism); (b), (c) done;
   (d) refresh the report with every finding after the draft.
4. Keep searching the idea backlog until 2026-11-28T09:46Z; then final report, check command, PR.

## Idea backlog

- Pump shorts on perp paths (5m, to 7 days); leverage 1-3x with liquidation on the perp path.
- Open interest / long-short ratios at entry as squeeze predictors (metrics download running).
- Transformer and TCN rankers (training), seed ensembles, a ranker trained only on the rule set.
- Listing shorts: combine with pump shorts in one book; listing pumps inside the first week.
- Long side: flush-day longs on perps (lower fees), CNN top 0.5% held 24h, a BTC-crash gate.
- Combined long/short book; capital split; Kelly sizing; drawdown control.
- Forward read-out on 2026-11-02..27 (spot synced; perp prices and funding via daily archive/API).
- Crazy ones: pyramiding into squeezes, basket shorts of the day's pumps, shorting extreme
  positive funding, weekday/hour effects, pump-then-dump chains (second pump within days).

## Blocked on

## Follow-ups
