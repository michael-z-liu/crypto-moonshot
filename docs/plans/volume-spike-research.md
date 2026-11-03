# Volume-spike strategy research

- **Status:** building <!-- building | blocked | review | done -->
- **Branch:** `claude/high-volume-trading-strategy-0c5d1e`
- **PR:** https://github.com/michael-z-liu/crypto-moonshot/pull/8
- **Updated:** 2026-11-03T01:00Z
- **Deadline:** 2026-11-02T22:00Z (09:00 Sydney). At or after the deadline, stop exploring, finish
  the write-up (M6) and open the PR, even if earlier milestones are only partly explored.

## Goal

The user hand-labelled about 930 Binance charts in July 2019 (`~/Documents/crypto-strategies/01`,
Good/Bad/Missed) where a sudden 5-minute volume spike preceded a price jump. They want an
overnight, unattended exploration that turns that starting point into a strategy with a real,
measured edge: rebuild the trigger mechanically, measure what happens after every spike in
2020–2026 USDT data, and search for filters and exits that leave positive expectancy after fees,
checked out of sample. The output is research code plus a findings write-up, not a live strategy.

## Ground rules

- Data: read-only, `MOONSHOT_DATA_DIR=/Users/michael/Documents/code/crypto-moonshot/data`
  (198 USDT pairs, 5m, 2020-01 to 2026-11). Never write there.
- Universe: drop stablecoins, fiat, gold, wrapped assets and tokenised stocks (`…BUSDT` such as
  `NVDABUSDT`). Note: the list is today's top coins, so results carry survivorship bias.
- No look-ahead: features use only candles up to the spike bar's close; entry is the next bar's
  open; exits are checked on later bars with the stop assumed hit first if both are touched in
  one bar.
- Costs: 0.1% fee per side plus 0.05% slippage per side (0.3% round trip) as the base case;
  report sensitivity at 0.2% and 0.4%.
- Splits: tune on 2020-01..2023-12 (train), choose on 2024 (validation), and look at 2025-01..
  2026-10 (test) once, at the end, for the final candidates only.
- The metric is mean net return per trade and its stability by year and by symbol, not win rate.
  Always compare against random entries with the same exits.
- No API keys, no orders, no network. Code lives in `research/volume_spike/`; big intermediate
  files go to `research/volume_spike/cache/` (gitignored); small result tables are committed.

## Out of scope

- Live or paper trading, wiring the strategy into `src/` (no strategy/backtest engine exists yet).
- Downloading more data (other intervals, delisted coins, order books).

## Milestones

### M1: Event dataset (done)

- **Delivers:** a loader for the clean universe and an event table: every candle where volume is
  a large multiple of its recent baseline, with features (volume multiple, taker-buy share, bar
  return, range, prior run-up, distance from recent high, volatility, liquidity, BTC context, hour)
  and forward outcomes (max up/down over 1..48 bars, close-to-close returns).
- **Done when:** the table covers all symbols and years, no feature uses future candles, and a
  spot check of a few events against the raw candles matches.
- **Worker:** master

### M2: Baseline edge (done)

- **Delivers:** expectancy of the raw spike trigger over a grid of take-profit, stop-loss and
  time-stop exits, versus random entries; the user's 2019 setup (+1% / −5%) included.
- **Done when:** a table answers "does the raw trigger beat random after costs, and with which
  exits?"
- **Worker:** master

### M3: Filters (done)

- **Delivers:** single-feature and two-feature conditional expectancy on train data (taker-buy
  share, bar return, run-up, BTC trend, liquidity, time of day, ...), then a small rule set or a
  simple model chosen on validation.
- **Done when:** candidate rules are ranked on train and validation, with trade counts.
- **Worker:** master

### M4: Robustness (done)

- **Delivers:** per-year, per-symbol and cost-sensitivity checks of the candidates; walk-forward.
- **Done when:** each candidate has a robustness verdict.
- **Worker:** master

### M5: Portfolio simulation (done)

- **Delivers:** a simple sequential simulation with a cap on open positions, equity curve,
  drawdown and trades per week, for the best one or two candidates; one look at the test period.
- **Done when:** results are saved and plotted.
- **Worker:** master

### M6: Write-up and PR (done)

- **Delivers:** `docs/research/volume-spike.md` with method, findings, honest caveats and next
  steps; PR opened.
- **Done when:** check command passes and the PR is open.
- **Worker:** master

### M7: Convolutional network on raw candles (done)

Added 2026-11-03 at the user's request: can a 1D CNN that sees the raw candle sequence before a
spike (price, volume, taker-buy volume) tell good entries from bad ones better than the
hand-made features?

- **Delivers:** `m11_cnn.py` and `build_seq.py`, a results table and a write-up section.
- **Protocol (fixed before training):**
  - Candidates: every 5m spike (volume >= 10x), the same 1.97M events as M1-M6.
  - Input: the last 96 candles (8 hours) up to and including the spike candle. Coin channels:
    open/high/low/close as log ratio to the spike close, log volume relative to the previous
    day's median volume, taker-buy share. Plus the coin's liquidity (log median USDT volume) as
    a scalar. A second variant adds BTC's close and volume channels.
  - Target: net 1h return (close of bar 12 after entry at the next open, minus 0.3%), clipped at
    +-20%. Same target and hold as the gradient-boosting model in M3 (`m10_model.py`).
  - Fit on 2020-2023; choose the variant, epochs and selection threshold on 2024 only; an
    ensemble of 3 seeds. Then run the chosen setup once on 2025-01..2026-10.
  - Report, per split: mean net bp per trade and per day (non-overlapping per symbol) at the
    train-quantile thresholds 0.99 / 0.995 / 0.998 / 0.999, also excluding 2025-10-10, side by
    side with the gradient-boosting model. And as an entry filter on top of F1: F1 trades with the
    CNN score above vs. below its median.
  - Caveat: the test period was already inspected for M5, so it is clean for this model's
    fitting and choices but not a fully untouched period.
- **Done when:** the table is committed and the write-up says whether the CNN adds an edge.
- **Worker:** master

### M8: Visual report of every method (done)

Added 2026-11-03 at the user's request: after M7, one report with plots, diagrams and the real
data for every method tried (including the ones that failed), to check the work and to evolve the
idea from.

- **Delivers:** `report_data.py` (recomputes every method per year on the same basis and exports
  the chart data as JSON) and an HTML report published as a private claude.ai artifact (source in
  `research/volume_spike/report/`): per method, a diagram of the rule, charts from the data, and
  a verdict.
- **Done when:** the report is published, linked from `docs/research/volume-spike.md`, and the
  PR is updated.
- **Worker:** master

### M9: Walk-forward CNN study (building)

Added 2026-11-03 at the user's request: a much more thorough, creative investigation of the CNN
approach, expanding its dimensions toward a more robust strategy.

- **Why walk-forward:** M7 rests on one 22-month test that has now been looked at. Retraining each
  year on all earlier data gives five out-of-sample years (2022-2026) per variant instead of one.
- **Candidates:** spikes (>= 10x) on liquid coins (median >= 5k USDT per 5m candle), 632k events.
- **Dataset (`build_seq2.py`):** 288 candles (24h) up to the spike candle; channels: coin OHLC,
  volume, taker share, trade count; BTC close and volume; ETH close and volume; market breadth
  (equal-weight universe return) and market spike share (share of coins with volume >= 5x);
  scalars: liquidity, coin age, hour of day. Forward returns at 15m/30m/1h/2h/4h.
- **Folds (`m12_cnn_wf.py`):** for each test year Y in 2022..2026: train on 2020-01 .. Y-1 Jul,
  early-stop on Y-1 Jul .. Y-1 Dec, predict Y. Scores are turned into percentiles of the fold's
  own training scores, so thresholds never use future data.
- **Variants screened (2 seeds each):** window length (2h / 8h / 24h), channel sets (coin, +BTC,
  +ETH, +market breadth, all), target horizon (30m / 1h / 4h, multi-horizon), loss (MSE, Huber,
  classification, within-bar cross-sectional rank target), architecture (M7 CNN, dilated TCN,
  CNN+GRU, MLP on the raw grid, boosted trees on the raw grid), training universe (all liquid
  spikes vs. crash context only).
- **Selection rule (fixed now):** a variant is judged on the 5-slot portfolio (top 0.5% by
  percentile, highest first, hold = its target horizon, 0.3% cost) over the walk-forward years
  **2022-2024 only**: mean of the three yearly net bp per trade, with the worst year as the
  tie-breaker. 2025-26 is reported for every variant, but is never used to choose. The finalists
  get 5 seeds.
- **Done when:** the variant table is committed with yearly OOS results.
- **Worker:** master

### M10: Trading rules on top of the scores (pending)

- On the chosen variant's walk-forward scores: threshold, slots, hold, exits (target / stop /
  trailing), score-weighted sizing, a BTC-flush gate, a per-day cap and cool-downs. Chosen on
  2022-2024 by the same rule; 2025-26 reported.
- **Worker:** master

### M11: Robustness of the final strategy (pending)

- Costs, latency, slot count, per-coin and per-day concentration, seed spread, a placebo that
  shuffles scores within each candle (does the ranking matter?), bootstrap confidence intervals.
- **Worker:** master

### M12: Report (pending)

- A new casebook section and write-up for M9-M11; PR updated.
- **Worker:** master

## Research log

<!-- Append one dated line per finding or dead end, so a resumed session knows what was tried. -->

- 11:59Z M1: universe 152 symbols (pegged + tokenised stocks dropped). Spike = volume >= 10x the
  previous day's median 5m volume (4x was 10% of all bars). 1.97M spikes + 296k random bars in
  `cache/events.parquet` + `cache/fwd.npy` (rebuild: `build_events.py`, ~15 s). Spot checks OK.
- 12:00Z M2 (train 2020-23): raw spike has NO edge. Gross 4h drift -4 bp (random +3 bp); spikes on
  up bars -12 bp (they mean-revert). User's +1%/-5% exit: 60% win rate but -31 bp/trade net.
  Best exit on raw spikes is still -22 bp net. Cost (30 bp) dominates; need filters worth >30 bp.
- 12:02Z M3 deciles (`m3_features.py`): spikes AFTER A PUMP fade (-50..-86 bp at 4h for top-decile
  run-up / bar return / range); spikes AFTER A CRASH bounce (+45 bp 4h, +90 bp 24h for bottom-decile
  ret_4h/ret_24h; random bars after the same drop only +15 bp at 4h). => "capitulation" idea.
- 12:04Z Capitulation (spike & ret_24h < -16%): +253 bp net 4h hold (train), +276 (valid). Survives
  dropping the 20 busiest crash days (+242). First spike per symbol per 24h: +169 bp, positive every
  year except 2025 (-47 bp: first spike = falling knife). Better when: drop is acute (ret_4h, runup_1h),
  at 24h low, big range bar, BTC also crashing / BTC volume spike, liquid coin (qv_base high),
  listed > ~2 months, mid taker-buy share (extremes worse).
- 12:06Z `m3_rules.py` grid: ret_4h <= -10..-20% & qv_base >= 20k USDT/bar: +600..+1200 bp/trade
  but only ~100-300 trades/yr, concentrated on market-wide crash days (2020-03-12, 2021-05-19,
  2021-12-04, ...). Checked: real moves, not bad ticks. Caveats: survivorship (LUNA/FTT absent),
  crash-time fills. NEXT: long-horizon fwd (288 bars) for crash subset to test stops/exits, then a
  slot-limited portfolio sim, cost sensitivity, and a search for a more frequent second signal.
- 12:08Z `build_long.py` (24h paths for 173k crash rows) + `portfolio.py` (5 slots, 1 position per
  symbol, compounding) + `m5_capitulation.py`. Candidates (train / valid):
  C1 spike & ret_4h<=-15% & qv_base>=5k, hold 48 bars: CAGR 82%/103%, maxDD -11%/-6%, 1.5 trades/wk.
  C2 spike & ret_4h<=-10% & qv_base>=20k, hold 48: CAGR 80%/72%, maxDD -22%/-20%, 3.3 trades/wk.
  Stops/targets don't beat plain time exits; a 5% trailing stop over 12h is close. NEXT: cost and
  entry-slippage stress, survivorship check (old-coin subset), filters to lift C2, frequency.
- 12:10Z `m4_robust.py`: C1/C2 positive every year 2020-24; survive 1% round-trip cost; work on
  coins listed by 2020-01 and on later ones. Entering one bar late costs ~1/3 of the edge (speed
  matters). Filter scan on C2: MARKET-WIDE flush is the key filter (BTC vol spike >=4x or BTC 4h
  <= -2%: +247/+367 bp; coin-only crash with calm BTC: +49/+20 bp, i.e. no edge).
- 12:11Z `m5_market.py` grid (240 combos, so validation is now mildly overfit): 1h holds best.
- 12:12Z Alternatives: user's 2019 rule (up bar, taker-buy > 60%, flat run-up) = -30 bp/trade at
  1h/4h, -19 bp at 24h. Breakouts to 7d highs: ~+20..+90 bp at 24h only, mostly bull beta.
  Uptrend pullback (7d > +20%, 4h < -3%, hold 24h): +78..+234 bp/trade but max DD -30..-84% =>
  rejected.
- 12:13Z PRE-REGISTERED finalists for the one-time test (2025-01..2026-10), 5 slots, 0.3% cost:
  F1 spike>=10x & ret_4h<=-8% & qv_base>=5k & btc_vol_mult>=6, exit after 12 bars (1h).
  F2 spike>=10x & ret_4h<=-10% & qv_base>=20k & btc_vol_mult>=6, exit after 12 bars.
  F3 (no BTC filter) spike>=10x & ret_4h<=-15% & qv_base>=5k, exit after 48 bars (= C1).
  References: user's 2019 rule; all spikes.
- 12:15Z TEST RESULT (`m6_test.py`, `results/equity.png`): edge mostly GONE in 2025-26. Per trade:
  F1 -3 bp, F2 +15 bp, F3 +42 bp (train/valid were +140..+570). 2025H1 bad for all, 2026H1 good
  for F2/F3. User 2019 rule -31 bp/trade in every period (reference).
- 12:18Z Post-hoc (test now seen): in 2025-26 only the most extreme flushes still bounce (ret_4h
  <= -15% or BTC 4h <= -5%). Alt-index regime filters (`alt_index.py`) and vol-normalised drops
  don't fix 2025. Results for 2025 swing -0.5%..+3% per trade with slot count => slot luck on a
  few crash days. Per-crash-day basket view: F2 +185..+360 bp/day (80-90% win days) in 2020-24,
  +50/+78 bp/day (58-64%) in 2025/26. Decayed, not dead.
- 12:24Z `m7_walkforward.py` (pick params on trailing 24m by t-stat, trade next 6m, per-day
  baskets, 50% of equity per crash day): OOS 2022-01..2026-10 CAGR +20..23%, max DD -8..-16%,
  ~40 crash days/yr; 2025 ~flat. This is the honest headline number.
- 12:27Z `m8_limit_entry.py`: limit buys below the spike close fill 30-40% (at offset 0) and only
  on trades that keep falling (adverse selection); no improvement in the test period.
- 12:29Z Majors only in flushes: +120..+350 bp/trade 2020-24 and 2026, negative 2025. Pump-fade
  shorts (hypothetical): erratic, big squeeze losses. Write-up `docs/research/volume-spike.md`.
- 12:33Z `m9_daily.py`: daily abnormal volume = market beta; daily capitulation (coin -10%/day,
  BTC -5%) same sign as 5m version but noisy.
- 12:36Z `m10_model.py` (HistGradientBoosting on 36 features): top 0.2% picks = capitulation;
  2025-26 positive only via 2025-10-10, otherwise -76 bp/trade.
- 12:39Z First-flush-after-calm filter: noise. BTC->alt lead-lag at 5m: ~0 in train.
- 12:42Z Exploration with the existing data pool is exhausted; remaining ideas need new data
  (delisted coins, 1m candles) => follow-ups for the user. Moving to M6 (PR) early.
- 2026-11-03 00:36Z M7 choice (train/valid only, `results/m11_cnn_selection_trainvalid.csv`): validation
  correlation peaks at epochs 1-3 then falls (overfits fast). The coin+BTC variant beats coin-only
  on 2024 at every threshold (per signal day +111/+220/+487/+614 bp vs +47/+79/+220/+217) and as
  an F1 filter (enter +309 / skip +75 bp per trade). CHOSEN before the test: coin+BTC, threshold =
  best 2024 per-day result among thresholds with >= 20 signal days = top 0.5% (train quantile
  0.995). GBM on 2024 at the same threshold: +277 bp/trade, +79 bp/day.
- 00:38Z M7 TEST (`m11_cnn.py report`, `results/m11_cnn_selection.csv`): chosen CNN (coin+BTC,
  top 0.5%) on 2025-26, every trade one per coin: -110 bp/trade but +177 bp per signal day (60% of
  101 days positive); without 2025-10-10 +64 bp/trade, +184 bp/day. GBM same threshold: -161 bp/trade,
  +56 bp/day. As an F1 enter/skip filter the CNN FAILED in the test (enter -155, skip +36 bp/trade).
- 00:41Z Post-hoc (`m11_cnn.py portfolio|stress`): in the 5-slot portfolio (highest score first)
  the CNN makes +114 bp/trade in 2025-26 (CAGR +33%, max DD -11%; 2025 +120, 2026 +106) vs GBM
  +10 bp (DD -26%) and F2 +15 bp (DD -22%). Holds without 2025-10-10 (+114), at 3/10 slots
  (+127/+87), at 1% cost (+44) and one candle late (+79). Coin-only CNN (rejected on 2024) loses
  (-21 bp, DD -40%): the BTC channels carry it. Its edge is in ranking signals within crash days.
- 00:45Z M8: `report_data.py` recomputes every method per year on one basis and builds the visual
  casebook (`report/template.html` + `cnn_section.js` + `cnn_text.js` -> `report/index.html`),
  published privately at https://claude.ai/artifact/RpfQqzAyte82MWYGXm7Yn5. Shape charts leave out
  2025-10-10 and clip paths at +-20% (that day's wicks put the 2025-26 flush average at +550 bp).
  Post-hoc rule check (`m11_cnn.py rules`): deep-flush thresholds don't reproduce the CNN.

## Decisions

- 2026-11-02: Bar interval is 5m (user). A "Good" label meant "price ran up afterwards" by eye;
  the original scanner code is lost, so the trigger is rebuilt from the screenshots.
- 2026-11-02: User: the 2019 setup is inspiration, not a constraint. Other signals and
  alternative strategies derived from it are in scope if they improve profitability.
- 2026-11-02: Research code stays outside `src/` as scripts; only promoted to `src/` once a
  strategy/backtest engine exists.

## Blocked on

## Follow-ups

- Retrain the CNN walk-forward (each year on all earlier years) and paper-trade it next to F2.

- Paper-trade F2 or the walk-forward version on the live stream (`paper` mode) to measure real fills
  on flush days; needs the strategy/backtest/live engine from `docs/architecture.md` first.
- Download 1m candles around past flush days to see how much of the bounce happens inside the 5m
  spike candle, and whether a faster trigger restores the 2025 edge.
- Add delisted symbols (Binance public data archive) to remove survivorship bias from dip-buying.
- Record order-book depth and spread on flush days before sizing anything.
