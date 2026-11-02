# Volume-spike strategy research

- **Status:** building <!-- building | blocked | review | done -->
- **Branch:** `claude/high-volume-trading-strategy-0c5d1e`
- **PR:** <!-- link, once opened -->
- **Updated:** 2026-11-02T13:35Z
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

### M3: Filters (in progress)

- **Delivers:** single-feature and two-feature conditional expectancy on train data (taker-buy
  share, bar return, run-up, BTC trend, liquidity, time of day, ...), then a small rule set or a
  simple model chosen on validation.
- **Done when:** candidate rules are ranked on train and validation, with trade counts.
- **Worker:** master

### M4: Robustness (todo)

- **Delivers:** per-year, per-symbol and cost-sensitivity checks of the candidates; walk-forward.
- **Done when:** each candidate has a robustness verdict.
- **Worker:** master

### M5: Portfolio simulation (todo)

- **Delivers:** a simple sequential simulation with a cap on open positions, equity curve,
  drawdown and trades per week, for the best one or two candidates; one look at the test period.
- **Done when:** results are saved and plotted.
- **Worker:** master

### M6: Write-up and PR (todo)

- **Delivers:** `docs/research/volume-spike.md` with method, findings, honest caveats and next
  steps; PR opened.
- **Done when:** check command passes and the PR is open.
- **Worker:** master

## Research log

<!-- Append one dated line per finding or dead end, so a resumed session knows what was tried. -->

- 12:15Z M1: universe 152 symbols (pegged + tokenised stocks dropped). Spike = volume >= 10x the
  previous day's median 5m volume (4x was 10% of all bars). 1.97M spikes + 296k random bars in
  `cache/events.parquet` + `cache/fwd.npy` (rebuild: `build_events.py`, ~15 s). Spot checks OK.
- 12:20Z M2 (train 2020-23): raw spike has NO edge. Gross 4h drift -4 bp (random +3 bp); spikes on
  up bars -12 bp (they mean-revert). User's +1%/-5% exit: 60% win rate but -31 bp/trade net.
  Best exit on raw spikes is still -22 bp net. Cost (30 bp) dominates; need filters worth >30 bp.
- 12:35Z M3 deciles (`m3_features.py`): spikes AFTER A PUMP fade (-50..-86 bp at 4h for top-decile
  run-up / bar return / range); spikes AFTER A CRASH bounce (+45 bp 4h, +90 bp 24h for bottom-decile
  ret_4h/ret_24h; random bars after the same drop only +15 bp at 4h). => "capitulation" idea.
- 12:45Z Capitulation (spike & ret_24h < -16%): +253 bp net 4h hold (train), +276 (valid). Survives
  dropping the 20 busiest crash days (+242). First spike per symbol per 24h: +169 bp, positive every
  year except 2025 (-47 bp: first spike = falling knife). Better when: drop is acute (ret_4h, runup_1h),
  at 24h low, big range bar, BTC also crashing / BTC volume spike, liquid coin (qv_base high),
  listed > ~2 months, mid taker-buy share (extremes worse).
- 12:55Z `m3_rules.py` grid: ret_4h <= -10..-20% & qv_base >= 20k USDT/bar: +600..+1200 bp/trade
  but only ~100-300 trades/yr, concentrated on market-wide crash days (2020-03-12, 2021-05-19,
  2021-12-04, ...). Checked: real moves, not bad ticks. Caveats: survivorship (LUNA/FTT absent),
  crash-time fills. NEXT: long-horizon fwd (288 bars) for crash subset to test stops/exits, then a
  slot-limited portfolio sim, cost sensitivity, and a search for a more frequent second signal.
- 13:10Z `build_long.py` (24h paths for 173k crash rows) + `portfolio.py` (5 slots, 1 position per
  symbol, compounding) + `m5_capitulation.py`. Candidates (train / valid):
  C1 spike & ret_4h<=-15% & qv_base>=5k, hold 48 bars: CAGR 82%/103%, maxDD -11%/-6%, 1.5 trades/wk.
  C2 spike & ret_4h<=-10% & qv_base>=20k, hold 48: CAGR 80%/72%, maxDD -22%/-20%, 3.3 trades/wk.
  Stops/targets don't beat plain time exits; a 5% trailing stop over 12h is close. NEXT: cost and
  entry-slippage stress, survivorship check (old-coin subset), filters to lift C2, frequency.
- 13:20Z `m4_robust.py`: C1/C2 positive every year 2020-24; survive 1% round-trip cost; work on
  coins listed by 2020-01 and on later ones. Entering one bar late costs ~1/3 of the edge (speed
  matters). Filter scan on C2: MARKET-WIDE flush is the key filter (BTC vol spike >=4x or BTC 4h
  <= -2%: +247/+367 bp; coin-only crash with calm BTC: +49/+20 bp, i.e. no edge).
- 13:30Z `m5_market.py` grid (240 combos, so validation is now mildly overfit): 1h holds best.
- 13:35Z Alternatives: user's 2019 rule (up bar, taker-buy > 60%, flat run-up) = -30 bp/trade at
  1h/4h, -19 bp at 24h. Breakouts to 7d highs: ~+20..+90 bp at 24h only, mostly bull beta.
  Uptrend pullback (7d > +20%, 4h < -3%, hold 24h): +78..+234 bp/trade but max DD -30..-84% =>
  rejected.
- 13:35Z PRE-REGISTERED finalists for the one-time test (2025-01..2026-10), 5 slots, 0.3% cost:
  F1 spike>=10x & ret_4h<=-8% & qv_base>=5k & btc_vol_mult>=6, exit after 12 bars (1h).
  F2 spike>=10x & ret_4h<=-10% & qv_base>=20k & btc_vol_mult>=6, exit after 12 bars.
  F3 (no BTC filter) spike>=10x & ret_4h<=-15% & qv_base>=5k, exit after 48 bars (= C1).
  References: user's 2019 rule; all spikes.

## Decisions

- 2026-11-02: Bar interval is 5m (user). A "Good" label meant "price ran up afterwards" by eye;
  the original scanner code is lost, so the trigger is rebuilt from the screenshots.
- 2026-11-02: User: the 2019 setup is inspiration, not a constraint. Other signals and
  alternative strategies derived from it are in scope if they improve profitability.
- 2026-11-02: Research code stays outside `src/` as scripts; only promoted to `src/` once a
  strategy/backtest engine exists.

## Blocked on

## Follow-ups
