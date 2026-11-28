# Alt catch-up after big BTC moves (1h-3d): X3

Scripts: `x3_build.py` (hourly spot/perp matrices), `x3_cands.py` (events, beta, lag signal, 72h
paths), `x3_land.py`, `x3_eval.py` (exits, 10-slot per-year books, grid), `x3_books.py`,
`x3_final.py` (picks, trade files, tables). Trades: `$OWN/explore/lag_trades.parquet`; non-entries
(12.4k): `lag_nonentries.parquet`. Everything below is in-sample (no train/test split was needed:
nothing worked).

## Hypothesis
After a large BTC move over 4/12/24h, liquid high-beta alts that have not yet moved by their
beta-implied amount (laggards) catch up over the next 4h-3d. Up move: long laggards (spot). Down
move: short laggards (perp).

## Method
- Hourly bars built from 5m spot candles (582 coins) and perp 1h bars (470 mapped perps).
- Events: BTC close-to-close move over w hours >= threshold: (4h: 3%, 5%), (12h: 5%, 8%), (24h: 6%,
  10%); up and down separately; 12h lock-out per direction. 2020-2026; trades only counted 2021+.
- Eligible alts: not BTC/ETH, trailing-30d hourly beta to BTC >= 0.8 and corr >= 0.3, trailing 24h
  spot volume >= $1M; shorts also need perp volume >= $50k in each of the last two hours.
- Lag = direction * (beta * BTC move - alt move) over the same window; z = lag / (residual hourly
  sd * sqrt(w)). Signal = cross-sectional percentile of z among the event's eligible alts (0..1).
  Threshold 0 (baseline) = no lag filter, random priority.
- Entry next hour open. Hold 4/12/24/72h, take profit none/5/10/20%, stop none/5/10% (stop first
  if both in one hourly bar; stop slippage 0.1%; isolated 1x short liquidated at +100% = -100%).
  Costs: spot 0.4% round trip, perp 0.3% plus actual funding.
- Book: per entry year a fresh $10k, 10 equal slots (equity/10), stake <= 0.5% of 24h spot volume,
  one position per coin, ordered by signal (random for baseline).
- Extra modes added after seeing the first landscape: `bounce_long` (long laggards after a BTC
  drop), `bounce_fell` (long the alts that fell MOST after a BTC drop), `fade_short` (short after
  a BTC rise).

## Configurations tried (all in-sample)
5 modes x 6 event definitions x thresholds {0, .8, .9, .95} x holds {4,12,24,72} x tp {none,5,10,20}
x stop {none,5,10} = 5760 candidate-level evaluations; the top 80 per mode by average net (>=150
candidates) got full books (400 books). Hypothesis modes (`up_long`, `down_short`) are the ones
that test the idea; the rest are diagnostics.

| Mode | Share of configs with avg net > 0 | Best candidate-level avg net |
| --- | --- | --- |
| up_long (hypothesis) | 54% | +3.0% (clustered, see below) |
| down_short (hypothesis) | 0% | -0.1% |
| fade_short | 2% | +0.5% |
| bounce_long | 67% | +3.9% |
| bounce_fell | 80% | +4.5% |

Candidate-level averages are inflated by clustering: 50-400 independent events, ~100 alts each,
mostly one market move.

## Does the lag signal matter?
Average net by lag-signal quintile (24h hold, no tp/stop, bp, low to high lag):

| Mode, event | Q1 | Q2 | Q3 | Q4 | Q5 |
| --- | --- | --- | --- | --- | --- |
| up_long, BTC +3%/4h | -16 | -7 | -2 | 0 | -14 |
| up_long, BTC +5%/12h | -18 | -33 | -37 | -38 | -55 |
| up_long, BTC +6%/24h | -2 | 9 | 17 | 2 | -19 |
| down_short, BTC -3%/4h | -139 | -140 | -151 | -145 | -130 |
| down_short, BTC -5%/12h | -272 | -260 | -238 | -245 | -224 |
| down_short, BTC -6%/24h | -309 | -306 | -298 | -274 | -259 |
| bounce_long, BTC -3%/4h | 69 | 74 | 61 | 57 | 22 |
| bounce_long, BTC -6%/24h | 186 | 200 | 199 | 188 | 141 |

Up moves: no lag effect (flat to negative at the top quintile). Down moves: the top lag quintile
loses slightly less when shorted (+10 to +50bp better), but every quintile loses 1.3-3% per trade
net: after a BTC drop alts bounce, they do not keep falling. The top lag quintile bounces LESS
than the rest (the alts that fell most bounce most). Per independent event, an equal-weight basket
of alts nets: up moves +0.2% to +1.1% (median +0.6-0.9%), top-lag-decile basket +0.0% to +0.5%;
BTC-down shorts -1.6% to -5.1%.

## Books (10 slots, per-year $10k): best hypothesis config vs its baseline
Best by mean yearly return among hypothesis modes (>= 60 trades): `up_long`, BTC +5% in 4h
(104 events), signal >= 0.8, hold 72h, TP 20%, no stop. Baseline: same without the lag filter.

| Config | Trades | Avg trade | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | Max DD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lag_best (thr .8) | 625 | +4.35% | +262% (259) | +90% (154) | +21% (60) | +34% (90) | -13% (30) | +24% (32) | -30% |
| lag_baseline (no lag filter) | 632 | +4.26% | +201% (263) | +87% (155) | +30% (61) | +63% (90) | -18% (30) | +15% (33) | -31% |

Per-year average trade, best: 2021 +5.4%, 2022 +4.5%, 2023 +3.3%, 2024 +3.5%, 2025 -4.4%, 2026 +7.5%.
The lag filter adds +0.1% per trade (noise). The result is a "buy liquid alts after a +5%/4h BTC
burst, hold 3 days with 20% TP" drift, not catch-up; it misses both targets (avg trade 4.4% vs
10%; worst year -13% vs +50%; 2021 alone is most of the return, 2025 negative).
Down-move short books: none has a positive worst year (best config's worst year -8%, shallowest
max DD -14%); most configs lose in most years.

Diagnostic best (not the hypothesis): long the alts that fell most after BTC -10%/24h
(`bounce_fell`, thr .8, 72h, TP 20%): 323 trades, +4.4% avg, 2021 +509% (207 trades), 2022 -59%,
2023 no events, 2024 +10%, 2025 -2%, 2026 +8%; baseline +3.3%. Max DD -54%. This is the already
known capitulation-long family on BTC-flush days; ~50 events in 5.8 years, 2021 dominates.

## What worked / what did not
- Did not work: laggard selection (no lift on up moves, negligible on down moves); shorting after
  BTC drops (alts rebound: -1.3% to -5% net per trade before any stop); shorting after BTC rises.
- Weakly positive: unfiltered long of liquid alts after a +5%/4h or larger BTC burst (+4% per trade
  in the candidate pool, ~+0.2% to +1% per event basket at 24h, negative in 2025).
- Bounce after BTC flushes is real (+1% to +4% per event basket at 24h) but concentrated in few
  events and already covered by the L sleeve (CNN capitulation longs).

## Caveats
- 2021 is 40% of trades and most of the return; later years have 30-90 trades.
- Hourly bars: stop vs TP order inside a bar assumed stop-first; entries at the hour open (no 5m
  delay modelled); equity-fraction stakes capped at 0.5% of spot 24h volume (also for perps).
- Beta/correlation from trailing hourly data; the lag signal is simple (no sector or listing
  effects); events cluster so t-stats from candidate counts are overstated.
- BTC-to-alt lag at 5m was about zero (volume-spike.md); here at 4-24h the lag is also not
  exploitable: alts move with or ahead of BTC, then revert.

## Why it failed
The catch-up premise is not visible: the lag of an alt after a BTC move does not predict its
subsequent excess return. The dominant effects are market-wide: a drift after up bursts (small) and
a bounce after drops (larger), neither selected by lag, and the bounce is the already-known
capitulation edge. Costs (0.4% spot, 0.3% + funding perp) and a negative short side leave no path
to +10% per trade or +50% every year.
