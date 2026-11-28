# Perp premium / basis extremes (x1)

Scripts: `x1_fetch_premium.py` (premium-index 1h klines, top 150 USD-M perps by volume, 147 MB),
`x1_premium.py`. Outputs: `x1_grid.csv` (864 configs), `x1_wf.csv`, `x1_books.csv`;
trades `$OWN/explore/premium_trades.parquet` (best + baseline), non-entries `premium_nonentries.parquet`.

## Hypothesis
Extreme perp premium (crowded longs) is followed by a fall, extreme discount (crowded shorts) by a rebound.

## Method
- Signal at the close of hour h: percentile of the premium close within the coin's trailing 30 days
  ('own') or across liquid coins that hour ('xs'). Short signal = pct, long signal = 1 - pct.
  Optional absolute gate |premium| >= 0 / 0.1% / 0.3%.
- Entry at the open of hour h+1; liquid only (24h perp volume >= $3M); per-coin cooldown = hold length.
- Exits on hourly bars: hold 8/24/72h, TP none/5%/10%, stop none/10%; stop first if both hit in a bar;
  short liquidated at +100% (net -1). Costs 0.3% round trip + realised funding from the funding file.
- Grid: side 2 x signal 2 x threshold {0.95, 0.98, 0.99, 0.995} x gate 3 x hold 3 x TP 3 x stop 2 = 864 configs,
  all of them in-sample on 2020-26 (tried: all listed in `x1_grid.csv`).
- Walk-forward: for each year 2022..26 choose the config with best mean net on earlier years (>= 60 trades).
- Book: per-year $10k, 10 slots, stake = realised equity/10, capped at 0.5% of 24h volume.

## Results
Baseline (short after own pct >= 0.99, 24h, no TP/stop): 35,960 trades, avg net -0.41%, win 49%.
Per-year book returns -68%, 0%, -75%, -86%, -50%, -52% (2021-26). Dead.

All 864 configs: 259 of the n>300 configs have avg > 0; none reaches +10%. Short side: best avg +0.4%
(`short_prem|own>=0.995|gate0.003|h24|tp0.1`), mean over all short configs about -0.2% to -0.5%.
Long-after-discount, 72h hold is the only family with a positive edge:

| Best in-sample: long_disc, own >= 0.995, |disc| >= 0.3%, 72h, no TP/stop | |
| --- | --- |
| trades / avg net / win | 3,869 / +2.76% / 51% |
| avg by year 2021-26 | +7.6%, -1.1%, +2.1%, +2.0%, +4.4%, +0.6% |
| 10-slot book return 2021-26 | +948%, +92%, +79%, +77%, +152%, +17% (max DD -14% .. -47%) |

Walk-forward (long_disc 72h picked every year): trade avg 2022 -1.1%, 2023 +2.2%, 2024 +1.8%, 2025 +3.9%,
2026 +0.6%; book +92%, +85%, +77%, +104%, +17%.

## What worked / did not
- Shorting premium: nothing. Premium is positive in bull trends and the coin keeps going up; costs eat the rest.
- Longs after a deep discount (<= -0.3%) held 72h have a small positive average, concentrated in 2021 and 2025.
  The absolute gate matters (without it the edge vanishes); the percentile threshold barely matters.
- Average trade +2.8% is far below the +10% target; the 2022 and 2026 trade averages are about 0.

## Caveats
- Book returns are flattered by compounding 10 slots over ~400-560 trades a year with equal-slot restaking
  and tp/stop-free mean-reversion in a few crash days: trades cluster (1,564 distinct entry days, median
  day mean about 0, 50% of days positive), so the mean is carried by a skewed tail. Max DD is large, 2026 only +17%.
- Hourly bars: no intra-bar ordering; stops fill exactly at the level (gaps ignored). The 24h volume gate is from
  the perp itself; discount events are crashes where fills at 0.05% slippage for $1-5k stakes are optimistic.
- Selection of best config is in-sample over 864 configs; no deflation applied. Walk-forward result is the honest read.
- Universe: top 150 perps by lifetime volume (survivorship toward big coins). Premium close is the last minute of hour h.

## Verdict
Fails the targets (avg trade +2.8% vs +10%; 2022/2026 not >= +50%). The long-after-discount leg is a weak,
crash-linked rebound likely overlapping the existing capitulation longs; the premium short leg has no edge.
