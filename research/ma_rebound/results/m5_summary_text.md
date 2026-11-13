**Nothing qualifies, and nothing beats the baseline.** Of the 27 screened variants (9 trained, 18
rules on the base scores), none meets the pre-registered conditions on 2022-24.

- **2022 fails everywhere but one.** 26 of 27 variants lose in 2022, per trade (-16 to -117 bp)
  and as a book (-30% to -82%). The exception is the daily-MAs-only rule (`rule-tf1d`): +155 /
  +104 / +686 bp per trade, +4% / +7% / +48% a year, not fragile, +268 bp under the stress. It
  fails on trade count alone (29 / 87 / 67 trades a year against the required 100), its
  day-bootstrap CI is -73 to +664 bp, and without its 5 best days its mean is -34 bp.
- **Three trained variants rank far above the base on the seed mean, but not seed by seed.**
  Market breadth (+80.5%), distances to the other MAs (+68.3%) and short windows (+65.8%) are
  above the +38.8% bar on the seed-mean score. Their seeds average +22.5%, +25.1% and +36.3%,
  against the base's +18.0%, with spreads of 25 to 47 points. Only short windows gain over the
  base in every seed. All three lose 63% to 74% in 2022 and are fragile, so none qualifies.
- **Three rules rank above the bar and also fail on 2022:** percentile >= 0.9 (+48.7%), 1h + 4h
  MAs only (+44.0%, the only variant whose CI excludes zero: +2 to +59 bp) and 20 slots (+38.9%,
  the only rule that gains in every seed).
- **Clearly not helping:** dropping the BTC inputs (-17 points), the bracket target (-23), one
  model per timeframe group (-20), thresholds below 0.8 (-35 and -53), exits E2, E4, E5 and E6
  (-14 to -42), 5 slots (-29), and weekly + monthly MAs alone (30 trades a year, -13% average).
- **About the same as the base, within seed noise:** long windows, the path target, touch-market
  entries, buffer 0 or 0.15, exit E3, the 200-length MAs only, majors only, percentile >= 0.95.
- **Selection is real but small.** Every trained variant except the bracket target beats its
  placebo, and trades earn 14 to 68 bp more than BTC over the same windows. That was not enough
  to make a long-only book positive in the 2022 bear market.
- **Seed noise limits what the screen can tell.** Per-seed spreads run from 4 to 57 points, and a
  seed-mean score can rank 40 to 60 points above its own seeds' average (breadth, other MAs).
  Differences below about 20 points are not distinguishable.
