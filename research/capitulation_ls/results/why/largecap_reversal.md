# Why the weekly large-cap reversal failed (E51)

Script: `w9_largecap_why.py` (same 30 coins, daily 00:00 UTC opens, Jan 2021 - Oct 2026, k=5 of up to 30, gross unless stated; 0.4% = round trip per leg). Diagnosis only.

## Bottom line
There is no reliable cross-sectional reversal in this universe at any horizon, and a small one would not clear costs anyway. The weekly rank IC (past 7d vs next 7d) is -0.030 (t -1.9, i.e. weakly reversal) but it is worth about 33 bp/week for a 5-coin book against a 40 bp round trip. The extremes behave the other way (losers -31 bp vs basket, winners +16), so the tails continue while the middle mean-reverts.

## 1. Horizons (formation F days, hold H days, non-overlapping; losers minus winners, gross; positive = reversal)
| F \ H | 1d | 3d | 7d | 14d | 28d |
| --- | --- | --- | --- | --- | --- |
| 1d | -15 (t -2.1) | +15 | +44 | +132 | +377 (n 76) |
| 3d | -10 | -24 | -48 | -73 | -133 |
| 7d | -15 (t -2.1) | -25 | -47 (t -0.9) | -130 | +205 (n 76) |
| 14d | -20 (t -2.8) | -67 (t -3.2) | -74 | -219 (t -1.8) | +119 |
| 28d | -16 (t -2.1) | -28 | -8 | -103 | +176 (n 76) |

bp per period. Short horizons (1-14d formation, 1-3d hold) show mild momentum (t up to -3.2 for 14d formation, 3d hold, -67 bp per 3 days gross), but a long-short needs 2 x 0.4% = 80 bp of cost per turnover, so net it is -95 to -150 bp. The positive cells are at 14-28d holds with only 76-151 non-overlapping periods (t about 1), and their sign flips with formation window (3d formation, 28d hold: -133). Nothing is consistent across neighbouring cells, which is what noise looks like. The fully long-only 7d/7d result: losers minus basket -31 bp/week, t -1.2.

## 2. Regime (7d/7d, losers minus basket, bp/week)
- BTC up last week (n 154): -42 (t -1.2); BTC down last week (n 148): -9. BTC last week below -5% (n 54): 0; above +5% (n 66): -97 (t -1.6).
- By next week's BTC move (look-ahead, diagnostic only): BTC next week up: +19; down: -72 (t -2.4). So the losers basket has beta about 1.01 to the basket but loses a bit extra in down weeks. It is not a buy-the-dip rebound.
- By year: 2021 -92, 2022 -31, 2023 -63, 2024 -43, 2025 -25, 2026 +132 (41 weeks). Five of six years negative; the sign "flips" only because 2026 alone is positive. Speculation: 2026's rebound may reflect a different market (lower beta dispersion, fewer new-narrative runs), not tested.

## 3. Is the drag in persistent downtrends?
Not really. Splitting the 5 weekly losers by 30-day return: worst-30d two coins -30 bp, best-30d two coins -24 bp (indistinguishable). Losers with 30d < -20% (n 396): -5 bp (t -0.1); the rest (n 1114): -33 bp. So persistent downtrends are not what hurts; if anything the milder losers do worse. Pooled regression of next-week excess on the past 7d excess has a tiny positive slope (0.03, momentum in magnitude, naive t 3 but cross-section is correlated so overstated) and the older 8-30d excess has zero (-0.002). Repeated losers are mixed: INJ +123, NEAR +131, AAVE +47 but SAND -58, HBAR -77 bp. No single mechanism; it is dispersion noise with fat tails (a few runaway winners that are not in the loser bucket).

## 4. Dispersion vs costs
Cross-sectional std of weekly returns: 17.6% (2021), 7.4%, 7.5%, 8.6%, 6.1%, 7.3% (2026); about 9.1% overall. Costs of 0.4% are 4% of one cross-sectional std, so there is plenty of spread to pay for trading; perfect foresight would earn about 24% per week on top5-bottom5. The problem is predictability, not spread: past 7d return explains about 0.1% of the next-week cross-section (pooled corr 0.03; rank IC -0.03). Required edge is small (a 40 bp cost needs IC of about -0.04 to -0.05 at k=5 net of nothing else), and the observed IC is just below that and unstable by year (2021 -0.09, 2022 +0.01, 2023 -0.01, 2024 -0.04, 2025 +0.03, 2026 -0.09).

## 5. Correlation
Average pairwise correlation of daily returns: 2021 0.48, 2022 0.71, 2023 0.54, 2024 0.53, 2025 0.72, 2026 0.59 (weekly: 0.43, 0.66, 0.56, 0.59, 0.66, 0.60). First principal component explains 52-73% of daily variance. Market-wide share of pooled weekly variance is 44%. Idiosyncratic spread still exists (std 6-9%), but high-correlation years (2022, 2025) are the ones with IC about 0 or positive, consistent with the reversal edge needing idiosyncratic overreaction that a common factor swamps. Note that a long-only losers book also carries about 1.0 beta, so its +40 bp/week is mostly the market (basket +71).

## What blocks it, what would have to be true
- Blocked by: (a) no stable sign of the 7d/7d effect (5 of 6 years negative gross vs basket, 2026 positive), (b) a measured edge of about 33 bp/week gross in rank terms against 40 bp round trip, (c) tails continue (winners +16 bp, losers -31), so a naive top/bottom-5 sort picks the wrong coins, (d) 0.4% cost on weekly turnover is a high bar for an effect of this size; maker fills at 0.2% would still leave about nothing.
- Would have to be true: a rank IC of about -0.06 or stronger and stable across years, or a conditioning variable that isolates idiosyncratic losers (low beta to BTC, no negative news) and lifts per-week excess above about 60-80 bp. The 30d-trend split tried here does not do it. Not tested: residual (beta-adjusted) returns, volume/news conditioning, and 28d holds, which look large (+90 to +377 bp) but have only 76 periods and inconsistent sign across formation windows, so they are not evidence.
