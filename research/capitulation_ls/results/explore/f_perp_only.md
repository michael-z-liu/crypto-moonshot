# F1. Pump shorts (F, P) on perp-only coins

Scripts: `f1_lib.py` (pools, perp simulation), `f1_eval.py` (this note). In-sample report of a fixed
rule; nothing chosen. F = x8_lib `sleeves()` F: 5m liquid spike rows with ret_4h >= +40%, perp active,
qv_base >= 20k, last settled funding rate >= 0 at the signal bar (none = 0), short the perp at the next
5m open, isolated 1x (stop at +100% adverse), 72h, cost 0.3% + funding, one position per coin.
P = F without the funding gate. Perp-only = p1 spikes (USD-M perps with no spot market); pre-spot =
p2 pre-spot spikes (perp window before a later spot listing). Funding keyed by the perp symbol.
The spot rows are the reference (re-simulated; F spot matches x8 sleeves() exactly: 143 trades, +9.53%).

Candidate rows (ret_4h >= 40%, active perp, qv_base >= 20k, 2022+): perp-only 4,799 (3,786 pass the gate), pre-spot 182 (145), spot 2,191 (1,089). Perp-only coins are mostly 2025-26 listings (p1 spikes: 1 coin in 2022, 8 in 2023, 37 in 2024, 166 in 2025, 183 in 2026), so 2022-24 are near empty.

## Verdict: F does not carry over to perp-only coins

* F perp-only: 313 trades, -5.2%/trade (median +13.4%), 19% stopped; on spot coins F is 143 trades, +9.5%, 5% stopped. P perp-only: 383 trades, -5.1%, 18% stopped. Pre-spot window: F 10 trades, -11.1% (too few to read). Almost all perp-only trades are 2025-26 (none in 2022, 1-6 a year in 2023-24).
* The typical trade works as on spot (median and non-stopped mean are the same or better); the loss is the squeeze tail: about 1 in 5 entries runs to +100% within 72h (closes, not wicks).
* The funding gate is inert there: funding is >= 0 on 79% of perp-only pump rows (spot 50%), and P -> F moves the mean -5.1% -> -5.2% (spot +2.5% -> +9.5%).
* Books: LMC+ + F (spot) +1034% (dd -32) / +184% (dd -12) in 2025 / 2026, worst dd -32%; adding F-perp-only gives +301% (dd -45) / +4% (dd -56), worst dd -56%, min year +102% -> +4%. Do not add F-perp-only (or P-perp-only).

## 1. Per year (trades, mean net)

| pool | 2022 | 2023 | 2024 | 2025 | 2026 | all | median | 95% CI (days) | stopped | funding paid | coins |
|---|---|---|---|---|---|---|---|---|---|---|---|
| F perp-only | 0 | 1, +2.7% | 4, +2.7% | 167, -5.7% | 141, -4.9% | 313, -5.2% | +13.4% | -10.7 .. +0.7% | 0.19 | +0.13% | 112 |
| P perp-only | 0 | 1, +2.7% | 6, +9.7% | 204, -4.8% | 172, -6.0% | 383, -5.1% | +13.3% | -10.4 .. +0.1% | 0.18 | +1.00% | 130 |
| F pre-spot | 0 | 1, +25.4% | 0 | 7, -20.2% | 2, +2.7% | 10, -11.1% | +0.6% | -36.4 .. +10.0% | 0.10 | +2.23% | 9 |
| P pre-spot | 0 | 2, +19.9% | 0 | 10, -9.3% | 2, +2.7% | 14, -3.4% | +6.4% | -24.0 .. +14.1% | 0.07 | +3.68% | 12 |
| F spot (ref) | 15, +18.5% | 11, +11.7% | 35, +1.7% | 57, +9.0% | 25, +15.4% | 143, +9.5% | +14.2% | +3.2 .. +15.4% | 0.05 | +0.97% | 99 |
| P spot (ref) | 35, +3.3% | 27, +7.2% | 45, +2.1% | 119, +0.1% | 69, +4.5% | 295, +2.5% | +9.9% | -2.5 .. +7.6% | 0.07 | +2.80% | 190 |

## 2. diag scorecards

| strategy | n | E_o | median_o | win | share_tp_win | share_time_win | share_near_miss | share_never_went | share_went_against | share_stopped | contrib_stopped | tp_progress_med | stop_progress_med | giveback_med | worst_year_E_o | years_positive |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F perp-only | 313 | -0.052 | +0.134 | 0.63 | 0.00 | 0.63 | 0.13 | 0.02 | 0.02 | 0.19 | -0.188 | 1.00 | 0.26 | 0.60 | -0.057 | 0.50 |
| P perp-only | 383 | -0.051 | +0.133 | 0.62 | 0.00 | 0.62 | 0.15 | 0.03 | 0.02 | 0.18 | -0.182 | 1.00 | 0.27 | 0.58 | -0.060 | 0.50 |
| F spot (ref) | 143 | +0.095 | +0.142 | 0.73 | 0.00 | 0.73 | 0.10 | 0.10 | 0.01 | 0.05 | -0.049 | 1.00 | 0.18 | 0.34 | +0.017 | 1.00 |
| P spot (ref) | 295 | +0.025 | +0.099 | 0.65 | 0.00 | 0.65 | 0.12 | 0.12 | 0.03 | 0.07 | -0.075 | 1.00 | 0.19 | 0.52 | +0.001 | 1.00 |
| F pre-spot | 10 | -0.111 | +0.006 | 0.50 | 0.00 | 0.50 | 0.20 | 0.20 | 0.00 | 0.10 | -0.100 | 1.00 | 0.35 | 0.83 | -0.202 | 0.67 |
| P pre-spot | 14 | -0.034 | +0.064 | 0.57 | 0.00 | 0.57 | 0.21 | 0.14 | 0.00 | 0.07 | -0.071 | 1.00 | 0.32 | 0.62 | -0.093 | 0.67 |


## 3. Calibration by pump size

Each ret_4h band simulated as its own pool (one position per coin within the band); bands below
40% are not entered by the rule (as if entered). o = net clipped to [-1, 1].

| pool | ret_4h | entered | n | E_o | median_o | win | p_stopped | giveback_med |
|---|---|---|---|---|---|---|---|---|
| F perp-only | 20%..30% | no | 975 | -0.022 | 0.080 | 0.626 | 0.094 | 0.559 |
| F perp-only | 30%..40% | no | 486 | -0.033 | 0.123 | 0.646 | 0.136 | 0.487 |
| F perp-only | 40%..60% | yes | 298 | -0.049 | 0.128 | 0.634 | 0.178 | 0.591 |
| F perp-only | 60%..80% | yes | 123 | -0.075 | 0.141 | 0.610 | 0.228 | 0.574 |
| F perp-only | 80%..120% | yes | 68 | -0.053 | 0.148 | 0.618 | 0.250 | 0.632 |
| F perp-only | >= 120% | yes | 27 | -0.074 | 0.233 | 0.556 | 0.296 | 0.438 |
| P perp-only | 20%..30% | no | 1165 | -0.020 | 0.080 | 0.630 | 0.094 | 0.558 |
| P perp-only | 30%..40% | no | 601 | -0.029 | 0.122 | 0.642 | 0.131 | 0.494 |
| P perp-only | 40%..60% | yes | 365 | -0.045 | 0.128 | 0.625 | 0.170 | 0.575 |
| P perp-only | 60%..80% | yes | 150 | -0.070 | 0.145 | 0.620 | 0.227 | 0.559 |
| P perp-only | 80%..120% | yes | 83 | -0.054 | 0.136 | 0.627 | 0.253 | 0.590 |
| P perp-only | >= 120% | yes | 38 | -0.062 | 0.199 | 0.579 | 0.263 | 0.491 |
| F spot (ref) | 20%..30% | no | 863 | 0.009 | 0.058 | 0.621 | 0.025 | 0.604 |
| F spot (ref) | 30%..40% | no | 275 | 0.027 | 0.087 | 0.651 | 0.044 | 0.455 |
| F spot (ref) | 40%..60% | yes | 132 | 0.109 | 0.147 | 0.758 | 0.038 | 0.315 |
| F spot (ref) | 60%..80% | yes | 43 | 0.133 | 0.177 | 0.767 | 0.047 | 0.358 |
| F spot (ref) | 80%..120% | yes | 24 | 0.023 | 0.053 | 0.583 | 0.167 | 0.666 |
| F spot (ref) | >= 120% | yes | 6 | 0.059 | 0.202 | 0.500 | 0.167 | 0.568 |
| P spot (ref) | 20%..30% | no | 1380 | -0.002 | 0.058 | 0.615 | 0.036 | 0.626 |
| P spot (ref) | 30%..40% | no | 514 | 0.017 | 0.083 | 0.640 | 0.045 | 0.492 |
| P spot (ref) | 40%..60% | yes | 267 | 0.036 | 0.117 | 0.682 | 0.067 | 0.446 |
| P spot (ref) | 60%..80% | yes | 98 | 0.038 | 0.112 | 0.673 | 0.082 | 0.560 |
| P spot (ref) | 80%..120% | yes | 51 | 0.038 | 0.139 | 0.588 | 0.118 | 0.649 |
| P spot (ref) | >= 120% | yes | 27 | -0.195 | -0.159 | 0.333 | 0.259 | 1.000 |

## 4. Books (10 slots, $10k each year, stake <= 0.5% of daily volume)

LMC+ = e25 L, M, C + L-perp-only (p2_robust.py). F rows: x8 priority 2 + ret_4h. Per year: return (max drawdown).

| book | trades | bp | 2022 | 2023 | 2024 | 2025 | 2026 | min yr | worst dd | F trades | F bp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LMC+ | 485 | 1145 | +62% (dd -10) | +81% (dd -7) | +152% (dd -16) | +790% (dd -17) | +111% (dd -10) | +62% | -17% | 0 | 0 |
| LMC+ + F (spot) | 589 | 1117 | +102% (dd -10) | +104% (dd -6) | +140% (dd -20) | +1034% (dd -32) | +184% (dd -12) | +102% | -32% | 129 | 1105 |
| LMC+ + F-perp-only | 759 | 466 | +62% (dd -10) | +82% (dd -7) | +146% (dd -16) | +100% (dd -49) | -20% (dd -60) | -20% | -60% | 297 | -569 |
| LMC+ + F (spot) + F-perp-only | 852 | 582 | +102% (dd -10) | +105% (dd -6) | +135% (dd -20) | +301% (dd -45) | +4% (dd -56) | +4% | -56% | 413 | 39 |
| LMC+ + F (spot) + F-perp-only + F-pre-spot | 864 | 525 | +102% (dd -10) | +110% (dd -6) | +135% (dd -20) | +179% (dd -53) | +5% (dd -56) | +5% | -56% | 428 | -49 |
| LMC+ + F (spot) + P-perp-only (info) | 911 | 540 | +102% (dd -10) | +105% (dd -6) | +144% (dd -20) | +344% (dd -51) | -32% (dd -69) | -32% | -69% | 473 | 21 |

## 5. Why it failed (numbers)

F trades cut by stop, repeat pumps, perp age (days since the perp's first candle) and liquidity:

| pool | cut | n | share | mean net | median net | stopped | mean price ret | mfe med | funding paid |
|---|---|---|---|---|---|---|---|---|---|
| F perp-only | all | 313 | 1.000 | -0.052 | 0.134 | 0.188 | -0.049 | 0.300 | 0.001 |
| F perp-only | not stopped | 254 | 0.812 | 0.168 | 0.212 | 0.000 | 0.172 | 0.336 | 0.001 |
| F perp-only | stopped | 59 | 0.188 | -0.998 | -1.000 | 1.000 | -1.000 | 0.087 | 0.002 |
| F perp-only | first pump (no prior 30d) | 136 | 0.435 | -0.046 | 0.131 | 0.176 | -0.043 | 0.284 | 0.000 |
| F perp-only | repeat pump (prior 30d) | 177 | 0.565 | -0.056 | 0.141 | 0.198 | -0.053 | 0.313 | 0.002 |
| F perp-only | perp age < 90d | 97 | 0.310 | -0.067 | 0.142 | 0.186 | -0.067 | 0.301 | -0.001 |
| F perp-only | perp age 90-365d | 180 | 0.575 | -0.037 | 0.124 | 0.178 | -0.032 | 0.286 | 0.003 |
| F perp-only | perp age >= 365d | 36 | 0.115 | -0.082 | 0.145 | 0.250 | -0.080 | 0.313 | 0.000 |
| F perp-only | qv_base < 100k | 172 | 0.550 | -0.049 | 0.115 | 0.163 | -0.045 | 0.282 | 0.002 |
| F perp-only | qv_base >= 100k | 141 | 0.450 | -0.055 | 0.151 | 0.220 | -0.053 | 0.334 | 0.000 |
| F perp-only | 2025-26 | 308 | 0.984 | -0.053 | 0.141 | 0.192 | -0.050 | 0.301 | 0.001 |
| F spot (ref) | all | 143 | 1.000 | 0.095 | 0.142 | 0.049 | 0.107 | 0.225 | 0.010 |
| F spot (ref) | not stopped | 136 | 0.951 | 0.152 | 0.153 | 0.000 | 0.164 | 0.233 | 0.010 |
| F spot (ref) | stopped | 7 | 0.049 | -1.000 | -1.000 | 1.000 | -1.000 | 0.092 | 0.009 |
| F spot (ref) | first pump (no prior 30d) | 96 | 0.671 | 0.099 | 0.125 | 0.021 | 0.112 | 0.213 | 0.010 |
| F spot (ref) | repeat pump (prior 30d) | 47 | 0.329 | 0.088 | 0.203 | 0.106 | 0.099 | 0.273 | 0.009 |
| F spot (ref) | perp age < 90d | 25 | 0.175 | 0.128 | 0.142 | 0.000 | 0.145 | 0.212 | 0.014 |
| F spot (ref) | perp age 90-365d | 31 | 0.217 | 0.112 | 0.193 | 0.032 | 0.119 | 0.269 | 0.006 |
| F spot (ref) | perp age >= 365d | 87 | 0.608 | 0.080 | 0.142 | 0.069 | 0.092 | 0.222 | 0.010 |
| F spot (ref) | qv_base < 100k | 114 | 0.797 | 0.110 | 0.165 | 0.053 | 0.125 | 0.247 | 0.012 |
| F spot (ref) | qv_base >= 100k | 29 | 0.203 | 0.036 | 0.077 | 0.034 | 0.038 | 0.199 | 0.001 |
| F spot (ref) | 2025-26 | 82 | 0.573 | 0.110 | 0.158 | 0.061 | 0.124 | 0.282 | 0.012 |

Funding at the signal bar (candidate rows):

| pool | rows | no rate yet | rate >= 0 (pass) | rate < 0 | rate <= -0.1% | median rate |
|---|---|---|---|---|---|---|
| perp-only | 4799 | 0 | 0.78891 | 0.21109 | 0.13961 | 0.00014 |
| spot (ref) | 2191 | 0 | 0.49703 | 0.50297 | 0.27978 | -0.00001 |

Trades the gate removes (P minus F): perp-only 70 (P -5.1% vs F -5.2%); spot 152 (P +2.5% vs F +9.5%).

Concentration: 112 coins, sum of net -16.16 stakes; the 10 worst coins sum to -17.41 (50 trades, 24 stops). Without them: n 263, mean +0.5%, stopped 0.13. Coins with a positive sum: 58 of 112.

| symbol | n | sum_net | stops |
|---|---|---|---|
| AIAUSDT | 5 | -2.160 | 3 |
| FHEUSDT | 4 | -2.060 | 2 |
| BRUSDT | 3 | -1.880 | 2 |
| RAVEUSDT | 3 | -1.760 | 2 |
| AKEUSDT | 9 | -1.740 | 3 |
| IRYSUSDT | 3 | -1.670 | 2 |
| COAIUSDT | 7 | -1.660 | 3 |
| AIOTUSDT | 5 | -1.530 | 2 |
| ZEREBROUSDT | 5 | -1.510 | 2 |
| PUMPBTCUSDT | 6 | -1.430 | 3 |

Price path after entry (perp close relative to the entry, short's view = minus the move; ignores the stop), and the share whose high reached +50% / +100% within 72h:

| pool | n | median short move 4h | mean short move 4h | median short move 24h | mean short move 24h | median short move 72h | mean short move 72h | median short move 7d | mean short move 7d | high >= +50% (72h) | high >= +100% (72h) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| F perp-only | 313 | 0.049 | -0.031 | 0.144 | 0.031 | 0.162 | -0.057 | 0.244 | -0.158 | 0.342 | 0.188 |
| F spot (ref) | 143 | 0.041 | 0.033 | 0.098 | 0.047 | 0.164 | 0.104 | 0.216 | 0.136 | 0.140 | 0.049 |

Were the stops wicks? (stop = the perp's last-price high reached +100%; a real isolated 1x short is liquidated on the mark price, which ignores single-print wicks):

| pool | stopped | max close >= +100% in 72h | max close +50..100% (stop on a wick) | max close < +50% (stop on a wick) | close at 72h >= +100% | median hours to stop |
|---|---|---|---|---|---|---|
| F perp-only | 59 | 48 | 10 | 1 | 17 | 11.0 |
| F spot (ref) | 7 | 5 | 1 | 1 | 2 | 6.5 |

F perp-only by quarter of entry:

| q | n | coins | mean_net | median_net | stopped |
|---|---|---|---|---|---|
| 2023-Q4 | 1 | 1 | 0.027 | 0.027 | 0.000 |
| 2024-Q1 | 2 | 2 | 0.012 | 0.012 | 0.000 |
| 2024-Q4 | 2 | 2 | 0.042 | 0.042 | 0.000 |
| 2025-Q1 | 6 | 6 | 0.176 | 0.255 | 0.000 |
| 2025-Q2 | 14 | 12 | -0.054 | 0.101 | 0.143 |
| 2025-Q3 | 41 | 27 | 0.007 | 0.143 | 0.146 |
| 2025-Q4 | 106 | 55 | -0.095 | 0.118 | 0.226 |
| 2026-Q1 | 28 | 23 | -0.082 | 0.028 | 0.107 |
| 2026-Q2 | 64 | 38 | 0.034 | 0.256 | 0.156 |
| 2026-Q3 | 30 | 20 | -0.082 | 0.128 | 0.233 |
| 2026-Q4 | 19 | 12 | -0.226 | -0.088 | 0.368 |

