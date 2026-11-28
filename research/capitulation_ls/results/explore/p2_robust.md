### F1. Robustness of LMC+

Per-trade mean, day-block bootstrap (4,000 draws):

| trades | n | days | mean net | 95% CI (days resampled) |
|---|---|---|---|---|
| L-perp-only | 328 | 42 | 0.131 | +0.089 .. +0.151 |
| spot L | 1883 | 25 | 0.129 | +0.063 .. +0.169 |
| all L (spot + perp-only) | 2211 | 50 | 0.129 | +0.069 .. +0.166 |

Book with the L entry days of each year resampled (400 paired draws; M and C fixed):

| book | bp (actual) | bp 5-95% | worst yr (actual) | worst yr 5% / 50% | P(worst yr < 0) | 2022 median (5%) | 2023 median (5%) | 2024 median (5%) | 2025 median (5%) | 2026 median (5%) |
|---|---|---|---|---|---|---|---|---|---|---|
| LMC | 1158 | 1070 .. 1209 | +51% | +2% / +34% | 0.045 | +49% (+2%) | +62% (+40%) | +111% (+75%) | +542% (+450%) | +48% (+22%) |
| LMC+ | 1145 | 1069 .. 1193 | +62% | +2% / +48% | 0.045 | +49% (+2%) | +62% (+40%) | +101% (+65%) | +677% (+569%) | +96% (+59%) |

Worst year LMC+ minus LMC over the same draws: median +1 pts, 5% +0 pts, share of draws where LMC+ is worse 0.03.

Starting equity (each year restarts at this amount; stake <= 0.5% of the coin's daily volume):

| book | trades | bp | 2022 | 2023 | 2024 | 2025 | 2026 | min yr |
|---|---|---|---|---|---|---|---|---|
| LMC $10k | 430 | 1158 | +62% (dd -10) | +81% (dd -7) | +156% (dd -16) | +625% (dd -17) | +51% (dd -11) | +51% |
| LMC+ $10k | 485 | 1145 | +62% (dd -10) | +81% (dd -7) | +152% (dd -16) | +790% (dd -17) | +111% (dd -10) | +62% |
| LMC $50k | 430 | 1158 | +61% (dd -10) | +82% (dd -7) | +156% (dd -16) | +504% (dd -16) | +50% (dd -11) | +50% |
| LMC+ $50k | 485 | 1145 | +62% (dd -10) | +82% (dd -7) | +151% (dd -16) | +606% (dd -16) | +110% (dd -10) | +62% |
| LMC $250k | 430 | 1158 | +55% (dd -10) | +62% (dd -7) | +123% (dd -16) | +300% (dd -15) | +53% (dd -10) | +53% |
| LMC+ $250k | 485 | 1145 | +55% (dd -10) | +62% (dd -7) | +122% (dd -16) | +358% (dd -15) | +87% (dd -10) | +55% |

Entries one 5m candle late (L, L-perp-only, C):

| book | trades | bp | 2022 | 2023 | 2024 | 2025 | 2026 | min yr |
|---|---|---|---|---|---|---|---|---|
| LMC, on time | 430 | 1158 | +62% (dd -10) | +81% (dd -7) | +156% (dd -16) | +625% (dd -17) | +51% (dd -11) | +51% |
| LMC+, on time | 485 | 1145 | +62% (dd -10) | +81% (dd -7) | +152% (dd -16) | +790% (dd -17) | +111% (dd -10) | +62% |
| LMC, 1 candle late | 426 | 1027 | +41% (dd -10) | +67% (dd -6) | +111% (dd -16) | +546% (dd -16) | +50% (dd -12) | +41% |
| LMC+, 1 candle late | 478 | 962 | +42% (dd -10) | +69% (dd -6) | +103% (dd -16) | +616% (dd -16) | +68% (dd -12) | +42% |

L-perp-only alone: on time n 328, mean net +0.1312; 1 candle late n 408, mean net +0.1322, TP rate 0.71.

LMCF (x8_lib.sleeves()) with L-perp-only:

| book | trades | bp | 2022 | 2023 | 2024 | 2025 | 2026 | min yr |
|---|---|---|---|---|---|---|---|---|
| LMCF | 539 | 1107 | +101% (dd -10) | +104% (dd -6) | +142% (dd -20) | +801% (dd -32) | +103% (dd -19) | +101% |
| LMCF + L-perp-only | 589 | 1117 | +102% (dd -10) | +104% (dd -6) | +140% (dd -20) | +1034% (dd -32) | +184% (dd -12) | +102% |

