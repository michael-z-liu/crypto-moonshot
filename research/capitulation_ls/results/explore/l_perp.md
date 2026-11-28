# X13: L on perps vs spot (candidates 4224; spot trades 1883; perp trades 1619; both 1603)

| year | n both | spot net bp | perp net bp | perp - spot bp | funding paid bp (neg = received) |
|---|---|---|---|---|---|
| 2022 | 46 | +930 | +1144 | +213 | -68 |
| 2023 | 131 | +1368 | +1446 | +77 | -7 |
| 2024 | 936 | +1072 | +1102 | +31 | +5 |
| 2025 | 461 | +1631 | +1528 | -103 | -12 |
| 2026 | 29 | +1366 | +1387 | +20 | -5 |
| all | 1603 | +1258 | +1259 | +1 | -3 |

## Verdict: no gain from trading L on perps (neutral)

On the 1,603 candidates with both a spot and a perp path, the perp long earns the same as spot
(+1259 vs +1258 bp). The lower cost (0.3% vs 0.4%) and the funding received in flushes (3 bp on
average; 68 bp in 2022) are offset in 2025, when the perp wicks on 2025-10-10 went deeper and the
perp recovered less within the take-profit window (-103 bp against spot). Spot stays the venue
where it exists (no liquidation risk, no funding); the perp is an equivalent venue where spot
does not exist, which is what makes the perp-only extension work.
