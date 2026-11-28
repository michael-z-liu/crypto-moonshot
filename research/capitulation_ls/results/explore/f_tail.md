# F's remaining squeeze tail: do crowding gates cut it without losing E[o]?

Scripts: `x6_lib.py` (pool, simulation, scorecard and book helpers), `x6_gates.py` (grid, split, diagnostics; run `uv run --group research python x6_gates.py`).
Outputs: `x6_scorecards.csv`, `x6_books.csv`, `x6_diag.csv` (all in this folder).

**Verdict: no gate works.** The gates that cut the stopped share also cut E[o] by about as much, and none improves the LMC+F book out of sample. F as built (funding >= 0) stays.

## Setup

* Base F is the d01 sleeve: r4 >= 0.4, perp, qv_base >= 20k, 2022+, last settled funding >= 0, 72h, isolated 1x, 0.3% cost + funding. It reproduces d01: 143 trades (no slot limit, one position per coin while held), E[o] +0.095, stopped 4.9%. Only **7 trades are stopped**, so every stopped-share number below has an error of about 2 points; differences of 1-3 points are noise.
* A gate filters the candidate rows before the simulation, so a coin's trigger row can move to a later row. All gate columns are known at entry (`feats`, `metric_feats`; the metrics cover every candidate row).
* Grid pre-declared, 15 gates plus the base. Levels come from the pool quartiles of the entry variables (not outcomes): ls_acc 0.75 / 1.0 (the 25th pct and about the median), oi_ch4h 0.12 / 0.34 (25th pct and median), taker_ls at its median, funding 1e-4, age 180 / 365 days. Both directions were run for OI and taker, because the sign of "new positions" is not known beforehand. Four combinations were added. This is 15 trials on 7 stop events, so any gate that looked good in-sample would still be a selection effect.
* Book = LMC (long, listing weak, crash short) + gated F, 10 slots, `e25_combined.book`, each year from $10k. `bp` = mean net bp per trade of the whole book; `geo` = geometric mean yearly return.

## 1. Scorecards (F trades, no slot limit)

n, E[o], stopped share and its contribution, years positive. Columns: all 2022-26 | choose-on 2022-24 | read 2025-26.

| gate | n | E[o] | stopped | contrib stopped | yrs+ | 22-24 n / E[o] / stopped | 25-26 n / E[o] / stopped |
|---|---|---|---|---|---|---|---|
| base | 143 | +0.095 | 4.9% | -0.049 | 5/5 | 61 / +0.076 / 3.3% | 82 / +0.110 / 6.1% |
| ls_acc >= 0.75 | 107 | +0.079 | 3.7% | -0.037 | 4/5 | 45 / +0.041 / 4.4% | 62 / +0.107 / 3.2% |
| ls_acc >= 1.0 | 82 | +0.091 | 2.4% | -0.024 | 5/5 | 35 / +0.078 / 0.0% | 47 / +0.100 / 4.3% |
| oi4h <= 0.34 | 105 | +0.096 | 5.7% | -0.057 | 5/5 | 38 / +0.090 / 2.6% | 67 / +0.099 / 7.5% |
| oi4h <= 0.12 | 57 | +0.147 | 5.3% | -0.053 | 5/5 | 19 / +0.194 / 0.0% | 38 / +0.124 / 7.9% |
| oi4h > 0.34 (reverse) | 61 | +0.072 | 4.9% | -0.049 | 5/5 | 34 / +0.079 / 2.9% | 27 / +0.062 / 7.4% |
| rate >= 1e-4 | 69 | +0.090 | 5.8% | -0.058 | 5/5 | 42 / +0.092 / 2.4% | 27 / +0.087 / 11.1% |
| rate24 >= 1e-4 | 49 | +0.059 | 10.2% | -0.102 | 5/5 | 31 / +0.065 / 3.2% | 18 / +0.049 / 22.2% |
| taker_ls <= median | 106 | +0.090 | 5.7% | -0.057 | 5/5 | 45 / +0.082 / 2.2% | 61 / +0.096 / 8.2% |
| taker_ls > median (reverse) | 130 | +0.101 | 4.6% | -0.046 | 5/5 | 56 / +0.083 / 1.8% | 74 / +0.114 / 6.8% |
| age >= 180d | 121 | +0.078 | 5.8% | -0.058 | 5/5 | 57 / +0.076 / 3.5% | 64 / +0.080 / 7.8% |
| age >= 365d | 107 | +0.095 | 5.6% | -0.056 | 5/5 | 52 / +0.093 / 3.8% | 55 / +0.097 / 7.3% |
| ls>=1.0 & rate>=1e-4 | 39 | +0.024 | 5.1% | -0.051 | 3/5 | 25 / +0.058 / 0.0% | 14 / -0.036 / 14.3% |
| ls>=1.0 & oi4h<=0.34 | 61 | +0.078 | 3.3% | -0.033 | 5/5 | 22 / +0.069 / 0.0% | 39 / +0.083 / 5.1% |
| ls>=1.0 & age>=180d | 69 | +0.086 | 2.9% | -0.029 | 5/5 | 32 / +0.083 / 0.0% | 37 / +0.088 / 5.4% |
| ls>=1.0 & rate>=1e-4 & oi4h<=0.34 | 25 | -0.034 | 8.0% | -0.080 | 2/5 | 14 / +0.028 / 0.0% | 11 / -0.113 / 18.2% |

(Years positive counted over the years that have trades; the exact fractions are in `x6_scorecards.csv`.)

## 2. The LMC+F book (10 slots), bp per trade of the whole book

| gate | all (trades) | 2022-24 | 2025-26 | geo yearly ret | worst year | years (22..26) | bp of the F trades, 22-24 / 25-26 |
|---|---|---|---|---|---|---|---|
| base | +1107 (539) | +1002 | +1199 | +183% | +101% | +101 +104 +142 +801 +103 | +762 / +1289 |
| ls >= 0.75 | +1088 | +968 | +1190 | +168% | +84% | +84 +91 +130 +826 +84 | +409 / +1237 |
| ls >= 1.0 | +1116 | +1056 | +1167 | +163% | +75% | +81 +82 +177 +694 +75 | +775 / +1141 |
| oi4h <= 0.34 | +1120 | +1054 | +1174 | +171% | +86% | +86 +98 +146 +685 +103 | +905 / +1242 |
| oi4h <= 0.12 | +1215 | +1170 | +1252 | +173% | +72% | +72 +99 +190 +730 +83 | +1937 / +1966 |
| oi4h > 0.34 | +1087 | +1045 | +1124 | +149% | +50% | +83 +85 +161 +622 +50 | +792 / +330 |
| rate >= 1e-4 | +1101 | +1055 | +1144 | +153% | +65% | +104 +98 +135 +563 +65 | +924 / +1086 |
| rate24 >= 1e-4 | +1081 | +1010 | +1147 | +140% | +60% | +76 +90 +133 +540 +60 | +649 / +1099 |
| taker <= med | +1106 | +1048 | +1157 | +171% | +69% | +87 +99 +162 +788 +69 | +824 / +1121 |
| taker > med | +1113 | +1023 | +1190 | +179% | +92% | +103 +98 +149 +786 +92 | +834 / +1233 |
| age >= 180d | +1093 | +1005 | +1171 | +174% | +103% | +103 +104 +134 +680 +103 | +757 / +1167 |
| age >= 365d | +1128 | +1054 | +1195 | +179% | +98% | +98 +104 +165 +631 +117 | +927 / +1428 |
| ls>=1.0 & rate>=1e-4 | +1088 | +1047 | +1126 | +140% | +51% | +82 +79 +148 +544 +51 | +581 / -114 |
| ls>=1.0 & oi4h<=0.34 | +1105 | +1065 | +1140 | +152% | +73% | +78 +84 +152 +614 +73 | +693 / +964 |
| ls>=1.0 & age>=180d | +1123 | +1068 | +1171 | +160% | +75% | +82 +82 +173 +651 +75 | +832 / +1134 |
| ls>=1.0 & rate>=1e-4 & oi4h<=0.34 | +1079 | +1051 | +1103 | +133% | +51% | +79 +81 +128 +508 +51 | +279 / -968 |

(Full columns, including the per-period F bp and trade counts, are in `x6_books.csv`. "Geo" is the geometric mean of the yearly returns, dominated by 2025's +801%; the bp column is the steadier read.)

Every gate moves the whole-book bp by less than +-30 bp except oi4h <= 0.12 (+108, on 57 F trades, 51 of them after the level was seen). The yearly returns fall for almost every gate (worst year +101% -> +50..+104%): fewer F trades free slots for weaker LMC trades or leave them empty, and the book's compounding is carried by a few large years.

## 3. The split (choose on 2022-24, read 2025-26)

Pre-declared rule: pick the gate with the highest 2022-24 E[o] among those that keep at least half of the base's 2022-24 trades (>= 31), and separately the gate with the lowest 2022-24 stopped contribution.

* By E[o]: oi4h <= 0.34 (+0.090 vs +0.076, n 38; oi4h <= 0.12 has +0.194 but only 19 trades, below the size floor). In 2025-26 it gives **+0.099 vs +0.110 for the base and stopped 7.5% vs 6.1%**: it does not carry over. Book 2025-26: +1174 vs +1199 bp.
* By stopped share: ls_acc >= 1.0 and every ls-combination have 0 stopped trades in 2022-24 (the base had 2, UNFI and XVG). In 2025-26 it keeps **4.3% vs 6.1% stopped** but E[o] drops to +0.100 vs +0.110, and the book gives +1167 vs +1199 bp and +2.7 vs +3.3 geo yearly return. The tail shrinks, the edge shrinks with it, and the book does not gain.
* Nothing selected in 2022-24 beats the base on E[o] or on the book in 2025-26. The two gates that reduce stopped in both halves (ls >= 1.0, ls >= 0.75) lose F per-trade bp in 2025-26 (+1289 -> +1141 for ls >= 1.0) and cost the book yearly return (2022-23 +101/+104% -> +81/+82%).

## 4. Why the gates fail (diag calibration, `x6_diag.csv`)

Calibration over terciles of the gate variable, on the base-F trades (n = 143, 7 stopped, 20 with max adverse >= +50%) and, for power, on P without the funding gate (r4 >= 0.4, any funding, 295 trades, 22 stopped, 53 with max adverse >= +50%).

1. **The crowding signal is real, as in `pump_shorts.md`, and it is priced.** AUC of ls_acc for adverse >= +50% is 0.32 in F and 0.36 in P-any (stopped: 0.34 / 0.33). Low ls_acc tercile in F: stopped 8.3% / 4.2% / 2.1% (low / mid / high), squeeze-50 22.9% / 12.5% / 6.4%. The same crowded tercile has the **highest** E[o], +0.111 / +0.097 / +0.078. Crowded shorts squeeze more often and also revert harder when they do not, so the extra tail is paid for in the body: a gate removes the squeezes together with the best mean reversions. It is the same flat-calibration pattern as P in `scorecards.md`, now inside F. Removing the low-ls_acc half (ls >= 1.0) deletes 61 trades (5 of the 7 stops) whose mean outcome is about +0.10; without the 5 stops the other 56 average about +0.20.
2. **With 7 events the tail is not predictable enough to gate.** ls >= 1.0 takes the stopped count from 7 to 2 of 82; under the base rate (4.9%, expected 4.0) a count of 2 or fewer has probability about 0.24. Of the 7 stopped trades, ALPACA (a frozen/delisted perp, noted in `pump_shorts.md`) and BAKE are not caught by any gate that keeps volume (BAKE has ls 1.35, the highest crowding-safe value in the list); the OI changes of the 7 (0.09-1.22) are spread through the whole range with a median of 0.23 vs 0.19 in the pool.
3. **OI during the pump has no monotone effect.** oi_ch4h AUC 0.43 (squeeze-50), 0.59 (stopped); terciles: stopped 2.1 / 8.3 / 4.3%, E[o] +0.186 / +0.044 / +0.056. The low-OI tercile is good for E[o] but its stopped share is 5.3% at oi4h <= 0.12 in the whole sample (0% in 2022-24, 7.9% in 2025-26). The tail came back out of sample. Rising OI (> 0.34, the reverse gate) does not raise stopped either (4.9%).
4. **Funding is already used up by the funding gate.** Within F, rate >= 1e-4 and rate24 >= 1e-4 do not lower the tail: stopped 5.8% and 10.2% (rate24's top tercile: 8.5% vs 4.2% and 2.1%). A higher positive rate means longs are paying to hold the pump, which is momentum as often as exhaustion; it earns the short more carry per hour but buys no price protection. Rate-based gates in the combinations drop E[o] to -0.034..+0.024 and the sleeve to 9-14 trades in 2025-26.
5. **Taker ratio is U-shaped.** AUC 0.67 for squeeze-50 but the tercile profile is 10.9% / 4.3% / 28.3%, stopped 6.5% / 0.0% / 8.7%; a one-sided gate (<= median or > median) cannot separate it (both give E[o] about +0.09..+0.10 and stopped 4.6-5.7%).
6. **Listing age** carries a small monotone trend (stopped 2.1 / 4.2 / 8.5% from young to old, E[o] +0.131 / +0.126 / +0.028), in the wrong direction for the usual intuition (old coins are the worse shorts); cutting young coins (age >= 180/365d) does not lower the tail (5.8% / 5.6%).

## Honest reading

* The only visible in-sample benefit is ls_acc >= 1.0 (stopped 4.9% -> 2.4%, E[o] +0.095 -> +0.091, F book bp +1107 -> +1116), and it is a trade of tail for edge, not a free cut: it removes 43% of F's trades and the book loses yearly return (geo +183% -> +163%, worst year +101% -> +75%). Selected after seeing the whole sample, from 15 variants and 7 events, it would not survive a multiple-testing correction.
* oi4h <= 0.12 is the best in-sample E[o] (+0.147) and book bp (+1215) but is the weakest on the split read (its 2022-24 half was a 0%-stopped sample of 19 trades) and has n = 57; treat as a lead for a future sample, not a result.
* The tail is a feature of the strategy (short volatility: crowded shorts squeeze more and pay more); the funding gate was the one gate that moved the price outcome (+1,100 bp gross difference). The next lever is more likely stake size per trade (smaller stake when ls_acc is low) than a binary gate; a size rule would keep the crowded-tercile's higher mean while capping its tail. It was not run here (it needs a per-trade weight in `book`, `w`), and with 7 stop events is hard to tune honestly.
