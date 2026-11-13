# M5 pick (written before any 2025-26 number of issue #10 was opened)

Plan: `docs/plans/cnn-improvements.md`, Combination rule. Runner: `i10_m5.py`. Every run: `CashCapped(5)` equal slots, pct >= 0.98, highest score first, fee 0.1% + slippage 0.05% + latency 0.05% per side (0.4% round trip), one continuous production-engine book; stresses keep the latency (0.6% = slippage 0.2% per side; late = one candle later); 20 within-day placebos. 2022-24 only, seed-mean scores (5 seeds); per-seed rows use each seed's own percentiles.

Score sets: `volscale5` (M4 volscale, 5 seeds), `day5` (M3 day, 5 seeds), `volday5` (volscale + day weights, 5 seeds). Rules: (a) `a_h1_1h` h1 head, 1h hold; (b) `b_h1_2h` h1, 2h; (c) `c_h2_2h` h2 (2h median) head, 2h; (d) `d_h2_1h` h2, 1h; comparison rows, not eligible (n/e): `esc` (h1, 1h, escalating threshold per cascade) and `esc_2h` (esc, 2h hold).

12 eligible-rule runs; 12 qualify; fragile among those: 1; 11 are eligible (qualified and not fragile). Rank = the Selection rule among them (seed-mean mean bp per trade 2022-24, worst year as the tiebreak). Reference: base3 +120.9 bp (per seed +110.1 / +102.6 / +129.4); noise band 26.8 bp; the beats-the-baseline bar is +147.7.

| Run | 2022 | 2023 | 2024 | Mean bp | Worst | 0.6% | Late | Placebo p95 | w/o 5 days | 95% CI | Trades/yr | Verdict | Fragile (w/o best 22 / 23 / 24) | Seed means | Seed avg | Spread | Rank |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| volscale5:a_h1_1h | +242 | +377 | +51 | **+223.6** | +51 | +193 | +92 | -86 | +161 | [+120, +335] | 190 | qualifies | no (+66 / +356 / +47) | +196 / +178 / +112 / +73 / +122 | +136.2 | 123.4 | 9 |
| volscale5:b_h1_2h | +399 | +514 | +88 | **+333.5** | +88 | +303 | +179 | -114 | +206 | [+192, +504] | 152 | qualifies | no (+199 / +485 / +80) | +281 / +239 / +160 / +147 / +198 | +205.1 | 134.1 | 2 |
| volscale5:c_h2_2h | +322 | +410 | +79 | **+270.3** | +79 | +240 | +161 | -103 | +138 | [+123, +456] | 164 | qualifies | no (+146 / +383 / +71) | +298 / +215 / +108 / +160 / +165 | +189.2 | 189.6 | 4 |
| volscale5:d_h2_1h | +152 | +278 | +41 | **+157.0** | +41 | +127 | +65 | -85 | +77 | [+7, +285] | 206 | qualifies | **yes** (-1 / +259 / +37) | +198 / +153 / +75 / +64 / +61 | +110.2 | 136.7 | - |
| volscale5:esc | +323 | +460 | +103 | **+295.3** | +103 | +264 | +128 | -90 | +181 | [+158, +453] | 154 | qualifies | no (+120 / +436 / +98) | +228 / +206 / +154 / +92 / +150 | +165.9 | 136.0 | n/e |
| volscale5:esc_2h | +510 | +552 | +100 | **+387.2** | +100 | +356 | +220 | -108 | +210 | [+206, +603] | 139 | qualifies | no (+295 / +520 / +91) | +289 / +261 / +182 / +173 / +212 | +223.4 | 116.5 | n/e |
| day5:a_h1_1h | +254 | +223 | +65 | **+180.4** | +65 | +150 | +87 | -76 | +108 | [+81, +270] | 210 | qualifies | no (+51 / +210 / +60) | +135 / +115 / +149 / +143 / +134 | +135.2 | 34.8 | 10 |
| day5:b_h1_2h | +373 | +282 | +87 | **+247.1** | +87 | +216 | +135 | -110 | +99 | [+64, +451] | 171 | qualifies | no (+129 / +266 / +80) | +232 / +141 / +282 / +223 / +166 | +208.7 | 140.6 | 7 |
| day5:c_h2_2h | +382 | +266 | +80 | **+242.7** | +80 | +212 | +142 | -111 | +92 | [+72, +436] | 174 | qualifies | no (+139 / +251 / +72) | +227 / +150 / +292 / +237 / +160 | +213.2 | 142.6 | 8 |
| day5:d_h2_1h | +211 | +233 | +60 | **+167.7** | +60 | +137 | +96 | -78 | +90 | [+63, +271] | 214 | qualifies | no (+10 / +220 / +53) | +106 / +115 / +170 / +155 / +116 | +132.2 | 63.5 | 11 |
| day5:esc | +290 | +251 | +103 | **+214.6** | +103 | +184 | +101 | -78 | +143 | [+118, +303] | 179 | qualifies | no (+64 / +237 / +98) | +139 / +113 / +170 / +176 / +143 | +148.1 | 62.3 | n/e |
| day5:esc_2h | +364 | +292 | +74 | **+243.4** | +74 | +213 | +126 | -112 | +100 | [+59, +447] | 162 | qualifies | no (+111 / +277 / +66) | +239 / +139 / +296 / +237 / +162 | +214.5 | 157.1 | n/e |
| volday5:a_h1_1h | +360 | +308 | +99 | **+255.3** | +99 | +225 | +145 | -78 | +128 | [+122, +393] | 159 | qualifies | no (+98 / +282 / +94) | +173 / +113 / +278 / +157 / +205 | +185.1 | 164.9 | 5 |
| volday5:b_h1_2h | +499 | +422 | +154 | **+358.2** | +154 | +327 | +235 | -127 | +176 | [+152, +621] | 122 | qualifies | no (+171 / +391 / +146) | +295 / +176 / +344 / +258 / +322 | +278.9 | 168.4 | 1 |
| volday5:c_h2_2h | +430 | +405 | +97 | **+311.0** | +97 | +280 | +223 | -114 | +141 | [+113, +546] | 130 | qualifies | no (+140 / +377 / +89) | +272 / +181 / +297 / +229 / +295 | +254.7 | 116.5 | 3 |
| volday5:d_h2_1h | +394 | +292 | +67 | **+251.4** | +67 | +221 | +158 | -78 | +123 | [+115, +377] | 170 | qualifies | no (+158 / +268 / +63) | +175 / +121 / +216 / +134 / +191 | +167.1 | 95.3 | 6 |
| volday5:esc | +390 | +407 | +180 | **+325.6** | +180 | +295 | +188 | -69 | +205 | [+187, +451] | 128 | qualifies | no (+102 / +376 / +174) | +211 / +109 / +303 / +207 / +306 | +227.1 | 196.8 | n/e |
| volday5:esc_2h | +512 | +479 | +168 | **+386.1** | +168 | +355 | +257 | -120 | +176 | [+161, +648] | 110 | qualifies | no (+165 / +447 / +159) | +324 / +225 / +384 / +280 / +365 | +315.3 | 159.1 | n/e |

## Pick

- **Pick: `M5:volday5:b_h1_2h`** (rank 1): mean +358.2 bp per trade 2022-24 (+499 / +422 / +154), worst year +154.5, 0.6% +327.2, late +235.4, placebo p95 -126.8, w/o 5 best days +176.2, CI [+152, +621], 122 trades a year; per seed +294.7 / +175.6 / +344.0 / +258.2 / +321.8 (avg +278.9, spread 168.4, 1 of 5 seeds fragile). Beats base3 (> +147.7): yes.
- **Runner-up: `M5:volscale5:b_h1_2h`** (rank 2): mean +333.5 bp per trade 2022-24 (+399 / +514 / +88), worst year +87.6, 0.6% +302.5, late +178.8, placebo p95 -113.7, w/o 5 best days +205.9, CI [+192, +504], 152 trades a year; per seed +281.3 / +238.5 / +160.0 / +147.2 / +198.5 (avg +205.1, spread 134.1, 2 of 5 seeds fragile). Beats base3 (> +147.7): yes.
- Rank 3: `M5:volday5:c_h2_2h`, mean +311.01 against the runner-up's +333.49 (worst year +97.2).
- Why: the plan's rule picks the top-ranked qualified, non-fragile eligible run, nothing else. Ranked at or above it by mean bp but not eligible: `M5:volscale5:esc_2h` (+387.2 bp; comparison row), `M5:volday5:esc_2h` (+386.1 bp; comparison row).

## Sanity: seeds 0-2 of the 5-seed runs (`i10_m5.py sanity`)

Per-seed percentiles (h1 and h2 heads) of `volscale5` / `day5` seeds 0-2 against the 3-seed `volscale` / `day` runs, every scored row 2022-26 (scores only):

```
volscale5 vs volscale pct_seed[0]: 1456656 rows, max abs diff 0, rows differing 0, same NaN pattern True
volscale5 vs volscale pct_seed[1]: 1456656 rows, max abs diff 0, rows differing 0, same NaN pattern True
volscale5 vs volscale pct_seed[2]: 1456656 rows, max abs diff 0, rows differing 0, same NaN pattern True
volscale5 vs volscale pct_seed_h2[0]: 1456656 rows, max abs diff 0, rows differing 0, same NaN pattern True
volscale5 vs volscale pct_seed_h2[1]: 1456656 rows, max abs diff 0, rows differing 0, same NaN pattern True
volscale5 vs volscale pct_seed_h2[2]: 1456656 rows, max abs diff 0, rows differing 0, same NaN pattern True
day5 vs day pct_seed[0]: 1456656 rows, max abs diff 0, rows differing 0, same NaN pattern True
day5 vs day pct_seed[1]: 1456656 rows, max abs diff 0, rows differing 0, same NaN pattern True
day5 vs day pct_seed[2]: 1456656 rows, max abs diff 0, rows differing 0, same NaN pattern True
day5 vs day pct_seed_h2[0]: 1456656 rows, max abs diff 0, rows differing 0, same NaN pattern True
day5 vs day pct_seed_h2[1]: 1456656 rows, max abs diff 0, rows differing 0, same NaN pattern True
day5 vs day pct_seed_h2[2]: 1456656 rows, max abs diff 0, rows differing 0, same NaN pattern True
```
