# S5 pick (written before any 2025-26 number of S5 was opened)

Plan: `docs/plans/cnn-robustness.md`, S5 Combination rule. Runner: `s5_confirm.py`. Every run: `CashCapped` equal slots, fee 0.1% + slippage 0.05% + latency 0.05% per side, continuous production-engine book; stresses keep the latency (0.6% = slippage 0.2% per side + latency 0.05%; late = one candle later). 2022-24 only.

24 runs; 24 qualify; 0 of those are fragile; 24 are eligible (qualified and not fragile). Rank = the Selection rule ranking among the eligible runs (mean bp per trade 2022-24, worst year as the tiebreak).

| Run | 2022 | 2023 | 2024 | Mean bp | Worst | 0.6% | Late | w/o 5 days | 95% CI | Trades/yr | DD 22 / 23 / 24 | Verdict | Fragile (w/o best 22 / 23 / 24) | Rank |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| path_all:a_1h | +147 | +206 | +62 | **+138** | +62 | +108 | +76 | +75 | [+58, +218] | 245 | -31% / -6% / -29% | qualifies | no (+17 / +191 / +50) | 24 |
| path_all:b_btcvol6 | +325 | +294 | +214 | **+278** | +214 | +247 | +173 | +152 | [+150, +443] | 140 | -8% / -2% / -14% | qualifies | no (+165 / +272 / +200) | 10 |
| path_all:c_btc4h-2 | +130 | +316 | +184 | **+210** | +130 | +179 | +116 | +144 | [+122, +315] | 155 | -14% / -2% / -15% | qualifies | no (+113 / +294 / +170) | 16 |
| path_all:d_hold2h | +234 | +312 | +85 | **+210** | +85 | +180 | +117 | +102 | [+78, +349] | 197 | -36% / -4% / -27% | qualifies | no (+75 / +294 / +65) | 15 |
| path_all:e_s10q99 | +513 | +426 | +213 | **+384** | +213 | +353 | +220 | +208 | [+194, +596] | 137 | -8% / 0% / -10% | qualifies | no (+206 / +404 / +200) | 4 |
| path_all:f_btcvol6_hold2h | +439 | +429 | +233 | **+367** | +233 | +336 | +243 | +187 | [+181, +630] | 119 | -20% / -1% / -15% | qualifies | no (+192 / +402 / +215) | 5 |
| path_all:g_btcvol6_tp5_2h | +183 | +236 | +173 | **+197** | +173 | +167 | +109 | +170 | [+144, +247] | 148 | -5% / -1% / -15% | qualifies | no (+181 / +234 / +171) | 18 |
| path_all:h_share75_hold2h | +385 | +363 | +123 | **+290** | +123 | +260 | +199 | +181 | [+169, +446] | 158 | -17% / -1% / -16% | qualifies | no (+203 / +343 / +99) | 8 |
| ctx:a_1h | +254 | +208 | +65 | **+176** | +65 | +145 | +98 | +101 | [+85, +266] | 228 | -17% / -4% / -17% | qualifies | no (+84 / +191 / +61) | 22 |
| ctx:b_btcvol6 | +538 | +257 | +177 | **+324** | +177 | +293 | +214 | +177 | [+164, +520] | 142 | -4% / -4% / -6% | qualifies | no (+274 / +234 / +166) | 7 |
| ctx:c_btc4h-2 | +192 | +275 | +134 | **+200** | +134 | +170 | +124 | +122 | [+105, +307] | 152 | -12% / -3% / -8% | qualifies | no (+169 / +253 / +122) | 17 |
| ctx:d_hold2h | +303 | +318 | +55 | **+225** | +55 | +195 | +135 | +87 | [+68, +417] | 185 | -32% / -2% / -21% | qualifies | no (+101 / +297 / +49) | 13 |
| ctx:e_s10q99 | +507 | +465 | +209 | **+394** | +209 | +363 | +263 | +217 | [+52, +658] | 128 | -24% / -0% / -6% | qualifies | no (+90 / +442 / +197) | 3 |
| ctx:f_btcvol6_hold2h | +625 | +404 | +153 | **+394** | +153 | +363 | +280 | +156 | [+138, +746] | 123 | -19% / -2% / -7% | qualifies | no (+219 / +377 / +139) | 2 |
| ctx:g_btcvol6_tp5_2h | +218 | +220 | +123 | **+187** | +123 | +157 | +90 | +157 | [+129, +238] | 152 | -6% / -2% / -8% | qualifies | no (+215 / +218 / +122) | 20 |
| ctx:h_share75_hold2h | +434 | +327 | +43 | **+268** | +43 | +237 | +188 | +146 | [+114, +446] | 163 | -19% / -2% / -21% | qualifies | no (+203 / +306 / +37) | 12 |
| path_all5:a_1h | +164 | +224 | +47 | **+145** | +47 | +115 | +78 | +69 | [+46, +244] | 235 | -37% / -4% / -20% | qualifies | no (+24 / +207 / +40) | 23 |
| path_all5:b_btcvol6 | +336 | +298 | +214 | **+283** | +214 | +252 | +169 | +155 | [+150, +451] | 137 | -9% / -3% / -10% | qualifies | no (+169 / +273 / +201) | 9 |
| path_all5:c_btc4h-2 | +119 | +278 | +158 | **+185** | +119 | +154 | +96 | +117 | [+94, +283] | 151 | -13% / -2% / -10% | qualifies | no (+101 / +253 / +144) | 21 |
| path_all5:d_hold2h | +263 | +326 | +46 | **+212** | +46 | +181 | +115 | +97 | [+83, +362] | 187 | -32% / -4% / -21% | qualifies | no (+88 / +306 / +28) | 14 |
| path_all5:e_s10q99 | +579 | +572 | +236 | **+462** | +236 | +431 | +268 | +272 | [+251, +692] | 127 | -9% / -0% / -9% | qualifies | no (+234 / +547 / +223) | 1 |
| path_all5:f_btcvol6_hold2h | +461 | +411 | +216 | **+363** | +216 | +332 | +232 | +186 | [+164, +621] | 116 | -22% / -2% / -11% | qualifies | no (+198 / +382 / +199) | 6 |
| path_all5:g_btcvol6_tp5_2h | +169 | +235 | +168 | **+191** | +168 | +160 | +98 | +161 | [+134, +239] | 145 | -6% / -1% / -10% | qualifies | no (+167 / +232 / +167) | 19 |
| path_all5:h_share75_hold2h | +375 | +374 | +65 | **+271** | +65 | +240 | +168 | +158 | [+143, +430] | 154 | -22% / -2% / -19% | qualifies | no (+180 / +352 / +41) | 11 |

## Pick

- **Pick: `S5:path_all5:e_s10q99`** (rank 1): mean +462.3 bp per trade 2022-24 (+579 / +572 / +236), worst year +236.2, 0.6% +431.0, late +268.1, w/o 5 best days +272.3, CI [+251, +692], 127 trades a year.
- **Runner-up: `S5:ctx:f_btcvol6_hold2h`** (rank 2): mean +394.1 (+625 / +404 / +153), worst year +153.0, 0.6% +362.9, late +280.2, w/o 5 best days +156.1, 123 trades a year.
- Rank 3: `S5:ctx:e_s10q99`, mean +393.91 against the runner-up's +394.08 (worst year +209.4): the rule ranks by the mean first, so it is third however small the gap.
- Reference (`S5:path_all:a_1h`, kept as a comparison by the rule): mean +138.4, worst year +62.1.
- Why: the plan's rule picks the top-ranked qualified, non-fragile run, nothing else. No ineligible run ranks above it.
