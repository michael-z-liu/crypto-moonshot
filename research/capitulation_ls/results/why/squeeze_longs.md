# Why long pumps at negative funding do not work (diagnosis, not a rescue)

Script: `w11_squeeze_longs.py` (run from this folder with `uv run --group research python w11_squeeze_longs.py`; every number below is its output).
Sample: perp-active pumps, `qv_base >= 20k`, 2022-26; the long is entered at the next 5m open, isolated 1x, 30 bp cost, perp funding settled from the funding history, no slot cap,
one position per coin while held (each pool is simulated on its own, so the booked headline with 10 slots differs). `rate` = last settled funding rate. "price" = price return of the
long before costs and funding; "funding recv" = funding the long received (negative-funding longs are paid by the shorts).

## Answers in one paragraph

The outcome is **not bimodal**. For r4 >= 0.2 and rate < 0 the 72h price return has median -9%, quartiles -19%/+7%, and only 8.4% of trades end above +30% against 9.3% below -30%
(rate >= 0: 8.3% / 6.1%). Negative funding does not change the price distribution much (mean price -283 vs -142 bp, so it is slightly worse, not better); there is no squeeze edge to ride.
What the negative sign changes is the **funding transfer**: the long receives +326 bp over 72h on average (median +105, positive in 85% of trades) against +17 bp at rate >= 0.
That income is real but is spent on the pump's normal fade: the mean price result of -283 bp (median -927) plus 30 bp cost leaves net +14 bp (n = 676, years +72/-166/+224/+293/-461).
So the pump long and the pump short at negative funding are both near zero for the same reason: price is a pump-fade drift with fat tails that no entry-time number separates, and funding is a transfer
that favours the long (+326 bp) but is about equal to the fade. The short at rate < 0 nets -164 bp (n = 689) and the long +14 bp: the funding gap (about 650 bp round trip) is almost the whole difference.

## 1. 72h return distribution, funding sign x year (price return before costs and funding)

### 72h price-return distribution, ret_4h >= 0.2, long, no TP

| group | n | q05 | q25 | q50 | q75 | q95 | >+30% | <-30% | mean price bp | funding recv bp | net bp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rate<0 all | 676 | -0.38 | -0.19 | -0.09 | +0.07 | +0.47 | 8.4% | 9.3% | -283 | +326 | +14 |
| rate>=0 all | 881 | -0.32 | -0.16 | -0.06 | +0.06 | +0.43 | 8.3% | 6.1% | -142 | +17 | -154 |
| rate<0 2022 | 95 | -0.32 | -0.15 | -0.04 | +0.11 | +0.44 | 9.5% | 6.3% | +2 | +100 | +72 |
| rate>=0 2022 | 94 | -0.32 | -0.16 | -0.06 | +0.05 | +0.23 | 3.2% | 6.4% | -362 | +4 | -388 |
| rate<0 2023 | 91 | -0.40 | -0.19 | -0.08 | +0.09 | +0.30 | 6.6% | 12.1% | -403 | +266 | -166 |
| rate>=0 2023 | 148 | -0.29 | -0.14 | -0.05 | +0.06 | +0.43 | 8.8% | 4.1% | -55 | +37 | -48 |
| rate<0 2024 | 87 | -0.23 | -0.16 | -0.05 | +0.07 | +0.56 | 11.5% | 2.3% | +49 | +205 | +224 |
| rate>=0 2024 | 339 | -0.24 | -0.14 | -0.04 | +0.10 | +0.45 | 10.0% | 2.7% | +115 | -13 | +72 |
| rate<0 2025 | 244 | -0.32 | -0.17 | -0.10 | +0.09 | +0.38 | 6.6% | 6.6% | -95 | +418 | +293 |
| rate>=0 2025 | 211 | -0.34 | -0.20 | -0.11 | +0.04 | +0.36 | 6.6% | 8.1% | -304 | +29 | -305 |
| rate<0 2026 | 159 | -0.48 | -0.25 | -0.17 | +0.00 | +0.70 | 10.1% | 17.6% | -853 | +422 | -461 |
| rate>=0 2026 | 89 | -0.49 | -0.22 | -0.12 | +0.02 | +0.70 | 10.1% | 18.0% | -644 | +81 | -593 |

### 72h price-return distribution, ret_4h >= 0.4, long, no TP

| group | n | q05 | q25 | q50 | q75 | q95 | >+30% | <-30% | mean price bp | funding recv bp | net bp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rate<0 all | 166 | -0.47 | -0.25 | -0.14 | +0.10 | +0.51 | 7.8% | 16.9% | -633 | +499 | -164 |
| rate>=0 all | 140 | -0.50 | -0.29 | -0.15 | -0.03 | +0.38 | 6.4% | 23.6% | -973 | +88 | -915 |
| rate<0 2022 | 22 | -0.35 | -0.23 | -0.07 | +0.13 | +0.38 | 13.6% | 13.6% | -238 | +111 | -156 |
| rate>=0 2022 | 15 | -0.40 | -0.25 | -0.20 | -0.11 | +0.01 | 0.0% | 13.3% | -1902 | +22 | -1909 |
| rate<0 2023 | 18 | -0.51 | -0.23 | -0.12 | +0.06 | +0.49 | 11.1% | 16.7% | -743 | +354 | -419 |
| rate>=0 2023 | 11 | -0.32 | -0.23 | -0.14 | -0.03 | +0.05 | 0.0% | 18.2% | -1501 | +300 | -1230 |
| rate<0 2024 | 13 | -0.36 | -0.18 | -0.14 | -0.01 | +0.15 | 0.0% | 15.4% | -1032 | +302 | -760 |
| rate>=0 2024 | 34 | -0.31 | -0.19 | -0.12 | +0.04 | +0.38 | 8.8% | 8.8% | -554 | +22 | -562 |
| rate<0 2025 | 67 | -0.36 | -0.24 | -0.10 | +0.16 | +0.29 | 4.5% | 9.0% | -77 | +705 | +597 |
| rate>=0 2025 | 55 | -0.53 | -0.30 | -0.15 | -0.01 | +0.23 | 5.5% | 27.3% | -434 | +32 | -432 |
| rate<0 2026 | 46 | -0.51 | -0.37 | -0.23 | +0.02 | +0.52 | 10.9% | 30.4% | -1477 | +497 | -1010 |
| rate>=0 2026 | 25 | -0.53 | -0.43 | -0.28 | -0.08 | +0.39 | 12.0% | 44.0% | -1942 | +246 | -1726 |

rate<0 & r4>=.2: median last rate -8.8 bp/8h, mean -26.9; share rate < -5bp 61%; funding received over 72h mean +326 bp, median +105 bp; price-return mean -283 bp, median -927. Share of trades where funding received > 0: 85%

### Year anatomy (r4>=.2, rate<0)

| year | n | price mean bp | median bp | >+30% | <-30% | fund recv bp | net bp | mkt_r24h mean | rate mean bp | share rate<0 among pumps | continuing (ret>0) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2022 | 95 | +2 | -377 | 9.5% | 6.3% | +100 | +72 | +0.014 | -19.7 | 50% | 37% |
| 2023 | 91 | -403 | -790 | 6.6% | 12.1% | +266 | -166 | +0.020 | -29.7 | 38% | 37% |
| 2024 | 87 | +49 | -492 | 11.5% | 2.3% | +205 | +224 | -0.000 | -33.8 | 20% | 39% |
| 2025 | 244 | -95 | -1004 | 6.6% | 6.6% | +418 | +293 | -0.013 | -27.3 | 54% | 34% |
| 2026 | 159 | -853 | -1659 | 10.1% | 17.6% | +422 | -461 | +0.001 | -25.2 | 64% | 26% |

2025-26 by half-year (r4>=.2, rate<0): price mean / n / share>+30% / share<-30%

- 2025H2: n=143, price -169 bp, median -903, >+30% 6%, <-30% 6%, net +109
- 2026H1: n=106, price -742 bp, median -1588, >+30% 10%, <-30% 19%, net -361
- 2026H2: n=53, price -1075 bp, median -1791, >+30% 9%, <-30% 15%, net -661
- 2025H1: n=101, price +11 bp, median -1138, >+30% 7%, <-30% 8%, net +553

r4>=.2 & rate<0: 676 trades on 496 days; share of trades on days with >=3 such pumps 22%.


Reading: in every year the median price return is negative (-4% to -17%) for both signs; the tails are symmetric-ish and thin (6-12% beyond +30%, 2-18% beyond -30%). The 2023 and 2026 rate < 0 rows
have a heavy collapse tail (12% and 18% below -30%) and 2024 the lightest (2%). For r4 >= 0.4 the collapse tail is larger (17% below -30% vs 8% above +30% at rate < 0; 24% vs 6% at rate >= 0):
a bigger pump is more likely to collapse than to squeeze, whatever the funding sign. Sample sizes at r4 >= 0.4 are small (n = 10-67 per year).

Funding detail (r4 >= 0.2, rate < 0): median last rate -8.8 bp per 8h, mean -26.9 bp (a few extreme rates); 61% of entries are below -5 bp. The long receives +326 bp on average in 72h (+105 median),
which is more than 9 settlements at the mean entry rate (9 x 27 = 242 bp) and more than at the median (9 x 8.8 = 79 vs 105 bp): the rate typically gets more negative after entry, in the days when the pump fades. Receipts are paid in 85% of trades, so there is no "paid then squeezed" asymmetry: the negative-funding long is a small positive carry on a negative-drift asset.

## 2. 2025 vs 2026

Funding received is the same in both years (+418 vs +422 bp); the whole difference is price: -95 bp in 2025 against -853 bp in 2026 (median -1,004 vs -1,659 bp; share below -30%: 6.6% vs 17.6%;
share of trades that continue up: 34% vs 26%). By half year net is +553 (2025H1), +109 (2025H2), -361 (2026H1), -661 (2026H2). The 2025 result is therefore a year in which pumps did not fade as hard, not a better
squeeze signal: share of squeeze trades (>+30%) is 6.6% in 2025 and 10.1% in 2026, so 2026 even has more squeezes, but its collapses are much larger and more frequent. Another change: negative funding became the norm
(share of pumps with rate < 0: 20% in 2024, 54% in 2025, 64% in 2026), so the sign is less selective than it was; in 2024 only a fifth of pumps had it and those were the best (+224 bp), in 2026 it flags
a typical pump. Market drift does not explain it (mean mkt_r24h -0.013 vs +0.001), and the entries are spread over 496 days (22% on days with >= 3 such pumps), so it is not one crash day.

## 3. Does anything at entry separate squeezes from collapses?

### Entry-time features vs 72h price return (r4>=.2, rate<0); Spearman, 2022-24 | 2025-26 | all; and AUC for squeeze (>+30%) vs collapse (<-30%)

| feature | rho 22-24 | rho 25-26 | rho all | AUC squeeze-vs-collapse |
|---|---|---|---|---|
| ret_24h | -0.18 | -0.29 | -0.25 | 0.35 |
| off_low4h | -0.14 | -0.14 | -0.14 | 0.29 |
| r7d | -0.11 | -0.16 | -0.14 | 0.32 |
| mkt_r4h | +0.16 | +0.07 | +0.13 | 0.61 |
| r30d | -0.06 | -0.18 | -0.12 | 0.33 |
| wick_up | -0.08 | -0.14 | -0.12 | 0.41 |
| btc_ret_4h | +0.17 | +0.03 | +0.11 | 0.62 |
| vol30d | +0.08 | -0.20 | -0.11 | 0.38 |
| ret_4h | -0.13 | -0.08 | -0.10 | 0.34 |
| n_hot_bars4h | +0.03 | +0.11 | +0.10 | 0.51 |
| rate24 | +0.09 | +0.08 | +0.09 | 0.59 |
| vol4h | -0.00 | +0.10 | +0.07 | 0.48 |
| off_high4h | -0.09 | -0.08 | -0.07 | 0.37 |
| top_ls_acc | +0.12 | +0.06 | +0.07 | 0.65 |
| taker4h | -0.09 | -0.02 | -0.06 | 0.40 |
| oi_ch24h | -0.08 | -0.08 | -0.06 | 0.53 |
| oi_to_vol | +0.05 | +0.04 | +0.06 | 0.43 |
| n_prior_pumps30 | +0.08 | -0.09 | -0.05 | 0.43 |
| rate | +0.08 | +0.02 | +0.04 | 0.55 |
| qv_base | +0.01 | +0.03 | +0.04 | 0.49 |
| dhi30d | -0.07 | -0.05 | -0.04 | 0.42 |
| ls_acc | +0.08 | +0.04 | +0.03 | 0.62 |
| ls_acc_ch4h | +0.04 | -0.03 | -0.03 | 0.51 |
| vol_mult | +0.01 | +0.01 | +0.03 | 0.54 |
| taker | -0.05 | -0.01 | -0.03 | 0.51 |
| taker_ls | +0.01 | +0.03 | +0.03 | 0.51 |
| listed_days | +0.05 | -0.07 | -0.02 | 0.50 |
| oi_ch4h | -0.02 | +0.00 | +0.01 | 0.53 |
| hour | +0.01 | -0.06 | -0.01 | 0.58 |
| top_ls_pos | -0.01 | +0.06 | -0.00 | 0.53 |
| mkt_r24h | +0.05 | -0.06 | -0.00 | 0.56 |

Features with |rho 22-24| > .08 and same sign in 25-26: ret_4h (-0.13/-0.08), ret_24h (-0.18/-0.29), rate24 (+0.09/+0.08), off_high4h (-0.09/-0.08), off_low4h (-0.14/-0.14), taker4h (-0.09/-0.02), r7d (-0.11/-0.16), top_ls_acc (+0.12/+0.06), ls_acc (+0.08/+0.04), mkt_r4h (+0.16/+0.07), btc_ret_4h (+0.17/+0.03)
signal vs price return 22-24: Spearman +0.246; top-half price mean +333 bp, bottom-half -636; top-half net +421
signal vs price return 25-26: Spearman +0.202; top-half price mean +117 bp, bottom-half -864; top-half net +347


Weak. Single features have |rho| <= 0.25 (the two `y*` forward-label columns were excluded: they are outcomes, not entry features, and would have shown rho of 0.7-0.98). The most stable direction is "the coin is
already extended" (high 24h return, high 7d/30d return, far above the 4h low) gives lower returns: the pump that starts from a flat base does better than a second leg. Squeeze-vs-collapse AUCs only reach
0.62-0.65 (top_ls_acc, btc_ret_4h, ls_acc) or 0.29-0.35 on the other side. A composite (oriented percentile of the 11 features whose sign agreed in 2022-24 and 2025-26) has Spearman +0.25 / +0.20 with the 72h price return
in 2022-24 / 2025-26, but the features were screened on the same pool and the result does not become money: on the 2025-26 holdout (features chosen on 2022-24 only) the pool has E_o -0.005 and the top half +0.002.

## 4. Best long variant and scorecard

Variants tried (24/72h; no TP or 30% TP; r4 >= 0.2/0.3/0.4; rate < 0 or < -5 bp). Best by positive years then mean net: r4 >= 0.2, rate < -5 bp, 72h, no TP (+120 bp, n = 469, years +444/-410/+317/+395/-266).
It is selected from 24 variants and is not corrected for that. The 30% TP lifts some means (r4 >= 0.4, 72h, rate < 0: -164 to +63 bp) but the median stays negative (-398 bp) and years still swing (+471/-566/-136/+399/-367). The no-TP r4 >= 0.4 72h rows are negative in 4 years of five.

### Variant grid (unconstrained; net bp, years = mean net by year 22/23/24/25/26)

| r4 min | rate | hold | tp | n | net bp | median bp | years bp |
|---|---|---|---|---|---|---|---|
| 0.2 | <0 | 24h | None | 738 | -14 | -505 | +59/-301/+135/+44/-59 |
| 0.2 | <-5bp | 24h | None | 505 | +48 | -459 | +194/-415/+333/+117/-5 |
| 0.3 | <0 | 24h | None | 335 | -66 | -857 | -16/-628/-224/+258/-236 |
| 0.3 | <-5bp | 24h | None | 235 | +2 | -709 | -44/-614/-12/+261/-111 |
| 0.4 | <0 | 24h | None | 173 | -167 | -709 | -194/-1033/-809/+327/-369 |
| 0.4 | <-5bp | 24h | None | 131 | -107 | -600 | -667/-1099/-896/+495/-213 |
| 0.2 | <0 | 24h | 0.3 | 835 | -65 | -450 | +73/-179/+118/-45/-199 |
| 0.2 | <-5bp | 24h | 0.3 | 567 | -25 | -424 | +158/-267/+288/+43/-229 |
| 0.3 | <0 | 24h | 0.3 | 385 | -135 | -614 | +150/-329/-31/-57/-340 |
| 0.3 | <-5bp | 24h | 0.3 | 263 | -143 | -563 | +71/-257/+142/-6/-473 |
| 0.4 | <0 | 24h | 0.3 | 206 | -122 | -444 | +72/-573/-450/+133/-345 |
| 0.4 | <-5bp | 24h | 0.3 | 148 | -166 | -347 | -496/-610/-694/+252/-363 |
| 0.2 | <0 | 72h | None | 676 | +14 | -671 | +72/-166/+224/+293/-461 |
| 0.2 | <-5bp | 72h | None | 469 | +120 | -493 | +444/-410/+317/+395/-266 |
| 0.3 | <0 | 72h | None | 320 | -247 | -954 | -73/-294/-80/+201/-982 |
| 0.3 | <-5bp | 72h | None | 223 | -94 | -746 | +96/-311/-1/+358/-818 |
| 0.4 | <0 | 72h | None | 166 | -164 | -954 | -156/-419/-760/+597/-1010 |
| 0.4 | <-5bp | 72h | None | 124 | +34 | -712 | -177/-1099/-691/+1001/-771 |
| 0.2 | <0 | 72h | 0.3 | 790 | +11 | -482 | +118/-216/+175/+182/-250 |
| 0.2 | <-5bp | 72h | 0.3 | 538 | +86 | -337 | +319/-420/+351/+286/-180 |
| 0.3 | <0 | 72h | 0.3 | 374 | +26 | -546 | +359/-271/+71/+241/-323 |
| 0.3 | <-5bp | 72h | 0.3 | 254 | +94 | -435 | +437/-314/+321/+353/-329 |
| 0.4 | <0 | 72h | 0.3 | 202 | +63 | -398 | +471/-566/-136/+399/-367 |
| 0.4 | <-5bp | 72h | 0.3 | 143 | +61 | -254 | +334/-827/-358/+554/-338 |

Best by (positive years, mean net): r4>=0.2, rate<-5bp, 72h, tp=None.

### Scorecard (diag.py), best long variant (signal = oriented-percentile composite of the stable-sign features, pool percentile, theta = 0.5 (top half))

| strategy | n | E_o | median_o | win | share_tp_win | share_time_win | share_near_miss | share_never_went | share_went_against | share_stopped | contrib_stopped | tp_progress_med | stop_progress_med | giveback_med | worst_year_E_o | years_positive |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| long r4>=0.2 rate<-5bp 72h tp=None | 469 | -0.001 | -0.049 | 0.39 | 0.00 | 0.39 | 0.26 | 0.34 | 0.01 | 0.00 | +0.000 | 0.78 | 0.17 | 1.00 | -0.046 | 0.60 |

Holdout 2025-26 only (features chosen on 2022-24): {'n': 289, 'E_o': -0.005, 'median_o': -0.071, 'win': 0.349}; top half: {'n': 143, 'E_o': 0.002, 'median_o': -0.065, 'win': 0.329}
Note: E_o clips net at +-1, so squeezes above +100% are truncated; the unclipped mean is +120 bp, share net > +100%: 1.5%
Calibration by composite-signal decile (E_o, win, p_tp, p_stopped):

```
shape: (10, 8)
┌──────────┬─────┬───────────┬──────────┬──────┬───────────┬─────────────┬─────────┐
│ bin      ┆ n   ┆ E_o       ┆ win      ┆ p_tp ┆ p_stopped ┆ tp_progress ┆ entered │
│ ---      ┆ --- ┆ ---       ┆ ---      ┆ ---  ┆ ---       ┆ ---         ┆ ---     │
│ f64      ┆ u32 ┆ f64       ┆ f64      ┆ f64  ┆ f64       ┆ f64         ┆ f64     │
╞══════════╪═════╪═══════════╪══════════╪══════╪═══════════╪═════════════╪═════════╡
│ 0.002132 ┆ 47  ┆ 0.018157  ┆ 0.404255 ┆ 0.0  ┆ 0.0       ┆ 1.0         ┆ 1.0     │
│ 0.101279 ┆ 47  ┆ -0.011213 ┆ 0.340426 ┆ 0.0  ┆ 0.0       ┆ 0.829555    ┆ 1.0     │
│ 0.202559 ┆ 47  ┆ -0.050996 ┆ 0.340426 ┆ 0.0  ┆ 0.0       ┆ 0.720061    ┆ 1.0     │
│ 0.302772 ┆ 47  ┆ -0.038885 ┆ 0.382979 ┆ 0.0  ┆ 0.0       ┆ 0.771289    ┆ 1.0     │
│ 0.402985 ┆ 47  ┆ -0.019582 ┆ 0.404255 ┆ 0.0  ┆ 0.0       ┆ 0.698173    ┆ 1.0     │
│ 0.503198 ┆ 47  ┆ -0.046913 ┆ 0.255319 ┆ 0.0  ┆ 0.0       ┆ 0.520987    ┆ 1.0     │
│ 0.603412 ┆ 47  ┆ -0.017662 ┆ 0.319149 ┆ 0.0  ┆ 0.0       ┆ 0.639527    ┆ 1.0     │
│ 0.703625 ┆ 47  ┆ 0.051167  ┆ 0.489362 ┆ 0.0  ┆ 0.0       ┆ 0.874199    ┆ 1.0     │
│ 0.803838 ┆ 47  ┆ 0.027224  ┆ 0.425532 ┆ 0.0  ┆ 0.0       ┆ 0.874915    ┆ 1.0     │
│ 0.904051 ┆ 46  ┆ 0.083989  ┆ 0.5      ┆ 0.0  ┆ 0.0       ┆ 0.939883    ┆ 1.0     │
└──────────┴─────┴───────────┴──────────┴──────┴───────────┴─────────────┴─────────┘
```
short ref r4>=.2 rate<0: n=689, net -164 bp (long same pool above)
short ref r4>=.2 rate>=0: n=890, net +74 bp (long same pool above)
long ref r4>=.2 rate<0: n=676, net +14 bp
long ref r4>=.2 rate>=0: n=881, net -154 bp

(The long reference at the end is the same pool without the funding filter split: long net +14 bp at rate < 0 and -154 bp at rate >= 0; the short net -164 / +74. The sign filter is a funding-income screen, not a price signal.)

## What blocks the idea and what would have to be true

1. The long's edge is only the funding transfer (+326 bp mean per 72h); the price drift of a pump that has already run >= 20% in 4h is negative (median -9%) and as large, so net is about zero (+14 bp at rate < 0).
2. No tradeable squeeze: only 8% of rate < 0 pumps end above +30% and as many end below -30%; this holds in 2022, 2023 and 2026, and a bigger pump (>= 0.4) is worse (collapse share 17% vs squeeze 8%).
3. Entry-time features separate the groups only weakly (AUC <= 0.65, composite holdout E_o about 0), and the sign itself has lost selectivity (64% of 2026 pumps have rate < 0).
4. Year dependence: 2025 works because the price drag happened to be small (-95 bp) with normal funding income; 2026 fails on price (-853 bp) with the same income. That is a regime difference in the fade, not in the carry.

For it to work, one of these would need to be true: (a) a feature that predicts "this pump does not fade" with a hit rate high enough to lift the median above about -3% (the funding income only covers 300 bp), tested
out of sample; the best composite found here lifts the 2025-26 holdout E_o from -0.005 to +0.002, i.e. nothing; (b) funding income several times larger than the 326 bp seen (not tested here which entries would give it); or (c) a stable regime in which pump fades stay below about 3% mean, which 2022-24 and 2026 contradict.
