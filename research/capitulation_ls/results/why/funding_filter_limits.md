# Why the funding-sign filter works for big pumps and not small ones (diagnosis, not a rescue)

Scripts: `w8_base.py` (loader), `w8_a.py` (size x sign, year x sign, bootstrap, crowding), `w8_b.py` (2024 anatomy, market funding),
`w8_c.py` (gross vs funding bootstrap, tail budget). Run from this folder with `uv run --group research python w8_a.py` (then `w8_b.py`, `w8_c.py`; they read `/tmp/w8_T.parquet` written by `w8_a.py`).

**Sample.** Perp-active pumps, `qv_base >= 20k`, 2022-26. Each `ret_4h` bin (.25-.3, .3-.4, .4-.6, >=.6) is simulated on its own, 72h, isolated 1x (liquidated at +100%),
30 bp cost, perp funding paid, one position per coin at a time, **no slot cap** (`slots=1000`), so numbers differ from the booked 10-slot headline (e.g. unconstrained >=.4 with funding >= 0 is
+1,087 / +877 bp in the .4-.6 / >=.6 bins, n = 130 / 51). n = 1,654 trades; 2022-26 only. "f>=0" = last settled funding rate >= 0. "gross" = net + funding paid (price result after the 30 bp cost).
Bootstrap = resample pump-days (clusters), 4,000 draws, 95% CI.

## Answers in one paragraph

For pumps >= 0.4 the funding sign carries real information about the price: shorts entered at f<0 earn about nothing before funding (gross +6 bp, n = 209) against +1,138 bp at f>=0 (n = 181). For pumps
0.25-0.4 the sign carries **no detectable price information** (gross +297 vs +254 bp; gap +44 bp, CI -374..+457): the whole +383 bp "net" gap is funding cost avoided (339 bp). Avoided funding is
about 340-370 bp at every size, so it is a constant; the filter's large effect at big sizes comes from the price part, which the small pumps do not have. At small sizes the filter is a funding-cost
screen that leaves the ~3-6% liquidation tail intact, and in 2024 it barely screened at all.

## 1. Size x sign (unconstrained, 2022-26, bp per trade)

| pump size | sign | n | net | median net | gross | funding paid | max adverse >= +50% | liquidated (+100%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| .25-.3 | f<0 | 310 | -160 | +720 | +216 | 377 | 14.5% | 5.5% |
| .25-.3 | f>=0 | 440 | +178 | +674 | +206 | 28 | 12.0% | 2.7% |
| .3-.4 | f<0 | 245 | -78 | +796 | +301 | 379 | 19.2% | 5.7% |
| .3-.4 | f>=0 | 269 | +390 | +913 | +447 | 57 | 11.9% | 3.3% |
| .4-.6 | f<0 | 137 | -334 | +815 | +107 | 441 | 19.0% | 9.5% |
| .4-.6 | f>=0 | 130 | +1,087 | +1,472 | +1,190 | 103 | 13.1% | 3.8% |
| >=.6 | f<0 | 72 | -744 | +131 | -186 | 558 | 23.6% | 12.5% |
| >=.6 | f>=0 | 51 | +877 | +1,404 | +1,008 | 131 | 15.7% | 7.8% |

Bootstrap of the difference (f>=0 minus f<0), bp per trade, pump-day clusters:

| bin | net diff [95% CI] | gross (price) diff [95% CI] | funding diff [95% CI] |
| --- | --- | --- | --- |
| .25-.3 | +338 [-82, +770], P(<=0) 0.06 | -10 [-428, +416] | -348 [-423, -280] |
| .25-.4 (small) | +383 [-27, +807], P 0.035 | +44 [-374, +457] | -339 [-414, -271] |
| >=.4 (big) | +1,503 [+607, +2,345], P 0.000 | +1,132 [+249, +1,979], P 0.005 | -370 [-535, -230] |
| .4-.6 | +1,421 [+549, +2,258] | +1,083 [+248, +1,899] | -338 [-464, -219] |

* Funding saved is statistically solid and about the same everywhere (-340..-370 bp, CIs exclude 0). The price gap is only established for >= 0.4.
* Squeeze profile: at f<0 the max-adverse >= +50% rate is 14-24% vs 12-16% at f>=0 and the liquidation rate is 5.5-12.5% vs 2.7-7.8% (roughly 1.5-2.5x), so f<0 raises squeeze risk at every size.
  Yet at small sizes this does not reach the mean price result (gross about equal), because the f>=0 group's liquidations (2.7-3.3%) and its body (median gross is similar) offset it. At big sizes
  the body also differs (mean gross excluding liquidations +1,716 at f>=0 vs +1,144 at f<0; liquidation share 5.0% vs 10.5%).
* In the f<0 group, squeeze trades pay only slightly more funding than non-squeeze trades (504 vs 353 bp small; 518 vs 471 big), so funding is a cost that is mostly present whether or not the squeeze comes.
  (This is softer than the earlier note's 682 vs 207 bp, which was measured on the S40 sample; not reconciled here.)
* The 2025-10-10 crash day has no pump trades in this sample, so excluding it changes nothing.

## 2. Year by year (f>=0 vs f<0, net bp per trade, n in brackets)

| size | year | f>=0 | f<0 | gross gap | share of pumps with f<0 | pumps continuing (gross < 0) |
| --- | --- | --- | --- | --- | --- | --- |
| small | 2022 | +342 (67) | -497 (80) | +731 [-96, +1638] | 54% | 45% |
| small | 2023 | +108 (108) | +98 (71) | -191 [-1200, +846] | 40% | 35% |
| small | 2024 | **-120 (258)** | -598 (55) | +211 [-853, +1547] | **18%** | 39% |
| small | 2025 | +590 (186) | -19 (203) | +177 [-436, +792] | 52% | 30% |
| small | 2026 | +775 (90) | +4 (146) | n/a | 62% | 29% |
| big | 2022 | +1,789 (16) | -887 (29) | +2,563 [+1129, +3666] | 64% | 38% |
| big | 2023 | +1,361 (16) | +952 (21) | +329 [-1691, +2514] | 57% | 22% |
| big | 2024 | +280 (43) | -610 (13) | +472 [-1520, +2671] | 23% | 30% |
| big | 2025 | +844 (70) | -823 (85) | +1,076 [-432, ...] | 55% | 35% |
| big | 2026 | +1,791 (36) | -257 (61) | n/a | 63% | 27% |

* Net-diff bootstrap by year (all sizes): 2022 +1,222 [+373, +2,018], 2023 -23 [-1,130, +1,175], 2024 +538 [-562, +1,914], 2025 +916 [+168, +1,694], 2026 +1,138 [-35, +2,273]. The sign of the gap is positive in 4 of 5
  years for both size groups; 2023 is about zero for both. Per-year CIs for big pumps are +/- 2,000 bp wide (n = 13-85 per cell), so "stable year by year" is not established, only "never clearly negative".
* Neither 2024 cell is distinguishable from 2023 or 2025 (all CIs overlap).

## 3. Why 2024 is weak

* **The filter hardly filters in 2024.** Market-wide funding prints were negative only 9% of the time in 2024 (2022: 32%, 2025: 23%, 2026: 20%), median pump funding +1.0 bp, 32% of pumps at >= 3 bp. Only 18% of small pumps
  and 23% of big pumps were at f<0 (2022/25/26: 54-64%). So "f>=0" kept 82% of 2024 small pumps and is almost the unfiltered 2024 sample (hypothesis: in a bull regime funding is positive for everyone, so its
  sign says nothing about whether this particular coin's shorts are crowded).
* **The loss is the tail, not a body failure.** Small f>=0 in 2024: median +597 bp (2023 +712, 2025 +911), mean after trimming the worst 5% +347 (2023 +413). The mean is -120 because 10 of 258 trades (3.9%) were
  liquidated, -388 bp/trade by themselves, concentrated in Nov-Dec 2024 (OM 15 Nov, XVG 2 Dec, RSR 3 Dec; 102 of the year's 258 trades are in Nov-Dec, with net +38 and -617 bp per trade in the two months). 2023 had no
  liquidations (0%), which is what made it look fine.
* **Not more continuation, not higher funding paid, not young listings.** Share of pumps that continued: 2024 39% vs 45/35/30/29% in other years (2022 was higher). Funding paid in 2024 small f>=0 was -2 bp (the short *received* funding), the
  best of any year, and still lost. Median listing age was the oldest of the sample (1,029 days vs 503-781), so it is not a young-coin cohort.
* **Regime.** Nov-Dec 2024 alt rally: mean 24h market move +4.0% / +2.1% on pump days (my label for it; I did not test it as a cause, and with 3-9 trades in the other months there is no way to separate regime from a few coins).
  The 2024 pump days also had more pumps per day (2.7 vs 1.7-2.1), i.e. a broad alt rally where many coins pumped together and a few kept going.
* The year-level liquidation shares for small f>=0 (2022 1.5%, 2023 0%, 2024 3.9%, 2025 2.7%, 2026 5.6%) show the rate is tail-driven and noisy: 2026 has the highest share and still the best mean because its body is much stronger (median +1,632).

## 4. Is "rate >= 0" crowding, or funding cost?

* Rank correlations over 1,652 trades with metrics: rate vs long/short account ratio +0.26 (top-trader +0.23), rate vs 24h OI change -0.19, vs 4h OI change -0.01, vs taker ratio 0.00. So the sign shares some
  information with "shorts are crowded" but overlaps with it only weakly (R of 0.2-0.26).
* Funding cost vs sign: rate vs funding actually paid over 72h: Spearman -0.72. So sign is mostly a forecast of funding paid, and is mostly independent of crowding.
* OLS on net (bp, iid SEs, all sizes; n = 1,652, R2 0.01-0.02): sign alone +645 (se 158); sign + ls_acc z + OI-24h z + log size: sign +454 (se 164), ls_acc +86 (81), OI-24h -329 (81), size -78 (253).
  Controlling for crowding removes about 30% of the sign effect; the remaining 70% is not explained by ls_acc/OI.
* Within long/short-ratio terciles the sign matters mainly where shorts are crowded (low ls_acc, all sizes): f<0 -535 (n 322) vs f>=0 +280 (n 229), gap +815; mid tercile gap +804 (-232 vs +573); high (few crowded
  shorts): +354 vs +380, gap +26. For big pumps the gap is +1,881 / +1,681 / +337 in low / mid / high.
  So where shorts are not crowded the sign adds nothing, and funding paid there is small whichever sign it has.
* Decomposition of the gap: small = +383 net = +44 price + 339 funding; big = +1,503 = +1,132 price + 370 funding.
  Reading (partly speculation): at small sizes "rate >= 0" is a cost filter; at big sizes it also marks pumps whose shorts are not yet crowded, and those fade much more.

## 5. What blocks the rule, and what would have to be true

**Blockers**
1. For small pumps (< 0.4) the filter's value is purely the ~340 bp of funding avoided. The remaining net of about +180 to +390 bp per trade is set against a liquidation tail of 3-6% (-300 bp/trade at 3%).
   Net excluding liquidations is +571 bp for small f>=0; the liquidation share that would zero the net is about 5.4%, versus 3.0% pooled, 3.9% in 2024 and 5.6% in 2026. So the margin is thin and year-dependent.
2. For big pumps the effect is large and statistically clear pooled, but each yearly cell has 13-85 trades; the mean is made by 5-8% liquidations vs a +1,700 bp body.
3. In 2024 (and probably any bull-funding regime) f>=0 is not selective. The rule also keeps almost all pumps in 2023-24, so it cannot protect in those years.
4. The sign only partly proxies for crowding, so refining toward "crowded shorts" with the measured ratios is not supported: ls_acc and OI explain about 30% of the sign effect, with AUC-level weakness in earlier notes.

**What would have to be true for the rule to be tradable at small sizes**
* The price part of the sign gap would have to be positive and stable below 0.4. Measured +44 bp [-374, +457] pooled and -191..+731 by year; it would need roughly +300 bp or more with a CI above 0, which
  needs a few times more trades than the 709 / 555 here (my rough power estimate: CI half-width about 410 bp at that n).
* Or the liquidation share at small f>=0 would have to stay below ~5% in every regime year. 2026 is at 5.6%, 2024 at 3.9%.
* For >= 0.4: the pooled gap is there (+1,503, CI +607..+2,345); what is untested is whether it persists in a bull-funding year with only 13-43 f>=0 trades (2024: +280 bp), so one more such year would decide it.

Caveats: unconstrained sample (not the booked 10-slot book); rate is the last settled print only, not the predicted rate; day-clustered bootstrap treats days as independent; bin and year cells are small;
the 2024 regime link and the crowding reading are descriptions, not tested causes; ret_4h bins are exclusive here, so the >=0.4 headline numbers are not identical to cumulative-threshold runs.
