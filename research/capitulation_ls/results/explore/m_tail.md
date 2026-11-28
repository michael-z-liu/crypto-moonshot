# M tail: can the stop/squeeze tail shrink without selling the retraces? (in-sample)

Script: `x5_m_tail.py` (new; re-simulates the 143 weak listings of `$OWN/e27_listings.parquet`, since_list <= 0, 2021-26, on the
perp 1h candles of e13; entry 72h after listing, 30 days, TP 50%, 0.3% cost, funding). The base row reproduces e27 exactly
(mean net +0.2000, 143 trades). Scorecard columns follow `diag.py` (o = net on the stake). `E_all` = sum of o over the 143
candidates / 143, so a variant that skips trades is not flattered by a smaller sample. LMC = `sleeve_long` + this M variant +
`sleeve_crash_short`, 10 slots, `e25_combined.book` (M from 2022; vol-scaled variants pass their weight as the stake multiplier).
All variants tried are listed (25), chosen after looking at the same data: every number is in-sample and the variant picked would
carry selection bias. Variant parameters (X, Y, thresholds) are single round values, not tuned further.

| variant | n | E_o | E_all (per 143) | win | share_stopped | contrib_stopped | mean w | years>0 | worst yr E_o | LMC bp | LMC min yr % | LMC years % 22..26 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V0 base (stop 50) | 143 | +0.200 | +0.200 | 0.78 | 0.15 | -0.080 | 1.00 | 1.00 | +0.072 | +1158 | +51 | +62/+81/+156/+625/+51 |
| stop 75 | 143 | +0.200 | +0.200 | 0.80 | 0.09 | -0.070 | 1.00 | 1.00 | +0.072 | +1171 | +42 | +62/+76/+154/+709/+42 |
| stop 100 (liq only) | 143 | +0.199 | +0.199 | 0.80 | 0.06 | -0.063 | 1.00 | 1.00 | +0.074 | +1174 | +50 | +62/+72/+151/+687/+50 |
| add dn10 (half+half) | 143 | +0.181 | +0.181 | 0.75 | 0.15 | -0.072 | 1.00 | 1.00 | +0.055 | +1101 | +53 | +60/+70/+145/+567/+53 |
| add dn20 | 143 | +0.171 | +0.171 | 0.75 | 0.15 | -0.049 | 1.00 | 1.00 | +0.046 | +1068 | +54 | +59/+60/+140/+553/+54 |
| add up10 | 143 | +0.153 | +0.153 | 0.80 | 0.15 | -0.070 | 1.00 | 1.00 | +0.059 | +1030 | +43 | +60/+82/+151/+425/+43 |
| add up20 | 143 | +0.126 | +0.126 | 0.80 | 0.15 | -0.061 | 1.00 | 1.00 | +0.038 | +943 | +37 | +55/+67/+138/+366/+37 |
| fund exit <-0.02% | 143 | +0.122 | +0.122 | 0.69 | 0.07 | -0.035 | 1.00 | 1.00 | +0.036 | +937 | +48 | +48/+52/+159/+352/+52 |
| fund exit <-0.05% | 143 | +0.126 | +0.126 | 0.69 | 0.09 | -0.045 | 1.00 | 1.00 | +0.038 | +951 | +46 | +46/+54/+133/+389/+54 |
| fund exit <-0.10% | 143 | +0.135 | +0.135 | 0.66 | 0.11 | -0.056 | 1.00 | 1.00 | +0.044 | +978 | +46 | +48/+52/+137/+448/+46 |
| skip fund<0 at entry | 61 | +0.189 | +0.081 | 0.80 | 0.15 | -0.082 | 1.00 | 1.00 | +0.018 | +923 | +38 | +62/+67/+112/+168/+38 |
| skip bounce>=20% | 117 | +0.185 | +0.151 | 0.78 | 0.16 | -0.085 | 1.00 | 1.00 | +0.089 | +1063 | +45 | +62/+78/+173/+337/+45 |
| skip bounce>=30% | 132 | +0.196 | +0.181 | 0.79 | 0.16 | -0.083 | 1.00 | 1.00 | +0.081 | +1130 | +51 | +62/+81/+160/+491/+51 |
| skip bounce>=50% | 140 | +0.198 | +0.194 | 0.79 | 0.16 | -0.082 | 1.00 | 1.00 | +0.072 | +1145 | +51 | +62/+81/+156/+588/+51 |
| trail: BE stop after +25% | 143 | +0.167 | +0.167 | 0.69 | 0.15 | -0.077 | 1.00 | 1.00 | +0.039 | +1080 | +52 | +62/+75/+137/+527/+52 |
| trail: stop at -10% after +25% | 143 | +0.142 | +0.142 | 0.80 | 0.15 | -0.077 | 1.00 | 1.00 | +0.014 | +1005 | +50 | +56/+80/+115/+429/+50 |
| trail: 25% above low after +25% | 143 | +0.133 | +0.133 | 0.80 | 0.15 | -0.077 | 1.00 | 1.00 | +0.041 | +970 | +40 | +52/+74/+138/+416/+40 |
| trail: 40% above low after +25% | 143 | +0.159 | +0.159 | 0.76 | 0.15 | -0.077 | 1.00 | 1.00 | +0.022 | +1056 | +49 | +62/+78/+127/+511/+49 |
| trail: exit 72h after +25% | 143 | +0.136 | +0.136 | 0.80 | 0.15 | -0.077 | 1.00 | 1.00 | +0.052 | +972 | +41 | +60/+72/+142/+419/+41 |
| trail: exit 168h after +25% | 143 | +0.146 | +0.146 | 0.79 | 0.15 | -0.080 | 1.00 | 1.00 | +0.013 | +1000 | +51 | +58/+73/+114/+454/+51 |
| vol-scaled size | 143 | +0.193 | +0.193 | 0.78 | 0.15 | -0.080 | 1.02 | 1.00 | +0.088 | +1155 | +48 | +61/+82/+162/+506/+48 |
| vol-scaled + stop 75 | 143 | +0.195 | +0.195 | 0.80 | 0.10 | -0.074 | 1.02 | 1.00 | +0.068 | +1168 | +39 | +61/+78/+154/+592/+39 |
| stop75 + skip bounce30 | 132 | +0.191 | +0.176 | 0.80 | 0.10 | -0.075 | 1.00 | 1.00 | +0.056 | +1118 | +42 | +62/+76/+138/+560/+42 |
| stop75 + fund exit 0.05% | 143 | +0.123 | +0.123 | 0.70 | 0.06 | -0.042 | 1.00 | 1.00 | +0.046 | +953 | +46 | +46/+50/+137/+385/+54 |
| stop75 + skip fund<0 | 61 | +0.196 | +0.084 | 0.82 | 0.08 | -0.065 | 1.00 | 0.83 | -0.066 | +934 | +30 | +62/+67/+106/+204/+30 |

## Reading

- Only one family cuts `contrib_stopped` at no cost to E[o]: **a wider stop**. Stop 75%: share stopped 15% -> 9%, contrib -0.080 -> -0.070,
  E[o] +0.200 -> +0.200, win 78% -> 80%; liquidation only (100%): 6%, -0.063, E[o] +0.199. LMC +1158 -> +1171 / +1174 bp. But the saving
  is partly illusory: the trades that no longer stop at -50% still lose, many of them at -75/-100% (stopped trades are fewer but each is
  worse), the gain is E[o]-neutral rather than positive, and the book result is uneven: 2025 +625% -> +709% / +687% (a handful of trades)
  while 2026 falls +51% -> +42% (75%) and 2022-23 move a few points. The worst single outcome rises from -50% to -75/-100% of the stake.
  Honest verdict: neutral on expectancy, smaller stopped *share*, fatter single-trade tail; not an improvement worth adopting by itself.
- **Funding exit (funding turns negative, 3 settlements averaged): clearly bad.** Halves the stopped share (7-11%) but E[o] drops to
  +0.12..+0.14 and win rate to 66-69%. Crowded shorts (negative funding) are the phase when these coins keep falling, so it sells
  the retraces the sleeve lives on. LMC -200 bp.
- **Skip negative funding at entry: bad.** 82 of the 143 listings have negative funding at entry; the remaining 61 keep E[o] +0.189 but
  the stopped contribution is unchanged (-0.082) and the sleeve loses 57% of its edge (E_all +0.081).
- **Skip a bounce >= 20/30/50% from the pre-entry low:** removes 26/11/3 trades; E[o] does not rise (+0.185/+0.196/+0.198) and
  contrib_stopped stays -0.083..-0.085 (the squeezed coins are not the ones that bounced before entry). Total edge E_all falls.
  The 72h-bounce pattern carries no information about the later stop (consistent with `why/listings.md`: the paths have no bounce).
- **Half-size entry plus a second half on a further fall (X=10/20%):** contrib_stopped improves (-0.072/-0.049) only because half the
  stake is committed to the trades that went on; E[o] falls to +0.181/+0.171 (the second half is idle when price rises, and
  entering lower has smaller upside to the 50% TP measured from the first entry). Adding on a rise (Y=10/20%) is worse (+0.153/+0.126):
  it doubles down into squeezes.
- **Trailing after +25% favourable** (breakeven stop, -10% locked, trail 25/40% above the low, time exit 72/168h): does nothing for the
  stopped tail (the stopped trades never reached +25%, contrib unchanged at -0.077) and costs E[o] (+0.13..+0.17) because the coins retrace
  to +25..+40% and then fall further to the TP/time exit. This is exactly "selling the retraces".
- **Volatility-scaled size** (hourly return std of the 72h before entry, weight = median/vol clipped 0.5-1.5): E[o] +0.193, stopped
  contribution unchanged (-0.080), LMC +1155 bp. Early volatility does not predict which listings get squeezed (the tail is not the
  volatile coins).
- Combinations (stop 75 with a skip, with funding exit, or with vol sizing) inherit the loss of the weaker component; none beats stop 75
  alone on E[o].

## Why most fail

The tail is event-driven (a fresh news pump, a second exchange listing) rather than visible in the entry state: the 72h bounce,
the entry funding sign and the early volatility do not separate stopped trades from winners, so any gate on them removes winners at the
same rate (E[o] of kept trades stays ~+0.19 while the count shrinks). Exits that react to the favourable move or to funding are
correlated with the retraces that produce the +0.50 take-profit wins, so they trade the edge for tail cover one for one.

## Verdict

Nothing tried reduces `contrib_stopped` while *raising* E[o] or the LMC book in a way that survives scrutiny. A wide stop (75% or
liquidation) is E[o]-neutral and shrinks the stopped share, at the cost of a worse single-trade loss and a mixed per-year book effect.
The lever the diagnostics README names (sizing/gates) found no usable signal here; the 15% stop tail looks like a price of the edge.
Any adoption would need an out-of-sample check (2027 listings) first.
