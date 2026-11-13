## Conclusion (2022-24 only)

Baselines: path_all +152 bp (not fragile), ens3 +149 bp (fragile through its 2022 LUNA trade).
46 new trials went into the ledger (23 rules x 2 score sets); 40 qualify and 6 fail (path_all
`tp5_stop10_1h`, `s3_q97`, `s5_q97`; ens3 `hold30m`, `s3_q97`, `s3_q99`).
The placebo criterion is met by every rule, so the binding ones are every-year positive, 0.6% cost
and one candle late.

- **Exits.** Holding longer helps and is the clearest exit result: 2h gives +221 (path_all) and
  +202 (ens3), 4h gives +235 / +253, against +152 / +149 at 1h; 30m is worse (+101 / +91). 4h is
  weak in 2024 (+54 / +84; ex-best-trade +16 for path_all) and has the lowest trades a year, so 2h
  is the safer step. The B1 take-profit finding is only partly confirmed: a +5% to +8% TP without
  a stop does not raise the mean for path_all (+109 / +110 vs +152) and is mixed for ens3 (+144
  to +158 vs +149), but it makes every year robust (ex-5-days +96 vs +85 for path_all; ens3 stops
  being fragile) and `tp5_2h` is the best of them (+146 path_all, +179 ens3, ex-5-days +129 /
  +148, not fragile). A +3% TP, trailing stops (3%: +54 / +71; 5%: +78 / +103) and the TP with a
  -10% stop (+60 / +82; fails for path_all) all hurt: they cut the right tail the strategy lives
  on. The -10% safety stop alone is neutral to mildly helpful (+133 path_all, +190 ens3).
- **Sizing.** Capping the stake at the cash (`cashcap`) is the same as the baseline for path_all
  (+149 vs +152) and +217 vs +149 for ens3, but that gain is the cash check skipping a LUNA-type
  entry in the baseline book, not a better rule (ens3 stops being fragile because the book now
  takes that trade as well). Score-weighted sizing (0.5 to 1.5) is slightly below `cashcap` in
  both (+145 vs +149, +210 vs +217): no gain, drop it.
- **Slots x threshold.** More slots at a high threshold is the best region: 10 slots at q 0.99
  gives +291 (path_all, 137 trades a year, not fragile) and +378 (ens3, 105 trades a year, not
  fragile, +247 one candle late), then 10 slots at q 0.98 (+170 / +168). The lower threshold 0.97
  is poor (path_all 3 and 5 slots fail; 10 slots +53), and 3 or 5 slots at q 0.99 are fragile
  (one trade in 2022). The gain at 10 x 0.99 comes with a smaller sample (300-400 trades in
  three years), so it is a candidate for S5, not a result.
- **Per-day cap.** Cap 2 and 3 lower path_all (+109 / +132) and raise ens3 (+181 / +173) while
  removing two thirds of the trades; ens3 with cap 2 or 3 is not fragile. Not a clear win.

Ranking by mean bp among qualifiers that are not fragile and keep every year positive without its
best trade: path_all `s10_q99` +291, `hold4h` +235, `hold2h` +221, `s10_q98` +170, then the
baseline +152; ens3 `s10_q99` +378, `hold4h`/`hold2h` (fragile, so not counted), `cashcap` +217,
`stop10_1h` +190, `cap2` +181, `tp5_2h` +179, `s10_q98` +168, then the baseline +149. For S2/S3
the useful carry-forwards are a 2h hold, 10 slots with q 0.98-0.99, cash-capped equal slots as the
sizing, and `tp5_2h` as the robustness exit; combinations were not tried here (the list was fixed).
