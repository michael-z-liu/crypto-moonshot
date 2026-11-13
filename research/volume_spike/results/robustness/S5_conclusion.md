## Conclusion

- **The pick did not hold up.** `S5:path_all5:e_s10q99` (10 slots, pct >= 0.99, 1h hold) ranked
  first on 2022-24 with +462 bp per trade and no fragile year. In 2025 it lost **-58 bp per trade**
  (125 trades, -12%, drawdown -16%). It was then +559 bp in 2026 on 36 trades, so 2025-26 is
  **+80 bp per trade** (161 trades, +7% over 22 months; +50 at 0.6%, +49 late). The 2025-26
  day-bootstrap 95% CI is [-136, +384] bp, and its 2025-26 Sharpe is 0.33. The S0 path_all
  baseline did as well or better over the same months (+94 bp, 384 trades, no latency).
- **The pick's out-of-sample result is a few days.** It traded on 15 days in 2025 and 7 in 2026.
  2025-10-10 alone was 31 trades summing to -12,832 bp (the 10 slots refilled through the
  crash). Without that day 2025 is +59 bp. Two days (2026-02-06 and 2026-06-04, 20 trades) carry
  2026. The high threshold makes the 2022-24 mean large and the sample small.
- **The runner-up** (`S5:ctx:f_btcvol6_hold2h`, 0.2 bp ahead of `S5:ctx:e_s10q99`) made +32 bp in
  2025 (+2 at 0.6%) and +276 in 2026, so +79 bp over 2025-26 with a -25% drawdown. Without
  2025-10-10 its 2025 is +109.
- **Across the 24 runs:** all are positive over 2025-26 (23 of 24 at 0.6% and late) because 2026
  is strong for every rule, but only 19 are positive in 2025, 14 at 0.6% and 10 one candle late.
  The 2022-24 ranking does not carry over to 2025: the Spearman correlation between the 2022-24
  mean and the 2025 bp is -0.20 (+0.36 against 2025-26).
  - The 2h-hold rules held up best. d_hold2h averaged +121 bp in 2025 and h_share75_hold2h
    averaged +142, against +40 for the 1h default.
  - The BTC-volume gates (b, f, g) lost most of their 2022-24 edge in 2025: +18, +1 and -10 bp
    on average. The take profit (g) was the worst rule.
  - 10 slots at q 0.99 (e) was worst in 2025 (-9 bp on average). Its 2025-26 average (+132) is
    one strong 2026.
  - ctx was the best score set (2025-26 average +103 vs +84 for path_all and +87 for path_all5),
    but that is within the spread of the rules.
- **1m fills change nothing for the pick and the runner-up.** Both exit on time only, and a time
  exit fills at the same 5m close on either resolution, so the check is identical by
  construction.
- **Deflated Sharpe.** The pick's 2022-24 Sharpe (1.83) beats the expected best of 115 no-edge
  trials (1.25) with probability 0.97 using its own skew and kurtosis. With normal returns the
  probability is 0.84. The 2022-24 test says the edge is unlikely to be pure selection. The
  2025-26 result (Sharpe 0.33, CI including 0) says it is much smaller now than any 2022-24
  number suggests.
