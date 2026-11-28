# X17: frozen settings (chosen on 2022-24) with and without the perp-only long

| book | slots (chosen on 2022-24) | trades | bp/trade | 2022 | 2023 | 2024 | 2025 | 2026 | 2025-26 bp/trade |
|---|---|---|---|---|---|---|---|---|---|
| LMC frozen | 8 | 611 | +719 | +106% (dd -9) | +188% (dd -9) | +165% (dd -22) | +371% (dd -34) | +54% (dd -26) | +634 |
| LMC+ frozen | 8 | 742 | +649 | +106% (dd -9) | +193% (dd -8) | +140% (dd -24) | +498% (dd -29) | +128% (dd -25) | +564 |

Perp-only long at the frozen settings: 647 trades, mean net +5.8% (2025-26: +6.0%).

(Importing `e45_frozen_2024` re-runs its own table, printed first; it has no main guard.)

## Reading

Under the honest protocol (every setting chosen on 2022-24, weak-listing filter measured at
entry), adding the perp-only long with the same frozen long settings raises 2025 from +371% to
+498% and 2026 from +54% to +128%; 2024 falls from +165% to +140% (slots shared). The per-trade
mean in 2025-26 stays below the +10% target in both books (+6.3% LMC, +5.6% LMC+): the frozen
long settings (pct >= 0.99, TP 10%) trade more and earn less per trade than the pick.
