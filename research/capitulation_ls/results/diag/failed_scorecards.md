# Scorecards of the failed variants (diag.py, d02_failed_scorecards.py)

Same scales as `scorecards.md`: outcome o = net return on the stake (clipped to [-1, +1]), shares of trades,
`contrib_*` sums to E_o. 2022-26 (quiet explorer: 2021-26), every candidate trade, no slot limit. Rows named `*_below` are the
candidates under the threshold, simulated as if entered. Calibration files: `calibration_<name>.csv`.

| strategy | n | E_o | median_o | win | share_tp_win | share_time_win | share_near_miss | share_never_went | share_went_against | share_stopped | contrib_stopped | tp_progress_med | stop_progress_med | giveback_med | worst_year_E_o | years_positive |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L_day5 | 1883 | +0.129 | +0.196 | 0.85 | 0.65 | 0.20 | 0.05 | 0.09 | 0.01 | 0.00 | -0.001 | 1.00 | 0.06 | 0.15 | +0.107 | 1.00 |
| L_day5_below | 14499 | +0.012 | +0.009 | 0.54 | 0.17 | 0.37 | 0.06 | 0.39 | 0.01 | 0.00 | -0.000 | 0.44 | 0.07 | 0.88 | -0.036 | 0.80 |
| L_tp20day | 1299 | +0.155 | +0.196 | 0.93 | 0.82 | 0.11 | 0.04 | 0.02 | 0.01 | 0.00 | -0.001 | 1.00 | 0.06 | 0.23 | -0.147 | 0.80 |
| L_tp20day_below | 10538 | +0.016 | +0.014 | 0.55 | 0.19 | 0.35 | 0.08 | 0.36 | 0.01 | 0.00 | -0.000 | 0.48 | 0.07 | 0.83 | -0.032 | 0.60 |
| L_day_transformer | 1803 | +0.121 | +0.196 | 0.85 | 0.59 | 0.26 | 0.05 | 0.08 | 0.01 | 0.00 | -0.001 | 1.00 | 0.07 | 0.17 | +0.099 | 1.00 |
| L_day_transformer_below | 15059 | +0.015 | +0.012 | 0.55 | 0.17 | 0.38 | 0.06 | 0.38 | 0.01 | 0.00 | -0.000 | 0.45 | 0.07 | 0.84 | -0.039 | 0.80 |
| L_day_tcn | 1687 | +0.124 | +0.196 | 0.84 | 0.63 | 0.21 | 0.05 | 0.10 | 0.01 | 0.00 | -0.001 | 1.00 | 0.06 | 0.16 | +0.100 | 1.00 |
| L_day_tcn_below | 15071 | +0.014 | +0.010 | 0.54 | 0.17 | 0.37 | 0.06 | 0.38 | 0.01 | 0.00 | -0.000 | 0.44 | 0.07 | 0.86 | -0.046 | 0.80 |
| L_day_gru | 1942 | +0.114 | +0.196 | 0.83 | 0.57 | 0.26 | 0.06 | 0.11 | 0.01 | 0.00 | -0.001 | 1.00 | 0.07 | 0.20 | +0.084 | 1.00 |
| L_day_gru_below | 15498 | +0.011 | +0.009 | 0.53 | 0.18 | 0.35 | 0.07 | 0.39 | 0.01 | 0.00 | -0.000 | 0.45 | 0.08 | 0.88 | -0.031 | 0.80 |
| L_day5_th099 | 2934 | +0.100 | +0.168 | 0.80 | 0.50 | 0.30 | 0.07 | 0.13 | 0.01 | 0.00 | -0.001 | 0.99 | 0.06 | 0.25 | +0.047 | 1.00 |
| L_day5_th099_below | 14370 | +0.010 | +0.008 | 0.53 | 0.17 | 0.37 | 0.06 | 0.39 | 0.01 | 0.00 | -0.000 | 0.43 | 0.07 | 0.89 | -0.037 | 0.80 |
| L_day5_th098 | 5348 | +0.047 | +0.049 | 0.66 | 0.28 | 0.37 | 0.08 | 0.25 | 0.02 | 0.00 | -0.000 | 0.67 | 0.07 | 0.59 | -0.031 | 0.80 |
| L_day5_th098_below | 13775 | +0.008 | +0.005 | 0.52 | 0.16 | 0.37 | 0.06 | 0.40 | 0.01 | 0.00 | -0.000 | 0.42 | 0.07 | 0.92 | -0.039 | 0.80 |
| L_day5_lev2 | 2254 | +0.178 | +0.392 | 0.80 | 0.66 | 0.15 | 0.04 | 0.04 | 0.04 | 0.08 | -0.081 | 1.00 | 0.12 | 0.00 | +0.105 | 1.00 |
| L_day5_lev2_below | 14771 | +0.018 | +0.021 | 0.54 | 0.18 | 0.36 | 0.06 | 0.31 | 0.07 | 0.02 | -0.018 | 0.44 | 0.15 | 0.72 | -0.074 | 0.60 |
| L_day5_lev3 | 2350 | +0.261 | +0.588 | 0.78 | 0.64 | 0.14 | 0.03 | 0.01 | 0.05 | 0.12 | -0.122 | 1.00 | 0.19 | 0.00 | +0.123 | 1.00 |
| L_day5_lev3_below | 15030 | +0.032 | +0.035 | 0.54 | 0.19 | 0.36 | 0.06 | 0.23 | 0.12 | 0.05 | -0.049 | 0.45 | 0.22 | 0.54 | -0.115 | 0.80 |
| P_nostop | 295 | +0.025 | +0.099 | 0.65 | 0.00 | 0.65 | 0.12 | 0.12 | 0.03 | 0.07 | -0.075 | 1.00 | 0.19 | 0.52 | +0.001 | 1.00 |
| P_nostop_below | 2507 | -0.006 | +0.042 | 0.60 | 0.00 | 0.60 | 0.11 | 0.22 | 0.05 | 0.03 | -0.027 | 0.67 | 0.12 | 0.68 | -0.027 | 0.60 |
| P_sl50 | 317 | +0.012 | +0.082 | 0.61 | 0.00 | 0.61 | 0.09 | 0.05 | 0.06 | 0.19 | -0.100 | 1.00 | 0.38 | 0.60 | -0.002 | 0.60 |
| P_sl50_below | 2594 | -0.008 | +0.040 | 0.59 | 0.00 | 0.59 | 0.09 | 0.11 | 0.10 | 0.10 | -0.052 | 0.67 | 0.24 | 0.69 | -0.026 | 0.40 |
| C_calm1 | 161 | +0.044 | +0.036 | 0.60 | 0.00 | 0.60 | 0.11 | 0.27 | 0.01 | 0.02 | -0.019 | 0.66 | 0.09 | 0.67 | +0.036 | 1.00 |
| C_calm1_below | 274 | +0.013 | +0.004 | 0.51 | 0.00 | 0.51 | 0.09 | 0.38 | 0.00 | 0.01 | -0.011 | 0.50 | 0.08 | 0.92 | -0.015 | 0.80 |
| C_calm2 | 198 | +0.028 | +0.033 | 0.59 | 0.00 | 0.59 | 0.09 | 0.28 | 0.02 | 0.02 | -0.020 | 0.59 | 0.09 | 0.67 | +0.014 | 1.00 |
| C_calm2_below | 440 | +0.004 | -0.004 | 0.47 | 0.00 | 0.47 | 0.07 | 0.44 | 0.01 | 0.01 | -0.007 | 0.44 | 0.08 | 1.00 | -0.016 | 0.60 |
| M_e27 | 143 | +0.200 | +0.333 | 0.78 | 0.37 | 0.41 | 0.01 | 0.01 | 0.04 | 0.15 | -0.080 | 0.91 | 0.31 | 0.23 | +0.072 | 1.00 |
| M_e27_below | 58 | +0.003 | +0.164 | 0.57 | 0.31 | 0.26 | 0.02 | 0.00 | 0.03 | 0.38 | -0.193 | 0.65 | 0.48 | 0.45 | -0.043 | 0.50 |
| E18_all_perps | 702 | +0.023 | +0.112 | 0.60 | 0.21 | 0.39 | 0.02 | 0.05 | 0.04 | 0.28 | -0.148 | 0.59 | 0.45 | 0.57 | -0.042 | 0.67 |
| E18_weak_perps | 419 | +0.055 | +0.124 | 0.64 | 0.23 | 0.41 | 0.03 | 0.06 | 0.04 | 0.24 | -0.128 | 0.62 | 0.38 | 0.52 | -0.010 | 0.83 |
| E18_weak_perps_below | 283 | -0.024 | +0.061 | 0.55 | 0.18 | 0.37 | 0.01 | 0.05 | 0.05 | 0.34 | -0.179 | 0.51 | 0.55 | 0.67 | -0.167 | 0.33 |
| Y_young_all | 5550 | +0.011 | +0.032 | 0.58 | 0.00 | 0.58 | 0.10 | 0.28 | 0.03 | 0.01 | -0.009 | 0.60 | 0.10 | 0.72 | -0.015 | 0.67 |
| Y_young_weak | 4443 | +0.010 | +0.028 | 0.58 | 0.00 | 0.58 | 0.10 | 0.29 | 0.02 | 0.01 | -0.008 | 0.57 | 0.10 | 0.74 | -0.017 | 0.50 |
| Y_young_weak_below | 1107 | +0.014 | +0.050 | 0.62 | 0.00 | 0.62 | 0.12 | 0.21 | 0.03 | 0.01 | -0.014 | 0.74 | 0.10 | 0.64 | -0.026 | 0.67 |
| quiet_BASE_long_30d_tp30_stop25_nofilter | 20659 | +0.014 | -0.019 | 0.47 | 0.29 | 0.19 | 0.07 | 0.06 | 0.18 | 0.22 | -0.056 | 0.52 | 0.56 | 1.00 | -0.051 | 0.50 |
| quiet_BASE_short_7d_tp30_nofilter | 22310 | +0.007 | +0.005 | 0.52 | 0.09 | 0.43 | 0.04 | 0.41 | 0.02 | 0.01 | -0.007 | 0.32 | 0.08 | 0.92 | -0.022 | 0.50 |
| quiet_Q_long_30d_tp30_stop25_thr0.5 | 8070 | +0.001 | -0.035 | 0.45 | 0.27 | 0.18 | 0.07 | 0.05 | 0.18 | 0.24 | -0.062 | 0.51 | 0.59 | 1.00 | -0.050 | 0.50 |
| quiet_Q_long_30d_tp30_stop25_thr0.5_below | 12589 | +0.022 | -0.009 | 0.49 | 0.29 | 0.19 | 0.07 | 0.06 | 0.18 | 0.20 | -0.052 | 0.54 | 0.54 | 1.00 | -0.053 | 0.67 |
| quiet_Q_short_7d_tp30_thr0.9 | 167 | +0.059 | +0.083 | 0.66 | 0.15 | 0.51 | 0.02 | 0.30 | 0.02 | 0.01 | -0.006 | 0.46 | 0.06 | 0.47 | -0.054 | 0.67 |


## Thresholds and calibration

- **L_day5** (entries by year n/E_o: 2022:73/+0.12 2023:138/+0.14 2024:1059/+0.11 2025:581/+0.17 2026:32/+0.13): theta 0.995; 11.5% of candidates reach it, 88.5% within 0.1 below; Spearman +0.248; calibration E_o by decile: +0.002 +0.002 -0.004 +0.004 +0.006 +0.011 +0.013 +0.026 +0.059 +0.132
- **L_tp20day** (entries by year n/E_o: 2022:113/+0.07 2023:38/+0.13 2024:207/+0.12 2025:937/+0.17 2026:4/-0.15): theta 0.995; 11.0% of candidates reach it, 89.0% within 0.1 below; Spearman +0.258; calibration E_o by decile: -0.002 +0.011 +0.015 +0.009 +0.012 +0.023 +0.025 +0.016 +0.044 +0.160
- **L_day_transformer** (entries by year n/E_o: 2022:100/+0.10 2023:180/+0.12 2024:920/+0.10 2025:596/+0.15 2026:7/+0.18): theta 0.995; 10.7% of candidates reach it, 89.3% within 0.1 below; Spearman +0.250; calibration E_o by decile: -0.001 +0.001 +0.006 +0.009 +0.006 +0.011 +0.019 +0.027 +0.064 +0.123
- **L_day_tcn** (entries by year n/E_o: 2022:81/+0.11 2023:49/+0.18 2024:967/+0.10 2025:563/+0.16 2026:27/+0.14): theta 0.995; 10.1% of candidates reach it, 89.9% within 0.1 below; Spearman +0.223; calibration E_o by decile: +0.002 +0.003 +0.003 +0.003 +0.011 +0.014 +0.016 +0.020 +0.051 +0.124
- **L_day_gru** (entries by year n/E_o: 2022:82/+0.09 2023:166/+0.12 2024:1015/+0.08 2025:667/+0.16 2026:12/+0.20): theta 0.995; 11.1% of candidates reach it, 88.9% within 0.1 below; Spearman +0.197; calibration E_o by decile: +0.001 +0.002 +0.005 +0.013 +0.008 +0.015 +0.015 +0.012 +0.036 +0.119
- **L_day5_th099** (entries by year n/E_o: 2022:184/+0.05 2023:241/+0.11 2024:1732/+0.09 2025:627/+0.14 2026:150/+0.10): theta 0.990; 17.0% of candidates reach it, 83.0% within 0.1 below; Spearman +0.244; calibration E_o by decile: +0.002 +0.004 -0.004 +0.006 +0.007 +0.012 +0.016 +0.031 +0.059 +0.123
- **L_day5_th098** (entries by year n/E_o: 2022:514/-0.03 2023:633/+0.09 2024:2831/+0.05 2025:1044/+0.05 2026:326/+0.04): theta 0.980; 28.0% of candidates reach it, 72.0% within 0.1 below; Spearman +0.167; calibration E_o by decile: +0.001 +0.003 -0.000 +0.004 +0.007 +0.018 +0.017 +0.026 +0.024 +0.087
- **L_day5_lev2** (entries by year n/E_o: 2022:99/+0.11 2023:138/+0.28 2024:1059/+0.21 2025:926/+0.13 2026:32/+0.26): theta 0.995; 13.2% of candidates reach it, 86.8% within 0.1 below; Spearman +0.245; calibration E_o by decile: -0.008 -0.003 -0.017 +0.002 +0.012 +0.020 +0.021 +0.051 +0.139 +0.178
- **L_day5_lev3** (entries by year n/E_o: 2022:117/+0.12 2023:138/+0.41 2024:1070/+0.32 2025:993/+0.19 2026:32/+0.39): theta 0.995; 13.5% of candidates reach it, 86.5% within 0.1 below; Spearman +0.242; calibration E_o by decile: -0.013 -0.005 -0.029 +0.005 +0.015 +0.030 +0.045 +0.079 +0.237 +0.263
- **P_nostop** (entries by year n/E_o: 2022:35/+0.03 2023:27/+0.07 2024:45/+0.02 2025:119/+0.00 2026:69/+0.04): theta 0.904; 10.5% of candidates reach it, 2.6% within 0.1 below; Spearman +0.105; calibration E_o by decile: +0.007 -0.015 -0.024 -0.015 -0.004 -0.025 +0.018 -0.003 +0.013 +0.020
- **P_sl50** (entries by year n/E_o: 2022:39/-0.00 2023:29/+0.06 2024:46/+0.02 2025:128/-0.00 2026:75/+0.01): theta 0.904; 10.9% of candidates reach it, 3.6% within 0.1 below; Spearman +0.063; calibration E_o by decile: +0.000 -0.015 -0.014 -0.003 -0.022 -0.016 +0.017 -0.006 -0.009 +0.007
- **C_calm1** (entries by year n/E_o: 2022:12/+0.07 2023:12/+0.05 2024:23/+0.04 2025:76/+0.04 2026:38/+0.05): theta 0.500; 37.0% of candidates reach it, 4.4% within 0.1 below; Spearman +0.106; calibration E_o by decile: -0.004 +0.021 +0.001 +0.024 +0.027 -0.009 +0.051 +0.034 +0.104 -0.009
- **C_calm2** (entries by year n/E_o: 2022:19/+0.04 2023:13/+0.01 2024:24/+0.02 2025:98/+0.02 2026:44/+0.04): theta 0.500; 31.0% of candidates reach it, 3.8% within 0.1 below; Spearman +0.151; calibration E_o by decile: -0.014 -0.018 +0.010 -0.013 +0.007 +0.042 +0.022 +0.029 +0.055 -0.009
- **M_e27** (entries by year n/E_o: 2021:7/+0.36 2022:4/+0.38 2023:15/+0.18 2024:30/+0.07 2025:73/+0.25 2026:14/+0.12): theta 0.294; 71.1% of candidates reach it, 10.0% within 0.1 below; Spearman +0.177; calibration E_o by decile: +0.101 -0.108 +0.048 +0.185 +0.170 +0.124 +0.110 +0.186 +0.365 +0.253
- **E18_weak_perps** (entries by year n/E_o: 2021:39/+0.02 2022:17/+0.29 2023:58/+0.01 2024:83/+0.05 2025:136/+0.10 2026:86/-0.01): theta 0.405; 59.7% of candidates reach it, 10.0% within 0.1 below; Spearman +0.128; calibration E_o by decile: +0.016 -0.100 -0.014 +0.000 -0.001 +0.036 +0.024 +0.063 +0.100 +0.108
- **Y_young_weak** (entries by year n/E_o: 2021:292/-0.02 2022:254/+0.02 2023:424/-0.01 2024:771/-0.00 2025:1900/+0.02 2026:802/+0.01): theta 0.200; 80.1% of candidates reach it, 10.0% within 0.1 below; Spearman -0.055; calibration E_o by decile: +0.017 +0.012 +0.010 +0.025 +0.026 +0.015 +0.011 +0.012 -0.007 -0.012
- **quiet_BASE_long_30d_tp30_stop25_nofilter** (entries by year n/E_o: 2021:3186/+0.07 2022:2156/-0.05 2023:4475/+0.04 2024:4777/+0.02 2025:2783/-0.03 2026:3282/-0.00): theta 0.000; 100.0% of candidates reach it, 0.0% within 0.1 below; Spearman -0.062; calibration E_o by decile: +0.036 +0.028 +0.020 +0.020 +0.009 +0.018 +0.012 +0.008 +0.008 -0.023
- **quiet_BASE_short_7d_tp30_nofilter** (entries by year n/E_o: 2021:1180/-0.02 2022:4151/+0.03 2023:2963/-0.00 2024:3451/-0.01 2025:6051/+0.01 2026:4514/+0.01): theta 0.000; 100.0% of candidates reach it, 0.0% within 0.1 below; Spearman +0.048; calibration E_o by decile: -0.017 -0.005 +0.004 +0.013 +0.016 +0.011 +0.018 +0.010 +0.008 +0.015
- **quiet_Q_long_30d_tp30_stop25_thr0.5** (entries by year n/E_o: 2021:1052/+0.05 2022:1280/-0.05 2023:1610/+0.03 2024:1427/+0.02 2025:1253/-0.04 2026:1448/-0.01): theta 0.500; 39.1% of candidates reach it, 17.7% within 0.1 below; Spearman -0.062; calibration E_o by decile: +0.036 +0.028 +0.020 +0.020 +0.009 +0.018 +0.012 +0.008 +0.008 -0.023
- **quiet_Q_short_7d_tp30_thr0.9** (entries by year n/E_o: 2021:2/+0.01 2022:21/+0.08 2023:58/+0.11 2024:42/+0.08 2025:22/-0.01 2026:22/-0.05): theta 0.900; 100.0% of candidates reach it, 0.0% within 0.1 below; Spearman -0.011; calibration E_o by decile: -0.009 +0.137 +0.057 +0.101 +0.076 +0.062 +0.065 +0.040 +0.083 -0.019

## Leverage: clustering of the entries by day (net on the stake, entries only)

- L_day5: 25 entry days; 1 days average <= -0.30 per trade (worst day -0.40 over 2 trades); trades on the 11 busiest days 96% of all
- L_day5_lev2: 25 entry days; 2 days average <= -0.30 per trade (worst day -0.52 over 5 trades); trades on the 11 busiest days 95% of all
- L_day5_lev3: 25 entry days; 2 days average <= -0.30 per trade (worst day -0.62 over 7 trades); trades on the 11 busiest days 95% of all


## Reading each failed variant (baseline first)

Baselines: L_day5 (+0.129, 85% wins, 65% TP, no stops), P_nostop (+0.025), C_calm1 (+0.044), M_e27 (+0.200).
Calibration shapes are the E_o-by-decile lines above; "mode" is the failure mode with the largest drag.

1. **Trade-aligned CNN and other architectures (L_tp20day, L_day_transformer / tcn / gru).** Per trade
   they are as good as day5 or better (E_o +0.155 / +0.121 / +0.124 / +0.114; tp20day wins 82% with only 2%
   `never_went`) and all four calibrations are the same hockey stick (top decile +0.12..+0.16, rest under
   +0.06). So the per-trade scorecard does not show why tp20day lost in the book (+517 vs +1064 bp at q 0.995,
   -26 vs +247 bp at q 0.98); what differs is where the events fall: 937 of its 1,299 entries are in 2025
   (day5: 1,059 in 2024 and 581 in 2025) and 2026 has 4 trades at -0.15, so it fires on different and fewer
   flush days. Dominant failure: not a trade mode but event selection on ~25 days; the architectures add
   nothing over day5. Implication: the signal is capped by the number of flush days, so do not tune the
   model; widen the event set (more coins / data).
2. **Lower threshold (day5 >= 0.99 and >= 0.98).** E_o falls monotonically as the threshold drops: +0.129 ->
   +0.100 (0.99) -> +0.047 (0.98); TP share 65% -> 50% -> 28%, `never_went` 9% -> 13% -> 25%, giveback 0.15 ->
   0.25 -> 0.59, and 2022 turns negative (-0.03) at 0.98. The pool below the threshold is ~0 (+0.008..+0.012,
   54% wins, 39% `never_went`, giveback 0.88). Calibration: hockey stick (flat ~0 below the 0.98 CNN
   signal, +0.026 / +0.059 / +0.132 in the top three deciles). Lowering theta only mixes in the flat part;
   lever: more qualifying events, not a lower threshold.
3. **Pump shorts with a 50% stop (P_sl50) vs none (P_nostop).** The stop makes the dominant failure worse:
   stopped share 7% -> 19%, contrib_stopped -0.075 -> -0.100, stop_progress median 0.19 -> 0.38, and E_o
   +0.025 -> +0.012 (2022 and 2025 at -0.00). Most pumps that go +50% later revert (giveback 0.52 -> 0.60), so
   the stop sells retraces; `never_went` falls 12% -> 5% (the stop converts them into stops). Calibration is
   flat in both (decile E_o -0.025..+0.020, Spearman +0.105 / +0.063, and the Spearman drops with the stop).
   Implication: flat signal, so no exit rule can fix it; tail control must be position sizing or gates (the
   funding gate F), not a tighter stop.
4. **Crash shorts with BTC calm 2% (C_calm2) vs 1% (C_calm1).** E_o +0.044 -> +0.028 (2022-23: +0.04 / +0.01
   instead of +0.07 / +0.05); 61 more entries (198 vs 161) are filler: the added trades have the same ~27-28%
   `never_went` and giveback 0.67. Calibration is not monotonic (a bump in decile 9, the top decile ~0 in both:
   -0.009), Spearman +0.106 / +0.151. Dominant mode: `never_went` plus giveback, not stops (2% stopped).
   The widening admits crashes that are market-driven; the signal does not separate them. Implication: a
   better entry signal or shorter holds; the BTC gate is a real filter but the CNN score is not monotonic.
5. **Leverage on the capitulation longs (lev 2, 3 vs 1x).** Per entry, on the stake, the levered sleeves look
   better, not worse: E_o +0.129 -> +0.178 (2x) -> +0.261 (3x); the cost is the stopped mode: 0% -> 8% -> 12%
   (contrib_stopped -0.081 / -0.122), and the below-threshold pool gets -0.018 / -0.049 from liquidations
   (worst year -0.115 at 3x). Calibration stays a hockey stick (top decile +0.178 / +0.263). The failure is
   therefore not the per-trade expectancy but correlated ruin: 95% of the trades sit on the 11 busiest days,
   25 entry days in all, and the worst day averages -0.40 (1x), -0.52 (2x), -0.62 (3x) per trade, i.e. every
   position in a flush is liquidated together; a per-trade scorecard cannot show that, the book's drawdown can.
   Implication: not an exit or signal problem; leverage needs the day-level (portfolio) view and sizing.
6. **Listing shorts.** On all new perps with e18 `trades(72, 30, 0.5, 0.5)` (E18_all_perps) E_o is +0.023: 28%
   stopped (contrib -0.148, about 6x the whole edge), giveback 0.57, stop_progress median 0.45. Restricting to
   perps already at or below their first price (E18_weak_perps) gives +0.055 and still 24% stopped; the perps
   above it get -0.024 with 34% stopped. The calibration is monotone but shallow (deciles -0.100..+0.108,
   Spearman +0.128), against the spot-listing M_e27: +0.200, 15% stopped, `never_went` 1%, 37% TP. The perp-listing
   pool fails on the stop tail, not on missing moves (median tp_progress 0.59-0.62). Young-coin weekly shorts
   (Y_young_all / weak): +0.011 / +0.010, no TP and no stops to blame (1% liquidated): 28% `never_went`,
   median giveback 0.72-0.74, and the calibration is inverted (Spearman -0.055: the weakest vs listing price
   earn least, deciles +0.017..-0.012). Dominant failure: no signal (flat/inverted) on 5,550 trades; the decay
   is already harvested in the first month (M).
7. **Quiet breakouts.** Long (30d, TP 30% / stop 25%): entries E_o +0.001, below-threshold +0.022 (the filter
   anti-selects), unfiltered baseline +0.014; Spearman -0.062 and the calibration is inverted, decile E_o
   falling from +0.036 to -0.023. Failure mode: stopped 22-24% plus `went_against` 18% (stop_progress median
   0.56-0.59, giveback 1.00 median: the breakout is given back in full); TP is hit in 27-29%, which is not
   enough against a 25% stop. Years positive 0.50. Shorts: the unfiltered 7d baseline is +0.007 (41%
   `never_went`); the filtered Q_short has only 167 trades, +0.059, calibrated flat inside the entries (no
   below-threshold pool, so no verdict). Implication: inverted signal on the long side: find a different signal.

## Ranking of failure causes across all variants

1. **No information in the signal (flat or inverted calibration), 5 variants**: pump shorts (P, both stops), young-coin
   shorts (inverted), quiet breakout longs (inverted), perp listings on all coins, C with a wider BTC band.
   Cost: the whole edge; exits cannot fix it. Lever: change the signal.
2. **Edge only in the extreme tail of a hockey-stick signal (L variants, 5 rows)**: below the threshold the
   outcome is ~0 and 39% `never_went`; lowering theta from 0.995 to 0.98 loses 64% of E_o and the other CNNs
   are the same curve. Lever: more flush events (data / universe), not a lower threshold or a new model.
3. **Stop and liquidation tail on shorts (P_sl50, E18, M, quiet longs)**: stopped 19-28% (contrib -0.10 to -0.15),
   2-6x the edge. A tighter stop worsens it (7% -> 19%). Lever: gates (funding, weak vs listing price),
   sizing, not stops.
4. **Concentrated, correlated risk invisible per trade (leverage, L)**: 25 flush days hold 95% of trades;
   the 3x sleeve has the best per-trade E_o (+0.261) yet loses in the book because positions are liquidated
   together (worst day -0.62). Lever: day-level sizing / drawdown view, not the trade scorecard.
5. **Giveback and `never_went` on weak calm-market shorts (C, young coins)**: 27-29% `never_went`, giveback
   0.67-0.74, E_o +0.01..+0.04. Lever: shorter hold / better entry.

