# Diagnostic scorecards (diag.py)

Outcome o = net return on the stake, clipped to [-1, +1]. Shares are of trades; `contrib_*` is the
mode's contribution to E_o (they sum to E_o). Every candidate trade, no slot limit, 2022-26.

| strategy | n | E_o | median_o | win | share_tp_win | share_time_win | share_near_miss | share_never_went | share_went_against | share_stopped | contrib_stopped | tp_progress_med | stop_progress_med | giveback_med | worst_year_E_o | years_positive |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L | 1883 | +0.129 | +0.196 | 0.85 | 0.65 | 0.20 | 0.05 | 0.09 | 0.01 | 0.00 | -0.001 | 1.00 | 0.06 | 0.15 | +0.107 | 1.00 |
| P | 295 | +0.025 | +0.099 | 0.65 | 0.00 | 0.65 | 0.12 | 0.12 | 0.03 | 0.07 | -0.075 | 1.00 | 0.19 | 0.52 | +0.001 | 1.00 |
| F | 143 | +0.095 | +0.142 | 0.73 | 0.00 | 0.73 | 0.10 | 0.10 | 0.01 | 0.05 | -0.049 | 1.00 | 0.18 | 0.34 | +0.017 | 1.00 |
| C | 161 | +0.044 | +0.036 | 0.60 | 0.00 | 0.60 | 0.11 | 0.27 | 0.01 | 0.02 | -0.019 | 0.66 | 0.09 | 0.67 | +0.036 | 1.00 |
| M | 143 | +0.200 | +0.333 | 0.78 | 0.37 | 0.41 | 0.01 | 0.01 | 0.04 | 0.15 | -0.080 | 0.91 | 0.31 | 0.23 | +0.072 | 1.00 |


## Thresholds

- **L**: threshold 0.995; 11.5% of candidates reach it, 88.5% fall within 0.1 below; Spearman(signal, outcome) +0.248
- **P**: threshold 0.904; 10.5% of candidates reach it, 2.6% fall within 0.1 below; Spearman(signal, outcome) +0.105
- **F**: threshold 0.904; 7.1% of candidates reach it, 2.4% fall within 0.1 below; Spearman(signal, outcome) +0.106
- **C**: threshold 0.500; 37.0% of candidates reach it, 4.4% fall within 0.1 below; Spearman(signal, outcome) +0.106
- **M**: threshold 0.294; 71.1% of candidates reach it, 10.0% fall within 0.1 below; Spearman(signal, outcome) +0.177
