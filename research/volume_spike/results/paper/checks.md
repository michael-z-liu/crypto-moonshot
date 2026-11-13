# Paper test checks (P1)

Written by `research/volume_spike/paper.py`.

## Model reproduction (2026 fold)

- retrain vs `path_all.npz` `pct` on 144,796 2026 rows: 144,796 equal (100.0000%), max abs diff 0
- retrain vs `path_all.npz` `raw` on 144,796 2026 rows: 144,796 equal (100.0000%), max abs diff 0
- `path` (all 96 quantiles): max abs diff 0
- saved model reloaded, `raw` vs the retrain: 144,796 equal (100.0000%), max abs diff 0
- saved model reloaded, `pct` vs the retrain: 144,796 equal (100.0000%), max abs diff 0

## Pipeline parity (2026-10-15 .. 2026-11-01)

Local store: candles from 2026-10-01; build 3s.

- shared seq2 rows in the window: 6,836; local liquid spikes: 6,852; matched by (symbol, open_time): 6,836 (100.00% of shared, 99.77% of local)
- qv_base on matched rows: max abs diff 0
- inputs, all 288 candles: 99.7161% of values equal, max abs diff 0.132; rows fully equal 97.38%
- inputs, the 96 candles the model sees: max abs diff 0.132; rows fully equal 97.38%
- max abs diff by channel (96 model candles): 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0.00467, 12: 0.132
- rows fully equal (model candles) by day: 10-15 713/713, 10-16 229/229, 10-17 409/409, 10-18 359/359, 10-19 288/288, 10-20 568/568, 10-21 373/373, 10-22 276/276, 10-23 384/384, 10-24 420/420, 10-25 214/214, 10-26 240/240, 10-27 254/254, 10-28 277/277, 10-29 958/958, 10-30 509/509, 10-31 186/186, 11-01 0/179
- scores, saved model on local inputs vs `path_all.npz`: raw max abs diff 0.32, pct equal 97.38%, pct max abs diff 0.346
  - on rows with equal inputs (6,657): raw max abs diff 0, pct equal 100.00%
  - same model on the shared inputs vs `path_all.npz`: raw max abs diff 0, pct equal 100.00%
  - signals at pct >= 0.98: shared 34, local 34, both 34
- rows only in shared: 0
- rows only local: 16
  - after the coin's last shared candle (outside-pool coins end 2026-10-31 23:55 in the shared store): 0
  - within 49 candles of it (build_events drops spikes without 48 forward candles): 16
  - other (a gap in the next 48 candles, which build_events also drops): 0
- market channels (11, 12) count coins: shared 392 on 10-31 and 152 on 11-01, local 392 and 392. The shared store's outside-pool coins end on 2026-10-31 23:55, so on 11-01 the shared market channels cover the pool only; that is the only input difference, and the local one is the complete universe the paper test uses.

## Edge effects (store from 2026-10-01 vs from 2026-10-22)

- spikes from 2026-11-02 00:00 to the data end: 1726 vs 1726, same rows: True
- inputs equal: True (max abs diff 0); qv_base max abs diff 0.0
- 5s
