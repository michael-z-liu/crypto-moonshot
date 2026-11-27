"""Offset for the i10 harness so a path model learns the realistic long outcome (TP +20%, exit
72h, spot paths, 0.4% cost) at every path step: offset = path target - outcome (percent, clipped
to +-30% as the path target). Written to cache_all/wf_i10/tp20_offset.npy (float32 [n, 32])."""

from __future__ import annotations

import numpy as np
from common_ls import CACHE_ALL, fwd2
from sim import Rule, exits, paths

PATH_STEPS = np.arange(3, 97, 3) - 1
n = fwd2().shape[0]
out = (
    np.lib.format.open_memmap(
        CACHE_ALL / "wf_i10" / "tp20_offset.npy",
        mode="w+",
        dtype=np.float32,
        shape=(n, len(PATH_STEPS)),
    )
    if (CACHE_ALL / "wf_i10").exists()
    else None
)
if out is None:
    (CACHE_ALL / "wf_i10").mkdir()
    out = np.lib.format.open_memmap(
        CACHE_ALL / "wf_i10" / "tp20_offset.npy",
        mode="w+",
        dtype=np.float32,
        shape=(n, len(PATH_STEPS)),
    )
rule = Rule(hold_min=4320, tp=0.2)
stats = []
for a in range(0, n, 200_000):
    rows = np.arange(a, min(n, a + 200_000))
    p = paths(rows)
    ret, _, _ = exits(p, np.ones(len(rows)), rule)
    outcome = np.clip(ret - 0.004, -0.3, 0.3) * 100
    fw = np.asarray(fwd2()[rows, 2, :], dtype=np.float32)
    path = np.clip(fw[:, PATH_STEPS] - 0.003, -0.3, 0.3) * 100
    path = np.nan_to_num(path, nan=0.0)
    out[rows] = path - outcome[:, None]
    stats.append(float(np.mean(outcome)))
out.flush()
print("rows", n, "mean outcome % by chunk", [round(s, 2) for s in stats])
