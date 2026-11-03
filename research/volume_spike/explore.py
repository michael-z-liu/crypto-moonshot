"""Ad-hoc helpers: evaluate a boolean event mask by split with a few exits."""

from __future__ import annotations

import numpy as np
import polars as pl
from evalx import load_events, summary, triple_barrier

EV, _FWD = load_events()
FWD = np.ascontiguousarray(_FWD)
SPLIT = EV["split"].to_numpy()


def ev(
    mask: pl.Series | np.ndarray,
    exits: list[tuple[float | None, float | None, int]],
    label: str = "",
    splits: tuple[str, ...] = ("train", "valid"),
) -> None:
    m = mask.to_numpy() if isinstance(mask, pl.Series) else mask
    m = np.asarray(m, dtype=bool) & ~np.isnan(EV["fwd_24h"].to_numpy())
    for s in splits:
        idx = np.flatnonzero(m & (s == SPLIT))
        parts = []
        for tp, sl, bars in exits:
            r = triple_barrier(FWD[idx], tp, sl, bars)
            x = summary(r)
            parts.append(f"{tp}/{sl}/{bars}:{x['mean_bp']:+5.0f}({x['win']:.2f})")
        f24 = EV["fwd_24h"].to_numpy()[idx]
        print(
            f"{label:28s} {s:5s} n={len(idx):6d} | "
            + " ".join(parts)
            + f" | 24h:{(f24.mean() - 0.003) * 1e4:+5.0f}"
        )
