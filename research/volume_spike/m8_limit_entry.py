"""M8: limit-order entry below the spike bar's close instead of a market buy at the next open.

A buy limit at close_t * (1 - offset) rests for `wait` bars. If bar t+1 opens at or below it, it
fills at that open; otherwise it fills at the limit once a bar's low reaches it. Unfilled orders
are cancelled (no trade). The position is closed at the close of the `hold`-th bar after the
fill bar. Cost: 0.1% maker fee in, 0.1% fee + 0.05% slippage out.
"""

from __future__ import annotations

import itertools

import numpy as np
import polars as pl
from common import HORIZON, RESULTS
from evalx import load_events
from m3_rules import non_overlapping

COST_LIMIT = 0.0025
CANDS = {
    "F1": (pl.col("ret_4h") <= -0.08)
    & (pl.col("qv_base") >= 5_000)
    & (pl.col("btc_vol_mult") >= 6),
    "F2": (pl.col("ret_4h") <= -0.10)
    & (pl.col("qv_base") >= 20_000)
    & (pl.col("btc_vol_mult") >= 6),
    "F3": (pl.col("ret_4h") <= -0.15) & (pl.col("qv_base") >= 5_000),
}


def limit_returns(
    fwd: np.ndarray, close_over_entry: np.ndarray, offset: float, wait: int, hold: int
) -> np.ndarray:
    lvl = close_over_entry * (1 - offset) - 1  # limit relative to the next open
    lo = fwd[:, 1, :wait]
    hit = lo <= lvl[:, None]
    filled = hit.any(1) | (lvl >= 0)
    k = np.where(lvl >= 0, 0, hit.argmax(1))
    fill = np.where(lvl >= 0, 0.0, lvl)
    exit_bar = np.minimum(k + hold, HORIZON - 1)
    out = (1 + fwd[np.arange(len(k)), 2, exit_bar]) / (1 + fill) - 1 - COST_LIMIT
    return np.where(filled, out, np.nan)


def main() -> None:
    ev, fwd = load_events()
    rows = []
    for name, cond in CANDS.items():
        idx = np.flatnonzero(
            ev.select((pl.col("is_spike") & cond).fill_null(False))["is_spike"].to_numpy()
        )
        sub = ev[idx]
        f = np.asarray(fwd[idx])
        coe = (sub["close"] / sub["entry"]).to_numpy()
        for offset, wait, hold in itertools.product([0.0, 0.01, 0.02, 0.03, 0.05], [1, 3, 6], [12]):
            r = limit_returns(f, coe, offset, wait, hold)
            d = (
                sub.select("symbol", "open_time", "split")
                .with_columns(r=pl.Series(r))
                .filter(pl.col("r").is_not_nan())
            )
            keep = non_overlapping(d["symbol"].to_numpy(), d["open_time"].to_numpy(), hold + wait)
            d = d.filter(pl.Series(keep))
            for split in ("train", "valid", "test"):
                s = d.filter(pl.col("split") == split)
                days = s.group_by(pl.col("open_time").dt.date()).agg(pl.col("r").mean())["r"]
                rows.append(
                    {
                        "cand": name,
                        "offset": offset,
                        "wait": wait,
                        "split": split,
                        "fill_rate": s.height / max(1, (sub["split"] == split).sum()),
                        "trades": s.height,
                        "mean_bp": s["r"].mean() * 1e4,
                        "day_bp": days.mean() * 1e4,
                        "win_days": (days > 0).mean(),
                    }
                )
    df = pl.DataFrame(rows)
    df.write_csv(RESULTS / "m8_limit_entry.csv")
    wide = df.pivot(
        on="split", index=["cand", "offset", "wait"], values=["fill_rate", "mean_bp", "day_bp"]
    )
    with pl.Config(tbl_rows=60, tbl_cols=14, tbl_width_chars=230, float_precision=2):
        print(wide)


if __name__ == "__main__":
    main()
