"""P1 (perp-only coins): parity of the p1_lib pipeline with the shared seq2 caches. Spot coins'
candles (`common.load`) go through `p1_lib` (spikes, inputs, fold scoring); the result is compared
with seq2_meta / seq2.npy rows and day5's walk-forward percentiles for the same spikes.

Run: uv run --group research python p1_parity.py
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import p1_lib
import polars as pl
from common_ls import meta, scores
from p1_lib import CACHE_ALL, Context, FoldModel, build, fold_name, spikes_and_coin_channels

COINS = ("SOLUSDT", "DOGEUSDT", "LUNCUSDT", "WIFUSDT", "BONKUSDT")


def main() -> None:
    from common import load

    m = meta().with_row_index("row")
    seq2 = np.load(CACHE_ALL / "seq2.npy", mmap_mode="r")
    day5 = scores("day5")
    ctx = Context()
    models = {y: FoldModel(fold_name(y)) for y in (2022, 2026)}
    for sym in COINS:
        raw = load(sym)
        got = build(sym, raw, ctx)
        if got is None:
            print(sym, "no spikes")
            continue
        mt, seq = got
        j = mt.with_row_index("k").join(m.select("row", "symbol", "open_time", "btc_ret_4h"),
                                        on=["symbol", "open_time"], how="left")  # fmt: skip
        hit = j.filter(pl.col("row").is_not_null())
        k, r = hit["k"].to_numpy(), hit["row"].to_numpy()
        a = np.asarray(seq[k], dtype=np.float32)
        b = np.asarray(seq2[np.sort(r)][np.argsort(np.argsort(r))], dtype=np.float32)
        d = np.abs(a - b).reshape(len(k), 13, -1).max(axis=(0, 2))
        print(
            f"{sym}: {mt.height} spikes, {hit.height} in seq2 ({j.height - hit.height} extra);"
            f" max abs diff per channel {np.round(d, 3).tolist()};"
            f" btc4h diff {np.nanmax(np.abs(hit['btc_ret_4h'].to_numpy() - mt['btc_ret_4h'].to_numpy()[k])):.2g}"  # noqa: E501
        )
        for y, fm in models.items():
            sel = (hit["open_time"].dt.year() == y).to_numpy()
            if not sel.any():
                continue
            _, pct = fm.score(seq[k[sel]], mt["qv_base"].to_numpy()[k[sel]])
            ref = day5[r[sel]]
            print(f"   fold {y}: {sel.sum()} rows, pct equal {np.mean(pct == ref):.3%},"
                  f" max diff {np.max(np.abs(pct - ref)):.3g}")  # fmt: skip
    _ = (spikes_and_coin_channels, p1_lib, datetime, UTC)


if __name__ == "__main__":
    main()
