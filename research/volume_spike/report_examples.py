"""Example trades for the casebook: 10 winners and 10 losers of the path forecaster.

    MOONSHOT_UNIVERSE=all uv run python research/volume_spike/report_examples.py  # after report_wf

The trades are ones the 5-slot portfolio actually took with the default rule (top 2% of scores,
hold 1h), on every USDT pair. Two winners and two losers per test year are drawn at random
(seed 7), so they show typical trades rather than the extremes. Also writes the forecast and the
outcome of every trade taken, for the overview charts.
"""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

import numpy as np
import polars as pl
from common import UNIVERSE, archive_groups, load
from m12_cnn_wf import PATH_STEPS, YEARS
from m13_policy import COST, Ctx, Rule
from report_data import OUT, build_html

VARIANT = "path_all5" if UNIVERSE == "pool" else "path_all"
BEFORE, AFTER = 48, 96  # candles shown before the spike and after it
PER_YEAR = 2
HOLD = 12
H1 = 3  # PATH_STEPS index of the 1h close


def main() -> None:
    ctx = Ctx(VARIANT)
    assert ctx.path is not None
    yrs = ctx.evaluate(Rule())
    groups = archive_groups()
    group = {s: "delisted" for s in groups["A_delisted"]}
    group |= {s: "listed, outside the top 200" for s in groups["B_listed_outside_pool"]}
    taken = [
        (int(y["year"]), i, r[2]) for y in yrs for i, r in zip(y["_taken"], y["_rows"], strict=True)
    ]  # type: ignore[call-overload]

    m = ctx.meta
    idx = np.array([t[1] for t in taken])
    ret = np.array([t[2] for t in taken])
    p = ctx.path[idx]
    fwd = np.asarray(ctx.fwd[idx], dtype=np.float32)
    trades = pl.DataFrame(
        {
            "year": [t[0] for t in taken],
            "symbol": m["symbol"].gather(idx),
            "ret_bp": ret * 1e4,
            "p10_bp": p[:, H1, 0] * 100,
            "p50_bp": p[:, H1, 1] * 100,
            "p90_bp": p[:, H1, 2] * 100,
            "peak8h_bp": p[:, :, 1].max(1) * 100,
            "best_bp": (fwd[:, 0, :HOLD].max(1) - COST) * 1e4,  # best high during the hold, net
            "worst_bp": (fwd[:, 1, :HOLD].min(1) - COST) * 1e4,
            "btc_ret_4h": m["btc_ret_4h"].gather(idx),
            "btc_vol_mult": m["btc_vol_mult"].gather(idx),
            "coin_ret_24h": m["ret_24h"].gather(idx),
            "vol_mult": m["vol_mult"].gather(idx),
        }
    ).with_columns(
        win=pl.col("ret_bp") > 0,
        group=pl.col("symbol").replace_strict(group, default="top 200"),
    )

    rng = np.random.default_rng(7)
    picks = []
    for year in YEARS:
        for win in (True, False):
            k = [j for j, t in enumerate(taken) if t[0] == year and (t[2] > 0) == win]
            picks += [(win, taken[j]) for j in sorted(rng.choice(k, PER_YEAR, replace=False))]
    cache: dict[str, pl.DataFrame] = {}
    examples = [example(ctx, cache, group, win, i, r) for win, (_, i, r) in picks]

    data = json.loads(OUT.read_text())
    data["trade_examples"] = {
        "variant": VARIANT, "universe": UNIVERSE, "hold": HOLD, "cost": COST,
        "trades": rows(trades), "examples": examples,
    }  # fmt: skip
    OUT.write_text(json.dumps(data, default=str, separators=(",", ":")))
    print(trades.group_by("win").agg(pl.len(), pl.col("ret_bp").mean(), pl.col("p50_bp").mean()))
    build_html()


def rows(df: pl.DataFrame) -> list[dict[str, Any]]:
    return [
        {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}
        for r in df.iter_rows(named=True)
    ]


def example(
    ctx: Ctx, cache: dict[str, pl.DataFrame], group: dict[str, str], win: bool, i: int, r: float
) -> dict[str, Any]:
    """Candles around one trade, with the forecast the model made at the entry."""
    assert ctx.path is not None
    meta = ctx.meta.row(i, named=True)
    sym, spike = meta["symbol"], meta["open_time"]
    if sym not in cache:
        cache[sym] = load(sym)
    w = cache[sym].filter(
        (pl.col("open_time") >= spike - timedelta(minutes=5 * BEFORE))
        & (pl.col("open_time") <= spike + timedelta(minutes=5 * AFTER))
    )
    entry = w.filter(pl.col("open_time") > spike)["open"][0]
    fc = ctx.path[i].astype(np.float64) + COST * 100  # gross % vs the entry price
    return {
        "win": win, "symbol": sym, "group": group.get(sym, "top 200"),
        "time": spike.isoformat(), "ret_bp": r * 1e4, "hold": HOLD, "entry": entry,
        "score": float(ctx.pct[i]), "btc_ret_4h": meta["btc_ret_4h"],
        "btc_vol_mult": meta["btc_vol_mult"], "coin_ret_4h": meta["ret_4h"],
        "coin_ret_24h": meta["ret_24h"], "vol_mult": meta["vol_mult"],
        "steps": (PATH_STEPS + 1).tolist(),
        **{q: fc[:, j].round(3).tolist() for j, q in enumerate(("p10", "p50", "p90"))},
        "t": [x.isoformat() for x in w["open_time"]],
        **{c: w[c].to_list() for c in ("open", "high", "low", "close", "volume")},
        "taker_buy": w["taker_buy_base_volume"].to_list(),
    }  # fmt: skip


if __name__ == "__main__":
    main()
