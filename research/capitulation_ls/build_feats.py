"""Feature table for big-move events (pumps r4 >= +15%, crashes r4 <= -10%), 2020-06 onward.

Per seq2 row: the meta scalars, features from the 24h sequence (seq2 channels), longer context
from hourly closes of the 5m store (7/30/90-day returns, distance to the 30/90-day high and low,
30-day volatility, 7-day volume trend), perp funding at entry, and targets (short/long returns at
4h/24h/72h from fwd72, max adverse/favourable moves over 24h).

Output: OWN/feats.parquet.  Run: uv run --group research python build_feats.py
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl
from common_ls import CACHE_ALL, OWN, fwd72, meta

sys.path.insert(0, str(CACHE_ALL.parent))
import os

os.environ.setdefault("MOONSHOT_UNIVERSE", "all")


def context(job: tuple[str, np.ndarray]) -> pl.DataFrame:
    from common import load  # volume_spike/common.py, universe "all"

    sym, times = job
    df = load(sym).select("open_time", "close", "high", "low", "quote_volume")
    h = df.group_by_dynamic("open_time", every="1h").agg(
        c=pl.col("close").last(), hi=pl.col("high").max(), lo=pl.col("low").min(), qv=pl.col("quote_volume").sum()  # noqa: E501
    ).sort("open_time")  # fmt: skip
    c = h["c"].to_numpy()
    hi, lo, qv = h["hi"].to_numpy(), h["lo"].to_numpy(), h["qv"].to_numpy()
    th = h["open_time"].dt.epoch("s").to_numpy()
    ev = np.asarray(times)
    # The hour containing the spike is still open at the spike: use the last *closed* hour.
    k = np.searchsorted(th, ev - 3600, side="right") - 1
    out: dict[str, np.ndarray] = {"symbol": np.full(len(ev), sym), "t": ev}

    def back(n: int) -> np.ndarray:
        j = k - n
        return np.where(j >= 0, c[np.maximum(j, 0)], np.nan)

    ck = np.where(k >= 0, c[np.maximum(k, 0)], np.nan)
    for lab, n in (("r7d", 168), ("r30d", 720), ("r90d", 2160)):
        out[lab] = ck / back(n) - 1
    s_hi = pl.Series(hi)
    s_lo = pl.Series(lo)
    for lab, n in (("30d", 720), ("90d", 2160)):
        rh = s_hi.rolling_max(n, min_samples=24).to_numpy()
        rl = s_lo.rolling_min(n, min_samples=24).to_numpy()
        out[f"dhi{lab}"] = np.where(k >= 0, ck / rh[np.maximum(k, 0)] - 1, np.nan)
        out[f"dlo{lab}"] = np.where(k >= 0, ck / rl[np.maximum(k, 0)] - 1, np.nan)
    lr = np.diff(np.log(np.maximum(c, 1e-12)), prepend=np.nan)
    vol30 = pl.Series(lr).rolling_std(720, min_samples=48).to_numpy()
    out["vol30d"] = np.where(k >= 0, vol30[np.maximum(k, 0)], np.nan)
    q7 = pl.Series(qv).rolling_sum(168, min_samples=24).to_numpy()
    q30 = pl.Series(qv).rolling_sum(720, min_samples=48).to_numpy()
    out["qv7_30"] = np.where(
        k >= 0, q7[np.maximum(k, 0)] / (q30[np.maximum(k, 0)] / 30 * 7), np.nan
    )
    out["n_prior_pumps30"] = np.zeros(len(ev))  # filled below from the event table
    return pl.DataFrame(out)


def main() -> None:
    m = meta().with_row_index("row")
    r4 = m["ret_4h"]
    ev = m.filter(
        ((r4 >= 0.15) | (r4 <= -0.10))
        & (pl.col("open_time") >= pl.datetime(2020, 6, 1, time_zone="UTC"))
    )
    rows = ev["row"].to_numpy()
    print(len(rows), "events", file=sys.stderr)
    seq = np.load(CACHE_ALL / "seq2.npy", mmap_mode="r")
    s = np.asarray(seq[rows], dtype=np.float32)
    f = np.asarray(fwd72()[rows], dtype=np.float64)
    vol = s[:, 4]
    feats = {
        "red": (s[:, 0, -1] > 0).astype(np.float32),
        "off_high4h": s[:, 1, -48:].max(1), "off_low4h": -s[:, 2, -48:].min(1),
        "off_high24h": s[:, 1].max(1), "off_low24h": -s[:, 2].min(1),
        "r5m": -s[:, 0, -1], "r15m": -s[:, 3, -4], "r1h": -s[:, 3, -13], "r2h": -s[:, 3, -25], "r8h": -s[:, 3, -97],  # noqa: E501
        "wick_up": s[:, 1, -1] - np.maximum(s[:, 0, -1], 0), "wick_dn": np.minimum(s[:, 0, -1], 0) - s[:, 2, -1],  # noqa: E501
        "taker": s[:, 5, -1], "taker1h": s[:, 5, -12:].mean(1), "taker4h": s[:, 5, -48:].mean(1),
        "trades_x": s[:, 6, -1], "vol1h": vol[:, -12:].mean(1), "vol4h": vol[:, -48:].mean(1),
        "vol_prev20h": vol[:, :-48].mean(1), "n_hot_bars4h": (vol[:, -48:] > np.log1p(5)).sum(1),
        "btc_r1h": -s[:, 7, -13], "btc_r24h": -s[:, 7, 0], "btc_vol1h": s[:, 8, -12:].mean(1),
        "eth_r4h": -s[:, 9, -49], "eth_r24h": -s[:, 9, 0],
        "mkt_r4h": -s[:, 11, -49], "mkt_r24h": -s[:, 11, 0], "spike_share": s[:, 12, -1], "spike_share1h": s[:, 12, -12:].mean(1),  # noqa: E501
    }  # fmt: skip
    out = ev.with_columns(**{k: pl.Series(v.astype(np.float32)) for k, v in feats.items()})
    for lab, hh in (("4h", 3), ("24h", 23), ("72h", 71)):
        out = out.with_columns(
            pl.Series(f"y{lab}", f[:, 2, hh].astype(np.float32))
        )  # long price return
    out = out.with_columns(
        hi24=pl.Series(f[:, 0, :24].max(1).astype(np.float32)), lo24=pl.Series(f[:, 1, :24].min(1).astype(np.float32)),  # noqa: E501
        hi72=pl.Series(f[:, 0, :72].max(1).astype(np.float32)), lo72=pl.Series(f[:, 1, :72].min(1).astype(np.float32)),  # noqa: E501
        t=pl.col("open_time").dt.epoch("s"),
    )  # fmt: skip
    jobs = [(str(sym), g["t"].to_numpy()) for (sym,), g in out.group_by(["symbol"])]
    with ProcessPoolExecutor(8) as ex:
        ctx = pl.concat(list(ex.map(context, jobs)))
    out = out.join(ctx.unique(["symbol", "t"]), on=["symbol", "t"], how="left")
    # Prior big pumps of the same coin in the last 30 days (one per day).
    days = (
        out.select("symbol", d=pl.col("t") // 86400, p=pl.col("ret_4h") >= 0.3)
        .filter("p")
        .unique(["symbol", "d"])
    )
    cnt = []
    dd = {k[0]: np.sort(g["d"].to_numpy()) for k, g in days.group_by(["symbol"])}
    for sym, t in zip(out["symbol"].to_numpy(), out["t"].to_numpy(), strict=True):
        a = dd.get(sym)
        d = t // 86400
        cnt.append(0 if a is None else int(np.searchsorted(a, d) - np.searchsorted(a, d - 30)))
    out = out.with_columns(n_prior_pumps30=pl.Series(cnt, dtype=pl.Float32))
    # Funding at entry (last settled rate, mean of the last 24h) and perp availability.
    from futures_ls import funding, has_perp

    fu = (
        funding()
        .select(symbol=pl.col("spot"), ft=pl.col("time").dt.epoch("s"), rate="rate")
        .sort("ft")
    )
    fu = fu.with_columns(rate24=pl.col("rate").rolling_mean(3, min_samples=1).over("symbol"))
    out = out.sort("t").join_asof(
        fu.sort("ft"), left_on="t", right_on="ft", by="symbol", strategy="backward"
    )
    out = out.with_columns(fund_age_h=(pl.col("t") - pl.col("ft")) / 3600).drop("ft")
    out = out.with_columns(perp=pl.Series(has_perp(out["symbol"].to_numpy(), out["t"].to_numpy())))
    out.write_parquet(OWN / "feats.parquet")
    print(out.shape, file=sys.stderr)


if __name__ == "__main__":
    main()
