"""B5: a 3-day path forecaster and what it says about the capitulation trades.

Model (`run`): the PR 8 path forecaster design (all 13 channels, a 24h window of 5m candles, coin
liquidity / age / hour) predicting the 10%, 50% and 90% quantiles of the net return (0.3% cost
taken off) at every hour 1..72 after entry (fwd72.npy, clipped to +-50%), pinball loss.
Walk-forward as M9 on the "all" universe: for test year Y it trains before Y-1 July, stops early
on Y-1 July-December (validation pinball loss) and forecasts Y; 2 seeds averaged.
Outputs cache_all/wf/path3d.npy (float16 [rows, 72, 3], NaN outside the test years).

`charts`: 20 trades of the B2 pick (R2 capitulation, $50, +20% or 3 days) taken by the book in
2022-26 and present in seq2; 4 a year (2 that hit +20%, 2 that did not, where possible), seed 3.
Writes results/universe_all/m25_charts.json: 24h before entry (15m bars), 72h after (hourly
bars), the forecast band, entry, target and exit.
`diag`: accuracy by horizon for those trades and for every liquid spike: band coverage, median
and mean forecast vs realised, rank correlation, and calibration by forecast decile.

Run: MOONSHOT_UNIVERSE=all uv run python research/volume_spike/m25_path3d.py run|charts|diag
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime

import numpy as np
import polars as pl
import torch
from common import BARS_PER_DAY, CACHE, RESULTS
from m12_cnn_wf import BATCH, BTC, COIN, DEVICE, ETH, MARKET, TRADES, WF, YEARS, Cfg, Data, make
from torch import nn

H = 72
Q = torch.tensor([0.1, 0.5, 0.9])
COST = 0.003
CFG = Cfg(channels=COIN + TRADES + BTC + ETH + MARKET, seq=288, epochs=6, target="path")
OUT = WF / "path3d.npy"


def target() -> np.ndarray:
    f = np.load(CACHE / "fwd72.npy", mmap_mode="r")
    return (np.clip(np.asarray(f[:, 2, :], dtype=np.float32) - COST, -0.5, 0.5) * 100).astype(
        np.float32
    )


def pinball(out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    o = out.view(len(out), H, 3)
    e = y[:, :, None] - o
    q = Q.to(out.device)[None, None, :]
    return torch.maximum(q * e, (q - 1) * e).mean()


def forecast(net: nn.Module, d: Data, rows: np.ndarray) -> np.ndarray:
    net.eval()
    out = []
    with torch.no_grad():
        for k in range(0, len(rows), 8192):
            x, s = d.batch(rows[k : k + 8192], CFG)
            out.append(net(x, s).float().cpu().numpy().reshape(-1, H, 3))
    return np.concatenate(out) if out else np.zeros((0, H, 3), dtype=np.float32)


def run() -> None:
    d = Data(mmap=True)
    y = target()
    yt = torch.from_numpy(y)
    t = d.meta["open_time"]
    years = d.meta["year"].to_numpy()
    res = np.lib.format.open_memmap(OUT, mode="w+", dtype=np.float16, shape=(len(y), H, 3))
    res[:] = np.nan
    logs = []
    for year in YEARS:
        cut = datetime(year - 1, 7, 1, tzinfo=UTC)
        tr = np.flatnonzero((t < cut).to_numpy())
        va = np.flatnonzero(((t >= cut) & (t < datetime(year, 1, 1, tzinfo=UTC))).to_numpy())
        va_s = np.sort(
            np.random.default_rng(0).choice(va, size=min(100_000, len(va)), replace=False)
        )
        te = np.flatnonzero(years == year)
        preds = []
        for seed in range(2):
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            net = make(CFG, sum(d.scal[k].shape[1] for k in CFG.scalars), H * 3).to(DEVICE)
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
            best, state, bad = np.inf, None, 0
            for epoch in range(CFG.epochs):
                net.train()
                t0 = time.time()
                perm = rng.permutation(tr)
                for k in range(0, len(perm), BATCH):
                    b = np.sort(perm[k : k + BATCH])
                    if len(b) < 32:
                        continue
                    x, s = d.batch(b, CFG)
                    loss = pinball(net(x, s), yt[b].to(DEVICE))
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                pv = forecast(net, d, va_s)
                e = y[va_s][:, :, None] - pv
                qn = np.array([0.1, 0.5, 0.9])[None, None, :]
                vloss = float(np.maximum(qn * e, (qn - 1) * e).mean())
                logs.append(
                    {
                        "year": year,
                        "seed": seed,
                        "epoch": epoch,
                        "valid_pinball": vloss,
                        "secs": time.time() - t0,
                    }
                )
                print(logs[-1], flush=True)
                if vloss < best:
                    best, bad = vloss, 0
                    state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= 2:
                        break
            assert state is not None
            net.load_state_dict(state)
            preds.append(forecast(net, d, te))
        p = np.mean(preds, 0)
        p.sort(axis=2)  # keep the quantiles ordered after averaging
        res[te] = p.astype(np.float16)
        res.flush()
    pl.DataFrame(logs).write_csv(WF / "path3d_log.csv")


# ----------------------------------------------------------------------------- trades
def pick_trades() -> pl.DataFrame:
    """The B2 pick's book trades in 2022-26, joined to seq2 rows (row in seq2 = `qi`)."""
    from m18_longshot import RULES, book, exits, load

    data = load().with_columns(vol_mult_neg=-pl.col("vol_mult"))
    mask, prio = RULES["R2 capitulation"]
    d = data.filter(mask.fill_null(False))
    ret, xb = exits(d, 0.2, 3)
    tk = book(d, ret, xb, prio)
    tr = d.with_columns(ret=pl.Series(ret), exit_bar=pl.Series(xb)).filter(pl.Series(tk))
    meta = pl.read_parquet(CACHE / "seq2_meta.parquet", columns=["event"]).with_row_index("qi")
    tr = tr.join(meta, left_on="row", right_on="event", how="inner")
    return tr.filter(pl.col("year") >= 2022).with_columns(
        won=pl.col("hit_20").is_between(0, 3 * BARS_PER_DAY - 1)
    )


def charts() -> None:
    from build_seq2 import STEP, T0, grid

    p = np.load(OUT, mmap_mode="r")
    f72 = np.load(CACHE / "fwd72.npy", mmap_mode="r")
    tr = pick_trades()
    rng = np.random.default_rng(3)
    picks = []
    for y in range(2022, 2027):
        for won in (True, False):
            g = tr.filter((pl.col("year") == y) & (pl.col("won") == won))
            if g.height:
                picks += g[rng.choice(g.height, size=min(2, g.height), replace=False)].to_dicts()
    out = []
    for r in sorted(picks, key=lambda r: r["open_time"]):
        off, a = grid(r["symbol"])
        gidx = int((r["open_time"] - T0) / STEP)
        e = gidx - off + 1  # entry candle on the symbol grid
        entry = a["open"][e]
        w = np.arange(e - 288, e)  # 24h of 5m candles before entry
        hi = a["high"][w].reshape(96, 3).max(1) / entry - 1
        lo = a["low"][w].reshape(96, 3).min(1) / entry - 1
        cl = a["close"][w].reshape(96, 3)[:, -1] / entry - 1
        fc = np.asarray(p[r["qi"]], dtype=np.float32) + COST * 100  # back to price, %
        post = np.asarray(f72[r["qi"]], dtype=np.float32) * 100
        hit = r["hit_20"]
        exit_h = (hit + 1) / 12 if r["won"] else 72.0
        out.append(
            {
                "symbol": r["symbol"],
                "time": r["open_time"].strftime("%Y-%m-%d %H:%M UTC"),
                "year": r["year"],
                "won": bool(r["won"]),
                "ret_pct": round(r["ret"] * 100, 2),
                "drop_4h_pct": round(r["ret_4h"] * 100, 1),
                "btc_vol_mult": round(r["btc_vol_mult"], 1),
                "exit_h": round(exit_h, 2),
                "pre": {
                    "h": [round(x, 3) for x in (np.arange(-96, 0) + 1) / 4],
                    "hi": np.round(hi * 100, 2).tolist(),
                    "lo": np.round(lo * 100, 2).tolist(),
                    "c": np.round(cl * 100, 2).tolist(),
                },
                "post": {
                    "h": list(range(1, H + 1)),
                    "hi": np.round(post[0], 2).tolist(),
                    "lo": np.round(post[1], 2).tolist(),
                    "c": np.round(post[2], 2).tolist(),
                },
                "fc": {
                    "q10": np.round(fc[:, 0], 2).tolist(),
                    "q50": np.round(fc[:, 1], 2).tolist(),
                    "q90": np.round(fc[:, 2], 2).tolist(),
                },
            }
        )
    (RESULTS / "m25_charts.json").write_text(json.dumps(out))
    print(len(out), "charts;", sum(o["won"] for o in out), "winners", flush=True)


def diag() -> None:
    p = np.load(OUT, mmap_mode="r")
    y = target()
    meta = pl.read_parquet(CACHE / "seq2_meta.parquet", columns=["open_time"]).with_columns(
        year=pl.col("open_time").dt.year()
    )
    test = np.flatnonzero(meta["year"].to_numpy() >= 2022)
    tr = pick_trades()
    sets = {
        "capitulation trades": np.sort(tr["qi"].to_numpy()),
        "all liquid spikes": test[:: max(1, len(test) // 300_000)],
    }
    rows, cal = [], []
    for name, idx in sets.items():
        f = np.asarray(p[idx], dtype=np.float32)
        r = y[idx]
        ok = ~np.isnan(f[:, 0, 0])
        f, r = f[ok], r[ok]
        for h in range(H):
            q10, q50, q90, real = f[:, h, 0], f[:, h, 1], f[:, h, 2], r[:, h]
            rows.append(
                {
                    "set": name,
                    "hour": h + 1,
                    "n": len(real),
                    "coverage_10_90": float(((real >= q10) & (real <= q90)).mean()),
                    "below_q10": float((real < q10).mean()),
                    "above_q90": float((real > q90).mean()),
                    "fc_median": float(np.median(q50)),
                    "real_median": float(np.median(real)),
                    "fc_mean": float(q50.mean()),
                    "real_mean": float(real.mean()),
                    "rank_corr": float(
                        pl.DataFrame({"a": q50, "b": real})
                        .select(pl.corr("a", "b", method="spearman"))
                        .item()
                    ),
                    "band_width": float(np.median(q90 - q10)),
                    "mae_model": float(np.abs(real - q50).mean()),
                    "mae_zero": float(np.abs(real).mean()),
                }
            )
        for h in (24, 72):
            q50, real = f[:, h - 1, 1], r[:, h - 1]
            dec = np.searchsorted(np.quantile(q50, np.linspace(0.1, 0.9, 9)), q50)
            for k in range(10):
                m = dec == k
                cal.append(
                    {
                        "set": name,
                        "hour": h,
                        "decile": k + 1,
                        "n": int(m.sum()),
                        "fc_mean": float(q50[m].mean()),
                        "real_mean": float(real[m].mean()),
                        "real_median": float(np.median(real[m])),
                        "fc_q90_mean": float(f[m, h - 1, 2].mean()),
                        "share_above_20": float((real[m] >= 20).mean()),
                    }
                )
    d, c = pl.DataFrame(rows), pl.DataFrame(cal)
    d.write_csv(RESULTS / "m25_diag.csv")
    c.write_csv(RESULTS / "m25_calibration.csv")
    with pl.Config(tbl_rows=60, float_precision=2):
        print(d.filter(pl.col("hour").is_in([1, 4, 12, 24, 48, 72])))
        print(c)


if __name__ == "__main__":
    {"run": run, "charts": charts, "diag": diag}[sys.argv[1]]()
