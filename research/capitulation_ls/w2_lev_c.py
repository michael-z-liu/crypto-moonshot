"""w2 part C: 2025-10-10 from 1m spot candles: taken trades, majors, wick depth vs close later, artifact check."""  # noqa: E501

import datetime as dt

import numpy as np
import polars as pl

pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_width_chars(220)
pl.Config.set_tbl_cols(20)
K = "/Users/mike/code/crypto-moonshot/data/klines/spot/interval=1m/symbol={}/2025-10.parquet"
U = dt.UTC


def anatomy(sym, t0):
    try:
        d = pl.read_parquet(K.format(sym)).sort("open_time")
    except Exception:
        return None
    w = d.filter((pl.col("open_time") >= t0) & (pl.col("open_time") < t0 + dt.timedelta(hours=3)))
    e = w.filter(pl.col("open_time") == t0)
    if w.height < 100 or e.height == 0:
        return None
    px = float(e["open"][0])
    i = int(w["low"].arg_min())
    lo = float(w["low"][i])
    tlo = w["open_time"][i]

    def cl(t):
        r = w.filter(pl.col("open_time") == t)
        return float(r["close"][0]) / px - 1 if r.height else np.nan

    half = px * (1 + 0.5 * (lo / px - 1))
    rec = w.filter((pl.col("open_time") > tlo) & (pl.col("close") >= half))
    return dict(
        sym=sym,
        low=lo / px - 1,
        tmin=(tlo - t0).total_seconds() / 60,
        c60=cl(t0 + dt.timedelta(minutes=60)),
        c_low60=cl(tlo + dt.timedelta(minutes=60)),
        c120=cl(t0 + dt.timedelta(minutes=120)),
        vol_ratio=float(w["volume"][i]) / float(w["volume"].median()),
        bar_range=float(w["high"][i] / w["low"][i] - 1),
        n_bars_below_half=int((w["low"] <= px * (1 + 0.5 * (lo / px - 1))).sum()),
        min_to_half=(rec["open_time"][0] - tlo).total_seconds() / 60 if rec.height else np.nan,
    )


t = pl.read_parquet("/tmp/w2_trades.parquet")
d = t.filter(pl.col("open_time").dt.date() == dt.date(2025, 10, 10)).sort("open_time")
rows = []
for s, ot, adv5, net in zip(d["symbol"], d["open_time"], d["adv"], d["net"], strict=False):
    r = anatomy(s, ot + dt.timedelta(minutes=5))
    if r:
        r["adv5m_sim"] = adv5
        r["net72h"] = net
        rows.append(r)
A = pl.DataFrame(rows)
print("== the 10 taken trades (entry = next 5m open, 1m detail)")
print(A)
MAJ = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "DOTUSDT",
    "LTCUSDT",
    "TRXUSDT",
    "BCHUSDT",
    "NEARUSDT",
    "MATICUSDT",
    "ATOMUSDT",
    "UNIUSDT",
    "ETCUSDT",
    "FILUSDT",
    "APTUSDT",
]
rows = []
for s in MAJ:
    r = anatomy(s, dt.datetime(2025, 10, 10, 20, 50, tzinfo=U))
    if r:
        rows.append(r)
M = pl.DataFrame(rows)
print("== majors, reference = open 20:50 UTC")
print(M)
print("== majors, reference = open 21:15 UTC")
print(
    pl.DataFrame(
        [r for r in (anatomy(s, dt.datetime(2025, 10, 10, 21, 15, tzinfo=U)) for s in MAJ) if r]
    )
)
# ATOM detail
a = pl.read_parquet(K.format("ATOMUSDT")).sort("open_time")
print(
    a.filter(
        (pl.col("open_time") >= dt.datetime(2025, 10, 10, 21, 10, tzinfo=U))
        & (pl.col("open_time") <= dt.datetime(2025, 10, 10, 21, 40, tzinfo=U))
    ).select("open_time", "open", "high", "low", "close", "volume", "trade_count")
)
# whole-market breadth: for every symbol with 1m on that day, wick depth from 21:00 open, and close 22:00; count >33/50%  # noqa: E501
import os  # noqa: E402

base = "/Users/mike/code/crypto-moonshot/data/klines/spot/interval=1m"
res = []
for s in sorted(os.listdir(base)):
    sym = s.split("=")[1]
    if not sym.endswith("USDT"):
        continue
    p = f"{base}/{s}/2025-10.parquet"
    if not os.path.exists(p):
        continue
    w = pl.read_parquet(p, columns=["open_time", "open", "low", "close", "quote_volume"]).filter(
        (pl.col("open_time") >= dt.datetime(2025, 10, 10, 20, 59, tzinfo=U))
        & (pl.col("open_time") < dt.datetime(2025, 10, 10, 23, 0, tzinfo=U))
    )
    if w.height < 100:
        continue
    o = w.filter(pl.col("open_time") == dt.datetime(2025, 10, 10, 21, 0, tzinfo=U))
    if o.height == 0:
        continue
    px = float(o["open"][0])
    lo = float(w["low"].min())
    c = w.filter(pl.col("open_time") == dt.datetime(2025, 10, 10, 22, 0, tzinfo=U))
    res.append(
        (
            sym,
            lo / px - 1,
            float(c["close"][0]) / px - 1 if c.height else np.nan,
            float(w["quote_volume"].sum()),
        )
    )
R = pl.DataFrame(res, schema=["sym", "low", "c22", "qv"], orient="row")
print("symbols with 1m on 10-10:", R.height)
for th in (-0.2, -0.33, -0.5, -0.7, -0.9):
    print(f"wick <= {th:.0%}: {(R['low'] <= th).sum()}  ({(R['low'] <= th).mean():.0%})")
print(
    "median low",
    R["low"].median(),
    "median close 22:00",
    R["c22"].median(),
    " median (c22 - low)",
    (R["c22"] - R["low"]).median(),
)
print(
    "wick<=-50% symbols: median 22:00 close vs 21:00 open",
    R.filter(pl.col("low") <= -0.5)["c22"].median(),
)
big = R.sort("qv", descending=True).head(40)
print("top-40 by 3h quote volume:")
print(big)
R.write_parquet("/tmp/w2_oct10_all.parquet")
