"""Would the take profit have filled? A 1-minute check of 5m take-profit hits (B3 step 4).

For each trade a book took that exited at its take profit, the 1m candles of the hit (Binance's
daily 1m files on data.binance.vision, checksum-verified, any symbol incl. delisted) decide:
- filled: a 1m candle closes at or above the target before the time limit (the price stayed
  there through a minute's close, not one tick)
- touched only: the high reached the target but no 1m close did; the trade then exits at the
  time limit instead (close after the limit, from the 5m data)
Also re-checks the entry: the 1m open of the entry minute against the 5m open used.

Run: MOONSHOT_UNIVERSE=all uv run python research/volume_spike/m21_fillcheck.py longshot
"""

from __future__ import annotations

import hashlib
import io
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import polars as pl
from common import BARS_PER_DAY, CACHE, RESULTS
from fetch_archive import UA

FILES = "https://data.binance.vision/data/spot/daily/klines"
M1 = CACHE / "m1"
COST = 0.003


def get(url: str) -> bytes | None:
    import urllib.error
    import urllib.request

    for k in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return bytes(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
        except Exception:
            pass
        __import__("time").sleep(2 * (k + 1))
    return None


def day_file(job: tuple[str, str]) -> str:
    """Download one symbol-day of 1m candles into cache/m1 (parquet); returns a status."""
    symbol, day = job
    out = M1 / symbol / f"{day}.parquet"
    if out.exists():
        return "exists"
    url = f"{FILES}/{symbol}/1m/{symbol}-1m-{day}.zip"
    raw = get(url)
    if raw is None:
        return "missing"
    chk = get(url + ".CHECKSUM")
    if chk is None or hashlib.sha256(raw).hexdigest() != chk.decode().split()[0]:
        return "checksum"
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        text = z.read(z.namelist()[0])
    df = pl.read_csv(io.BytesIO(text), has_header=False, infer_schema_length=0)
    if not df[0, 0].isdigit():
        df = df.slice(1)
    t = df[:, 0].cast(pl.Int64)
    unit = "us" if int(t.max() or 0) > 10**14 else "ms"
    df = pl.DataFrame(
        {
            "open_time": pl.from_epoch(t, time_unit=unit).dt.replace_time_zone("UTC"),
            **{
                c: df[:, j].cast(pl.Float64)
                for j, c in enumerate(("open", "high", "low", "close"), 1)
            },
        }
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    return "ok"


def check(trades: pl.DataFrame, tp: float, days: int, name: str) -> None:
    """trades: symbol, open_time (signal candle), entry, hit (5m offset of the TP hit), c_limit."""
    trades = trades.with_columns(
        entry_time=pl.col("open_time") + timedelta(minutes=5),
        target=pl.col("entry") * (1 + tp),
    ).with_columns(
        hit_time=pl.col("entry_time") + pl.duration(minutes=5 * pl.col("hit")),
        limit_time=pl.col("entry_time") + timedelta(days=days),
    )
    jobs = set()
    for r in trades.iter_rows(named=True):
        d = r["hit_time"].date()
        while d <= min(r["limit_time"].date(), r["hit_time"].date() + timedelta(days=days)):
            jobs.add((r["symbol"], d.isoformat()))
            d += timedelta(days=1)
        jobs.add((r["symbol"], r["entry_time"].date().isoformat()))
    jobs_l = sorted(jobs)
    print(len(trades), "take-profit trades,", len(jobs_l), "symbol-days of 1m data", flush=True)
    with ThreadPoolExecutor(12) as pool:
        status = list(pool.map(day_file, jobs_l))
    print({s: status.count(s) for s in set(status)}, flush=True)
    rows = []
    for (sym,), g in trades.group_by("symbol"):
        files = sorted((M1 / sym).glob("*.parquet")) if (M1 / sym).exists() else []
        if not files:
            for r in g.iter_rows(named=True):
                rows.append({**r, "status": "no 1m data"})
            continue
        m1 = pl.concat([pl.read_parquet(f) for f in files]).sort("open_time")
        for r in g.iter_rows(named=True):
            w = m1.filter(
                (pl.col("open_time") >= r["hit_time"]) & (pl.col("open_time") < r["limit_time"])
            )
            e = m1.filter(pl.col("open_time") == r["entry_time"])
            entry_1m = float(e["open"][0]) if e.height else None
            if w.height == 0:
                rows.append({**r, "status": "no 1m data", "entry_1m": entry_1m})
                continue
            closes = w.filter(pl.col("close") >= r["target"])
            touched = w.filter(pl.col("high") >= r["target"])
            first_close = closes["open_time"][0] if closes.height else None
            first_touch = touched["open_time"][0] if touched.height else None
            status = (
                "filled"
                if first_close is not None
                else ("touched only" if first_touch is not None else "not reached in 1m")
            )
            rows.append(
                {
                    **r,
                    "status": status,
                    "entry_1m": entry_1m,
                    "minutes_to_close": (first_close - first_touch).total_seconds() / 60
                    if first_close is not None and first_touch is not None
                    else None,
                }
            )
    out = pl.DataFrame(rows).with_columns(
        ret_5m=pl.lit(tp - COST),
        ret_1m=pl.when(pl.col("status") == "filled")
        .then(tp - COST)
        .otherwise(pl.col("c_limit") - COST),
        year=pl.col("open_time").dt.year(),
    )
    out.write_csv(RESULTS / f"m21_fillcheck_{name}.csv")
    with pl.Config(tbl_rows=40, float_precision=3):
        print(out.group_by("status").agg(pl.len(), pl.col("minutes_to_close").median()))
        print(
            out.group_by("year")
            .agg(
                pl.len(),
                filled=(pl.col("status") == "filled").mean(),
                lost_usd=(50 * (pl.col("ret_5m") - pl.col("ret_1m"))).sum(),
            )
            .sort("year")
        )
        e = out.filter(pl.col("entry_1m").is_not_null())
        print(
            "entry: 1m open vs 5m open, max abs diff", (e["entry_1m"] / e["entry"] - 1).abs().max()
        )


def longshot() -> None:
    """B2 pick: R2 capitulation, +20% take profit or 3 days, the $50 book."""
    from m18_longshot import RULES, book, exits, load

    data = load().with_columns(vol_mult_neg=-pl.col("vol_mult"))
    mask, prio = RULES["R2 capitulation"]
    d = data.filter(mask.fill_null(False))
    ev = pl.read_parquet(CACHE / "events.parquet", columns=["row", "entry"])
    d = d.join(ev, on="row", how="left")
    ret, xb = exits(d, 0.2, 3)
    taken = book(d, ret, xb, prio)
    hit = d["hit_20"].to_numpy()
    tp_mask = taken & (hit >= 0) & (hit < 3 * BARS_PER_DAY)
    trades = d.filter(pl.Series(tp_mask)).select(
        "symbol", "open_time", "entry", hit="hit_20", c_limit="c_3d"
    )
    check(trades, 0.2, 3, "longshot_r2_tp20_3d")


if __name__ == "__main__":
    {"longshot": longshot}[sys.argv[1]]()
