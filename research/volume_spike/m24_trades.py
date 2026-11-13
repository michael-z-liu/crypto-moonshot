"""B4 part C: what a real bot would pay, from trade-by-trade data, for a sample of 1m entries.

The backtest buys at the 1m open after the trigger minute, which is by definition the first trade
of that minute. A bot first has to see the minute close and send an order. For a random sample of
trades, Binance's daily aggTrades files (data.binance.vision, checksum-verified) give the price of
the last trade at or before 1, 2, 5 and 10 seconds after the minute starts, compared with the open
the backtest used, plus how much USDT traded in those first seconds (can $50 fill there?).

Run: MOONSHOT_UNIVERSE=all uv run python research/volume_spike/m24_trades.py <sample.parquet> <name>
  (sample.parquet: columns symbol, entry_time (UTC datetime of the entry minute), entry (price))
"""

from __future__ import annotations

import hashlib
import io
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import polars as pl
from common import RESULTS
from m21_fillcheck import get

AGG = "https://data.binance.vision/data/spot/daily/aggTrades"
DELAYS = (1, 2, 5, 10)  # seconds after the entry minute starts


def day_trades(job: tuple[str, str]) -> tuple[str, str, pl.DataFrame | None]:
    symbol, day = job
    url = f"{AGG}/{symbol}/{symbol}-aggTrades-{day}.zip"
    raw = get(url)
    chk = get(url + ".CHECKSUM") if raw is not None else None
    if raw is None or chk is None or hashlib.sha256(raw).hexdigest() != chk.decode().split()[0]:
        return symbol, day, None
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        text = z.read(z.namelist()[0])
    df = pl.read_csv(io.BytesIO(text), has_header=False, infer_schema_length=0)
    if not df[0, 0].isdigit():
        df = df.slice(1)
    t = df[:, 5].cast(pl.Int64)
    unit = "us" if int(t.max() or 0) > 10**14 else "ms"
    return (
        symbol,
        day,
        pl.DataFrame(
            {
                "ts": pl.from_epoch(t, time_unit=unit).dt.replace_time_zone("UTC"),
                "price": df[:, 1].cast(pl.Float64),
                "qty": df[:, 2].cast(pl.Float64),
            }
        ).sort("ts"),
    )


def main(sample_path: str, name: str) -> None:
    s = pl.read_parquet(sample_path)
    jobs = sorted(
        {(r["symbol"], r["entry_time"].date().isoformat()) for r in s.iter_rows(named=True)}
    )
    print(len(s), "trades,", len(jobs), "symbol-days of aggTrades", flush=True)
    with ThreadPoolExecutor(8) as pool:
        days = {(sym, d): df for sym, d, df in pool.map(day_trades, jobs)}
    rows = []
    for r in s.iter_rows(named=True):
        df = days.get((r["symbol"], r["entry_time"].date().isoformat()))
        if df is None:
            rows.append({**r, "status": "no trade data"})
            continue
        t0 = r["entry_time"]
        w = df.filter((pl.col("ts") >= t0) & (pl.col("ts") < t0 + pl.duration(seconds=60)))
        if w.height == 0:
            rows.append({**r, "status": "no trades in the minute"})
            continue
        out = {**r, "status": "ok", "first_price": w["price"][0], "trades_first_min": w.height}
        for sec in DELAYS:
            upto = w.filter(pl.col("ts") < t0 + pl.duration(seconds=sec))
            out[f"px_{sec}s"] = upto["price"][-1] if upto.height else w["price"][0]
            out[f"usdt_{sec}s"] = float((upto["price"] * upto["qty"]).sum()) if upto.height else 0.0
        rows.append(out)
    res = pl.DataFrame(rows)
    res.write_csv(RESULTS / f"m24_trades_{name}.csv")
    ok = res.filter(pl.col("status") == "ok")
    print(res.group_by("status").len())
    q = lambda c: [round(float(np.nanquantile(ok[c].to_numpy(), p)), 3) for p in (0.1, 0.5, 0.9)]  # noqa: E731
    ok = ok.with_columns(
        first_vs_open=(pl.col("first_price") / pl.col("entry") - 1) * 100,
        **{f"slip_{s}s": (pl.col(f"px_{s}s") / pl.col("entry") - 1) * 100 for s in DELAYS},
    )
    print("first trade vs backtest open, % (10/50/90th):", q("first_vs_open"))
    for sec in DELAYS:
        print(
            f"price after {sec}s vs open, % (10/50/90th):",
            q(f"slip_{sec}s"),
            "mean",
            round(float(ok[f"slip_{sec}s"].mean()), 3),
            f"| USDT traded in the first {sec}s, median",
            round(float(ok[f"usdt_{sec}s"].median())),
        )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
