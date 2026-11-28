"""X12 fetch: Binance USD-M perp order-book depth (bookDepth) for the sleeve candidates.

data.binance.vision data/futures/um/daily/bookDepth/<SYM>/<SYM>-bookDepth-YYYY-MM-DD.zip holds
~30-second snapshots of the cumulative book notional within +-1..5% of mid (2026 adds +-0.2%).
For every 2023+ candidate of x8_sleeves.parquet this downloads the entry day and the previous
day of its perp (spot name, else the 1000-prefixed perp), keeps the last snapshot of each
5-minute bucket (with its own timestamp, so a backward as-of join has no look-ahead) and writes
wide parquet parts to OWN/bookdepth/parts/. Zips go to OWN/bookdepth/raw/ and are deleted after
parsing. Self-contained (no local imports) so it can run with `python -I` on downloaded data.

Run: uv run --group research python -I x12_fetch.py [--workers 6]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import io
import os
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import polars as pl

DATA_DIR = Path(os.environ.get("MOONSHOT_DATA_DIR", "/Users/mike/code/crypto-moonshot/data"))
OWN = DATA_DIR / "cache" / "capitulation_ls"
BD = OWN / "bookdepth"
RAW = BD / "raw"
PARTS = BD / "parts"
LOG = BD / "fetch_log.csv"
URL = "https://data.binance.vision/data/futures/um/daily/bookDepth/{s}/{s}-bookDepth-{d}.zip"
PREFIXES = ("1000000", "1000", "1M")
LEVELS = (1, 2, 3, 4, 5)
MAX_CSV = 60_000_000  # bytes; a daily file is ~1.6 MB, refuse anything absurd


def spot_name(perp: str) -> str:
    for p in PREFIXES:
        if perp.startswith(p) and len(perp) > len(p) + 4:
            return perp[len(p) :]
    return perp


def perp_map() -> tuple[set[str], dict[str, list[str]]]:
    f = pl.read_parquet(OWN / "futures" / "funding.parquet", columns=["symbol"])
    perps = set(f["symbol"].unique().to_list())
    alt: dict[str, list[str]] = {}
    for p in sorted(perps):
        alt.setdefault(spot_name(p), []).append(p)
    return perps, alt


def to_perp(sym: str, perps: set[str], alt: dict[str, list[str]]) -> str | None:
    if sym in perps:
        return sym
    c = [p for p in alt.get(sym, []) if p != sym]
    return c[0] if c else None


def jobs() -> list[tuple[str, str]]:
    s = pl.read_parquet(OWN / "explore" / "x8_sleeves.parquet").filter(pl.col("year") >= 2023)
    perps, alt = perp_map()
    out: set[tuple[str, str]] = set()
    for sym, t_in in s.select("symbol", "t_in").iter_rows():
        p = to_perp(sym, perps, alt)
        if p is None:
            continue
        d = dt.datetime.fromtimestamp(t_in, tz=dt.UTC).date()
        for k in (0, 1):
            out.add((p, (d - dt.timedelta(days=k)).isoformat()))
    return sorted(out)


def parse(blob: bytes, perp: str, day: str) -> pl.DataFrame:
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        names = z.namelist()
        want = f"{perp}-bookDepth-{day}.csv"
        if names != [want] or z.getinfo(want).file_size > MAX_CSV:
            raise ValueError(f"unexpected archive content {names[:3]}")
        raw = z.read(want)
    d = pl.read_csv(io.BytesIO(raw), schema_overrides={"percentage": pl.Float64})
    d = d.select(
        ts=pl.col("timestamp").str.to_datetime("%Y-%m-%d %H:%M:%S", time_zone="UTC").dt.epoch("s"),
        lvl=pl.col("percentage").round(1),
        notional=pl.col("notional").cast(pl.Float64),
    ).filter(pl.col("lvl").abs().is_in([float(x) for x in LEVELS]))
    d = d.with_columns(
        col=pl.when(pl.col("lvl") < 0)
        .then(pl.lit("b") + (-pl.col("lvl")).cast(pl.Int32).cast(pl.Utf8))
        .otherwise(pl.lit("a") + pl.col("lvl").cast(pl.Int32).cast(pl.Utf8))
    )
    w = d.pivot(on="col", index="ts", values="notional", aggregate_function="last").sort("ts")
    w = w.with_columns(bucket=pl.col("ts") // 300).group_by("bucket", maintain_order=True).last()
    cols = [f"{s}{k}" for s in "ba" for k in LEVELS]
    for c in cols:
        if c not in w.columns:
            w = w.with_columns(pl.lit(None, dtype=pl.Float64).alias(c))
    return w.select(pl.lit(perp).alias("perp"), "ts", *cols)


def fetch_one(perp: str, day: str) -> tuple[str, str, str, pl.DataFrame | None]:
    url = URL.format(s=perp, d=day)
    zp = RAW / f"{perp}-{day}.zip"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                blob = r.read()
            zp.write_bytes(blob)
            try:
                df = parse(blob, perp, day)
            finally:
                zp.unlink(missing_ok=True)
            return perp, day, "ok", df
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return perp, day, "404", None
            time.sleep(2 + 3 * attempt)
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(2 + 3 * attempt)
        except Exception as e:  # malformed file: record and move on
            return perp, day, f"bad:{type(e).__name__}", None
    return perp, day, "fail", None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    PARTS.mkdir(parents=True, exist_ok=True)
    done: set[tuple[str, str]] = set()
    if LOG.exists():
        lg = pl.read_csv(LOG)
        done = {(r[0], r[1]) for r in lg.filter(pl.col("status") != "fail").iter_rows()}
    else:
        LOG.write_text("perp,day,status,rows\n")
    todo = [j for j in jobs() if j not in done]
    if a.limit:
        todo = todo[: a.limit]
    print(f"jobs to fetch: {len(todo)} (already done {len(done)})", flush=True)
    buf: list[pl.DataFrame] = []
    pend: list[str] = []
    part = len(list(PARTS.glob("part_*.parquet")))  # leftovers of an interrupted run
    with cf.ThreadPoolExecutor(a.workers) as ex, LOG.open("a") as log:
        futs = [ex.submit(fetch_one, p, d) for p, d in todo]
        for i, f in enumerate(cf.as_completed(futs), 1):
            perp, day, st, df = f.result()
            if df is not None:
                buf.append(df)
            pend.append(f"{perp},{day},{st},{0 if df is None else df.height}\n")
            if len(buf) >= 100 or i == len(futs):
                if buf:
                    pl.concat(buf).write_parquet(PARTS / f"part_{part:04d}.parquet")
                    part += 1
                    buf = []
                log.write("".join(pend))  # logged only once the data is on disk
                log.flush()
                pend = []
            if i % 100 == 0:
                print(f"{i}/{len(futs)}", flush=True)
    parts = sorted(PARTS.glob("part_*.parquet"))
    out = BD / "depth5m.parquet"
    frames = [pl.read_parquet(p) for p in parts]
    if out.exists():
        frames.insert(0, pl.read_parquet(out))
    if frames:
        allp = pl.concat(frames).unique(["perp", "ts"]).sort("perp", "ts")
        allp.write_parquet(out)
        for p in parts:  # folded into depth5m.parquet
            p.unlink()
        print("depth5m rows", allp.height, flush=True)


if __name__ == "__main__":
    main()
