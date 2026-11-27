"""M1: USD-M perpetual data from data.binance.vision (funding rates, 1h klines; metrics later).

Lists every USDT-margined symbol in the archive (delisted ones included), downloads the monthly
zips into memory and writes one parquet per dataset to OWN/futures/:

- funding.parquet: symbol, time (UTC), rate (per funding interval)
- perp_1h.parquet: symbol, open_time, open, high, low, close, volume, quote_volume

Run: uv run --group research python fetch_futures.py [funding|klines1h]
"""

from __future__ import annotations

import io
import re
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from xml.etree import ElementTree

import polars as pl
from common_ls import OWN

S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
DL = "https://data.binance.vision/"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
OUT = OWN / "futures"
OUT.mkdir(parents=True, exist_ok=True)


def list_prefix(prefix: str, dirs: bool) -> list[str]:
    out, marker = [], ""
    while True:
        url = f"{S3}?delimiter=/&prefix={prefix}&marker={marker}"
        root = ElementTree.fromstring(urllib.request.urlopen(url, timeout=60).read())
        if dirs:
            got = [e.find(f"{NS}Prefix").text for e in root.findall(f"{NS}CommonPrefixes")]  # type: ignore[union-attr]
        else:
            got = [e.find(f"{NS}Key").text for e in root.findall(f"{NS}Contents")]  # type: ignore[union-attr]
        out += [g for g in got if g]
        nxt = root.find(f"{NS}NextMarker")
        trunc = root.find(f"{NS}IsTruncated")
        if trunc is None or trunc.text != "true":
            return out
        marker = nxt.text if nxt is not None and nxt.text else out[-1]


def fetch_csv(key: str) -> bytes | None:
    for _ in range(4):
        try:
            data = urllib.request.urlopen(DL + key, timeout=60).read()
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                return z.read(z.namelist()[0])
        except Exception:  # noqa: BLE001
            continue
    print("failed", key, file=sys.stderr)
    return None


def symbols(kind: str) -> list[str]:
    ps = list_prefix(f"data/futures/um/monthly/{kind}/", dirs=True)
    names = (p.rstrip("/").split("/")[-1] for p in ps)
    return sorted(n for n in names if n.endswith("USDT") and n.isascii() and n.isalnum())


def funding() -> None:
    syms = symbols("fundingRate")
    print(len(syms), "symbols", file=sys.stderr)
    with ThreadPoolExecutor(16) as ex:
        keys = [k for ks in ex.map(lambda s: list_prefix(f"data/futures/um/monthly/fundingRate/{s}/", False), syms) for k in ks if k.endswith(".zip")]
    print(len(keys), "files", file=sys.stderr)

    def one(key: str) -> pl.DataFrame | None:
        raw = fetch_csv(key)
        if raw is None:
            return None
        sym = key.split("/")[5]
        lines = [ln for ln in raw.decode().splitlines() if ln and ln[0].isdigit()]
        if not lines:
            return None
        rows = [ln.split(",") for ln in lines]
        return pl.DataFrame({"symbol": [sym] * len(rows), "ms": [int(r[0]) for r in rows],
                             "rate": [float(r[2]) for r in rows]})  # fmt: skip

    with ThreadPoolExecutor(32) as ex:
        parts = [p for p in ex.map(one, keys) if p is not None]
    df = pl.concat(parts).with_columns(time=pl.from_epoch("ms", "ms").dt.replace_time_zone("UTC")).drop("ms")
    df.unique(["symbol", "time"]).sort("symbol", "time").write_parquet(OUT / "funding.parquet")
    print(df.shape, file=sys.stderr)


def klines1h() -> None:
    syms = symbols("klines")
    print(len(syms), "symbols", file=sys.stderr)
    with ThreadPoolExecutor(16) as ex:
        keys = [k for ks in ex.map(lambda s: list_prefix(f"data/futures/um/monthly/klines/{s}/1h/", False), syms) for k in ks if k.endswith(".zip")]
    print(len(keys), "files", file=sys.stderr)

    def one(key: str) -> pl.DataFrame | None:
        raw = fetch_csv(key)
        if raw is None:
            return None
        sym = key.split("/")[5]
        rows = [ln.split(",") for ln in raw.decode().splitlines() if ln and ln[0].isdigit()]
        if not rows:
            return None
        return pl.DataFrame({
            "symbol": [sym] * len(rows), "ms": [int(r[0]) for r in rows],
            **{c: [float(r[i]) for r in rows] for i, c in enumerate(("open", "high", "low", "close", "volume"), 1)},
            "quote_volume": [float(r[7]) for r in rows],
        })  # fmt: skip

    with ThreadPoolExecutor(32) as ex:
        parts = [p for p in ex.map(one, keys) if p is not None]
    df = pl.concat(parts)
    df = df.with_columns(ms=pl.when(pl.col("ms") > 10**14).then(pl.col("ms") // 1000).otherwise(pl.col("ms")))
    df = df.with_columns(open_time=pl.from_epoch("ms", "ms").dt.replace_time_zone("UTC")).drop("ms")
    df.unique(["symbol", "open_time"]).sort("symbol", "open_time").write_parquet(OUT / "perp_1h.parquet")
    print(df.shape, file=sys.stderr)


if __name__ == "__main__" and sys.argv[1] in ("funding", "klines1h"):
    {"funding": funding, "klines1h": klines1h}[sys.argv[1]]()


def klines5m() -> None:
    """5m perp klines for every perp whose spot coin had a +20% 4h pump (feats.parquet), stored
    per symbol as OWN/futures/perp5m/<PERP>.parquet (float32 OHLC + quote volume)."""
    from futures_ls import spot_name

    d = pl.read_parquet(OWN / "feats.parquet", columns=["symbol", "ret_4h"]).filter(pl.col("ret_4h") >= 0.2)
    want = set(d["symbol"].unique().to_list())
    syms = [s for s in symbols("klines") if spot_name(s) in want]
    out = OUT / "perp5m"
    out.mkdir(exist_ok=True)
    syms = [s for s in syms if not (out / f"{s}.parquet").exists()]
    print(len(syms), "perps to fetch", file=sys.stderr)

    def one_symbol(sym: str) -> int:
        keys = [k for k in list_prefix(f"data/futures/um/monthly/klines/{sym}/5m/", False) if k.endswith(".zip")]
        parts = []
        for key in keys:
            raw = fetch_csv(key)
            if raw is None:
                continue
            rows = [ln.split(",") for ln in raw.decode().splitlines() if ln and ln[0].isdigit()]
            if rows:
                parts.append(pl.DataFrame({
                    "ms": [int(r[0]) for r in rows],
                    **{c: [float(r[i]) for r in rows] for i, c in enumerate(("open", "high", "low", "close"), 1)},
                    "qv": [float(r[7]) for r in rows],
                }))  # fmt: skip
        if not parts:
            return 0
        df = pl.concat(parts).with_columns(ms=pl.when(pl.col("ms") > 10**14).then(pl.col("ms") // 1000).otherwise(pl.col("ms")))
        df = df.unique("ms").sort("ms").with_columns(pl.col("open", "high", "low", "close", "qv").cast(pl.Float32))
        df.write_parquet(out / f"{sym}.parquet")
        return df.height

    with ThreadPoolExecutor(12) as ex:
        n = sum(ex.map(one_symbol, syms))
    print("rows", n, file=sys.stderr)


if __name__ == "__main__" and sys.argv[1:] == ["klines5m"]:
    klines5m()
