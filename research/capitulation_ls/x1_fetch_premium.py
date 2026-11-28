"""x1: premium index 1h klines (data.binance.vision) for the ~150 most liquid USD-M perps.

Writes OWN/explore/premium_1h.parquet: symbol, open_time, p_open, p_high, p_low, p_close.
Run: uv run --group research python x1_fetch_premium.py
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor

import polars as pl
from common_ls import OWN
from fetch_futures import fetch_csv, list_prefix

OUT = OWN / "explore"
OUT.mkdir(parents=True, exist_ok=True)
TOP = 150


def main() -> None:
    p = pl.read_parquet(OWN / "futures" / "perp_1h.parquet", columns=["symbol", "quote_volume"])
    rank = p.group_by("symbol").agg(v=pl.col("quote_volume").sum()).sort("v", descending=True)
    syms = [s for s in rank["symbol"].to_list() if s.endswith("USDT") and s.isascii()][:TOP]
    with ThreadPoolExecutor(16) as ex:
        keys = [
            k
            for ks in ex.map(
                lambda s: list_prefix(f"data/futures/um/monthly/premiumIndexKlines/{s}/1h/", False),
                syms,
            )
            for k in ks
            if k.endswith(".zip")
        ]
    print(len(syms), "symbols", len(keys), "files", file=sys.stderr)

    def one(key: str) -> pl.DataFrame | None:
        raw = fetch_csv(key)
        if raw is None:
            return None
        rows = [ln.split(",") for ln in raw.decode().splitlines() if ln and ln[0].isdigit()]
        if not rows:
            return None
        sym = key.split("/")[5]
        return pl.DataFrame({
            "symbol": [sym] * len(rows), "ms": [int(r[0]) for r in rows],
            "p_open": [float(r[1]) for r in rows], "p_high": [float(r[2]) for r in rows],
            "p_low": [float(r[3]) for r in rows], "p_close": [float(r[4]) for r in rows],
        })  # fmt: skip

    with ThreadPoolExecutor(32) as ex:
        parts = [d for d in ex.map(one, keys) if d is not None]
    df = pl.concat(parts)
    df = df.with_columns(
        ms=pl.when(pl.col("ms") > 10**14).then(pl.col("ms") // 1000).otherwise(pl.col("ms"))
    )
    df = df.with_columns(open_time=pl.from_epoch("ms", "ms").dt.replace_time_zone("UTC")).drop("ms")
    df.unique(["symbol", "open_time"]).sort("symbol", "open_time").write_parquet(
        OUT / "premium_1h.parquet"
    )
    print(df.shape, file=sys.stderr)


if __name__ == "__main__":
    main()
