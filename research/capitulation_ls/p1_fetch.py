"""P1 (perp-only coins): list the USD-M perpetuals that have no Binance spot market and fetch their
5m klines (2020-2026) from data.binance.vision.

Perp-only = a USDT-margined perp in the archive's monthly klines (delisted ones included) whose
symbol and spot name (1000X -> X, `futures_ls.spot_name`) both lack spot 5m klines in the shared
data folder, minus non-crypto underlyings: exchangeInfo's `underlyingType` must be COIN with
contract type PERPETUAL (TradFi stocks, ETFs, commodities, FX and the INDEX perps are dropped);
perps no longer in exchangeInfo (delisted) are kept unless they are in `MANUAL_DROP`.

Raw zips go to OWN/futures/raw_perponly/ (one at a time per worker, deleted after parsing);
output per symbol: OWN/futures/perp5m/<PERP>.parquet with the existing columns (ms, open, high,
low, close, qv as float32) plus vol (base volume), tbv (taker-buy base volume) and n (trades),
which the CNN inputs need. The symbol list goes to OWN/futures/perp_only.json.

Run: uv run --group research python p1_fetch.py [list|fetch]
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
from common_ls import OWN
from fetch_futures import DL, list_prefix, symbols
from futures_ls import spot_name

FUT = OWN / "futures"
RAW = FUT / "raw_perponly"
P5 = FUT / "perp5m"
LIST = FUT / "perp_only.json"
SPOT_5M = Path("/Users/mike/code/crypto-moonshot/data/klines/spot/interval=5m")
# Delisted perps (not in exchangeInfo) that are not crypto coins: index perps.
MANUAL_DROP = {"DEFIUSDT", "BTCDOMUSDT", "FOOTBALLUSDT", "BLUEBIRDUSDT"}
STABLE = {"USDEXUSDT", "USUSDT", "FRAXUSDT", "USDBRLUSDT"}


def perp_only() -> list[str]:
    spot = {p.name.removeprefix("symbol=") for p in SPOT_5M.glob("symbol=*")}
    perps = symbols("klines")
    info = json.loads((RAW / "exinfo.json").read_text())["symbols"]
    kind = {s["symbol"]: (s.get("underlyingType"), s.get("contractType")) for s in info}
    out, gone = [], []
    for p in perps:
        if p in spot or spot_name(p) in spot or p in MANUAL_DROP or p in STABLE:
            continue
        k = kind.get(p)
        if k is None:
            gone.append(p)
            out.append(p)
        elif k == ("COIN", "PERPETUAL"):
            out.append(p)
    print(len(perps), "perps in the archive;", len(out), "perp-only coins;",
          len(gone), "of them delisted:", " ".join(gone), file=sys.stderr)  # fmt: skip
    LIST.write_text(json.dumps({"perp_only": out, "delisted": gone}, indent=1))
    return out


def fetch_zip(key: str) -> bytes | None:
    """Download one zip into RAW, return the CSV bytes and delete the zip."""
    dst = RAW / key.split("/")[-1]
    for _ in range(4):
        try:
            urllib.request.urlretrieve(DL + key, dst)
            with zipfile.ZipFile(dst) as z:
                names = z.namelist()
                if len(names) != 1 or not names[0].endswith(".csv"):
                    return None
                raw = z.read(names[0])
            return raw
        except Exception:
            continue
        finally:
            if dst.exists():
                os.remove(dst)
    print("failed", key, file=sys.stderr)
    return None


def num(x: str) -> float:
    try:
        return float(x)
    except ValueError:
        return float("nan")


def one_symbol(sym: str) -> int:
    out = P5 / f"{sym}.parquet"
    if out.exists():
        return 0
    keys = [
        k
        for k in list_prefix(f"data/futures/um/monthly/klines/{sym}/5m/", False)
        if k.endswith(".zip")
    ]
    parts = []
    for key in keys:
        raw = fetch_zip(key)
        if raw is None:
            continue
        rows = [ln.split(",") for ln in raw.decode("ascii", "replace").splitlines()]
        rows = [r for r in rows if len(r) >= 11 and r[0].isdigit()]
        if rows:
            parts.append(pl.DataFrame({
                "ms": [int(r[0]) for r in rows],
                **{c: [num(r[i]) for r in rows] for i, c in enumerate(("open", "high", "low", "close", "vol"), 1)},  # noqa: E501
                "qv": [num(r[7]) for r in rows],
                "n": [int(num(r[8])) if r[8].isdigit() else 0 for r in rows],
                "tbv": [num(r[9]) for r in rows],
            }))  # fmt: skip
    if not parts:
        return 0
    df = pl.concat(parts).with_columns(
        ms=pl.when(pl.col("ms") > 10**14).then(pl.col("ms") // 1000).otherwise(pl.col("ms"))
    )
    df = (
        df.unique("ms")
        .sort("ms")
        .with_columns(
            pl.col("open", "high", "low", "close", "qv", "vol", "tbv").cast(pl.Float32),
            pl.col("n").cast(pl.Int32),
        )
        .select("ms", "open", "high", "low", "close", "qv", "vol", "tbv", "n")
    )
    df.write_parquet(out)
    return df.height


def main() -> None:
    syms = perp_only()
    if sys.argv[1:] == ["list"]:
        return
    with ThreadPoolExecutor(24) as ex:
        n = 0
        for i, k in enumerate(ex.map(one_symbol, syms)):
            n += k
            if i % 20 == 0:
                print(i, "symbols,", n, "rows", file=sys.stderr, flush=True)
    print("rows", n, file=sys.stderr)


if __name__ == "__main__":
    main()
