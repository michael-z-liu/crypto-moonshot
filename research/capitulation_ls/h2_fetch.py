"""H2 (holdout, BTC pairs): fetch Binance spot 5m klines of the BTC-quoted pairs (XXXBTC) of
2017-08 .. 2020-01 whose coin had no USDT pair at the time, and convert them to USDT terms.

Which pairs: plain ASCII `^[A-Z0-9]+BTC$` tickers in data.binance.vision, minus pegged / wrapped
bases (stablecoins, WBTC, RENBTC, gold). A pair is kept when its coin XXX had no XXXUSDT candles
(h1's symbols.json) in at least one month of 2017-08 .. 2019-12. Only the months it needs are
fetched: those in which XXXUSDT had no data, plus the following month (72h exit paths). Spikes
are later kept only while the USDT pair did not exist (h2_build.py), so no coin-time is counted
in both holdouts.

Conversion, on the same 5m open time (BTCUSDT from h1's holdout store; candles without a BTCUSDT
candle are dropped):
- open = open_btc x BTCUSDT open, close = close_btc x BTCUSDT close;
- high / low = high_btc / low_btc x BTCUSDT close (an approximation: the intrabar BTC and alt
  extremes need not coincide), widened to cover open and close;
- base volume, trade count, taker buy base volume unchanged; quote volume and taker buy quote
  volume x BTCUSDT close;
- converted prices and quote volumes are kept at float32 precision (relative error < 1e-7) so
  the store compresses.

Downloads are untrusted data: each zip is written only to the new folder holdout/raw_btc/ while
parsed as CSV (`fetch_archive.parse`), checked against its sha256 and deleted. Misaligned candles
after the 2018-02-08/09 outage are floored to the 5m grid, as in h1_fetch.py. Converted candles go
to the store holdout/klines_btc/ (symbol folders keep the BTC-pair name, e.g. symbol=XRPBTC).

Run: uv run --group research python h2_fetch.py
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "volume_spike"))

import polars as pl  # noqa: E402
from fetch_archive import FILES, get, parse, s3_list  # noqa: E402
from h1_fetch import months_of  # noqa: E402

from crypto_moonshot.data.store import KlineStore  # noqa: E402
from crypto_moonshot.models import Interval  # noqa: E402

HOLDOUT = Path("/Users/mike/code/crypto-moonshot/data/cache/capitulation_ls/holdout")
RAW = HOLDOUT / "raw_btc"
STORE = HOLDOUT / "klines_btc"
LISTING = HOLDOUT / "symbols_btc.json"
PLAN = HOLDOUT / "plan_btc.json"
PLAIN = re.compile(r"^[A-Z0-9]+BTC$")
NOT_ALTS = {
    "WBTC", "RENBTC", "BTCB", "PAXG", "USDT", "USDC", "TUSD", "PAX", "USDS", "BUSD", "DAI",
    "USDSB", "BKRW", "EUR", "GBP", "AUD", "XAUT", "BETH",
}  # fmt: skip
LAST_SPIKE_MONTH = "2019-12"


def next_month(m: str) -> str:
    y, mo = int(m[:4]), int(m[5:])
    return f"{y + mo // 12}-{mo % 12 + 1:02d}"


def btc_symbols() -> dict[str, list[str]]:
    """Archive BTC-quoted pairs with 5m months in 2017-08 .. 2020-01 (cached)."""
    if LISTING.exists():
        out: dict[str, list[str]] = json.loads(LISTING.read_text())
        return out
    prefixes, _ = s3_list("data/spot/monthly/klines/")
    names = {p.rstrip("/").rsplit("/", 1)[1] for p in prefixes}
    cand = sorted(
        s for s in names if s.isascii() and PLAIN.match(s) and s.removesuffix("BTC") not in NOT_ALTS
    )
    print(len(cand), "candidate BTC pairs in the archive", flush=True)
    with ThreadPoolExecutor(16) as pool:
        months = dict(zip(cand, pool.map(months_of, cand), strict=True))
    out = {s: ms for s, ms in months.items() if ms}
    LISTING.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    return out


def plan() -> dict[str, list[str]]:
    """Months to fetch per BTC pair: no-USDT months (to 2019-12) and the month after each."""
    usdt: dict[str, list[str]] = json.loads((HOLDOUT / "symbols.json").read_text())
    out: dict[str, list[str]] = {}
    for sym, months in sorted(btc_symbols().items()):
        have = set(usdt.get(sym.removesuffix("BTC") + "USDT", []))
        free = [m for m in months if m not in have and m <= LAST_SPIKE_MONTH]
        if not free:
            continue
        want = sorted((set(free) | {next_month(m) for m in free}) & set(months))
        out[sym] = want
    PLAN.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    return out


def btcusdt() -> pl.DataFrame:
    src = HOLDOUT / "klines" / "spot" / "interval=5m" / "symbol=BTCUSDT"
    return pl.read_parquet(src / "*.parquet").select("open_time", b_open="open", b_close="close")


BTC = pl.DataFrame()


def convert(df: pl.DataFrame, btc: pl.DataFrame) -> pl.DataFrame:
    schema = df.schema
    j = df.join(btc, on="open_time", how="inner")
    o = pl.col("open") * pl.col("b_open")
    c = pl.col("close") * pl.col("b_close")
    j = j.with_columns(open=o, close=c, high=pl.col("high") * pl.col("b_close"),
                       low=pl.col("low") * pl.col("b_close"),
                       quote_volume=pl.col("quote_volume") * pl.col("b_close"),
                       taker_buy_quote_volume=pl.col("taker_buy_quote_volume") * pl.col("b_close"))  # fmt: skip  # noqa: E501
    j = j.with_columns(
        high=pl.max_horizontal("high", "open", "close"),
        low=pl.min_horizontal("low", "open", "close"),
    )
    # products of two prices do not compress: keep float32 precision (relative error < 1e-7),
    # stored as float64 with zero low mantissa bits, which roughly halves the store
    f32 = ("open", "high", "low", "close", "quote_volume", "taker_buy_quote_volume")
    j = j.with_columns(pl.col(c).cast(pl.Float32).cast(pl.Float64) for c in f32)
    return j.select(list(schema)).cast(schema)  # type: ignore[arg-type]


def fetch_month(job: tuple[str, str]) -> tuple[str, str, int | str]:
    symbol, month = job
    store = KlineStore(STORE)
    y, m = int(month[:4]), int(month[5:])
    start = datetime(y, m, 1, tzinfo=UTC)
    if store.month_path(symbol, Interval("5m"), start).exists():
        return symbol, month, "exists"
    url = f"{FILES}/data/spot/monthly/klines/{symbol}/5m/{symbol}-5m-{month}.zip"
    tmp = RAW / f"{symbol}-{month}.zip"
    try:
        raw = get(url)
        want = get(url + ".CHECKSUM").decode().split()[0]
        if hashlib.sha256(raw).hexdigest() != want:
            return symbol, month, "checksum mismatch"
        tmp.write_bytes(raw)  # kept only while parsing
        df = parse(raw)
        df = df.with_columns(pl.col("open_time").dt.truncate("5m"))
        df = (
            df.filter(pl.col("open_time").dt.strftime("%Y-%m") == month)
            .unique("open_time")
            .sort("open_time")
        )
        df = convert(df, BTC)
        if df.height == 0:
            return symbol, month, "empty after conversion"
        store.write_month(symbol, Interval("5m"), start, df)
        return symbol, month, df.height
    except Exception as exc:
        return symbol, month, f"error: {type(exc).__name__}: {str(exc)[:120]}"
    finally:
        tmp.unlink(missing_ok=True)


def main() -> None:
    global BTC
    RAW.mkdir(parents=True, exist_ok=True)
    STORE.mkdir(parents=True, exist_ok=True)
    BTC = btcusdt()
    todo = plan()
    jobs = [(s, m) for s, ms in todo.items() for m in ms]
    print(len(todo), "BTC pairs,", len(jobs), "months to fetch", flush=True)
    if "--plan" in sys.argv:
        return
    done, bad = 0, []
    with ThreadPoolExecutor(12) as pool:
        for s, m, res in pool.map(fetch_month, jobs):
            done += 1
            if isinstance(res, str) and res != "exists":
                bad.append((s, m, res))
            if done % 200 == 0:
                print(done, "done,", len(bad), "problems", flush=True)
    (HOLDOUT / "fetch_btc_problems.json").write_text(json.dumps(bad, indent=1) + "\n")
    left = list(RAW.glob("*"))
    print("finished:", done, "months,", len(bad), "problems,", len(left), "files left in raw_btc")


if __name__ == "__main__":
    main()
