"""H1 (holdout): fetch Binance spot 5m klines of every USDT pair for 2017-08 .. 2020-01 from the
public archive data.binance.vision (monthly zips with sha256 checksums), including pairs that were
later delisted. 2020-01 is fetched only so that trades opened in late December 2019 have their 72h
path; spikes are taken before 2020-01-01.

Symbol filter as `volume_spike/fetch_archive.py` (the "all" universe): plain ASCII
`^[A-Z0-9]+USDT$` tickers, no leveraged tokens (UP/DOWN/BULL/BEAR), no pegged assets.

Downloads are untrusted data: each zip is written to its own new empty folder
`$DATA/cache/capitulation_ls/holdout/raw/`, checked against its checksum, parsed only as CSV (the
archive's 2017-2018 timestamps are milliseconds; `fetch_archive.parse` detects the unit) and
deleted. Misaligned candles after the February 2018 outage are floored to the 5m grid.
Candles go to the separate store `$DATA/cache/capitulation_ls/holdout/klines/` (same
schema and layout as the shared store; the shared store is never written).

Run: uv run --group research python h1_fetch.py
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
from common import PEGGED  # noqa: E402
from fetch_archive import EXTRA_PEGGED, FILES, LEVERAGED, get, parse, s3_list  # noqa: E402

from crypto_moonshot.data.store import KlineStore  # noqa: E402
from crypto_moonshot.models import Interval  # noqa: E402

HOLDOUT = Path("/Users/mike/code/crypto-moonshot/data/cache/capitulation_ls/holdout")
RAW = HOLDOUT / "raw"
FIRST, LAST = "2017-08", "2020-01"
PLAIN = re.compile(r"^[A-Z0-9]+USDT$")
LISTING = HOLDOUT / "symbols.json"


def months_of(symbol: str) -> list[str]:
    _, keys = s3_list(f"data/spot/monthly/klines/{symbol}/5m/")
    out = sorted({m.group(1) for k in keys if (m := re.search(r"-5m-(\d{4}-\d{2})\.zip$", k))})
    return [m for m in out if FIRST <= m <= LAST]


def symbols() -> dict[str, list[str]]:
    """Archive USDT pairs with 5m months in [FIRST, LAST] (cached in symbols.json)."""
    if LISTING.exists():
        out: dict[str, list[str]] = json.loads(LISTING.read_text())
        return out
    prefixes, _ = s3_list("data/spot/monthly/klines/")
    names = {p.rstrip("/").rsplit("/", 1)[1] for p in prefixes}
    cand = sorted(
        s
        for s in names
        if s.isascii()
        and PLAIN.match(s)
        and not LEVERAGED.search(s)
        and s not in PEGGED
        and s not in EXTRA_PEGGED
    )
    print(len(cand), "candidate USDT pairs in the archive", flush=True)
    with ThreadPoolExecutor(16) as pool:
        months = dict(zip(cand, pool.map(months_of, cand), strict=True))
    out = {s: ms for s, ms in months.items() if ms}
    LISTING.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    return out


def fetch_month(job: tuple[str, str]) -> tuple[str, str, int | str]:
    symbol, month = job
    store = KlineStore(HOLDOUT)
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
        # After the 2018-02-08/09 exchange outage the archive's candles are offset by 3m14.789s
        # for about 20 hours (241 per pair); they are floored to the 5m grid (no collisions).
        df = df.with_columns(pl.col("open_time").dt.truncate("5m"))
        df = (
            df.filter(pl.col("open_time").dt.strftime("%Y-%m") == month)
            .unique("open_time")
            .sort("open_time")
        )
        store.write_month(symbol, Interval("5m"), start, df)
        return symbol, month, df.height
    except Exception as exc:
        return symbol, month, f"error: {type(exc).__name__}: {str(exc)[:120]}"
    finally:
        tmp.unlink(missing_ok=True)


def main() -> None:
    HOLDOUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    syms = symbols()
    jobs = [(s, m) for s, ms in sorted(syms.items()) for m in ms]
    print(len(syms), "symbols,", len(jobs), "months to fetch", flush=True)
    done, bad = 0, []
    with ThreadPoolExecutor(12) as pool:
        for s, m, res in pool.map(fetch_month, jobs):
            done += 1
            if isinstance(res, str) and res != "exists":
                bad.append((s, m, res))
            if done % 200 == 0:
                print(done, "done,", len(bad), "problems", flush=True)
    (HOLDOUT / "fetch_problems.json").write_text(json.dumps(bad, indent=1) + "\n")
    print("finished:", done, "months,", len(bad), "problems", flush=True)


if __name__ == "__main__":
    main()
