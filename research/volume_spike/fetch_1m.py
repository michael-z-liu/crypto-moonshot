"""B4: 1-minute candles for every coin in the "all" universe from Binance's public archive.

For each coin and each month it has 5m candles (2020-01..2026-10), the monthly 1m zip from
data.binance.vision is checked against its sha256 checksum, parsed (fetch_archive.parse) and
written with the repo's KlineStore into the data folder under interval=1m. Months already stored
are skipped, so re-running resumes. Months with no archive file are listed in
results/universe_all/fetch_1m_problems.json.

Run: MOONSHOT_UNIVERSE=all uv run python research/volume_spike/fetch_1m.py
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import polars as pl
from common import DATA_DIR, KLINES_5M, RESULTS, universe
from fetch_archive import FILES, get, parse

from crypto_moonshot.data.store import KlineStore
from crypto_moonshot.models import Interval

LAST = "2026-10"
M1 = Interval("1m")


def months(symbol: str) -> list[str]:
    d = KLINES_5M / f"symbol={symbol}"
    return sorted(p.stem for p in d.glob("20*.parquet") if p.stem <= LAST)


def fetch(job: tuple[str, str]) -> tuple[str, str, int | str]:
    symbol, month = job
    store = KlineStore(DATA_DIR)
    y, m = int(month[:4]), int(month[5:])
    start = datetime(y, m, 1, tzinfo=UTC)
    if store.month_path(symbol, M1, start).exists():
        return symbol, month, "exists"
    url = f"{FILES}/data/spot/monthly/klines/{symbol}/1m/{symbol}-1m-{month}.zip"
    try:
        raw = get(url)
        want = get(url + ".CHECKSUM").decode().split()[0]
        if hashlib.sha256(raw).hexdigest() != want:
            return symbol, month, "checksum mismatch"
        df = parse(raw)
        df = (
            df.filter(pl.col("open_time").dt.strftime("%Y-%m") == month)
            .unique("open_time")
            .sort("open_time")
        )
        store.write_month(symbol, M1, start, df)
        return symbol, month, df.height
    except Exception as exc:
        return symbol, month, f"error: {type(exc).__name__}: {str(exc)[:120]}"


def main() -> None:
    jobs = [(s, m) for s in universe() for m in months(s)]
    print(len(jobs), "symbol-months to fetch", flush=True)
    done, bad, rows = 0, [], 0
    with ThreadPoolExecutor(16) as pool:
        for s, m, res in pool.map(fetch, jobs):
            done += 1
            if isinstance(res, str) and res != "exists":
                bad.append((s, m, res))
            elif isinstance(res, int):
                rows += res
            if done % 1000 == 0:
                print(done, "done,", len(bad), "problems,", rows, "candles", flush=True)
    (RESULTS / "fetch_1m_problems.json").write_text(json.dumps(bad, indent=1) + "\n")
    print("finished:", done, "months,", len(bad), "problems,", rows, "new candles", flush=True)


if __name__ == "__main__":
    main()
