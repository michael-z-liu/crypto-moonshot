"""M0: download daily candles from before 2020 for coins listed earlier.

The store starts on 2020-01-01, so the long averages (1d 200, the weekly and monthly ones) would
only exist months or years later for coins that were already trading. This fetches their 1d
candles from the listing date to 2020-01-01 into `cache/daily_pre2020/<SYMBOL>.parquet`. They are
used only to complete the 1d/1w/1M averages, never to trade.

    uv run --group research python fetch_daily.py

Listed pairs come from the public market-data API (no key). Delisted pairs are not served there,
so they come from the data.binance.vision archive (monthly zips, checked against their sha256,
parsed as CSV only). Rerunning skips coins already fetched.
"""

from __future__ import annotations

import hashlib
import io
import json
import time
import zipfile
from datetime import UTC, datetime

import httpx
import polars as pl
from common import DAILY_PRE2020, KLINES_5M, RESULTS, STORE_START, universe

from crypto_moonshot.data.binance import klines_to_frame
from crypto_moonshot.models import KLINE_SCHEMA

API = "https://data-api.binance.vision/api/v3/klines"
FILES = "https://data.binance.vision/data/spot/monthly/klines"
FIRST_MONTH = (2017, 7)  # Binance opened in July 2017
UA = {"User-Agent": "crypto-moonshot-research/0.1"}


def listed_before_2020(symbol: str) -> bool:
    """True when the coin's first stored 5m candle is the store's first candle."""
    first = sorted((KLINES_5M / f"symbol={symbol}").glob("*.parquet"))
    if not first or first[0].stem != "2020-01":
        return False
    t = pl.read_parquet(first[0], columns=["open_time"])["open_time"].min()
    return isinstance(t, datetime) and t == STORE_START


def get(client: httpx.Client, url: str, **params: str | int) -> httpx.Response:
    for attempt in range(5):
        try:
            r = client.get(url, params=params)
        except httpx.HTTPError:
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code in (418, 429):
            raise SystemExit(f"rate limited by Binance ({r.status_code}); stop and retry later")
        if r.status_code >= 500:
            time.sleep(2 * (attempt + 1))
            continue
        return r
    raise RuntimeError(f"no answer from {url}")


def from_api(client: httpx.Client, symbol: str) -> pl.DataFrame | None:
    """Daily candles before 2020 from the API, or None when the API does not know the pair."""
    end_ms = int(STORE_START.timestamp() * 1000) - 1
    r = get(client, API, symbol=symbol, interval="1d", startTime=0, endTime=end_ms, limit=1000)
    if r.status_code == 400:  # invalid symbol: delisted
        return None
    r.raise_for_status()
    rows = r.json()
    if len(rows) >= 1000:
        raise RuntimeError(f"{symbol}: more than one page of daily candles before 2020")
    return klines_to_frame(rows) if rows else pl.DataFrame(schema=KLINE_SCHEMA)


def from_archive(client: httpx.Client, symbol: str) -> pl.DataFrame:
    """Daily candles before 2020 from the monthly archive zips."""
    frames = []
    year, month = FIRST_MONTH
    while year < 2020:
        name = f"{symbol}-1d-{year}-{month:02d}.zip"
        r = get(client, f"{FILES}/{symbol}/1d/{name}")
        if r.status_code == 200:
            want = get(client, f"{FILES}/{symbol}/1d/{name}.CHECKSUM").text.split()[0]
            if hashlib.sha256(r.content).hexdigest() != want:
                raise RuntimeError(f"{name}: checksum mismatch")
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                raw = z.read(z.namelist()[0])
            csv = pl.read_csv(io.BytesIO(raw), has_header=False, infer_schema_length=0)
            rows = [list(row) for row in csv.iter_rows() if row[0].isdigit()]
            frames.append(klines_to_frame(rows))
        elif r.status_code != 404:
            r.raise_for_status()
        year, month = (year, month + 1) if month < 12 else (year + 1, 1)
    if not frames:
        return pl.DataFrame(schema=KLINE_SCHEMA)
    return pl.concat(frames).unique("open_time", keep="first").sort("open_time")


def main() -> None:
    DAILY_PRE2020.mkdir(parents=True, exist_ok=True)
    symbols = [s for s in universe() if listed_before_2020(s)]
    summary: dict[str, dict[str, str | int | None]] = {}
    with httpx.Client(timeout=60, headers=UA) as client:
        for symbol in symbols:
            out = DAILY_PRE2020 / f"{symbol}.parquet"
            if out.exists():
                df, source = pl.read_parquet(out), "cached"
            else:
                api = from_api(client, symbol)
                source = "api" if api is not None else "archive"
                df = api if api is not None else from_archive(client, symbol)
                df = df.filter(pl.col("open_time") < STORE_START).sort("open_time")
                tmp = out.with_suffix(".tmp")
                df.write_parquet(tmp)
                tmp.rename(out)
                time.sleep(0.2)
            first = df["open_time"].min()
            summary[symbol] = {
                "source": source,
                "days": df.height,
                "first": first.astimezone(UTC).date().isoformat()
                if isinstance(first, datetime)
                else None,
            }
            print(symbol, summary[symbol], flush=True)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "daily_pre2020.json").write_text(json.dumps(summary, indent=1) + "\n")
    empty = [s for s, v in summary.items() if not v["days"]]
    print(f"{len(symbols)} coins listed before 2020; {len(empty)} without daily history: {empty}")


if __name__ == "__main__":
    main()
