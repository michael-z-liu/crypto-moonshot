"""M13: fetch 5m candles of delisted and non-pool USDT pairs from Binance's public data archive.

The live API (used by `moonshot data sync`) has no history for delisted pairs; data.binance.vision
keeps them as monthly zips with sha256 checksums. Candles are written with the repo's KlineStore
into a separate store, ARCHIVE (gitignored), which `common.load` reads alongside the main one and
`moonshot data import` can merge into it.

    uv run python research/volume_spike/fetch_archive.py groups   # list groups A and B
    uv run python research/volume_spike/fetch_archive.py fetch    # download and store

Downloads are untrusted data: zips are checked against their checksum, only parsed as CSV, kept in
their own temporary folder and deleted after parsing.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import sys
import tempfile
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
from common import ARCHIVE, DATA_DIR, HERE, PEGGED

from crypto_moonshot.data.store import KlineStore
from crypto_moonshot.models import KLINE_SCHEMA, Interval

S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
FILES = "https://data.binance.vision"
EXCHANGE_INFO = "https://data-api.binance.vision/api/v3/exchangeInfo"
FIRST, LAST = "2020-01", "2026-10"
RESULTS = HERE / "results"
GROUPS = RESULTS / "archive_groups.json"
LEVERAGED = re.compile(r"(UP|DOWN|BULL|BEAR)USDT$")
EXTRA_PEGGED = {
    "BUSDUSDT", "TUSDUSDT", "USDPUSDT", "PAXUSDT", "USDSUSDT", "USDSBUSDT", "DAIUSDT", "SUSDUSDT",
    "USTUSDT", "AEURUSDT", "GBPUSDT", "AUDUSDT", "BKRWUSDT", "IDRTUSDT", "BIDRUSDT", "FRAXUSDT",
    "USDSOLDUSDT", "BETHUSDT", "BNSOLUSDT", "USDEBUSDT",
}  # fmt: skip
UA = {"User-Agent": "crypto-moonshot-research/0.1"}


def get(url: str, tries: int = 4) -> bytes:
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                return bytes(r.read())
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))
    raise RuntimeError(url)


def s3_list(prefix: str) -> tuple[list[str], list[str]]:
    """(sub-prefixes, keys) under an S3 prefix, following pagination."""
    prefixes, keys, marker = [], [], ""
    while True:
        xml = get(f"{S3}?delimiter=/&prefix={prefix}&marker={marker}").decode()
        prefixes += [x for x in re.findall(r"<Prefix>([^<]+/)</Prefix>", xml) if x != prefix]
        keys += re.findall(r"<Key>([^<]+)</Key>", xml)
        nxt = re.findall(r"<NextMarker>([^<]+)</NextMarker>", xml)
        if "<IsTruncated>true</IsTruncated>" not in xml:
            break
        marker = nxt[0] if nxt else (keys[-1] if keys else prefixes[-1])
    return prefixes, keys


def months_of(symbol: str) -> list[str]:
    _, keys = s3_list(f"data/spot/monthly/klines/{symbol}/5m/")
    out = sorted({m.group(1) for k in keys if (m := re.search(r"-5m-(\d{4}-\d{2})\.zip$", k))})
    return [m for m in out if FIRST <= m <= LAST]


def groups() -> None:
    prefixes, _ = s3_list("data/spot/monthly/klines/")
    archive = {p.rstrip("/").rsplit("/", 1)[1] for p in prefixes}
    info = json.loads(get(EXCHANGE_INFO))
    status = {s["symbol"]: s["status"] for s in info["symbols"]}
    pool = set((DATA_DIR / "top200.txt").read_text().split())
    plain = re.compile(r"^[A-Z0-9]+USDT$")  # the store only takes plain tickers
    cand = sorted(
        s for s in archive if plain.match(s) and s not in pool and not LEVERAGED.search(s)
    )
    cand = [s for s in cand if s not in PEGGED and s not in EXTRA_PEGGED]
    with ThreadPoolExecutor(16) as pool_:
        months = dict(zip(cand, pool_.map(months_of, cand), strict=True))
    out: dict[str, dict[str, list[str]]] = {
        "A_delisted": {},
        "B_listed_outside_pool": {},
        "excluded_stocks": {},
    }
    for s, ms in months.items():
        if not ms:
            continue
        # Binance's tokenised stocks and ETFs: "...BUSDT" tickers first listed from 2025-06
        if s.endswith("BUSDT") and ms[0] >= "2025-06":
            out["excluded_stocks"][s] = ms
        elif status.get(s) == "TRADING":
            out["B_listed_outside_pool"][s] = ms
        else:
            out["A_delisted"][s] = ms
    GROUPS.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    for g, v in out.items():
        print(g, len(v), "symbols,", sum(len(m) for m in v.values()), "months")


def parse(raw: bytes) -> pl.DataFrame:
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        name = z.namelist()[0]
        text = z.read(name)
    df = pl.read_csv(io.BytesIO(text), has_header=False, infer_schema_length=0)
    if not df[0, 0].isdigit():  # some archive files carry a header row
        df = df.slice(1)
    df = df.select(df.columns[:11])
    df.columns = [
        "open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
        "trade_count", "taker_buy_base_volume", "taker_buy_quote_volume",
    ]  # fmt: skip
    t = df["open_time"].cast(pl.Int64)
    unit = (
        "us" if int(t.max() or 0) > 10**14 else "ms"
    )  # spot files switched to microseconds in 2025
    df = df.with_columns(
        open_time=pl.from_epoch(t, time_unit=unit)
        .dt.replace_time_zone("UTC")
        .dt.cast_time_unit("us")
    )
    return df.select(
        pl.col(c).cast(dtype) if c != "open_time" else pl.col(c)
        for c, dtype in KLINE_SCHEMA.items()
    )


def fetch_month(job: tuple[str, str, Path]) -> tuple[str, str, int | str]:
    symbol, month, tmp = job
    store = KlineStore(ARCHIVE)
    y, m = int(month[:4]), int(month[5:])
    if store.month_path(symbol, Interval("5m"), datetime(y, m, 1, tzinfo=UTC)).exists():
        return symbol, month, "exists"
    url = f"{FILES}/data/spot/monthly/klines/{symbol}/5m/{symbol}-5m-{month}.zip"
    try:
        raw = get(url)
        want = get(url + ".CHECKSUM").decode().split()[0]
        if hashlib.sha256(raw).hexdigest() != want:
            return symbol, month, "checksum mismatch"
        (tmp / f"{symbol}-{month}.zip").write_bytes(raw)  # kept only while parsing
        df = parse(raw)
        df = (
            df.filter(pl.col("open_time").dt.strftime("%Y-%m") == month)
            .unique("open_time")
            .sort("open_time")
        )
        store.write_month(symbol, Interval("5m"), datetime(y, m, 1, tzinfo=UTC), df)
        (tmp / f"{symbol}-{month}.zip").unlink()
        return symbol, month, df.height
    except Exception as exc:
        return symbol, month, f"error: {type(exc).__name__}: {str(exc)[:120]}"


def fetch() -> None:
    g = json.loads(GROUPS.read_text())
    jobs = []
    tmp = Path(tempfile.mkdtemp(prefix="binance_archive_"))
    for grp in ("A_delisted", "B_listed_outside_pool"):
        for s, ms in g[grp].items():
            jobs += [(s, m, tmp) for m in ms]
    print(len(jobs), "months to fetch into", ARCHIVE, flush=True)
    done, bad = 0, []
    with ThreadPoolExecutor(12) as pool:
        for s, m, res in pool.map(fetch_month, jobs):
            done += 1
            if isinstance(res, str) and res != "exists":
                bad.append((s, m, res))
            if done % 500 == 0:
                print(done, "done,", len(bad), "problems", flush=True)
    (RESULTS / "archive_fetch_problems.json").write_text(json.dumps(bad, indent=1) + "\n")
    print("finished:", done, "months,", len(bad), "problems", flush=True)


def publish() -> None:
    """Import the fetched store into the main data folder, so any session can reuse it.

    Uses the repo's own `moonshot data import` (validates every month file, never overwrites
    stored candles), then writes the group lists next to `top200.txt`.
    """
    import subprocess

    cmd = [
        "uv",
        "run",
        "moonshot",
        "data",
        "import",
        str(ARCHIVE),
        "--data-dir",
        str(DATA_DIR),
        "--json",
    ]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    (ARCHIVE.parent / "archive_import_report.json").write_text(out.strip() + "\n")  # large
    report = json.loads(out)
    summary = {"destination": report["destination"], "totals": report["totals"]}
    (RESULTS / "archive_import_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    g = json.loads(GROUPS.read_text())
    have = {
        p.name.removeprefix("symbol=")
        for p in (DATA_DIR / "klines/spot/interval=5m").glob("symbol=*")
    }
    for name, key in (
        ("delisted.txt", "A_delisted"),
        ("outside_pool.txt", "B_listed_outside_pool"),
    ):
        syms = sorted(s for s in g[key] if s in have)
        (DATA_DIR / name).write_text("\n".join(syms) + "\n")
        print(name, len(syms), "symbols")


if __name__ == "__main__":
    {"groups": groups, "fetch": fetch, "publish": publish}[sys.argv[1]]()
