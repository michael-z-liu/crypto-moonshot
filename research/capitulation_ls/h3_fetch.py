"""H3 (other quotes, 2020-26): fetch the Binance spot 5m klines of pairs quoted in BUSD, BTC, FDUSD,
USDC, TUSD, ETH or BNB for coin-months of 2020-01 .. 2026-10 in which the coin had no USDT spot
pair, and convert them to USDT terms with the quote's USDT pair, as h2_fetch.py does for the
2017-19 BTC pairs (`h2_fetch.convert`, unchanged).

Which pairs: plain ASCII `^[A-Z0-9]+(FDUSD|BUSD|TUSD|USDC|BTC|ETH|BNB)$` tickers in
data.binance.vision, minus pegged / wrapped / fiat / leveraged bases (h2's `NOT_ALTS` plus the
newer stablecoins). The coin's USDT months come from the study's own USDT stores (the shared spot
folder and the volume-spike archive store, which together are the "all" universe) and, for a
USDT pair that is in the Binance archive but in neither store, from the archive's month list.
Months fetched per pair, within its archive months of 2019-12 .. 2026-10:
- the months in which XXXUSDT had no data (2020-01 .. 2026-10);
- the month before and after each of them (history for the spike detector's first week and for
  spikes right after a USDT delisting; 72h exit paths);
- the pair's first month when the USDT pair starts in that same month (the pre-listing window).
Spikes are kept only while the coin had no USDT pair (h3_build.py), so no coin-time is counted
twice. Perp-only coins (p1) are handled in h3_build.py.

Conversion series (the quote's USDT pair: BTCUSDT, ETHUSDT, BNBUSDT, BUSDUSDT, FDUSDUSDT,
USDCUSDT, TUSDUSDT): the shared spot folder where it has them, else fetched here (BUSDUSDT,
TUSDUSDT, and every quote's 2019-12) into OWN/h3/quotes/. For the stablecoins the factor is ~1,
but it is applied all the same.

Downloads are untrusted data: each zip is written only to the new folder OWN/h3/raw/ while it is
parsed as CSV (`fetch_archive.parse`), checked against its sha256 and deleted. Converted candles
go to OWN/h3/pairs/ (symbol folders keep the pair name, e.g. symbol=XYZBTC).

Run: uv run --group research python h3_fetch.py [--plan] [--reverse]
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
from h2_fetch import NOT_ALTS, convert, next_month  # noqa: E402

from crypto_moonshot.data.store import KlineStore  # noqa: E402
from crypto_moonshot.models import Interval  # noqa: E402

H3 = Path("/Users/mike/code/crypto-moonshot/data/cache/capitulation_ls/h3")
RAW = H3 / "raw"
PAIRS = H3 / "pairs"
QUOTES = H3 / "quotes"
LISTING = H3 / "symbols.json"
PLAN = H3 / "plan.json"
SPOT_5M = Path("/Users/mike/code/crypto-moonshot/data/klines/spot/interval=5m")
ARCH_5M = Path(
    "/Users/mike/code/crypto-moonshot/data/cache/volume_spike/archive/klines/spot/interval=5m"
)
QUOTE_LIST = ("FDUSD", "BUSD", "TUSD", "USDC", "BTC", "ETH", "BNB")  # longest suffix first
CONV = {q: f"{q}USDT" for q in QUOTE_LIST}
FIRST, LAST = "2019-12", "2026-10"
SPIKE_FIRST = "2020-01"
EXTRA_NOT_ALTS = {
    "FDUSD", "USDP", "AEUR", "EURI", "USDE", "XUSD", "USD1", "BFUSD", "UST", "SUSD", "IDRT",
    "BIDR", "TRY", "BRL", "RUB", "NGN", "UAH", "ZAR", "PLN", "RON", "ARS", "JPY", "MXN", "COP",
    "CZK", "BNSOL", "WBETH", "USDSOLD", "FRAX", "RLUSD", "USDEB", "BTC", "ETH", "BNB",
    "USD",  # USDTUSD parses as USD/TUSD (a stablecoin pair, fetched before this was added)
}  # fmt: skip
LEVER = re.compile(r"(UP|DOWN|BULL|BEAR)$")


def split(sym: str) -> tuple[str, str] | None:
    """(base, quote) for a plain ticker quoted in one of QUOTE_LIST, else None."""
    if not sym.isascii() or not re.fullmatch(r"[A-Z0-9]+", sym):
        return None
    for q in QUOTE_LIST:
        if sym.endswith(q) and len(sym) > len(q):
            base = sym[: -len(q)]
            if base in NOT_ALTS or base in EXTRA_NOT_ALTS or LEVER.search(base):
                return None
            return base, q
    return None


def months_in(symbol: str) -> list[str]:
    _, keys = s3_list(f"data/spot/monthly/klines/{symbol}/5m/")
    out = sorted({m.group(1) for k in keys if (m := re.search(r"-5m-(\d{4}-\d{2})\.zip$", k))})
    return [m for m in out if FIRST <= m <= LAST]


def listing() -> dict[str, dict[str, list[str]]]:
    """{"pairs": candidate pair -> months, "usdt_extra": archive USDT pair outside both stores ->
    months} (cached)."""
    if LISTING.exists():
        out: dict[str, dict[str, list[str]]] = json.loads(LISTING.read_text())
        return out
    prefixes, _ = s3_list("data/spot/monthly/klines/")
    names = {p.rstrip("/").rsplit("/", 1)[1] for p in prefixes}
    cand = sorted(s for s in names if split(s) is not None)
    bases = {b for s in cand if (sp := split(s)) is not None for b in [sp[0]]}
    stored = {
        p.name.removeprefix("symbol=") for d in (SPOT_5M, ARCH_5M) for p in d.glob("symbol=*")
    }
    extra = sorted(f"{b}USDT" for b in bases if f"{b}USDT" in names and f"{b}USDT" not in stored)
    conv = [s for s in CONV.values() if s in names]
    print(len(cand), "candidate pairs,", len(extra), "archive USDT pairs outside the stores",
          flush=True)  # fmt: skip
    with ThreadPoolExecutor(16) as pool:
        pm = dict(zip(cand, pool.map(months_in, cand), strict=True))
        um = dict(zip(extra, pool.map(months_in, extra), strict=True))
        cm = dict(zip(conv, pool.map(months_in, conv), strict=True))
    out = {
        "pairs": {s: ms for s, ms in pm.items() if ms},
        "usdt_extra": {s: ms for s, ms in um.items() if ms},
        "conv": cm,
    }
    LISTING.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    return out


def store_months(symbol: str) -> set[str]:
    out: set[str] = set()
    for d in (SPOT_5M, ARCH_5M):
        p = d / f"symbol={symbol}"
        if p.exists():
            out |= {f.stem for f in p.glob("*.parquet")}
    return out


def prev_month(m: str) -> str:
    y, mo = int(m[:4]), int(m[5:])
    return f"{y - 1}-12" if mo == 1 else f"{y}-{mo - 1:02d}"


def plan() -> dict[str, list[str]]:
    """Months to fetch per pair (see the module docstring)."""
    lst = listing()
    extra = lst["usdt_extra"]
    out: dict[str, list[str]] = {}
    for sym, months in sorted(lst["pairs"].items()):
        sp = split(sym)
        assert sp is not None
        usdt = f"{sp[0]}USDT"
        have = store_months(usdt) | set(extra.get(usdt, []))
        free = [m for m in months if m not in have and m >= SPIKE_FIRST]
        want = set(free) | {next_month(m) for m in free} | {prev_month(m) for m in free}
        if have and months[0] >= SPIKE_FIRST and months[0] == min(have):
            want.add(months[0])  # pre-listing window inside the USDT pair's first month
        want &= set(months)
        if free or want:
            out[sym] = sorted(want)
    PLAN.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    return out


def quote_frame(sym: str) -> pl.DataFrame:
    """The quote's USDT 5m open / close (shared folder + OWN/h3/quotes), as b_open / b_close."""
    parts = []
    for d in (SPOT_5M, QUOTES / "klines" / "spot" / "interval=5m"):
        p = d / f"symbol={sym}"
        if p.exists() and any(p.glob("*.parquet")):
            parts.append(pl.read_parquet(p / "*.parquet", columns=["open_time", "open", "close"]))
    df = pl.concat(parts).unique("open_time", keep="first").sort("open_time")
    return df.select("open_time", b_open="open", b_close="close")


def download(symbol: str, month: str) -> pl.DataFrame | str:
    url = f"{FILES}/data/spot/monthly/klines/{symbol}/5m/{symbol}-5m-{month}.zip"
    tmp = RAW / f"{symbol}-{month}.zip"
    try:
        raw = get(url)
        want = get(url + ".CHECKSUM").decode().split()[0]
        if hashlib.sha256(raw).hexdigest() != want:
            return "checksum mismatch"
        tmp.write_bytes(raw)  # kept only while parsing
        df = parse(raw)
        df = df.with_columns(pl.col("open_time").dt.truncate("5m"))
        return (
            df.filter(pl.col("open_time").dt.strftime("%Y-%m") == month)
            .unique("open_time")
            .sort("open_time")
        )
    except Exception as exc:
        return f"error: {type(exc).__name__}: {str(exc)[:120]}"
    finally:
        tmp.unlink(missing_ok=True)


def month_start(month: str) -> datetime:
    return datetime(int(month[:4]), int(month[5:]), 1, tzinfo=UTC)


def fetch_quote(job: tuple[str, str]) -> tuple[str, str, int | str]:
    symbol, month = job
    store = KlineStore(QUOTES)
    if store.month_path(symbol, Interval("5m"), month_start(month)).exists():
        return symbol, month, "exists"
    df = download(symbol, month)
    if isinstance(df, str):
        return symbol, month, df
    store.write_month(symbol, Interval("5m"), month_start(month), df)
    return symbol, month, df.height


QF: dict[str, pl.DataFrame] = {}


def fetch_pair(job: tuple[str, str]) -> tuple[str, str, int | str]:
    symbol, month = job
    store = KlineStore(PAIRS)
    if store.month_path(symbol, Interval("5m"), month_start(month)).exists():
        return symbol, month, "exists"
    df = download(symbol, month)
    if isinstance(df, str):
        return symbol, month, df
    sp = split(symbol)
    assert sp is not None
    df = convert(df, QF[sp[1]])
    if df.height == 0:
        return symbol, month, "empty after conversion"
    store.write_month(symbol, Interval("5m"), month_start(month), df)
    return symbol, month, df.height


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    PAIRS.mkdir(parents=True, exist_ok=True)
    QUOTES.mkdir(parents=True, exist_ok=True)
    lst = listing()
    todo = plan()
    jobs = [(s, m) for s, ms in todo.items() for m in ms]
    if "--reverse" in sys.argv:  # a second worker from the other end (existing months are skipped)
        jobs = jobs[::-1]
    by_q: dict[str, list[int]] = {}
    for s, ms in todo.items():
        sp = split(s)
        assert sp is not None
        by_q.setdefault(sp[1], [0, 0])
        by_q[sp[1]][0] += 1
        by_q[sp[1]][1] += len(ms)
    print(len(lst["pairs"]), "listed pairs;", len(todo), "planned pairs,", len(jobs), "months;",
          "per quote (pairs, months):", by_q, flush=True)  # fmt: skip
    if "--plan" in sys.argv:
        return
    qjobs = []
    for sym in CONV.values():
        have = {f.stem for f in (SPOT_5M / f"symbol={sym}").glob("*.parquet")}
        qjobs += [(sym, m) for m in lst["conv"].get(sym, []) if m not in have]
    print(len(qjobs), "quote months to fetch", flush=True)
    bad: list[tuple[str, str, str]] = []
    with ThreadPoolExecutor(12) as pool:
        for s, m, res in pool.map(fetch_quote, qjobs):
            if isinstance(res, str) and res != "exists":
                bad.append((s, m, res))
    for q, sym in CONV.items():
        QF[q] = quote_frame(sym)
        print(q, sym, QF[q].height, "candles", QF[q]["open_time"].min(), QF[q]["open_time"].max(),
              flush=True)  # fmt: skip
    done = 0
    with ThreadPoolExecutor(12) as pool:
        for s, m, res in pool.map(fetch_pair, jobs):
            done += 1
            if isinstance(res, str) and res != "exists":
                bad.append((s, m, res))
            if done % 500 == 0:
                print(done, "done,", len(bad), "problems", flush=True)
    name = "fetch_problems_rev.json" if "--reverse" in sys.argv else "fetch_problems.json"
    (H3 / name).write_text(json.dumps(bad, indent=1) + "\n")
    left = list(RAW.glob("*"))
    print("finished:", done, "months,", len(bad), "problems,", len(left), "files left in raw")


if __name__ == "__main__":
    main()
