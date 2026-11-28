"""H3 (other quotes, 2020-26): liquid 5m volume spikes of the USDT-converted pairs (h3_fetch.py) on
the main study's grid (T0 = 2020-01-01), their 13 CNN channels and their outcomes.

- spikes: h1_build.spike_job's selection, unchanged (`build_events.features`, vol_mult >= 10, a
  series' first week skipped, 48 gap-free forward candles, qv_base >= 5,000 USDT per 5m), here on
  the main grid; open time in [2020-01-02, 2026-10-29) (the market channels start on 2020-01-01
  and the fetched months end with 2026-10, so a 72h path fits);
- kept only while the coin had no USDT spot pair (open time before the XXXUSDT series' first
  candle or after its last, from the shared folder and the volume-spike archive store, as
  `h2_build.keep_free`; a USDT pair only in the Binance archive blocks its whole months);
- perp-only coins (p1's universe, OWN/futures/perp_only.json): spikes from 2022 on while the perp
  traded (between its first and last perp5m candle) are dropped, since p1's L already covers that
  coin-time;
- one coin can have several quote pairs: rows keep the pair name in `pair` and the coin
  (`XXXUSDT`) in `symbol`, so the simulator's one-position-per-symbol rule is per coin;
- channels: the 7 coin channels as `build_seq2.coin_job` (via `p1_lib.spikes_and_coin_channels`,
  the same arrays), BTC / ETH and the market channels 11-12 from the study's own caches
  (`p1_lib.Context`: shared BTCUSDT / ETHUSDT and cache_all/market.parquet, the inputs the folds
  were trained on);
- outcomes: `build_seq2.fwd_job` (fwd2) and `build_fwd72.job` (fwd72) on the converted series
  (`common.load` reads the pair from OWN/h3/pairs through the archive-store switch, as h2_build).

Writes OWN/h3/{meta.parquet, seq.npy, fwd2.npy, fwd72.npy, coverage.json}.
Run: uv run --group research python h3_build.py
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

os.environ["MOONSHOT_UNIVERSE"] = "all"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "volume_spike"))

import build_events  # noqa: E402
import build_fwd72  # noqa: E402
import build_seq2  # noqa: E402
import common  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from build_seq2 import CHANNELS, SEQ  # noqa: E402
from futures_ls import spot_name  # noqa: E402
from h3_fetch import ARCH_5M, H3, LISTING, PAIRS, SPOT_5M, split  # noqa: E402
from p1_lib import P5, Context, perp_only, spikes_and_coin_channels  # noqa: E402

PAIRS_5M = PAIRS / "klines" / "spot" / "interval=5m"
common.ARCHIVE_5M = PAIRS_5M  # the converted pairs; USDT symbols still come from the shared folder
START = datetime(2020, 1, 2, tzinfo=UTC)
END = datetime(2026, 10, 29, tzinfo=UTC)
P1_START = datetime(2022, 1, 1, tzinfo=UTC)
BPD = common.BARS_PER_DAY
T0 = build_seq2.T0
STEP = build_seq2.STEP


def pairs() -> list[str]:
    names = (p.name.removeprefix("symbol=") for p in PAIRS_5M.glob("symbol=*"))
    return sorted(n for n in names if split(n) is not None)


def usdt_blocks(base: str) -> list[tuple[datetime, datetime]]:
    """Time spans [lo, hi] in which XXXUSDT traded (store first..last candle; archive-only
    pairs: whole months)."""
    sym = f"{base}USDT"
    out = []
    lo: datetime | None = None
    hi: datetime | None = None
    for d in (SPOT_5M, ARCH_5M):
        p = d / f"symbol={sym}"
        if p.exists() and any(p.glob("*.parquet")):
            t = pl.read_parquet(p / "*.parquet", columns=["open_time"])["open_time"]
            a, b = t.min(), t.max()
            assert isinstance(a, datetime)
            assert isinstance(b, datetime)
            lo = a if lo is None else min(lo, a)
            hi = b if hi is None else max(hi, b)
    if lo is not None and hi is not None:
        out.append((lo, hi))
    extra = json.loads(LISTING.read_text())["usdt_extra"].get(sym, [])
    for m in extra:
        a = datetime(int(m[:4]), int(m[5:]), 1, tzinfo=UTC)
        b = datetime(a.year + a.month // 12, a.month % 12 + 1, 1, tzinfo=UTC)
        out.append((a, b))
    return out


def perp_spans() -> dict[str, tuple[datetime, datetime]]:
    """Coin (XXXUSDT) -> first..last perp5m candle of its perp-only perp (p1's universe)."""
    out: dict[str, tuple[datetime, datetime]] = {}
    for perp in perp_only():
        p = P5 / f"{perp}.parquet"
        if not p.exists():
            continue
        ms = pl.read_parquet(p, columns=["ms"])["ms"]
        lo = datetime.fromtimestamp(int(ms.min()) / 1000, tz=UTC)  # type: ignore[arg-type]
        hi = datetime.fromtimestamp(int(ms.max()) / 1000, tz=UTC)  # type: ignore[arg-type]
        coin = spot_name(perp)
        a, b = out.get(coin, (lo, hi))
        out[coin] = (min(a, lo), max(b, hi))
    return out


def job(pair: str) -> tuple[pl.DataFrame, np.ndarray, dict[str, int]] | None:
    """Spikes of one converted pair (h1_build.spike_job's selection), the coin-time filters and
    the 7 coin channels."""
    sp = split(pair)
    assert sp is not None
    base, quote = sp
    raw = common.load(pair)
    if raw.height < build_events.WARMUP + BPD:
        return None
    df = build_events.features(build_events.full_grid(raw)).with_row_index("i")
    n = df.height
    sel = df.filter(
        (pl.col("i") >= build_events.WARMUP)
        & (pl.col("i") < n - common.HORIZON - 1)
        & pl.col("volume").is_not_null()
        & pl.col("entry").is_not_null()
        & (pl.col("vol_base") > 0)
        & (pl.col("vol_mult") >= build_events.SPIKE_MULT)
        & (pl.col("open_time") >= START)
        & (pl.col("open_time") < END)
    )
    if sel.height == 0:
        return None
    idx = sel["i"].to_numpy()
    hi, lo, cl = (df[c].to_numpy() for c in ("high", "low", "close"))
    entry = sel["entry"].to_numpy()
    k = idx[:, None] + 1 + np.arange(common.HORIZON)[None, :]
    fwd = np.stack([hi[k] / entry[:, None] - 1, lo[k] / entry[:, None] - 1,
                    cl[k] / entry[:, None] - 1], axis=1)  # fmt: skip
    ok = ~np.isnan(fwd).any(axis=(1, 2))  # drop windows that cross an exchange gap
    sel = sel.filter(pl.Series(ok) & (pl.col("qv_base") >= build_seq2.LIQ))
    n_liq = sel.height
    t = pl.col("open_time")
    for a, b in usdt_blocks(base):
        sel = sel.filter((t < a) | (t > b))
    n_free = sel.height
    ps = PERP.get(f"{base}USDT")
    if ps is not None:
        sel = sel.filter(~((t >= max(ps[0], P1_START)) & (t <= ps[1])))
    counts = {"liquid": n_liq, "no_usdt": n_free, "kept": sel.height}
    if sel.height == 0:
        return pl.DataFrame(), np.zeros((0, 7, SEQ), np.float16), counts
    got = spikes_and_coin_channels(pair, raw, START, END)
    assert got is not None
    meta, coin = got
    keep = np.isin(meta["open_time"].to_numpy(), sel["open_time"].to_numpy())
    meta = meta.filter(pl.Series(keep))
    coin = coin[keep]
    assert meta.height == sel.height, pair
    extra = sel.select("open_time", "trades_base", "i")
    meta = meta.join(extra, on="open_time", how="left").with_columns(
        pair=pl.lit(pair), quote=pl.lit(quote), symbol=pl.lit(f"{base}USDT"),
        listed_days=(pl.col("i") / BPD).cast(pl.Float32),
    ).drop("i")  # fmt: skip
    return meta, coin, counts


PERP = perp_spans()


def main() -> None:
    ps = pairs()
    print(len(ps), "converted pairs", file=sys.stderr)
    metas, coins, cov = [], [], {}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for p, got in zip(ps, pool.map(job, ps), strict=True):
            if got is None:
                continue
            meta, coin, counts = got
            cov[p] = counts
            if meta.height:
                metas.append(meta)
                coins.append(coin)
    meta = pl.concat(metas, how="vertical_relaxed")
    coin = np.concatenate(coins)
    order = np.lexsort((meta["pair"].to_numpy(), meta["open_time"].dt.epoch("s").to_numpy()))
    meta = meta[order]
    coin = coin[order]
    n = meta.height
    print(n, "kept spikes on", meta["pair"].n_unique(), "pairs,", meta["symbol"].n_unique(),
          "coins", file=sys.stderr)  # fmt: skip
    (H3 / "coverage.json").write_text(json.dumps(cov, indent=1, sort_keys=True) + "\n")
    seq = np.lib.format.open_memmap(
        H3 / "seq.npy", mode="w+", dtype=np.float16, shape=(n, CHANNELS, SEQ)
    )
    ctx = Context()
    btc4 = np.zeros(n)
    for a in range(0, n, 50_000):
        part = np.zeros((min(50_000, n - a), CHANNELS, SEQ), np.float16)
        part[:, :7] = coin[a : a + 50_000]
        btc4[a : a + 50_000] = ctx.fill(part, meta["open_time"][a : a + 50_000])
        seq[a : a + 50_000] = part
    seq.flush()
    del seq, coin
    meta = meta.with_columns(
        btc_ret_4h=pl.Series(btc4),
        gidx=((pl.col("open_time") - pl.lit(T0)) / pl.lit(STEP)).cast(pl.Int64),
        hour=pl.col("open_time").dt.hour(),
    )
    gidx = meta["gidx"].to_numpy()
    pr = meta["pair"].to_numpy()
    f2 = H3 / "fwd2.npy"
    arr = np.lib.format.open_memmap(f2, mode="w+", dtype=np.float32, shape=(n, 3, build_seq2.FWD))
    del arr
    f72 = H3 / "fwd72.npy"
    arr = np.lib.format.open_memmap(f72, mode="w+", dtype=np.float32, shape=(n, 3, 72))
    del arr
    jobs2 = [(s, gidx[pr == s], np.flatnonzero(pr == s), str(f2)) for s in np.unique(pr)]
    jobs72 = [(s, gidx[pr == s], np.flatnonzero(pr == s), str(f72)) for s in np.unique(pr)]
    with ProcessPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(build_seq2.fwd_job, jobs2)) == n
        assert sum(pool.map(build_fwd72.job, jobs72)) == n
    meta.write_parquet(H3 / "meta.parquet")
    f = np.load(f72, mmap_mode="r")
    print("written", H3, "; fwd72 NaN share", float(np.isnan(f).mean()),
          "; check gidx span", timedelta(minutes=5) * int(gidx.min()), file=sys.stderr)  # fmt: skip


if __name__ == "__main__":
    main()
