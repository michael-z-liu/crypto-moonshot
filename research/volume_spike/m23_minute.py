"""B4: a trigger in the first minute of a spike, on 1-minute candles. Protocol: the plan, B4.

Part A (`speed`): the B2 pick (R2 capitulation, $50, +20% or 3 days, no stop) entered at the
first spike minute inside the 5m spike candle (the first minute with volume >= 10x the 1m median
of the previous 24 hours, else the last minute), versus the 5m entry; outcomes on 1m candles.
Part B (`quiet`): 1m spikes (volume >= 20x the 24h 1m median, an up minute, >= 5k USDT) on coins
quiet before (Q7 decile <= 2) or busy (>= 5); entry at the next minute versus when 5m data would
allow (the open after the 5m candle containing the spike minute); TP +5/10/20% or 15m/1h/4h/1d.

On a gap-free 1m grid (missing minutes carry the last close; after a series ends the last close
holds). Book: $10,000, $50 a trade, <= 200 open, one per coin, no compounding, 0.3% costs.

Run: MOONSHOT_UNIVERSE=all uv run python research/volume_spike/m23_minute.py speed|quiet|report
"""

from __future__ import annotations

import heapq
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
from common import CACHE, DATA_DIR, RESULTS

K1M = DATA_DIR / "klines" / "spot" / "interval=1m"
DAY = 1440
STAKE, CAP, COST = 50.0, 200, 0.003
LAST_ENTRY = datetime(2026, 10, 1, tzinfo=UTC)
CASCADE = (datetime(2025, 10, 9, tzinfo=UTC), datetime(2025, 10, 12, tzinfo=UTC))
MIN_NS = 60 * 10**9


# ----------------------------------------------------------------------------- 1m data
def load_1m(symbol: str) -> tuple[np.datetime64, dict[str, np.ndarray]] | None:
    files = sorted((K1M / f"symbol={symbol}").glob("*.parquet"))
    if not files:
        return None
    df = pl.concat([pl.read_parquet(f) for f in files]).sort("open_time").unique("open_time")
    df = df.sort("open_time")
    grid = pl.DataFrame(
        {
            "open_time": pl.datetime_range(
                df["open_time"][0], df["open_time"][-1], timedelta(minutes=1), eager=True
            )
        }
    )
    df = grid.join(df, on="open_time", how="left")
    c = df["close"].fill_null(strategy="forward").to_numpy().astype(np.float64)
    a = {"c": c}
    for k in ("open", "high", "low"):
        x = df[k].to_numpy().astype(np.float64)
        a[k[0]] = np.where(np.isnan(x), c, x)
    a["v"] = df["volume"].fill_null(0.0).to_numpy()
    a["qv"] = df["quote_volume"].fill_null(0.0).to_numpy()
    t0 = df["open_time"].dt.replace_time_zone(None).to_numpy()[0]
    return t0, a


def sparse_max(a: np.ndarray, levels: int) -> list[np.ndarray]:
    st = [a]
    for k in range(1, levels + 1):
        prev, h = st[-1], 1 << (k - 1)
        nxt = prev.copy()
        nxt[: len(prev) - h] = np.maximum(prev[: len(prev) - h], prev[h:])
        st.append(nxt)
    return st


def first_hit(st: list[np.ndarray], start: np.ndarray, thr: np.ndarray, span: int) -> np.ndarray:
    """First index in [start, start + span) with high >= thr, as an offset; -1 if none."""
    pos, end = start.copy(), start + span
    for k in range(len(st) - 2, -1, -1):
        step = 1 << k
        skip = (pos + step <= end) & (st[k][pos] < thr)
        pos = np.where(skip, pos + step, pos)
    hit = (pos < end) & (st[0][pos] >= thr)
    return np.where(hit, pos - start, -1).astype(np.int32)


def padded(a: dict[str, np.ndarray], n_pad: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    last = a["c"][-1]
    pad = np.full(n_pad, last)
    return (
        np.concatenate([a["h"], pad]),
        np.concatenate([a["c"], pad]),
        np.concatenate([a["o"], pad]),
    )


# ----------------------------------------------------------------------------- the book
def book(
    entry_ns: np.ndarray, exit_ns: np.ndarray, sym: np.ndarray, prio: np.ndarray
) -> np.ndarray:
    """Signals in entry order (ties by prio); capacity CAP, one position per coin."""
    order = np.lexsort((prio, entry_ns))
    taken = np.zeros(len(entry_ns), dtype=bool)
    heap: list[tuple[int, int]] = []
    held: set[int] = set()
    for j in order:
        t = entry_ns[j]
        while heap and heap[0][0] <= t:
            held.discard(heapq.heappop(heap)[1])
        if len(heap) >= CAP or sym[j] in held:
            continue
        heapq.heappush(heap, (int(exit_ns[j]), int(sym[j])))
        held.add(int(sym[j]))
        taken[j] = True
    return taken


def book_stats(
    d: pl.DataFrame, ret: str, entry: str, exit_: str, label: str, cost: float = COST
) -> list[dict[str, object]]:
    sym = d["symbol"].cast(pl.Categorical).to_physical().to_numpy()
    tk = book(d[entry].to_numpy(), d[exit_].to_numpy(), sym, d["prio"].to_numpy())
    r = d[ret].to_numpy() + COST - cost
    yr = d["year"].to_numpy()
    casc = d["cascade"].to_numpy()
    out = []
    for y in range(2020, 2027):
        m = tk & (yr == y)
        n = int(m.sum())
        out.append(
            {
                "rule": label,
                "year": y,
                "trades": n,
                "mean_pct": float(r[m].mean() * 100) if n else None,
                "win": float((r[m] > 0).mean()) if n else None,
                "pnl_usd": float(STAKE * r[m].sum()),
                "pnl_usd_ex_cascade": float(STAKE * r[m & ~casc].sum()),
                "worst_pct": float(r[m].min() * 100) if n else None,
            }
        )
    return out


# ----------------------------------------------------------------------------- part A
def speed_job(job: tuple[str, np.ndarray]) -> pl.DataFrame | None:
    symbol, t5 = job
    got = load_1m(symbol)
    if got is None:
        return None
    t0, a = got
    n = len(a["c"])
    i = ((t5 - t0) // np.timedelta64(1, "m")).astype(np.int64)  # first minute of the 5m candle
    ok = (i >= DAY) & (i + 5 < n)
    if not ok.any():
        return None
    t5, i = t5[ok], i[ok]
    vmed = pl.Series(a["v"]).shift(1).rolling_median(DAY, min_samples=DAY // 2).to_numpy()
    k = np.full(len(i), 4)
    for j in range(4, -1, -1):  # first minute in the candle with volume >= 10x the median
        hot = a["v"][i + j] >= 10 * np.nan_to_num(vmed[i + j], nan=np.inf)
        k = np.where(hot, j, k)
    hp, cp, op = padded(a, 3 * DAY + 2048)
    st = sparse_max(hp, 13)
    out: dict[str, object] = {"symbol": symbol, "t5": t5, "trigger_min": k}
    for tag, e in (("m1", i + k + 1), ("m5", i + 5)):
        ent = op[e]
        hit = first_hit(st, e, ent * 1.2, 3 * DAY)
        r = np.where(hit >= 0, 0.2, cp[e + 3 * DAY - 1] / ent - 1) - COST
        out[f"entry_{tag}"] = ent
        out[f"ret_{tag}"] = r
        out[f"exit_off_{tag}"] = np.where(hit >= 0, hit + 1, 3 * DAY)
        out[f"e_{tag}"] = e - i  # minutes from the 5m candle's open to entry
        out[f"r1h_{tag}"] = cp[e + 59] / ent - 1
    return pl.DataFrame(out)


def speed() -> None:
    from m18_longshot import RULES, load

    data = load().with_columns(vol_mult_neg=-pl.col("vol_mult"))
    mask, _ = RULES["R2 capitulation"]
    sig = data.filter(mask.fill_null(False)).select("symbol", "open_time", "ret_4h", "year")
    jobs = [
        (s, g["open_time"].dt.replace_time_zone(None).to_numpy())
        for (s,), g in sig.group_by("symbol")
    ]
    parts = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        for res in pool.map(speed_job, jobs):
            if res is not None:
                parts.append(res)
    r = pl.concat(parts).with_columns(pl.col("t5").dt.replace_time_zone("UTC"))
    r = r.join(sig.rename({"open_time": "t5"}), on=["symbol", "t5"], how="left")
    t5ns = r["t5"].dt.epoch("ns").to_numpy()
    r = r.with_columns(
        prio=pl.col("ret_4h"),
        cascade=pl.col("t5").is_between(*CASCADE, closed="left"),
        **{f"in_{t}": pl.Series(t5ns + r[f"e_{t}"].to_numpy() * MIN_NS) for t in ("m1", "m5")},
    ).with_columns(
        **{f"out_{t}": pl.col(f"in_{t}") + pl.col(f"exit_off_{t}") * MIN_NS for t in ("m1", "m5")}
    )
    r.write_parquet(CACHE / "m23_speed.parquet")
    rows = []
    for t, label in (("m5", "5m entry (as before)"), ("m1", "1m trigger entry")):
        rows += book_stats(r, f"ret_{t}", f"in_{t}", f"out_{t}", label)
        rows += book_stats(r, f"ret_{t}", f"in_{t}", f"out_{t}", label + ", 0.6% cost", 0.006)
    df = pl.DataFrame(rows)
    df.write_csv(RESULTS / "m23_speed.csv")
    print(
        r.select(
            pl.len(),
            pl.col("trigger_min").mean(),
            (pl.col("trigger_min") < 4).mean(),
            ((pl.col("entry_m1") / pl.col("entry_m5") - 1) * 100).median().alias("m1_vs_m5_pct"),
            (pl.col("r1h_m1") * 100).mean(),
            (pl.col("r1h_m5") * 100).mean(),
        )
    )
    with pl.Config(tbl_rows=40, float_precision=2):
        print(df.pivot(on="year", index="rule", values="pnl_usd"))


# ----------------------------------------------------------------------------- part B
HORIZONS = {"15m": 15, "1h": 60, "4h": 240, "1d": DAY}
TPS = (0.05, 0.10, 0.20)


def quiet_job(symbol: str) -> pl.DataFrame | None:
    got = load_1m(symbol)
    if got is None:
        return None
    t0, a = got
    n = len(a["c"])
    v = a["v"]
    vmed = pl.Series(v).shift(1).rolling_median(DAY, min_samples=DAY // 2).to_numpy()
    spike = (v >= 20 * np.nan_to_num(vmed, nan=np.inf)) & (vmed > 0) & (a["c"] > a["o"])
    spike &= a["qv"] >= 5_000
    i = np.flatnonzero(spike)
    i = i[(i >= DAY) & (i + 6 + DAY < n)]
    if len(i) == 0:
        return None
    minute = ((t0 - np.datetime64("1970-01-01T00:00")) // np.timedelta64(1, "m")).astype(np.int64)
    abs_m = minute + i
    hp, cp, op = padded(a, DAY + 2048)
    st = sparse_max(hp, 11)
    times = t0 + i.astype("timedelta64[m]")
    out: dict[str, object] = {
        "symbol": symbol,
        "t": times,
        "spike_pct": a["c"][i] / a["o"][i] - 1,
        "vol_mult": v[i] / vmed[i],
        "qv": a["qv"][i],
    }
    for tag, e in (("m1", i + 1), ("m5", i + (5 - abs_m % 5))):
        ent = op[e]
        out[f"lag_{tag}"] = e - i
        for tp in TPS:
            out[f"hit{round(tp * 100)}_{tag}"] = first_hit(st, e, ent * (1 + tp), DAY)
        for h, w in HORIZONS.items():
            out[f"c{h}_{tag}"] = cp[e + w - 1] / ent - 1
    return pl.DataFrame(out)


def quiet() -> None:
    from common import universe
    from m19_quiet import load as load_quiet

    parts = []
    with ProcessPoolExecutor(max_workers=6) as pool:
        for s, res in zip(universe(), pool.map(quiet_job, universe()), strict=True):
            if res is not None:
                parts.append(res)
            print(s, 0 if res is None else res.height, file=sys.stderr, flush=True)
    ev = pl.concat(parts).with_columns(pl.col("t").dt.replace_time_zone("UTC"))
    ev = ev.filter(pl.col("t") < LAST_ENTRY)
    q = (
        load_quiet()
        .select("symbol", "open_time", "q7_dec")
        .with_columns(snap_end=pl.col("open_time") + timedelta(minutes=5))
        .drop("open_time")
        .sort("snap_end")
    )
    ev = ev.sort("t").join_asof(
        q,
        left_on="t",
        right_on="snap_end",
        by="symbol",
        strategy="backward",
        tolerance=timedelta(hours=4),
    )
    ev = ev.filter(pl.col("q7_dec").is_not_null()).with_columns(
        year=pl.col("t").dt.year(),
        prio=-pl.col("vol_mult"),
        cascade=pl.col("t").is_between(*CASCADE, closed="left"),
    )
    ev.write_parquet(CACHE / "m23_quiet_events.parquet")
    print(ev.height, "1m spikes with a quiet measure", flush=True)
    rows = []
    for ctx, cmask in (("quiet", pl.col("q7_dec") <= 2), ("busy", pl.col("q7_dec") >= 5)):
        d0 = ev.filter(cmask)
        tns = d0["t"].dt.epoch("ns").to_numpy()
        for tag in ("m1", "m5"):
            lag = d0[f"lag_{tag}"].to_numpy()
            for tp in (*TPS, None):
                for h, w in HORIZONS.items():
                    if tp is None:
                        ret, off = d0[f"c{h}_{tag}"].to_numpy(), np.full(d0.height, w)
                    else:
                        hit = d0[f"hit{round(tp * 100)}_{tag}"].to_numpy()
                        took = (hit >= 0) & (hit < w)
                        ret = np.where(took, tp, d0[f"c{h}_{tag}"].to_numpy())
                        off = np.where(took, hit + 1, w)
                    d = d0.with_columns(
                        r=pl.Series(ret - COST),
                        tin=pl.Series(tns + lag * MIN_NS),
                        tout=pl.Series(tns + (lag + off) * MIN_NS),
                    )
                    label = f"{ctx} | {tag} | TP {'none' if tp is None else f'{tp:.0%}'} | {h}"
                    rows += book_stats(d, "r", "tin", "tout", label)
                    rows += book_stats(d, "r", "tin", "tout", label + " | 0.6% cost", 0.006)
            print(ctx, tag, "done", flush=True)
    df = pl.DataFrame(rows)
    df.write_csv(RESULTS / "m23_quiet.csv")


def report() -> None:
    d = pl.read_csv(RESULTS / "m23_quiet.csv")
    g = d.filter(pl.col("year") >= 2022).with_columns(
        per=pl.when(pl.col("year") <= 2024).then(pl.lit("a")).otherwise(pl.lit("b"))
    )
    s = (
        g.group_by("rule", "per")
        .agg(
            tr=pl.col("trades").sum()
            / pl.when(pl.col("per").first() == "a").then(3.0).otherwise(1.75),
            mean=(pl.col("mean_pct") * pl.col("trades")).sum() / pl.col("trades").sum(),
            pnl=pl.col("pnl_usd").sum()
            / pl.when(pl.col("per").first() == "a").then(3.0).otherwise(1.75),
        )
        .pivot(on="per", index="rule", values=["tr", "mean", "pnl"])
        .sort("pnl_a", descending=True)
    )
    with pl.Config(tbl_rows=80, tbl_width_chars=200, fmt_str_lengths=50, float_precision=1):
        print(s.filter(~pl.col("rule").str.contains("0.6%")).head(30))


if __name__ == "__main__":
    {"speed": speed, "quiet": quiet, "report": report}[sys.argv[1]]()
