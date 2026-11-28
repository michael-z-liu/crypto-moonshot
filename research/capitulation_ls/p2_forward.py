"""P2.2 (perp-only follow-ups): forward window 2026-11-02 .. data end for the perp-only coins.

Fetches the perp-only coins' November 2026 5m klines (data.binance.vision daily files; raw zips in
OWN/futures/raw_p2fwd/, deleted after parsing), joins them to the Sep-Oct candles of p1, finds the
liquid spikes from 2026-11-02 to the spot data end (as forward_signals.py: the latest BTC / ETH
candle), and scores them with the saved day5 2026 fold exactly as forward_signals.py does (market
channels from `paper.market_series` over the shared store). The perp activity gate is computed
from the fetched candles (>= 50k USDT in each of the last two closed hours, with a moving price).
L signals (pct >= 0.995, active perp) are followed on the perp to the latest candle.

Writes OWN/p2/forward_perp_only.parquet and results/explore/p2_forward.md.
Run: uv run --group research python p2_forward.py
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
os.environ["PAPER_DIR"] = str(ROOT / ".cache" / "paper_day5")
os.environ["PAPER_DATA_DIR"] = os.environ.get(
    "MOONSHOT_DATA_DIR", "/Users/mike/code/crypto-moonshot/data"
)
sys.path.insert(0, str(HERE.parent / "volume_spike"))

from concurrent.futures import ThreadPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import p1_lib  # noqa: E402
import paper  # noqa: E402
import polars as pl  # noqa: E402
from p1_lib import Context, perp_only, perp_paths  # noqa: E402
from p2_lib import P2, RL, fetch_csv, finish, md, parse_klines, score_coins, sim_perp  # noqa: E402

FUT = p1_lib.FUT
RAW = FUT / "raw_p2fwd"
FWD5 = P2 / "fwd5m"
START = datetime(2026, 11, 2, tzinfo=UTC)
DAYS = [f"2026-11-{d:02d}" for d in range(1, 28)]
OUTMD = HERE / "results" / "explore" / "p2_forward.md"


def fetch(sym: str) -> int:
    out = FWD5 / f"{sym}.parquet"
    if out.exists():
        return 0
    parts = []
    old = pl.read_parquet(p1_lib.P5 / f"{sym}.parquet")
    old = old.filter(pl.col("ms") >= int(datetime(2026, 9, 1, tzinfo=UTC).timestamp() * 1000))
    if old.height:
        parts.append(old)
    for d in DAYS:
        raw = fetch_csv(f"data/futures/um/daily/klines/{sym}/5m/{sym}-5m-{d}.zip", RAW)
        if raw is not None:
            k = parse_klines(raw)
            if k is not None:
                parts.append(k)
    if not parts:
        return 0
    df = finish(parts)
    df.write_parquet(out)
    return df.height


def gate(sym: str, t: np.ndarray) -> np.ndarray:
    """has_perp on the fetched candles: hourly quote volume (only hours whose price moved)."""
    p = pl.read_parquet(FWD5 / f"{sym}.parquet")
    h = (
        p.with_columns(h=(pl.col("ms") // 3_600_000) * 3600)
        .group_by("h")
        .agg(qv=pl.col("qv").sum(), hi=pl.col("high").max(), lo=pl.col("low").min())
        .sort("h")
        .with_columns(qv=pl.when(pl.col("hi") > pl.col("lo")).then("qv").otherwise(0.0))
    )
    hours, qv = h["h"].to_numpy(), h["qv"].to_numpy()
    out = np.zeros(len(t), dtype=bool)
    for i, ti in enumerate(t):
        k = np.searchsorted(hours, (ti // 3600) * 3600) - 1
        if k < 1 or hours[k] < ti - 2 * 3600:
            continue
        out[i] = qv[k] >= 50_000 and qv[k - 1] >= 50_000
    return out


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    FWD5.mkdir(parents=True, exist_ok=True)
    syms = perp_only()
    with ThreadPoolExecutor(16) as ex:
        n = sum(ex.map(fetch, syms))
    print("rows fetched/merged", n, file=sys.stderr)
    end = paper.data_end()
    market = paper.market_series(end)
    ctx = Context()
    ctx.cum = np.cumsum(market["alt_r"].to_numpy())
    ctx.share = market["spike_share"].to_numpy()
    p1_lib.P5 = FWD5  # load_perp / perp_paths read the merged Sep-Nov candles
    raws = {}
    for s in syms:
        f = FWD5 / f"{s}.parquet"
        if f.exists():
            df = p1_lib.load_perp(s)
            if df["open_time"].max() >= START:
                raws[s] = df
    sp = score_coins(raws, dict.fromkeys(raws, START), dict.fromkeys(raws, end), ctx,
                     fold=lambda y: "day5")  # fmt: skip
    perp_flag = np.zeros(sp.height, dtype=bool)
    sym_arr = sp["symbol"].to_numpy()
    t_arr = sp["t"].to_numpy()
    for s in np.unique(sym_arr):
        m = sym_arr == s
        perp_flag[m] = gate(str(s), t_arr[m])
    sp = sp.with_columns(perp=pl.Series(perp_flag))
    sp.write_parquet(P2 / "forward_perp_only.parquet")
    top = (
        sp.sort("pct", descending=True)
        .head(10)
        .select("symbol", "open_time", "pct", "ret_4h", "btc_ret_4h", "qv_base", "perp")
    )
    sig = sp.filter((pl.col("pct") >= 0.995) & pl.col("perp"))
    lines = ["### F2. Forward window 2026-11-02 .. data end, perp-only coins", "",
             f"Data end (latest BTC/ETH spot candle, as forward_signals.py): {end:%Y-%m-%d %H:%M} UTC. "  # noqa: E501
             f"Perp candles fetched to 2026-11-27 23:55. {len(raws)} perp-only coins traded in the window; "  # noqa: E501
             f"{sp.height:,} liquid spikes ({int(sp['perp'].sum()):,} with an active perp), scored with the saved day5 2026 fold.", "",  # noqa: E501
             f"pct >= 0.99: {int((sp['pct'] >= 0.99).sum())}; pct >= 0.995: {int((sp['pct'] >= 0.995).sum())}; "  # noqa: E501
             f"L signals (pct >= 0.995 and active perp): {sig.height}.", "",
             "Top 10 scores:", "", md(top, 4), ""]  # fmt: skip
    if sig.height:
        t = sim_perp(sig.with_columns(prio=pl.col("pct")), 1, RL, paths=perp_paths)
        last = int(pl.read_parquet(FWD5 / f"{sig['symbol'][0]}.parquet")["ms"].max()) // 1000 + 300
        t = t.with_columns(
            status=pl.when(pl.col("exit_kind") == "tp").then(pl.lit("closed: TP"))
            .when(pl.col("t_out") <= last).then(pl.lit("closed: 72h"))
            .otherwise(pl.lit("open (marked at the last candle)"))
        )  # fmt: skip
        lines += [
            "L signals and outcomes so far:",
            "",
            md(t.select("symbol", "open_time", "pct", "ret_4h", "net", "mfe", "adv", "status"), 4),
            "",
        ]
    else:
        lines += ["No L signal: the perp-only coins had no flush day in the window either.", ""]
    OUTMD.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
