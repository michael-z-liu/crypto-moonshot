"""Long shots (B2): many $50 trades, no stop, take profit +20..+100% or a 1-30 day time limit.

Book: $10,000, $50 per trade, at most 200 open positions, one per coin, signals in time order (ties
in one candle by priority), no compounding; 0.3% round trip. Outcomes from build_long30.py.
Protocol: docs/plans/cnn-replication.md, milestone B2 (pre-registered).

Run (MOONSHOT_UNIVERSE=all for the real study):
  uv run python research/volume_spike/m18_longshot.py rules
"""

from __future__ import annotations

import heapq
import sys
from datetime import UTC, datetime

import numpy as np
import polars as pl
from common import BARS_PER_DAY, CACHE, RESULTS

COST = 0.003
STAKE, CAPITAL, CAP = 50.0, 10_000.0, 200
TPS = (0.2, 0.3, 0.5, 1.0, None)
DAYS = (1, 3, 7, 30)
LAST_ENTRY = datetime(2026, 10, 1, tzinfo=UTC)
FIVE_MIN = np.timedelta64(5, "m").astype("timedelta64[ns]").astype(np.int64)


def load() -> pl.DataFrame:
    cols = [
        "row",
        "symbol",
        "open_time",
        "is_spike",
        "is_random",
        "qv_base",
        "vol_mult",
        "ret_bar",
        "ret_4h",
        "ret_24h",
        "dist_hi_7d",
        "btc_vol_mult",
    ]
    ev = pl.read_parquet(CACHE / "events.parquet", columns=cols)
    lo = pl.read_parquet(CACHE / "long30.parquet")
    return (
        ev.join(lo, on="row", how="inner")
        .filter(pl.col("open_time") < LAST_ENTRY)
        .with_columns(year=pl.col("open_time").dt.year())
    )


def exits(d: pl.DataFrame, tp: float | None, days: int) -> tuple[np.ndarray, np.ndarray]:
    """Net return and exit candle offset (from entry) for take profit `tp` or `days` limit."""
    n = days * BARS_PER_DAY
    close = d[f"c_{days}d"].to_numpy().astype(np.float64)
    if tp is None:
        return close - COST, np.full(d.height, n - 1)
    hit = d[f"hit_{round(tp * 100)}"].to_numpy()
    took = (hit >= 0) & (hit < n)
    return np.where(took, tp, close) - COST, np.where(took, hit, n - 1)


def book(d: pl.DataFrame, ret: np.ndarray, exit_bar: np.ndarray, prio: str | None) -> np.ndarray:
    """Which signals the $50 book takes (time order, capacity CAP, one position per coin)."""
    order = ["open_time"] + ([prio] if prio else [])
    idx = d.with_row_index("_j").sort(order)["_j"].to_numpy()
    t = d["open_time"].dt.epoch("ns").to_numpy()[idx]
    sym = (
        d["symbol"].to_physical().to_numpy()[idx]
        if d["symbol"].dtype == pl.Categorical
        else d["symbol"].cast(pl.Categorical).to_physical().to_numpy()[idx]
    )
    t_exit = t + FIVE_MIN * (exit_bar[idx] + 2)  # entry at t + 5m, out at the end of the exit bar
    taken = np.zeros(d.height, dtype=bool)
    heap: list[tuple[int, int]] = []
    held: set[int] = set()
    for j in range(len(idx)):
        tj = t[j] + FIVE_MIN
        while heap and heap[0][0] <= tj:
            held.discard(heapq.heappop(heap)[1])
        if len(heap) >= CAP or sym[j] in held:
            continue
        heapq.heappush(heap, (int(t_exit[j]), int(sym[j])))
        held.add(int(sym[j]))
        taken[idx[j]] = True
    return taken


RULES: dict[str, tuple[pl.Expr, str | None]] = {
    "R0 random": (pl.col("is_random"), None),
    "R1 every spike": (pl.col("is_spike") & (pl.col("qv_base") >= 1_000), "vol_mult_neg"),
    "R2 capitulation": (
        pl.col("is_spike")
        & (pl.col("qv_base") >= 1_000)
        & (pl.col("ret_4h") <= -0.10)
        & (pl.col("btc_vol_mult") >= 6),
        "ret_4h",
    ),
    "R3 deep crash": (
        pl.col("is_spike") & (pl.col("qv_base") >= 1_000) & (pl.col("ret_24h") <= -0.30),
        "ret_24h",
    ),
    "R4 breakout": (
        pl.col("is_spike")
        & (pl.col("qv_base") >= 1_000)
        & (pl.col("ret_bar") > 0)
        & (pl.col("dist_hi_7d") >= 0),
        "vol_mult_neg",
    ),
}


def run_book(d: pl.DataFrame, name: str, prio: str | None, years: range) -> list[dict[str, object]]:
    """All exits for one signal set: per entry year, the book's trades and every signal's mean."""
    rows = []
    yr = d["year"].to_numpy()
    for tp in TPS:
        for days in DAYS:
            ret, xb = exits(d, tp, days)
            taken = book(d, ret, xb, prio)
            pnl_t = d["open_time"].dt.epoch("ns").to_numpy() + FIVE_MIN * (xb + 2)
            # realised P&L curve of the book, by exit time, for the drawdown
            k = np.flatnonzero(taken)
            o = np.argsort(pnl_t[k])
            curve = np.cumsum(STAKE * ret[k][o])
            dd = float((curve - np.maximum.accumulate(np.maximum(curve, 0))).min()) if len(k) else 0
            day = d["open_time"].dt.date().cast(pl.String).to_numpy()
            cascade = np.isin(day, ["2025-10-10", "2025-10-11"])
            for y in years:
                m = taken & (yr == y)
                mx = m & ~cascade
                a = yr == y
                n = int(m.sum())
                won = ret[m] > (tp or 0) - COST - 1e-9 if tp else ret[m] > 0
                rows.append(
                    {
                        "entry": name,
                        "tp": tp or 0.0,
                        "days": days,
                        "year": y,
                        "signals": int(a.sum()),
                        "all_mean_pct": float(ret[a].mean() * 100) if a.any() else None,
                        "trades": n,
                        "hit_rate": float(won.mean()) if n else None,
                        "mean_pct": float(ret[m].mean() * 100) if n else None,
                        "pnl_usd": float(STAKE * ret[m].sum()),
                        "pnl_usd_ex1010": float(STAKE * ret[mx].sum()),
                        "trades_ex1010": int(mx.sum()),
                        "worst_pct": float(ret[m].min() * 100) if n else None,
                        "book_dd_usd": dd,
                    }
                )
        print(name, tp, "done", flush=True)
    return rows


def rules() -> None:
    data = load().with_columns(vol_mult_neg=-pl.col("vol_mult"))
    out = []
    for name, (mask, prio) in RULES.items():
        d = data.filter(mask.fill_null(False))
        print(name, d.height, "signals", flush=True)
        out += run_book(d, name, prio, range(2020, 2027))
    df = pl.DataFrame(out)
    df.write_csv(RESULTS / "m18_rules.csv")
    show(df)


def show(df: pl.DataFrame) -> None:
    g = df.filter(pl.col("year") >= 2022).with_columns(
        per=pl.when(pl.col("year") <= 2024).then(pl.lit("22-24")).otherwise(pl.lit("25-26"))
    )
    s = g.group_by("entry", "tp", "days", "per").agg(
        trades_yr=pl.col("trades").sum()
        / pl.when(pl.col("per").first() == "22-24").then(3.0).otherwise(1.75),
        mean_pct=(pl.col("mean_pct") * pl.col("trades")).sum() / pl.col("trades").sum(),
        pnl_pct_yr=pl.col("pnl_usd").sum()
        / CAPITAL
        * 100
        / pl.when(pl.col("per").first() == "22-24").then(3.0).otherwise(1.75),
        hit=(pl.col("hit_rate") * pl.col("trades")).sum() / pl.col("trades").sum() * 100,
        pnl_x1010=pl.col("pnl_usd_ex1010").sum()
        / CAPITAL
        * 100
        / pl.when(pl.col("per").first() == "22-24").then(3.0).otherwise(1.75),
    )
    w = s.pivot(
        on="per",
        index=["entry", "tp", "days"],
        values=["trades_yr", "hit", "mean_pct", "pnl_pct_yr", "pnl_x1010"],
    )
    with pl.Config(tbl_rows=200, tbl_cols=14, tbl_width_chars=250, float_precision=1):
        print(w.sort("entry", "tp", "days"))


# ----------------------------------------------------------------------------- CNN
CNN_TARGET = ("hit_30", 7)  # classify: +30% reached within 7 days


def cnn_run(seeds: int = 2) -> None:
    """Walk-forward classifier on the liquid-spike sequences (seq2 on this universe)."""
    import time

    import torch
    from m12_cnn_wf import (
        BATCH,
        BTC,
        COIN,
        DEVICE,
        ETH,
        MARKET,
        TRADES,
        WF,
        YEARS,
        Cfg,
        Data,
        make,
        predict,
    )
    from torch import nn

    cfg = Cfg(
        channels=COIN + TRADES + BTC + ETH + MARKET,
        seq=288,
        scalars=("liq", "age", "hour"),
        epochs=6,
    )
    data = Data(mmap=True)
    lo = pl.read_parquet(CACHE / "long30.parquet", columns=["row", CNN_TARGET[0]])
    hit = (
        data.meta.select("event")
        .join(lo, left_on="event", right_on="row", how="left")[CNN_TARGET[0]]
        .to_numpy()
    )
    y = ((hit >= 0) & (hit < CNN_TARGET[1] * BARS_PER_DAY)).astype(np.float32)
    print("base rate", y.mean(), flush=True)
    t = data.meta["open_time"]
    years = data.meta["year"].to_numpy()
    pct = np.full(len(y), np.nan, dtype=np.float32)
    logs = []
    for year in YEARS:
        cut_tr = datetime(year - 1, 7, 1, tzinfo=UTC)
        tr = np.flatnonzero((t < cut_tr).to_numpy())
        va = np.flatnonzero(((t >= cut_tr) & (t < datetime(year, 1, 1, tzinfo=UTC))).to_numpy())
        test = np.flatnonzero(years == year)
        ref = np.sort(tr[:: max(1, len(tr) // 50_000)])
        o_test, o_ref = [], []
        for seed in range(seeds):
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            net = make(cfg, sum(data.scal[k].shape[1] for k in cfg.scalars), 1).to(DEVICE)
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
            best, state, bad = -np.inf, None, 0
            for epoch in range(cfg.epochs):
                net.train()
                t0 = time.time()
                perm = rng.permutation(tr)
                for k in range(0, len(perm), BATCH):
                    b = np.sort(perm[k : k + BATCH])
                    if len(b) < 32:
                        continue
                    x, sc = data.batch(b, cfg)
                    loss = nn.functional.binary_cross_entropy_with_logits(
                        net(x, sc).squeeze(1), torch.from_numpy(y[b]).to(DEVICE)
                    )
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                pv = predict(net, data, cfg, va)[:, 0]
                # early stopping: hit rate of the top 2% of validation scores
                top = pv >= np.quantile(pv, 0.98)
                score = float(y[va][top].mean())
                logs.append(
                    {
                        "year": year,
                        "seed": seed,
                        "epoch": epoch,
                        "valid_top2_hit": score,
                        "valid_base": float(y[va].mean()),
                        "secs": time.time() - t0,
                    }
                )
                if score > best:
                    best, bad = score, 0
                    state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= 2:
                        break
            assert state is not None
            net.load_state_dict(state)
            o_test.append(predict(net, data, cfg, test)[:, 0])
            o_ref.append(predict(net, data, cfg, ref)[:, 0])
        s_test, s_ref = np.mean(o_test, 0), np.mean(o_ref, 0)
        pct[test] = np.searchsorted(np.sort(s_ref), s_test) / len(s_ref)
        print(year, "scored", len(test), logs[-1], flush=True)
    WF.mkdir(exist_ok=True)
    np.savez(WF / "longshot.npz", pct=pct, event=data.meta["event"].to_numpy())
    pl.DataFrame(logs).write_csv(WF / "longshot_log.csv")


def cnn_eval() -> None:
    from m12_cnn_wf import WF

    z = np.load(WF / "longshot.npz")
    sc = pl.DataFrame({"row": z["event"], "pct": z["pct"]}).drop_nans()
    data = load().join(sc, on="row", how="inner").with_columns(neg_pct=-pl.col("pct"))
    out = []
    out += run_book(data, "C0 all liquid spikes", "neg_pct", range(2022, 2027))
    for q in (0.95, 0.98, 0.99):
        d = data.filter(pl.col("pct") >= q)
        out += run_book(d, f"CNN top {(1 - q) * 100:g}%", "neg_pct", range(2022, 2027))
    df = pl.DataFrame(out)
    df.write_csv(RESULTS / "m18_cnn.csv")
    show(df)


if __name__ == "__main__":
    {"rules": rules, "cnn_run": cnn_run, "cnn_eval": cnn_eval}[sys.argv[1]]()
