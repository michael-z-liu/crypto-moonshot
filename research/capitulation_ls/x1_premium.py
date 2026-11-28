"""x1: perp premium extremes. Shorts after extreme premium, longs after extreme discount.

Signal (0..1): percentile of the hourly premium-index close within the coin's trailing 30 days
('own') or across all liquid coins at that hour ('xs'); short signal = pct, long signal = 1-pct.
Entry at the next hour's perp open, exit at hold end / take profit / stop / liquidation on hourly
bars (stop first when both hit in one bar). Costs 0.3% round trip + realised funding.

Run: uv run --group research python x1_fetch_premium.py
     uv run --group research python x1_premium.py
"""

from __future__ import annotations

import itertools
import sys

import numpy as np
import polars as pl
from common_ls import OWN, PERP_COST, RESULTS
from futures_ls import funding_cum, spot_name
from numpy.lib.stride_tricks import sliding_window_view

EX = OWN / "explore"
W = 720  # 30 days
NMAX = 72
MIN_VOL24 = 3e6
rng = np.random.default_rng(1)


# ---------------------------------------------------------------- data
def load() -> dict:
    pr = pl.read_parquet(EX / "premium_1h.parquet")
    syms = sorted(pr["symbol"].unique().to_list())
    pk = pl.read_parquet(OWN / "futures" / "perp_1h.parquet").filter(pl.col("symbol").is_in(syms))
    pk = pk.with_columns(h=pl.col("open_time").dt.epoch("s") // 3600)
    pr = pr.with_columns(h=pl.col("open_time").dt.epoch("s") // 3600)
    h0 = int(pk["h"].min())
    T = int(pk["h"].max()) - h0 + 1
    S = len(syms)
    idx = {s: i for i, s in enumerate(syms)}
    arr = {k: np.full((S, T), np.nan, dtype=np.float32) for k in ("o", "h", "l", "c", "qv", "p")}
    si = pk["symbol"].replace_strict(idx, return_dtype=pl.Int64).to_numpy()
    hi = pk["h"].to_numpy() - h0
    for k, col in (
        ("o", "open"),
        ("h", "high"),
        ("l", "low"),
        ("c", "close"),
        ("qv", "quote_volume"),
    ):
        arr[k][si, hi] = pk[col].to_numpy()
    si = pr["symbol"].replace_strict(idx, return_dtype=pl.Int64).to_numpy()
    hi = pr["h"].to_numpy() - h0
    ok = (hi >= 0) & (hi < T)
    arr["p"][si[ok], hi[ok]] = pr["p_close"].to_numpy()[ok]
    # cumulative funding at each hour (settlements with time <= hour)
    fc = funding_cum()
    cumf = np.zeros((S, T))
    hours = (h0 + np.arange(T)) * 3600
    for s, i in idx.items():
        c = fc.get(spot_name(s))
        if c is None:
            continue
        t, cs = c
        k = np.searchsorted(t, hours, side="right") - 1
        cumf[i] = np.where(k >= 0, cs[np.maximum(k, 0)], 0.0)
    qv = np.nan_to_num(arr["qv"])
    cq = np.cumsum(qv, axis=1)
    vol24 = cq - np.concatenate([np.zeros((S, 24)), cq[:, :-24]], axis=1)  # hours h-23..h
    return dict(syms=syms, h0=h0, T=T, cumf=cumf, vol24=vol24, **arr)


def own_pct(p: np.ndarray) -> np.ndarray:
    S, T = p.shape
    out = np.full((S, T), np.nan, dtype=np.float32)
    for s in range(S):
        x = p[s]
        if T <= W:
            continue
        win = sliding_window_view(x, W)  # win[i] = x[i:i+W]
        cur = x[W:]
        prev = win[: T - W]  # prev[j] = x[j:j+W] -> history of hour j+W
        less = (prev < cur[:, None]).sum(1) + 0.5 * (prev == cur[:, None]).sum(1)
        n = np.isfinite(prev).sum(1)
        v = np.where((n >= W // 2) & np.isfinite(cur), less / np.maximum(n, 1), np.nan)
        out[s, W:] = v
    return out


def xs_pct(p: np.ndarray, liquid: np.ndarray) -> np.ndarray:
    x = np.where(liquid & np.isfinite(p), p, np.inf)
    r = np.argsort(np.argsort(x, axis=0), axis=0).astype(np.float32)
    n = (np.isfinite(x)).sum(0)
    out = r / np.maximum(n - 1, 1)
    return np.where(np.isfinite(x) & (n >= 20), out, np.nan).astype(np.float32)


# ---------------------------------------------------------------- simulation
def gather(D: dict, s: np.ndarray, h: np.ndarray, n: int) -> tuple[np.ndarray, ...]:
    """Path arrays relative to the entry open (hour h+1): high, low, close, plus entry open."""
    cols = (h + 1)[:, None] + np.arange(n)[None, :]
    cols = np.minimum(cols, D["T"] - 1)
    o = D["o"][s, h + 1]
    hh = D["h"][s[:, None], cols] / o[:, None] - 1
    ll = D["l"][s[:, None], cols] / o[:, None] - 1
    cc = D["c"][s[:, None], cols] / o[:, None] - 1
    return hh, ll, cc, o


def exits(
    side: int,
    hh: np.ndarray,
    ll: np.ndarray,
    cc: np.ndarray,
    n: int,
    tp: float | None,
    stop: float | None,
) -> tuple[np.ndarray, ...]:
    """ret (price return on position), exit bar, kind code 0 time 1 tp 2 stop 3 liq, mfe, mae."""
    hh, ll, cc = hh[:, :n], ll[:, :n], cc[:, :n]
    if side == -1:
        fav, adv = -ll, hh
        cl = -cc
    else:
        fav, adv = hh, -ll
        cl = cc
    liq = 1.0  # isolated 1x: liquidated at a 100% adverse move (either side)
    sl = liq if stop is None else min(stop, liq)
    big = n + 5
    stop_hit = adv >= sl - 1e-12
    sb = np.where(stop_hit.any(1), stop_hit.argmax(1), big)
    if tp is not None:
        tph = fav >= tp
        tb = np.where(tph.any(1), tph.argmax(1), big)
    else:
        tb = np.full(len(hh), big)
    kb = np.minimum(sb, tb)
    kb = np.minimum(kb, n - 1)
    is_stop = (sb <= tb) & (sb < n)
    is_tp = (tb < sb) & (tb < n)
    ret = np.where(is_stop, -sl, np.where(is_tp, tp if tp is not None else 0.0, cl[:, n - 1]))
    kind = np.where(is_stop, np.where(sl >= 1.0 - 1e-12, 3, 2), np.where(is_tp, 1, 0))
    m = np.arange(n)[None, :] <= kb[:, None]
    mfe = np.maximum(np.where(m, fav, 0).max(1), 0)
    mae = np.maximum(np.where(m, adv, 0).max(1), 0)
    return ret, kb, kind, mfe, mae


KINDS = np.array(["time", "tp", "stop", "liq"])


def pick(key: np.ndarray, cool: int) -> np.ndarray:
    """Greedy: first event, then the next one at least `cool` hours later (per symbol; key=s*BIG+h)."""  # noqa: E501
    out = []
    i = 0
    while i < len(key):
        out.append(i)
        i = int(np.searchsorted(key, key[i] + cool, side="left"))
    return np.array(out, dtype=int)


class Ctx:
    def __init__(self, D: dict) -> None:
        self.D = D
        liquid = D["vol24"] >= MIN_VOL24
        D["liquid"] = liquid
        p = D["p"]
        self.sig = {"own": own_pct(p), "xs": xs_pct(p, liquid)}
        self.p = p
        self.BIG = D["T"] + 100

    def events(self, side: int, kind: str, thr: float, gate: float):
        """(s, h, signal) of hours with signal >= thr, liquid, valid next open, sorted by key."""
        D = self.D
        sg = self.sig[kind] if side == -1 else 1 - self.sig[kind]
        m = np.isfinite(sg) & (sg >= thr) & D["liquid"]
        m[:, -NMAX - 2 :] = False
        if gate > 0:
            m &= (self.p * (-side)) >= gate  # short: premium >= gate; long: premium <= -gate
        s, h = np.nonzero(m)
        k = s.astype(np.int64) * self.BIG + h
        o = np.argsort(k)
        s, h = s[o], h[o]
        ok = np.isfinite(D["o"][s, h + 1])
        s, h = s[ok], h[ok]
        return s, h, sg[s, h]

    def trades(self, cfg: dict, want_paths: bool = False) -> pl.DataFrame:
        side = cfg["side"]
        n = cfg["hold"]
        s, h, sg = self.events(side, cfg["sig"], cfg["thr"], cfg["gate"])
        key = s.astype(np.int64) * self.BIG + h
        sel = pick(key, n)
        s, h, sg = s[sel], h[sel], sg[sel]
        return self.eval_events(cfg, s, h, sg)

    def eval_events(self, cfg: dict, s: np.ndarray, h: np.ndarray, sg: np.ndarray) -> pl.DataFrame:
        D = self.D
        side, n = cfg["side"], cfg["hold"]
        hh, ll, cc, o = gather(D, s, h, n)
        good = np.isfinite(hh).all(1) & np.isfinite(ll).all(1) & np.isfinite(cc).all(1)
        s, h, sg, hh, ll, cc, o = (a[good] for a in (s, h, sg, hh, ll, cc, o))
        ret, kb, kind, mfe, mae = exits(side, hh, ll, cc, n, cfg["tp"], cfg["stop"])
        e = h + 1
        x = e + kb + 1
        x = np.minimum(x, D["T"] - 1)
        fund = side * (D["cumf"][s, x] - D["cumf"][s, e])
        net = np.maximum(-1.0, ret - PERP_COST - fund)
        t_in = (D["h0"] + e) * 3600
        t_out = (D["h0"] + x) * 3600
        yr = (t_in.astype("datetime64[s]").astype("datetime64[Y]").astype(int) + 1970).astype(int)
        return pl.DataFrame(
            {
                "strategy": "premium",
                "config": cfg_name(cfg),
                "symbol": np.array(D["syms"])[s],
                "side": np.full(len(s), side),
                "t_in": t_in,
                "t_out": t_out,
                "signal": sg.astype(float),
                "threshold": float(cfg["thr"]),
                "net": net,
                "mfe": mfe.astype(float),
                "mae": mae.astype(float),
                "tp": [cfg["tp"]] * len(s),
                "stop": [1.0 if cfg["stop"] is None else cfg["stop"]] * len(s),
                "exit_kind": KINDS[kind],
                "year": yr,
                "vol24": D["vol24"][s, h],
            },
            schema_overrides={"tp": pl.Float64},
        )


def cfg_name(c: dict) -> str:
    sd = "short_prem" if c["side"] == -1 else "long_disc"
    return f"{sd}|{c['sig']}>={c['thr']}|gate{c['gate']}|h{c['hold']}|tp{c['tp']}|stop{c['stop']}"


# ---------------------------------------------------------------- book
def book(tr: pl.DataFrame, year: int, slots: int = 10, eq0: float = 10_000.0) -> dict:
    t = tr.filter(pl.col("year") == year).sort(["t_in", "signal"], descending=[False, True])
    if t.height == 0:
        return dict(year=year, n=0, ret=0.0, dd=0.0)
    cash_eq = eq0
    active: list[tuple[int, float, float]] = []  # t_out, stake, pnl
    peak, dd = eq0, 0.0
    taken = 0
    for r in t.iter_rows(named=True):
        still = []
        for a in sorted(active):
            if a[0] <= r["t_in"]:
                cash_eq += a[2]
                peak = max(peak, cash_eq)
                dd = min(dd, cash_eq / peak - 1)
            else:
                still.append(a)
        active = still
        if len(active) >= slots:
            continue
        stake = min(cash_eq / slots, 0.005 * r["vol24"])
        active.append((r["t_out"], stake, stake * r["net"]))
        taken += 1
    for a in active:
        cash_eq += a[2]
    return dict(year=year, n=taken, ret=cash_eq / eq0 - 1, dd=dd)


def summarize(tr: pl.DataFrame) -> dict:
    if tr.height == 0:
        return dict(n=0, avg=np.nan)
    return dict(n=tr.height, avg=float(tr["net"].mean()), med=float(tr["net"].median()),
                win=float((tr["net"] > 0).mean()))  # fmt: skip


def main() -> None:
    D = load()
    print("grid", D["T"], len(D["syms"]), file=sys.stderr)
    C = Ctx(D)
    grid = list(
        itertools.product(
            (-1, 1), ("own", "xs"), (0.95, 0.98, 0.99, 0.995), (0.0, 0.001, 0.003),
            (8, 24, 72), (None, 0.05, 0.10), (None, 0.10),
        )
    )  # fmt: skip
    rows = []
    cache: dict = {}
    for side, sig, thr, gate, hold, tp, stop in grid:
        cfg = dict(side=side, sig=sig, thr=thr, gate=gate, hold=hold, tp=tp, stop=stop)
        tr = C.trades(cfg)
        cache[cfg_name(cfg)] = cfg
        if tr.height == 0:
            continue
        by = tr.group_by("year").agg(n=pl.len(), avg=pl.col("net").mean()).sort("year")
        row = dict(config=cfg_name(cfg), **cfg, n=tr.height, avg=float(tr["net"].mean()),
                   win=float((tr["net"] > 0).mean()))  # fmt: skip
        for r in by.iter_rows(named=True):
            row[f"n{r['year']}"] = r["n"]
            row[f"a{r['year']}"] = r["avg"]
        rows.append(row)
    g = pl.DataFrame(rows, infer_schema_length=None)
    g.write_csv(RESULTS / "explore" / "x1_grid.csv")
    print(
        g.sort("avg", descending=True).head(15).select(["config", "n", "avg", "win"]),
        file=sys.stderr,
    )
    print(
        g.group_by("side", "hold").agg(m=pl.col("avg").mean(), mx=pl.col("avg").max(), k=pl.len()),
        file=sys.stderr,
    )

    # walk-forward: pick best mean-net config (n>=60) on earlier years, trade the next year
    years = sorted({int(c[1:]) for c in g.columns if c.startswith("a2")})
    wf = []
    for ty in range(2022, 2027):
        best, bavg = None, -9.0
        for r in g.iter_rows(named=True):
            tot = sum(r.get(f"n{y}") or 0 for y in years if y < ty)
            if tot < 60:
                continue
            avg = sum((r.get(f"a{y}") or 0) * (r.get(f"n{y}") or 0) for y in years if y < ty) / tot
            if avg > bavg:
                best, bavg = r["config"], avg
        if best is None:
            continue
        tr = C.trades(cache[best])
        b = book(tr, ty)
        sub = tr.filter(pl.col("year") == ty)
        wf.append(dict(year=ty, config=best, train_avg=bavg, n=sub.height,
                       avg=float(sub["net"].mean()) if sub.height else np.nan, book_ret=b["ret"], book_n=b["n"], dd=b["dd"]))  # fmt: skip  # noqa: E501
    wfd = pl.DataFrame(wf)
    wfd.write_csv(RESULTS / "explore" / "x1_wf.csv")
    print(wfd, file=sys.stderr)

    # best in-sample config (n>=300) and baseline
    best = g.filter(pl.col("n") >= 300).sort("avg", descending=True).row(0, named=True)
    base = dict(side=-1, sig="own", thr=0.99, gate=0.0, hold=24, tp=None, stop=None)
    out = []
    bk = []
    for name, cfg in (("best", cache[best["config"]]), ("baseline", base)):
        tr = C.trades(cfg)
        out.append(tr)
        for y in range(2021, 2027):
            b = book(tr, y)
            bk.append(dict(which=name, config=cfg_name(cfg), **b))
    print(pl.DataFrame(bk), file=sys.stderr)
    pl.DataFrame(bk).write_csv(RESULTS / "explore" / "x1_books.csv")
    allt = pl.concat(out).drop("vol24")
    allt.write_parquet(EX / "premium_trades.parquet")
    # also best-config yearly trade stats, long and short separately for the table
    for name, tr in zip(("best", "baseline"), out, strict=True):
        print(name, tr.group_by("year").agg(n=pl.len(), avg=pl.col("net").mean(), win=(pl.col("net") > 0).mean(),  # noqa: E501
              mfe=pl.col("mfe").mean(), mae=pl.col("mae").mean()).sort("year"), file=sys.stderr)  # fmt: skip  # noqa: E501

    # non-entries for the best config: signal in [0.5, thr), liquid, as if entered
    cfg = cache[best["config"]]
    sdn = C.sig[cfg["sig"]] if cfg["side"] == -1 else 1 - C.sig[cfg["sig"]]
    m = np.isfinite(sdn) & (sdn >= 0.5) & (sdn < cfg["thr"]) & D["liquid"]
    m[:, -NMAX - 2 :] = False
    s, h = np.nonzero(m)
    pick_i = rng.choice(len(s), size=min(len(s), 25000), replace=False)
    s, h = s[pick_i], h[pick_i]
    ok = np.isfinite(D["o"][s, h + 1])
    s, h = s[ok], h[ok]
    ne = C.eval_events(cfg, s, h, sdn[s, h]).drop("vol24")
    ne = ne.sample(n=min(ne.height, 20000), seed=1)
    ne.write_parquet(EX / "premium_nonentries.parquet")
    print("nonentries", ne.height, "avg net", ne["net"].mean(), file=sys.stderr)


if __name__ == "__main__":
    main()
