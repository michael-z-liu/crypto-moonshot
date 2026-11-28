"""A1: red-team audit of the LMC headline book (results/why/audit.md).

Rebuilds the three LMC sleeves exactly as e25_combined does, but keeps each trade's seq2 row so
that its price path can be marked to market, replays e25's book() with the accepted trades and
their stakes recorded (checked against e25.book), and then measures what the headline numbers
leave out:

1. mark-to-market drawdown (hourly closes of every open position) vs the closed-trade drawdown
   that book() reports;
2. stake-weighted vs equal-weighted mean per trade (book() averages nets, whatever the stake);
3. trades that cross a year end (book() credits them to the entry year; dsr.daily_returns drops
   their P&L because their exit day is outside the entry year);
4. the deflated Sharpe ratio on mark-to-market daily returns instead of realised-P&L days, with
   the Sharpe-ratio variance hard-coded in e43 (0.48 annualised) and with one measured on a
   family of LMC-like books;
5. long trades whose path ends flat (delisting or halt: exit at the last traded price).

Run: uv run --group research python a1_audit.py
"""

from __future__ import annotations

import itertools

import e13_listing_book as e13
import e25_combined as e25
import numpy as np
import polars as pl
from common_ls import OWN, meta, scores
from dsr import daily_returns, dsr
from sim import PERP_STEP_END_MIN, STEP_END_MIN, Rule, paths, perp_paths, simulate

Y = (2022, 2023, 2024, 2025, 2026)
m = meta()
QV = m["qv_base"].to_numpy()
VAR_SR_E43 = (0.48 / np.sqrt(365)) ** 2


def _frame(t: pl.DataFrame, sleeve: str, prio: pl.Expr, side: int) -> pl.DataFrame:
    return t.select(
        "symbol",
        "adv",
        "row",
        t_in=pl.col("t") + 300,
        t_out="t_out",
        net="net",
        qv=pl.col("row").map_elements(lambda r: float(QV[r]) * 288, return_dtype=pl.Float64),
        sleeve=pl.lit(sleeve),
        prio=prio,
        side=pl.lit(side),
    )


def sleeve_L(
    q: float = 0.995, tp: float = 0.2, cost: float = 0.004, through: float = 0.0
) -> pl.DataFrame:
    """e25.sleeve_long; `through` > 0: the take profit fills at `tp` only once a bar's high has
    traded `through` beyond it (no fills at a touch)."""
    pct = scores("day5")
    yr = m["year"].to_numpy()
    idx = np.flatnonzero((yr >= 2022) & np.isfinite(pct) & (pct >= q))
    c = pl.DataFrame({"row": idx, "side": 1, "prio": pct[idx]})
    rule = Rule(hold_min=4320, tp=tp + through, slots=100_000, cost=cost, liq_frac=None)
    t = simulate(c, rule, 1e12).trades
    t = t.with_columns(
        net=pl.col("net") - pl.when(pl.col("kind") == 2).then(through).otherwise(0.0)
    )
    return _frame(t, "L", pl.lit(3.0) + pl.col("prio"), 1)


def sleeve_C() -> pl.DataFrame:
    d = pl.read_parquet(OWN / "feats.parquet").filter(
        pl.col("perp")
        & (pl.col("qv_base") >= 20_000)
        & (pl.col("open_time").dt.year() >= 2022)
        & (pl.col("ret_4h") <= -0.15)
        & (pl.col("btc_ret_4h").abs() <= 0.01)
    )
    pct = scores("day5")
    d = d.filter(pl.Series(pct[d["row"].to_numpy()] <= 0.5))
    b = simulate(
        d.select("row", side=pl.lit(-1), prio=-pl.col("ret_4h")),
        Rule(hold_min=1440, slots=100_000, cost=0.003, sl=1.0, liq_frac=None),
        start_equity=1e12,
        perp_funding=True,
        source="perp",
    )
    return _frame(b.trades, "C", pl.lit(1.5), -1)


def sleeve_M() -> pl.DataFrame:
    x = pl.read_parquet(OWN / "e27_listings.parquet").filter(
        (pl.col("year") >= 2022) & (pl.col("since_list") <= 0)
    )
    return x.select(
        "symbol", "adv", row=pl.lit(-1), t_in="t_in", t_out="t_out", net="net", qv="qv",
        sleeve=pl.lit("N"), prio=pl.lit(1.0), side=pl.lit(-1),
    )  # fmt: skip


def book_replay(tr: pl.DataFrame, slots: int = 10, start: float = 10_000.0) -> pl.DataFrame:
    """e25.book (gross 1, no caps / weights), returning the accepted trades with their stakes."""
    tr = tr.with_columns(year=pl.from_epoch("t_in").dt.year()).sort(
        ["t_in", "prio"], descending=[False, True]
    )
    acc = []
    for (_y,), g in tr.group_by(["year"], maintain_order=True):
        eq, cash, pos, held = start, start, [], set()
        for r in g.iter_rows(named=True):
            keep = []
            for p in pos:
                if p[0] <= r["t_in"]:
                    cash += p[1] * (1 + p[2])
                    eq += p[1] * p[2]
                    held.discard(p[3])
                else:
                    keep.append(p)
            pos = keep
            if len(pos) >= slots or r["symbol"] in held:
                continue
            stake = min(eq / slots, cash, r["qv"] * 0.005)
            if stake < 1:
                continue
            cash -= stake
            pos.append((r["t_out"], stake, r["net"], r["symbol"]))
            held.add(r["symbol"])
            acc.append({**r, "stake": stake, "eq_at_entry": eq})
    return pl.DataFrame(acc)


def marks(a: pl.DataFrame, adverse: bool = False) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per accepted trade: (times in epoch s, mark-to-market return on the stake), the return being
    side x close / entry - 1 minus the round-trip cost (what closing at that moment would give).
    `adverse`: mark at each step's worst price instead of its close (a pessimistic bound that
    assumes every open position sits at its step extreme at once)."""
    ch = 2
    out: list[tuple[np.ndarray, np.ndarray]] = [(np.zeros(0), np.zeros(0))] * a.height
    rows = a["row"].to_numpy()
    side = a["side"].to_numpy()
    sl = a["sleeve"].to_numpy()
    tin = a["t_in"].to_numpy()
    cost = np.where(sl == "L", 0.004, 0.003)
    iL = np.flatnonzero(sl == "L")
    pL = paths(rows[iL])
    for j, i in enumerate(iL):
        out[i] = (tin[i] + STEP_END_MIN * 60, side[i] * pL[j, 1 if adverse else ch] - cost[i])
    iC = np.flatnonzero(sl == "C")
    pC, ok = perp_paths(rows[iC])
    assert ok.all()
    for j, i in enumerate(iC):
        out[i] = (tin[i] + PERP_STEP_END_MIN * 60, side[i] * pC[j, 0 if adverse else ch] - cost[i])
    for i in np.flatnonzero(sl == "N"):
        h, o, hi, _lo, c, _qv = e13.perp[a["symbol"][int(i)]]
        c = hi if adverse else c
        k = int(np.searchsorted(h, tin[i]))
        kk = np.arange(k, min(k + 31 * 24 + 1, len(h)))
        out[i] = (h[kk] + 3600, -(c[kk] / o[k] - 1) - cost[i])
    return out


def mtm(
    a: pl.DataFrame, start: float = 10_000.0, adverse: bool = False
) -> tuple[pl.DataFrame, np.ndarray]:
    """Hourly mark-to-market equity per entry year; returns per-year stats and daily returns
    (calendar days of each year, 2026 to Nov 1, equity at each day's end)."""
    mk = marks(a, adverse)
    res, daily = [], []
    for y in Y:
        g = np.flatnonzero(a["year"].to_numpy() == y)
        t0 = int(np.datetime64(f"{y}-01-01", "s").astype(int))
        t_end = int(
            max(a["t_out"].to_numpy()[g].max(), np.datetime64(f"{y + 1}-01-01", "s").astype(int))
        )
        grid = np.arange(t0, t_end + 3600, 3600)
        eq = np.full(len(grid), start)
        for i in g:
            st, net = a["stake"][int(i)], a["net"][int(i)]
            ti, to = a["t_in"][int(i)], a["t_out"][int(i)]
            closed = grid >= to
            eq[closed] += st * net
            op = np.flatnonzero((grid >= ti) & (grid < to))
            tt, vv = mk[int(i)]
            k = np.searchsorted(tt, grid[op], side="right") - 1
            v = np.where(k >= 0, vv[np.maximum(k, 0)], -0.004)
            eq[op] += st * np.nan_to_num(v, nan=0.0)
        peak = np.maximum.accumulate(eq)
        dd = float((eq / peak - 1).min())
        # closed-trade equity on the same grid
        ce = np.full(len(grid), start)
        for i in g:
            ce[grid >= a["t_out"][int(i)]] += a["stake"][int(i)] * a["net"][int(i)]
        cdd = float((ce / np.maximum.accumulate(ce) - 1).min())
        res.append({"year": y, "ret": eq[-1] / start - 1, "dd_closed": cdd, "dd_mtm": dd,
                    "min_mtm_vs_start": float(eq.min() / start - 1)})  # fmt: skip
        last = (
            np.datetime64(f"{y + 1}-01-01", "s") if y < 2026 else np.datetime64("2026-11-02", "s")
        )
        ends = np.arange(t0 + 86400, int(last.astype(int)) + 1, 86400)
        e_d = np.concatenate([[start], eq[np.searchsorted(grid, ends) - 1]])
        daily.append(e_d[1:] / e_d[:-1] - 1)
    return pl.DataFrame(res), np.concatenate(daily)


def main() -> None:
    pl.Config.set_tbl_width_chars(220)
    pl.Config.set_tbl_cols(20)
    L, C, N = sleeve_L(), sleeve_C(), sleeve_M()
    tr = pl.concat([L, N, C], how="vertical_relaxed")
    ref = e25.book(tr.drop("row", "side"), 10)
    a = book_replay(tr)
    print("e25.book:", ref.select("year", "trades", "ret", "dd").to_dicts())
    # replay check: same trade count per year
    cnt = a.group_by("year").len().sort("year")["len"].to_list()
    assert cnt == ref["trades"].to_list(), (cnt, ref["trades"].to_list())
    n = a.height
    print(f"\n== 1. per-trade mean: equal-weighted {a['net'].mean() * 100:+.2f}% (n={n}),"
          f" stake-weighted {float((a['stake'] * a['net']).sum() / a['stake'].sum()) * 100:+.2f}%")  # fmt: skip  # noqa: E501
    print(a.group_by("sleeve").agg(
        n=pl.len(), eq_w=pl.col("net").mean() * 100,
        stake_w=(pl.col("stake") * pl.col("net")).sum() / pl.col("stake").sum() * 100,
        stake=pl.col("stake").mean(), capped=(pl.col("stake") < pl.col("eq_at_entry") / 10 - 1).mean(),  # noqa: E501
    ).sort("sleeve"))  # fmt: skip

    print("\n== 2. mark-to-market drawdown (hourly closes) vs closed-trade drawdown")
    st, r_mtm = mtm(a)
    st_adv, _ = mtm(a, adverse=True)
    print(st.join(ref.select("year", dd_book="dd"), on="year").join(
        st_adv.select("year", dd_adverse_bound="dd_mtm"), on="year"))  # fmt: skip

    print("\n== 3. trades crossing a year end")
    a = a.with_columns(
        exit_year=pl.from_epoch("t_out").dt.year(), pnl=pl.col("stake") * pl.col("net")
    )
    x = a.filter(pl.col("exit_year") > pl.col("year"))
    print(x.select("year", "sleeve", "symbol", "stake", "net", "pnl"))

    print("\n== 4. deflated Sharpe")
    tr_out = a.select("year", "t_out", "pnl", "sleeve").to_dicts()
    r_real = daily_returns(pl.DataFrame(tr_out), Y)
    for lab, r in (("realised (e43)", r_real), ("mark-to-market", r_mtm)):
        for nt in (1000, 5000):
            d = dsr(r, nt, VAR_SR_E43)
            print(f"{lab:16s} N={nt}: SR {d['sr_annual']:.2f} SR0 {d['sr0_annual']:.2f}"
                  f" skew {d['skew']:.1f} kurt {d['kurt']:.0f} DSR {d['dsr']:.3f}")  # fmt: skip
    # Sharpe variance on a family of LMC-like books (thresholds, take profits, slots), MTM
    srs = []
    fam = {}
    for q, tp in itertools.product((0.99, 0.995, 0.997), (0.15, 0.2, 0.3)):
        fam[(q, tp)] = sleeve_L(q, tp)
    for (q, tp), slots, combo in itertools.product(fam, (8, 10, 15), ("L", "LM", "LC", "LMC")):
        parts = [fam[(q, tp)]] + ([N] if "M" in combo else []) + ([C] if "C" in combo else [])
        aa = book_replay(pl.concat(parts, how="vertical_relaxed"), slots)
        _, rr = mtm(aa)
        srs.append(rr.mean() / rr.std(ddof=1))
    var_fam = float(np.var(srs, ddof=1))
    print(f"LMC family: {len(srs)} books, MTM annual Sharpe mean {np.mean(srs) * np.sqrt(365):.2f}"
          f" sd {np.std(srs, ddof=1) * np.sqrt(365):.2f}")  # fmt: skip
    for nt in (1000, 5000):
        d = dsr(r_mtm, nt, var_fam)
        print(f"MTM, family variance N={nt}: SR0 {d['sr0_annual']:.2f} DSR {d['dsr']:.3f}")

    print("DSR sensitivity, MTM daily returns (rows: Sharpe sd across trials, annualised):")
    for sd in (0.48, 0.75, 1.0):
        v = (sd / np.sqrt(365)) ** 2
        cells = [f"N={nt}: {dsr(r_mtm, nt, v)['dsr']:.3f}" for nt in (100, 1000, 5000, 20000)]
        print(f"  sd {sd:.2f}: " + "  ".join(cells))

    print("\n== 5. long trades whose last 24 hourly steps are flat (delisting / halt)")
    la = a.filter(pl.col("sleeve") == "L")
    p = paths(la["row"].to_numpy())
    rng = p[:, 0, -24:] - p[:, 1, -24:]
    flat = np.flatnonzero((np.abs(rng) < 1e-9).all(1))
    print(la[flat].select("year", "symbol", pl.from_epoch("t_in").alias("t_in"), "net", "stake"))

    print("\n== 6. take profit must trade through the level (no fills at a touch)")
    for th in (0.0, 0.002, 0.005, 0.01):
        aa = book_replay(pl.concat([sleeve_L(through=th), N, C], how="vertical_relaxed"))
        b = (
            aa.group_by("year")
            .agg(pnl=(pl.col("stake") * pl.col("net")).sum(), nl=(pl.col("sleeve") == "L").sum())
            .sort("year")
        )
        rets = (b["pnl"] / 10_000).to_list()
        nL = aa.filter(pl.col("sleeve") == "L")["net"].mean()
        print(f"through {th * 100:.1f}%: {aa['net'].mean() * 100:+.2f}% a trade (L {nL * 100:+.2f}%),"  # noqa: E501
              f" worst year {min(rets) * 100:+.0f}% | " + " ".join(f"{r * 100:+.0f}%" for r in rets))  # fmt: skip  # noqa: E501

    print("\n== 7. trades whose exit is after the price data ends (2026-10-25 spot store end)")
    end = int(np.datetime64("2026-10-25", "s").astype(int))
    print(a.filter(pl.col("t_out") > end).select("sleeve", "symbol", pl.from_epoch("t_in"), "net"))


if __name__ == "__main__":
    main()
