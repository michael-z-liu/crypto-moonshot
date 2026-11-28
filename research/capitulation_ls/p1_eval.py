"""P1 (perp-only coins): the pre-registered sleeves L and C applied to the USD-M perps without a
spot market, and the 10-slot LMC book with the additions.

- L-perp: long when the fold's day5 pct >= 0.995 on a liquid spike with an active perp
  (`futures_ls.has_perp`), entry next 5m perp open, TP +20%, 72h time exit, isolated 1x, cost 0.3%
  round trip + funding, one position per coin.
- C-perp: short when ret_4h <= -15%, |BTC 4h| <= 1%, pct <= 0.5, qv_base >= 20k, active perp,
  24h hold, isolated 1x (liquidation at +100%), cost 0.3% + funding.

Every candidate is simulated (no slot limit, one position per coin), with the candidates below the
threshold simulated as if entered for the calibration (`diag.py`). The book is e25's `book` with
10 slots, L + M + C (x5's `lmc`) with and without the perp-only trades.

Writes results/explore/perp_only_tables.md (pasted into perp_only.md) and OWN/p1/trades_*.parquet.
Run: uv run --group research python p1_eval.py
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import polars as pl
from common_ls import meta, scores
from diag import annotate, calibration, scorecard, table, threshold_reach
from futures_ls import funding_paid, spot_name
from p1_lib import P1, perp_paths
from sim import PERP_STEP_END_MIN, Rule, exits

OUT = Path(__file__).resolve().parent / "results" / "explore"
KIND = {0: "time", 1: "stop", 2: "tp", 3: "liq", 4: "stop"}
RL = Rule(hold_min=4320, tp=0.2, slots=100_000, cost=0.003, liq_frac=None)
RC = Rule(hold_min=1440, slots=100_000, cost=0.003, sl=1.0, liq_frac=None)
YEARS = (2022, 2023, 2024, 2025, 2026)


def load() -> pl.DataFrame:
    sp = pl.read_parquet(P1 / "spikes.parquet")
    sc = pl.read_parquet(P1 / "scores.parquet").sort("row")
    assert sc.height == sp.height
    return sp.with_columns(pct=sc["pct"].cast(pl.Float64), fold=sc["fold"])


def sim(c: pl.DataFrame, side: int, rule: Rule) -> pl.DataFrame:
    """Every candidate of `c` (symbol, t, prio, ...) on its perp path, one position per coin."""
    parts = []
    for (sym,), g in c.group_by(["symbol"]):
        ok, p = perp_paths(str(sym), g["t"].to_numpy())
        if not ok.any():
            continue
        g = g.filter(pl.Series(ok))
        sd = np.full(g.height, side)
        ret, t_exit, kind = exits(p, sd, rule, PERP_STEP_END_MIN)
        hold = PERP_STEP_END_MIN[t_exit]
        last = np.minimum(t_exit, p.shape[2] - 1)
        advp = np.nan_to_num(-p[:, 1] if side > 0 else p[:, 0], nan=0.0)
        favp = np.nan_to_num(p[:, 0] if side > 0 else -p[:, 1], nan=0.0)
        adv = np.maximum.accumulate(advp, axis=1)[np.arange(g.height), last]
        mfe = np.maximum.accumulate(favp, axis=1)[np.arange(g.height), last]
        t_in = g["t"].to_numpy() + 300
        t_out = t_in + hold * 60
        fund = funding_paid(np.array([spot_name(str(sym))] * g.height), t_in, t_out, sd)
        net = np.maximum(rule.lev * (ret - rule.cost - fund), rule.loss_floor)
        parts.append(g.with_columns(
            t_in=pl.Series(t_in), t_out=pl.Series(t_out), net=pl.Series(net), ret=pl.Series(ret),
            fund=pl.Series(fund), kind=pl.Series(kind), adv=pl.Series(adv.astype(np.float64)),
            mfe=pl.Series(mfe.astype(np.float64)),
        ))  # fmt: skip
    if not parts:
        return pl.DataFrame()
    t = pl.concat(parts).sort(["t_in", "prio"], descending=[False, True])
    keep, busy = [], {}
    for r in t.select("symbol", "t_in", "t_out").iter_rows():
        if busy.get(r[0], -1) > r[1]:
            keep.append(False)
            continue
        busy[r[0]] = r[2]
        keep.append(True)
    return t.filter(pl.Series(keep)).with_columns(
        exit_kind=pl.col("kind").replace_strict(KIND, default="time")
    )


def frame(
    t: pl.DataFrame, sig: str, theta: float, tp: float | None, stop: float | None
) -> pl.DataFrame:
    return t.select(
        signal=pl.col(sig).cast(pl.Float64), threshold=pl.lit(theta), net="net", mfe="mfe",
        mae="adv", tp=pl.lit(tp, dtype=pl.Float64), stop=pl.lit(stop, dtype=pl.Float64),
        exit_kind="exit_kind", year="year",
    )  # fmt: skip


def by_year(t: pl.DataFrame) -> pl.DataFrame:
    return (
        t.group_by("year")
        .agg(
            n=pl.len(),
            coins=pl.col("symbol").n_unique(),
            days=pl.col("day").n_unique(),
            mean_net=pl.col("net").mean(),
            median_net=pl.col("net").median(),
            win=(pl.col("net") > 0).mean(),
            tp_rate=(pl.col("exit_kind") == "tp").mean(),
            fund=pl.col("fund").mean(),
        )
        .sort("year")
    )


def md(df: pl.DataFrame, digits: int = 3) -> str:
    cols = df.columns
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.iter_rows():
        out.append(
            "| "
            + " | ".join(f"{v:.{digits}f}" if isinstance(v, float) else str(v) for v in r)
            + " |"
        )
    return "\n".join(out)


def bins(t: pl.DataFrame, edges: list[float]) -> pl.DataFrame:
    """Outcome in fixed signal bins (as quoted for spot L in results/diag/README.md)."""
    lab = [f"{a:.4f}-{b:.4f}" for a, b in itertools.pairwise(edges)]
    a = annotate(t)
    a = a.with_columns(bin=pl.col("signal").cut(edges[1:-1], labels=lab, left_closed=True))
    return (
        a.group_by("bin")
        .agg(
            n=pl.len(),
            E_o=pl.col("o").mean(),
            win=(pl.col("o") > 0).mean(),
            p_tp=(pl.col("mode") == "tp_win").mean(),
        )
        .sort("bin")
    )


def main() -> None:
    from e25_combined import book, sleeve_crash_short, sleeve_listing_weak, sleeve_long

    d = load().with_columns(day=pl.col("open_time").dt.date())
    lines: list[str] = []
    # ---------------------------------------------------------------- coverage
    p5 = {s: pl.read_parquet(P1.parent / "futures" / "perp5m" / f"{s}.parquet", columns=["ms"])
          for s in d["symbol"].unique().to_list()}  # fmt: skip
    months = sum(
        int(pl.from_epoch(v["ms"], "ms").dt.strftime("%Y-%m").n_unique()) for v in p5.values()
    )
    cov = d.group_by("year").agg(
        spikes=pl.len(), coins=pl.col("symbol").n_unique(), perp_active=pl.col("perp").mean(),
        q995=(pl.col("pct") >= 0.995).sum(),
        q995_active=((pl.col("pct") >= 0.995) & pl.col("perp")).sum(),
        q995_days=pl.col("day").filter((pl.col("pct") >= 0.995) & pl.col("perp")).n_unique(),
    ).sort("year")  # fmt: skip
    # spot L flush days (raw candidates, 2022+)
    m = meta()
    pct_s = scores("day5")
    yr_s = m["year"].to_numpy()
    sel = (yr_s >= 2022) & np.isfinite(pct_s) & (pct_s >= 0.995)
    spot_days = m.filter(pl.Series(sel)).select("year", "day").unique()
    po_days = d.filter((pl.col("pct") >= 0.995) & pl.col("perp")).select("year", "day").unique()
    new_days = po_days.join(spot_days, on=["year", "day"], how="anti")
    fl = (
        spot_days.group_by("year").agg(spot_L_days=pl.len())
        .join(po_days.group_by("year").agg(perp_only_days=pl.len()), on="year", how="full", coalesce=True)  # noqa: E501
        .join(new_days.group_by("year").agg(new_days=pl.len()), on="year", how="full", coalesce=True)  # noqa: E501
        .fill_null(0).sort("year")
    )  # fmt: skip
    spot_L_n = m.filter(pl.Series(sel)).group_by("year").agg(spot_L_events=pl.len())
    fl = fl.join(spot_L_n, on="year", how="left")
    lines += ["## Coverage", "",
              f"{d['symbol'].n_unique()} perp-only coins with liquid spikes in 2022-01..2026-10, "
              f"{months} coin-months of perp 5m klines, {d.height:,} liquid spikes.", "",
              md(cov), "", "### Flush days (UTC days with an L candidate, pct >= 0.995)", "",
              md(fl), "",
              "Days with a perp-only L candidate that have no spot L candidate: "
              + ", ".join(str(x) for x in new_days.sort("day")["day"].to_list()), ""]  # fmt: skip

    # ---------------------------------------------------------------- L
    pool = d.filter(pl.col("perp") & (pl.col("pct") >= 0.95)).with_columns(prio=pl.col("pct"))
    ent = sim(pool.filter(pl.col("pct") >= 0.995), 1, RL)
    non = sim(pool.filter(pl.col("pct") < 0.995), 1, RL)
    ent.write_parquet(P1 / "trades_L.parquet")
    fe, fn = frame(ent, "pct", 0.995, 0.2, None), frame(non, "pct", 0.995, 0.2, None)
    by_day = ent.group_by("day").agg(n=pl.len(), e=pl.col("net").mean()).sort("n", descending=True)
    lines += ["## L on perp-only coins (perp long, 1x, cost 0.3% + funding)", "",
              md(by_year(ent)), "",
              f"All years: n {ent.height}, mean net {ent['net'].mean():+.4f}, "
              f"TP rate {(ent['exit_kind'] == 'tp').mean():.2f}; by entry day (top 10 days by trades):", "",  # noqa: E501
              md(by_day.head(10)), ""]  # fmt: skip
    # Robustness: trades are clustered by day, so read the day as the unit.
    new_set = set(new_days["day"].to_list())
    ent = ent.with_columns(new_day=pl.col("day").is_in(pl.Series(list(new_set), dtype=pl.Date)))
    dm = by_day["e"].to_numpy()
    rob = (
        ent.group_by("new_day")
        .agg(
            n=pl.len(),
            days=pl.col("day").n_unique(),
            mean_net=pl.col("net").mean(),
            tp_rate=(pl.col("exit_kind") == "tp").mean(),
        )
        .sort("new_day")
    )
    big = by_day["day"][0]
    rest = ent.filter(pl.col("day") != big)
    lines += ["Robustness (day = unit):", "",
              f"- {len(dm)} entry days; mean of day means {dm.mean():+.4f}, median {np.median(dm):+.4f}, "  # noqa: E501
              f"{(dm > 0).mean():.0%} of days positive; t-stat over days {dm.mean() / (dm.std(ddof=1) / np.sqrt(len(dm))):.1f}.",  # noqa: E501
              f"- Without the largest day ({big}): n {rest.height}, mean net {rest['net'].mean():+.4f}.",  # noqa: E501
              "- Trades on days with no spot L candidate (new_day) vs days shared with spot L:", "",
              md(rob), ""]  # fmt: skip
    # spot L reference with the same simulator semantics (from e25's sleeve)
    Ls = sleeve_long()
    Ls_y = Ls.with_columns(year=pl.from_epoch("t_in").dt.year()).group_by("year").agg(
        n=pl.len(), mean_net=pl.col("net").mean(), tp_rate=(pl.col("net") > 0.19).mean()
    ).sort("year")  # fmt: skip
    lines += ["Spot L for reference (e25 `sleeve_long`, spot, cost 0.4%):", "", md(Ls_y), ""]

    cards = {"L-perp-only": scorecard(annotate(fe))}
    tr = threshold_reach(fe, fn)
    cal = calibration(fe, fn)
    lines += ["### Diag scorecard and calibration (L)", "",
              f"Threshold 0.995: {tr['share_reaching']:.1%} of pct >= 0.95 candidates reach it; "
              f"Spearman(signal, outcome) {tr['spearman_signal_outcome']:+.3f}.", "",
              "Calibration by signal decile (equal-count bins over the pct >= 0.95 pool, label = lowest pct):", "",  # noqa: E501
              md(cal), "",
              "Same fixed bins as quoted for spot L (spot: ~0 / +0.026 / +0.059 / +0.132):", "",
              md(bins(pl.concat([fe, fn]), [0.95, 0.976, 0.984, 0.9955, 1.0001])), ""]  # fmt: skip

    # ---------------------------------------------------------------- C
    cc = d.filter(
        pl.col("perp") & (pl.col("qv_base") >= 20_000) & (pl.col("ret_4h") <= -0.15)
        & (pl.col("btc_ret_4h").abs() <= 0.01) & pl.col("pct").is_not_nan()
    ).with_columns(prio=-pl.col("ret_4h"), sig=1 - pl.col("pct"))  # fmt: skip
    ce = sim(cc.filter(pl.col("sig") >= 0.5), -1, RC)
    cn = sim(cc.filter(pl.col("sig") < 0.5), -1, RC)
    ce.write_parquet(P1 / "trades_C.parquet")
    fce, fcn = frame(ce, "sig", 0.5, None, 1.0), frame(cn, "sig", 0.5, None, 1.0)
    cards["C-perp-only"] = scorecard(annotate(fce))
    trc = threshold_reach(fce, fcn)
    lines += ["## C on perp-only coins (perp short, 1x, 24h, cost 0.3% + funding)", "",
              f"{cc.height} crash candidates (ret_4h <= -15%, BTC calm, active perp, qv_base >= 20k).", "",  # noqa: E501
              md(by_year(ce)), "",
              f"All years: n {ce.height}, mean net {ce['net'].mean() if ce.height else float('nan'):+.4f}; "  # noqa: E501
              f"below-threshold candidates (pct > 0.5) as if shorted: n {cn.height}, mean net "
              f"{cn['net'].mean() if cn.height else float('nan'):+.4f}. Spearman(signal, outcome) {trc['spearman_signal_outcome']:+.3f}.", "",  # noqa: E501
              "Calibration (signal = 1 - pct; equal-count bins):", "", md(calibration(fce, fcn, bins=5)), ""]  # fmt: skip              "Calibration (signal = 1 - pct; equal-count bins):", "", md(calibration(fce, fcn, bins=5)), ""]  # fmt: skip  # noqa: E501
    lines += ["### Scorecards (diag.py)", "", table(cards)]

    # ---------------------------------------------------------------- book
    cols = ["symbol", "adv", "t_in", "t_out", "net", "qv", "sleeve", "prio", "w"]
    base_parts = [sleeve_long(), sleeve_listing_weak(), sleeve_crash_short()]
    base = [p.with_columns(w=pl.lit(1.0)).select(cols) for p in base_parts]

    def add(t: pl.DataFrame, sleeve: str, prio: pl.Expr) -> pl.DataFrame:
        return t.select(
            "symbol", adv=pl.col("adv").clip(0, 1), t_in="t_in", t_out="t_out", net="net",
            qv=pl.col("qv_base") * 288, sleeve=pl.lit(sleeve), prio=prio, w=pl.lit(1.0),
        ).select(cols)  # fmt: skip

    xl = add(ent, "L", 3.0 + pl.col("pct"))
    xc = add(ce, "C", pl.lit(1.5))
    variants = {
        "LMC (base)": base,
        "LMC + L-perp-only": [*base, xl],
        "LMC + C-perp-only": [*base, xc],
        "LMC + both": [*base, xl, xc],
        "LMC + L-perp-only, new days only": [
            *base,
            add(ent.filter("new_day"), "L", 3.0 + pl.col("pct")),
        ],
        "LMC + L-perp-only, 2x cost": [
            *base,
            add(ent.with_columns(net=pl.col("net") - 0.003), "L", 3.0 + pl.col("pct")),
        ],
    }
    rows = []
    for name, parts in variants.items():
        b = book(pl.concat(parts, how="vertical_relaxed"), 10)
        n = int(b["trades"].sum())
        bp = float((b["bp"] * b["trades"]).sum() / n)
        yrs = {int(r["year"]): r for r in b.iter_rows(named=True)}
        row: dict[str, object] = {"book": name, "trades": n, "bp": round(bp)}
        for y in YEARS:
            r = yrs.get(y)
            row[str(y)] = (f"{r['ret'] * 100:+.0f}% (dd {r['dd'] * 100:.0f}, L{r['nL']} C{r['nC']})"
                           if r else "-")  # fmt: skip
        row["min yr"] = f"{min(float(r['ret']) for r in yrs.values()) * 100:+.0f}%"
        rows.append(row)
    lines += ["## 10-slot LMC book (e25 `book`, $10k each year)", "",
              md(pl.DataFrame(rows)), ""]  # fmt: skip
    (OUT / "perp_only_tables.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
