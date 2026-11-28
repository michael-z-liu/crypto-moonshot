# ruff: noqa: E501
"""X7a-2: second pumps on the F/P event definition (5m volume-spike rows of feats.parquet with
ret_4h >= 40%, perp active, qv_base >= 20k) and the d01 simulator (5m-then-hourly perp paths to 7
days; spot paths to 72 h). Splits the pool into repeat pumps (another pump of the coin 1-14 days
before) and first pumps, short (perp, isolated 1x, gate off/on) and long (spot). In-sample.

Pump history (for the "prior pump" flag): feats rows with ret_4h >= 0.4 or ret_24h >= 0.4 (any
liquidity) plus the hourly spot events of x7_second_pump (w 4 h / 24 h, thr 0.4).

Writes results/explore/second_pump_f_raw.md and x7_second_pump_f_grid.csv.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import polars as pl
from common_ls import OWN, meta
from diag import calibration, scorecard, table, threshold_reach
from sim import Rule, simulate
from x7_lib import T0, books, split_mean, yr_cells

OUT = Path(__file__).resolve().parent / "results" / "explore"
KIND = {0: "time", 1: "stop", 2: "tp", 3: "liq", 4: "stop"}
m = meta()
QVB = m["qv_base"].to_numpy()


def history() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """symbol -> (sorted pump times in epoch s, pump size)."""
    import x7_second_pump as sp

    f = pl.read_parquet(OWN / "feats.parquet").filter(
        (pl.col("ret_4h") >= 0.4) | (pl.col("ret_24h") >= 0.4)
    )
    parts = [f.select("symbol", t="t", r=pl.max_horizontal("ret_4h", "ret_24h"))]
    for w in (4, 24):
        ev = sp.events(w, 0.4)
        parts.append(ev.select("symbol", t=(pl.col("h") * 3600 + T0 + 3600).cast(pl.Int64), r="r"))
    h = pl.concat(parts, how="vertical_relaxed").sort("symbol", "t")
    return {
        str(s): (g["t"].to_numpy(), g["r"].to_numpy())
        for (s,), g in h.group_by(["symbol"], maintain_order=True)
    }


def pool(hist: dict[str, tuple[np.ndarray, np.ndarray]]) -> pl.DataFrame:
    d = pl.read_parquet(OWN / "feats.parquet").filter(
        pl.col("perp")
        & (pl.col("qv_base") >= 20_000)
        & (pl.col("year") >= 2021)
        & (pl.col("ret_4h") >= 0.4)
    )
    pm, gap = [], []
    for s, t in zip(d["symbol"], d["t"], strict=True):
        a = hist.get(s)
        if a is None:
            pm.append(0.0)
            gap.append(-1.0)
            continue
        tt, rr = a
        k = (tt <= t - 86400) & (tt >= t - 14 * 86400)
        pm.append(float(rr[k].max()) if k.any() else 0.0)
        gap.append(float((t - tt[k].max()) / 3600) if k.any() else -1.0)
    return d.with_columns(prior_max=pl.Series(pm), gap_h=pl.Series(gap)).with_columns(
        repeat=pl.col("prior_max") > 0
    )


def run(
    sub: pl.DataFrame, side: int, hold_h: int, tp: float | None, sl: float | None
) -> pl.DataFrame:
    if side < 0:
        rule = Rule(
            hold_min=hold_h * 60,
            tp=tp,
            sl=1.0 if sl is None else sl,
            slots=100_000,
            cost=0.003,
            liq_frac=None,
        )
        kw: dict[str, object] = {"perp_funding": True, "source": "perp"}
    else:
        rule = Rule(hold_min=hold_h * 60, tp=tp, sl=sl, slots=100_000, cost=0.004, liq_frac=None)
        kw = {}
    c = sub.select("row", side=pl.lit(side), prio="ret_4h")
    t = simulate(c, rule, start_equity=1e12, **kw).trades  # type: ignore[arg-type]
    t = t.join(sub.select("row", "prior_max", "gap_h", "repeat", "rate"), on="row", how="left")
    if side < 0 and sl is None:
        t = t.with_columns(kind=pl.when(pl.col("kind") == 1).then(3).otherwise(pl.col("kind")))
    return t.with_columns(
        year=pl.col("open_time").dt.year(),
        exit_kind=pl.col("kind").replace_strict(KIND, default="time"),
        mae=pl.col("adv"),
        t_in=pl.col("t") + 300,
        qv24=pl.col("row").map_elements(lambda r: float(QVB[r]) * 288, return_dtype=pl.Float64),
    )


def frame(t: pl.DataFrame, theta: float, tp: float | None, stop: float | None) -> pl.DataFrame:
    return t.select(
        signal="signal", threshold=pl.lit(theta), net="net", mfe="mfe", mae="adv",
        tp=pl.lit(tp, dtype=pl.Float64), stop=pl.lit(stop, dtype=pl.Float64),
        exit_kind="exit_kind", year="year",
    )  # fmt: skip


def main() -> None:
    P = pool(history())
    print("pool", P.height, "repeat rows", int(P["repeat"].sum()), flush=True)
    sets = {"all": P, "repeat": P.filter(pl.col("repeat")), "first": P.filter(~pl.col("repeat"))}
    gated = pl.col("rate").fill_null(0) >= 0
    grid: list[dict] = []
    tc: dict[tuple, pl.DataFrame] = {}
    specs: list[tuple[str, int, bool, int, str]] = []
    for s, g, h, ex in itertools.product(sets, (False, True), (24, 72, 168), ("liq", "tp30")):
        specs.append((s, -1, g, h, ex))
    for s, h, ex in itertools.product(sets, (24, 72), ("none", "tp30", "sl20")):
        specs.append((s, 1, False, h, ex))
    EXS = {"liq": (None, None), "tp30": (0.3, None), "none": (None, None), "sl20": (None, 0.2)}
    for s, side, g, h, ex in specs:
        tp, sl = EXS[ex]
        sub = sets[s].filter(gated) if g else sets[s]
        t = run(sub, side, h, tp, sl)
        tc[(s, side, g, h, ex)] = t
        stop = (1.0 if sl is None else sl) if side < 0 else sl
        c = scorecard(frame(t.with_columns(signal=pl.lit(1.0)), 0.0, tp, stop))
        n1, e1 = split_mean(t, 2022, 2024)
        n2, e2 = split_mean(t, 2025, 2026)
        grid.append({"set": s, "side": "short" if side < 0 else "long", "gate": g, "hold": h, "exit": ex,
                     "n": t.height, "mean": float(t["net"].mean()), "win": c["win"], "stopped": c["share_stopped"],
                     "contrib_stopped": c["contrib_stopped"], "giveback": c["giveback_med"],
                     "n22_24": n1, "e22_24": e1, "n25_26": n2, "e25_26": e2, "years": yr_cells(t)})  # fmt: skip
    g = pl.DataFrame(grid)
    g.drop("years").write_csv(OUT / "x7_second_pump_f_grid.csv")
    md = [
        "| set | side | gate | hold | exit | n | mean | win | stopped | contrib_stopped | giveback | e22_24 (n) | e25_26 (n) | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |",
        "|" + "---|" * 19,
    ]
    for r in g.iter_rows(named=True):
        md.append(
            f"| {r['set']} | {r['side']} | {'on' if r['gate'] else 'off'} | {r['hold']} | {r['exit']} | {r['n']} | {r['mean'] * 100:+.1f}% | "
            f"{r['win']:.2f} | {r['stopped']:.2f} | {r['contrib_stopped']:+.3f} | {r['giveback']:.2f} | {r['e22_24'] * 100:+.1f}% ({r['n22_24']}) | "
            f"{r['e25_26'] * 100:+.1f}% ({r['n25_26']}) | {r['years']} |"
        )
    # picks among repeat sets, by 2022-24
    cards: dict[str, dict[str, float]] = {}
    extra: list[str] = []
    for name, side, need_gate in (
        ("short", "short", False),
        ("short_gated", "short", True),
        ("long", "long", False),
    ):
        sub = g.filter(
            (pl.col("set") == "repeat")
            & (pl.col("side") == side)
            & (pl.col("n22_24") >= (10 if need_gate else 15))
        )
        if need_gate:
            sub = sub.filter(pl.col("gate"))
        p = sub.sort("e22_24", descending=True).row(0, named=True)
        sd = -1 if side == "short" else 1
        tp, sl = EXS[p["exit"]]
        stop = (1.0 if sl is None else sl) if sd < 0 else sl
        t = tc[("repeat", sd, p["gate"], p["hold"], p["exit"])]
        label = f"F-def repeat {side}: gate={'on' if p['gate'] else 'off'} hold {p['hold']}h {p['exit']}"
        cards[label] = scorecard(frame(t.with_columns(signal=pl.lit(1.0)), 0.0, tp, stop))
        tf = tc[("first", sd, p["gate"], p["hold"], p["exit"])]
        cards[label.replace("repeat", "first")] = scorecard(
            frame(tf.with_columns(signal=pl.lit(1.0)), 0.0, tp, stop)
        )
        # calibration: whole pool (same gate/exit), signal = rank of the prior pump size (0 = none)
        ta = tc[("all", sd, p["gate"], p["hold"], p["exit"])]
        ta = ta.with_columns(signal=pl.col("prior_max").rank("ordinal") / pl.len())
        theta = float(ta.filter(pl.col("repeat"))["signal"].min())
        fr = frame(ta, theta, tp, stop)
        e_, n_ = fr.filter(pl.col("signal") >= theta), fr.filter(pl.col("signal") < theta)
        cal = calibration(e_, n_)
        cal.write_csv(OUT / f"x7_calibration_second_pump_f_{name}.csv")
        tr = threshold_reach(e_, n_)
        extra += [f"### {label}", "", f"Pool: all F-def pumps with the same gate/exit (n={fr.height}); signal = rank of the largest prior pump "
                  f"(first pumps = 0, ties ordinal); threshold {theta:.3f} (= any prior pump); Spearman {tr['spearman_signal_outcome']:+.3f}.", "",
                  "| bin (signal >=) | n | E_o | win | p_tp | p_stopped | tp_progress | entered |", "|---|---|---|---|---|---|---|---|"]  # fmt: skip
        for r in cal.iter_rows(named=True):
            extra.append(
                f"| {r['bin']:.3f} | {r['n']} | {r['E_o']:+.3f} | {r['win']:.2f} | {r['p_tp']:.2f} | {r['p_stopped']:.2f} | {r['tp_progress']:.2f} | {r['entered']:.2f} |"
            )
        bd = t.with_columns(
            gap=pl.when(pl.col("gap_h") <= 72).then(pl.lit("1-3d")).otherwise(pl.lit("3-14d"))
        )
        agg = bd.group_by("gap").agg(n=pl.len(), mean=pl.col("net").mean()).sort("gap")
        extra += [
            "",
            "By gap: "
            + "; ".join(
                f"{r['gap']} n={r['n']} mean {r['mean'] * 100:+.1f}%"
                for r in agg.iter_rows(named=True)
            ),
            "",
        ]
        b = books(t.select("symbol", "mae", "t_in", "t_out", "net", "qv24"))
        extra += [
            f"Books: alone {b['alone']}",
            "",
            f"LMC + X: {b['lmc_plus']}",
            "",
            f"(LMC: {b['lmc']})",
            "",
        ]
    out = [
        "## Grid",
        "",
        *md,
        "",
        "## Picks (repeat sets, best e22_24 with n22_24 >= 15; gated short: n22_24 >= 10) and the same config on first pumps",
        "",
        table(cards),
        "",
        *extra,
    ]
    (OUT / "second_pump_f_raw.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
