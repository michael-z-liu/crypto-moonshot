"""E25: one book for three sleeves sharing capital: capitulation longs (CNN), pump shorts (perp,
isolated 1x) and spot-listing shorts (perp). Each sleeve's trades are generated with its own exit
rule (all candidates, no slot limit), then one portfolio takes them in time order with K equal
slots of equity (each year from $10k), one position per coin, stake <= liquidity cap."""

from __future__ import annotations

import itertools
import sys

import numpy as np
import polars as pl
from common_ls import OWN, meta, scores
from sim import Rule, simulate

Y = (2022, 2023, 2024, 2025, 2026)
m = meta()


def sleeve_long(q: float = 0.995, tp: float = 0.2, hold: int = 4320, score: str = "day5") -> pl.DataFrame:
    pct = scores(score)
    yr = m["year"].to_numpy()
    idx = np.flatnonzero((yr >= 2022) & np.isfinite(pct) & (pct >= q))
    c = pl.DataFrame({"row": idx, "side": 1, "prio": pct[idx]})
    b = simulate(c, Rule(hold_min=hold, tp=tp, slots=100_000, cost=0.004, liq_frac=None), start_equity=1e12)
    t = b.trades
    return t.select("symbol", "adv", t_in=pl.col("t") + 300, t_out="t_out", net="net", qv=pl.col("row").map_elements(lambda r: float(m["qv_base"][r]) * 288, return_dtype=pl.Float64), sleeve=pl.lit("L"), prio=pl.lit(3.0) + pl.col("prio"))


def sleeve_pump(thr: float = 0.4, hold: int = 4320) -> pl.DataFrame:
    d = pl.read_parquet(OWN / "feats.parquet").filter(pl.col("perp") & (pl.col("qv_base") >= 20_000) & (pl.col("ret_4h") >= thr) & (pl.col("open_time").dt.year() >= 2022))
    c = d.select("row", side=pl.lit(-1), prio="ret_4h")
    b = simulate(c, Rule(hold_min=hold, slots=100_000, cost=0.003, sl=1.0, liq_frac=None), start_equity=1e12, perp_funding=True, source="perp")
    t = b.trades
    return t.select("symbol", "adv", t_in=pl.col("t") + 300, t_out="t_out", net="net", qv=pl.col("row").map_elements(lambda r: float(m["qv_base"][r]) * 288, return_dtype=pl.Float64), sleeve=pl.lit("P"), prio=pl.lit(2.0) + pl.col("prio").clip(0, 0.99))


def sleeve_listing() -> pl.DataFrame:
    import e13_listing_book as e13

    d = e13.trades(72, 30, 0.5, 0.5).filter(pl.col("year") >= 2022)
    return d.select("symbol", "adv", "t_in", "t_out", "net", "qv", sleeve=pl.lit("N"), prio=pl.lit(1.0))


def book(tr: pl.DataFrame, slots: int, gross: float = 1.0) -> pl.DataFrame:
    tr = tr.with_columns(year=pl.from_epoch("t_in").dt.year()).sort(["t_in", "prio"], descending=[False, True])
    res = []
    for (y,), g in tr.group_by(["year"], maintain_order=True):
        eq, cash, pos, nets, peak, dd, held = 10_000.0, 10_000.0 * gross, [], [], 10_000.0, 0.0, set()
        worst = 0.0
        by = {"L": [], "P": [], "N": []}
        for r in g.iter_rows(named=True):
            keep = []
            for p in pos:
                if p[0] <= r["t_in"]:
                    cash += p[1] * (1 + p[2]); eq += p[1] * p[2]; held.discard(p[3])
                    peak = max(peak, eq); dd = min(dd, eq / peak - 1)
                else:
                    keep.append(p)
            pos = keep
            if len(pos) >= slots or r["symbol"] in held:
                continue
            stake = min(gross * eq / slots, cash, r["qv"] * 0.005)
            if stake < 1:
                continue
            cash -= stake
            pos.append((r["t_out"], stake, r["net"], r["symbol"], r["adv"])); held.add(r["symbol"])
            worst = max(worst, sum(p[1] * p[4] for p in pos) / eq)
            nets.append(r["net"]); by[r["sleeve"]].append(r["net"])
        for p in sorted(pos):
            eq += p[1] * p[2]; peak = max(peak, eq); dd = min(dd, eq / peak - 1)
        res.append({"year": y, "trades": len(nets), "bp": float(np.mean(nets)) * 1e4, "ret": eq / 10_000 - 1, "dd": dd, "worst_mtm": worst,
                    **{f"n{k}": len(v) for k, v in by.items()}, **{f"bp{k}": float(np.mean(v)) * 1e4 if v else 0.0 for k, v in by.items()}})  # fmt: skip
    return pl.DataFrame(res).sort("year")


if __name__ == "__main__":
    L, P, N = sleeve_long(), sleeve_pump(), sleeve_listing()
    print("candidates:", L.height, P.height, N.height)
    pl.Config.set_tbl_width_chars(220); pl.Config.set_tbl_cols(20)
    for combo, slots, gross in itertools.product(("L", "N", "LN", "LPN"), (15, 20), (1.0, 1.5)):
        tr = pl.concat([{"L": L, "P": P, "N": N}[k] for k in combo], how="vertical_relaxed")
        b = book(tr, slots, gross)
        yrs = " | ".join(f"{r['year']}: {r['trades']:>3}t {r['bp']:+5.0f}bp {r['ret'] * 100:+5.0f}% dd{r['dd'] * 100:+4.0f} w{r['worst_mtm'] * 100:3.0f}" for r in b.iter_rows(named=True))
        n = int(b["trades"].sum())
        bp = float((b["bp"] * b["trades"]).sum() / n)
        print(f"{combo:3s} slots {slots:>2} gross {gross}: {bp:+5.0f}bp min {b['ret'].min() * 100:+4.0f}% | {yrs}")
