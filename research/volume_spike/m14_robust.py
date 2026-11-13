"""M11: robustness of a chosen variant + rule (costs, latency, slots, concentration, placebo).

uv run python research/volume_spike/m14_robust.py <variant> '<Rule(...) kwargs as JSON>'
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace

import numpy as np
import polars as pl
from common import RESULTS
from m13_policy import COST, Ctx, Rule

YEARS = (2022, 2023, 2024, 2025, 2026)


def flat(yrs: list[dict[str, object]]) -> dict[str, object]:
    rows = [r for y in yrs for r in y["_rows"]]  # type: ignore[attr-defined]
    ret = np.array([r[2] for r in rows]) if rows else np.array([np.nan])
    by = {y["year"]: y for y in yrs}
    return {
        "trades": len(rows), "mean_bp": float(np.nanmean(ret) * 1e4),
        **{f"bp_{y}": by[y]["mean_bp"] for y in YEARS},
        **{f"cagr_{y}": by[y]["cagr"] for y in YEARS},
        "worst_dd": min(float(by[y]["max_dd"]) for y in YEARS),  # type: ignore[arg-type]
    }  # fmt: skip


def late_fwd(fwd: np.ndarray, k: int) -> np.ndarray:
    """Forward paths re-based to an entry k candles later (at that candle's close)."""
    base = 1 + fwd[:, 2:3, k - 1 : k]
    shifted = (1 + fwd[:, :, k:]) / base - 1
    pad = np.repeat(shifted[:, :, -1:], k, axis=2)
    return np.concatenate([shifted, pad], axis=2).astype(np.float32)


def main(variant: str, rule: Rule) -> None:
    ctx = Ctx(variant)
    base = ctx.evaluate(rule)
    out = [{"case": "as chosen", **flat(base)}]
    for c in (0.006, 0.010):
        ctx.cost = c
        out.append({"case": f"cost {c:.1%}", **flat(ctx.evaluate(rule))})
    ctx.cost = COST
    full = np.asarray(ctx.fwd, dtype=np.float32)
    for k in (1, 2):
        ctx.fwd = late_fwd(full, k)
        out.append({"case": f"enter {k} candle(s) late", **flat(ctx.evaluate(rule))})
    ctx.fwd = full
    for sl in (3, 10):
        out.append({"case": f"{sl} slots", **flat(ctx.evaluate(replace(rule, slots=sl)))})
    rows = [r for y in base for r in y["_rows"]]  # type: ignore[attr-defined]
    t = pl.DataFrame(rows, schema=["d", "pnl", "ret", "symbol"], orient="row")
    top_sym = (
        t.group_by("symbol").agg(pl.col("pnl").sum()).sort("pnl", descending=True)["symbol"][:3]
    )
    ctx.exclude = ctx.meta["symbol"].is_in(top_sym.to_list()).to_numpy()
    out.append(
        {"case": f"without the 3 best coins ({', '.join(top_sym)})", **flat(ctx.evaluate(rule))}
    )
    top_days = t.group_by("d").agg(pl.col("pnl").sum()).sort("pnl", descending=True)["d"][:5]
    ctx.exclude = ctx.meta["open_time"].dt.date().is_in(top_days.to_list()).to_numpy()
    out.append({"case": "without the 5 best days", **flat(ctx.evaluate(rule))})
    ctx.exclude[:] = False
    # placebo: shuffle scores among the same day's spikes (keeps which days are traded and how many)
    real = ctx.pct.copy()
    day = ctx.meta["open_time"].dt.date().to_physical().to_numpy()
    rng = np.random.default_rng(0)
    order = np.argsort(day, kind="stable")
    bounds = np.flatnonzero(np.diff(day[order])) + 1
    plac = []
    for _ in range(20):
        p = real.copy()
        for g in np.split(order, bounds):
            p[g] = real[rng.permutation(g)]
        ctx.pct = p
        plac.append(flat(ctx.evaluate(rule)))
    ctx.pct = real
    pm = np.array([x["mean_bp"] for x in plac])
    out.append({"case": "placebo: scores shuffled within each day (mean of 20)",
                "trades": int(np.mean([x["trades"] for x in plac])), "mean_bp": float(pm.mean()),
                **{f"bp_{y}": float(np.nanmean([x[f"bp_{y}"] for x in plac]))
                   for y in YEARS}})  # fmt: skip
    # bootstrap over trading days, all years pooled
    g = t.group_by("d").agg(pl.col("ret").sum(), pl.len())
    sums, cnts = g["ret"].to_numpy(), g["len"].to_numpy()
    boot = []
    for _ in range(2000):
        i = rng.integers(0, len(sums), len(sums))
        boot.append(sums[i].sum() / cnts[i].sum() * 1e4)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    df = pl.DataFrame(out, infer_schema_length=None)
    df.write_csv(RESULTS / f"m14_robust_{variant}.csv")
    with pl.Config(
        tbl_rows=30, tbl_cols=14, tbl_width_chars=260, float_precision=2, fmt_str_lengths=60
    ):
        print(df)
    print(f"placebo mean bp: {pm.mean():.0f} (range {pm.min():.0f}..{pm.max():.0f})")
    print(f"real {out[0]['mean_bp']:.0f}; bootstrap 95% CI of real: {lo:.0f}..{hi:.0f} bp")
    info = {"placebo": pm.tolist(), "ci": [lo, hi], "rule": rule.__dict__}
    (RESULTS / f"m14_robust_{variant}.json").write_text(json.dumps(info, default=str) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], Rule(**json.loads(sys.argv[2])) if len(sys.argv) > 2 else Rule())
