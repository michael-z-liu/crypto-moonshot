"""Chart data for the walk-forward CNN study (M9-M11), merged into report/data.json.

uv run python research/volume_spike/report_wf.py   # after report_data.py and m12..m15
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import polars as pl
from common import RESULTS
from m13_policy import Ctx, Rule
from report_data import OUT, build_html, rows

STRATS: dict[str, tuple[str, Rule]] = {
    "Official pick: 4h CNN, hold 4h, 3 slots": ("t4h5", Rule(q=0.98, hold=48, slots=3)),
    "Ensemble of all variants, 1h, 5 slots": ("ens_all", Rule(q=0.97)),
    "Base CNN, 1h, 5 slots": ("base", Rule()),
    "Path forecaster, 1h, 5 slots": ("path_all5", Rule()),
}
PEAK = {"q": 0.95, "rank": "peak", "min_peak": 0.5, "hold_from_forecast": True}
FORECAST_RULES: dict[str, Rule] = {
    "Rank by 1h forecast, hold 1h (default)": Rule(),
    "Rank by forecast peak, exit at peak time": Rule(**PEAK),
    "… + take profit at the p90 peak": Rule(**PEAK, tp_from="p90"),
    "… + stop at the p10 trough": Rule(**PEAK, tp_from="p90", sl_from_p10=True),
    "… the same, looking 8h ahead": Rule(**PEAK, tp_from="p90", sl_from_p10=True, horizon=96),
    "Need +2% forecast upside within 8h": Rule(
        q=0.98, rank="peak", min_peak=2.0, hold_from_forecast=True, horizon=96
    ),
}


def equity(yrs: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Chain the yearly simulations (each restarts at 1) into one curve, by exit day."""
    days: list[str] = []
    eq: list[float] = []
    level = 1.0
    for y in yrs:
        r = y["_rows"]
        if not r:
            continue
        t = pl.DataFrame(r, schema=["d", "pnl", "ret", "symbol"], orient="row")
        g = t.group_by("d").agg(pl.col("pnl").sum()).sort("d")
        cum = 1 + np.cumsum(g["pnl"].to_numpy())
        days += [str(d) for d in g["d"]]
        eq += (level * cum).round(4).tolist()
        level *= float(cum[-1])
    return {"day": days, "equity": eq}


def yearly(yrs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ("year", "trades", "mean_bp", "cagr", "max_dd", "win")
    return [{k: y.get(k) for k in keys} for y in yrs]


def describe(r: dict[str, Any]) -> str:
    s = f"top {100 - r['q'] * 100:.0f}%, hold {r['hold'] * 5} min, {r['slots']} slots"
    if r["sl"]:
        s += f", stop {r['sl']:.0%}"
    if r["gate"] == "flush":
        s += ", BTC-flush gate"
    return s


def main() -> None:
    data = json.loads(OUT.read_text())
    wf: dict[str, Any] = {}
    wf["variants"] = rows(pl.read_csv(RESULTS / "m12_variants.csv").drop("cfg"))
    wf["subsets"] = rows(pl.read_csv(RESULTS / "m12_ensemble_subsets.csv"))
    strats = {}
    for name, (variant, rule) in STRATS.items():
        yrs = Ctx(variant).evaluate(rule)
        strats[name] = {"variant": variant, "yearly": yearly(yrs), "equity": equity(yrs)}
    wf["strategies"] = strats
    tuned = []
    for f in sorted(RESULTS.glob("m13_policy_*.csv")):
        d = pl.read_csv(f, infer_schema_length=None)
        ok = d.filter(pl.col("select_worst_dd") >= -0.25).sort("select_cagr", descending=True)
        best, default = ok.row(0, named=True), d.row(0, named=True)
        tuned.append({
            "variant": d["variant"][0], "best_rule": describe(best),
            "best_select": best["select_cagr"], "best_late": best["late_cagr"],
            "default_select": default["select_cagr"], "default_late": default["late_cagr"],
        })  # fmt: skip
    wf["rule_tuning"] = tuned
    names = ("ens_all", "t4h5", "base")
    wf["robust"] = {v: rows(pl.read_csv(RESULTS / f"m14_robust_{v}.csv")) for v in names}
    wf["placebo"] = {v: json.loads((RESULTS / f"m14_robust_{v}.json").read_text()) for v in names}
    wf["forecast"] = json.loads((RESULTS / "m15_forecast_path_all5.json").read_text())
    ctx = Ctx("path_all5")
    wf["forecast_rules"] = [
        {"rule": n, "yearly": yearly(ctx.evaluate(r))} for n, r in FORECAST_RULES.items()
    ]
    data["wf"] = wf
    OUT.write_text(json.dumps(data, default=str, separators=(",", ":")))
    print(OUT, OUT.stat().st_size // 1024, "KB")
    build_html()


if __name__ == "__main__":
    main()
