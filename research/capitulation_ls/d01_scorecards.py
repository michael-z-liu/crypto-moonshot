"""D01: diagnostic scorecards (diag.py) for the sleeves L, C, P, F and M, on every candidate trade
(no slot limit; one position per coin while held), with the candidates below each entry threshold
simulated as if entered, so the signal's calibration and the failure modes can be read.

Signals on a 0..1 scale: L = day5 percentile (theta 0.995); C = 1 - day5 percentile among
idiosyncratic crashes (theta 0.5); P = percentile of the 4h pump size among pumps >= 15% (theta =
the percentile of a 40% pump); F = P with the funding gate (entries need funding >= 0);
M = percentile of the fall since listing among spot listings (theta = no fall).

Writes results/diag/scorecards.md and results/diag/calibration_<sleeve>.csv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from common_ls import OWN, meta, scores
from diag import annotate, calibration, scorecard, table, threshold_reach
from sim import Rule, simulate

OUT = Path(__file__).resolve().parent / "results" / "diag"
OUT.mkdir(parents=True, exist_ok=True)
KIND = {0: "time", 1: "stop", 2: "tp", 3: "liq", 4: "stop"}
m = meta()
pct = scores("day5")
yr = m["year"].to_numpy()


def sim_trades(c: pl.DataFrame, rule: Rule, **kw: object) -> pl.DataFrame:
    t = simulate(c, rule, start_equity=1e12, **kw).trades  # type: ignore[arg-type]
    return t.with_columns(
        year=pl.col("open_time").dt.year(),
        exit_kind=pl.col("kind").replace_strict(KIND, default="time"),
    )


def frame(
    t: pl.DataFrame, sig: dict[int, float], theta: float, tp: float | None, stop: float | None
) -> pl.DataFrame:
    return t.select(
        signal=pl.col("row").replace_strict(sig, default=None).cast(pl.Float64),
        threshold=pl.lit(theta),
        net="net", mfe="mfe", mae="adv", tp=pl.lit(tp, dtype=pl.Float64), stop=pl.lit(stop, dtype=pl.Float64),  # noqa: E501
        exit_kind="exit_kind", year="year",
    )  # fmt: skip


cards: dict[str, dict[str, float]] = {}
notes: list[str] = []


def report(name: str, ent: pl.DataFrame, non: pl.DataFrame | None) -> None:
    cards[name] = scorecard(annotate(ent))
    cal = calibration(ent, non)
    cal.write_csv(OUT / f"calibration_{name}.csv")
    tr = threshold_reach(ent, non)
    notes.append(f"- **{name}**: threshold {tr['threshold']:.3f}; {tr['share_reaching']:.1%} of candidates reach it, "  # noqa: E501
                 f"{tr['share_within_0.1_below']:.1%} fall within 0.1 below; Spearman(signal, outcome) "  # noqa: E501
                 f"{tr['spearman_signal_outcome']:+.3f}")  # fmt: skip


# L: capitulation longs
RL = Rule(hold_min=4320, tp=0.2, slots=100_000, cost=0.004, liq_frac=None)
idx = np.flatnonzero((yr >= 2022) & np.isfinite(pct) & (pct >= 0.95))
sig = dict(zip(idx.tolist(), pct[idx].tolist(), strict=True))
ie, ino = idx[pct[idx] >= 0.995], idx[pct[idx] < 0.995]
fe = frame(
    sim_trades(pl.DataFrame({"row": ie, "side": 1, "prio": pct[ie]}), RL), sig, 0.995, 0.2, None
)
fn = frame(
    sim_trades(pl.DataFrame({"row": ino, "side": 1, "prio": pct[ino]}), RL), sig, 0.995, 0.2, None
)
report("L", fe, fn)

# P and F: pump shorts (perp, isolated 1x, 72h)
d = pl.read_parquet(OWN / "feats.parquet").filter(
    pl.col("perp")
    & (pl.col("qv_base") >= 20_000)
    & (pl.col("open_time").dt.year() >= 2022)
    & (pl.col("ret_4h") >= 0.15)
)
d = d.with_columns(sig=pl.col("ret_4h").rank() / pl.len())
theta_p = float(d.filter(pl.col("ret_4h") >= 0.4)["sig"].min())
RP = Rule(hold_min=4320, slots=100_000, cost=0.003, sl=1.0, liq_frac=None)
sig = dict(zip(d["row"].to_list(), d["sig"].to_list(), strict=True))


def pump(sub: pl.DataFrame) -> pl.DataFrame:
    t = sim_trades(
        sub.select("row", side=pl.lit(-1), prio="ret_4h"), RP, perp_funding=True, source="perp"
    )
    return frame(t, sig, theta_p, None, 1.0)


big = pl.col("sig") >= theta_p
report("P", pump(d.filter(big)), pump(d.filter(~big)))
okf = big & (pl.col("rate").fill_null(0) >= 0)
report("F", pump(d.filter(okf)), pump(d.filter(~okf)))

# C: idiosyncratic crash shorts (perp, isolated 1x, 24h)
dc = pl.read_parquet(OWN / "feats.parquet").filter(
    pl.col("perp")
    & (pl.col("qv_base") >= 20_000)
    & (pl.col("open_time").dt.year() >= 2022)
    & (pl.col("ret_4h") <= -0.15)
    & (pl.col("btc_ret_4h").abs() <= 0.01)
)
dc = dc.with_columns(sig=pl.Series(1 - pct[dc["row"].to_numpy()])).drop_nans("sig")
RC = Rule(hold_min=1440, slots=100_000, cost=0.003, sl=1.0, liq_frac=None)
sig = dict(zip(dc["row"].to_list(), dc["sig"].to_list(), strict=True))


def crash(sub: pl.DataFrame) -> pl.DataFrame:
    t = sim_trades(
        sub.select("row", side=pl.lit(-1), prio=-pl.col("ret_4h")),
        RC,
        perp_funding=True,
        source="perp",
    )
    return frame(t, sig, 0.5, None, 1.0)


report("C", crash(dc.filter(pl.col("sig") >= 0.5)), crash(dc.filter(pl.col("sig") < 0.5)))

# M: listing shorts (perp 1h paths from e13)
import e13_listing_book as e13  # noqa: E402

x = pl.read_parquet(OWN / "e27_listings.parquet")
mfe = []
for r in x.iter_rows(named=True):
    h, o, hi, lo, c, qv = e13.perp[r["symbol"]]
    i, k = np.searchsorted(h, r["t_in"]), np.searchsorted(h, r["t_out"] - 3600)
    mfe.append(float(-(lo[i : k + 1].min() / o[i] - 1)))
x = x.with_columns(mfe=pl.Series(mfe), sig=(-pl.col("since_list")).rank() / pl.len())
theta_m = float(x.filter(pl.col("since_list") <= 0)["sig"].min())
fm = x.select(
    signal="sig", threshold=pl.lit(theta_m), net="net", mfe="mfe", mae="adv", tp=pl.lit(0.5), stop=pl.lit(0.5),  # noqa: E501
    exit_kind=pl.when(pl.col("ret") >= 0.5).then(pl.lit("tp")).when(pl.col("ret") <= -0.5).then(pl.lit("stop")).otherwise(pl.lit("time")),  # noqa: E501
    year="year",
)  # fmt: skip
report("M", fm.filter(pl.col("signal") >= theta_m), fm.filter(pl.col("signal") < theta_m))

md = ["# Diagnostic scorecards (diag.py)", "", "Outcome o = net return on the stake, clipped to [-1, +1]. Shares are of trades; `contrib_*` is the",  # noqa: E501
      "mode's contribution to E_o (they sum to E_o). Every candidate trade, no slot limit, 2022-26.", "", table(cards), "", "## Thresholds", "", *notes]  # fmt: skip  # noqa: E501
(OUT / "scorecards.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
