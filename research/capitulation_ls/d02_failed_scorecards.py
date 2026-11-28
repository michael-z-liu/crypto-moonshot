"""D02: diagnostic scorecards (diag.py) for the variants that failed, next to the working baseline
each was compared with. Same method as d01_scorecards.py: entries (signal >= theta) and the
below-threshold candidates are simulated separately, one position per coin, no slot limit.

Writes results/diag/failed_scorecards.md and results/diag/calibration_<name>.csv.
Commentary is read from d02_commentary.md (written after reading the numbers).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from common_ls import OWN, meta, scores
from diag import annotate, calibration, scorecard, table, threshold_reach
from sim import Rule, simulate

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "diag"
OUT.mkdir(parents=True, exist_ok=True)
KIND = {0: "time", 1: "stop", 2: "tp", 3: "liq", 4: "stop"}
m = meta()
yr = m["year"].to_numpy()
cards: dict[str, dict[str, float]] = {}
notes: list[str] = []
ENTRY_DAYS: dict[str, pl.DataFrame] = {}


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


def report(name: str, ent: pl.DataFrame, non: pl.DataFrame | None) -> None:
    cards[name] = scorecard(annotate(ent))
    if non is not None and non.height:
        cards[name + "_below"] = scorecard(annotate(non))
    cal = calibration(ent, non)
    cal.write_csv(OUT / f"calibration_{name}.csv")
    tr = threshold_reach(ent, non)
    e = cal["E_o"].to_list()
    yy = annotate(ent).group_by("year").agg(n=pl.len(), e=pl.col("o").mean()).sort("year")
    peryear = " ".join(f"{r['year']}:{r['n']}/{r['e']:+.2f}" for r in yy.iter_rows(named=True))
    notes.append(
        f"- **{name}** (entries by year n/E_o: {peryear}): theta {tr['threshold']:.3f}; {tr['share_reaching']:.1%} of candidates reach it, "  # noqa: E501
        f"{tr['share_within_0.1_below']:.1%} within 0.1 below; Spearman {tr['spearman_signal_outcome']:+.3f}; "  # noqa: E501
        f"calibration E_o by decile: " + " ".join(f"{x:+.3f}" for x in e)
    )


def long_sleeve(
    name: str,
    pct: np.ndarray,
    theta: float,
    rule: Rule | None = None,
    stop: float | None = None,
    floor: float = 0.95,
) -> None:
    rule = rule or Rule(hold_min=4320, tp=0.2, slots=100_000, cost=0.004, liq_frac=None)
    idx = np.flatnonzero((yr >= 2022) & np.isfinite(pct) & (pct >= floor))
    sig = dict(zip(idx.tolist(), pct[idx].tolist(), strict=True))
    ie, ino = idx[pct[idx] >= theta], idx[pct[idx] < theta]
    raw = sim_trades(pl.DataFrame({"row": ie, "side": 1, "prio": pct[ie]}), rule)
    ENTRY_DAYS[name] = raw.select(day=pl.col("open_time").dt.date(), net="net")
    ent = frame(raw, sig, theta, 0.2, stop)
    non = frame(
        sim_trades(pl.DataFrame({"row": ino, "side": 1, "prio": pct[ino]}), rule), sig, theta, 0.2, stop  # noqa: E501
    ) if len(ino) else None  # fmt: skip
    report(name, ent, non)


# 1. CNN variants: tp20day and the architectures, vs day5 (theta 0.995, TP 20%, 72h)
long_sleeve("L_day5", scores("day5"), 0.995)
for nm in ("tp20day", "day_transformer", "day_tcn", "day_gru"):
    long_sleeve(f"L_{nm}", scores(nm), 0.995)

# 2. lower thresholds on day5
long_sleeve("L_day5_th099", scores("day5"), 0.99)
long_sleeve("L_day5_th098", scores("day5"), 0.98)

# 5. leverage on the capitulation longs (day5 >= 0.995): outcome on the stake, spot paths
for lev in (2, 3):
    liq = 1.0 / lev - 0.01
    long_sleeve(f"L_day5_lev{lev}", scores("day5"), 0.995,
                Rule(hold_min=4320, tp=0.2, slots=100_000, cost=0.004, lev=float(lev), liq_frac=None),  # noqa: E501
                stop=liq)  # fmt: skip

# 3. pump shorts: sl 0.5 vs none (perp, isolated 1x, 72h, r4 >= 0.4)
feats = pl.read_parquet(OWN / "feats.parquet")
d = feats.filter(
    pl.col("perp") & (pl.col("qv_base") >= 20_000) & (pl.col("open_time").dt.year() >= 2022)
    & (pl.col("ret_4h") >= 0.15)
).with_columns(sig=pl.col("ret_4h").rank() / pl.len())  # fmt: skip
theta_p = float(d.filter(pl.col("ret_4h") >= 0.4)["sig"].min())
psig = dict(zip(d["row"].to_list(), d["sig"].to_list(), strict=True))
big = pl.col("sig") >= theta_p
for tag, sl in (("P_nostop", 1.0), ("P_sl50", 0.5)):
    rule = Rule(hold_min=4320, slots=100_000, cost=0.003, sl=sl, liq_frac=None)

    def pump(sub: pl.DataFrame, rule: Rule = rule, sl: float = sl) -> pl.DataFrame:
        t = sim_trades(sub.select("row", side=pl.lit(-1), prio="ret_4h"), rule,
                       perp_funding=True, source="perp")  # fmt: skip
        return frame(t, psig, theta_p, None, sl)

    report(tag, pump(d.filter(big)), pump(d.filter(~big)))

# 4. crash shorts: |BTC 4h| <= 1% vs 2%
pct5 = scores("day5")
for tag, band in (("C_calm1", 0.01), ("C_calm2", 0.02)):
    dc = feats.filter(
        pl.col("perp") & (pl.col("qv_base") >= 20_000) & (pl.col("open_time").dt.year() >= 2022)
        & (pl.col("ret_4h") <= -0.15) & (pl.col("btc_ret_4h").abs() <= band)
    )  # fmt: skip
    dc = dc.with_columns(sig=pl.Series(1 - pct5[dc["row"].to_numpy()])).drop_nans("sig")
    csig = dict(zip(dc["row"].to_list(), dc["sig"].to_list(), strict=True))
    RC = Rule(hold_min=1440, slots=100_000, cost=0.003, sl=1.0, liq_frac=None)

    def crash(sub: pl.DataFrame, csig: dict[int, float] = csig, RC: Rule = RC) -> pl.DataFrame:
        t = sim_trades(sub.select("row", side=pl.lit(-1), prio=-pl.col("ret_4h")), RC,
                       perp_funding=True, source="perp")  # fmt: skip
        return frame(t, csig, 0.5, None, 1.0)

    report(tag, crash(dc.filter(pl.col("sig") >= 0.5)), crash(dc.filter(pl.col("sig") < 0.5)))

# 6. listing shorts
import e13_listing_book as e13  # noqa: E402
import e18_perp_listings as e18  # noqa: E402


def short_kind(ret: pl.Expr, tp: float, sl: float) -> pl.Expr:
    return (pl.when(ret >= tp - 1e-9).then(pl.lit("tp")).when(ret <= -sl + 1e-9).then(pl.lit("stop"))  # noqa: E501
            .otherwise(pl.lit("time")))  # fmt: skip


def path_stats(
    df: pl.DataFrame, book: dict[str, tuple]
) -> tuple[list[float], list[float], list[float]]:  # type: ignore[type-arg]
    mfe, adv, since = [], [], []
    for r in df.iter_rows(named=True):
        h, o, hi, lo, _c, _qv = book[r["symbol"]]
        i, k = np.searchsorted(h, r["t_in"]), np.searchsorted(h, r["t_out"] - 3600)
        mfe.append(float(-(lo[i : k + 1].min() / o[i] - 1)))
        adv.append(float(hi[i : k + 1].max() / o[i] - 1))
        since.append(float(o[i] / o[0] - 1))
    return mfe, adv, since


# 6a. M = weak spot listings (e27), baseline (as in d01)
x = pl.read_parquet(OWN / "e27_listings.parquet")
mfe = []
for r in x.iter_rows(named=True):
    h, o, hi, lo, c, qv = e13.perp[r["symbol"]]
    i, k = np.searchsorted(h, r["t_in"]), np.searchsorted(h, r["t_out"] - 3600)
    mfe.append(float(-(lo[i : k + 1].min() / o[i] - 1)))
x = x.with_columns(mfe=pl.Series(mfe), sig=(-pl.col("since_list")).rank() / pl.len())
theta_m = float(x.filter(pl.col("since_list") <= 0)["sig"].min())
fm = x.select(signal="sig", threshold=pl.lit(theta_m), net="net", mfe="mfe", mae="adv",
              tp=pl.lit(0.5), stop=pl.lit(0.5), exit_kind=short_kind(pl.col("ret"), 0.5, 0.5),
              year="year")  # fmt: skip
report("M_e27", fm.filter(pl.col("signal") >= theta_m), fm.filter(pl.col("signal") < theta_m))

# 6b. all new perps (e18 trades(72, 30, 0.5, 0.5)); signal = percentile of the fall since the
# perp's first candle (weak = at or below it, as e28)
t18 = e18.trades(72, 30, 0.5, 0.5)
mfe, adv, since = path_stats(t18, e18.perp)
t18 = t18.with_columns(mfe=pl.Series(mfe), adv2=pl.Series(adv), since=pl.Series(since))
t18 = t18.with_columns(sig=(-pl.col("since")).rank() / pl.len())
theta18 = float(t18.filter(pl.col("since") <= 0)["sig"].min())
f18 = t18.select(signal="sig", threshold=pl.lit(theta18), net="net", mfe="mfe", mae="adv2",
                 tp=pl.lit(0.5), stop=pl.lit(0.5), exit_kind=short_kind(pl.col("ret"), 0.5, 0.5),
                 year="year")  # fmt: skip
cards["E18_all_perps"] = scorecard(annotate(f18))
report(
    "E18_weak_perps",
    f18.filter(pl.col("signal") >= theta18),
    f18.filter(pl.col("signal") < theta18),
)

# 6c. young-coin weekly shorts (e44), 7d, liquidation at +100%; weak = at or below listing price
y = pl.read_parquet(OWN / "e44_young.parquet")
mfe, adv = [], []
for r in y.iter_rows(named=True):
    h, o, hi, lo, c, qv = e13.perp[r["symbol"]]
    i, k = np.searchsorted(h, r["t_in"]), np.searchsorted(h, r["t_out"] - 3600)
    mfe.append(float(-(lo[i : k + 1].min() / o[i] - 1)))
    adv.append(float(hi[i : k + 1].max() / o[i] - 1))
y = y.with_columns(mfe=pl.Series(mfe), adv=pl.Series(adv).clip(0, 1.0),
                   sig=(-pl.col("rel")).rank() / pl.len())  # fmt: skip
theta_y = float(y.filter(pl.col("rel") <= 0)["sig"].min())
fy = y.select(signal="sig", threshold=pl.lit(theta_y), net="net", mfe="mfe", mae="adv",
              tp=pl.lit(None, dtype=pl.Float64), stop=pl.lit(None, dtype=pl.Float64),
              exit_kind=pl.when(pl.col("ret") <= -1.0 + 1e-9).then(pl.lit("liq")).otherwise(pl.lit("time")),  # noqa: E501
              year="year")  # fmt: skip
cards["Y_young_all"] = scorecard(annotate(fy))
report(
    "Y_young_weak", fy.filter(pl.col("signal") >= theta_y), fy.filter(pl.col("signal") < theta_y)
)

# 7. quiet-breakout explorer (already in diag's schema)
qt = pl.read_parquet(OWN / "explore" / "quiet_trades.parquet")
qn = pl.read_parquet(OWN / "explore" / "quiet_nonentries.parquet")
for cfg in qt["config"].unique().sort().to_list():
    sub = qt.filter(pl.col("config") == cfg)
    non = qn.filter(pl.col("config") == cfg)
    report(
        "Q_"
        + cfg.split("_30d")[0].split("_7d")[0]
        + ("_" + cfg if not cfg.startswith("Q_") else ""),
        sub,
        non if non.height else None,
    ) if False else None
    nm = "quiet_" + cfg
    report(nm, sub, non if non.height else None)

cluster = ["", "## Leverage: clustering of the entries by day (net on the stake, entries only)", ""]
for k in ("L_day5", "L_day5_lev2", "L_day5_lev3"):
    g = ENTRY_DAYS[k].group_by("day").agg(n=pl.len(), e=pl.col("net").mean()).sort("e")
    cluster.append(
        f"- {k}: {g.height} entry days; {int((g['e'] <= -0.3).sum())} days average <= -0.30 per trade "  # noqa: E501
        f"(worst day {g['e'][0]:+.2f} over {g['n'][0]} trades); trades on the 11 busiest days "
        f"{g.sort('n', descending=True).head(11)['n'].sum() / g['n'].sum():.0%} of all"
    )
notes += cluster
md = ["# Scorecards of the failed variants (diag.py, d02_failed_scorecards.py)", "",
      "Same scales as `scorecards.md`: outcome o = net return on the stake (clipped to [-1, +1]), shares of trades,",  # noqa: E501
      "`contrib_*` sums to E_o. 2022-26 (quiet explorer: 2021-26), every candidate trade, no slot limit. Rows named `*_below` are the",  # noqa: E501
      "candidates under the threshold, simulated as if entered. Calibration files: `calibration_<name>.csv`.", "",  # noqa: E501
      table(cards), "", "## Thresholds and calibration", "", *notes, ""]  # fmt: skip
com = HERE / "d02_commentary.md"
if com.exists():
    md += ["", com.read_text()]
(OUT / "failed_scorecards.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
