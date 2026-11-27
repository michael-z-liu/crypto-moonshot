"""w8-A: funding-sign effect by pump size/year, bootstrap CI, 2024 anatomy, crowding vs funding cost."""  # noqa: E501

from dataclasses import replace

import numpy as np
import polars as pl
from sim import simulate_yearly
from w8_base import RULE, feats

pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_width_chars(250)
pl.Config.set_tbl_cols(30)
bp = 1e4
F = feats()
# unconstrained: every spike with ret_4h>=0.25 is a trade (slots huge), isolated 1x, 72h, perp funding  # noqa: E501
R = replace(RULE, slots=1000, liq_frac=None)
bins = [(0.25, 0.3), (0.3, 0.4), (0.4, 0.6), (0.6, 99)]
parts = []
for lo, hi in bins:
    c = F.filter((pl.col("ret_4h") >= lo) & (pl.col("ret_4h") < hi)).select(
        "row", pl.lit(-1).alias("side"), pl.col("ret_4h").alias("prio")
    )
    parts.append(
        simulate_yearly(
            c, R, perp_funding=True, source="perp", years=(2022, 2023, 2024, 2025, 2026)
        ).trades
    )
T = pl.concat(parts)
T = T.join(
    F.select(
        "row",
        "ret_4h",
        "rate",
        "rate24",
        "listed_days",
        "btc_r24h",
        "mkt_r4h",
        "mkt_r24h",
        "r7d",
        "r30d",
        "ls_acc",
        "top_ls_acc",
        "oi_ch4h",
        "oi_ch24h",
        "oi_to_vol",
        "taker_ls",
        "day",
    ),
    on="row",
)
T = T.with_columns(
    sgn=pl.when(pl.col("rate") >= 0).then(pl.lit("f>=0")).otherwise(pl.lit("f<0")),
    size=pl.when(pl.col("ret_4h") < 0.3)
    .then(pl.lit("a .25-.3"))
    .when(pl.col("ret_4h") < 0.4)
    .then(pl.lit("b .3-.4"))
    .when(pl.col("ret_4h") < 0.6)
    .then(pl.lit("c .4-.6"))
    .otherwise(pl.lit("d >=.6")),
    sq100=pl.col("adv") >= 1,
    sq50=pl.col("adv") >= 0.5,
    gross=pl.col("net") + pl.col("fund"),
)
print(
    "unconstrained n",
    T.height,
    "mean bp",
    T["net"].mean() * bp,
    "check kinds",
    T["kind"].unique().to_list(),
)
T.write_parquet("/tmp/w8_T.parquet")


def agg(g):
    return g.agg(
        n=pl.len(),
        net=pl.col("net").mean() * bp,
        med=pl.col("net").median() * bp,
        gross=pl.col("gross").mean() * bp,
        fund=pl.col("fund").mean() * bp,
        sq50=pl.col("sq50").mean().round(3),
        liq=pl.col("sq100").mean().round(3),
        rate_bp=pl.col("rate").mean() * bp,
        win=(pl.col("net") > 0).mean().round(3),
    )


print("\n== 1. size x sign (2022-26, unconstrained)")
print(agg(T.group_by("size", "sgn")).sort("size", "sgn"))
print("\n== 1b. size x sign, excluding 2025-10-10 day")
print(
    agg(T.filter(pl.col("day").cast(pl.Utf8) != "2025-10-10").group_by("size", "sgn")).sort(
        "size", "sgn"
    )
)
print("\n== 2. year x sign, by size group (small<.4, big>=.4)")
T = T.with_columns(
    sz2=pl.when(pl.col("ret_4h") < 0.4).then(pl.lit("small .25-.4")).otherwise(pl.lit("big >=.4"))
)
print(agg(T.group_by("sz2", "year", "sgn")).sort("sz2", "year", "sgn"))
# bootstrap diff, cluster by day
rng = np.random.default_rng(8)


def boot(d, B=4000):
    days = d["day"].cast(pl.Utf8).to_numpy()
    u, inv = np.unique(days, return_inverse=True)
    pos = d["rate"].to_numpy() >= 0
    x = d["net"].to_numpy() * bp
    sp = np.bincount(inv, weights=x * pos, minlength=len(u))
    np_ = np.bincount(inv, weights=pos, minlength=len(u))
    sn = np.bincount(inv, weights=x * ~pos, minlength=len(u))
    nn = np.bincount(inv, weights=~pos, minlength=len(u))
    if pos.sum() < 3 or (~pos).sum() < 3:
        return None
    est = sp.sum() / np_.sum() - sn.sum() / nn.sum()
    out = []
    for _ in range(B):
        k = rng.integers(0, len(u), len(u))
        a, b, cc, dd = np_[k].sum(), nn[k].sum(), sp[k].sum(), sn[k].sum()
        if a > 0 and b > 0:
            out.append(cc / a - dd / b)
    lo, hi = np.percentile(out, [2.5, 97.5])
    return est, lo, hi, (np.array(out) <= 0).mean(), int(pos.sum()), int((~pos).sum())


print("\n== 3. bootstrap diff (f>=0 minus f<0), bp/trade, day-cluster, 95% CI, P(diff<=0), n+/n-")
rows = []
for name, flt in [
    ("all .25+", pl.lit(True)),
    ("small .25-.4", pl.col("ret_4h") < 0.4),
    ("big >=.4", pl.col("ret_4h") >= 0.4),
    (".25-.3", pl.col("ret_4h") < 0.3),
    (".3-.4", (pl.col("ret_4h") >= 0.3) & (pl.col("ret_4h") < 0.4)),
    (".4-.6", (pl.col("ret_4h") >= 0.4) & (pl.col("ret_4h") < 0.6)),
    (">=.6", pl.col("ret_4h") >= 0.6),
]:
    for y in [None, 2022, 2023, 2024, 2025, 2026]:
        d = T.filter(flt) if y is None else T.filter(flt & (pl.col("year") == y))
        r = boot(d)
        if r:
            rows.append(
                (name, y or "all", *[round(v, 3) if isinstance(v, float) else v for v in r])
            )
print(
    pl.DataFrame(
        rows, schema=["size", "year", "diff", "lo", "hi", "p<=0", "n+", "n-"], orient="row"
    )
)
# same without 2025-10-10
print("-- excluding 2025-10-10")
rows = []
for name, flt in [("small .25-.4", pl.col("ret_4h") < 0.4), ("big >=.4", pl.col("ret_4h") >= 0.4)]:
    d = T.filter(flt & (pl.col("day").cast(pl.Utf8) != "2025-10-10"))
    r = boot(d)
    rows.append((name, *[round(v, 3) if isinstance(v, float) else v for v in r]))
print(pl.DataFrame(rows, schema=["size", "diff", "lo", "hi", "p<=0", "n+", "n-"], orient="row"))

print("\n== 4. share f<0 by year x size; mean rate; pumps that continued")
T = T.with_columns(cont=pl.col("gross") < 0, dbl=pl.col("sq100"))
print(
    T.group_by("sz2", "year")
    .agg(
        n=pl.len(),
        neg_share=(pl.col("rate") < 0).mean().round(3),
        rate_bp=pl.col("rate").mean() * bp,
        med_rate_bp=pl.col("rate").median() * bp,
        rate24_bp=pl.col("rate24").mean() * bp,
        cont=pl.col("cont").mean().round(3),
        sq50=pl.col("sq50").mean().round(3),
        gross=pl.col("gross").mean() * bp,
        fund=pl.col("fund").mean() * bp,
        listed_med=pl.col("listed_days").median(),
        r30d=pl.col("r30d").median().round(3),
        mkt24=pl.col("mkt_r24h").mean().round(4),
        days=pl.col("day").n_unique(),
    )
    .sort("sz2", "year")
)
print("\n== 5. f>=0 only, small pumps by year: continuation, gross, listing age")
print(
    T.filter(pl.col("sgn") == "f>=0")
    .group_by("sz2", "year")
    .agg(
        n=pl.len(),
        net=pl.col("net").mean() * bp,
        gross=pl.col("gross").mean() * bp,
        cont=pl.col("cont").mean().round(3),
        sq50=pl.col("sq50").mean().round(3),
        young60=(pl.col("listed_days") < 60).mean().round(3),
        net_young=pl.col("net").filter(pl.col("listed_days") < 60).mean() * bp,
        net_old=pl.col("net").filter(pl.col("listed_days") >= 60).mean() * bp,
        r7d=pl.col("r7d").median().round(3),
    )
    .sort("sz2", "year")
)

print("\n== 6. crowding vs funding cost")
D = T.drop_nulls("ls_acc")
print("n with metrics", D.height, "frac", D.height / T.height)
for c in ("ls_acc", "top_ls_acc", "oi_ch4h", "oi_ch24h", "taker_ls"):
    d = D.drop_nulls(c)
    print(
        c,
        "spearman w/ rate:",
        round(d.select(pl.corr(pl.col(c), pl.col("rate"), method="spearman")).item(), 3),
        " w/ net:",
        round(d.select(pl.corr(pl.col(c), pl.col("net"), method="spearman")).item(), 3),
        " rate-sign point-biserial (rate>=0):",
        round(
            d.select(
                pl.corr(pl.col(c), (pl.col("rate") >= 0).cast(pl.Float64), method="spearman")
            ).item(),
            3,
        ),
    )
print(
    "spearman rate vs net:",
    round(T.select(pl.corr("rate", "net", method="spearman")).item(), 3),
    " rate vs fund paid:",
    round(T.select(pl.corr("rate", "fund", method="spearman")).item(), 3),
)
D = D.with_columns(lsq=pl.col("ls_acc").qcut(3, labels=["low(crowded short)", "mid", "high"]))
print("-- ls_acc tercile x sign, all sizes")
print(agg(D.group_by("lsq", "sgn")).sort("lsq", "sgn"))
print("-- ls_acc tercile x sign, big only")
print(agg(D.filter(pl.col("ret_4h") >= 0.4).group_by("lsq", "sgn")).sort("lsq", "sgn"))
D = D.with_columns(oiq=pl.col("oi_ch24h").qcut(3, labels=["oi low", "oi mid", "oi high"]))
print("-- oi_ch24h tercile x sign")
print(agg(D.drop_nulls("oiq").group_by("oiq", "sgn")).sort("oiq", "sgn"))
# gross vs funding decomposition of the sign gap
for nm, d in [
    ("small", T.filter(pl.col("sz2") == "small .25-.4")),
    ("big", T.filter(pl.col("sz2") == "big >=.4")),
]:
    a = d.filter(pl.col("sgn") == "f>=0")
    b = d.filter(pl.col("sgn") == "f<0")
    print(
        nm,
        "gap net",
        round((a["net"].mean() - b["net"].mean()) * bp),
        "= gross gap",
        round((a["gross"].mean() - b["gross"].mean()) * bp),
        "+ funding gap",
        round((b["fund"].mean() - a["fund"].mean()) * bp),
    )
# OLS: net ~ sign + ls_acc z + oi z + size
import numpy.linalg as la  # noqa: E402

Dd = D.drop_nulls(["oi_ch24h", "ls_acc"]).filter(pl.col("ret_4h") < 5)


def z(c):
    v = Dd[c].to_numpy()
    return (v - v.mean()) / v.std()


X = np.column_stack(
    [
        np.ones(Dd.height),
        (Dd["rate"].to_numpy() >= 0).astype(float),
        z("ls_acc"),
        z("oi_ch24h"),
        np.log(Dd["ret_4h"].to_numpy()),
    ]
)
y = Dd["net"].to_numpy() * bp
for nm, cols in [
    ("sign only", [0, 1]),
    ("sign+ls_acc+oi+size", [0, 1, 2, 3, 4]),
    ("ls_acc+oi+size (no sign)", [0, 2, 3, 4]),
]:
    b, *_ = la.lstsq(X[:, cols], y, rcond=None)
    res = y - X[:, cols] @ b
    cov = la.inv(X[:, cols].T @ X[:, cols]) * res.var()
    print(
        nm,
        "coef",
        np.round(b, 1),
        "se(iid)",
        np.round(np.sqrt(np.diag(cov)), 1),
        "R2",
        round(1 - res.var() / y.var(), 4),
    )
