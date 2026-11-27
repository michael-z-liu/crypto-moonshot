"""w8-C: bootstrap of the sign gap in GROSS (price) return vs funding paid; tail budget of small f>=0 pumps."""  # noqa: E501

import numpy as np
import polars as pl

bp = 1e4
rng = np.random.default_rng(3)
T = pl.read_parquet("/tmp/w8_T.parquet")


def boot(d, col, B=4000):
    u, inv = np.unique(d["day"].cast(pl.Utf8).to_numpy(), return_inverse=True)
    pos = d["rate"].to_numpy() >= 0
    x = d[col].to_numpy() * bp
    a = np.bincount(inv, weights=pos, minlength=len(u))
    b = np.bincount(inv, weights=~pos, minlength=len(u))
    sa = np.bincount(inv, weights=x * pos, minlength=len(u))
    sb = np.bincount(inv, weights=x * ~pos, minlength=len(u))
    est = sa.sum() / a.sum() - sb.sum() / b.sum()
    o = []
    for _ in range(B):
        k = rng.integers(0, len(u), len(u))
        o.append(sa[k].sum() / a[k].sum() - sb[k].sum() / b[k].sum())
    return (
        round(est),
        *np.round(np.percentile(o, [2.5, 97.5])),
        round(float((np.array(o) <= 0).mean()), 3),
    )


print("size  col  diff(f>=0 minus f<0) lo hi P(<=0)")
for nm, fl in [
    ("small .25-.4", pl.col("ret_4h") < 0.4),
    ("big >=.4", pl.col("ret_4h") >= 0.4),
    (".25-.3", pl.col("ret_4h") < 0.3),
    (".4-.6", (pl.col("ret_4h") >= 0.4) & (pl.col("ret_4h") < 0.6)),
]:
    d = T.filter(fl)
    for c in ("net", "gross", "fund"):
        print(nm, c, boot(d, c))
# gross gap by year, small / big
print("\ngross gap by year (f>=0 minus f<0), bp")
for nm, fl in [("small", pl.col("ret_4h") < 0.4), ("big", pl.col("ret_4h") >= 0.4)]:
    r = []
    for y in (2022, 2023, 2024, 2025, 2026):
        d = T.filter(fl & (pl.col("year") == y))
        r.append((y, boot(d, "gross", 1500)[:3]))
    print(nm, r)
# winsorised: cap at -100% loss already; median gross by sign (typical-trade)
print("\nmedian gross bp by size x sign")
T = T.with_columns(
    sz=pl.when(pl.col("ret_4h") < 0.4).then(pl.lit("small")).otherwise(pl.lit("big")),
    sg=pl.col("rate") >= 0,
)
print(
    T.group_by("sz", "sg")
    .agg(
        n=pl.len(),
        mean=pl.col("gross").mean() * bp,
        med=pl.col("gross").median() * bp,
        p5=pl.col("gross").quantile(0.05) * bp,
        liq=pl.col("sq100").mean(),
        mean_ex_liq=pl.col("gross").filter(~pl.col("sq100")).mean() * bp,
        win_ex_liq=(pl.col("gross") > 0).mean(),
    )
    .sort("sz", "sg")
)
# gross price-move: mean short return on non-liq trades
# what share of f<0 funding paid sits in squeezes
print("\nfunding paid f<0: squeeze (sq50) vs not, bp")
print(
    T.filter(~pl.col("sg"))
    .group_by("sz", "sq50")
    .agg(n=pl.len(), fund=pl.col("fund").mean() * bp, gross=pl.col("gross").mean() * bp)
    .sort("sz", "sq50")
)
# break-even: small f>=0 liq rate needed for 0 net with median trade
s = T.filter((pl.col("sz") == "small") & pl.col("sg"))
ex = s.filter(~pl.col("sq100"))
print(
    "\nsmall f>=0: net ex-liq",
    ex["net"].mean() * bp,
    "liq share",
    s["sq100"].mean(),
    "liq loss bp/trade",
    -s["sq100"].mean() * 1e4,
    "net all",
    s["net"].mean() * bp,
)
print("liq rate that zeroes net:", ex["net"].mean() / (ex["net"].mean() + 1.0))
