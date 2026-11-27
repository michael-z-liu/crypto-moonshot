"""W3-A: label noise vs signal and variance decomposition of the 1h / 72h outcomes."""

import polars as pl

d = pl.read_parquet("/tmp/w3_tab.parquet").with_columns(
    y1c=pl.col("y1h").clip(-0.2, 0.2),
    y4c=pl.col("y4h").clip(-0.2, 0.5),
    c72c=pl.col("c72").clip(-0.5, 1.0),
)
d = d.with_columns(day=pl.col("day").cast(pl.Date))


def r2adj(df, y, g):
    x = df.select(pl.col(y).alias("y"), pl.col(g).alias("g")).drop_nulls()
    n = x.height
    mu = x["y"].mean()
    sst = ((x["y"] - mu) ** 2).sum()
    gm = x.group_by("g").agg(m=pl.col("y").mean(), c=pl.len())
    k = gm.height
    sse = x.join(gm, on="g").select(((pl.col("y") - pl.col("m")) ** 2).sum()).item()
    return 1 - (sse / (n - k)) / (sst / (n - 1)), k


subsets = {
    "all 2022+": d,
    "crash (ret_4h<=-10%)": d.filter(pl.col("ret_4h") <= -0.10),
    "day5>=0.995": d.filter(pl.col("day5") >= 0.995),
    "crash & day5>=.98": d.filter((pl.col("ret_4h") <= -0.10) & (pl.col("day5") >= 0.98)),
}
print("== variance decomposition (adj R2 of one-way group means)")
for nm, s in subsets.items():
    s = s.with_columns(sy=pl.col("symbol"), ym=pl.col("day").dt.strftime("%Y-%m"))
    print(nm, "n=", s.height, "days", s["day"].n_unique(), "symbols", s["symbol"].n_unique())
    for y in ("y1c", "y4c", "c72c", "o72"):
        print(f"  {y:5s} sd={s[y].std():.4f} mean={s[y].mean():+.4f}", end="")
        for g in ("year", "ym", "day", "symbol"):
            r, k = r2adj(s, y, g)
            print(f"  R2[{g}]={r:+.3f}(k={k})", end="")
        print()


# within-day-demeaned correlation
def ics(s, a, b):
    s = s.with_columns(
        ra=pl.col(a).rank().over("day"), rb=pl.col(b).rank().over("day"), n=pl.len().over("day")
    ).filter(pl.col("n") >= 5)
    pooled = s.select(
        pl.corr(pl.col(a) - pl.col(a).mean().over("day"), pl.col(b) - pl.col(b).mean().over("day"))
    ).item()
    per = s.group_by("day").agg(ic=pl.corr("ra", "rb"), n=pl.len()).drop_nulls()
    return pooled, per["ic"].mean(), per.height


print("== within-day IC (pooled demeaned Pearson, mean per-day Spearman, ndays), score vs outcome")
base = {"all 2022+": d, "crash": d.filter(pl.col("ret_4h") <= -0.10)}
for nm, s in base.items():
    for sc in ("day5", "tp20day"):
        for y in ("y1c", "y4c", "c72c", "o72"):
            p, sp, nd = ics(s, sc, y)
            tot = s.select(pl.corr(pl.col(sc).rank(), pl.col(y).rank())).item()
            print(
                f" {nm:10s} {sc:8s} vs {y:4s}: total spearman {tot:+.3f}  within-day pooled {p:+.3f} per-day spearman {sp:+.3f} (days {nd})"  # noqa: E501
            )
print("== cross-correlation of scores within day (crash rows, spearman pooled-demeaned)")
cr = d.filter(pl.col("ret_4h") <= -0.10)
ns = ["day5", "tp20day", "day10b", "day_transformer", "day_tcn", "day_gru"]
for a in ns:
    print(
        f" {a:16s}",
        " ".join(
            f"{ics(cr.with_columns(pl.col(a).rank().alias('A'), pl.col(b).rank().alias('B')), 'A', 'B')[0]:+.2f}"  # noqa: E501
            for b in ns
        ),
    )
print("== within-day IC of each architecture vs y1c / o72 (all rows 2022+; crash rows)")
for nm, s in base.items():
    for a in ns:
        print(
            f" {nm:10s} {a:16s} y1c {ics(s, a, 'y1c')[0]:+.3f}  y4c {ics(s, a, 'y4c')[0]:+.3f}  o72 {ics(s, a, 'o72')[0]:+.3f}"  # noqa: E501
        )
print("== coin-day ICC (spikes in same symbol-day) of y1c within the crash set")
c = cr.with_columns(cell=pl.col("symbol") + pl.col("day").cast(pl.String))
c = c.with_columns(
    dm=pl.col("y1c") - pl.col("y1c").mean().over("day"),
    dm2=pl.col("o72") - pl.col("o72").mean().over("day"),
)
for y, lab in (("dm", "y1c day-demeaned"), ("dm2", "o72 day-demeaned")):
    g = (
        c.group_by("cell")
        .agg(n=pl.len(), m=pl.col(y).mean(), v=pl.col(y).var())
        .filter(pl.col("n") >= 2)
    )
    tot = c[y].var()
    within = (g["v"] * (g["n"] - 1)).sum() / (g["n"] - 1).sum()
    print(
        f" {lab}: total var {tot:.5f}, within-cell var {within:.5f} -> same-coin-day share ~{1 - within / tot:.3f} ({g.height} cells)"  # noqa: E501
    )
