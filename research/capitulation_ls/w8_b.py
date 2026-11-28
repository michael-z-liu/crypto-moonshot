"""w8-B: 2024 anatomy (small pumps, f>=0) and market-wide funding level by year."""

import polars as pl
from common_ls import OWN

pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_width_chars(220)
pl.Config.set_tbl_cols(30)
bp = 1e4
T = pl.read_parquet("/tmp/w8_T.parquet").with_columns(mon=pl.col("day").dt.strftime("%Y-%m"))
print(
    "day dtype",
    T["day"].dtype,
    "trades on 2025-10-10:",
    T.filter(pl.col("day").cast(pl.Utf8) == "2025-10-10").height,
)
S = T.filter((pl.col("ret_4h") < 0.4) & (pl.col("rate") >= 0))
for y in (2023, 2024, 2025):
    d = S.filter(pl.col("year") == y).sort("net")
    n = d.height
    tot = d["net"].sum() * bp
    print(
        f"\n== small f>=0, {y}: n={n} mean {d['net'].mean() * bp:.0f} bp; median {d['net'].median() * bp:.0f}; trimmed (drop worst 5%) {d['net'][int(n * 0.05) :].mean() * bp:.0f}; worst10 contribute {d['net'][:10].sum() * bp / n:.0f} bp/trade; liq share {d['sq100'].mean():.3f}; sq50 {d['sq50'].mean():.3f}"  # noqa: E501
    )
    print(
        " worst 6:",
        [
            (r["symbol"], str(r["day"]), round(r["net"] * bp), round(r["ret_4h"], 2))
            for r in d.head(6).iter_rows(named=True)
        ],
    )
    print(
        " distinct coins",
        d["symbol"].n_unique(),
        "top3 coins by trades",
        d.group_by("symbol").len().sort("len", descending=True).head(3).rows(),
    )
d = S.filter(pl.col("year") == 2024)
print("\n2024 small f>=0 by month")
print(
    d.group_by("mon")
    .agg(
        n=pl.len(),
        net=pl.col("net").mean() * bp,
        med=pl.col("net").median() * bp,
        cont=pl.col("cont").mean().round(2)
        if "cont" in d.columns
        else (pl.col("gross") < 0).mean().round(2),
        sq50=pl.col("sq50").mean().round(2),
        mkt24=pl.col("mkt_r24h").mean().round(4),
    )
    .sort("mon")
)
print("\nstop-event concentration 2024: share of trades by top-day")
print(d.group_by("day").len().sort("len", descending=True).head(6))
# episodes/day clustering of 2024 trades vs other years
print("\ntrades per active day (small f>=0):")
print(
    S.group_by("year")
    .agg(n=pl.len(), days=pl.col("day").n_unique())
    .with_columns(per=pl.col("n") / pl.col("days"))
    .sort("year")
)
# size distribution of ret_4h within small by year
print(
    S.group_by("year")
    .agg(
        r4=pl.col("ret_4h").mean().round(3),
        listed_med=pl.col("listed_days").median(),
        liq=pl.col("sq100").mean().round(3),
        liq_bp=(pl.col("sq100").cast(pl.Float64) * -1e4).mean(),
    )
    .sort("year")
)
# market-wide funding by year
f = pl.read_parquet(OWN / "futures" / "funding.parquet")
print(f.columns, f.height)
tc = next(c for c in f.columns if "time" in c)
f = f.with_columns(year=pl.col(tc).dt.year())
print(
    f.filter(pl.col("year") >= 2022)
    .group_by("year")
    .agg(
        prints=pl.len(),
        neg=(pl.col("rate") < 0).mean().round(3),
        mean_bp=pl.col("rate").mean() * bp,
        med_bp=pl.col("rate").median() * bp,
        p10_bp=pl.col("rate").quantile(0.1) * bp,
        syms=pl.col("symbol").n_unique(),
    )
    .sort("year")
)
# at-spike population (all pumps >=0.25 perp): funding level by year
F = pl.read_parquet(OWN / "feats.parquet").filter(
    pl.col("perp")
    & (pl.col("qv_base") >= 20000)
    & (pl.col("year") >= 2022)
    & (pl.col("ret_4h") >= 0.25)
)
print(
    F.group_by("year")
    .agg(
        n=pl.len(),
        neg=(pl.col("rate") < 0).mean().round(3),
        mean_bp=pl.col("rate").mean() * bp,
        med_bp=pl.col("rate").median() * bp,
        hi=(pl.col("rate") >= 0.0003).mean().round(3),
    )
    .sort("year")
)
