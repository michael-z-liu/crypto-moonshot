"""X3 step 5: 10-slot per-year $10k books for the most promising configs of every mode."""

import polars as pl
from x3_eval import EX, run_config

g = pl.read_parquet(EX / "lag_grid.parquet").filter(pl.col("n") >= 150)
rows = []
for mode in g["mode"].unique().sort():
    s = g.filter(pl.col("mode") == mode)
    top = s.sort("avg", descending=True).head(80)
    for r in top.iter_rows(named=True):
        b = run_config(mode, r["combo"], r["thr"], r["hold"], r["tp"], r["stop"])
        if b:
            b.pop("_taken")
            rows.append(b)
out = pl.DataFrame(rows)
out.write_parquet(EX / "lag_books.parquet")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_width_chars(250)
for mode in out["mode"].unique().sort():
    o = out.filter(pl.col("mode") == mode)
    print(mode, o.height, "configs with books; trades>=15/yr?")
    print(
        o.sort("book_avg", descending=True)
        .head(8)
        .select(
            "combo",
            "thr",
            "hold",
            "tp",
            "stop",
            "n",
            "avg",
            "book_n",
            "book_avg",
            *[f"y{y}" for y in range(2021, 2027)],
            "worst",
            "maxdd",
        )
        .with_columns(pl.col(pl.Float64).round(3))
    )
