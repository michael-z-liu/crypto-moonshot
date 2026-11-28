"""X4 step 2: configuration grid (all in-sample). Stage 1 no book; stage 2 books for the top by mean per-year avg trade."""  # noqa: E501

import itertools

import polars as pl
from x4_lib import EX, run

rows = []
for mode, kind, k, fell, thr, hold in itertools.product(
    ("short", "long", "neutral"),
    (0, 1),
    (3, 5, 10),
    (None, -0.05, -0.02),
    (0.0, 0.5, 0.7, 0.85),
    (4, 12, 24, 48, 72),
):
    tps = (None,) if mode == "neutral" else (None, 0.05, 0.10)
    stops = (None,) if mode == "neutral" else (None, 0.10)
    for tp, stop in itertools.product(tps, stops):
        r = run(mode, kind, k, fell, thr, hold, tp, stop, do_book=False)
        if r:
            rows.append(r)
g = pl.DataFrame(rows, infer_schema_length=None)
g.write_parquet(EX / "contagion_grid.parquet")
print("configs", g.height)
g = g.with_columns(my=pl.mean_horizontal([f"a{y}" for y in range(2022, 2027)]))
for mode in ("short", "long", "neutral"):
    s = g.filter((pl.col("mode") == mode) & (pl.col("n") >= 100))
    print(
        mode,
        "n configs",
        s.height,
        "avg of config avgs %.3f%%" % (s["avg"].mean() * 100),
        "share avg>0 {:.2f}".format((s["avg"] > 0).mean()),
    )
    print(
        s.sort("avg", descending=True)
        .head(6)
        .select("kind", "k", "fell", "thr", "hold", "tp", "stop", "n", "avg", "my")
    )
# books for the top 40 of each mode by mean per-year avg
bk = []
for mode in ("short", "long", "neutral"):
    s = (
        g.filter((pl.col("mode") == mode) & (pl.col("n") >= 100))
        .sort("my", descending=True)
        .head(40)
    )
    for c in s.iter_rows(named=True):
        r = run(mode, c["kind"], c["k"], c["fell"], c["thr"], c["hold"], c["tp"], c["stop"])
        r.pop("_taken")
        bk.append(r)
b = pl.DataFrame(bk, infer_schema_length=None)
b.write_parquet(EX / "contagion_books.parquet")
print(
    b.sort("worst", descending=True)
    .head(8)
    .select(
        "mode",
        "kind",
        "k",
        "fell",
        "thr",
        "hold",
        "tp",
        "stop",
        "book_n",
        "book_avg",
        "worst",
        "maxdd",
        *[f"y{y}" for y in range(2021, 2027)],
    )
)
