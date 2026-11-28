"""E09: do perp prices follow spot after pumps? Same-hour close-to-close returns, 24h and 72h."""

from __future__ import annotations

import numpy as np
import polars as pl
from common_ls import OWN, first_per_event, meta
from futures_ls import spot_name

m = meta()
p = pl.read_parquet(OWN / "futures" / "perp_1h.parquet").with_columns(
    spot=pl.col("symbol").map_elements(spot_name, return_dtype=pl.Utf8),
    h=pl.col("open_time").dt.epoch("s"),
)
d = pl.read_parquet(OWN / "feats.parquet").filter(
    pl.col("perp") & (pl.col("ret_4h") >= 0.3) & (pl.col("qv_base") >= 20_000)
)
keep = set(first_per_event(np.isin(np.arange(len(m)), d["row"].to_numpy())).tolist())
d = d.filter(pl.col("row").is_in(list(keep)))
# Hour grid: hour 0 = the first full hour after the entry (t + 5 min).
d = d.with_columns(h0=((pl.col("t") + 300) // 3600 + 1) * 3600)
pc = p.select("spot", "h", "open", "close")
out = d.select("row", "symbol", "h0", "year").join(
    pc.rename({"spot": "symbol", "h": "h0", "open": "p0"}).drop("close"), on=["symbol", "h0"]
)
for k in (24, 72):
    out = (
        out.with_columns(hk=pl.col("h0") + (k - 1) * 3600)
        .join(
            pc.rename({"spot": "symbol", "h": "hk", "close": f"pc{k}"}).drop("open"),
            on=["symbol", "hk"],
            how="left",
        )
        .drop("hk")
    )
# Spot over the same hours from 5m candles is not cached; use fwd72 (entry = next 5m open) as close proxy.  # noqa: E501
from common_ls import fwd72  # noqa: E402

f = np.asarray(fwd72()[out["row"].to_numpy()], dtype=np.float64)
out = out.with_columns(
    perp24=pl.col("pc24") / pl.col("p0") - 1, perp72=pl.col("pc72") / pl.col("p0") - 1,
    spot24=pl.Series(f[:, 2, 23]), spot72=pl.Series(f[:, 2, 71]),
)  # fmt: skip
print(out.height, "pump events with perp")
print(
    out.select(
        c24=pl.corr("perp24", "spot24"),
        c72=pl.corr("perp72", "spot72"),
        perp24=-pl.col("perp24").mean() * 1e4,
        spot24=-pl.col("spot24").mean() * 1e4,
        perp72=-pl.col("perp72").mean() * 1e4,
        spot72=-pl.col("spot72").mean() * 1e4,
        perp72med=-pl.col("perp72").median() * 1e4,
        spot72med=-pl.col("spot72").median() * 1e4,
    )
)
print(
    out.group_by("year")
    .agg(n=pl.len(), perp72=-pl.col("perp72").mean() * 1e4, spot72=-pl.col("spot72").mean() * 1e4)
    .sort("year")
)

# Apples to apples: spot over the same clock hours, from the 5m store.
import os  # noqa: E402
import sys  # noqa: E402

os.environ["MOONSHOT_UNIVERSE"] = "all"
from common_ls import CACHE_ALL  # noqa: E402

sys.path.insert(0, str(CACHE_ALL.parent))
from common import load  # noqa: E402

rows = []
for (sym,), g in out.group_by(["symbol"]):
    s = load(str(sym)).select(h=pl.col("open_time").dt.epoch("s"), o="open", c="close")
    so = dict(zip(s["h"].to_list(), s["o"].to_list(), strict=True))
    sc = dict(zip(s["h"].to_list(), s["c"].to_list(), strict=True))
    for r in g.iter_rows(named=True):
        o = so.get(r["h0"])
        c24 = sc.get(r["h0"] + 24 * 3600 - 300)
        c72 = sc.get(r["h0"] + 72 * 3600 - 300)
        rows.append(
            {
                "row": r["row"],
                "s_same24": (c24 / o - 1) if o and c24 else None,
                "s_same72": (c72 / o - 1) if o and c72 else None,
            }
        )
out = out.join(pl.DataFrame(rows), on="row")
print(
    out.select(
        c72=pl.corr("perp72", "s_same72"),
        perp24=-pl.col("perp24").mean() * 1e4,
        spot_same24=-pl.col("s_same24").mean() * 1e4,
        perp72=-pl.col("perp72").mean() * 1e4,
        spot_same72=-pl.col("s_same72").mean() * 1e4,
        perp72med=-pl.col("perp72").median() * 1e4,
        spot_same72med=-pl.col("s_same72").median() * 1e4,
        gap_abs_med=(pl.col("perp72") - pl.col("s_same72")).abs().median() * 1e4,
    )
)
print(
    out.group_by("year")
    .agg(
        n=pl.len(),
        perp72=-pl.col("perp72").mean() * 1e4,
        spot_same72=-pl.col("s_same72").mean() * 1e4,
    )
    .sort("year")
)
print(
    out.with_columns(gap=pl.col("perp72") - pl.col("s_same72"))
    .sort("gap", descending=True)
    .select("symbol", "h0", "perp24", "s_same24", "perp72", "s_same72", "gap")
    .with_columns(h0=pl.from_epoch("h0"))
    .head(12)
)
g = (
    out.with_columns(gap=pl.col("perp72") - pl.col("s_same72"), h0=pl.from_epoch("h0"))
    .filter(pl.col("year") == 2025)
    .drop_nulls("gap")
)
print(
    g.sort("gap", descending=True)
    .select("symbol", "h0", "perp24", "s_same24", "perp72", "s_same72", "gap")
    .head(10)
)
print(
    "2025 gap sum",
    g["gap"].sum(),
    "n",
    g.height,
    "top5 share",
    g.sort("gap", descending=True)["gap"].head(5).sum(),
)
