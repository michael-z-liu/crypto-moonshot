"""W1: how certain is the 'edge is a 2025-26 phenomenon' claim? Event-level and
coin-cluster bootstrap
of the realistic net return (iso 1x liquidation at +100%, 30bp cost, funding) by period, for the
deduped (first spike per coin per 24h) and non-deduped samples."""

from __future__ import annotations

import numpy as np
import polars as pl
from common_ls import OWN, first_per_event, meta
from futures_ls import funding_paid
from sim import perp_paths

K = 96 + 63
bp = 1e4
rng = np.random.default_rng(1)


def build(thr: float, dedup: bool) -> pl.DataFrame:
    f = pl.read_parquet(OWN / "feats.parquet").filter(
        pl.col("perp")
        & (pl.col("qv_base") >= 20_000)
        & (pl.col("year") >= 2021)
        & (pl.col("ret_4h") >= thr)
    )
    if dedup:
        keep = first_per_event(np.isin(np.arange(meta().height), f["row"].to_numpy()))
        f = f.filter(pl.col("row").is_in(keep.tolist()))
    f = f.sort("row")
    pp, ok = perp_paths(f["row"].to_numpy())
    f = f.filter(pl.Series(ok))
    pc = pp[:, 2, : K + 1]
    last = np.array([c[~np.isnan(c)][-1] if (~np.isnan(c)).any() else 0.0 for c in pc])
    adv = np.nan_to_num(pp[:, 0, : K + 1], nan=0).max(1)
    t0 = f["t"].to_numpy() + 300
    fund = funding_paid(f["symbol"].to_numpy(), t0, t0 + 72 * 3600, -np.ones(len(f)))
    net = np.where(adv >= 0.99, -1.0, -last) - 0.003 - fund
    return f.with_columns(net=pl.Series(net), fade=pl.Series(-last), sq100=pl.Series(adv >= 1))


def cluster_boot(df: pl.DataFrame, col: str, n: int = 1500) -> tuple[float, float, float]:
    """mean and 90% CI, resampling coin-months (clusters) with replacement."""
    key = (df["symbol"] + "_" + df["open_time"].dt.strftime("%Y%m")).to_numpy()
    u, inv = np.unique(key, return_inverse=True)
    s = np.bincount(inv, weights=df[col].to_numpy())
    c = np.bincount(inv)
    m = []
    for _ in range(n):
        i = rng.integers(0, len(u), len(u))
        m.append(s[i].sum() / c[i].sum())
    return float(s.sum() / c.sum()), *np.percentile(m, [5, 95])


for thr, dedup in ((0.15, False), (0.15, True), (0.25, True), (0.4, True)):
    d = build(thr, dedup)
    print(
        f"\n=== ret_4h>={thr} dedup={dedup}: rows {d.height}, distinct coin-months {d.select((pl.col('symbol') + pl.col('open_time').dt.strftime('%Y%m')).n_unique()).item()}"  # noqa: E501
    )
    for lab, sub in (
        ("2021", d.filter(pl.col("year") == 2021)),
        ("2022-24", d.filter(pl.col("year").is_between(2022, 2024))),
        ("2025-26", d.filter(pl.col("year") >= 2025)),
    ):
        m, lo, hi = cluster_boot(sub, "net")
        mf, lof, hif = cluster_boot(sub, "fade")
        print(
            f"  {lab:8s} n={sub.height:6d} net {m * bp:+6.0f} bp (90% CI {lo * bp:+.0f}..{hi * bp:+.0f}) | mean 72h fade {mf * bp:+.0f} ({lof * bp:+.0f}..{hif * bp:+.0f}) | median fade {sub['fade'].median() * bp:+.0f}"  # noqa: E501
        )
    a, b = d.filter(pl.col("year").is_between(2022, 2024)), d.filter(pl.col("year") >= 2025)
    print(
        "  2025-26 minus 2022-24 net (bp): point", round((b["net"].mean() - a["net"].mean()) * bp)
    )
