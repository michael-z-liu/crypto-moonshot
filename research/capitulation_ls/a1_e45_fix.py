"""A1: E45 ("settings chosen on 2022-24, read on 2025-26") with the weak-listing filter measured at
each candidate's own entry time.

E45 keeps the listing symbols whose perp traded at or below its listing price 72 hours after the
spot listing (E27's `since_list`, measured at the 72h entry) and then lets the 2022-24 grid choose
the entry delay. It chose 24 hours, so that book shorts at listing + 24h only the listings that
will be weak 48 hours later. Here `since_list` is taken at the entry of each delay (perp open at
the entry hour over the perp open at the listing hour, as E27 does at 72h). L and C are chosen
as E45 does (their choice does not involve M) and slots are re-chosen on 2022-24.

Run: uv run --group research python a1_e45_fix.py
"""

from __future__ import annotations

import itertools

import e13_listing_book as e13
import numpy as np
import polars as pl
from e18_perp_listings import book as lbook
from e25_combined import book, sleeve_crash_short, sleeve_long
from e45_frozen_2024 import SEL, score

LISTED = {r["symbol"]: int(r["listed"].timestamp()) for r in e13.L.iter_rows(named=True)}


def since_at_entry(d: pl.DataFrame) -> pl.Series:
    out = []
    for s, t_in in zip(d["symbol"], d["t_in"], strict=True):
        h, o, *_ = e13.perp[s]
        i = int(np.searchsorted(h, t_in))
        k0 = int(np.searchsorted(h, LISTED[s]))
        out.append(o[i] / o[k0] - 1 if i > k0 else np.nan)
    return pl.Series(out, dtype=pl.Float64)


def m_frame(d: pl.DataFrame) -> pl.DataFrame:
    return d.select(
        "symbol", adv=pl.lit(0.0), t_in="t_in", t_out="t_out", net="net", qv="qv",
        sleeve=pl.lit("N"), prio=pl.lit(1.0),
    )  # fmt: skip


def main() -> None:
    print("\n== A1 fix: weak filter at the entry time of each delay")
    L = sleeve_long(q=0.99, tp=0.1, hold=4320)  # E45's 2022-24 choices
    C = sleeve_crash_short(r4=-0.1, calm=0.01, cnn=0.5, hold=4320)
    mres = []
    for delay, hold, tp, sl in itertools.product((24, 72), (7, 14, 30), (None, 0.5), (None, 0.5)):
        d = e13.trades(delay, hold, tp, sl).filter(pl.col("year") >= 2022)
        d = d.with_columns(since=since_at_entry(d)).filter(pl.col("since") <= 0)
        b = lbook(d, 10)
        s = b.filter(pl.col("year").is_in(SEL))
        sc = float(np.log1p(s["ret"].clip(-0.99, None)).mean() - 0.5 * abs(s["dd"].min()))
        mres.append((sc, dict(delay=delay, hold=hold, tp=tp, sl=sl), d))
    mres.sort(key=lambda r: -r[0])
    print(f"M: best on 2022-24 {mres[0][1]} score {mres[0][0]:.3f} (weak filter at entry)")
    for sc, cfg, d in mres[:4]:
        print(f"   {cfg} score {sc:.3f} n={d.height} mean {d['net'].mean() * 100:+.1f}%")
    # the 24h book E45 picked, with the honest filter, for a like-for-like comparison
    d24 = e13.trades(24, 30, 0.5, 0.5).filter(pl.col("year") >= 2022)
    d24 = d24.with_columns(since=since_at_entry(d24))
    from common_ls import OWN

    x27 = pl.read_parquet(OWN / "e27_listings.parquet").filter(pl.col("since_list") <= 0)
    leak = d24.filter(pl.col("symbol").is_in(x27["symbol"].to_list()))
    print(
        f"delay 24h, 30d, TP/SL 50%: E45 filter (measured at 72h) n={leak.height} mean {leak['net'].mean() * 100:+.1f}%"  # noqa: E501
    )
    print(
        f"delay 24h, 30d, TP/SL 50%: honest filter n={d24.filter(pl.col('since') <= 0).height}"
        f" mean {d24.filter(pl.col('since') <= 0)['net'].mean() * 100:+.1f}%"
    )
    for lab, M in (
        ("chosen (honest)", m_frame(mres[0][2])),
        ("E45 delay 24h, honest", m_frame(d24.filter(pl.col("since") <= 0))),
    ):
        allb = []
        for slots in (8, 10, 15):
            b = book(pl.concat([L, M, C], how="vertical_relaxed"), slots)
            allb.append((score(b), slots, b))
        allb.sort(key=lambda r: -r[0])
        _, slots, b = allb[0]
        n = int(b["trades"].sum())
        bp = float((b["bp"] * b["trades"]).sum() / n)
        print(f"{lab}: slots {slots}, {bp:+.0f}bp a trade over all years")
        for r in b.iter_rows(named=True):
            print(
                f"  {r['year']}: {r['trades']:>3}t {r['bp']:+5.0f}bp {r['ret'] * 100:+5.0f}% dd {r['dd'] * 100:+4.0f}%"  # noqa: E501
            )


if __name__ == "__main__":
    main()
