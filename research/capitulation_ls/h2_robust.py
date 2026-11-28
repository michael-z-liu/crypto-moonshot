"""H2 (holdout, BTC pairs): robustness of L on the converted BTC pairs (day5 2026 fold), all
post hoc checks of the pre-registered rule, not new rules:

1. highs haircut: every forward high lowered by 1% / 2% of price ((1 + h)(1 - x) - 1), against
   the conversion's high approximation and thin books at the take profit;
2. without the two largest flush days (2018-01-11, 2018-01-16) and without all of 2017-12 ..
   2018-01 (the altcoin top and first crash);
3. by liquidity (qv_base, USDT per 5m) at 0.4% and 1.2%;
4. tick size: BTC pairs were quoted in satoshis; one satoshi relative to the entry price in BTC is
   a floor on the spread. A round trip pays at least that tick (cost 1.2% + 1 tick).

Writes holdout/btc/robust.txt.
Run: uv run --group research python h2_robust.py
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import sim
from h1_eval import HY, book_table, clustered, trades
from h2_eval import HOLDOUT, OUT, WIDE_BTC, use_btc

lines: list[str] = []


def emit(s: str) -> None:
    lines.append(s)
    print(s)


def summary(t: pl.DataFrame, label: str, extra: float = 0.0) -> str:
    if t.height == 0:
        return f"{label}: no trades"
    net = t["net"] - extra
    b = book_table(t.with_columns(net=net))
    books = ", ".join(f"{r['year']} {r['ret']:+.1%}" for r in b.iter_rows(named=True))
    return (
        f"{label}: n {t.height}, mean {net.mean() * 1e4:+.0f} bp, TP rate"
        f" {(t['kind'] == 2).mean():.2f}, win {(net > 0).mean():.2f}, days {t['day'].n_unique()};"
        f" book {books}"
    )


def main() -> None:
    m = use_btc()
    pct = pl.read_parquet(OUT / "scores.parquet")["pct_day5"].to_numpy()
    base = trades(m, pct, HY, 0.995)
    emit(summary(base, "L as tested (0.4%)"))

    emit("\n1. Highs haircut")
    f2 = np.load(OUT / "fwd2.npy")
    f72 = np.load(OUT / "fwd72.npy")
    for x in (0.01, 0.02):
        a, b = f2.copy(), f72.copy()
        a[:, 0] = (1 + a[:, 0]) * (1 - x) - 1
        b[:, 0] = (1 + b[:, 0]) * (1 - x) - 1
        sim.fwd2, sim.fwd72 = (lambda a=a: a), (lambda b=b: b)  # type: ignore[assignment,misc]
        t = trades(m, pct, HY, 0.995)
        emit(summary(t, f"highs -{x:.0%} (0.4%)"))
        emit(summary(t, f"highs -{x:.0%} (1.2%)", WIDE_BTC))
    use_btc()

    emit("\n2. Without the largest days")
    big = [dt.date(2018, 1, 11), dt.date(2018, 1, 16)]
    t2 = base.filter(~pl.col("day").is_in(big))
    emit(clustered(t2, "without 2018-01-11 and 2018-01-16 (0.4%)"))
    emit(summary(t2, "without 2018-01-11 and 2018-01-16 (1.2%)", WIDE_BTC))
    mania = (pl.col("day") >= dt.date(2017, 12, 1)) & (pl.col("day") < dt.date(2018, 2, 1))
    t3 = base.filter(~mania)
    emit(clustered(t3, "without 2017-12 .. 2018-01 (0.4%)"))
    emit(clustered(t3.with_columns(net=pl.col("net") - WIDE_BTC), "same at 1.2%"))
    emit(summary(t3, "without 2017-12 .. 2018-01 (1.2%)", WIDE_BTC))

    emit("\n3. By liquidity (qv_base in USDT per 5m candle at the spike)")
    qv = m["qv_base"].to_numpy()[base["row"].to_numpy()]
    for lo, hi in ((5e3, 1e4), (1e4, 3e4), (3e4, 1e12)):
        t = base.filter(pl.Series((qv >= lo) & (qv < hi)))
        emit(summary(t, f"qv_base {lo:,.0f}-{min(hi, 1e9):,.0f} (1.2%)", WIDE_BTC))

    emit("\n4. Tick size (1 satoshi relative to the entry price in BTC)")
    btc = pl.read_parquet(
        HOLDOUT / "klines/spot/interval=5m/symbol=BTCUSDT/*.parquet", columns=["open_time", "close"]
    ).rename({"close": "btc_close"})
    mm = (
        m.with_row_index("row")
        .select(pl.col("row").cast(pl.Int64), "open_time", "close")
        .join(btc, on="open_time", how="left")
    )
    tb = base.join(mm.select("row", "close", "btc_close"), on="row", how="left").with_columns(
        tick=1e-8 / (pl.col("close") / pl.col("btc_close"))
    )
    emit(
        f"tick share of price at L entries: median {tb['tick'].median():.4%}, p75"
        f" {tb['tick'].quantile(0.75):.4%}, max {tb['tick'].max():.4%}; share > 0.2%"
        f" {(tb['tick'] > 0.002).mean():.2f}, > 0.5% {(tb['tick'] > 0.005).mean():.2f}"
    )
    t4 = tb.with_columns(net=pl.col("net") - WIDE_BTC - pl.col("tick"))
    emit(clustered(t4, "cost 1.2% + 1 tick"))
    emit(summary(t4, "cost 1.2% + 1 tick"))
    for lab, f in (
        ("tick <= 0.2%", pl.col("tick") <= 0.002),
        ("tick > 0.2%", pl.col("tick") > 0.002),
    ):
        emit(summary(t4.filter(f), f"{lab}, cost 1.2% + 1 tick"))
    (OUT / "robust.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
