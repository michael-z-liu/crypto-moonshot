"""Entries to leave out because the coin's price series has a redenomination in reach.

    uv run --group research python build_exclusions.py

A redenomination or token swap (COCOS 1000:1 in 2021, the new LUNA under the old symbol in 2022)
shows in the candles as a trading halt followed by an open at a multiple of the last close. It
is not a price move: it makes false +30% hits and false stops in the labels, and false returns
in the factors. Rule, the same for every year: a *break* is a pair of consecutive 5m candles
whose open is more than 3 times, or less than a third of, the previous close. An entry is
excluded when a break of its coin lies in `(D - 200 days, D + 30 days]`: inside its outcome
window or inside its longest factor lookback.

Writes `cache/exclusions.parquet` (symbol, d_us, break_t) and `results/exclusions.md` (the
breaks and the number of entries excluded per split; no outcomes).
"""

from __future__ import annotations

import numpy as np
import polars as pl
from common import CACHE, DAY, PANEL, RESULTS, WINDOW_US, read_5m, universe, utc

BREAK_RATIO = 3.0
LOOKBACK_US = 200 * DAY
EXCLUSIONS = CACHE / "exclusions.parquet"


def breaks(symbol: str) -> list[tuple[str, int, float, float]]:
    """(symbol, open time of the candle after the break, open / previous close, halt hours)."""
    c = read_5m(symbol)
    if c.height < 2:
        return []
    t = c["t"].to_numpy().astype(np.int64)
    o, cl = c["open"].to_numpy(), c["close"].to_numpy()
    r = o[1:] / cl[:-1]
    idx = np.flatnonzero((r > BREAK_RATIO) | (r < 1 / BREAK_RATIO))
    return [(symbol, int(t[i + 1]), float(r[i]), float((t[i + 1] - t[i]) / 3.6e9)) for i in idx]


def main() -> None:
    found = [b for s in universe() for b in breaks(s)]
    br = pl.DataFrame(found, schema=["symbol", "break_t", "ratio", "halt_hours"], orient="row")
    panel = pl.read_parquet(PANEL, columns=["symbol", "d_us", "split"])
    ex = (
        panel.join(br.select("symbol", "break_t"), on="symbol")
        .filter(
            (pl.col("break_t") > pl.col("d_us") - LOOKBACK_US)
            & (pl.col("break_t") <= pl.col("d_us") + WINDOW_US)
        )
        .group_by("symbol", "d_us", "split")
        .agg(pl.col("break_t").min())
        .sort("symbol", "d_us")
    )
    ex.select("symbol", "d_us", "break_t").write_parquet(EXCLUSIONS)
    lines = [
        "# Excluded entries: price-series breaks",
        "",
        "A break is a pair of consecutive 5m candles whose open is more than 3 times, or less",
        "than a third of, the previous close (a redenomination or token swap after a trading",
        "halt). An entry is excluded when a break of its coin lies within 200 days before `D`",
        "or inside its 30-day window. Built by `build_exclusions.py`.",
        "",
        "| coin | reopened (UTC) | open / last close | halt (hours) |",
        "| --- | --- | --- | --- |",
    ]
    lines += [
        f"| {s} | {utc(t):%Y-%m-%d %H:%M} | {r:g} | {h:.0f} |" for s, t, r, h in sorted(found)
    ]
    lines += ["", "| split | entries excluded | of |", "| --- | --- | --- |"]
    tot = dict(panel.group_by("split").len().iter_rows())
    cnt = dict(ex.group_by("split").len().iter_rows())
    lines += [f"| {k} | {cnt.get(k, 0)} | {tot[k]} |" for k in sorted(tot)]
    (RESULTS / "exclusions.md").write_text("\n".join(lines) + "\n")
    print(f"{len(found)} breaks, {ex.height} entries excluded")


if __name__ == "__main__":
    main()
