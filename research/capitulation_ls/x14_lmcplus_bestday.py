"""X14: LMC and LMC+ without each year's best flush day(s) for the long trades (stress test).

A flush day's value = the sum of net over that day's L candidates (spot and perp-only together).
Removes each year's best 1 and best 2 days from L (both universes) and rebuilds the books.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from e25_combined import book
from p2_lib import book_row, lmc, lperp, to_book

OUT = Path(__file__).resolve().parent / "results" / "explore"


def drop_best(frames: list[pl.DataFrame], k: int) -> list[pl.DataFrame]:
    lt = pl.concat(
        [f.filter(pl.col("sleeve").is_in(["L", "LP"])) for f in frames], how="vertical_relaxed"
    )
    day = lt.with_columns(d=pl.from_epoch("t_in").dt.date(), y=pl.from_epoch("t_in").dt.year())
    best = (
        day.group_by("y", "d").agg(v=pl.col("net").sum()).sort("v", descending=True)
        .group_by("y").head(k)["d"].to_list()
    )  # fmt: skip
    return [
        f.filter(~(pl.col("sleeve").is_in(["L", "LP"]) & pl.from_epoch("t_in").dt.date().is_in(best)))  # noqa: E501
        for f in frames
    ]  # fmt: skip


if __name__ == "__main__":
    base = lmc()
    lp = to_book(lperp(), "LP")
    rows = []
    for k in (0, 1, 2):
        a = drop_best(base, k) if k else base
        b = drop_best([*base, lp], k) if k else [*base, lp]
        rows.append(
            book_row(
                f"LMC, without best {k} L day(s)/yr", book(pl.concat(a, how="vertical_relaxed"), 10)
            )
        )
        rows.append(
            book_row(
                f"LMC+, without best {k} L day(s)/yr",
                book(pl.concat(b, how="vertical_relaxed"), 10),
            )
        )
    df = pl.DataFrame(rows)
    cols = df.columns
    txt = (
        "# X14: LMC / LMC+ without each year's best long flush days\n\n| "
        + " | ".join(cols)
        + " |\n|"
        + "---|" * len(cols)
        + "\n"
    )
    txt += "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in df.iter_rows()) + "\n"
    print(txt)
    (OUT / "lmcplus_bestday.md").write_text(txt)
