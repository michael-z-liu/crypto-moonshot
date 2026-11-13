"""M1: candidate counts per MA and trigger year, per entry family (no outcomes).

    uv run --group research python m1_counts.py

Reads `cache/candidates.parquet` and writes `results/M1_counts.md`: counts of candidates only,
which the holdout rule allows for 2025-26 too. Nothing about fills, exits or returns.
"""

from __future__ import annotations

import polars as pl
from common import CACHE, RESULTS
from events import TF_NAMES


def table(cands: pl.DataFrame, family: str, placebo: bool) -> str:
    sub = cands.filter(pl.col("family") == family, pl.col("placebo") == placebo)
    years = sorted(cands["year"].unique().to_list())
    wide = (
        sub.group_by("tf", "n", "year")
        .len()
        .pivot("year", index=["tf", "n"], values="len", sort_columns=True)
        .fill_null(0)
    )
    order = {tf: i for i, tf in enumerate(TF_NAMES)}
    wide = wide.with_columns(o=pl.col("tf").replace_strict(order)).sort("o", "n").drop("o")
    cols = [str(y) for y in years if str(y) in wide.columns]
    head = "| MA | " + " | ".join(cols) + " | total |"
    lines = [head, "|" + " --- |" * (len(cols) + 2)]
    for r in wide.iter_rows(named=True):
        vals = [r[c] for c in cols]
        lines.append(f"| {r['tf']} {r['n']} | " + " | ".join(map(str, vals)) + f" | {sum(vals)} |")
    tot = [int(wide[c].sum()) for c in cols]
    lines.append("| **all** | " + " | ".join(map(str, tot)) + f" | {sum(tot)} |")
    return "\n".join(lines)


def main() -> None:
    cands = pl.read_parquet(CACHE / "candidates.parquet")
    parts = ["# M1 candidate counts per MA and trigger year", ""]
    parts.append(
        "Candidates after the liquidity floor and with `atr_d` defined, by the UTC year of the "
        "trigger candle. Counts only; 2025-26 is the sealed holdout (counts are allowed)."
    )
    for placebo in (False, True):
        for family in ("limit", "touch"):
            kind = "placebo" if placebo else "study"
            name = "limit (approach-zone trigger)" if family == "limit" else "touch-market"
            parts += ["", f"## {kind} MAs, {name}", "", table(cands, family, placebo)]
    coins = cands.group_by("year").agg(pl.col("symbol").n_unique()).sort("year")
    parts += ["", "## Coins with at least one candidate", "", "| year | coins |", "| --- | --- |"]
    parts += [f"| {r['year']} | {r['symbol']} |" for r in coins.iter_rows(named=True)]
    out = RESULTS / "M1_counts.md"
    out.write_text("\n".join(parts) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
