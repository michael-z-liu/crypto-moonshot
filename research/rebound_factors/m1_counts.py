"""M1: entry counts per year, and base rates of the labels on discovery entries only.

    uv run --group research python m1_counts.py

Reads `cache/panel.parquet` (counts, every year) and `cache/labels.parquet` (discovery labels).
It never opens `cache/sealed/`: for 2024-26 it writes counts of entries only. Writes
`results/m1_counts.md` and `results/m1_counts.csv`.
"""

from __future__ import annotations

import polars as pl
from common import DISCOVERY_END_US, LABELS, PANEL, RESULTS, SEALED_START_US


def main() -> None:
    panel = pl.read_parquet(PANEL)
    labels = pl.read_parquet(LABELS)  # D before 2024 only
    assert int(labels.select(pl.col("d_us").max()).item()) < SEALED_START_US
    disc = labels.filter(pl.col("d_us") <= DISCOVERY_END_US)
    counts = panel.group_by("year").agg(
        entries=pl.len(), coins=pl.col("symbol").n_unique(), median_liq=pl.col("liq").median()
    )
    rates = (
        disc.group_by("year")
        .agg(
            discovery_entries=pl.len(),
            hit10=pl.col("hit10").mean(),
            hit20=pl.col("hit20").mean(),
            hit_ns=pl.col("hit_ns").mean(),
            no_entry=(pl.col("reason_10") == "no_entry").sum(),
            stopped_early=pl.col("stopped_early").sum(),
            entry_late_1h=((pl.col("entry_t") - pl.col("d_us")) >= 3_600_000_000).sum(),
        )
        .with_columns(pl.col("^hit.*$").round(4))
    )
    table = counts.join(rates, on="year", how="left").sort("year")
    RESULTS.mkdir(parents=True, exist_ok=True)
    table.write_csv(RESULTS / "m1_counts.csv")

    splits = panel.group_by("split").agg(entries=pl.len(), coins=pl.col("symbol").n_unique())
    order = {"discovery": 0, "gap": 1, "confirm": 2, "unused": 3}
    splits = splits.sort(pl.col("split").replace_strict(order))

    def pct(x: float | None) -> str:
        return "" if x is None else f"{100 * x:.1f}%"

    lines = [
        "# M1 entry counts and discovery base rates",
        "",
        "Entries are coin-days at UTC midnight `D` (protocol in `docs/plans/rebound-factors.md`;",
        "definitions in the docstrings of `build_panel.py` and `build_labels.py`). Counts of",
        "entries cover every year; label base rates are for **discovery** entries only",
        "(D 2020-01-31 .. 2023-12-01). 2024-26 is sealed: counts only.",
        "",
        "| year | entries | coins | median liq ($M) | discovery entries | hit10 | hit20 | hit_ns |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in table.iter_rows(named=True):
        de = "" if r["discovery_entries"] is None else str(r["discovery_entries"])
        lines.append(
            f"| {r['year']} | {r['entries']} | {r['coins']} | {r['median_liq'] / 1e6:.1f} | {de} "
            f"| {pct(r['hit10'])} | {pct(r['hit20'])} | {pct(r['hit_ns'])} |"
        )
    total = panel.height
    lines += [
        f"| **all** | {total} | {panel['symbol'].n_unique()} | | {disc.height} | | | |",
        "",
        "## By split",
        "",
        "| split | D range | entries | coins |",
        "| --- | --- | --- | --- |",
    ]
    ranges = {
        "discovery": "2020-01-31 .. 2023-12-01",
        "gap": "2023-12-02 .. 2023-12-31 (unused)",
        "confirm": "2024-01-01 .. 2026-10-02 (sealed)",
        "unused": "2026-10-03 (complete window, after the confirm range; sealed)",
    }
    for r in splits.iter_rows(named=True):
        lines.append(f"| {r['split']} | {ranges[r['split']]} | {r['entries']} | {r['coins']} |")
    flags = table.filter(pl.col("discovery_entries").is_not_null())
    lines += [
        "",
        "## Gaps and delistings (discovery entries)",
        "",
        "Missing 5m candles are simply absent. If a coin's candles stop more than a day before",
        "the window end (a delisting), the entry is kept, the trade exits at the last close in",
        "the window (reason `series_end`) and `stopped_early` is set. An entry with no candle in",
        "the window would have reason `no_entry` and null outcomes. An entry needs a candle in",
        "the day before `D` for its price `P`, so a coin delisted days earlier is not an entry.",
        "These flags are not reported for 2024-26 (they come from the sealed labels).",
        "",
        "| year | stopped early | no entry | entry 1 h or more after D |",
        "| --- | --- | --- | --- |",
    ]
    for r in flags.iter_rows(named=True):
        lines.append(
            f"| {r['year']} | {r['stopped_early']} | {r['no_entry']} | {r['entry_late_1h']} |"
        )
    lines.append("")
    (RESULTS / "m1_counts.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
