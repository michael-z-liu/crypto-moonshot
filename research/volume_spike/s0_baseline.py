"""S0 of the CNN robustness study (docs/plans/cnn-robustness.md): baselines and sanity check.

    cd research/volume_spike
    export MOONSHOT_UNIVERSE=all
    uv run --group research python s0_baseline.py score base market path_all ens3
    uv run --group research python s0_baseline.py sanity

`score` puts each variant in the ledger (results/robustness/ledger.csv) and writes its 2022-24
card to results/robustness/cards/<variant>.txt. `ens3` is the equal-weight mean of the base,
market and path_all percentiles where all three are present, traded with the same rule
(mean >= 0.98).

`sanity` compares, for base, the research simulator (`m12_cnn_wf.portfolio_years`: one book per
signal year, additive 0.3% cost, no cash check) with the engine: per-year books without the cash
check (parity), per-year books as designed, and the scorecard's continuous book. Writes
results/robustness/S0_sanity.csv. 2022-24 only.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import polars as pl
import scorecard as sc
from common import CACHE

from crypto_moonshot.backtest.engine import BacktestConfig, run_backtest
from crypto_moonshot.portfolio import Book, EqualSlots, Sizing

MEMBERS = ("base", "market", "path_all")


def scores(name: str) -> tuple[np.ndarray, str]:
    """The variant's score array and its source description (part of the config hash)."""
    from m12_cnn_wf import VARIANTS

    if name == "ens3":
        ens = np.mean(np.stack([sc.wf_scores(n) for n in MEMBERS]), 0)  # NaN unless all present
        return ens, "mean pct of m12 " + ", ".join(MEMBERS) + " (cache_all/wf, 2 seeds each)"
    cfg = asdict(VARIANTS[name])
    cfg.pop("note")
    return sc.wf_scores(name), f"m12_cnn_wf {name} {cfg} (cache_all/wf/{name}.npz)"


def score(names: list[str]) -> None:
    cards = sc.OUT / "cards"
    cards.mkdir(parents=True, exist_ok=True)
    for name in names:
        s, source = scores(name)
        card = sc.scorecard(name, s, sc.Rule(), stage="S0", source=source)
        text = sc.show(card)
        print(text, flush=True)
        (cards / f"{name}.txt").write_text(text + "\n")


# ----------------------------------------------------------------------------- sanity


@dataclass(frozen=True)
class CashCapped:
    """Equal slots, the stake capped at the cash left: takes every entry the research took."""

    slots: int

    def stake(self, book: Book) -> float:
        return min(book.equity / self.slots, book.cash)


def engine_years(sig: pl.DataFrame, sizing: Sizing) -> dict[int, tuple[int, float]]:
    """One engine book per signal year (as `portfolio_years`), bp by signal year."""
    out = {}
    for year in sc.SELECT_YEARS:
        config = BacktestConfig(
            exit=sc.Rule().exit,
            sizing=sizing,
            max_positions=5,
            start=datetime(year, 1, 1, tzinfo=UTC),
            end=datetime(year + 1, 1, 1, tzinfo=UTC),
        )
        t = run_backtest(sig, sc.source(), config).trades
        out[year] = (t.height, float(t.select(pl.col("net_return").mean()).item()) * 1e4)
    return out


def sanity() -> None:
    from m12_cnn_wf import portfolio_years

    pct = sc.wf_scores("base")
    meta = pl.read_parquet(CACHE / "seq2_meta.parquet").with_columns(
        year=pl.col("open_time").dt.year(), day=pl.col("open_time").dt.date()
    )
    data = SimpleNamespace(meta=meta, fwd=np.load(CACHE / "fwd2.npy", mmap_mode="r"))
    research = {}
    for r in portfolio_years(data, np.nan_to_num(pct, nan=-1), 12, q=0.98):  # type: ignore[arg-type]
        if r["year"] in sc.SELECT_YEARS:
            research[int(str(r["year"]))] = (int(str(r["trades"])), float(str(r["mean_bp"])))
    sig = sc.signals(pct, sc.Rule())
    parity = engine_years(sig, CashCapped(5))
    per_year = engine_years(sig, EqualSlots(5))
    cont = sc.run_book(sig, sc.Rule(), sc.SELECT_END).years
    rows = []
    for y in sc.SELECT_YEARS:
        rows.append(
            {
                "year": y,
                "research_trades": research[y][0],
                "research_bp": round(research[y][1], 1),
                "parity_trades": parity[y][0],
                "parity_bp": round(parity[y][1], 1),
                "per_year_trades": per_year[y][0],
                "per_year_bp": round(per_year[y][1], 1),
                "continuous_trades": int(cont[y]["trades"]),
                "continuous_bp": round(cont[y]["bp"], 1),
            }
        )
    table = pl.DataFrame(rows)
    bp = [c for c in table.columns if c.endswith("_bp")]
    mean = {c: round(float(table.select(pl.col(c).mean()).item()), 1) for c in bp}
    table.write_csv(sc.OUT / "S0_sanity.csv")
    with pl.Config(tbl_cols=12, tbl_width_chars=200):
        print(table)
    print("mean 2022-24:", mean)


if __name__ == "__main__":
    if sys.argv[1] == "score":
        score(sys.argv[2:])
    elif sys.argv[1] == "sanity":
        sanity()
    else:
        raise SystemExit(__doc__)
