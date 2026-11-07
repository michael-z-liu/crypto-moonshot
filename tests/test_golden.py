"""Golden tests: the backtest engine against the research results, on the real data.

Opt-in: every test here is marked `golden`, which `pyproject.toml` deselects by default. Run them
with `uv run pytest -m golden` (add `-s` for the per-year tables). They read the shared data
folder (`MOONSHOT_DATA_DIR`, default `/Users/mike/code/crypto-moonshot/data`) and skip when the
candles or research caches they need are missing. They never write there: signal and score
caches go to this worktree's `.cache/`.

References and tolerances were fixed before the runs (plan `docs/plans/backtest-engine.md`, M4).
Each case has two kinds of test:

- **golden**: the engine as designed against the published research numbers, with the agreed
  tolerance;
- **parity**: the engine fed the research's own signals (from its event tables) with its cash
  check neutralised, i.e. with the documented, deliberate differences removed. The trade sets
  then match the research books exactly, so per-year counts must be equal and the remaining gap
  is the cost model (multiplicative here, additive 0.3% in the research).

`docs/backtest-validation.md` explains and measures every difference.
"""

from __future__ import annotations

import os
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from crypto_moonshot.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from crypto_moonshot.backtest.source import ParquetCandleSource
from crypto_moonshot.data.universe import universe_symbols
from crypto_moonshot.portfolio import CashCapped, EqualSlots, FixedStake, Sizing
from crypto_moonshot.strategy.base import Signals
from crypto_moonshot.strategy.capitulation import F2, R2, R2_LAST_ENTRY, CapitulationRule
from crypto_moonshot.strategy.compute import compute_signals
from crypto_moonshot.strategy.scores import BASE_CNN, convert_wf_npz

pytestmark = pytest.mark.golden

ROOT = Path(os.environ.get("MOONSHOT_DATA_DIR", "/Users/mike/code/crypto-moonshot/data"))
CACHE = Path(__file__).resolve().parents[1] / ".cache"
UNTIL = datetime(2026, 11, 2, tzinfo=UTC)
"""The data end of every research run here (exclusive)."""
KLINES = "klines/spot/interval=5m"
KLINES_1M = "klines/spot/interval=1m"
POOL_CACHE = "cache/volume_spike"
ALL_CACHE = "cache/volume_spike_all"

YEARS = range(2020, 2027)
CNN_YEARS = range(2022, 2027)

# Case 1: F2, PR #8 (`m6_test.py`), net bp per trade and trades by entry year.
F2_BP = dict(zip(YEARS, (222, 319, 223, 287, 284, -44, 187), strict=True))
F2_TRADES = dict(zip(YEARS, (114, 243, 126, 74, 156, 133, 45), strict=True))
# Case 2: R2 capitulation pick, cnn-replication B2 (`m18_longshot.py`), $ P&L by signal year.
R2_USD = dict(zip(YEARS, (845, 8_651, -1_082, 2_967, 7_584, 2_479, 1_337), strict=True))
# The B2 book's trade counts: not in the published table; from `m18_longshot.book` rerun over
# the research caches (it reproduces the $ row above to the dollar).
R2_TRADES = dict(zip(YEARS, (711, 2_668, 1_467, 1_013, 2_290, 1_170, 649), strict=True))
# Case 3: m21 fill check, what 1m candles and a close at the target cost over 2020..2026.
R2_1M_DROP = 1_678.0
# Case 4: base CNN, PR #8 walk-forward (`m12_cnn_wf.py portfolio_years`), bp by year, and the
# per-year books' trade counts (rerun of `portfolio_years` over the research caches).
CNN_BP = dict(zip(CNN_YEARS, (334, 170, 57, 67, 53), strict=True))
CNN_TRADES = dict(zip(CNN_YEARS, (81, 100, 197, 233, 135), strict=True))


def require(*paths: str) -> None:
    """Skip unless every path exists under the data root."""
    missing = [p for p in paths if not (ROOT / p).exists()]
    if missing:
        pytest.skip(f"MOONSHOT_DATA_DIR={ROOT} lacks {', '.join(missing)}")


def per_year(
    trades: pl.DataFrame, value: pl.Expr, *, by: str = "entry_time"
) -> dict[int, tuple[int, float]]:
    """Trades and `value` (an aggregation) by the year of column `by`."""
    table = (
        trades.group_by(pl.col(by).dt.year().alias("year")).agg(n=pl.len(), v=value).sort("year")
    )
    return {int(y): (int(n), float(v)) for y, n, v in table.iter_rows()}


def mean_bp(trades: pl.DataFrame, *, by: str = "entry_time") -> dict[int, tuple[int, float]]:
    return per_year(trades, pl.col("net_return").mean() * 1e4, by=by)


def usd(trades: pl.DataFrame) -> dict[int, tuple[int, float]]:
    return per_year(trades, pl.col("pnl").sum(), by="signal_time")


def total(trades: pl.DataFrame, column: str = "pnl") -> float:
    return float(trades.select(pl.col(column).sum()).item())


def show(title: str, got: dict[int, tuple[int, float]], ref: dict[int, float]) -> None:
    print(f"\n{title}")
    for year, value in ref.items():
        n, v = got.get(year, (0, float("nan")))
        print(f"  {year}  trades {n:5d}  engine {v:+10.1f}  reference {value:+9.0f}")


def research_signals(cache: str, rule: CapitulationRule, end: datetime) -> pl.DataFrame:
    """The rule's signals from the research's own event table (read only)."""
    mask = (
        pl.col("is_spike")
        & (pl.col("ret_4h") <= rule.max_ret_4h)
        & (pl.col("qv_base") >= rule.min_qv_base)
        & (pl.col("btc_vol_mult") >= rule.min_btc_vol_mult)
    ).fill_null(False)
    return (
        pl.scan_parquet(ROOT / cache / "events.parquet")
        .select("symbol", "open_time", "is_spike", "ret_4h", "qv_base", "btc_vol_mult")
        .filter(mask & (pl.col("open_time") < end))
        .select(
            pl.col("symbol").cast(pl.String),
            pl.col("open_time").dt.convert_time_zone("UTC").dt.cast_time_unit("us"),
            priority=pl.col("ret_4h"),
        )
        .collect()
    )


@pytest.fixture(scope="module")
def source() -> ParquetCandleSource:
    require(KLINES)
    return ParquetCandleSource(ROOT, until=UNTIL)


def rule_signals(rule: CapitulationRule, universe: str) -> Signals:
    lists = ["top200.txt", *(["delisted.txt", "outside_pool.txt"] if universe == "all" else [])]
    require(KLINES, *lists)
    symbols = universe_symbols(universe, ROOT)
    return compute_signals(rule, symbols, ROOT, until=UNTIL, cache_dir=CACHE).signals


@pytest.fixture(scope="module")
def f2() -> Signals:
    return rule_signals(F2, "pool")


@pytest.fixture(scope="module")
def r2() -> Signals:
    return rule_signals(R2, "all")


def f2_config(sizing: Sizing) -> BacktestConfig:
    return BacktestConfig(exit=F2.exit, sizing=sizing, max_positions=5)


def r2_config(*, fine: bool = False, capital: float = 10_000.0) -> BacktestConfig:
    return BacktestConfig(
        exit=replace(R2.exit, tp_fill="close") if fine else R2.exit,
        sizing=FixedStake(50.0),
        max_positions=200,
        initial_capital=capital,
        end=R2_LAST_ENTRY,
        use_fine=fine,
    )


@pytest.fixture(scope="module")
def r2_coarse(r2: Signals, source: ParquetCandleSource) -> BacktestResult:
    return run_backtest(r2.frame, source, r2_config())


# --------------------------------------------------------------------------- case 1: F2


def test_f2_pool(f2: Signals, source: ParquetCandleSource) -> None:
    result = run_backtest(f2.frame, source, f2_config(EqualSlots(5)))
    got = mean_bp(result.trades)
    show("F2, pool, 5 slots, 1h: bp per trade by entry year", got, dict(F2_BP))
    print("  skipped", result.skipped)
    assert sorted(got) == list(YEARS)
    for year in YEARS:
        n, bp = got[year]
        assert abs(bp - F2_BP[year]) <= 15, year
        assert abs(n - F2_TRADES[year]) <= 0.05 * F2_TRADES[year], year


def test_f2_parity_with_research_book(source: ParquetCandleSource) -> None:
    """Research signals (no gap look-ahead filter difference) and no cash check: the research
    book's trades, year by year; bp within 2 (costs: about -0.7 bp, plus the reference's
    rounding)."""
    require(f"{POOL_CACHE}/events.parquet")
    signals = research_signals(POOL_CACHE, F2, UNTIL)
    result = run_backtest(signals, source, f2_config(CashCapped(5)))
    got = mean_bp(result.trades)
    show("F2 parity: research signals, no cash check", got, dict(F2_BP))
    assert result.skipped["cash"] == 0
    for year in YEARS:
        n, bp = got[year]
        assert n == F2_TRADES[year], year
        assert abs(bp - F2_BP[year]) <= 2, year


# --------------------------------------------------------------------------- case 2: R2


def test_r2_all(r2_coarse: BacktestResult) -> None:
    got = usd(r2_coarse.trades)
    show("R2, all coins, $50, +20% or 3 days: $ P&L by signal year", got, dict(R2_USD))
    print("  skipped", r2_coarse.skipped)
    assert sorted(got) == list(YEARS)
    for year in YEARS:
        ref = R2_USD[year]
        assert abs(got[year][1] - ref) <= max(0.05 * abs(ref), 150), year


def test_r2_parity_with_research_book(source: ParquetCandleSource) -> None:
    """Research signals and ample cash: the B2 book's trades, year by year; $ within 40 a year
    (costs: -$27 to +$3 a year)."""
    require(f"{ALL_CACHE}/events.parquet")
    signals = research_signals(ALL_CACHE, R2, R2_LAST_ENTRY)
    result = run_backtest(signals, source, r2_config(capital=1e9))
    got = usd(result.trades)
    show("R2 parity: research signals, ample cash", got, dict(R2_USD))
    for year in YEARS:
        n, pnl = got[year]
        assert n == R2_TRADES[year], year
        assert abs(pnl - R2_USD[year]) <= 40, year


# --------------------------------------------------------------------------- case 3: R2 on 1m


def test_r2_one_minute_fills(
    r2: Signals, r2_coarse: BacktestResult, source: ParquetCandleSource
) -> None:
    """1m fills with the `close` rule: the total falls by m21's $1,678 +-25% on the trades both
    books take. m21 re-priced the 5m book's take-profit exits and kept its trades; the engine
    re-runs the book, and a take profit that no 1m close confirms holds its slot until the time
    exit, so the 1m book also skips some later signals. That part is measured separately."""
    require(KLINES_1M)
    fine = run_backtest(r2.frame, source, r2_config(fine=True))
    coarse_t, fine_t = r2_coarse.trades, fine.trades
    key = ["symbol", "entry_time"]
    both = coarse_t.select(*key, "pnl").join(fine_t.select(*key, fine_pnl="pnl"), on=key)
    repricing = total(both) - total(both, "fine_pnl")
    drop = total(coarse_t) - total(fine_t)
    only_coarse = coarse_t.join(fine_t, on=key, how="anti")
    only_fine = fine_t.join(coarse_t, on=key, how="anti")
    rerun = drop - repricing
    print(
        f"\nR2 1m fills: total drop ${drop:,.0f} = re-pricing of {both.height} common trades "
        f"${repricing:,.0f} + book re-run ${rerun:,.0f} ({only_coarse.height} trades only on 5m "
        f"${total(only_coarse):,.0f}, {only_fine.height} only on 1m "
        f"${total(only_fine):,.0f}); reference ${R2_1M_DROP:,.0f}"
    )
    print("  skipped 5m", r2_coarse.skipped, "\n  skipped 1m", fine.skipped)
    assert abs(repricing - R2_1M_DROP) <= 0.25 * R2_1M_DROP
    # The documented book effect: confirmed exits come later, so slots stay full longer.
    assert fine.skipped["slots_full"] > r2_coarse.skipped["slots_full"]
    assert rerun == pytest.approx(total(only_coarse) - total(only_fine), abs=1e-6)


# --------------------------------------------------------------------------- case 4: base CNN


@pytest.fixture(scope="module")
def cnn() -> Signals:
    npz, meta = f"{POOL_CACHE}/wf/base.npz", f"{POOL_CACHE}/seq2_meta.parquet"
    require(npz, meta)
    out = CACHE / "scores" / "base.parquet"
    convert_wf_npz(ROOT / npz, ROOT / meta, out)
    return BASE_CNN.load(out)


def cnn_years(
    cnn: Signals, source: ParquetCandleSource, sizing: Sizing
) -> tuple[dict[int, tuple[int, float]], int]:
    """One book per signal year, as the research (`portfolio_years`); and the cash skips."""
    got: dict[int, tuple[int, float]] = {}
    cash = 0
    for year in CNN_YEARS:
        config = BacktestConfig(
            exit=cnn.exit,
            sizing=sizing,
            max_positions=5,
            start=datetime(year, 1, 1, tzinfo=UTC),
            end=datetime(year + 1, 1, 1, tzinfo=UTC),
        )
        result = run_backtest(cnn.frame, source, config)
        got[year] = mean_bp(result.trades, by="signal_time")[year]
        cash += result.skipped["cash"]
    return got, cash


def test_base_cnn(cnn: Signals, source: ParquetCandleSource) -> None:
    got, cash = cnn_years(cnn, source, EqualSlots(5))
    show("Base CNN, top 2%, 5 slots, 1h, one book per year: bp per trade", got, dict(CNN_BP))
    print("  cash skips", cash)
    for year in CNN_YEARS:
        assert abs(got[year][1] - CNN_BP[year]) <= 20, year


def test_base_cnn_parity_with_research_books(cnn: Signals, source: ParquetCandleSource) -> None:
    """No cash check: the research books' trades; bp within 2 (costs: -0.2 to -1.0 bp)."""
    got, cash = cnn_years(cnn, source, CashCapped(5))
    show("Base CNN parity: no cash check", got, dict(CNN_BP))
    assert cash == 0
    for year in CNN_YEARS:
        n, bp = got[year]
        assert n == CNN_TRADES[year], year
        assert abs(bp - CNN_BP[year]) <= 2, year
