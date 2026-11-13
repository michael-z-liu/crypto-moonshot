import os
from datetime import timedelta
from pathlib import Path

import polars as pl
import pytest
from strategy_fakes import LONE, SPIKE, at, write_capitulation_store

from crypto_moonshot.broker.base import ExitPlan
from crypto_moonshot.strategy.capitulation import F2, R2, CapitulationRule
from crypto_moonshot.strategy.compute import candidates, compute_signals

SYMBOLS = ["DIPUSDT", "SHALLOWUSDT", "THINUSDT"]


def table(**columns: list[object]) -> pl.DataFrame:
    n = len(columns["ret_4h"])
    base = {
        "symbol": [f"S{k}USDT" for k in range(n)],
        "open_time": [at(SPIKE)] * n,
        "vol_base": [1.0] * n,
        "vol_mult": [10.0] * n,
        "qv_base": [50_000.0] * n,
        "ret_bar": [0.0] * n,
        "ret_4h": [-0.2] * n,
        "ret_24h": [0.0] * n,
        "dist_hi_7d": [0.0] * n,
        "btc_vol_mult": [8.0] * n,
    }
    return pl.DataFrame(
        {**base, **columns}, schema_overrides={"open_time": pl.Datetime("us", "UTC")}
    )


def test_the_rule_masks_at_its_thresholds() -> None:
    cands = table(
        ret_4h=[-0.10, -0.0999, -0.3, -0.3, -0.3, None, -0.3, -0.3],
        qv_base=[20_000.0, 50_000.0, 19_999.0, 1_000.0, 50_000.0, 50_000.0, 50_000.0, 50_000.0],
        btc_vol_mult=[6.0, 8.0, 8.0, 8.0, 5.99, 8.0, None, 8.0],
    )
    assert F2.select(cands)["symbol"].to_list() == ["S7USDT", "S0USDT"]
    assert sorted(R2.select(cands)["symbol"].to_list()) == ["S0USDT", "S2USDT", "S3USDT", "S7USDT"]
    no_btc = CapitulationRule("x", min_qv_base=20_000, exit=F2.exit, min_btc_vol_mult=None)
    assert sorted(no_btc.select(cands)["symbol"].to_list()) == [
        "S0USDT",
        "S4USDT",
        "S6USDT",
        "S7USDT",
    ]


def test_priority_is_ret_4h_deepest_first() -> None:
    cands = table(ret_4h=[-0.12, -0.30, -0.20])
    signals = F2.select(cands)
    assert signals["symbol"].to_list() == ["S1USDT", "S2USDT", "S0USDT"]
    assert signals["priority"].to_list() == [-0.30, -0.20, -0.12]
    assert signals.columns[:3] == ["symbol", "open_time", "priority"]
    assert "btc_vol_mult" in signals.columns


def test_presets() -> None:
    assert (F2.min_qv_base, F2.max_ret_4h, F2.min_btc_vol_mult) == (20_000, -0.10, 6.0)
    assert F2.exit == ExitPlan(max_hold=timedelta(hours=1))
    assert (R2.min_qv_base, R2.max_ret_4h, R2.min_btc_vol_mult) == (1_000, -0.10, 6.0)
    assert R2.exit == ExitPlan(take_profit=0.20, max_hold=timedelta(days=3))


@pytest.fixture(scope="module")
def root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("store")
    write_capitulation_store(path)
    return path


def test_f2_and_r2_on_a_store_with_btc_context(root: Path) -> None:
    f2 = compute_signals(F2, SYMBOLS, root, workers=1).signals
    assert f2.frame.select("symbol", "open_time").rows() == [("DIPUSDT", at(SPIKE))]
    assert f2.exit == F2.exit
    row = f2.frame.row(0, named=True)
    assert row["priority"] == pytest.approx(-0.15)
    assert row["btc_vol_mult"] == pytest.approx(10.0)
    assert row["vol_mult"] == pytest.approx(20.0)
    r2 = compute_signals(R2, SYMBOLS, root, workers=1).signals
    assert r2.frame.select("symbol", "open_time").rows() == [
        ("DIPUSDT", at(SPIKE)),
        ("THINUSDT", at(SPIKE)),
    ]


def test_candidates_carry_btc_context_even_where_btc_is_quiet(root: Path) -> None:
    table = candidates(F2, ["DIPUSDT"], root, workers=1)
    assert table["open_time"].to_list() == [at(SPIKE), at(LONE)]
    assert table["btc_vol_mult"].to_list() == pytest.approx([10.0, 1.0])


def test_until_hides_later_candles(root: Path) -> None:
    assert len(compute_signals(F2, SYMBOLS, root, until=at(SPIKE), workers=1).signals) == 0
    assert len(compute_signals(F2, SYMBOLS, root, until=at(SPIKE + 1), workers=1).signals) == 1


def test_parallel_workers_give_the_same_candidates(root: Path) -> None:
    serial = candidates(R2, SYMBOLS, root, workers=1)
    parallel = candidates(R2, SYMBOLS, root, workers=2)
    assert parallel.equals(serial)


def test_signals_are_cached_by_rule_universe_and_data(root: Path, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    first = compute_signals(F2, SYMBOLS, root, workers=1, cache_dir=cache)
    again = compute_signals(F2, SYMBOLS, root, workers=1, cache_dir=cache)
    assert (first.cached, again.cached) == (False, True)
    assert again.path == first.path
    assert again.path is not None
    assert again.path.with_suffix(".json").exists()
    assert again.signals.frame.equals(first.signals.frame)
    # Another rule, universe or data end is another entry.
    assert not compute_signals(R2, SYMBOLS, root, workers=1, cache_dir=cache).cached
    assert not compute_signals(F2, SYMBOLS[:2], root, workers=1, cache_dir=cache).cached
    later = compute_signals(F2, SYMBOLS, root, until=at(SPIKE + 1), workers=1, cache_dir=cache)
    assert not later.cached
    # Changed data is recomputed.
    month = next((root / "klines/spot/interval=5m/symbol=DIPUSDT").glob("*.parquet"))
    stat = month.stat()
    os.utime(month, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert not compute_signals(F2, SYMBOLS, root, workers=1, cache_dir=cache).cached


def test_an_empty_universe_gives_no_signals(root: Path) -> None:
    signals = compute_signals(F2, [], root, workers=1).signals
    assert len(signals) == 0
    assert signals.frame.columns[:3] == ["symbol", "open_time", "priority"]
