from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from crypto_moonshot.broker.base import ExitPlan
from crypto_moonshot.strategy.scores import (
    BASE_CNN,
    ScoreStrategy,
    convert_wf_npz,
    read_scores,
    write_scores,
)

T = datetime(2024, 3, 1, tzinfo=UTC)
M5 = timedelta(minutes=5)


def scores(*rows: tuple[str, int, float | None], **extra: list[object]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": [r[0] for r in rows],
            "open_time": [T + r[1] * M5 for r in rows],
            "score": [r[2] for r in rows],
            **extra,
        },
        schema_overrides={"open_time": pl.Datetime("us", "UTC"), "score": pl.Float64},
    )


def test_threshold_and_priority() -> None:
    frame = scores(
        ("AUSDT", 0, 0.97),
        ("BUSDT", 0, 0.99),
        ("CUSDT", 0, 0.98),
        ("AUSDT", 1, None),
        ("BUSDT", 1, float("nan")),
        ("CUSDT", 2, 1.0),
        model=["m"] * 6,
    )
    signals = ScoreStrategy(threshold=0.98).signals(frame)
    assert signals.frame.select("symbol", "open_time", "priority").rows() == [
        ("BUSDT", T, -0.99),
        ("CUSDT", T, -0.98),
        ("CUSDT", T + 2 * M5, -1.0),
    ]
    assert signals.frame.columns == ["symbol", "open_time", "priority", "score", "model"]
    assert signals.exit == ExitPlan(max_hold=timedelta(hours=1))


def test_base_cnn_preset() -> None:
    assert BASE_CNN.threshold == 0.98
    assert BASE_CNN.exit == ExitPlan(max_hold=timedelta(hours=1))


def test_score_file_round_trip(tmp_path: Path) -> None:
    frame = scores(("AUSDT", 0, 0.5), ("BUSDT", 3, 0.99), event=[7, 9])
    path = tmp_path / "s" / "scores.parquet"
    write_scores(frame, path)
    back = read_scores(path)
    assert back.equals(frame)
    assert not list(path.parent.glob(".*"))  # no temporary file left behind
    signals = ScoreStrategy(threshold=0.9).load(path)
    assert signals.frame["event"].to_list() == [9]


def test_score_files_are_validated() -> None:
    with pytest.raises(ValueError, match="missing columns: score"):
        ScoreStrategy(0.5).signals(scores(("AUSDT", 0, 0.5)).drop("score"))
    with pytest.raises(ValueError, match="duplicate"):
        ScoreStrategy(0.5).signals(scores(("AUSDT", 0, 0.5), ("AUSDT", 0, 0.6)))
    naive = scores(("AUSDT", 0, 0.5)).with_columns(pl.col("open_time").dt.replace_time_zone(None))
    with pytest.raises(ValueError, match="timezone-aware"):
        ScoreStrategy(0.5).signals(naive)
    # Another time unit or zone is normalised to µs UTC.
    ms = scores(("AUSDT", 0, 0.5)).with_columns(
        pl.col("open_time").dt.cast_time_unit("ms").dt.convert_time_zone("Europe/Paris")
    )
    assert ScoreStrategy(0.5).signals(ms).frame["open_time"].to_list() == [T]


def write_wf(tmp_path: Path, n: int = 5) -> tuple[Path, Path]:
    meta = pl.DataFrame(
        {
            "event": list(range(10, 10 + n)),
            "symbol": ["AUSDT", "BUSDT", "AUSDT", "CUSDT", "BUSDT"][:n],
            "open_time": [T + k * M5 for k in range(n)],
            "qv_base": [1.0] * n,
        },
        schema_overrides={"open_time": pl.Datetime("us", "UTC")},
    )
    meta_path = tmp_path / "seq2_meta.parquet"
    meta.write_parquet(meta_path)
    npz_path = tmp_path / "base.npz"
    np.savez(
        npz_path,
        pct=np.array([np.nan, 0.5, 0.99, np.nan, 0.985], dtype=np.float32)[:n],
        raw=np.arange(n, dtype=np.float32),
        pct_seed=np.zeros((2, n), dtype=np.float32),
    )
    return npz_path, meta_path


def test_convert_wf_npz(tmp_path: Path) -> None:
    npz_path, meta_path = write_wf(tmp_path)
    out = tmp_path / "cache" / "base.parquet"
    frame = convert_wf_npz(npz_path, meta_path, out)
    assert frame.columns == ["symbol", "open_time", "score", "event", "raw"]
    assert frame.select("symbol", "event", "raw").rows() == [
        ("BUSDT", 11, 1.0),
        ("AUSDT", 12, 2.0),
        ("BUSDT", 14, 4.0),
    ]
    assert frame["score"].to_list() == pytest.approx([0.5, 0.99, 0.985])
    assert read_scores(out).equals(frame)
    signals = BASE_CNN.load(out)
    assert signals.frame.select("symbol", "open_time").rows() == [
        ("AUSDT", T + 2 * M5),
        ("BUSDT", T + 4 * M5),
    ]


def test_convert_wf_npz_checks_alignment(tmp_path: Path) -> None:
    npz_path, meta_path = write_wf(tmp_path)
    pl.read_parquet(meta_path).head(4).write_parquet(meta_path)
    with pytest.raises(ValueError, match="expected \\(4,\\)"):
        convert_wf_npz(npz_path, meta_path, tmp_path / "out.parquet")
    with pytest.raises(ValueError, match="no 'missing' array"):
        convert_wf_npz(npz_path, meta_path, tmp_path / "out.parquet", key="missing")
