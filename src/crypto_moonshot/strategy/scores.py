"""Signals from precomputed model scores: a score file, a threshold, and an exit plan.

Score file format (Parquet), one row per scored candle:

| Column      | Type                    | Meaning                                              |
| ----------- | ----------------------- | ---------------------------------------------------- |
| `symbol`    | `String`                | Binance symbol, e.g. `BTCUSDT`                       |
| `open_time` | `Datetime("us", "UTC")` | open time of the signal candle *t* (scored on its close) |
| `score`     | `Float64`               | higher is better; null or NaN means not scored       |
| any other   | any                     | carried into the signals and the trade list          |

`(symbol, open_time)` is unique. A `ScoreStrategy` enters every candle whose score is at least
its threshold, with `priority = -score` (the best score first).

`convert_wf_npz` writes such a file from the volume-spike research's walk-forward CNN output
(`cache/volume_spike/wf/<variant>.npz`, cnn-replication `m12_cnn_wf.py`): its `pct` array holds,
for each row of `seq2_meta.parquet` in order, the score's percentile among the fold's training
scores, NaN outside the test years. The base CNN traded `pct >= 0.98` with a 1-hour hold
(`BASE_CNN`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl

from crypto_moonshot.broker.base import ExitPlan
from crypto_moonshot.models import Interval
from crypto_moonshot.strategy.base import Signals

UTC_TIME = pl.Datetime("us", "UTC")
SCORE_COLUMNS: dict[str, pl.DataType] = {
    "symbol": pl.String(),
    "open_time": UTC_TIME,
    "score": pl.Float64(),
}


def _check_scores(frame: pl.DataFrame) -> pl.DataFrame:
    """Validate a score frame; returns it with `open_time` in µs UTC and `score` as Float64."""
    missing = [c for c in SCORE_COLUMNS if c not in frame.columns]
    if missing:
        msg = f"score frame is missing columns: {', '.join(missing)}"
        raise ValueError(msg)
    dtype = frame.schema["open_time"]
    if not isinstance(dtype, pl.Datetime) or dtype.time_zone is None:
        msg = f"score open_time must be a timezone-aware datetime, got {dtype}"
        raise ValueError(msg)
    if frame.schema["symbol"] != pl.String:
        msg = f"score symbol must be a string, got {frame.schema['symbol']}"
        raise ValueError(msg)
    frame = frame.with_columns(
        pl.col("open_time").dt.convert_time_zone("UTC").dt.cast_time_unit("us"),
        pl.col("score").cast(pl.Float64),
    )
    if frame.select(pl.struct("symbol", "open_time").is_duplicated().any()).item():
        msg = "score frame has duplicate (symbol, open_time) rows"
        raise ValueError(msg)
    return frame


def read_scores(path: Path) -> pl.DataFrame:
    """Read and validate a score file."""
    return _check_scores(pl.read_parquet(path))


def write_scores(frame: pl.DataFrame, path: Path) -> None:
    """Validate and atomically write a score file (via a hidden temporary file)."""
    frame = _check_scores(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp"
    frame.write_parquet(tmp)
    os.replace(tmp, path)


@dataclass(frozen=True, slots=True)
class ScoreStrategy:
    """Enter where `score >= threshold`, best score first; every entry carries `exit`."""

    threshold: float
    exit: ExitPlan = field(default_factory=lambda: ExitPlan(max_hold=timedelta(hours=1)))
    interval: Interval = Interval.M5
    name: str = "scores"

    def signals(self, scores: pl.DataFrame) -> Signals:
        """Signals from a score frame (the score file format)."""
        scores = _check_scores(scores)
        frame = (
            scores.filter(pl.col("score").fill_nan(None) >= self.threshold)
            .with_columns(priority=-pl.col("score"))
            .select(
                "symbol", "open_time", "priority", pl.exclude("symbol", "open_time", "priority")
            )
            .sort("open_time", "priority", "symbol")
        )
        return Signals(frame, self.exit, self.interval)

    def load(self, path: Path) -> Signals:
        """Signals from a score file."""
        return self.signals(read_scores(path))


BASE_CNN = ScoreStrategy(threshold=0.98, name="base_cnn")
"""The walk-forward base CNN's trade rule: top 2% of the fold's training scores, 1-hour hold."""


def convert_wf_npz(
    npz_path: Path, meta_path: Path, out_path: Path, *, key: str = "pct"
) -> pl.DataFrame:
    """Write a score file from a research walk-forward `.npz` and its `seq2_meta.parquet`.

    Row *i* of every 1-D array in the npz belongs to row *i* of the meta table. `key` becomes
    `score`; other 1-D arrays of the same length (e.g. `raw`) and the meta's `event` column (the
    row in the research event table), when present, are kept as extra columns. Rows with a NaN
    score (outside the walk-forward test years) are dropped. Returns the written frame.
    """
    meta = pl.read_parquet(meta_path)
    for column in ("symbol", "open_time"):
        if column not in meta.columns:
            msg = f"{meta_path} has no {column!r} column"
            raise ValueError(msg)
    with np.load(npz_path) as npz:
        if key not in npz.files:
            msg = f"{npz_path} has no {key!r} array (has: {', '.join(npz.files)})"
            raise ValueError(msg)
        arrays = {name: npz[name] for name in npz.files}
    score = arrays.pop(key)
    if score.ndim != 1 or len(score) != meta.height:
        msg = (
            f"{npz_path}[{key!r}] has shape {score.shape}, expected ({meta.height},) to match "
            f"the rows of {meta_path}"
        )
        raise ValueError(msg)
    keep = ["symbol", "open_time", *(["event"] if "event" in meta.columns else [])]
    extra = {
        name: pl.Series(name, values.astype(np.float64))
        for name, values in sorted(arrays.items())
        if values.ndim == 1 and len(values) == meta.height and name not in keep
    }
    frame = (
        meta.select(keep)
        .with_columns(score=pl.Series("score", score.astype(np.float64)), **extra)
        .filter(pl.col("score").is_not_nan())
        .select(
            "symbol", "open_time", "score", *[c for c in keep if c not in SCORE_COLUMNS], *extra
        )
    )
    write_scores(frame, out_path)
    return read_scores(out_path)
