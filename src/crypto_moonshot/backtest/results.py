"""Writing a run's result files: the format shared by backtests and replays.

A run folder holds `trades.parquet` and `trades.csv` (the trade list), `equity.csv` (the
mark-to-market curve), `yearly.csv` (metrics per calendar year) and `metrics.json` (a document
the caller builds: metrics, configuration, ...).
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl


def json_safe(value: Any) -> Any:
    """JSON-safe: NaN and infinity become null, datetimes ISO strings, timedeltas seconds,
    paths strings (recursively through dicts and lists)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, Path):
        return str(value)
    return value


def write_run_files(
    out: Path,
    trades: pl.DataFrame,
    equity: pl.DataFrame,
    yearly: pl.DataFrame,
    document: dict[str, Any],
) -> None:
    """Write a run's files into `out` (created if needed)."""
    out.mkdir(parents=True, exist_ok=True)
    trades.write_parquet(out / "trades.parquet")
    trades.write_csv(out / "trades.csv")
    equity.write_csv(out / "equity.csv")
    yearly.write_csv(out / "yearly.csv")
    (out / "metrics.json").write_text(json.dumps(json_safe(document), indent=2) + "\n", "utf-8")
