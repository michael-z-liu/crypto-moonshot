# ruff: noqa: E501  (markdown text and table rows read better unsplit)
"""M7: the forward test of the frozen rule with the market gate (protocol in the plan).

    uv run --group research python forward.py                       # real run, writes results/
    uv run --group research python forward.py --dry-run-from 2026-03-01
    uv run --group research python forward.py --list-sync-symbols   # prints the symbols to sync

What it does. Forward entries are the coin-days with `D` from 2026-11-02 (the data end of every
study so far) to the day before this run's own data end: the last complete UTC day of BTCUSDT in
the candle store (`MOONSHOT_DATA_DIR`). The study's builders stop at `common.DATA_END`; this
script calls their pure functions (`build_panel.panel_rows`, `build_labels.label_entry`,
`build_factors.coin_rows` / `assemble`, `build_exclusions.breaks` / `excluded_entries`) with its
own end, reading all candles (the factors need the history before 2026-11-02), and writes its
tables only to `cache/forward/` (a dry run: `cache/forward_dryrun/`), never to the study's cache
files. Entries are the same: liquidity at least $1M over the previous 30 days, at least 30 days
listed, the same labels, costs, exclusions and book. An entry's label exists only when its
30-day window is complete by the data end; the others are *pending* (their picks are listed
without outcomes).

Scoring is `model.score_period` with the spec's training rule: the percentile reference is the
transparent score of every study entry (discovery, gap, confirm and the unused last entry, minus
exclusions) whose 30-day window ended by the forward period's start (`D + 30 d <= start`), as
M5 does for 1 January of each year. One reference for the whole forward period (no retraining
inside it). The gate (`m7_gate.gate_open_expr`: `m_btc_sma50 > 0`) is applied to the picks
before the book. Arm A: gate open against closed on every usable entry. Arm B: the frozen rule on
gate-open days, the same 10-slot book, the stress run (0.3% a side), and same-day random books
drawn on the open days (100 draws, `SEED`).

Readout. `interim` until at least 180 days of forward entries have complete windows and arm B has
at least 100 trades (10% stop, base). At the first run meeting both, the protocol's **Confirmed**
clauses are computed and written to `results/forward_verdict.json` (created exclusively: it is
never overwritten); later runs read it back and add an "after the readout" section. Always
written: `results/forward.md`, `results/forward_gate_days.csv`.

Dry run (`--dry-run-from YYYY-MM-DD`): treats a past date as the forward start, to test the
pipeline on existing candles. Output goes to `cache/forward_dryrun/` and stdout, is marked DRY
RUN, never touches `results/` and never writes a verdict file.

Extending the store (not done here; run it yourself, then rerun this script):

    cd /Users/mike/code/crypto-moonshot && uv run moonshot data sync \\
        --data-dir /Users/mike/code/crypto-moonshot/data --start 2026-10-01 --progress none \\
        $(uv run --group research python research/rebound_factors/forward.py --list-sync-symbols)

The symbol list is `paper.py`'s `listed_symbols`: `top200.txt` minus pegged and tokenised-stock
symbols, plus `outside_pool.txt`, plus BTCUSDT and ETHUSDT. Delisted coins cannot gain new
candles; a coin listed after the lists were made appears only once it is added to `top200.txt` or
`outside_pool.txt` and synced. A coin the store does not have is simply not an entry.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import build_factors as bf
import m5_confirm as m5
import numpy as np
import polars as pl
from build_exclusions import breaks, excluded_entries
from build_labels import LABEL_SCHEMA, Value, label_entry
from build_panel import PANEL_SCHEMA, panel_rows
from common import (
    CACHE,
    DATA_DIR,
    DAY,
    FIVE,
    I64,
    KLINES_5M,
    LABELS,
    RESULTS,
    SEALED_LABELS,
    WINDOW_US,
    first_day_pre2020,
    read_5m,
    universe,
    us,
    utc,
)
from m4_model import SEED
from m5_confirm import Rows, pc
from m7_gate import (
    Tidy,
    arm_a,
    arm_a_table,
    arm_b,
    arm_b_table,
    ci,
    gate_state,
    month_of,
    other_stops_table,
    pp,
    sma50_factor,
    tidy,
    with_gate,
)
from model import PickRule, Spec, assemble, book, load_spec, picks, score_period, train_mask

from crypto_moonshot.data.universe import EXCLUDED

FORWARD_START = datetime(2026, 11, 2, tzinfo=UTC)  # the data end of every study so far
FWD_CACHE = CACHE / "forward"
DRY_CACHE = CACHE / "forward_dryrun"
VERDICT_JSON = RESULTS / "forward_verdict.json"
OUT_MD = RESULTS / "forward.md"
OUT_DAYS = RESULTS / "forward_gate_days.csv"
BTC, ETH = bf.BTC, bf.ETH

MIN_COMPLETE_DAYS = 180  # forward days whose 30-day window is complete
MIN_TRADES = 100  # arm B trades (10% stop, base)
MIN_GATE_DAYS = 30  # days of each gate state for arm A's comparison
SCOPE = "forward"

SYNC_COMMAND = (
    "cd /Users/mike/code/crypto-moonshot && uv run moonshot data sync "
    "--data-dir /Users/mike/code/crypto-moonshot/data --start 2026-10-01 --progress none "
    "$(uv run --group research python research/rebound_factors/forward.py --list-sync-symbols)"
)


# --- the store and the forward period ---------------------------------------------------------


def complete_day_end(last_open_us: int) -> int:
    """The exclusive end: the midnight at or before the close of the last candle."""
    return (last_open_us + FIVE) // DAY * DAY


def store_end_us(root: Path = KLINES_5M) -> int:
    """This run's data end: the last complete UTC day of BTCUSDT in the candle store."""
    files = sorted((root / f"symbol={BTC}").glob("*.parquet"))
    if not files:
        raise RuntimeError(f"{BTC} has no candles in {root}")
    last = pl.read_parquet(files[-1], columns=["open_time"]).select(
        pl.col("open_time").dt.epoch("us").max()
    )
    return complete_day_end(int(last.item()))


def forward_days(start_us: int, end_us: int) -> int:
    """Entry days `D` in `[start, end)` (a day needs one complete candle day after `D`)."""
    return max(0, (end_us - start_us) // DAY)


def complete_days(start_us: int, end_us: int) -> int:
    """Forward days whose window `[D, D + 30 d)` is complete by the end."""
    return max(0, (end_us - WINDOW_US - start_us) // DAY + 1)


def earliest_readout_end_us(start_us: int) -> int:
    """The earliest data end with `MIN_COMPLETE_DAYS` complete days (arm B trades aside)."""
    return start_us + WINDOW_US + (MIN_COMPLETE_DAYS - 1) * DAY


def sync_symbols() -> list[str]:
    """The symbols `moonshot data sync` should fetch: `paper.py`'s `listed_symbols`."""
    top = (DATA_DIR / "top200.txt").read_text().split()
    outside = (DATA_DIR / "outside_pool.txt").read_text().split()
    pool = [s for s in top if s not in EXCLUDED]
    return sorted(set(pool) | set(outside) | {BTC, ETH})


# --- builders: the study's pure functions with this run's end -----------------------------------


@dataclass(frozen=True)
class Job:
    symbol: str
    start_us: int
    end_us: int


@dataclass
class CoinOut:
    panel: pl.DataFrame
    coins: pl.DataFrame | None
    labels: pl.DataFrame | None
    qv: pl.DataFrame
    breaks: list[tuple[str, int, float, float]]


@dataclass
class Built:
    """The forward tables: entries, factors, labels (null while the window is open), exclusions."""

    panel: pl.DataFrame
    factors: pl.DataFrame
    labels: pl.DataFrame
    exclusions: pl.DataFrame

    @property
    def empty(self) -> bool:
        return self.panel.height == 0


_BTC_BARS: dict[int, bf.Bars] = {}


def btc_bars(end_us: int) -> bf.Bars:
    if end_us not in _BTC_BARS:
        _BTC_BARS[end_us] = bf.daily_bars(read_5m(BTC, end_us, flow=True))
    return _BTC_BARS[end_us]


def coin_job(job: Job) -> CoinOut:
    """One coin: its forward entries (study panel rule), their factors, labels and breaks."""
    sym, end = job.symbol, job.end_us
    c5 = read_5m(sym, end, flow=True)
    empty_qv = pl.DataFrame(schema={"day": pl.Int64, "qv": pl.Float64})
    if c5.height == 0:
        return CoinOut(pl.DataFrame(schema=PANEL_SCHEMA), None, None, empty_qv, [])
    t = c5["t"].to_numpy().astype(np.int64)
    o, h, lo, c = (c5[k].to_numpy().astype(np.float64) for k in ("open", "high", "low", "close"))
    qv = c5["quote_volume"].to_numpy().astype(np.float64)
    pan = panel_rows(sym, t, c, qv, first_day_pre2020(sym), last_d_us=end - DAY)
    pan = pan.filter(pl.col("d_us") >= job.start_us)
    b = bf.daily_bars(c5)
    daily = pl.DataFrame({"day": b.d0 + np.arange(b.n, dtype=np.int64), "qv": b.qv})
    brk = breaks(sym, end, c5)
    if pan.height == 0:
        return CoinOut(pan, None, None, daily, brk)
    d_list = pan["d_us"].to_list()
    p_list = pan["P"].to_list()
    coins = bf.coin_rows(
        sym,
        c5,
        btc_bars(end),
        np.asarray(d_list, dtype=np.int64),
        np.asarray(p_list, dtype=np.float64),
    )
    rows: list[dict[str, Value]] = [
        label_entry(t, o, h, lo, c, d, p)
        if d + WINDOW_US <= end
        else dict.fromkeys(LABEL_SCHEMA)  # window not complete: no outcome yet
        for d, p in zip(d_list, p_list, strict=True)
    ]
    lab = pl.DataFrame(rows, schema=LABEL_SCHEMA, orient="row").with_columns(
        symbol=pl.lit(sym), d_us=pl.Series(d_list, dtype=pl.Int64)
    )
    return CoinOut(pan, coins, lab, daily, brk)


def write_atomic(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


def empty_built() -> Built:
    e = pl.DataFrame()
    return Built(pl.DataFrame(schema=PANEL_SCHEMA), e, e, e)


def build_forward(
    start_us: int,
    end_us: int,
    cache_dir: Path,
    symbols: list[str] | None = None,
    workers: int = 8,
) -> Built:
    """Panel, factors, labels and exclusions of the entries with `D` in `[start, end)`; written
    to `cache_dir` only. `symbols` defaults to the study's universe."""
    if forward_days(start_us, end_us) == 0:
        return empty_built()
    syms = universe() if symbols is None else symbols
    jobs = [Job(s, start_us, end_us) for s in syms]
    if workers <= 1:
        outs = [coin_job(j) for j in jobs]
    else:
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as pool:
            outs = list(pool.map(coin_job, jobs, chunksize=2))
    panel = pl.concat([o.panel for o in outs]).sort("d_us", "symbol")
    if panel.height == 0:
        return empty_built()
    coins = pl.concat([o.coins for o in outs if o.coins is not None])
    daily_qv = pl.concat([o.qv for o in outs if o.qv.height])
    btc = btc_bars(end_us)
    eth = bf.daily_bars(read_5m(ETH, end_us, flow=True))
    factors = bf.assemble(panel, coins, daily_qv, btc, eth).sort("d_us", "symbol")
    keys = panel.select("symbol", "d_us", "year", "split")
    labels = (
        pl.concat([o.labels for o in outs if o.labels is not None])
        .join(keys, on=["symbol", "d_us"], how="left")
        .select(["symbol", "d_us", "year", "split", *LABEL_SCHEMA])
        .sort("d_us", "symbol")
    )
    found = [b for o in outs for b in o.breaks]
    br_schema = {
        "symbol": pl.String,
        "break_t": pl.Int64,
        "ratio": pl.Float64,
        "halt_hours": pl.Float64,
    }
    br = pl.DataFrame(found, schema=br_schema, orient="row")
    excl = excluded_entries(panel.select("symbol", "d_us", "split"), br).select(
        "symbol", "d_us", "break_t"
    )
    write_atomic(panel, cache_dir / "panel.parquet")
    write_atomic(factors, cache_dir / "factors.parquet")
    write_atomic(labels, cache_dir / "labels.parquet")
    write_atomic(excl, cache_dir / "exclusions.parquet")
    return Built(panel, factors, labels, excl)


# --- frames and scoring -----------------------------------------------------------------------


def study_frame() -> pl.DataFrame:
    """Every study entry with ranks, as `model.assemble` builds it: discovery, gap, confirm and
    the unused last entry (D 2026-10-03), minus exclusions and entries without an outcome."""
    labels = pl.concat([pl.read_parquet(LABELS), pl.read_parquet(SEALED_LABELS)])
    return assemble(pl.read_parquet(m5.FACTORS_PQ), labels, pl.read_parquet(m5.EXCLUSIONS_PQ))


def forward_frame(b: Built, end_us: int) -> pl.DataFrame:
    """The forward entries with factors and ranks, minus exclusions. Entries whose window is
    complete and which have no outcome (`no_entry`) are dropped as in the study; entries whose
    window is still open are kept (`complete` False, outcomes null)."""
    df = assemble(b.factors, b.labels, b.exclusions, require_outcome=False)
    for c in (*m5.HITS, *m5.RETS):
        df = df.with_columns(pl.col(c).cast(pl.Float64))
    df = df.join(b.panel.select("symbol", "d_us", "liq"), on=["symbol", "d_us"], how="left")
    df = df.with_columns(
        month=pl.Series(month_of(df["d_us"].to_numpy().astype(np.int64))),
        complete=(pl.col("d_us") + WINDOW_US <= end_us),
    )
    return df.filter(~pl.col("complete") | pl.col("hit10").is_not_null())


def score_forward(
    spec: Spec, study: pl.DataFrame, fwd: pl.DataFrame, start_us: int
) -> pl.DataFrame:
    """`model.score_period` for the forward period: trained on the study entries whose window
    ended by `start_us`; the gate column is added."""
    assert spec.rule.gate is None
    assert spec.method.kind == "transparent"
    d = study["d_us"].to_numpy().astype(np.int64)
    train = study.filter(pl.Series(train_mask(d, start_us)))
    sc = score_period(spec.method, train, fwd, None, start_us)
    return with_gate(sc.frame)


# --- the gate day by day ----------------------------------------------------------------------


def gate_days(btc: bf.Bars, start_us: int, end_us: int) -> tuple[I64, Any, int]:
    """(forward days `D`, gate state per day (1 open, 0 closed, -1 null), the state at `end`).

    From BTC's daily bars: the factor of the entry at `D` is the one of the bar of day `D - 1`.
    The state "at end" is the next entry day's, the one a trader would act on next."""
    sma = sma50_factor(btc.close)
    d = np.arange(start_us, max(start_us, end_us), DAY, dtype=np.int64)
    d = d[d < end_us]
    st = gate_state(sma[d // DAY - 1 - btc.d0]) if len(d) else gate_state(np.zeros(0))
    now = int(gate_state(sma[[btc.n - 1]])[0])
    return d, st, now


# --- readout ----------------------------------------------------------------------------------


def readout_ready(n_complete_days: int, trades: int) -> bool:
    return n_complete_days >= MIN_COMPLETE_DAYS and trades >= MIN_TRADES


def write_verdict_once(path: Path, verdict: dict[str, Any]) -> None:
    """Create the verdict file; raises FileExistsError if it exists (never overwritten)."""
    with path.open("x") as f:
        json.dump(verdict, f, indent=1, sort_keys=True, allow_nan=False)
        f.write("\n")


def readout_state(
    path: Path | None,
    n_complete_days: int,
    trades: int,
    make: Callable[[], dict[str, Any]],
) -> tuple[str, dict[str, Any] | None]:
    """The readout state machine.

    - a verdict file exists: ("after", the recorded verdict), whatever the numbers are now;
    - else not ready (fewer than 180 complete days or fewer than 100 trades): ("interim", None);
    - else ready: the verdict is made and written (exclusively) and returned as ("verdict", v).
    `path` None (a dry run) never writes: a ready run returns ("dry_ready", v).
    """
    if path is not None and path.exists():
        return "after", json.loads(path.read_text())
    if not readout_ready(n_complete_days, trades):
        return "interim", None
    v = make()
    if path is None:
        return "dry_ready", v
    write_verdict_once(path, v)
    return "verdict", v


def jsonable(x: Any) -> Any:
    if isinstance(x, float):
        return None if math.isnan(x) or math.isinf(x) else x
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [jsonable(v) for v in x]
    return x


def build_verdict(g: Tidy, n_done: int, start_us: int, end_us: int) -> dict[str, Any]:
    """The record written once at the readout: the clauses, the numbers behind them and the
    tables as they stood (json-safe: NaN becomes null)."""
    v = jsonable(
        {
            "data_end": f"{utc(end_us):%Y-%m-%d}",
            "forward_start": f"{utc(start_us):%Y-%m-%d}",
            "complete_days": n_done,
            "arm_b_trades": trades_of(g),
            "clauses": clauses(g),
            "tables_md": table_lines(g),
        }
    )
    assert isinstance(v, dict)
    return v


def clauses(g: Tidy) -> dict[str, Any]:
    """The protocol's Confirmed clauses for arm B on the forward period, and arm A's reading."""
    s, v = SCOPE, "gate_open"
    stress = g("trades", v, s, "mean_ret", "10", "stress")
    bk = g("book", v, s, "book_return", "10", "base")
    rnd50 = g("random", v, s, "book_return_p50", "10", "base")
    hit = g("picks", v, s, "hit10")
    p95 = g("random", v, s, "pick_hit10_p95")
    c1, c2, c3 = bool(stress > 0), bool(bk > 0 and bk > rnd50), bool(hit > p95)
    d_open, d_closed = g("armA", "all", s, "days_open"), g("armA", "all", s, "days_closed")
    tested = bool(d_open >= MIN_GATE_DAYS and d_closed >= MIN_GATE_DAYS)
    return {
        "c1_mean_net_return_with_0.3pct_stress_above_0": {"value": stress, "ok": c1},
        "c2_book_positive_and_above_random_median": {"book": bk, "random_p50": rnd50, "ok": c2},
        "c3_pick_hit10_above_random_p95": {"pick_hit10": hit, "random_p95": p95, "ok": c3},
        "confirmed": c1 and c2 and c3,
        "arm_a": {
            "days_open": d_open,
            "days_closed": d_closed,
            "tested": tested,
            "hit10_open": g("armA", "all", s, "hit10:open"),
            "hit10_closed": g("armA", "all", s, "hit10:closed"),
            "diff": g("armA", "all", s, "hit10:diff"),
            "diff_lo": g("armA", "all", s, "hit10:diff_lo"),
            "diff_hi": g("armA", "all", s, "hit10:diff_hi"),
        },
    }


# --- evaluation -------------------------------------------------------------------------------


def evaluate(scored: pl.DataFrame, rule: PickRule) -> tuple[Tidy, pl.DataFrame, pl.DataFrame]:
    """Arms A and B on the forward entries with complete windows. Returns the tidy numbers
    (variants `all`, `ungated`, `gate_open`; scope `forward`), the tidy frame, and the picks of
    the ungated rule on the complete days (before the gate)."""
    done = scored.filter(pl.col("complete"))
    rows_a, rows_u, rows_g = Rows([]), Rows([]), Rows([])
    pk_u = picks(done, rule) if done.height else done
    if done.height:
        arm_a(rows_a, SCOPE, done)
    if pk_u.height:
        arm_b(rows_u, rows_g, SCOPE, done, pk_u, SEED)
    df = tidy([("all", rows_a), ("ungated", rows_u), ("gate_open", rows_g)])
    return Tidy(df), df, pk_u


def trades_of(g: Tidy) -> int:
    n = g("trades", "gate_open", SCOPE, "trades", "10", "base")
    return 0 if math.isnan(n) else int(n)


def table_lines(g: Tidy) -> list[str]:
    """The forward result tables (arm A, arm B, other stops) as markdown lines."""
    s = (SCOPE,)
    out = ["#### Arm A: every usable entry, gate open against closed", ""]
    labels = {
        "hit10": "hit10",
        "hit20": "hit20",
        "hit_ns": "hit_ns",
        "ret_10": "mean net return, 10% stop",
    }
    for c, name in labels.items():
        out += [f"**{name}**", "", *arm_a_table(g, c, s), ""]
    out += ["#### Arm B: the frozen rule on gate-open days", "", *arm_b_table(g, s), ""]
    out += ["**Other stops**", "", *other_stops_table(g, s), ""]
    return out


def pending_picks(scored: pl.DataFrame, rule: PickRule) -> pl.DataFrame:
    """Picks of the rule on forward days whose window is still open: no outcomes, so allowed."""
    pend = scored.filter(~pl.col("complete"))
    if pend.height == 0:
        return pend
    return picks(pend, rule).select("symbol", "d_us", "pct", "score", "pick_rank", "gate_ok", "liq")


# --- dry-run checks against the study ---------------------------------------------------------


def dryrun_checks(
    b: Built, scored: pl.DataFrame, rule: PickRule, start_us: int, end_us: int
) -> list[str]:
    """The pipeline against the study's own cache on the overlapping entries."""
    out: list[str] = []
    # factors
    study_f = pl.read_parquet(m5.FACTORS_PQ)
    both = b.factors.join(study_f, on=["symbol", "d_us"], suffix="_s", how="inner")
    worst = ("", 0.0)
    null_mismatch = 0
    for name in bf.NAMES:
        a, c = both[name], both[f"{name}_s"]
        null_mismatch += int((a.is_null() != c.is_null()).sum())
        m = a.is_not_null() & c.is_not_null()
        if m.any():
            diff = float((a.filter(m) - c.filter(m)).abs().max())  # type: ignore[arg-type]
            if diff > worst[1]:
                worst = (name, diff)
    out.append(
        f"- factors: {both.height} of {b.factors.height} forward entries are also in the study's "
        f"factor table; null patterns differ in {null_mismatch} cells; largest absolute difference "
        f"{worst[1]:.3g} ({worst[0] or 'none'})"
    )
    # labels
    study_l = pl.concat([pl.read_parquet(LABELS), pl.read_parquet(SEALED_LABELS)])
    lab = b.labels.filter(pl.col("hit10").is_not_null()).join(
        study_l, on=["symbol", "d_us"], suffix="_s", how="inner"
    )
    bad = 0
    for col in ("hit10", "hit20", "hit_ns", "ret_10", "ret_20", "ret_ns", "exit_t_10"):
        gap = (lab[col].cast(pl.Float64) - lab[f"{col}_s"].cast(pl.Float64)).abs()
        bad += int((gap > 1e-12).sum())
    n_lab = b.labels.filter(pl.col("hit10").is_not_null()).height
    out.append(
        f"- labels: {lab.height} of {n_lab} forward labels are in the study's label tables; "
        f"{bad} differing values among hit10/20/ns, ret_10/20/ns and exit_t_10"
    )
    # exclusions vs the study's
    study_e = pl.read_parquet(m5.EXCLUSIONS_PQ)
    e_in = study_e.filter((pl.col("d_us") >= start_us) & (pl.col("d_us") < end_us - 0))
    e_f = b.exclusions
    e_both = e_f.join(e_in, on=["symbol", "d_us"], how="inner").height
    out.append(
        f"- exclusions: {e_f.height} forward vs {e_in.height} study entries in the same days; "
        f"{e_both} in both (the study's breaks use candles to 2026-11-02, as here)"
    )
    # ungated rule per year against M5
    ref = pl.read_csv(
        m5.OUT_CSV, schema_overrides={"scope": pl.String, "stop": pl.String, "stress": pl.String}
    )
    look = {
        (r["table"], r["scope"], r["stop"] or "", r["stress"] or "", r["metric"]): r["value"]
        for r in ref.iter_rows(named=True)
    }
    done = scored.filter(pl.col("complete"))
    out.append("")
    out.append(
        "Per year against `results/m5_confirm.csv` (fresh book per year; M5 retrains the percentile reference each 1 January, this run keeps one reference from the start, so 2025 and 2026 differ slightly; years starting before the forward start are skipped):"
    )
    out.append("")
    out.append(
        "| year | picks (here / M5) | trades (here / M5) | hit10 of picks (here / M5) | mean net return per trade (here / M5) |"
    )
    out.append("| --- | --- | --- | --- | --- |")
    for yr in (2024, 2025, 2026):
        if start_us > us(datetime(yr, 1, 1, tzinfo=UTC)):
            continue  # a part year is not comparable with M5's whole-year numbers
        ty = done.filter(pl.col("year") == yr)
        if ty.height == 0:
            continue
        pk = picks(ty, rule)
        bk = book(pk, "10", 0.0)
        mine_hit = float(pk["hit10"].mean())  # type: ignore[arg-type]
        mine_ret = float(bk.trades["ret"].mean()) if bk.trades.height else math.nan  # type: ignore[arg-type]
        k = str(yr)
        out.append(
            f"| {yr} | {pk.height} / {look[('picks', k, '', '', 'n_picks')]:.0f} "
            f"| {bk.trades.height} / {look[('trades', k, '10', 'base', 'trades')]:.0f} "
            f"| {pc(mine_hit)}% / {pc(look[('picks', k, '', '', 'hit10')])}% "
            f"| {pp(mine_ret)}% / {pp(look[('trades', k, '10', 'base', 'mean_ret')])}% |"
        )
    return out


# --- report -----------------------------------------------------------------------------------


def report(
    *,
    dry_from: datetime | None,
    start_us: int,
    end_us: int,
    state: str,
    verdict: dict[str, Any] | None,
    g: Tidy,
    gdays: tuple[I64, Any, int],
    pend: pl.DataFrame,
    n_entries: int,
    n_train: int,
    extra: list[str],
) -> str:
    dry = dry_from is not None
    n_fwd, n_done = forward_days(start_us, end_us), complete_days(start_us, end_us)
    n_fwd, n_done = n_fwd, min(n_done, n_fwd)
    trades = trades_of(g)
    _d, st, now = gdays
    n_open, n_closed, n_null = int((st == 1).sum()), int((st == 0).sum()), int((st == -1).sum())
    word = {1: "open", 0: "closed", -1: "null (closed)"}[now]
    L: list[str] = []
    a = L.append
    if dry:
        a("# DRY RUN: forward test pipeline on past candles (not a result)")
        a("")
        a(
            f"**DRY RUN from {utc(start_us):%Y-%m-%d}.** A past date is treated as the forward start to test the"
        )
        a(
            "pipeline end to end on existing candles. Nothing here is a forward result, no verdict file is"
        )
        a("written, and this output never goes to `results/`.")
    else:
        a("# Forward test of the frozen rule with the BTC gate (M7)")
    a("")
    a(
        'Generated by `forward.py` (protocol: `docs/plans/rebound-factors.md`, "Gate protocol"). The rule is'
    )
    a(
        "`results/spec.json`, unchanged; the gate is open when BTC's last daily close is above the mean of its"
    )
    a("last 50 daily closes, closed otherwise and when the factor is null.")
    a("")
    a("## Data and status")
    a("")
    a(
        f"- candle store: `{DATA_DIR}`; data end found (BTCUSDT, last complete UTC day, exclusive): **{utc(end_us):%Y-%m-%d %H:%M} UTC**"
    )
    a(
        f"- forward start: {utc(start_us):%Y-%m-%d}; forward entry days so far (`D` from the start to the day before the data end): **{n_fwd}**"
    )
    a(
        f"- days whose 30-day window is complete: **{n_done}** of the {MIN_COMPLETE_DAYS} needed; arm B trades (10% stop, base): **{trades}** of the {MIN_TRADES} needed"
    )
    a(
        f"- readout state: **{state.upper().replace('_', ' ')}**; the earliest data end with {MIN_COMPLETE_DAYS} complete days is {utc(earliest_readout_end_us(start_us)):%Y-%m-%d} (later if arm B has fewer than {MIN_TRADES} trades by then)"
    )
    if n_entries:
        a(
            f"- forward entries scored: {n_entries} (percentile reference: the transparent score of the {n_train} study entries whose window ended by the forward start, `D + 30 d <= start`, minus exclusions; one reference for the whole period)"
        )
    else:
        a("- forward entries scored: none yet")
    a("")
    a("### Extending the candle store")
    a("")
    a("This script never syncs. To add candles, run (writes into the shared data folder):")
    a("")
    a("```")
    a(SYNC_COMMAND)
    a("```")
    a("")
    a(
        f"then rerun `uv run --group research python forward.py` from `research/rebound_factors/`. The symbols are `paper.py`'s `listed_symbols` ({len(sync_symbols()) if (DATA_DIR / 'top200.txt').exists() else 0} now): `top200.txt` minus pegged assets and tokenised stocks, plus `outside_pool.txt`, BTCUSDT and ETHUSDT. Delisted coins cannot gain new candles; a coin listed after those lists were made appears only after it is added to `top200.txt` or `outside_pool.txt` and synced. A coin missing from the store is simply not an entry."
    )
    a("")
    a("## The gate, day by day")
    a("")
    a(
        f"- forward days: open {n_open}, closed {n_closed}, null {n_null} (days with `m_btc_sma50` missing count as closed)"
    )
    a(
        f"- current state (the entry day {utc(end_us):%Y-%m-%d}, from BTC's close of {utc(end_us - DAY):%Y-%m-%d}): **{word}**"
    )
    if not dry:
        a("- every day: `results/forward_gate_days.csv`")
    a("")
    a("## Open picks (window not complete: no outcomes shown)")
    a("")
    if pend.height == 0:
        a("None: there are no forward entries with an open window.")
    else:
        a(
            f"{pend.height} picks of the ungated rule on days whose window is still open ({int((pend['gate_ok']).sum())} on gate-open days, which the gated rule would take, slots permitting)."
        )
        if dry and pend.height > 30:
            a("DRY RUN: the last 30 shown.")
            pend = pend.tail(30)
        a("")
        a("| symbol | D | score percentile | rank that day | gate | 30-day mean volume ($) |")
        a("| --- | --- | --- | --- | --- | --- |")
        for r in pend.iter_rows(named=True):
            a(
                f"| {r['symbol']} | {utc(r['d_us']):%Y-%m-%d} | {r['pct'] * 100:.1f} | {r['pick_rank'] + 1} | {'open' if r['gate_ok'] else 'closed'} | {r['liq']:,.0f} |"
            )
    a("")
    if dry:
        a("## DRY RUN checks against the study")
        a("")
        L.extend(extra)
        a("")
    if verdict is not None and state in ("verdict", "after"):
        a("## Readout (recorded once; later runs do not change it)")
        a("")
        L.extend(verdict_lines(verdict))
        a("")
    if n_done == 0 or math.isnan(g("picks", "ungated", SCOPE, "n_picks")):
        a("## Tables")
        a("")
        a(
            "No forward entry has a complete window yet, so there are no outcomes and no tables."
            if n_done == 0
            else "No picks on complete days yet."
        )
        a("")
    else:
        head = {
            "interim": "## Interim tables (not a verdict)",
            "verdict": "## Tables at the readout",
            "after": "## After the readout (updated numbers; they do not change the verdict above)",
            "dry_ready": "## Tables (DRY RUN: the readout conditions would be met; no verdict file written)",
        }[state]
        a(head)
        a("")
        if state == "interim":
            a(
                f"Interim: fewer than {MIN_COMPLETE_DAYS} complete days or fewer than {MIN_TRADES} arm B trades. These numbers can move a lot and are not the verdict."
            )
            a("")
        if dry:
            a("DRY RUN: ungated rows are the rule without the gate, for the reproduction check.")
            a("")
        L.extend(table_lines(g))
        cl = clauses(g)
        a("#### Confirmed clauses (arm B, forward period)")
        a("")
        L.extend(clause_lines(cl))
        a("")
    return "\n".join(L) + "\n"


def clause_lines(cl: dict[str, Any]) -> list[str]:
    c1 = cl["c1_mean_net_return_with_0.3pct_stress_above_0"]
    c2 = cl["c2_book_positive_and_above_random_median"]
    c3 = cl["c3_pick_hit10_above_random_p95"]
    a = cl["arm_a"]

    def w(b: bool) -> str:
        return "pass" if b else "FAIL"

    def f(x: float | None) -> float:
        return math.nan if x is None else float(x)

    out = [
        f"- mean net return per trade with the 0.3% extra cost above 0: {pp(f(c1['value']))}% -> {w(c1['ok'])}",
        f"- 10%-stop book return positive and above the same-day random books' median: {pp(f(c2['book']), 0)}% against {pp(f(c2['random_p50']), 0)}% -> {w(c2['ok'])}",
        f"- picks' hit10 above the same-day random 95th percentile: {pc(f(c3['pick_hit10']))}% against {pc(f(c3['random_p95']))}% -> {w(c3['ok'])}",
        f"- **Confirmed: {'yes' if cl['confirmed'] else 'no'}** (all three clauses must pass)",
        f"- also reported, arm A hit10 with the gate open against closed: {pc(f(a['hit10_open']))}% against {pc(f(a['hit10_closed']))}% (difference {pp(f(a['diff']))} pp, month-bootstrap 95% {ci(f(a['diff_lo']), f(a['diff_hi']))}); gate-open days {f(a['days_open']):.0f}, closed {f(a['days_closed']):.0f}; {'enough days of each (at least 30)' if a['tested'] else 'fewer than 30 days of one state: not tested'}",
    ]
    return out


def verdict_lines(v: dict[str, Any]) -> list[str]:
    out = [
        f"Read out on the run with data end {v['data_end']} ({v['complete_days']} days with complete windows, {v['arm_b_trades']} arm B trades; forward start {v['forward_start']}).",
        "",
        *clause_lines(v["clauses"]),
        "",
        *v["tables_md"],
    ]
    return out


# --- main -------------------------------------------------------------------------------------


def run(dry_from: datetime | None, workers: int) -> str:
    t0 = time.perf_counter()
    dry = dry_from is not None
    start_us = us(dry_from) if dry_from is not None else us(FORWARD_START)
    end_us = store_end_us()
    cache_dir = DRY_CACHE if dry else FWD_CACHE
    spec = load_spec(m5.SPEC_JSON)
    assert spec.rule.name == "k10_q0.95"
    assert spec.extra["name"] == "rule_liq_dd_nogate"
    assert spec.stop == "10"
    print(
        f"data end {utc(end_us):%Y-%m-%d}; forward start {utc(start_us):%Y-%m-%d}; {forward_days(start_us, end_us)} forward days",
        flush=True,
    )
    btc = btc_bars(end_us)
    gdays = gate_days(btc, start_us, end_us)
    b = build_forward(start_us, end_us, cache_dir, workers=workers)
    print(f"built in {time.perf_counter() - t0:.0f}s: {b.panel.height} entries", flush=True)
    empty = pl.DataFrame()
    g = Tidy(tidy([("all", Rows([]))]))
    pend, n_train, n_entries, extra = empty, 0, 0, []
    state, verdict = "interim", None
    if not b.empty:
        study = study_frame()
        fwd = forward_frame(b, end_us)
        scored = score_forward(spec, study, fwd, start_us)
        n_entries = scored.height
        n_train = int(train_mask(study["d_us"].to_numpy().astype(np.int64), start_us).sum())
        g, df, _pk = evaluate(scored, spec.rule)
        pend = pending_picks(scored, spec.rule)
        n_done = min(complete_days(start_us, end_us), forward_days(start_us, end_us))

        def make() -> dict[str, Any]:
            return build_verdict(g, n_done, start_us, end_us)

        state, verdict = readout_state(None if dry else VERDICT_JSON, n_done, trades_of(g), make)
        if dry:
            extra = dryrun_checks(b, scored, spec.rule, start_us, end_us)
            df.write_csv(cache_dir / "forward_dryrun.csv", float_precision=6)
    elif not dry and VERDICT_JSON.exists():
        state, verdict = "after", json.loads(VERDICT_JSON.read_text())
    md = report(
        dry_from=dry_from,
        start_us=start_us,
        end_us=end_us,
        state=state,
        verdict=verdict,
        g=g,
        gdays=gdays,
        pend=pend,
        n_entries=n_entries,
        n_train=n_train,
        extra=extra,
    )
    if dry:
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "forward_dryrun.md").write_text(md)
    else:
        RESULTS.mkdir(parents=True, exist_ok=True)
        OUT_MD.write_text(md)
        d, st, _now = gdays
        names = {1: "open", 0: "closed", -1: "null"}
        lines = [
            "date,gate",
            *(f"{utc(int(x)):%Y-%m-%d},{names[int(s)]}" for x, s in zip(d, st, strict=True)),
        ]
        OUT_DAYS.write_text("\n".join(lines) + "\n")
    print(f"done in {time.perf_counter() - t0:.0f}s; state {state}", flush=True)
    return md


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--dry-run-from",
        metavar="YYYY-MM-DD",
        help="treat a past date as the forward start (output to cache/forward_dryrun/ and stdout only)",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--list-sync-symbols",
        action="store_true",
        help="print the symbols `moonshot data sync` should fetch, then exit (syncs nothing)",
    )
    args = parser.parse_args()
    if args.list_sync_symbols:
        print(" ".join(sync_symbols()))
        return
    dry_from = None
    if args.dry_run_from:
        dry_from = datetime.strptime(args.dry_run_from, "%Y-%m-%d").replace(tzinfo=UTC)
        assert us(dry_from) < store_end_us(), "a dry run needs a start inside the stored candles"
    md = run(dry_from, args.workers)
    if dry_from is not None:
        print(md)


if __name__ == "__main__":
    main()
