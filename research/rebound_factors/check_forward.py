"""M7: checks of the market gate (`m7_gate.py`) and the forward test (`forward.py`).

    uv run --group research python check_forward.py [--logic-only]

1. Logic, on synthetic data: the gate on BTC bars (above, below, null, exactly equal is closed);
   gated picks are a subset of the ungated picks and none falls on a closed day; the readout
   state machine (interim, then one verdict, then never overwritten) in a temp folder; the
   Confirmed clauses on synthetic numbers; the data end and the day counts.
2. Real data: the forward builders write only into the folder they are given (a temp cache root;
   the study's cache files keep their size and mtime), and for a few coins the forward panel and
   labels equal the study's on the overlapping entries; a dry run writes nothing into `results/`
   and no verdict file.

Prints one pass/fail line per check; exits non-zero on a failure.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import forward as fw
import numpy as np
import polars as pl
from common import CACHE, DAY, LABELS, PANEL, RESULTS, SEALED_LABELS, WINDOW_US, us
from m5_confirm import Rows
from m7_gate import Tidy, gate_open_expr, gate_state, gated_picks, sma50_factor, tidy
from model import PickRule, picks

FAILURES: list[str] = []


def check(name: str, ok: bool) -> None:
    print(f"{'pass' if ok else 'FAIL'}: {name}")
    if not ok:
        FAILURES.append(name)


# --- logic ------------------------------------------------------------------------------------


def logic_gate() -> None:
    rising = np.linspace(100.0, 200.0, 120)
    s = gate_state(sma50_factor(rising))
    check("gate: null while fewer than 50 closes", bool((s[:49] == -1).all()))
    check("gate: rising BTC is open from the 50th close", bool((s[49:] == 1).all()))
    falling = rising[::-1].copy()
    s = gate_state(sma50_factor(falling))
    check("gate: falling BTC is closed", bool((s[49:] == 0).all()))
    flat = np.full(80, 123.456)
    f = sma50_factor(flat)
    check("gate: a flat series gives a factor of exactly 0", bool((f[49:] == 0.0).all()))
    check("gate: exactly equal to the mean is closed", bool((gate_state(f)[49:] == 0).all()))
    # one close above a flat series is open, one close below is closed
    up, dn = flat.copy(), flat.copy()
    up[-1] *= 1.001
    dn[-1] *= 0.999
    check("gate: just above the mean is open", int(gate_state(sma50_factor(up))[-1]) == 1)
    check("gate: just below the mean is closed", int(gate_state(sma50_factor(dn))[-1]) == 0)
    # the factor matches the study's definition: close over the mean of the last 50, minus 1
    rng = np.random.default_rng(1)
    c = 100 + np.cumsum(rng.normal(size=90))
    manual = c[89] / c[40:90].mean() - 1
    check(
        "gate: factor = close / mean of last 50 closes - 1",
        abs(sma50_factor(c)[89] - manual) < 1e-12,
    )
    df = pl.DataFrame({"m_btc_sma50": [0.1, 0.0, -0.1, None]}, schema={"m_btc_sma50": pl.Float64})
    ok = df.select(gate_open_expr())["gate_ok"].to_list()
    check(
        "gate: open only when the factor is above 0 (null and 0 closed)",
        ok == [True, False, False, False],
    )


def logic_gated_picks() -> None:
    rng = np.random.default_rng(7)
    n_days, n_coins = 40, 25
    d = np.repeat(np.arange(n_days, dtype=np.int64) * DAY, n_coins)
    sym = np.tile(np.array([f"C{i}USDT" for i in range(n_coins)]), n_days)
    open_day = rng.random(n_days) < 0.5
    frame = pl.DataFrame(
        {
            "d_us": d,
            "symbol": sym,
            "score": rng.random(len(d)),
            "pct": rng.random(len(d)),
            "gate_ok": np.repeat(open_day, n_coins),
        }
    )
    rule = PickRule(k=5, q=0.5, gate=None)
    pk_u = picks(frame, rule)
    pk_g = gated_picks(pk_u)
    keys_u = set(zip(pk_u["symbol"].to_list(), pk_u["d_us"].to_list(), strict=True))
    keys_g = set(zip(pk_g["symbol"].to_list(), pk_g["d_us"].to_list(), strict=True))
    check("gated picks are a subset of the ungated picks", keys_g <= keys_u and len(keys_g) > 0)
    closed_days = {int(x) for x in d[~np.repeat(open_day, n_coins)]}
    check("no gated pick on a closed day", all(k[1] not in closed_days for k in keys_g))
    check(
        "every ungated pick on an open day is kept",
        len(keys_g) == sum(1 for k in keys_u if k[1] not in closed_days),
    )
    check("some ungated picks fall on closed days (the test has teeth)", len(keys_u) > len(keys_g))


def logic_readout() -> None:
    check("readout not ready: 179 days", not fw.readout_ready(179, 500))
    check("readout not ready: 99 trades", not fw.readout_ready(300, 99))
    check("readout ready: 180 days and 100 trades", fw.readout_ready(180, 100))
    made: list[int] = []

    def make(tag: int) -> dict[str, object]:
        made.append(tag)
        return {"tag": tag, "confirmed": False}

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "forward_verdict.json"
        st, v = fw.readout_state(path, 100, 50, lambda: make(0))
        check(
            "readout: interim below the thresholds, nothing written",
            st == "interim" and v is None and not path.exists() and not made,
        )
        st, v = fw.readout_state(path, 300, 99, lambda: make(0))
        check("readout: interim with 99 trades", st == "interim" and not path.exists())
        st, v = fw.readout_state(None, 200, 150, lambda: make(9))
        check(
            "readout: a dry run never writes",
            st == "dry_ready" and v == {"tag": 9, "confirmed": False} and not path.exists(),
        )
        made.clear()
        st, v = fw.readout_state(path, 180, 100, lambda: make(1))
        first = path.read_bytes() if path.exists() else b""
        check(
            "readout: the first run meeting both writes the verdict",
            st == "verdict" and v is not None and v["tag"] == 1 and bool(first),
        )
        st, v = fw.readout_state(path, 400, 500, lambda: make(2))
        check(
            "readout: a later run returns the recorded verdict and does not remake it",
            st == "after" and v is not None and v["tag"] == 1 and made == [1],
        )
        check("readout: the verdict file is unchanged", path.read_bytes() == first)
        st, v = fw.readout_state(path, 10, 1, lambda: make(3))
        check(
            "readout: after the readout even a thin run keeps the verdict",
            st == "after" and v is not None and v["tag"] == 1,
        )
        try:
            fw.write_verdict_once(path, {"tag": 4})
            refused = False
        except FileExistsError:
            refused = True
        check(
            "readout: writing over the verdict is refused", refused and path.read_bytes() == first
        )


def synthetic_numbers(stress: float, book: float, rnd50: float, hit: float, p95: float) -> Tidy:
    ra, ru, rg = Rows([]), Rows([]), Rows([])
    s = fw.SCOPE
    rg.add("trades", s, "mean_ret", stress, stop="10", stress="stress")
    rg.add("book", s, "book_return", book, stop="10", stress="base")
    rg.add("random", s, "book_return_p50", rnd50, stop="10", stress="base")
    rg.add("picks", s, "hit10", hit)
    rg.add("random", s, "pick_hit10_p95", p95)
    ra.add("armA", s, "days_open", 40.0)
    ra.add("armA", s, "days_closed", 20.0)
    return Tidy(tidy([("all", ra), ("ungated", ru), ("gate_open", rg)]))


def logic_clauses() -> None:
    good = fw.clauses(synthetic_numbers(0.01, 0.5, 0.1, 0.3, 0.25))
    check("clauses: all three pass -> confirmed", good["confirmed"] is True)
    check("clauses: arm A with 20 closed days is not tested", good["arm_a"]["tested"] is False)
    for name, args in (
        ("stressed return <= 0", (0.0, 0.5, 0.1, 0.3, 0.25)),
        ("book below the random median", (0.01, 0.05, 0.1, 0.3, 0.25)),
        ("book negative", (0.01, -0.1, -0.5, 0.3, 0.25)),
        ("hit10 not above the random 95th percentile", (0.01, 0.5, 0.1, 0.25, 0.25)),
    ):
        got = fw.clauses(synthetic_numbers(*args))
        check(f"clauses: {name} -> not confirmed", got["confirmed"] is False)
    g = synthetic_numbers(0.01, 0.5, 0.1, 0.3, 0.25)
    v = fw.build_verdict(
        g, 200, us(datetime(2026, 11, 2, tzinfo=UTC)), us(datetime(2027, 6, 1, tzinfo=UTC))
    )
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "v.json"
        fw.write_verdict_once(p, v)  # allow_nan=False: raises if a NaN slipped through
        back = json.loads(p.read_text())
    check(
        "verdict record round-trips through json and renders",
        back == v and len(fw.verdict_lines(back)) > 5 and v["clauses"]["confirmed"] is True,
    )
    check(
        "clauses: json-safe with NaN",
        fw.jsonable({"a": math.nan, "b": [1.0, math.inf]}) == {"a": None, "b": [1.0, None]},
    )


def logic_end_and_days() -> None:
    start = us(datetime(2026, 11, 2, tzinfo=UTC))
    check(
        "end: candles to 23:55 close at midnight", fw.complete_day_end(start - 300_000_000) == start
    )
    check(
        "end: a partial day is not complete",
        fw.complete_day_end(start + 14 * 3600 * 1_000_000) == start,
    )
    check("days: no forward day at the data end", fw.forward_days(start, start) == 0)
    check("days: one day after the start", fw.forward_days(start, start + DAY) == 1)
    check(
        "days: none complete before 30 days", fw.complete_days(start, start + WINDOW_US - DAY) == 0
    )
    check(
        "days: the first is complete at start + 30 days",
        fw.complete_days(start, start + WINDOW_US) == 1,
    )
    check(
        "days: 180 complete at start + 209 days",
        fw.complete_days(start, fw.earliest_readout_end_us(start)) == 180,
    )
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "symbol=BTCUSDT"
        d.mkdir()
        t0 = datetime(2026, 12, 1, tzinfo=UTC)
        times = [t0 + timedelta(minutes=5 * i) for i in range(288 + 174)]  # to 14:25 open on 12-02
        pl.DataFrame({"open_time": pl.Series(times, dtype=pl.Datetime("us", "UTC"))}).write_parquet(
            d / "2026-12.parquet"
        )
        end = fw.store_end_us(Path(tmp))
        check(
            "end: store_end_us is the last complete UTC day",
            end == us(datetime(2026, 12, 2, tzinfo=UTC)),
        )


# --- real data --------------------------------------------------------------------------------


def snapshot(paths: list[Path]) -> dict[str, tuple[int, int]]:
    return {str(p): (p.stat().st_mtime_ns, p.stat().st_size) for p in paths}


def real_builders() -> None:
    study_files = sorted(
        p
        for p in CACHE.rglob("*")
        if p.is_file() and not p.relative_to(CACHE).parts[0].startswith("forward")
    )
    check("study cache has files to guard", len(study_files) >= 6)
    before = snapshot(study_files)
    top_before = {p.name for p in CACHE.iterdir() if p.is_file()}
    check(
        "forward cache folders are not the study's cache files",
        fw.FWD_CACHE != CACHE and fw.DRY_CACHE != CACHE and fw.FWD_CACHE.parent == CACHE,
    )
    end = fw.store_end_us()
    start = us(datetime(2026, 9, 1, tzinfo=UTC))
    syms = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "SUIUSDT"]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        b = fw.build_forward(start, end, root, symbols=syms, workers=1)
        names = sorted(p.name for p in root.iterdir())
        check(
            "builders wrote their four tables into the given folder",
            names == ["exclusions.parquet", "factors.parquet", "labels.parquet", "panel.parquet"],
        )
        check(
            "forward builders left the study's cache files untouched",
            snapshot(study_files) == before,
        )
        check(
            "no new file next to the study's cache files",
            {p.name for p in CACHE.iterdir() if p.is_file()} == top_before,
        )
    check("forward panel has entries for the sample coins", b.panel.height > 100)
    # the panel equals the study's on the overlap (study D <= 2026-10-03)
    sp = pl.read_parquet(PANEL).filter(pl.col("symbol").is_in(syms) & (pl.col("d_us") >= start))
    fp = b.panel.filter(pl.col("d_us") <= int(sp["d_us"].max()))  # type: ignore[arg-type]
    cols = ["symbol", "d_us", "P", "liq", "days_listed"]
    check(
        "forward panel equals the study panel on the overlap",
        fp.select(cols).sort(cols[:2]).equals(sp.select(cols).sort(cols[:2])),
    )
    last_d = b.panel["d_us"].max()
    check("forward panel goes to the day before the data end", last_d == end - DAY)
    # labels: equal to the study's where complete; null where the window is open
    sl = pl.concat([pl.read_parquet(LABELS), pl.read_parquet(SEALED_LABELS)]).filter(
        pl.col("symbol").is_in(syms)
    )
    done = b.labels.filter(pl.col("d_us") + WINDOW_US <= end)
    j = done.join(sl, on=["symbol", "d_us"], suffix="_s", how="inner")
    same = all(
        bool(
            (
                (j[c].cast(pl.Float64) - j[f"{c}_s"].cast(pl.Float64)).abs().fill_null(0) < 1e-12
            ).all()
        )
        for c in ("hit10", "ret_10", "ret_20", "ret_ns", "exit_t_10")
    )
    check(
        "forward labels equal the study's labels where the window is complete",
        j.height > 50 and same,
    )
    open_rows = b.labels.filter(pl.col("d_us") + WINDOW_US > end)
    check(
        "labels are null while the window is open",
        open_rows.height > 0 and open_rows["hit10"].null_count() == open_rows.height,
    )
    check(
        "the sample's forward factors include the gate factor", "m_btc_sma50" in b.factors.columns
    )


def real_dry_run() -> None:
    results_before = snapshot(sorted(p for p in RESULTS.rglob("*") if p.is_file()))
    verdict_existed = fw.VERDICT_JSON.exists()
    md = fw.run(datetime(2026, 3, 1, tzinfo=UTC), 8)
    results_after = snapshot(sorted(p for p in RESULTS.rglob("*") if p.is_file()))
    check("dry run: nothing in results/ changed", results_before == results_after)
    check("dry run: no verdict file written", fw.VERDICT_JSON.exists() == verdict_existed)
    out = fw.DRY_CACHE / "forward_dryrun.md"
    check(
        "dry run: output in cache/forward_dryrun/ marked DRY RUN",
        out.exists() and md.startswith("# DRY RUN") and "DRY RUN" in md,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logic-only", action="store_true")
    a = ap.parse_args()
    logic_gate()
    logic_gated_picks()
    logic_readout()
    logic_clauses()
    logic_end_and_days()
    if not a.logic_only:
        real_builders()
        real_dry_run()
    if FAILURES:
        print(f"{len(FAILURES)} failed")
        sys.exit(1)
    print("all passed")


if __name__ == "__main__":
    main()
