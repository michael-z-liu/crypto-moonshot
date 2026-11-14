"""M4: checks of `model.py` (pick rule, book, ranks, training-set rule) and of the M4 outputs.

    uv run --group research python check_model.py [--logic-only]

1. Logic, on small synthetic cases (no data needed): within-day ranks, percentile of the
   training scores, the training-set rule, the pick rule (threshold, per-day cap, score order,
   gate), and the book (slot limit, one position per coin, a slot freeing at `exit_t`, score
   order within a day, equity compounding with realised-equity slots, the extra cost per side,
   the cash cap), and a spec round trip through json.
2. Real data: for each walk-forward year the training entries' windows end before the year
   starts (also by their recorded `exit_t`), and no sealed row is anywhere: the discovery frame,
   the cached walk-forward scores and the committed results stop before 2024-01-01. The sealed
   folder is never opened.

Prints one pass/fail line per check; exits non-zero on a failure.
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np
import polars as pl
from common import DAY, RESULTS, SEALED_START_US, WINDOW_US
from model import (
    START_EQUITY,
    Gate,
    Method,
    PickRule,
    Settings,
    Spec,
    TransparentRule,
    add_ranks,
    book,
    percentile_of,
    picks,
    spec_from_json,
    spec_to_json,
    stressed,
    train_mask,
    transparent_score,
    year_start_us,
)

FAILURES: list[str] = []
TOL = 1e-9
D0 = year_start_us(2021)


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'' if ok else '  ' + detail}", flush=True)
    if not ok:
        FAILURES.append(name)


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=TOL, abs_tol=TOL)


def trades(rows: list[tuple[str, int, float, int, float]]) -> pl.DataFrame:
    """(symbol, day number, score, exit day number, ret) -> a picks-like frame (stop "10")."""
    return pl.DataFrame(
        {
            "symbol": [r[0] for r in rows],
            "d_us": [D0 + r[1] * DAY for r in rows],
            "score": [r[2] for r in rows],
            "exit_t_10": [D0 + r[3] * DAY for r in rows],
            "ret_10": [r[4] for r in rows],
        }
    ).sort(["d_us", "score"], descending=[False, True])


# --- logic -----------------------------------------------------------------------------------


def logic_ranks() -> None:
    fac = pl.DataFrame(
        {
            "symbol": ["A", "B", "C", "D", "A", "B"],
            "d_us": [0, 0, 0, 0, DAY, DAY],
            "c_x": [1.0, 3.0, 2.0, None, float("nan"), 5.0],
        }
    )
    r = add_ranks(fac, ["c_x"])["r_c_x"].to_list()
    # day 0: ranks 1, 3, 2 of 3 non-null -> (r - .5) / 3; day 1: NaN is null, B alone -> 0.5
    want = [0.5 / 3, 2.5 / 3, 1.5 / 3, None, None, 0.5]
    ok = all(
        (a is None and b is None) or (a is not None and b is not None and close(a, b))
        for a, b in zip(r, want, strict=True)
    )
    check("ranks: within-day percentile, nulls and NaN stay null", ok, str(r))


def logic_percentile_and_train() -> None:
    ref = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    p = percentile_of(ref, np.array([0.05, 0.2, 0.35, 0.9]))
    check("percentile: share of training scores at or below", list(p) == [0.0, 0.5, 0.75, 1.0])
    start = year_start_us(2022)
    d = np.array([start - WINDOW_US - DAY, start - WINDOW_US, start - WINDOW_US + DAY, start])
    m = train_mask(d.astype(np.int64), start)
    check("training rule: D + 30 d <= start", list(m) == [True, True, False, False], str(m))


def logic_picks() -> None:
    g = Gate("m_g", 0.5)
    sc = pl.DataFrame(
        {
            "symbol": ["A", "B", "C", "D", "E", "F", "G"],
            "d_us": [0, 0, 0, 0, DAY, DAY, 2 * DAY],
            "score": [0.9, 0.95, 0.5, 0.8, 0.99, 0.97, 0.99],
            "pct": [0.96, 0.97, 0.5, 0.95, 0.99, 0.98, 0.99],
            "gate_ok": [True, True, True, True, False, False, True],
        }
    )
    p = picks(sc, PickRule(k=2, q=0.95, gate=None))
    check(
        "picks: threshold, cap per day, score order",
        p["symbol"].to_list() == ["B", "A", "E", "F", "G"]
        and p["pick_rank"].to_list() == [0, 1, 0, 1, 0],
        str(p["symbol"].to_list()),
    )
    p = picks(sc, PickRule(k=5, q=0.95, gate=g))
    check("picks: gate drops days failing it", p["symbol"].to_list() == ["B", "A", "D", "G"])
    p = picks(sc, PickRule(k=5, q=0.99, gate=None))
    check("picks: q is inclusive", p["symbol"].to_list() == ["E", "G"])


def logic_transparent() -> None:
    rule = TransparentRule(Gate("m_g", 0.5), (("c_a", -1), ("c_b", 1)))
    fr = pl.DataFrame({"m_g": [1.0, 1.0, 0.0], "r_c_a": [0.2, 0.8, 0.2], "r_c_b": [0.6, None, 0.6]})
    s = transparent_score(rule, fr, threshold=0.5)
    want = [(0.8 + 0.6) / 2, (0.2 + 0.5) / 2, (0.8 + 0.6) / 2 - 1.0]
    check("transparent: oriented rank mean, null as 0.5, gate offset", np.allclose(s, want), str(s))


def logic_book() -> None:
    # slot limit: 12 coins on day 0, all held 10 days -> 10 taken in score order
    rows = [(f"S{i:02d}", 0, 1.0 - i / 100, 10, 0.0) for i in range(12)]
    b = book(trades(rows), "10")
    check(
        "book: slot limit (10), taken in score order",
        b.trades.height == 10 and b.skipped == 2 and "S10" not in b.trades["symbol"].to_list(),
    )
    # freeing at exit_t: a slot frees when exit_t <= D of the new pick
    rows = [(f"S{i:02d}", 0, 1.0 - i / 100, 5 if i == 0 else 10, 0.0) for i in range(10)]
    rows += [("N1", 4, 0.9, 20, 0.0), ("N2", 5, 0.9, 20, 0.0), ("N3", 5, 0.8, 20, 0.0)]
    b = book(trades(rows), "10")
    got = b.trades["symbol"].to_list()
    check(
        "book: slot frees at exit_t, not before",
        "N1" not in got and "N2" in got and "N3" not in got,
        str(got),
    )
    # one position per coin: same coin while held is skipped, again after exit is taken
    rows = [("A", 0, 0.9, 3, 0.0), ("A", 1, 0.9, 5, 0.0), ("A", 3, 0.9, 6, 0.0)]
    b = book(trades(rows), "10")
    check(
        "book: one position per coin, re-entry after exit",
        b.trades["d_us"].to_list() == [D0, D0 + 3 * DAY],
    )
    # compounding: equity / 10 per slot; equity changes at closes only
    rows = [("A", 0, 0.9, 2, 0.10), ("B", 1, 0.9, 3, -0.10), ("C", 3, 0.9, 4, 0.20)]
    b = book(trades(rows), "10")
    eq = START_EQUITY
    s_a = eq / 10
    s_b = eq / 10  # A still open on day 1
    eq += s_a * 0.10  # A closes on day 2
    eq += s_b * -0.10  # B closes on day 3, before C enters on day 3
    s_c = eq / 10
    eq += s_c * 0.20
    sizes = b.trades["size"].to_list()
    check(
        "book: sizes are realised equity / 10, equity compounds",
        close(sizes[0], s_a)
        and close(sizes[1], s_b)
        and close(sizes[2], s_c)
        and close(b.final, eq),
        f"{sizes} {b.final} {eq}",
    )
    check(
        "book: equity curve and drawdown",
        b.equity["equity"].to_list()[0] == START_EQUITY and b.max_drawdown < 0,
    )
    # extra cost per side
    c = 0.003
    b2 = book(trades(rows), "10", extra_cost=c)
    r = b2.trades["ret"].to_list()
    check(
        "book: extra cost per side",
        close(r[0], 1.10 * (1 - c) / (1 + c) - 1) and close(r[1], 0.90 * (1 - c) / (1 + c) - 1),
        str(r),
    )
    check(
        "book: stressed() matches",
        close(float(stressed(np.array([0.0]), c)[0]), (1 - c) / (1 + c) - 1),
    )
    check("book: stress lowers the final equity", b2.final < b.final)
    # cash cap: after losses the open sizes exceed equity / 10 ... cap at untied cash
    rows = [
        (f"S{i}", 0, 1.0 - i / 100, 100 if i else 1, -1.0 if i == 0 else 0.0) for i in range(10)
    ]
    rows += [("Z", 2, 0.9, 50, 0.0)]
    b = book(trades(rows), "10")
    z = b.trades.filter(pl.col("symbol") == "Z")["size"].to_list()
    # S0 lost its 1,000; equity 9,000, tied 9,000 -> Z gets min(900, 0) = 0
    check("book: size capped at the untied cash", len(z) == 1 and close(z[0], 0.0), str(z))
    # null trade results are not booked
    rows2 = trades([("A", 0, 0.9, 2, 0.1)]).vstack(
        pl.DataFrame(
            {"symbol": ["B"], "d_us": [D0], "score": [0.5], "exit_t_10": [None], "ret_10": [None]},
            schema_overrides={"exit_t_10": pl.Int64, "ret_10": pl.Float64},
        )
    )
    check("book: entries without a trade result are skipped", book(rows2, "10").trades.height == 1)


def logic_spec() -> None:
    spec = Spec(
        method=Method(
            kind="gbm",
            features=["m_a", "c_b", "r_c_b"],
            settings=Settings("x", 0.1, 10, 3, 5, 0.0, 0.5),
            seeds=[1, 2],
        ),
        rule=PickRule(k=3, q=0.95, gate=Gate("m_a", 0.6)),
        stop="10",
        extra={"note": "x"},
    )
    back = spec_from_json(json.loads(json.dumps(spec_to_json(spec))))
    check("spec: json round trip", back == spec)


# --- real data -------------------------------------------------------------------------------


def real() -> None:
    import m4_model as m4

    df = m4.load()
    d = df["d_us"].to_numpy().astype(np.int64)
    check("data: discovery frame ends before 2024", int(d.max()) < SEALED_START_US)
    check("data: discovery only", set(df["split"].unique().to_list()) == {"discovery"})
    for y in m4.TEST_YEARS:
        start = year_start_us(y)
        tr = df.filter(pl.Series(train_mask(d, start)))
        ex = tr.select(pl.max_horizontal("exit_t_10", "exit_t_20", "exit_t_ns").max()).item()
        check(
            f"training rule {y}: windows and exits end by {y}-01-01",
            int(tr["d_us"].max()) + WINDOW_US <= start and int(ex) <= start,  # type: ignore[arg-type]
        )
    if m4.SCORES.exists():
        s = pl.read_parquet(m4.SCORES, columns=["d_us"])
        check("scores cache: no row from 2024", int(s["d_us"].max()) < SEALED_START_US)  # type: ignore[arg-type]
    for name in ("m4_results.csv", "m4_ledger.csv"):
        p = RESULTS / name
        if p.exists():
            t = pl.read_csv(p, infer_schema_length=0)
            years = (
                [int(v) for v in t["year"].to_list() if v and v.isdigit()]
                if "year" in t.columns
                else []
            )
            check(f"{name}: no year from 2024", not years or max(years) < 2024)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logic-only", action="store_true")
    a = ap.parse_args()
    logic_ranks()
    logic_percentile_and_train()
    logic_picks()
    logic_transparent()
    logic_book()
    logic_spec()
    if not a.logic_only:
        real()
    if FAILURES:
        print(f"{len(FAILURES)} failed")
        sys.exit(1)
    print("all passed")


if __name__ == "__main__":
    main()
