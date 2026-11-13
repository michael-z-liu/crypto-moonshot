"""M3 checks of the book and the scorecard: PASS/FAIL lines.

    uv run --group research python check_book.py [--skip-parity]

1. Synthetic cases, computed by hand: slots, one position per coin, a slot freed at exactly the
   next entry time, the cash skip, the engine's sell accounting, the processing order, the
   midnight marks, and the scorecard's day statistics (best days, fragile, bootstrap).
2. Parity with the production engine: touch-market candidates on 1h MAs of 50 coins with
   triggers in 2022-2023, exit E1 (time exit at 8 hours), run through `run_backtest` (Parquet
   source over the data root, `until=DATA_END`, 1m fills, fee 0.1% + slippage 0.1% per side,
   10 equal slots, $10,000) and through the book from the M1 labels. Trade lists and the daily
   equity curve must agree to rounding (relative 1e-9); every difference is listed.
3. Causality: the book cut at 2024-01-01 equals the 2022-23 part of the book cut at 2025-01-01.

Nothing here reads a fill on or after 2025-01-01.
"""

from __future__ import annotations

import argparse
import math
import time
from datetime import UTC, datetime, timedelta

import book as bk
import numpy as np
import polars as pl
from common import DATA_DIR, DATA_END, FEE, INITIAL_CAPITAL

from crypto_moonshot.backtest.engine import BacktestConfig, run_backtest
from crypto_moonshot.backtest.source import ParquetCandleSource
from crypto_moonshot.broker.base import Costs, ExitPlan
from crypto_moonshot.portfolio import EqualSlots

PARITY_SEED = 20261113
PARITY_COINS = 50
REL = 1e-9
H = 3_600_000_000
D = bk.DAY

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'} {name}" + (f" ({detail})" if detail else ""), flush=True)
    if not ok:
        failures.append(name)


def close(a: float | None, b: float | None, rel: float = REL) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= rel * max(abs(a), abs(b), 1e-12)


# --- 1. synthetic ----------------------------------------------------------------------------


def _sim(
    rows: list[tuple[int, int, int, float, float]], slots: int, symbols: list[str]
) -> tuple[list[int], list[float], list[float], list[float], dict[str, int]]:
    """rows: (sym, entry_t, exit_t, buy, sell), already in processing order."""
    a = np.array(rows, dtype=np.float64)
    return bk.simulate(
        a[:, 0].astype(np.int64),
        a[:, 1].astype(np.int64),
        a[:, 2].astype(np.int64),
        a[:, 3],
        a[:, 4],
        symbols,
        slots=slots,
    )


def synthetic() -> None:
    s = ["AAAUSDT", "BBBUSDT", "CCCUSDT"]
    # A: 2 slots, three entries at once: the third finds the slots full; stakes 10000 / 2
    taken, stakes, _, _, sk = _sim(
        [(0, 0, 10, 1.0, 1.0), (1, 0, 10, 1.0, 1.0), (2, 0, 10, 1.0, 1.0)], 2, s
    )
    check("slots: third entry skipped", taken == [0, 1] and sk["slots_full"] == 1)
    check("slots: stake = realised equity / slots", stakes == [5000.0, 5000.0])

    # B: an exit at exactly the next entry time frees its slot first (engine: exit <= entry)
    taken, stakes, _, procs, sk = _sim([(0, 0, 10, 100.0, 110.0), (1, 10, 20, 1.0, 1.0)], 1, s)
    q = 10_000 / (100 * (1 + FEE))
    p = q * 110 - q * 110 * FEE
    check("close at the entry time frees the slot", taken == [0, 1])
    check("sell accounting = engine's", close(procs[0], p), f"{procs[0]:.6f} vs {p:.6f}")
    net = 110 * (1 - FEE) / (100 * (1 + FEE)) - 1
    check("pnl = stake x net", close(procs[0] - 10_000, 10_000 * net))
    check("next stake uses realised equity after the close", close(stakes[1], p))

    # C: one position per coin; the coin is free again once its exit has passed
    taken, _, _, _, sk = _sim(
        [(0, 0, 10, 1.0, 1.0), (0, 5, 15, 1.0, 1.0), (0, 11, 20, 1.0, 1.0)], 5, s
    )
    check("one position per coin", taken == [0, 2] and sk["symbol_held"] == 1)

    # D: 2 slots; A loses half, closes; the next stake (7500 / 2 = 3750) exceeds the cash (2500)
    taken, stakes, _, procs, sk = _sim(
        [
            (0, 0, 10, 1.0, 0.5 / (1 - FEE) * (1 + FEE)),
            (1, 1, 100, 1.0, 1.0),
            (2, 20, 30, 1.0, 1.0),
        ],
        2,
        s,
    )
    check("cash skip after a loss", taken == [0, 1] and sk["cash"] == 1, f"{procs[0]:.2f}")

    # E: processing order: fill time, score (high first), trigger, symbol
    e = np.array([5, 5, 5, 5, 1], dtype=np.int64)
    trig = np.array([3, 2, 2, 2, 0], dtype=np.int64)
    sym = np.array([0, 2, 1, 0, 0], dtype=np.int64)
    score = np.array([0.9, 0.5, 0.5, 0.5, 0.1])
    check("order with scores", bk.processing_order(e, trig, sym, score).tolist() == [4, 0, 3, 2, 1])
    check("order without scores", bk.processing_order(e, trig, sym).tolist() == [4, 3, 2, 1, 0])

    # F: marks. Day 0 = 2022-01-01. Close matrix: coin 0 closes 10, 11, (no bar), 13 on days
    # 0..3, forward-filled. Position 1: entry day 0 12:00 at raw 9.5, exit day 3 06:00 ->
    # marks at days 1, 2, 3 = close of day 0, 1, 1 (no bar day 2) = 10, 11, 11. Position 2:
    # entry exactly at day 2 00:00 at raw 20 -> mark at day 2 = raw entry, day 3 = coin 1's day 2.
    day0 = int(datetime(2022, 1, 1, tzinfo=UTC).timestamp()) // 86_400
    cl = bk.Closes(np.array([[10.0, 11.0, 11.0, 13.0], [30.0, 31.0, 32.0, 33.0]]), day0)
    t0 = day0 * D
    br = bk.BookRun(
        row=np.array([0, 1]),
        sym=np.array([0, 1]),
        entry_t=np.array([t0 + 12 * H, t0 + 2 * D]),
        exit_t=np.array([t0 + 3 * D + 6 * H, t0 + 3 * D + 1]),
        stake=np.array([1000.0, 2000.0]),
        quantity=np.array([100.0, 50.0]),
        proceeds=np.array([1300.0, 2100.0]),
        entry_raw=np.array([9.5, 20.0]),
    )
    eq = bk.equity_curve(br, cl)
    ov = eq["open_value"].to_list()
    want = [0.0, 100 * 10.0, 100 * 11.0 + 50 * 20.0, 100 * 11.0 + 50 * 32.0, 0.0]
    check(
        "midnight marks",
        eq.height == 5 and all(close(a, b) for a, b in zip(ov, want, strict=True)),
        str(ov),
    )
    cash = eq["cash"].to_list()
    want_cash = [10_000.0, 9000.0, 7000.0, 7000.0, 7000 + 3400.0]
    check("curve cash", all(close(a, b) for a, b in zip(cash, want_cash, strict=True)), str(cash))

    # G: scorecard day statistics on a hand-made trade list
    import scorecard as sc

    def tr(day: str, net: float) -> dict[str, object]:
        t = datetime.fromisoformat(day).replace(tzinfo=UTC)
        return {"entry_time": t, "net_return": net}

    rows = [
        tr("2022-01-01", 0.10),
        tr("2022-01-01", 0.0),
        tr("2022-02-01", -0.01),
        tr("2023-01-01", 0.02),
        tr("2023-01-02", -0.01),
        tr("2024-03-01", 0.01),
        tr("2024-03-02", 0.03),
    ]
    t = pl.DataFrame(rows)
    # yearly means: 2022 (0.10 + 0 - 0.01)/3 = 0.03, 2023 0.005, 2024 0.02 -> mean 0.018333
    yb = sc.yearly_bp(t)
    check("yearly bp", close(yb[2022], 300) and close(yb[2023], 50) and close(yb[2024], 200))
    # without the best day (2022-01-01, sum 0.10): 2022 = -0.01 -> -100 bp
    ex1 = sc.without_best_days(t, 1)
    check("without the best day", close(ex1, (-100 + 50 + 200) / 3), f"{ex1:.3f}")
    # without each year's best trade: 2022 (0 - 0.01)/2 = -50 bp -> fragile
    eb = sc.ex_best_trade(t)
    check("without each year's best trade", close(eb[2022], -50) and close(eb[2023], -100))
    lo, hi = sc.bootstrap_ci(t)
    check(
        "bootstrap CI brackets the mean", lo <= (300 + 50 + 200) / 3 <= hi, f"[{lo:.0f}, {hi:.0f}]"
    )


# --- 2. parity with the engine ---------------------------------------------------------------


def parity() -> None:
    t_start = time.perf_counter()
    c = bk.cands()
    lg = bk.legs("touch", None, "base", "E1")
    a = bk.us(datetime(2022, 1, 1, tzinfo=UTC))
    b = bk.us(datetime(2024, 1, 1, tzinfo=UTC))
    ok = (c.family == "touch") & (c.tf == "1h") & (c.trigger_t >= a) & (c.trigger_t < b)
    coins = sorted(set(np.unique(c.sym[ok]).tolist()))
    rng = np.random.default_rng(PARITY_SEED)
    pick = np.sort(rng.choice(coins, PARITY_COINS, replace=False))
    sel = ok & np.isin(c.sym, pick)
    sel_idx = np.flatnonzero(sel)
    names = [c.symbols[i] for i in pick]
    delisted = (
        set((DATA_DIR / "delisted.txt").read_text().split())
        if (DATA_DIR / "delisted.txt").exists()
        else set()
    )
    print(
        f"  parity: {len(names)} coins ({len(set(names) & delisted)} delisted), "
        f"{len(sel_idx)} candidates",
        flush=True,
    )

    # the engine
    signals = pl.DataFrame(
        {
            "symbol": [c.symbols[s] for s in c.sym[sel_idx]],
            "open_time": c.trigger_t[sel_idx],
            "priority": c.trigger_t[sel_idx].astype(np.float64),
            "cand": sel_idx,
        }
    ).with_columns(pl.from_epoch("open_time", "us").dt.replace_time_zone("UTC"))
    config = BacktestConfig(
        exit=ExitPlan(max_hold=timedelta(hours=8)),
        costs=Costs(fee=0.001, slippage=0.001),
        sizing=EqualSlots(10),
        max_positions=10,
        initial_capital=INITIAL_CAPITAL,
        start=datetime(2022, 1, 1, tzinfo=UTC),
        end=datetime(2024, 1, 1, tzinfo=UTC),
        use_fine=True,
    )
    src = ParquetCandleSource(DATA_DIR, until=DATA_END)
    t = time.perf_counter()
    eng = run_backtest(signals, src, config)
    print(
        f"  engine: {eng.trades.height} trades in {time.perf_counter() - t:.0f}s, "
        f"skipped {dict((k, v) for k, v in eng.skipped.items() if v)}",
        flush=True,
    )

    # the book
    mask = np.isin(lg.cand, sel_idx)
    # labels whose entry is not the engine's entry candle (a gap after the touch candle)
    gap = mask & (lg.entry_t != c.trigger_t[lg.cand] + bk.FIVE)
    rows = bk.order_rows(lg, mask)
    res = bk.book(lg, rows)
    print(
        f"  book: {res.trades.height} trades, skipped {res.run.skipped}; "
        f"labels entering after a gap: {int(gap.sum())}",
        flush=True,
    )

    mine = res.trades.select(
        "symbol",
        "entry_time",
        "exit_time",
        "entry_ref",
        "entry_price",
        "exit_price",
        "stake",
        "pnl",
        "net_return",
        "cand",
    )
    theirs = eng.trades.select(
        "symbol",
        "entry_time",
        "exit_time",
        "exit_reason",
        "entry_ref",
        "entry_price",
        "exit_price",
        "stake",
        "pnl",
        "net_return",
        "cand",
    )
    j = mine.join(theirs, on=["symbol", "entry_time"], how="full", suffix="_e", coalesce=True)
    only_book = j.filter(pl.col("exit_time_e").is_null())
    only_eng = j.filter(pl.col("exit_time").is_null())
    both = j.filter(pl.col("exit_time").is_not_null() & pl.col("exit_time_e").is_not_null())
    check(
        "parity: same trades (symbol, entry time)",
        only_book.height == 0 and only_eng.height == 0,
        f"{both.height} matched, {only_book.height} only in the book, "
        f"{only_eng.height} only in the engine",
    )
    if only_book.height or only_eng.height:
        print(only_book.head(10))
        print(only_eng.head(10))

    check("parity: exit times", bool((both["exit_time"] == both["exit_time_e"]).all()))
    for col in ("entry_ref", "entry_price", "exit_price", "stake", "pnl", "net_return"):
        x, y = both[col].to_numpy(), both[f"{col}_e"].to_numpy()
        scale = np.maximum(np.maximum(np.abs(x), np.abs(y)), 1e-12)
        rel = np.abs(x - y) / scale
        # pnl and net return are differences: compare them on the stake's scale
        if col == "pnl":
            rel = np.abs(x - y) / both["stake_e"].to_numpy()
        if col == "net_return":
            rel = np.abs(x - y)
        okc = rel <= REL
        check(
            f"parity: {col}", bool(okc.all()), f"{int((~okc).sum())} differ, worst {rel.max():.1e}"
        )
        if not okc.all():
            print(both.filter(pl.Series(~okc)).head(5))
    cand_ok = (both["cand"] == both["cand_e"]).all()
    check("parity: same candidate taken", bool(cand_ok))

    em = res.equity.select("time", "equity")
    ee = eng.equity.select("time", pl.col("equity").alias("equity_e"))
    je = em.join(ee, on="time", how="full", coalesce=True)
    missing = je.filter(pl.col("equity").is_null() | pl.col("equity_e").is_null()).height
    dif = (je["equity"] - je["equity_e"]).abs() / je["equity_e"].abs()
    worst = float(dif.max() or 0)
    check(
        "parity: daily equity curve",
        missing == 0 and worst <= REL,
        f"{je.height} days, {missing} unmatched, worst rel {worst:.1e}",
    )
    ym = res.yearly.select("year", "return", "trades", "mean_bp")
    ye = eng.yearly.select("year", "return", "trades", "mean_bp")
    yj = ym.join(ye, on="year", suffix="_e")
    yok = all(
        close(r["return"], r["return_e"])
        and r["trades"] == r["trades_e"]
        and close(r["mean_bp"], r["mean_bp_e"])
        for r in yj.iter_rows(named=True)
    )
    check("parity: yearly metrics", yok)
    for r in yj.iter_rows(named=True):
        print(
            f"  {r['year']} (book/engine): trades {r['trades']:.0f}/{r['trades_e']:.0f}, "
            f"return {r['return'] * 100:+.4f}%/{r['return_e'] * 100:+.4f}%, "
            f"bp {r['mean_bp'] or math.nan:+.3f}/{r['mean_bp_e'] or math.nan:+.3f}"
        )
    reasons = dict(eng.trades.group_by("exit_reason").len().sort("exit_reason").iter_rows())
    print(f"  engine exit reasons: {reasons}")
    print(f"  parity took {time.perf_counter() - t_start:.0f}s", flush=True)


# --- 3. causality ----------------------------------------------------------------------------


def causality() -> None:
    lg = bk.legs("limit", 0.05, "base", "E1")
    cut = bk.us(datetime(2024, 1, 1, tzinfo=UTC))
    short = bk.book(lg, bk.order_rows(lg, bk.in_book(lg, cut)))
    long = bk.book(lg, bk.order_rows(lg, bk.in_book(lg, bk.HOLDOUT_US)))
    n = short.trades.height
    head = long.trades.head(n)
    same = n == long.trades.filter(pl.col("entry_time") < datetime(2024, 1, 1, tzinfo=UTC)).height
    same = same and short.trades.select("row", "stake", "pnl").equals(
        head.select("row", "stake", "pnl")
    )
    check("causality: the 2022-23 trades do not depend on later fills", same)


# --- 4. scorecard plumbing ------------------------------------------------------------------


def plumbing() -> None:
    import scorecard as sc

    c = bk.cands()
    every = sc.scorecard(
        "check-every", sc.Rule(), stage="check", source="check", ledger=False, seal=False
    )
    check(
        "card carries 2022-24 only",
        list(every.years) == [2022, 2023, 2024] and list(every.stress) == [2022, 2023, 2024],
    )
    check(
        "every candidate: no placebo, verdict says so",
        not every.placebo
        and every.verdict.criteria["c_placebo"] is None
        and "n/a" in every.verdict.reasons[-1],
    )
    # all scores 1.0 at pct 0.8 select every candidate: same book as no scores
    ones = pl.DataFrame({"cid": c.cid, "score": np.ones(len(c))})
    same = sc.scorecard(
        "check-ones",
        sc.Rule(),
        ones,
        stage="check",
        source="check",
        placebos=2,
        ledger=False,
        seal=False,
    )
    check(
        "scores of 1 = every candidate",
        all(
            same.years[y]["trades"] == every.years[y]["trades"]
            and close(same.years[y]["ret"], every.years[y]["ret"])
            for y in (2022, 2023, 2024)
        ),
    )
    check(
        "placebo of constant scores = the variant",
        all(close(p, same.mean_bp) for p in same.placebo),
    )
    # NaN scores are never traded
    half = np.where(np.arange(len(c)) % 2 == 0, 0.9, np.nan)
    hs = sc.scorecard(
        "check-half",
        sc.Rule(),
        pl.DataFrame({"cid": c.cid, "score": half}),
        stage="check",
        source="check",
        placebos=0,
        ledger=False,
        seal=False,
    )
    check("NaN scores are not traded", hs.n_selected < every.n_selected)

    # ledger upsert and seeds, on a scratch ledger
    old = sc.LEDGER, sc.LOCK
    sc.LEDGER = bk.M3 / "check_ledger.csv"
    sc.LOCK = bk.M3 / "check_ledger.lock"
    try:
        sc.LEDGER.unlink(missing_ok=True)
        sc.write_ledger(every)
        sc.write_ledger(every)
        n1 = pl.read_csv(sc.LEDGER).height
        sc.write_ledger(same)
        n2 = pl.read_csv(sc.LEDGER).height
        check("ledger: same config replaces, a new config adds", n1 == 1 and n2 == 2)
        rng = np.random.default_rng(7)
        seeds = [pl.DataFrame({"cid": c.cid, "score": rng.random(len(c))}) for _ in range(2)]
        res = sc.score_seeds(
            "check-seeds", sc.Rule(), seeds, stage="check", source="check", placebos=0, seal=False
        )
        r = res.rank_values
        check(
            "seeds: per-seed rank values and spread",
            len(r) == 2
            and close(res.spread, max(r) - min(r))
            and pl.read_csv(sc.LEDGER).height == 4,
        )
    finally:
        sc.LEDGER.unlink(missing_ok=True)
        sc.LEDGER, sc.LOCK = old


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-parity", action="store_true")
    args = ap.parse_args()
    synthetic()
    causality()
    plumbing()
    if not args.skip_parity:
        parity()
    print(f"{len(failures)} failures" + (f": {', '.join(failures)}" if failures else ""))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
