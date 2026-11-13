"""M3: the protocol's portfolio book over the M1 labels.

    uv run --group research python book.py --prepare   # build cache/m3/ (about a minute)

The production backtest engine has no limit entries, so the study keeps its own book. It replays
trades whose entry and exit are already resolved in `labels.parquet` (by the production
`resolve_exit`) through the production `Book`, `EqualSlots` sizing and the `MaxPositions` /
`OnePerSymbol` risk checks, with the engine's order of events:

1. Rows are ordered by fill (entry) time, then score (highest first), then the earliest trigger,
   then symbol. Before each new entry time, every open position whose exit time is at or before
   it is closed (in exit-time order), freeing its slot and returning its proceeds.
2. The stake is realised equity / slots (cash plus open cost basis, `EqualSlots`). An entry is
   skipped if the slots are full (`slots_full`), the coin is held (`symbol_held`), or the stake
   exceeds the cash (`cash`); otherwise it spends `min(stake, cash)`.
3. Accounting is the engine's (`SimulatedBroker`): the buy fill price is the raw entry plus the
   scenario's entry costs, quantity = stake / (buy fill x (1 + fee)); the sell fill price is the
   raw exit minus the scenario's exit costs, proceeds = quantity x sell fill x (1 - fee). So
   P&L = stake x the label's net return, up to float rounding; `net_return` = P&L / stake.
4. The mark-to-market curve has a point at every midnight UTC from the first entry's day to the
   day after the last exit, as the engine's: cash plus each open position's quantity x the
   coin's last close at or before that midnight (the daily bar's close), or x the raw entry
   price at a midnight that is the entry time itself (no candle has closed yet). Yearly returns
   and the overall metrics come from `backtest.metrics.summarize`.

The data end: a `censored` label (a position whose exit lies past the coin's last candle on a
series that has not ended) is closed at the coin's last 5m close, at that candle's close time,
reason `series_end`, with market exit costs: what the engine does at its data end. A censored
*fill* (an order still resting at the data end) is no trade. Neither exists before 2025.

Holdout: every reader takes `until` (fills before it). Development and everything printed use
`until=HOLDOUT_START`; only the scorecard's sealing path books the full range.
"""

from __future__ import annotations

import argparse
import heapq
import time
from dataclasses import dataclass, field
from datetime import datetime
from functools import cache

import numpy as np
import numpy.typing as npt
import polars as pl
from common import (
    BOOK_START,
    CACHE,
    DATA_END,
    FEE,
    HOLDOUT_START,
    INITIAL_CAPITAL,
    KLINES_5M,
    MAJORS_TOP,
    MARKET_SLIPPAGE,
    STRESS_EXTRA_COST,
    TIMEFRAMES,
)

from crypto_moonshot.backtest.metrics import summarize
from crypto_moonshot.broker.candles import from_us
from crypto_moonshot.portfolio import Book, EqualSlots
from crypto_moonshot.risk import MaxPositions, OnePerSymbol, RiskCheck, first_rejection

type F64 = npt.NDArray[np.float64]
type I64 = npt.NDArray[np.int64]
type Bools = npt.NDArray[np.bool_]

M3 = CACHE / "m3"
MINUTE = 60_000_000
FIVE = 5 * MINUTE
DAY = 86_400_000_000
EPOCH_DAY0 = int(datetime(2020, 1, 1, tzinfo=BOOK_START.tzinfo).timestamp()) // 86_400
MAJOR_SYMBOLS = ("BTCUSDT", "ETHUSDT")
TF_ORDER = tuple(TIMEFRAMES)
SCENARIOS = ("base", "stress")
MARKET_EXITS = ("stop", "time", "series_end", "trailing_stop")


def us(t: datetime) -> int:
    return int(t.timestamp()) * 1_000_000


BOOK_START_US = us(BOOK_START)
HOLDOUT_US = us(HOLDOUT_START)
DATA_END_US = us(DATA_END)


# --- prepared tables -------------------------------------------------------------------------


def key_name(family: str, buffer: float | None, scenario: str, exit_name: str) -> str:
    b = "touch" if family == "touch" else f"b{buffer:g}"
    return f"{b}-{scenario}-{exit_name}"


def _daily_parts() -> pl.DataFrame:
    from runner import PARTS

    frames = [
        pl.read_parquet(f, columns=["t", "close", "liq"]).with_columns(symbol=pl.lit(f.stem))
        for f in sorted((PARTS / "bars_1d").glob("*.parquet"))
    ]
    return pl.concat(frames)


def _majors(cands: pl.DataFrame, daily: pl.DataFrame) -> pl.Series:
    """BTC, ETH, or one of the 20 highest `liq` across all coins on the trigger's UTC day
    (M2's definition)."""
    liq = daily.filter(pl.col("liq").is_not_null()).with_columns(
        rank=pl.col("liq").rank("ordinal", descending=True).over("t")
    )
    top = (
        liq.filter(pl.col("rank") <= MAJORS_TOP)
        .select(pl.col("t").alias("day"), "symbol")
        .with_columns(major=pl.lit(True))
    )
    day = pl.col("trigger_t") - pl.col("trigger_t") % DAY
    out = (
        cands.select("symbol", day=day)
        .join(top, on=["day", "symbol"], how="left", maintain_order="left")
        .select(pl.col("major").fill_null(False) | pl.col("symbol").is_in(MAJOR_SYMBOLS))
    )
    return out.to_series()


def _last_closes(symbols: list[str]) -> pl.DataFrame:
    """Each coin's last 5m candle before the data end: close time and close."""
    rows = []
    for s in symbols:
        files = sorted((KLINES_5M / f"symbol={s}").glob("*.parquet"))
        for f in reversed(files):
            df = pl.read_parquet(f, columns=["open_time", "close"])
            df = df.with_columns(t=pl.col("open_time").dt.epoch("us")).filter(
                pl.col("t") + FIVE <= DATA_END_US
            )
            if df.height:
                last = df.sort("t").row(-1, named=True)
                rows.append((s, int(last["t"]) + FIVE, float(last["close"])))
                break
    return pl.DataFrame(
        rows,
        schema={"symbol": pl.String, "last_t": pl.Int64, "last_close": pl.Float64},
        orient="row",
    )


def prepare() -> None:
    """Build `cache/m3/`: study candidates with their attributes, one label table per
    (entry, scenario, exit) with censored labels closed at the coin's last close, the daily close
    matrix and BTC's 5m closes."""
    t0 = time.perf_counter()
    M3.mkdir(parents=True, exist_ok=True)
    cols = ["cid", "family", "symbol", "tf", "n", "trigger_t", "decision_t", "u", "placebo"]
    cands = (
        pl.read_parquet(CACHE / "candidates.parquet", columns=cols)
        .filter(~pl.col("placebo"))
        .drop("placebo")
        .sort("family", "trigger_t", "symbol", "tf", "n")
    )
    daily = _daily_parts()
    symbols = sorted(cands["symbol"].unique().to_list())
    half = pl.col("tf").replace_strict(
        {tf: TIMEFRAMES[tf].hold.total_seconds() * 500_000 for tf in TF_ORDER},
        return_dtype=pl.Float64,
    )
    window_end = (
        pl.when(pl.col("family") == "touch")
        .then(pl.col("decision_t") + FIVE)
        .otherwise(pl.col("decision_t") + half.cast(pl.Int64))
    )
    cands = cands.with_columns(
        major=_majors(cands, daily),
        window_end=window_end,
        sym=pl.col("symbol").replace_strict(
            {s: i for i, s in enumerate(symbols)}, return_dtype=pl.Int32
        ),
    ).with_row_index("cand")
    cands.write_parquet(M3 / "cands.parquet")
    pl.DataFrame({"symbol": symbols}).write_parquet(M3 / "symbols.parquet")

    # daily closes, forward-filled: close_ff[sym, d] = last daily close on day d or before
    days = (DATA_END_US // DAY) - EPOCH_DAY0
    mat = np.full((len(symbols), days), np.nan)
    d = daily.filter(pl.col("symbol").is_in(symbols), pl.col("t") >= EPOCH_DAY0 * DAY)
    d = d.with_columns(
        sym=pl.col("symbol").replace_strict(
            {s: i for i, s in enumerate(symbols)}, return_dtype=pl.Int64
        ),
        day=pl.col("t") // DAY - EPOCH_DAY0,
    ).filter(pl.col("day") < days)
    mat[d["sym"].to_numpy(), d["day"].to_numpy()] = d["close"].to_numpy()
    for i in range(len(symbols)):  # forward fill
        row = mat[i]
        ok = ~np.isnan(row)
        idx = np.where(ok, np.arange(days), -1)
        np.maximum.accumulate(idx, out=idx)
        mat[i] = np.where(idx >= 0, row[np.maximum(idx, 0)], np.nan)
    np.save(M3 / "close_ff.npy", mat)

    from events import read_5m

    btc = read_5m("BTCUSDT", DATA_END_US).select("t", "close")
    btc.write_parquet(M3 / "btc_5m.parquet")

    lab = pl.scan_parquet(CACHE / "labels.parquet")
    cand_idx = cands.select("cid", "cand", "symbol")
    censored_syms = (
        lab.filter(pl.col("status") == "censored")
        .select("cid")
        .collect()
        .join(cand_idx, on="cid")["symbol"]
        .unique()
        .to_list()
    )
    last = _last_closes(sorted(censored_syms))
    keys = (
        lab.select("buffer", "scenario", "exit")
        .unique()
        .collect()
        .sort("buffer", "scenario", "exit")
    )
    for buffer, scenario, exit_name in keys.iter_rows():
        fam = "touch" if buffer is None else "limit"
        cond = pl.col("buffer").is_null() if buffer is None else pl.col("buffer") == buffer
        df = (
            lab.filter(cond, pl.col("scenario") == scenario, pl.col("exit") == exit_name)
            .collect()
            .join(cand_idx, on="cid", how="inner")  # study MAs only
            .join(last, on="symbol", how="left")
        )
        cen = pl.col("status") == "censored"
        df = df.with_columns(
            censored=cen,
            exit_t=pl.when(cen).then(pl.col("last_t")).otherwise(pl.col("exit_t")),
            exit_price=pl.when(cen).then(pl.col("last_close")).otherwise(pl.col("exit_price")),
            reason=pl.when(cen).then(pl.lit("series_end")).otherwise(pl.col("reason")),
        )
        df = df.select(
            "cand",
            "entry_t",
            "entry_price",
            "entry_kind",
            "exit_t",
            "exit_price",
            "reason",
            "net",
            "censored",
        ).sort("entry_t", "cand")
        df.write_parquet(M3 / f"lab-{key_name(fam, buffer, scenario, exit_name)}.parquet")
    print(f"prepared cache/m3 in {time.perf_counter() - t0:.0f}s", flush=True)


# --- loaded data -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Cands:
    """The study candidates (both families), row = candidate index `cand`."""

    cid: pl.Series
    family: npt.NDArray[np.str_]
    sym: I64
    tf: npt.NDArray[np.str_]
    n: I64
    trigger_t: I64
    decision_t: I64
    window_end: I64
    major: Bools
    symbols: list[str]

    def __len__(self) -> int:
        return len(self.sym)


@dataclass(frozen=True)
class Legs:
    """Trades of one label key: one row per (candidate, fill), sorted by entry time.

    `buy` / `sell` are the fill prices after the scenario's costs (before fees); `gross` is the raw
    exit over the raw entry; `btc` BTCUSDT's gross return over the same window (5m closes)."""

    cand: I64
    entry_t: I64
    exit_t: I64
    entry_raw: F64
    buy: F64
    sell: F64
    net: F64
    gross: F64
    btc: F64
    censored: Bools

    def __len__(self) -> int:
        return len(self.cand)


@dataclass(frozen=True)
class Closes:
    """Daily closes forward-filled: `close_ff[sym, d]` is the last close on day `day0 + d` or
    before (NaN before the coin's first bar)."""

    close_ff: F64
    day0: int  # epoch day of column 0

    def at_midnight(self, sym: I64, t: I64) -> F64:
        """The last close at or before midnight `t` (µs): the daily bar of the day before."""
        return self.close_ff[sym, t // DAY - 1 - self.day0]


@cache
def cands() -> Cands:
    if not (M3 / "cands.parquet").exists():
        prepare()
    df = pl.read_parquet(M3 / "cands.parquet")
    symbols = pl.read_parquet(M3 / "symbols.parquet")["symbol"].to_list()
    return Cands(
        cid=df["cid"],
        family=df["family"].to_numpy(),
        sym=df["sym"].to_numpy().astype(np.int64),
        tf=df["tf"].to_numpy(),
        n=df["n"].to_numpy().astype(np.int64),
        trigger_t=df["trigger_t"].to_numpy(),
        decision_t=df["decision_t"].to_numpy(),
        window_end=df["window_end"].to_numpy(),
        major=df["major"].to_numpy(),
        symbols=symbols,
    )


@cache
def closes() -> Closes:
    return Closes(np.load(M3 / "close_ff.npy"), EPOCH_DAY0)


@cache
def _btc() -> tuple[I64, F64]:
    b = pl.read_parquet(M3 / "btc_5m.parquet")
    return b["t"].to_numpy() + FIVE, b["close"].to_numpy()


def btc_return(a: I64, b: I64) -> F64:
    """BTCUSDT's return from the last 5m close at or before `a` to the last at or before `b`."""
    ct, c = _btc()
    ia = np.searchsorted(ct, a, side="right") - 1
    ib = np.searchsorted(ct, b, side="right") - 1
    return np.where(ia >= 0, c[ib] / c[np.maximum(ia, 0)] - 1, np.nan)


@cache
def legs(family: str, buffer: float | None, scenario: str, exit_name: str) -> Legs:
    """One label key's trades (all years: callers filter on entry time)."""
    df = pl.read_parquet(M3 / f"lab-{key_name(family, buffer, scenario, exit_name)}.parquet")
    extra = STRESS_EXTRA_COST if scenario == "stress" else 0.0
    taker = (df["entry_kind"] == "taker").to_numpy()
    market = df["reason"].is_in(MARKET_EXITS).to_numpy()
    entry = df["entry_price"].to_numpy()
    exit_ = df["exit_price"].to_numpy()
    # same association as the label's net and as `Costs.buy_price` / `Costs.sell_price`
    buy = entry * (1 + np.where(taker, MARKET_SLIPPAGE, 0.0) + extra)
    sell = exit_ * (1 - np.where(market, MARKET_SLIPPAGE, 0.0) - extra)
    net = sell * (1 - FEE) / (buy * (1 + FEE)) - 1
    censored = df["censored"].to_numpy()
    lab_net = df["net"].to_numpy()
    done = ~censored
    assert np.allclose(net[done], lab_net[done], rtol=0, atol=1e-12), "net differs from the label"
    entry_t = df["entry_t"].to_numpy()
    exit_t = df["exit_t"].to_numpy()
    return Legs(
        cand=df["cand"].to_numpy().astype(np.int64),
        entry_t=entry_t,
        exit_t=exit_t,
        entry_raw=entry,
        buy=buy,
        sell=sell,
        net=net,
        gross=exit_ / entry - 1,
        btc=btc_return(entry_t, exit_t),
        censored=censored,
    )


# --- the book --------------------------------------------------------------------------------


@dataclass
class BookRun:
    """A book's trades (in entry order) and skip counts. Indices `row` point into the `Legs`."""

    row: I64
    sym: I64
    entry_t: I64
    exit_t: I64
    stake: F64
    quantity: F64
    proceeds: F64
    entry_raw: F64
    skipped: dict[str, int] = field(default_factory=dict)

    @property
    def pnl(self) -> F64:
        return self.proceeds - self.stake

    @property
    def net_return(self) -> F64:
        return self.pnl / self.stake


def simulate(
    sym: I64,
    entry_t: I64,
    exit_t: I64,
    buy: F64,
    sell: F64,
    symbols: list[str],
    *,
    slots: int = 10,
    initial: float = INITIAL_CAPITAL,
) -> tuple[list[int], list[float], list[float], list[float], dict[str, int]]:
    """Run the book over rows already in processing order. Returns, per taken row (in entry
    order): its position in the input, stake, quantity and proceeds; and the skip counts."""
    book = Book(cash=initial)
    sizing = EqualSlots(slots)
    checks: list[RiskCheck] = [MaxPositions(slots), OnePerSymbol()]
    heap: list[tuple[int, int, str, float]] = []  # (exit time, seq, symbol, proceeds)
    taken: list[int] = []
    stakes: list[float] = []
    qtys: list[float] = []
    procs: list[float] = []
    skipped = {"slots_full": 0, "symbol_held": 0, "cash": 0}
    last = -1
    ent = entry_t.tolist()
    ext = exit_t.tolist()
    sy = sym.tolist()
    bl = buy.tolist()
    sl = sell.tolist()
    for i in range(len(ent)):
        e = ent[i]
        if e != last:
            while heap and heap[0][0] <= e:
                xt, _, s, p = heapq.heappop(heap)
                book.close(s, p, from_us(xt))
            last = e
        symbol = symbols[sy[i]]
        stake = sizing.stake(book)
        reason = first_rejection(checks, book, symbol, stake)
        if reason is None and not 0 < stake <= book.cash * (1 + 1e-9):
            reason = "cash"
        if reason is not None:
            skipped[reason] += 1
            continue
        stake = min(stake, book.cash)
        book.open(symbol, stake, from_us(e))
        q = stake / (bl[i] * (1 + FEE))
        notional = sl[i] * q
        proceeds = notional - notional * FEE
        heapq.heappush(heap, (ext[i], len(taken), symbol, proceeds))
        taken.append(i)
        stakes.append(stake)
        qtys.append(q)
        procs.append(proceeds)
    return taken, stakes, qtys, procs, skipped


def run(lg: Legs, rows: I64, *, slots: int = 10, initial: float = INITIAL_CAPITAL) -> BookRun:
    """Book the `Legs` rows `rows` (already filtered and in processing order)."""
    c = cands()
    sym = c.sym[lg.cand[rows]]
    taken, stakes, qtys, procs, skipped = simulate(
        sym,
        lg.entry_t[rows],
        lg.exit_t[rows],
        lg.buy[rows],
        lg.sell[rows],
        c.symbols,
        slots=slots,
        initial=initial,
    )
    t = rows[np.asarray(taken, dtype=np.int64)]
    return BookRun(
        row=t,
        sym=c.sym[lg.cand[t]],
        entry_t=lg.entry_t[t],
        exit_t=lg.exit_t[t],
        stake=np.asarray(stakes),
        quantity=np.asarray(qtys),
        proceeds=np.asarray(procs),
        entry_raw=lg.entry_raw[t],
        skipped=skipped,
    )


def equity_curve(br: BookRun, cl: Closes, initial: float = INITIAL_CAPITAL) -> pl.DataFrame:
    """Mark-to-market equity at every midnight UTC from the first entry's day to the day after
    the last exit (the engine's grid and formula)."""
    schema = {
        "time": pl.Datetime("us", "UTC"),
        "equity": pl.Float64(),
        "cash": pl.Float64(),
        "open_value": pl.Float64(),
        "open_positions": pl.Int64(),
    }
    if len(br.row) == 0:
        return pl.DataFrame(schema=schema)
    order_in = np.argsort(br.entry_t, kind="stable")
    order_out = np.argsort(br.exit_t, kind="stable")
    entry = br.entry_t[order_in]
    exit_ = br.exit_t[order_out]
    t0 = int(entry[0]) // DAY * DAY
    t1 = -(-int(exit_[-1]) // DAY) * DAY
    grid = np.arange(t0, t1 + DAY, DAY, dtype=np.int64)
    n_in = np.searchsorted(entry, grid, side="right")
    n_out = np.searchsorted(exit_, grid, side="right")
    spent = np.concatenate(([0.0], np.cumsum(br.stake[order_in])))[n_in]
    returned = np.concatenate(([0.0], np.cumsum(br.proceeds[order_out])))[n_out]
    cash = initial - spent + returned
    # marks: each midnight in [entry, exit)
    first = -(-br.entry_t // DAY) * DAY
    count = np.maximum(-(-(br.exit_t - first) // DAY), 0)
    k = np.repeat(np.arange(len(br.row)), count)
    offs = np.arange(int(count.sum())) - np.repeat(np.cumsum(count) - count, count)
    mt = first[k] + offs * DAY
    price = np.where(mt == br.entry_t[k], br.entry_raw[k], cl.at_midnight(br.sym[k], mt))
    assert not np.isnan(price).any(), "a mark without a daily close"
    open_value = np.bincount((mt - t0) // DAY, price * br.quantity[k], minlength=len(grid))
    return pl.DataFrame(
        {
            "time": grid,
            "equity": cash + open_value,
            "cash": cash,
            "open_value": open_value,
            "open_positions": (n_in - n_out).astype(np.int64),
        }
    ).with_columns(pl.from_epoch("time", "us").dt.replace_time_zone("UTC"))


def trade_frame(br: BookRun, lg: Legs) -> pl.DataFrame:
    """The trade list in the engine's column names (plus the label row and candidate)."""
    c = cands()
    cand = lg.cand[br.row]
    return pl.DataFrame(
        {
            "row": br.row,
            "cand": cand,
            "symbol": np.asarray(c.symbols, dtype=object)[br.sym].astype(str),
            "tf": c.tf[cand],
            "entry_time": br.entry_t,
            "exit_time": br.exit_t,
            "entry_ref": br.entry_raw,
            "entry_price": lg.buy[br.row],
            "exit_price": lg.sell[br.row],
            "quantity": br.quantity,
            "stake": br.stake,
            "pnl": br.pnl,
            "net_return": br.net_return,
            "gross": lg.gross[br.row],
            "btc": lg.btc[br.row],
            "censored": lg.censored[br.row],
        }
    ).with_columns(
        pl.from_epoch("entry_time", "us").dt.replace_time_zone("UTC"),
        pl.from_epoch("exit_time", "us").dt.replace_time_zone("UTC"),
    )


@dataclass(frozen=True)
class Result:
    run: BookRun
    trades: pl.DataFrame
    equity: pl.DataFrame
    metrics: dict[str, float]
    yearly: pl.DataFrame


def book(lg: Legs, rows: I64, *, slots: int = 10) -> Result:
    """Run the book, its equity curve and the production metrics."""
    br = run(lg, rows, slots=slots)
    trades = trade_frame(br, lg)
    equity = equity_curve(br, closes())
    metrics, yearly = summarize(trades, equity, INITIAL_CAPITAL)
    return Result(br, trades, equity, metrics, yearly)


def in_book(lg: Legs, until: int = HOLDOUT_US) -> Bools:
    """Rows filled in [BOOK_START, until)."""
    return (lg.entry_t >= BOOK_START_US) & (lg.entry_t < until)


def processing_order(entry_t: I64, trigger_t: I64, sym: I64, score: F64 | None = None) -> I64:
    """Positions in processing order: fill time, score (highest first), earliest trigger, then
    symbol (codes follow the alphabetical order of the symbols)."""
    keys: list[npt.NDArray[np.generic]] = [sym, trigger_t]
    if score is not None:
        keys.append(-score)
    keys.append(entry_t)
    return np.lexsort(keys)


def order_rows(lg: Legs, mask: Bools, score: F64 | None = None) -> I64:
    """Rows of `mask` in processing order; `score` is per candidate."""
    c = cands()
    k = np.flatnonzero(mask)
    cand = lg.cand[k]
    sc = None if score is None else score[cand]
    return k[processing_order(lg.entry_t[k], c.trigger_t[cand], c.sym[cand], sc)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    args = ap.parse_args()
    if args.prepare:
        prepare()
    t = time.perf_counter()
    lg = legs("limit", 0.05, "base", "E1")
    c = cands()
    print(f"loaded {len(c)} candidates, {len(lg)} legs in {time.perf_counter() - t:.1f}s")
    rows = order_rows(lg, in_book(lg))
    t = time.perf_counter()
    book(lg, rows)
    print(f"book of {len(rows)} rows (fills 2022-24): {time.perf_counter() - t:.2f}s")


if __name__ == "__main__":
    main()
