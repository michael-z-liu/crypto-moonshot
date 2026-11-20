from __future__ import annotations

import os
import signal
import sqlite3
import subprocess
import sys
import textwrap
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from broker_fakes import T0, Bar, candles, flat

from crypto_moonshot.broker.base import (
    Costs,
    ExitPlan,
    ExitReason,
    Fill,
    Order,
    OrderType,
    Side,
    client_order_id,
)
from crypto_moonshot.broker.paper import (
    BrokerEvent,
    EntryFilled,
    ExitFilled,
    OrderCancelled,
    PaperBroker,
)
from crypto_moonshot.live.state import (
    MIGRATIONS,
    SCHEMA_VERSION,
    BatchStatus,
    Control,
    Controls,
    EventLevel,
    Migration,
    OrderRecord,
    OrderStatus,
    PositionStatus,
    ProviderRecord,
    SchemaVersionError,
    SignalOutcome,
    SignalRow,
    StateError,
    StateStore,
    StrategyRecord,
    parse_time,
    time_text,
)
from crypto_moonshot.models import Interval
from crypto_moonshot.portfolio import Book

M5 = timedelta(minutes=5)
PLAN = ExitPlan(take_profit=0.2, stop=0.05, trailing=0.03, max_hold=timedelta(hours=3))
PLUS2 = timezone(timedelta(hours=2))


@pytest.fixture
def store(tmp_path: Path) -> StateStore:
    opened = StateStore.open(tmp_path / "state.sqlite")
    opened.upsert_strategy("alpha", {"preset": "capitulation", "stake": 50}, 1_000.0)
    opened.upsert_strategy("beta", {}, 500.0)
    return opened


def is_utc(dt: datetime | None) -> bool:
    return dt is not None and dt.tzinfo is UTC


def order_record(
    strategy: str = "alpha",
    symbol: str = "ABCUSDT",
    k: int = 0,
    side: Side = Side.BUY,
    signal_id: int | None = None,
) -> OrderRecord:
    when = T0 + k * M5
    if side is Side.BUY:
        order = Order(
            symbol,
            side,
            when,
            quote_amount=50.0,
            exit_plan=PLAN,
            id=client_order_id(strategy, symbol, when - M5, side),
        )
    else:
        order = Order(symbol, side, when, id=client_order_id(strategy, symbol, when - M5, side))
    return OrderRecord.from_order(order, strategy, Interval.M5, signal_id)


# Schema and migrations -------------------------------------------------------------------


def test_a_new_file_gets_the_schema_and_safe_settings(store: StateStore) -> None:
    db = store._db
    assert db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert db.execute("PRAGMA synchronous").fetchone()[0] == 2  # FULL
    assert db.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert store.schema_version == SCHEMA_VERSION == 1
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert tables >= {
        "meta",
        "strategies",
        "ticks",
        "batches",
        "signals",
        "orders",
        "fills",
        "positions",
        "controls",
        "events",
        "providers",
    }


V2: Migration = (
    "CREATE TABLE notes (id INTEGER PRIMARY KEY, text TEXT NOT NULL)",
    "ALTER TABLE strategies ADD COLUMN enabled INTEGER NOT NULL DEFAULT 1",
)


def test_opening_an_older_file_runs_the_missing_migrations(tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite"
    with StateStore.open(path) as old:
        old.upsert_strategy("alpha", {"x": 1}, 10.0)
        assert old.schema_version == 1
    with StateStore.open(path, migrations=(*MIGRATIONS, V2)) as new:
        assert new.schema_version == 2
        assert new.strategies() == [StrategyRecord("alpha", {"x": 1}, 10.0)]
        row = new._db.execute("SELECT enabled FROM strategies").fetchone()
        assert row[0] == 1
        new._db.execute("INSERT INTO notes (text) VALUES ('ok')")
    with StateStore.open(path, migrations=(*MIGRATIONS, V2)) as again:
        assert again.schema_version == 2  # nothing left to run


def test_a_failed_migration_changes_nothing(tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite"
    StateStore.open(path).close()
    broken: Migration = ("CREATE TABLE notes (id INTEGER)", "THIS IS NOT SQL")
    with pytest.raises(sqlite3.OperationalError):
        StateStore.open(path, migrations=(*MIGRATIONS, broken))
    with StateStore.open(path) as store:
        assert store.schema_version == 1
        tables = {r[0] for r in store._db.execute("SELECT name FROM sqlite_master")}
        assert "notes" not in tables


def test_a_newer_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite"
    with StateStore.open(path, migrations=(*MIGRATIONS, V2)):
        pass
    with pytest.raises(SchemaVersionError, match="schema version 2, newer than this build's 1"):
        StateStore.open(path)
    with StateStore.open(path, migrations=(*MIGRATIONS, V2)) as store:
        assert store.schema_version == 2  # untouched


# Round trips -----------------------------------------------------------------------------


def test_times_are_stored_as_sortable_utc_text() -> None:
    local = datetime(2024, 3, 1, 1, 30, tzinfo=PLUS2)
    assert time_text(local) == "2024-02-29T23:30:00.000000Z"  # a leap day
    assert parse_time(time_text(local)) == local
    assert is_utc(parse_time(time_text(local)))
    with pytest.raises(ValueError, match="timezone-aware"):
        time_text(datetime(2024, 1, 1))  # noqa: DTZ001 - checking the refusal
    with pytest.raises(StateError, match="time zone"):
        parse_time("2024-01-01T00:00:00")


def test_strategies_ticks_controls_and_providers_round_trip(store: StateStore) -> None:
    store.upsert_strategy("alpha", {"preset": "f2"}, 2_000.0)
    assert store.strategies() == [
        StrategyRecord("alpha", {"preset": "f2"}, 2_000.0),
        StrategyRecord("beta", {}, 500.0),
    ]
    assert store.strategy("beta") == StrategyRecord("beta", {}, 500.0)
    assert store.strategy("gamma") is None

    assert store.last_tick(Interval.M5) is None
    store.set_last_tick(Interval.M5, T0.astimezone(PLUS2))
    store.set_last_tick(Interval.H1, T0)
    store.set_last_tick(Interval.M5, T0 + M5)
    assert store.last_tick(Interval.M5) == T0 + M5
    assert is_utc(store.last_tick(Interval.M5))
    assert store.last_ticks() == {Interval.M5: T0 + M5, Interval.H1: T0}

    assert store.controls() == Controls()
    store.set_control(Control.PAUSED, True)
    store.set_control(Control.FLATTEN_REQUESTED, True)
    store.set_control(Control.FLATTEN_REQUESTED, False)
    assert store.controls() == Controls(paused=True, flatten_requested=False)

    store.provider_seen("cnn", T0, batch_time=T0 - M5)
    store.provider_seen("cnn", T0 + M5)  # a heartbeat without a batch keeps the batch time
    [provider] = store.providers()
    assert provider == ProviderRecord("cnn", T0 + M5, T0 - M5)
    assert is_utc(provider.last_seen)
    assert is_utc(provider.last_batch_time)


def test_events_since_an_id(store: StateStore) -> None:
    first = store.add_event(T0, EventLevel.INFO, "start", "daemon started")
    second = store.add_event(
        T0 + M5, EventLevel.WARNING, "stale", "BTCUSDT is stale", {"symbols": ["BTCUSDT"]}
    )
    third = store.add_event(T0 + 2 * M5, EventLevel.ERROR, "provider", "cnn missed a batch")
    assert first < second < third
    events = store.events_since(first)
    assert [e.id for e in events] == [second, third]
    assert events[0].data == {"symbols": ["BTCUSDT"]}
    assert events[0].level is EventLevel.WARNING
    assert all(is_utc(e.time) for e in events)
    assert [e.kind for e in store.events_since()] == ["start", "stale", "provider"]
    assert [e.id for e in store.events_since(0, limit=1)] == [first]


def test_batches_and_signals_round_trip(store: StateStore) -> None:
    batch = store.record_batch(
        "alpha",
        Interval.M5,
        T0,
        BatchStatus.LATE,
        received_at=(T0 + M5 + timedelta(seconds=7)).astimezone(PLUS2),
        lateness=timedelta(seconds=7, microseconds=250),
        signals=[
            SignalRow("ABCUSDT", priority=1.5, weight=0.5, info={"score": 0.9, "tag": "x"}),
            SignalRow(
                "XYZUSDT",
                action="exit",
                exit_override={"take_profit": 0.1},
                outcome=SignalOutcome.IGNORED,
                reason="not_holding",
            ),
        ],
    )
    assert (batch.strategy, batch.interval, batch.candle_time) == ("alpha", Interval.M5, T0)
    assert batch.status is BatchStatus.LATE
    assert batch.received_at == T0 + M5 + timedelta(seconds=7)
    assert is_utc(batch.received_at)
    assert batch.lateness == timedelta(seconds=7, microseconds=250)
    enter, leave = batch.signals
    assert (enter.symbol, enter.action, enter.priority, enter.weight) == (
        "ABCUSDT",
        "enter",
        1.5,
        0.5,
    )
    assert enter.info == {"score": 0.9, "tag": "x"}
    assert (enter.outcome, enter.reason, enter.exit_override) == (None, None, None)
    assert (leave.action, leave.exit_override) == ("exit", {"take_profit": 0.1})
    assert (leave.outcome, leave.reason) == (SignalOutcome.IGNORED, "not_holding")

    store.set_signal_outcome(enter.id, SignalOutcome.REJECTED, "slots_full")
    again = store.batch(batch.id)
    assert again is not None
    assert (again.signals[0].outcome, again.signals[0].reason) == (
        SignalOutcome.REJECTED,
        "slots_full",
    )
    with pytest.raises(KeyError):
        store.set_signal_outcome(999, SignalOutcome.PLACED)

    missed = store.record_batch("beta", Interval.M5, T0, BatchStatus.MISSED)
    assert (missed.received_at, missed.lateness, missed.signals) == (None, None, ())
    with pytest.raises(sqlite3.IntegrityError):  # one batch per strategy, interval and candle
        store.record_batch("beta", Interval.M5, T0, BatchStatus.ON_TIME)
    with pytest.raises(sqlite3.IntegrityError):  # an unknown strategy
        store.record_batch("gamma", Interval.M5, T0, BatchStatus.ON_TIME)
    assert store.batch(12345) is None


def test_orders_and_fills_round_trip(store: StateStore) -> None:
    batch = store.record_batch(
        "alpha", Interval.M5, T0, BatchStatus.ON_TIME, signals=[SignalRow("ABCUSDT")]
    )
    record = order_record(k=1, signal_id=batch.signals[0].id)
    store.insert_order(record)
    assert store.order(record.id) == record
    assert store.order(record.id).to_order() == record.to_order()  # type: ignore[union-attr]
    assert record.to_order().exit_plan == PLAN
    assert is_utc(store.order(record.id).time)  # type: ignore[union-attr]
    sell = order_record(k=5, side=Side.SELL)
    store.insert_order(sell)
    assert [o.id for o in store.pending_orders()] == [record.id, sell.id]
    assert store.pending_orders("beta") == []

    with pytest.raises(sqlite3.IntegrityError):
        store.insert_order(record)  # the client order ID is the key: never twice
    with pytest.raises(StateError, match="must be pending"):
        store.insert_order(replace(record, id="other", status=OrderStatus.FILLED))

    fill = Fill(record.id, "ABCUSDT", Side.BUY, (T0 + M5).astimezone(PLUS2), 100.05, 0.4995, 0.05)
    stored = store.record_fill(fill)
    assert (stored.order_id, stored.time, stored.price, stored.quantity, stored.fee) == (
        record.id,
        T0 + M5,
        100.05,
        0.4995,
        0.05,
    )
    assert is_utc(stored.time)
    assert stored.reason is None
    store.mark_order_filled(record.id)
    exit_fill = Fill(
        sell.id, "ABCUSDT", Side.SELL, T0 + 6 * M5, 99.0, 0.4995, 0.04, ExitReason.STOP
    )
    store.record_fill(exit_fill)
    assert [f.reason for f in store.fills()] == [None, "stop"]
    assert store.fills(record.id) == [stored]
    with pytest.raises(sqlite3.IntegrityError):
        store.record_fill(Fill("nope", "ABCUSDT", Side.BUY, T0, 1.0, 1.0, 0.0))

    assert store.order(record.id).status is OrderStatus.FILLED  # type: ignore[union-attr]
    with pytest.raises(StateError, match="filled, not pending"):
        store.mark_order_cancelled(record.id, "late")
    store.mark_order_rejected(sell.id, "not_holding")
    rejected = store.order(sell.id)
    assert rejected is not None
    assert (rejected.status, rejected.reason) == (OrderStatus.REJECTED, "not_holding")
    with pytest.raises(KeyError):
        store.mark_order_filled("missing")
    assert store.pending_orders() == []


def test_position_lifecycle(store: StateStore) -> None:
    entry = order_record(k=1)
    store.insert_order(entry)
    position = store.insert_position(
        "alpha",
        "ABCUSDT",
        Interval.M5,
        (T0 + M5).astimezone(PLUS2),
        entry.id,
        50.0,
        exit_plan=PLAN,
        signal_info={"score": 3},
    )
    assert position.status is PositionStatus.PENDING
    assert (position.placed_at, position.cost, position.exit_plan) == (T0 + M5, 50.0, PLAN)
    assert position.signal_info == {"score": 3}
    assert store.live_position("ABCUSDT") == position
    assert store.position_for_order(entry.id) == position
    with pytest.raises(StateError, match="not open"):
        position.to_paper()
    with pytest.raises(StateError, match="pending, not open"):
        store.close_position(
            position.id, exit_time=T0, exit_price=1.0, exit_reason="stop", proceeds=1.0
        )

    bracket = OrderRecord(
        "bracket-1",
        "alpha",
        "ABCUSDT",
        Interval.M5,
        Side.SELL,
        OrderType.BRACKET,
        T0 + M5,
        quantity=0.4995,
        exit_plan=PLAN,
    )
    store.insert_order(bracket)
    store.open_position(
        position.id,
        entry_time=T0 + M5,
        entry_ref=100.0,
        entry_price=100.05,
        quantity=0.4995,
        cost=50.0,
        bracket_id=bracket.id,
    )
    opened = store.position(position.id)
    assert opened is not None
    assert opened.status is PositionStatus.OPEN
    assert (opened.entry_time, opened.entry_ref, opened.entry_price) == (T0 + M5, 100.0, 100.05)
    assert is_utc(opened.entry_time)
    assert store.position_for_order(bracket.id) == opened
    paper = opened.to_paper()
    assert (paper.symbol, paper.quantity, paper.bracket_id, paper.exit_plan) == (
        "ABCUSDT",
        0.4995,
        bracket.id,
        PLAN,
    )
    store.set_position_bracket(position.id, None)
    store.set_position_bracket(position.id, bracket.id)
    with pytest.raises(StateError, match="open, not pending"):
        store.open_position(
            position.id, entry_time=T0, entry_ref=1, entry_price=1, quantity=1, cost=1
        )

    closed = store.close_position(
        position.id,
        exit_time=(T0 + 9 * M5).astimezone(PLUS2),
        exit_price=119.94,
        exit_reason="take_profit",
        proceeds=59.85,
    )
    assert closed.status is PositionStatus.CLOSED
    assert closed.exit_time == T0 + 9 * M5
    assert is_utc(closed.exit_time)
    assert (closed.exit_price, closed.exit_reason, closed.proceeds) == (
        119.94,
        "take_profit",
        59.85,
    )
    assert closed.pnl == pytest.approx(9.85)
    assert store.live_position("ABCUSDT") is None
    assert store.closed_positions() == [closed]
    assert store.closed_positions("beta") == []

    other = order_record("beta", k=2)
    store.insert_order(other)
    pending = store.insert_position("beta", "ABCUSDT", Interval.M5, T0 + 2 * M5, other.id, 20.0)
    store.cancel_position(pending.id, T0 + 3 * M5, "no_entry_candle")
    cancelled = store.position(pending.id)
    assert cancelled is not None
    assert (cancelled.status, cancelled.exit_time, cancelled.exit_reason) == (
        PositionStatus.CANCELLED,
        T0 + 3 * M5,
        "no_entry_candle",
    )
    assert store.live_positions() == []
    with pytest.raises(KeyError):
        store.cancel_position(999, T0, "x")


def test_only_one_strategy_holds_a_symbol(store: StateStore) -> None:
    first, second = order_record("alpha", k=1), order_record("beta", k=1)
    store.insert_order(first)
    store.insert_order(second)
    store.insert_position("alpha", "ABCUSDT", Interval.M5, T0 + M5, first.id, 50.0)
    with pytest.raises(sqlite3.IntegrityError):
        store.insert_position("beta", "ABCUSDT", Interval.M5, T0 + M5, second.id, 50.0)
    with pytest.raises(sqlite3.IntegrityError):  # the entry order must exist
        store.insert_position("beta", "XYZUSDT", Interval.M5, T0 + M5, "missing", 50.0)


# Transactions and crashes ----------------------------------------------------------------


def test_an_exception_rolls_the_whole_transaction_back(store: StateStore) -> None:
    def step() -> None:
        with store.transaction():
            store.add_event(T0, EventLevel.INFO, "a", "a")
            store.set_control(Control.PAUSED, True)
            store.insert_order(order_record())
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        step()
    assert store.events_since() == []
    assert store.controls() == Controls()
    assert store.pending_orders() == []
    assert not store.in_transaction


def test_nested_transactions_are_savepoints(store: StateStore) -> None:
    def inner() -> None:
        with store.transaction():
            store.add_event(T0, EventLevel.INFO, "inner", "dropped")
            raise ValueError("inner")

    with store.transaction():
        store.add_event(T0, EventLevel.INFO, "outer", "kept")
        with pytest.raises(ValueError, match="inner"):
            inner()
        assert store.in_transaction
        with store.transaction():
            store.add_event(T0, EventLevel.INFO, "inner2", "kept")
    assert [e.kind for e in store.events_since()] == ["outer", "inner2"]


def test_writes_are_visible_to_another_connection_only_after_commit(
    store: StateStore, tmp_path: Path
) -> None:
    other = sqlite3.connect(tmp_path / "state.sqlite")
    count = "SELECT count(*) FROM events"
    with store.transaction():
        store.add_event(T0, EventLevel.INFO, "a", "a")
        assert other.execute(count).fetchone()[0] == 0
    assert other.execute(count).fetchone()[0] == 1
    other.close()


CHILD_SETUP = """
import os, sys
from datetime import UTC, datetime
from crypto_moonshot.broker.base import Fill, Order, Side
from crypto_moonshot.live.state import Control, EventLevel, OrderRecord, StateStore
from crypto_moonshot.models import Interval

T0 = datetime(2024, 1, 1, tzinfo=UTC)
store = StateStore.open(sys.argv[1])
"""


def run_child(script: str, path: Path) -> subprocess.Popen[bytes]:
    code = CHILD_SETUP + textwrap.dedent(script)
    return subprocess.Popen([sys.executable, "-c", code, str(path)])


def test_a_process_dying_mid_transaction_leaves_the_last_committed_state(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.sqlite"
    child = run_child(
        """
        store.upsert_strategy("alpha", {}, 100.0)
        store.add_event(T0, EventLevel.INFO, "committed", "kept")
        with store.transaction():
            store.add_event(T0, EventLevel.INFO, "uncommitted", "lost")
            store.set_control(Control.PAUSED, True)
            store.upsert_strategy("beta", {}, 1.0)
            os._exit(3)
        """,
        path,
    )
    assert child.wait(timeout=30) == 3
    with StateStore.open(path) as store:
        assert store._db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert [e.kind for e in store.events_since()] == ["committed"]
        assert store.controls() == Controls()
        assert [s.name for s in store.strategies()] == ["alpha"]


def test_a_killed_writer_leaves_a_consistent_file(tmp_path: Path) -> None:
    """kill -9 at an arbitrary point of a loop of transactions, each writing an order, its fill
    and an event: on reopen every committed step is whole and none is half there."""
    path = tmp_path / "state.sqlite"
    StateStore.open(path).close()
    child = run_child(
        """
        store.upsert_strategy("alpha", {}, 100.0)
        k = 0
        while True:
            with store.transaction():
                oid = f"order-{k}"
                store.insert_order(OrderRecord.from_order(
                    Order("ABCUSDT", Side.BUY, T0, quote_amount=1.0, id=oid), "alpha", Interval.M5
                ))
                store.record_fill(Fill(oid, "ABCUSDT", Side.BUY, T0, 1.0, 1.0, 0.0))
                store.mark_order_filled(oid)
                store.add_event(T0, EventLevel.INFO, "step", str(k))
            k += 1
        """,
        path,
    )
    reader = sqlite3.connect(path)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            if reader.execute("SELECT count(*) FROM events").fetchone()[0] >= 50:
                break
        except sqlite3.OperationalError:
            pass
        time.sleep(0.01)
    reader.close()
    os.kill(child.pid, signal.SIGKILL)
    child.wait(timeout=30)
    with StateStore.open(path) as store:
        db = store._db
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        orders = db.execute("SELECT count(*) FROM orders").fetchone()[0]
        assert orders >= 50
        assert db.execute("SELECT count(*) FROM fills").fetchone()[0] == orders
        assert db.execute("SELECT count(*) FROM events").fetchone()[0] == orders
        assert store.pending_orders() == []
        steps = [int(e.message) for e in store.events_since()]
        assert steps == list(range(orders))


# Derived books ---------------------------------------------------------------------------


class Ledger:
    """Writes positions the way the engine will, and keeps the events a `Book` would see."""

    def __init__(self, store: StateStore, strategy: str = "alpha") -> None:
        self.store = store
        self.strategy = strategy
        self.n = 0

    def place(self, symbol: str, at: datetime, cost: float) -> int:
        self.n += 1
        order = Order(symbol, Side.BUY, at, quote_amount=cost, id=f"{self.strategy}-{self.n}")
        with self.store.transaction():
            self.store.insert_order(OrderRecord.from_order(order, self.strategy, Interval.M5))
            position = self.store.insert_position(
                self.strategy, symbol, Interval.M5, at, order.id, cost
            )
        return position.id

    def fill(self, position_id: int, at: datetime) -> None:
        position = self.store.position(position_id)
        assert position is not None
        self.store.mark_order_filled(position.entry_order_id)
        self.store.open_position(
            position_id,
            entry_time=at,
            entry_ref=100.0,
            entry_price=100.0,
            quantity=position.cost / 100,
            cost=position.cost,
        )

    def close(self, position_id: int, at: datetime, proceeds: float) -> None:
        self.store.close_position(
            position_id, exit_time=at, exit_price=1.0, exit_reason="stop", proceeds=proceeds
        )

    def cancel(self, position_id: int, at: datetime) -> None:
        position = self.store.position(position_id)
        assert position is not None
        with self.store.transaction():
            self.store.mark_order_cancelled(position.entry_order_id, "no_entry_candle")
            self.store.cancel_position(position_id, at, "no_entry_candle")


def event_fed_book(store: StateStore, strategy: str, capital: float, now: datetime) -> Book:
    """A `Book` fed every non-cancelled entry and every close, in time order (closes first at
    equal times, as the backtest settles exits before entries), then rolled to `now`."""
    positions = store._positions(("pending", "open", "closed"), strategy, "id")
    events: list[tuple[datetime, int, int, str, float]] = []
    for p in positions:
        events.append((p.placed_at, 1, p.id, p.symbol, p.cost))
        if p.status is PositionStatus.CLOSED:
            assert p.exit_time is not None
            assert p.proceeds is not None
            events.append((p.exit_time, 0, p.id, p.symbol, p.proceeds))
    book = Book(cash=capital)
    for at, kind, _, symbol, amount in sorted(events):
        if kind == 1:
            book.open(symbol, amount, at)
        else:
            book.close(symbol, amount, at)
    book.roll_day(now)
    return book


def assert_books_equal(derived: Book, fed: Book) -> None:
    assert derived.cash == pytest.approx(fed.cash, abs=1e-9)
    assert derived.costs == pytest.approx(fed.costs)
    assert derived.open_cost == pytest.approx(fed.open_cost, abs=1e-9)
    assert derived.day == fed.day
    assert derived.day_start_equity == pytest.approx(fed.day_start_equity, abs=1e-9)
    assert derived.day_pnl == pytest.approx(fed.day_pnl, abs=1e-9)
    assert derived.day_worst_pnl == pytest.approx(fed.day_worst_pnl, abs=1e-9)
    assert derived.day_entries == fed.day_entries


def test_the_derived_book_across_midnight_with_a_cancelled_entry(store: StateStore) -> None:
    ledger = Ledger(store)
    night = datetime(2024, 1, 1, 22, tzinfo=UTC)
    a = ledger.place("AAAUSDT", night, 100.0)
    ledger.fill(a, night)
    b = ledger.place("BBBUSDT", night + timedelta(minutes=30), 100.0)
    ledger.fill(b, night + timedelta(minutes=30))
    ledger.close(a, night + timedelta(hours=1), 95.0)  # -5 on day 1
    c = ledger.place("CCCUSDT", night + timedelta(minutes=90), 100.0)
    ledger.cancel(c, night + timedelta(minutes=95))  # a hole: never counts
    ledger.close(b, night + timedelta(hours=2, minutes=10), 103.0)  # +3 on day 2
    d = ledger.place("DDDUSDT", night + timedelta(hours=2, minutes=20), 100.0)
    ledger.fill(d, night + timedelta(hours=2, minutes=20))
    ledger.close(d, night + timedelta(hours=3), 92.0)  # -8: day 2 at -5, worst -5
    e = ledger.place("EEEUSDT", night + timedelta(hours=4), 100.0)  # still pending
    c2 = ledger.place("CCCUSDT", night + timedelta(hours=4, minutes=5), 50.0)
    ledger.cancel(c2, night + timedelta(hours=4, minutes=10))  # cancelled the same day

    now = night + timedelta(hours=5)
    derived = store.derive_book("alpha", 1_000.0, now)
    assert derived.cash == pytest.approx(1_000 - 5 + 3 - 8 - 100)
    assert derived.costs == {"EEEUSDT": 100.0}
    assert derived.day_start_equity == pytest.approx(995)
    assert derived.day_pnl == pytest.approx(-5)
    assert derived.day_worst_pnl == pytest.approx(-5)
    assert derived.day_entries == 2  # D and E; the cancelled C2 does not count
    assert_books_equal(derived, event_fed_book(store, "alpha", 1_000.0, now))
    assert e in {p.id for p in store.live_positions("alpha")}
    # Other strategies' positions never leak in.
    assert store.derive_book("beta", 500.0, now) == Book(
        cash=500.0, day=derived.day, day_start_equity=500.0
    )


@pytest.mark.parametrize("seed", range(8))
def test_the_derived_book_matches_an_event_fed_book_at_every_step(
    store: StateStore, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    ledger = Ledger(store)
    capital = 1_000.0
    pending: dict[int, str] = {}
    held: dict[int, str] = {}
    now = datetime(2024, 1, 1, 18, tzinfo=UTC)
    for _ in range(300):
        now += timedelta(minutes=int(rng.integers(1, 40)))
        action = rng.random()
        used = set(pending.values()) | set(held.values())
        free = [s for s in ("AUSDT", "BUSDT", "CUSDT", "DUSDT", "EUSDT") if s not in used]
        if action < 0.35 and free:
            symbol = free[int(rng.integers(len(free)))]
            pending[ledger.place(symbol, now, float(rng.uniform(10, 200)))] = symbol
        elif action < 0.6 and pending:
            pid = list(pending)[int(rng.integers(len(pending)))]
            held[pid] = pending.pop(pid)
            ledger.fill(pid, now)
        elif action < 0.7 and pending:
            pid = list(pending)[int(rng.integers(len(pending)))]
            pending.pop(pid)
            ledger.cancel(pid, now)
        elif held:
            pid = list(held)[int(rng.integers(len(held)))]
            held.pop(pid)
            position = store.position(pid)
            assert position is not None
            ledger.close(pid, now, position.cost * float(rng.uniform(0.8, 1.25)))
        derived = store.derive_book("alpha", capital, now)
        assert_books_equal(derived, event_fed_book(store, "alpha", capital, now))


# The paper broker through the store ------------------------------------------------------


class Mirror:
    """What the engine does with the broker: intent before action, then each event."""

    def __init__(self, store: StateStore, broker: PaperBroker) -> None:
        self.store = store
        self.broker = broker

    def buy(self, order: Order) -> None:
        with self.store.transaction():
            self.store.insert_order(OrderRecord.from_order(order, "alpha", Interval.M5))
            assert order.quote_amount is not None
            self.store.insert_position(
                "alpha",
                order.symbol,
                Interval.M5,
                order.time,
                order.id,
                order.quote_amount,
                exit_plan=order.exit_plan,
            )
            self.broker.place_order(order)

    def sell(self, order: Order) -> None:
        with self.store.transaction():
            self.store.insert_order(OrderRecord.from_order(order, "alpha", Interval.M5))
            self.broker.place_order(order)

    def advance(self, until: datetime) -> list[BrokerEvent]:
        with self.store.transaction():
            events = self.broker.advance(until)
            for event in events:
                self.record(event)
        return events

    def record(self, event: BrokerEvent) -> None:
        store = self.store
        if isinstance(event, EntryFilled):
            position = store.position_for_order(event.fill.order_id)
            assert position is not None
            store.mark_order_filled(event.fill.order_id)
            store.record_fill(event.fill)
            paper = event.position
            bracket = paper.bracket()
            if bracket is not None:
                store.insert_order(OrderRecord.from_order(bracket, "alpha", paper.interval))
            store.open_position(
                position.id,
                entry_time=paper.entry_time,
                entry_ref=paper.entry_ref,
                entry_price=paper.entry_price,
                quantity=paper.quantity,
                cost=paper.cost,
                bracket_id=paper.bracket_id,
            )
        elif isinstance(event, ExitFilled):
            position = store.live_position(event.fill.symbol)
            assert position is not None
            store.mark_order_filled(event.fill.order_id)
            store.record_fill(event.fill)
            bracket_id = event.position.bracket_id
            if bracket_id is not None and bracket_id != event.fill.order_id:
                store.mark_order_cancelled(bracket_id, "position_sold")
            reason = "signal" if event.fill.reason is None else event.fill.reason.value
            store.close_position(
                position.id,
                exit_time=event.fill.time,
                exit_price=event.fill.price,
                exit_reason=reason,
                proceeds=event.proceeds,
            )
        else:
            assert isinstance(event, OrderCancelled)
            store.mark_order_cancelled(event.order.id, event.reason.value)
            if event.order.side is Side.BUY:
                position = store.position_for_order(event.order.id)
                assert position is not None
                store.cancel_position(position.id, event.time, event.reason.value)


PATHS: dict[str, list[Bar | None]] = {
    "AAAUSDT": [*flat(3), (100, 100, 93, 94), *flat(30, 94)],  # stop at the close of 3
    "BBBUSDT": [*flat(2), None, *flat(30)],  # entry at 2 is a hole
    "CCCUSDT": [*flat(40, 100)],  # time exit after 3 hours, or sold
    "DDDUSDT": [*flat(5), (100, 125, 99, 120), *flat(30, 120)],  # take profit at 5
}


def drive(mirror: Mirror, start: int, stop: int, *, sell_at: int = 8) -> list[BrokerEvent]:
    events = []
    for k in range(start, stop):
        for symbol, bars in PATHS.items():
            bar = bars[k] if k < len(bars) else None
            if bar is not None:
                mirror.broker.add_candles(symbol, candles([bar], start=T0 + k * M5))
        events.extend(mirror.advance(T0 + (k + 1) * M5))
        if k + 1 == sell_at and "CCCUSDT" in mirror.broker.positions:
            mirror.sell(Order("CCCUSDT", Side.SELL, T0 + sell_at * M5, id="sell-ccc"))
    return events


def test_the_broker_snapshot_from_the_store_resumes_the_broker(tmp_path: Path) -> None:
    costs = Costs()
    capital = 1_000.0

    def start(path: Path) -> Mirror:
        store = StateStore.open(path)
        store.upsert_strategy("alpha", {}, capital)
        mirror = Mirror(store, PaperBroker(cash=capital, costs=costs))
        mirror.buy(Order("AAAUSDT", Side.BUY, T0 + M5, quote_amount=100.0, exit_plan=PLAN, id="a"))
        mirror.buy(
            Order("BBBUSDT", Side.BUY, T0 + 2 * M5, quote_amount=100.0, exit_plan=PLAN, id="b")
        )
        mirror.buy(
            Order("CCCUSDT", Side.BUY, T0 + 2 * M5, quote_amount=100.0, exit_plan=PLAN, id="c")
        )
        mirror.buy(
            Order("DDDUSDT", Side.BUY, T0 + 4 * M5, quote_amount=100.0, exit_plan=PLAN, id="d")
        )
        return mirror

    whole = start(tmp_path / "whole.sqlite")
    whole_events = drive(whole, 0, 40)
    assert [type(e).__name__ for e in whole_events].count("ExitFilled") == 3

    crashed = start(tmp_path / "crashed.sqlite")
    drive(crashed, 0, 3)  # A and C are open, B's entry candle not decided, D pending
    snapshot = crashed.store.broker_snapshot(capital)
    assert snapshot.cash == pytest.approx(crashed.broker.cash)
    assert snapshot.positions == crashed.broker.snapshot().positions
    assert snapshot.orders == crashed.broker.snapshot().orders
    crashed.store.close()

    # Restart: a new store connection and a broker rebuilt from it, candles from each entry.
    store = StateStore.open(tmp_path / "crashed.sqlite")
    broker = PaperBroker.restore(store.broker_snapshot(capital), costs)
    assert {s for s, _, _ in broker.required_history()} == {"AAAUSDT", "CCCUSDT"}
    resumed = Mirror(store, broker)
    for symbol, interval, since in broker.required_history():
        k = (since - T0) // M5
        broker.add_candles(symbol, candles(PATHS[symbol][k:3], start=since, interval=interval))
    drive(resumed, 3, 40)

    def trades(s: StateStore) -> list[tuple[object, ...]]:
        return [
            (p.symbol, p.entry_time, p.exit_time, p.exit_reason, p.proceeds, p.pnl)
            for p in s.closed_positions()
        ]

    assert trades(store) == trades(whole.store)
    assert [f.__class__ for f in store.fills()] == [f.__class__ for f in whole.store.fills()]
    assert [(f.order_id, f.time, f.price, f.quantity, f.fee, f.reason) for f in store.fills()] == [
        (f.order_id, f.time, f.price, f.quantity, f.fee, f.reason) for f in whole.store.fills()
    ]
    assert broker.cash == pytest.approx(whole.broker.cash)
    assert store.broker_snapshot(capital).cash == pytest.approx(broker.cash)
    assert store.derive_book("alpha", capital, T0 + 40 * M5) == whole.store.derive_book(
        "alpha", capital, T0 + 40 * M5
    )
    cancelled = store.position_for_order("b")
    assert cancelled is not None
    assert (cancelled.status, cancelled.exit_reason) == (
        PositionStatus.CANCELLED,
        "no_entry_candle",
    )
    assert store.pending_orders() == []


def test_batch_lookups_and_meta(store: StateStore) -> None:
    store.record_batch("beta", Interval.M5, T0, BatchStatus.MISSED)
    alpha = store.record_batch(
        "alpha", Interval.M5, T0, BatchStatus.ON_TIME, signals=[SignalRow("ABCUSDT")]
    )
    store.record_batch("alpha", Interval.M5, T0 + M5, BatchStatus.ON_TIME)
    assert [b.strategy for b in store.batches_for(Interval.M5, T0)] == ["alpha", "beta"]
    assert store.batches_for(Interval.H1, T0) == []
    assert store.batch_for("alpha", Interval.M5, T0) == alpha
    assert store.batch_for("alpha", Interval.M5, T0 + 2 * M5) is None
    assert store.signal(alpha.signals[0].id) == alpha.signals[0]
    assert store.signal(999) is None

    assert store.meta("initial_cash") is None
    store.set_meta("initial_cash", "100.0")
    store.set_meta("initial_cash", "200.0")
    assert store.meta("initial_cash") == "200.0"
    with pytest.raises(StateError, match="migrations only"):
        store.set_meta("schema_version", "9")
