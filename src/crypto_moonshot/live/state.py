"""The daemon's state file: one SQLite database per run, written only by the daemon.

Everything the engine must not lose lives here: strategies and their capital, the last processed
tick per interval, signal batches and each signal's outcome, orders (written *before* the broker
is called), fills, positions, controls, the event feed and provider heartbeats. The strategy books
are not stored: `derive_book` rebuilds one from its positions, and `broker_snapshot` rebuilds the
paper broker after a restart.

Storage rules
-------------
- WAL journal, `synchronous=FULL` (a committed transaction survives a power cut), foreign keys
  on. The `meta` table holds the schema version; `MIGRATIONS` are applied at open, in one
  transaction, and a file from a newer build (a version beyond the known migrations) is refused.
- Datetimes are stored as fixed-width ISO-8601 UTC text (`2024-01-01T00:00:00.000000Z`, so they
  sort as text) and always come back as aware UTC datetimes. Durations are integer µs. JSON
  columns hold objects.
- `transaction()` makes a block atomic (one tick step = one transaction); nested use becomes a
  savepoint, so an inner failure that is caught rolls back only the inner block. Every write
  method outside a transaction runs in its own.
- Status columns only move forward: an order goes `new` (the intent, written before the broker
  is called) -> `pending` (accepted by the broker) or `unknown` (the broker could not say; it
  is queried) -> filled / cancelled / rejected, each final once; a position goes pending ->
  open -> closed, or pending -> cancelled. A write that would move one backwards raises
  `StateError`.
- One live (pending or open) position per symbol across all strategies, enforced by a unique
  index: the account rule that only one strategy holds a symbol at a time.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Collection, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from crypto_moonshot.broker.base import ExitPlan, Fill, Order, OrderType, Side
from crypto_moonshot.broker.paper import PaperPosition, PaperSnapshot, PendingOrder
from crypto_moonshot.models import Interval, as_utc
from crypto_moonshot.portfolio import Book, utc_day

type JsonObject = dict[str, Any]
type Migration = tuple[str, ...]

_US = timedelta(microseconds=1)
_DAY = timedelta(days=1)


class StateError(Exception):
    """The state file cannot be used, or a write contradicts what it holds."""


class SchemaVersionError(StateError):
    """The file's schema is newer than this build knows."""


# -- schema -------------------------------------------------------------------------------

_V1: Migration = (
    "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    """CREATE TABLE strategies (
        name TEXT PRIMARY KEY,
        config TEXT NOT NULL,
        capital REAL NOT NULL
    )""",
    "CREATE TABLE ticks (interval TEXT PRIMARY KEY, candle_time TEXT NOT NULL)",
    """CREATE TABLE batches (
        id INTEGER PRIMARY KEY,
        strategy TEXT NOT NULL REFERENCES strategies(name),
        interval TEXT NOT NULL,
        candle_time TEXT NOT NULL,
        received_at TEXT,
        lateness_us INTEGER,
        status TEXT NOT NULL CHECK (status IN ('on_time', 'late', 'missed')),
        UNIQUE (strategy, interval, candle_time)
    )""",
    """CREATE TABLE signals (
        id INTEGER PRIMARY KEY,
        batch_id INTEGER NOT NULL REFERENCES batches(id),
        symbol TEXT NOT NULL,
        action TEXT NOT NULL CHECK (action IN ('enter', 'exit')),
        priority REAL NOT NULL,
        weight REAL,
        exit_override TEXT,
        info TEXT NOT NULL,
        outcome TEXT CHECK (outcome IN ('placed', 'rejected', 'expired', 'ignored')),
        reason TEXT
    )""",
    "CREATE INDEX signals_batch ON signals (batch_id)",
    """CREATE TABLE orders (
        id TEXT PRIMARY KEY,
        strategy TEXT NOT NULL REFERENCES strategies(name),
        symbol TEXT NOT NULL,
        interval TEXT NOT NULL,
        side TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
        type TEXT NOT NULL CHECK (type IN ('market', 'bracket')),
        quote_amount REAL,
        quantity REAL,
        exit_plan TEXT,
        time TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('pending', 'filled', 'cancelled', 'rejected')),
        reason TEXT,
        signal_id INTEGER REFERENCES signals(id)
    )""",
    "CREATE INDEX orders_pending ON orders (strategy) WHERE status = 'pending'",
    "CREATE INDEX orders_signal ON orders (signal_id)",
    """CREATE TABLE fills (
        id INTEGER PRIMARY KEY,
        order_id TEXT NOT NULL REFERENCES orders(id),
        time TEXT NOT NULL,
        price REAL NOT NULL,
        quantity REAL NOT NULL,
        fee REAL NOT NULL,
        reason TEXT
    )""",
    "CREATE INDEX fills_order ON fills (order_id)",
    """CREATE TABLE positions (
        id INTEGER PRIMARY KEY,
        strategy TEXT NOT NULL REFERENCES strategies(name),
        symbol TEXT NOT NULL,
        interval TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('pending', 'open', 'closed', 'cancelled')),
        placed_at TEXT NOT NULL,
        entry_order_id TEXT NOT NULL UNIQUE REFERENCES orders(id),
        entry_time TEXT,
        entry_ref REAL,
        entry_price REAL,
        quantity REAL,
        cost REAL NOT NULL,
        exit_plan TEXT,
        bracket_id TEXT REFERENCES orders(id),
        exit_time TEXT,
        exit_price REAL,
        exit_reason TEXT,
        proceeds REAL,
        pnl REAL,
        signal_info TEXT
    )""",
    "CREATE UNIQUE INDEX positions_live_symbol ON positions (symbol)"
    " WHERE status IN ('pending', 'open')",
    "CREATE INDEX positions_strategy ON positions (strategy, status)",
    "CREATE INDEX positions_exit ON positions (strategy, exit_time) WHERE status = 'closed'",
    "CREATE INDEX positions_placed ON positions (strategy, placed_at)",
    "CREATE TABLE controls (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    """CREATE TABLE events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        time TEXT NOT NULL,
        level TEXT NOT NULL CHECK (level IN ('info', 'warning', 'error')),
        kind TEXT NOT NULL,
        message TEXT NOT NULL,
        data TEXT
    )""",
    """CREATE TABLE providers (
        name TEXT PRIMARY KEY,
        last_batch_time TEXT,
        last_seen TEXT NOT NULL
    )""",
)

_V2: Migration = (
    """CREATE TABLE alerts (
        key TEXT PRIMARY KEY,
        rule TEXT NOT NULL,
        level TEXT NOT NULL CHECK (level IN ('info', 'warning', 'error')),
        detail TEXT NOT NULL,
        fired_at TEXT NOT NULL,
        acknowledged_at TEXT
    )""",
    "CREATE INDEX events_kind ON events (kind, id)",
    "CREATE INDEX batches_time ON batches (candle_time)",
)
"""Alerts (the active ones; a row lives from firing to resolution, so an acknowledgement dies
with it) and indexes for the dashboard's queries."""

_ORDER_COLUMNS = (
    "id, strategy, symbol, interval, side, type, quote_amount, quantity, exit_plan, time,"
    " status, reason, signal_id"
)

_V3: Migration = (
    """CREATE TABLE orders_v3 (
        id TEXT PRIMARY KEY,
        strategy TEXT NOT NULL REFERENCES strategies(name),
        symbol TEXT NOT NULL,
        interval TEXT NOT NULL,
        side TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
        type TEXT NOT NULL CHECK (type IN ('market', 'bracket')),
        quote_amount REAL,
        quantity REAL,
        exit_plan TEXT,
        time TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN
            ('new', 'pending', 'unknown', 'filled', 'cancelled', 'rejected')),
        reason TEXT,
        signal_id INTEGER REFERENCES signals(id)
    )""",
    f"INSERT INTO orders_v3 (rowid, {_ORDER_COLUMNS}) SELECT rowid, {_ORDER_COLUMNS} FROM orders",
    "DROP TABLE orders",
    "ALTER TABLE orders_v3 RENAME TO orders",
    "CREATE INDEX orders_pending ON orders (strategy)"
    " WHERE status IN ('new', 'pending', 'unknown')",
    "CREATE INDEX orders_signal ON orders (signal_id)",
)
"""Order states for exchange brokers: `new` (the intent is committed, the broker not yet
called) and `unknown` (the broker could not tell whether it placed the order). SQLite cannot
change a CHECK constraint, so the table is rebuilt (rowids kept; foreign keys are off while
migrations run and checked before they commit)."""

MIGRATIONS: tuple[Migration, ...] = (_V1, _V2, _V3)
"""Migration `k` (0-based) takes a file from schema version `k` to `k + 1`."""
SCHEMA_VERSION = len(MIGRATIONS)


# -- records ------------------------------------------------------------------------------


class BatchStatus(StrEnum):
    ON_TIME = "on_time"
    LATE = "late"
    MISSED = "missed"


class SignalOutcome(StrEnum):
    PLACED = "placed"
    REJECTED = "rejected"
    EXPIRED = "expired"
    IGNORED = "ignored"


class OrderStatus(StrEnum):
    NEW = "new"
    """The intent is committed; the broker has not answered yet (a crash leaves it so)."""
    PENDING = "pending"
    """Accepted by the broker and working (a bracket: live)."""
    UNKNOWN = "unknown"
    """The broker could not tell whether it placed the order: it must be queried."""
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

    @property
    def working(self) -> bool:
        """Not final: new, pending or unknown."""
        return self in WORKING


WORKING = frozenset({OrderStatus.NEW, OrderStatus.PENDING, OrderStatus.UNKNOWN})
"""Order statuses that are not final."""
_WORKING_SQL = "('new', 'pending', 'unknown')"


class PositionStatus(StrEnum):
    PENDING = "pending"
    """The entry order is placed but not filled; its stake is reserved."""
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"
    """The entry never filled (e.g. `no_entry_candle`); it never counts in the book."""


class EventLevel(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class Control(StrEnum):
    PAUSED = "paused"
    FLATTEN_REQUESTED = "flatten_requested"


@dataclass(frozen=True, slots=True)
class StrategyRecord:
    name: str
    config: JsonObject
    capital: float


@dataclass(frozen=True, slots=True)
class SignalRow:
    """A signal to record with its batch; `outcome` may be set now or later."""

    symbol: str
    action: Literal["enter", "exit"] = "enter"
    priority: float = 0.0
    weight: float | None = None
    exit_override: JsonObject | None = None
    info: JsonObject = field(default_factory=dict)
    outcome: SignalOutcome | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class SignalRecord:
    id: int
    batch_id: int
    symbol: str
    action: Literal["enter", "exit"]
    priority: float
    weight: float | None
    exit_override: JsonObject | None
    info: JsonObject
    outcome: SignalOutcome | None
    reason: str | None


@dataclass(frozen=True, slots=True)
class BatchRecord:
    id: int
    strategy: str
    interval: Interval
    candle_time: datetime
    status: BatchStatus
    received_at: datetime | None
    lateness: timedelta | None
    signals: tuple[SignalRecord, ...]


@dataclass(frozen=True, slots=True)
class OrderRecord:
    """An order as the engine intends or placed it. `id` is the client order ID."""

    id: str
    strategy: str
    symbol: str
    interval: Interval
    side: Side
    type: OrderType
    time: datetime
    quote_amount: float | None = None
    quantity: float | None = None
    exit_plan: ExitPlan | None = None
    status: OrderStatus = OrderStatus.PENDING
    reason: str | None = None
    signal_id: int | None = None

    @classmethod
    def from_order(
        cls, order: Order, strategy: str, interval: Interval, signal_id: int | None = None
    ) -> OrderRecord:
        return cls(
            id=order.id,
            strategy=strategy,
            symbol=order.symbol,
            interval=interval,
            side=order.side,
            type=order.type,
            time=order.time,
            quote_amount=order.quote_amount,
            quantity=order.quantity,
            exit_plan=order.exit_plan,
            signal_id=signal_id,
        )

    def to_order(self) -> Order:
        return Order(
            self.symbol,
            self.side,
            self.time,
            self.type,
            quote_amount=self.quote_amount,
            quantity=self.quantity,
            exit_plan=self.exit_plan,
            id=self.id,
        )


@dataclass(frozen=True, slots=True)
class FillRecord:
    id: int
    order_id: str
    time: datetime
    price: float
    quantity: float
    fee: float
    reason: str | None


@dataclass(frozen=True, slots=True)
class PositionRecord:
    """One position's life: placed (pending), open, then closed or cancelled.

    `cost` is the USDT spent including the entry fee (the reserved stake while pending). For a
    cancelled position `exit_time` is when it was cancelled and `exit_reason` why.
    """

    id: int
    strategy: str
    symbol: str
    interval: Interval
    status: PositionStatus
    placed_at: datetime
    entry_order_id: str
    cost: float
    exit_plan: ExitPlan | None = None
    signal_info: JsonObject | None = None
    entry_time: datetime | None = None
    entry_ref: float | None = None
    entry_price: float | None = None
    quantity: float | None = None
    bracket_id: str | None = None
    exit_time: datetime | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    proceeds: float | None = None
    pnl: float | None = None

    def to_paper(self) -> PaperPosition:
        """The open position as the paper broker holds it."""
        if self.status is not PositionStatus.OPEN:
            msg = f"position {self.id} is {self.status}, not open"
            raise StateError(msg)
        assert self.entry_time is not None
        assert self.quantity is not None
        assert self.entry_ref is not None
        assert self.entry_price is not None
        return PaperPosition(
            symbol=self.symbol,
            interval=self.interval,
            quantity=self.quantity,
            entry_time=self.entry_time,
            entry_ref=self.entry_ref,
            entry_price=self.entry_price,
            cost=self.cost,
            entry_order_id=self.entry_order_id,
            exit_plan=self.exit_plan,
            bracket_id=self.bracket_id,
        )


@dataclass(frozen=True, slots=True)
class EventRecord:
    id: int
    time: datetime
    level: EventLevel
    kind: str
    message: str
    data: JsonObject | None


@dataclass(frozen=True, slots=True)
class AlertRecord:
    """An active alert: `key` (`rule` or `rule:subject`), when it fired and whether (and when)
    it was acknowledged."""

    key: str
    rule: str
    level: EventLevel
    detail: str
    fired_at: datetime
    acknowledged_at: datetime | None = None

    @property
    def acknowledged(self) -> bool:
        return self.acknowledged_at is not None


@dataclass(frozen=True, slots=True)
class BatchStats:
    """One strategy's batches over a window: counts per status and lateness (seconds)."""

    strategy: str
    on_time: int
    late: int
    missed: int
    median_lateness: float | None
    worst_lateness: float | None
    last_batch_time: datetime | None
    """Candle time of the latest batch that was received (not missed), in or before the window."""


@dataclass(frozen=True, slots=True)
class SignalListing:
    """A signal with its batch's strategy, interval, candle and status (for the API)."""

    signal: SignalRecord
    strategy: str
    interval: Interval
    candle_time: datetime
    batch_status: BatchStatus


@dataclass(frozen=True, slots=True)
class ProviderRecord:
    name: str
    last_seen: datetime
    last_batch_time: datetime | None


@dataclass(frozen=True, slots=True)
class Controls:
    paused: bool = False
    flatten_requested: bool = False


# -- conversions --------------------------------------------------------------------------


def time_text(dt: datetime) -> str:
    """Fixed-width ISO-8601 UTC text of an aware datetime."""
    return as_utc(dt).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def parse_time(text: str) -> datetime:
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        msg = f"stored datetime without a time zone: {text!r}"
        raise StateError(msg)
    return dt.astimezone(UTC)


def _opt_time(text: str | None) -> datetime | None:
    return None if text is None else parse_time(text)


def _opt_text(dt: datetime | None) -> str | None:
    return None if dt is None else time_text(dt)


def _json(value: JsonObject | None) -> str | None:
    return None if value is None else json.dumps(value, sort_keys=True)


def _unjson(text: str | None) -> JsonObject | None:
    if text is None:
        return None
    value = json.loads(text)
    if not isinstance(value, dict):
        msg = f"expected a JSON object, got {text!r}"
        raise StateError(msg)
    return value


def plan_to_json(plan: ExitPlan) -> JsonObject:
    return {
        "take_profit": plan.take_profit,
        "stop": plan.stop,
        "trailing": plan.trailing,
        "max_hold_us": None if plan.max_hold is None else plan.max_hold // _US,
        "tp_fill": plan.tp_fill,
    }


def plan_from_json(value: JsonObject) -> ExitPlan:
    hold = value.get("max_hold_us")
    return ExitPlan(
        take_profit=value.get("take_profit"),
        stop=value.get("stop"),
        trailing=value.get("trailing"),
        max_hold=None if hold is None else timedelta(microseconds=hold),
        tp_fill=value.get("tp_fill", "touch"),
    )


def _plan_text(plan: ExitPlan | None) -> str | None:
    return None if plan is None else _json(plan_to_json(plan))


def _plan(text: str | None) -> ExitPlan | None:
    value = _unjson(text)
    return None if value is None else plan_from_json(value)


def _signal(row: sqlite3.Row) -> SignalRecord:
    return SignalRecord(
        id=row["id"],
        batch_id=row["batch_id"],
        symbol=row["symbol"],
        action=row["action"],
        priority=row["priority"],
        weight=row["weight"],
        exit_override=_unjson(row["exit_override"]),
        info=_unjson(row["info"]) or {},
        outcome=None if row["outcome"] is None else SignalOutcome(row["outcome"]),
        reason=row["reason"],
    )


def _order(row: sqlite3.Row) -> OrderRecord:
    return OrderRecord(
        id=row["id"],
        strategy=row["strategy"],
        symbol=row["symbol"],
        interval=Interval(row["interval"]),
        side=Side(row["side"]),
        type=OrderType(row["type"]),
        time=parse_time(row["time"]),
        quote_amount=row["quote_amount"],
        quantity=row["quantity"],
        exit_plan=_plan(row["exit_plan"]),
        status=OrderStatus(row["status"]),
        reason=row["reason"],
        signal_id=row["signal_id"],
    )


def _fill(row: sqlite3.Row) -> FillRecord:
    return FillRecord(
        id=row["id"],
        order_id=row["order_id"],
        time=parse_time(row["time"]),
        price=row["price"],
        quantity=row["quantity"],
        fee=row["fee"],
        reason=row["reason"],
    )


def _position(row: sqlite3.Row) -> PositionRecord:
    return PositionRecord(
        id=row["id"],
        strategy=row["strategy"],
        symbol=row["symbol"],
        interval=Interval(row["interval"]),
        status=PositionStatus(row["status"]),
        placed_at=parse_time(row["placed_at"]),
        entry_order_id=row["entry_order_id"],
        cost=row["cost"],
        exit_plan=_plan(row["exit_plan"]),
        signal_info=_unjson(row["signal_info"]),
        entry_time=_opt_time(row["entry_time"]),
        entry_ref=row["entry_ref"],
        entry_price=row["entry_price"],
        quantity=row["quantity"],
        bracket_id=row["bracket_id"],
        exit_time=_opt_time(row["exit_time"]),
        exit_price=row["exit_price"],
        exit_reason=row["exit_reason"],
        proceeds=row["proceeds"],
        pnl=row["pnl"],
    )


def _alert(row: sqlite3.Row) -> AlertRecord:
    return AlertRecord(
        key=row["key"],
        rule=row["rule"],
        level=EventLevel(row["level"]),
        detail=row["detail"],
        fired_at=parse_time(row["fired_at"]),
        acknowledged_at=_opt_time(row["acknowledged_at"]),
    )


def _event(row: sqlite3.Row) -> EventRecord:
    return EventRecord(
        id=row["id"],
        time=parse_time(row["time"]),
        level=EventLevel(row["level"]),
        kind=row["kind"],
        message=row["message"],
        data=_unjson(row["data"]),
    )


# -- the store ----------------------------------------------------------------------------


class StateStore:
    """A run's state file. Open with `StateStore.open(path)`; see the module docstring."""

    def __init__(self, connection: sqlite3.Connection, path: Path) -> None:
        self._db = connection
        self.path = path
        self._depth = 0

    @classmethod
    def open(cls, path: Path | str, *, migrations: Sequence[Migration] = MIGRATIONS) -> StateStore:
        """Open (creating if needed) the state file at `path` and bring its schema up to date.

        Raises `SchemaVersionError` for a file written by a newer build.
        """
        path = Path(path)
        db = sqlite3.connect(path, autocommit=True)
        try:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA journal_mode = WAL")
            db.execute("PRAGMA synchronous = FULL")
            db.execute("PRAGMA foreign_keys = ON")
            store = cls(db, path)
            store._migrate(migrations)
        except BaseException:
            db.close()
            raise
        return store

    @classmethod
    def open_readonly(
        cls, path: Path | str, *, migrations: Sequence[Migration] = MIGRATIONS
    ) -> StateStore:
        """Open an existing state file for reading only (`live status` with the daemon down):
        nothing is created, migrated or written. Raises `StateError` for a file that is not a
        state file of this build's schema (older or newer)."""
        path = Path(path)
        if not path.is_file():
            msg = f"no state file at {path}"
            raise StateError(msg)
        db = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, autocommit=True)
        try:
            db.row_factory = sqlite3.Row
            store = cls(db, path)
            version = store.schema_version
            if version != len(migrations):
                msg = f"{path} has schema version {version}; this build reads {len(migrations)}"
                raise SchemaVersionError(msg)
        except sqlite3.DatabaseError as exc:
            db.close()
            msg = f"{path} is not a readable state file: {exc}"
            raise StateError(msg) from None
        except BaseException:
            db.close()
            raise
        return store

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> StateStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def schema_version(self) -> int:
        exists = self._db.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'meta'"
        ).fetchone()
        if exists is None:
            return 0
        row = self._db.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        return 0 if row is None else int(row["value"])

    def _migrate(self, migrations: Sequence[Migration]) -> None:
        version = self.schema_version
        if version > len(migrations):
            msg = (
                f"{self.path} has schema version {version}, newer than this build's "
                f"{len(migrations)}; upgrade crypto-moonshot to open it"
            )
            raise SchemaVersionError(msg)
        if version == len(migrations):
            return
        # Foreign keys are off while migrations run (a table rebuild drops the old table), and
        # checked before the migrations commit; PRAGMA foreign_keys cannot change in a
        # transaction, so it is switched around it.
        self._db.execute("PRAGMA foreign_keys = OFF")
        try:
            with self.transaction():
                for migration in migrations[version:]:
                    for statement in migration:
                        self._db.execute(statement)
                broken = self._db.execute("PRAGMA foreign_key_check").fetchall()
                if broken:
                    msg = f"migrating {self.path} broke {len(broken)} foreign key(s)"
                    raise StateError(msg)
                self._db.execute(
                    "INSERT INTO meta (key, value) VALUES ('schema_version', ?)"
                    " ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                    (str(len(migrations)),),
                )
        finally:
            self._db.execute("PRAGMA foreign_keys = ON")

    # -- transactions ---------------------------------------------------------------------

    @property
    def in_transaction(self) -> bool:
        return self._depth > 0

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Make the block atomic: commit at its end, roll everything back on an exception.

        Nested blocks are savepoints: an exception that leaves one rolls back that block only
        (and, unless caught, the outer one too).
        """
        if self._depth == 0:
            self._db.execute("BEGIN IMMEDIATE")
            self._depth = 1
            try:
                yield
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
            else:
                try:
                    self._db.execute("COMMIT")
                except BaseException:
                    self._db.execute("ROLLBACK")
                    raise
            finally:
                self._depth = 0
            return
        name = f"sp{self._depth}"
        self._db.execute(f"SAVEPOINT {name}")
        self._depth += 1
        try:
            yield
        except BaseException:
            self._db.execute(f"ROLLBACK TO {name}")
            self._db.execute(f"RELEASE {name}")
            raise
        else:
            self._db.execute(f"RELEASE {name}")
        finally:
            self._depth -= 1

    @contextmanager
    def _atomic(self) -> Iterator[None]:
        if self._depth:
            yield
        else:
            with self.transaction():
                yield

    def _exec(self, sql: str, params: Sequence[object] = ()) -> sqlite3.Cursor:
        return self._db.execute(sql, params)

    # -- strategies, ticks, controls, providers --------------------------------------------

    def upsert_strategy(self, name: str, config: JsonObject, capital: float) -> None:
        with self._atomic():
            self._exec(
                "INSERT INTO strategies (name, config, capital) VALUES (?, ?, ?)"
                " ON CONFLICT (name) DO UPDATE SET config = excluded.config,"
                " capital = excluded.capital",
                (name, _json(config), capital),
            )

    def strategies(self) -> list[StrategyRecord]:
        rows = self._exec("SELECT * FROM strategies ORDER BY name").fetchall()
        return [StrategyRecord(r["name"], _unjson(r["config"]) or {}, r["capital"]) for r in rows]

    def strategy(self, name: str) -> StrategyRecord | None:
        row = self._exec("SELECT * FROM strategies WHERE name = ?", (name,)).fetchone()
        return (
            None
            if row is None
            else StrategyRecord(name, _unjson(row["config"]) or {}, row["capital"])
        )

    def meta(self, key: str) -> str | None:
        """A value from the `meta` table (the schema version lives there too)."""
        row = self._exec("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def set_meta(self, key: str, value: str) -> None:
        if key == "schema_version":
            msg = "the schema version is set by migrations only"
            raise StateError(msg)
        with self._atomic():
            self._exec(
                "INSERT INTO meta (key, value) VALUES (?, ?)"
                " ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def set_last_tick(self, interval: Interval, candle_time: datetime) -> None:
        with self._atomic():
            self._exec(
                "INSERT INTO ticks (interval, candle_time) VALUES (?, ?)"
                " ON CONFLICT (interval) DO UPDATE SET candle_time = excluded.candle_time",
                (interval.value, time_text(candle_time)),
            )

    def last_tick(self, interval: Interval) -> datetime | None:
        row = self._exec(
            "SELECT candle_time FROM ticks WHERE interval = ?", (interval.value,)
        ).fetchone()
        return None if row is None else parse_time(row["candle_time"])

    def last_ticks(self) -> dict[Interval, datetime]:
        rows = self._exec("SELECT * FROM ticks").fetchall()
        return {Interval(r["interval"]): parse_time(r["candle_time"]) for r in rows}

    def controls(self) -> Controls:
        rows = self._exec("SELECT key, value FROM controls").fetchall()
        values = {r["key"]: json.loads(r["value"]) for r in rows}
        return Controls(
            paused=bool(values.get(Control.PAUSED, False)),
            flatten_requested=bool(values.get(Control.FLATTEN_REQUESTED, False)),
        )

    def set_control(self, control: Control, value: bool) -> None:
        with self._atomic():
            self._exec(
                "INSERT INTO controls (key, value) VALUES (?, ?)"
                " ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                (control.value, json.dumps(value)),
            )

    def provider_seen(
        self, name: str, seen_at: datetime, batch_time: datetime | None = None
    ) -> None:
        """Record a provider's heartbeat; `batch_time` (the candle of its latest batch) is kept
        when not given."""
        with self._atomic():
            self._exec(
                "INSERT INTO providers (name, last_batch_time, last_seen) VALUES (?, ?, ?)"
                " ON CONFLICT (name) DO UPDATE SET last_seen = excluded.last_seen,"
                " last_batch_time = coalesce(excluded.last_batch_time, last_batch_time)",
                (name, _opt_text(batch_time), time_text(seen_at)),
            )

    def providers(self) -> list[ProviderRecord]:
        rows = self._exec("SELECT * FROM providers ORDER BY name").fetchall()
        return [
            ProviderRecord(r["name"], parse_time(r["last_seen"]), _opt_time(r["last_batch_time"]))
            for r in rows
        ]

    # -- events ---------------------------------------------------------------------------

    def add_event(
        self,
        time: datetime,
        level: EventLevel,
        kind: str,
        message: str,
        data: JsonObject | None = None,
    ) -> int:
        with self._atomic():
            cursor = self._exec(
                "INSERT INTO events (time, level, kind, message, data) VALUES (?, ?, ?, ?, ?)",
                (time_text(time), level.value, kind, message, _json(data)),
            )
        assert cursor.lastrowid is not None
        return cursor.lastrowid

    def events_since(self, after_id: int = 0, limit: int | None = None) -> list[EventRecord]:
        """Events with an id above `after_id`, oldest first (ids only grow)."""
        rows = self._exec(
            "SELECT * FROM events WHERE id > ? ORDER BY id LIMIT ?",
            (after_id, -1 if limit is None else limit),
        ).fetchall()
        return [_event(r) for r in rows]

    def recent_events(self, limit: int) -> list[EventRecord]:
        """The latest `limit` events, oldest first."""
        rows = self._exec(
            "SELECT * FROM (SELECT * FROM events ORDER BY id DESC LIMIT ?) ORDER BY id", (limit,)
        ).fetchall()
        return [_event(r) for r in rows]

    def recent_events_of_kind(self, kind: str, limit: int) -> list[EventRecord]:
        """The latest `limit` events of `kind`, newest first."""
        rows = self._exec(
            "SELECT * FROM events WHERE kind = ? ORDER BY id DESC LIMIT ?", (kind, limit)
        ).fetchall()
        return [_event(r) for r in rows]

    def count_events(self, kinds: Sequence[str], since: datetime) -> int:
        """How many events of these kinds happened at or after `since`."""
        marks = ", ".join("?" for _ in kinds)
        row = self._exec(
            f"SELECT count(*) AS n FROM events WHERE kind IN ({marks}) AND time >= ?",
            (*kinds, time_text(since)),
        ).fetchone()
        return int(row["n"])

    # -- alerts ---------------------------------------------------------------------------

    def alerts(self) -> list[AlertRecord]:
        """The active alerts, oldest first."""
        rows = self._exec("SELECT * FROM alerts ORDER BY fired_at, key").fetchall()
        return [_alert(r) for r in rows]

    def alert(self, key: str) -> AlertRecord | None:
        row = self._exec("SELECT * FROM alerts WHERE key = ?", (key,)).fetchone()
        return None if row is None else _alert(row)

    def fire_alert(
        self, key: str, rule: str, level: EventLevel, detail: str, fired_at: datetime
    ) -> None:
        """Make `key` active (unacknowledged). A key that is already active is left as it is."""
        with self._atomic():
            self._exec(
                "INSERT OR IGNORE INTO alerts (key, rule, level, detail, fired_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (key, rule, level.value, detail, time_text(fired_at)),
            )

    def resolve_alert(self, key: str) -> bool:
        """Make `key` inactive (its acknowledgement goes with it); whether it was active."""
        with self._atomic():
            return self._exec("DELETE FROM alerts WHERE key = ?", (key,)).rowcount == 1

    def acknowledge_alert(self, key: str, at: datetime) -> bool:
        """Acknowledge an active alert (the first time counts); false if `key` is not active."""
        with self._atomic():
            cursor = self._exec(
                "UPDATE alerts SET acknowledged_at = coalesce(acknowledged_at, ?) WHERE key = ?",
                (time_text(at), key),
            )
        return cursor.rowcount == 1

    # -- batches and signals --------------------------------------------------------------

    def record_batch(
        self,
        strategy: str,
        interval: Interval,
        candle_time: datetime,
        status: BatchStatus,
        *,
        received_at: datetime | None = None,
        lateness: timedelta | None = None,
        signals: Sequence[SignalRow] = (),
    ) -> BatchRecord:
        """Record one strategy's batch for a candle (a `missed` one has no `received_at`)."""
        with self._atomic():
            cursor = self._exec(
                "INSERT INTO batches (strategy, interval, candle_time, received_at, lateness_us,"
                " status) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    strategy,
                    interval.value,
                    time_text(candle_time),
                    _opt_text(received_at),
                    None if lateness is None else lateness // _US,
                    status.value,
                ),
            )
            batch_id = cursor.lastrowid
            assert batch_id is not None
            self._insert_signals(batch_id, signals)
        batch = self.batch(batch_id)
        assert batch is not None
        return batch

    def _insert_signals(self, batch_id: int, signals: Sequence[SignalRow]) -> None:
        for signal in signals:
            self._exec(
                "INSERT INTO signals (batch_id, symbol, action, priority, weight,"
                " exit_override, info, outcome, reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    batch_id,
                    signal.symbol,
                    signal.action,
                    signal.priority,
                    signal.weight,
                    _json(signal.exit_override),
                    _json(signal.info),
                    None if signal.outcome is None else signal.outcome.value,
                    signal.reason,
                ),
            )

    def fill_missed_batch(
        self,
        batch_id: int,
        *,
        received_at: datetime,
        lateness: timedelta,
        signals: Sequence[SignalRow] = (),
    ) -> BatchRecord:
        """A batch recorded as missed arrived after all: missed -> late, with its signals."""
        with self._atomic():
            cursor = self._exec(
                "UPDATE batches SET status = 'late', received_at = ?, lateness_us = ?"
                " WHERE id = ? AND status = 'missed'",
                (time_text(received_at), lateness // _US, batch_id),
            )
            if cursor.rowcount != 1:
                current = self.batch(batch_id)
                if current is None:
                    raise KeyError(batch_id)
                msg = f"batch {batch_id} is {current.status}, not missed"
                raise StateError(msg)
            self._insert_signals(batch_id, signals)
        batch = self.batch(batch_id)
        assert batch is not None
        return batch

    def batch_for(
        self, strategy: str, interval: Interval, candle_time: datetime
    ) -> BatchRecord | None:
        """`strategy`'s batch for the candle, if one was recorded."""
        row = self._exec(
            "SELECT id FROM batches WHERE strategy = ? AND interval = ? AND candle_time = ?",
            (strategy, interval.value, time_text(candle_time)),
        ).fetchone()
        return None if row is None else self.batch(row["id"])

    def batches_for(self, interval: Interval, candle_time: datetime) -> list[BatchRecord]:
        """Every strategy's batch recorded for the candle, by strategy name."""
        rows = self._exec(
            "SELECT id FROM batches WHERE interval = ? AND candle_time = ? ORDER BY strategy",
            (interval.value, time_text(candle_time)),
        ).fetchall()
        out = []
        for row in rows:
            batch = self.batch(row["id"])
            assert batch is not None
            out.append(batch)
        return out

    def recent_batch_statuses(self, strategy: str, limit: int) -> list[BatchStatus]:
        """The statuses of `strategy`'s latest `limit` batches, newest candle first."""
        rows = self._exec(
            "SELECT status FROM batches WHERE strategy = ? ORDER BY candle_time DESC LIMIT ?",
            (strategy, limit),
        ).fetchall()
        return [BatchStatus(r["status"]) for r in rows]

    def batch_stats(self, since: datetime) -> list[BatchStats]:
        """Per strategy, the batches whose candle is at or after `since`: counts per status,
        the median and worst lateness (late and on-time batches), and the latest received
        batch's candle time (looked up beyond the window too)."""
        rows = self._exec(
            "SELECT strategy, status, lateness_us FROM batches WHERE candle_time >= ?"
            " ORDER BY strategy",
            (time_text(since),),
        ).fetchall()
        counts: dict[str, dict[str, int]] = {}
        lateness: dict[str, list[float]] = {}
        for row in rows:
            per = counts.setdefault(row["strategy"], {"on_time": 0, "late": 0, "missed": 0})
            per[row["status"]] += 1
            if row["lateness_us"] is not None:
                lateness.setdefault(row["strategy"], []).append(row["lateness_us"] / 1e6)
        out = []
        for record in self.strategies():
            per = counts.get(record.name, {"on_time": 0, "late": 0, "missed": 0})
            values = sorted(lateness.get(record.name, []))
            mid = len(values) // 2
            median = (
                None
                if not values
                else values[mid]
                if len(values) % 2
                else (values[mid - 1] + values[mid]) / 2
            )
            last = self._exec(
                "SELECT max(candle_time) AS t FROM batches WHERE strategy = ?"
                " AND status != 'missed'",
                (record.name,),
            ).fetchone()["t"]
            out.append(
                BatchStats(
                    strategy=record.name,
                    on_time=per["on_time"],
                    late=per["late"],
                    missed=per["missed"],
                    median_lateness=median,
                    worst_lateness=values[-1] if values else None,
                    last_batch_time=_opt_time(last),
                )
            )
        return out

    def recent_signals(self, limit: int, strategy: str | None = None) -> list[SignalListing]:
        """The latest `limit` signals (newest first), optionally of one strategy."""
        sql = (
            "SELECT s.*, b.strategy AS b_strategy, b.interval AS b_interval,"
            " b.candle_time AS b_candle_time, b.status AS b_status"
            " FROM signals s JOIN batches b ON b.id = s.batch_id"
        )
        params: list[object] = []
        if strategy is not None:
            sql += " WHERE b.strategy = ?"
            params.append(strategy)
        rows = self._exec(sql + " ORDER BY s.id DESC LIMIT ?", [*params, limit]).fetchall()
        return [
            SignalListing(
                signal=_signal(r),
                strategy=r["b_strategy"],
                interval=Interval(r["b_interval"]),
                candle_time=parse_time(r["b_candle_time"]),
                batch_status=BatchStatus(r["b_status"]),
            )
            for r in rows
        ]

    def signal(self, signal_id: int) -> SignalRecord | None:
        row = self._exec("SELECT * FROM signals WHERE id = ?", (signal_id,)).fetchone()
        return None if row is None else _signal(row)

    def signal_outcomes(self) -> dict[str, int]:
        """How many signals ended how: `outcome` or `outcome:reason` (`pending` for none yet)."""
        rows = self._exec(
            "SELECT outcome, reason, count(*) AS n FROM signals GROUP BY outcome, reason"
            " ORDER BY outcome, reason"
        ).fetchall()
        out: dict[str, int] = {}
        for row in rows:
            outcome = row["outcome"] or "pending"
            key = outcome if row["reason"] is None else f"{outcome}:{row['reason']}"
            out[key] = int(row["n"])
        return out

    def set_signal_outcome(
        self, signal_id: int, outcome: SignalOutcome, reason: str | None = None
    ) -> None:
        with self._atomic():
            cursor = self._exec(
                "UPDATE signals SET outcome = ?, reason = ? WHERE id = ?",
                (outcome.value, reason, signal_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(signal_id)

    def batch(self, batch_id: int) -> BatchRecord | None:
        row = self._exec("SELECT * FROM batches WHERE id = ?", (batch_id,)).fetchone()
        if row is None:
            return None
        signals = self._exec(
            "SELECT * FROM signals WHERE batch_id = ? ORDER BY id", (batch_id,)
        ).fetchall()
        return BatchRecord(
            id=row["id"],
            strategy=row["strategy"],
            interval=Interval(row["interval"]),
            candle_time=parse_time(row["candle_time"]),
            status=BatchStatus(row["status"]),
            received_at=_opt_time(row["received_at"]),
            lateness=None
            if row["lateness_us"] is None
            else timedelta(microseconds=row["lateness_us"]),
            signals=tuple(_signal(s) for s in signals),
        )

    # -- orders and fills -----------------------------------------------------------------

    def insert_order(self, order: OrderRecord) -> None:
        """Record an order before it goes to the broker (intent before action): `new` for an
        intent, `pending` for an order the broker already holds (a bracket opened by a fill)."""
        if order.status not in (OrderStatus.NEW, OrderStatus.PENDING):
            msg = f"a new order must be pending or new, got {order.status}"
            raise StateError(msg)
        with self._atomic():
            self._exec(
                "INSERT INTO orders (id, strategy, symbol, interval, side, type, quote_amount,"
                " quantity, exit_plan, time, status, reason, signal_id)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    order.id,
                    order.strategy,
                    order.symbol,
                    order.interval.value,
                    order.side.value,
                    order.type.value,
                    order.quote_amount,
                    order.quantity,
                    _plan_text(order.exit_plan),
                    time_text(order.time),
                    order.status.value,
                    order.reason,
                    order.signal_id,
                ),
            )

    def _finish_order(self, order_id: str, status: OrderStatus, reason: str | None) -> None:
        with self._atomic():
            cursor = self._exec(
                "UPDATE orders SET status = ?, reason = ?"
                f" WHERE id = ? AND status IN {_WORKING_SQL}",
                (status.value, reason, order_id),
            )
            if cursor.rowcount == 1:
                return
            current = self.order(order_id)
            if current is None:
                raise KeyError(order_id)
            msg = f"order {order_id} is {current.status}, not pending"
            raise StateError(msg)

    def mark_order_accepted(self, order_id: str) -> None:
        """The broker holds the order: new or unknown -> pending (pending stays pending)."""
        with self._atomic():
            cursor = self._exec(
                "UPDATE orders SET status = 'pending'"
                " WHERE id = ? AND status IN ('new', 'unknown')",
                (order_id,),
            )
            if cursor.rowcount == 1:
                return
            current = self.order(order_id)
            if current is None:
                raise KeyError(order_id)
            if current.status is not OrderStatus.PENDING:
                msg = f"order {order_id} is {current.status}, not new or unknown"
                raise StateError(msg)

    def mark_order_unknown(self, order_id: str) -> None:
        """The broker could not tell whether it placed the order: new -> unknown."""
        with self._atomic():
            cursor = self._exec(
                "UPDATE orders SET status = 'unknown' WHERE id = ? AND status = 'new'",
                (order_id,),
            )
            if cursor.rowcount == 1:
                return
            current = self.order(order_id)
            if current is None:
                raise KeyError(order_id)
            if current.status is not OrderStatus.UNKNOWN:
                msg = f"order {order_id} is {current.status}, not new"
                raise StateError(msg)

    def mark_order_filled(self, order_id: str) -> None:
        self._finish_order(order_id, OrderStatus.FILLED, None)

    def mark_order_cancelled(self, order_id: str, reason: str) -> None:
        self._finish_order(order_id, OrderStatus.CANCELLED, reason)

    def mark_order_rejected(self, order_id: str, reason: str) -> None:
        self._finish_order(order_id, OrderStatus.REJECTED, reason)

    def order(self, order_id: str) -> OrderRecord | None:
        row = self._exec("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone()
        return None if row is None else _order(row)

    def pending_orders(self, strategy: str | None = None) -> list[OrderRecord]:
        """Orders not yet filled, cancelled or rejected (new, pending or unknown; brackets
        included), oldest first."""
        sql = f"SELECT * FROM orders WHERE status IN {_WORKING_SQL}"
        params: tuple[str, ...] = ()
        if strategy is not None:
            sql += " AND strategy = ?"
            params = (strategy,)
        rows = self._exec(sql + " ORDER BY time, rowid", params).fetchall()
        return [_order(r) for r in rows]

    def record_fill(self, fill: Fill) -> FillRecord:
        with self._atomic():
            cursor = self._exec(
                "INSERT INTO fills (order_id, time, price, quantity, fee, reason)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    fill.order_id,
                    time_text(fill.time),
                    fill.price,
                    fill.quantity,
                    fill.fee,
                    None if fill.reason is None else fill.reason.value,
                ),
            )
        assert cursor.lastrowid is not None
        row = self._exec("SELECT * FROM fills WHERE id = ?", (cursor.lastrowid,)).fetchone()
        return _fill(row)

    def fills(self, order_id: str | None = None) -> list[FillRecord]:
        if order_id is None:
            rows = self._exec("SELECT * FROM fills ORDER BY time, id").fetchall()
        else:
            rows = self._exec(
                "SELECT * FROM fills WHERE order_id = ? ORDER BY time, id", (order_id,)
            ).fetchall()
        return [_fill(r) for r in rows]

    # -- positions ------------------------------------------------------------------------

    def insert_position(
        self,
        strategy: str,
        symbol: str,
        interval: Interval,
        placed_at: datetime,
        entry_order_id: str,
        cost: float,
        *,
        exit_plan: ExitPlan | None = None,
        signal_info: JsonObject | None = None,
    ) -> PositionRecord:
        """A pending position for an entry order (insert the order first): `cost` is the stake
        reserved, `placed_at` the time the entry is for (the open of *t+1*), which is the time
        the backtest's book counts the entry at."""
        with self._atomic():
            cursor = self._exec(
                "INSERT INTO positions (strategy, symbol, interval, status, placed_at,"
                " entry_order_id, cost, exit_plan, signal_info)"
                " VALUES (?, ?, ?, 'pending', ?, ?, ?, ?, ?)",
                (
                    strategy,
                    symbol,
                    interval.value,
                    time_text(placed_at),
                    entry_order_id,
                    cost,
                    _plan_text(exit_plan),
                    _json(signal_info),
                ),
            )
        assert cursor.lastrowid is not None
        position = self.position(cursor.lastrowid)
        assert position is not None
        return position

    def _move_position(
        self, position_id: int, expected: PositionStatus, sql: str, params: Sequence[object]
    ) -> None:
        with self._atomic():
            cursor = self._exec(
                f"UPDATE positions SET {sql} WHERE id = ? AND status = ?",
                (*params, position_id, expected.value),
            )
            if cursor.rowcount == 1:
                return
            current = self.position(position_id)
            if current is None:
                raise KeyError(position_id)
            msg = f"position {position_id} is {current.status}, not {expected}"
            raise StateError(msg)

    def open_position(
        self,
        position_id: int,
        *,
        entry_time: datetime,
        entry_ref: float,
        entry_price: float,
        quantity: float,
        cost: float,
        bracket_id: str | None = None,
    ) -> None:
        """The entry filled: pending -> open."""
        self._move_position(
            position_id,
            PositionStatus.PENDING,
            "status = 'open', entry_time = ?, entry_ref = ?, entry_price = ?, quantity = ?,"
            " cost = ?, bracket_id = ?",
            (time_text(entry_time), entry_ref, entry_price, quantity, cost, bracket_id),
        )

    def set_position_bracket(self, position_id: int, bracket_id: str | None) -> None:
        """Change an open position's bracket (e.g. `None` once cancelled)."""
        self._move_position(position_id, PositionStatus.OPEN, "bracket_id = ?", (bracket_id,))

    def close_position(
        self,
        position_id: int,
        *,
        exit_time: datetime,
        exit_price: float,
        exit_reason: str,
        proceeds: float,
    ) -> PositionRecord:
        """The position was sold: open -> closed, `pnl = proceeds - cost` (`proceeds` after the
        exit fee)."""
        self._move_position(
            position_id,
            PositionStatus.OPEN,
            "status = 'closed', exit_time = ?, exit_price = ?, exit_reason = ?, proceeds = ?,"
            " pnl = ? - cost",
            (time_text(exit_time), exit_price, exit_reason, proceeds, proceeds),
        )
        position = self.position(position_id)
        assert position is not None
        return position

    def cancel_position(self, position_id: int, time: datetime, reason: str) -> None:
        """The entry never filled: pending -> cancelled; its stake is free again."""
        self._move_position(
            position_id,
            PositionStatus.PENDING,
            "status = 'cancelled', exit_time = ?, exit_reason = ?",
            (time_text(time), reason),
        )

    def position(self, position_id: int) -> PositionRecord | None:
        row = self._exec("SELECT * FROM positions WHERE id = ?", (position_id,)).fetchone()
        return None if row is None else _position(row)

    def position_for_order(self, order_id: str) -> PositionRecord | None:
        """The position an order belongs to, as its entry or its bracket."""
        row = self._exec(
            "SELECT * FROM positions WHERE entry_order_id = ? OR bracket_id = ?"
            " ORDER BY id DESC LIMIT 1",
            (order_id, order_id),
        ).fetchone()
        return None if row is None else _position(row)

    def live_position(self, symbol: str) -> PositionRecord | None:
        """The pending or open position in `symbol` (at most one across strategies)."""
        row = self._exec(
            "SELECT * FROM positions WHERE symbol = ? AND status IN ('pending', 'open')",
            (symbol,),
        ).fetchone()
        return None if row is None else _position(row)

    def live_positions(self, strategy: str | None = None) -> list[PositionRecord]:
        """Pending and open positions, in placement order."""
        return self._positions(("pending", "open"), strategy, "placed_at, id")

    def closed_positions(self, strategy: str | None = None) -> list[PositionRecord]:
        """Closed positions (the trade list), in exit order."""
        return self._positions(("closed",), strategy, "exit_time, id")

    def recent_closed_positions(
        self, limit: int, strategy: str | None = None
    ) -> list[PositionRecord]:
        """The latest `limit` closed positions, newest exit first."""
        sql = "SELECT * FROM positions WHERE status = 'closed'"
        params: list[object] = []
        if strategy is not None:
            sql += " AND strategy = ?"
            params.append(strategy)
        rows = self._exec(sql + " ORDER BY exit_time DESC, id DESC LIMIT ?", [*params, limit])
        return [_position(r) for r in rows.fetchall()]

    def equity_points(
        self, strategy: str, capital: float, limit: int
    ) -> list[tuple[datetime, float]]:
        """`strategy`'s realised equity after each of its latest `limit` closed positions
        (`capital` plus the cumulative P&L in exit order), oldest first."""
        rows = self._exec(
            "SELECT exit_time, equity FROM (SELECT exit_time, id,"
            " ? + sum(pnl) OVER (ORDER BY exit_time, id) AS equity FROM positions"
            " WHERE strategy = ? AND status = 'closed') ORDER BY exit_time DESC, id DESC LIMIT ?",
            (capital, strategy, limit),
        ).fetchall()
        return [(parse_time(r["exit_time"]), float(r["equity"])) for r in reversed(rows)]

    def realised_pnl(self, strategy: str) -> float:
        """The sum of `strategy`'s closed positions' P&L."""
        row = self._exec(
            "SELECT coalesce(sum(pnl), 0.0) AS pnl FROM positions"
            " WHERE strategy = ? AND status = 'closed'",
            (strategy,),
        ).fetchone()
        return float(row["pnl"])

    def _positions(
        self, statuses: tuple[str, ...], strategy: str | None, order_by: str
    ) -> list[PositionRecord]:
        marks = ", ".join("?" for _ in statuses)
        sql = f"SELECT * FROM positions WHERE status IN ({marks})"
        params: list[object] = list(statuses)
        if strategy is not None:
            sql += " AND strategy = ?"
            params.append(strategy)
        rows = self._exec(f"{sql} ORDER BY {order_by}", params).fetchall()
        return [_position(r) for r in rows]

    # -- derived state --------------------------------------------------------------------

    def derive_book(
        self,
        strategy: str,
        capital: float,
        now: datetime,
        *,
        exclude: Collection[int] = (),
    ) -> Book:
        """`strategy`'s book rebuilt from its positions, as of the UTC day of `now`.

        - `cash = capital + sum(pnl of closed) - sum(cost of open and pending)`; `costs` the
          open and pending positions' costs (a pending entry's stake is reserved, as the
          backtest spends it at entry).
        - `day_start_equity = capital + sum(pnl of positions closed before today)`; `day_pnl`
          the P&L of today's closes; `day_worst_pnl` the lowest running total of today's P&L
          in exit order (exit time, then position id), starting at 0.
        - `day_entries`: positions placed today (`placed_at`) that were not cancelled. A
          cancelled entry never counts, as the backtest never counts a skipped signal.

        This equals a `Book` fed every non-cancelled entry (at `placed_at`) and every close (at
        `exit_time`) in time order, then `roll_day(now)`.

        `exclude`: ids of pending or open positions to leave out, as if not placed yet (the
        engine reapplies a resumed tick's entries one by one, exactly as it first did).
        """
        day = utc_day(now)
        skip = ""
        if exclude:
            skip = " AND id NOT IN (" + ", ".join(str(int(i)) for i in exclude) + ")"
        midnight = as_utc(now).replace(hour=0, minute=0, second=0, microsecond=0)
        start, end = time_text(midnight), time_text(midnight + _DAY)
        db = self._db
        live = db.execute(
            "SELECT symbol, cost FROM positions WHERE strategy = ?"
            f" AND status IN ('pending', 'open'){skip} ORDER BY placed_at, id",
            (strategy,),
        ).fetchall()
        before = db.execute(
            "SELECT coalesce(sum(pnl), 0.0) AS pnl FROM positions WHERE strategy = ?"
            " AND status = 'closed' AND exit_time < ?",
            (strategy, start),
        ).fetchone()["pnl"]
        later = db.execute(
            "SELECT pnl, exit_time < ? AS today FROM positions WHERE strategy = ?"
            " AND status = 'closed' AND exit_time >= ? ORDER BY exit_time, id",
            (end, strategy, start),
        ).fetchall()
        entries = db.execute(
            "SELECT count(*) AS n FROM positions WHERE strategy = ? AND status != 'cancelled'"
            f" AND placed_at >= ? AND placed_at < ?{skip}",
            (strategy, start, end),
        ).fetchone()["n"]
        costs = {r["symbol"]: float(r["cost"]) for r in live}
        open_cost = sum(costs.values())
        day_pnl = worst = after = 0.0
        for row in later:
            if row["today"]:
                day_pnl += row["pnl"]
                worst = min(worst, day_pnl)
            else:  # closed after `now`'s day: only the cash sees it
                after += row["pnl"]
        return Book(
            cash=capital + before + day_pnl + after - open_cost,
            costs=costs,
            open_cost=open_cost,
            day=day,
            day_start_equity=capital + before,
            day_pnl=day_pnl,
            day_worst_pnl=worst,
            day_entries=entries,
        )

    def broker_snapshot(self, initial_cash: float) -> PaperSnapshot:
        """The paper broker's state from the file, for a restart: open positions with their
        brackets, working market orders (an intent the crash left `new` included: the paper
        broker is deterministic, so restoring it is placing it), and free cash `initial_cash +
        sum(pnl of closed) - sum(cost of open and pending)`. Its candles must be added again
        (see `PaperBroker.required_history`)."""
        live = self.live_positions()
        pnl = self._db.execute(
            "SELECT coalesce(sum(pnl), 0.0) AS pnl FROM positions WHERE status = 'closed'"
        ).fetchone()["pnl"]
        cash = initial_cash + pnl - sum(p.cost for p in live)
        orders = tuple(
            PendingOrder(o.to_order(), o.interval)
            for o in self.pending_orders()
            if o.type is OrderType.MARKET
        )
        positions = tuple(p.to_paper() for p in live if p.status is PositionStatus.OPEN)
        return PaperSnapshot(cash, positions, orders)
