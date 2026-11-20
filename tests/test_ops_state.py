"""State file additions of P2: the schema v2 migration, alerts and the dashboard's queries."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from ops_fakes import add_batch, add_position

from crypto_moonshot.live.state import (
    MIGRATIONS,
    SCHEMA_VERSION,
    BatchStatus,
    EventLevel,
    SignalOutcome,
    SignalRow,
    StateStore,
)
from crypto_moonshot.models import Interval

T0 = datetime(2024, 3, 1, tzinfo=UTC)
HOUR = timedelta(hours=1)


def at(k: int) -> datetime:
    return T0 + k * HOUR


@pytest.fixture
def store(tmp_path: Path) -> StateStore:
    opened = StateStore.open(tmp_path / "state.sqlite")
    opened.upsert_strategy("alpha", {}, 1000.0)
    opened.upsert_strategy("beta", {}, 500.0)
    return opened


def test_a_version_1_file_migrates_to_version_2_keeping_its_data(tmp_path: Path) -> None:
    path = tmp_path / "old.sqlite"
    with StateStore.open(path, migrations=MIGRATIONS[:1]) as old:
        assert old.schema_version == 1
        old.upsert_strategy("alpha", {"x": 1}, 10.0)
        old.add_event(T0, EventLevel.WARNING, "kept", "an event from before")
        tables = {r[0] for r in old._db.execute("SELECT name FROM sqlite_master")}
        assert "alerts" not in tables
    with StateStore.open(path) as new:
        assert SCHEMA_VERSION == 2
        assert new.schema_version == 2
        assert [s.name for s in new.strategies()] == ["alpha"]
        assert [e.kind for e in new.events_since()] == ["kept"]
        assert new.alerts() == []  # the new table works
        new.fire_alert("disk_low", "disk_low", EventLevel.WARNING, "low", T0)
        assert new.acknowledge_alert("disk_low", T0)
        indexes = {r[1] for r in new._db.execute("SELECT * FROM sqlite_master WHERE type='index'")}
        assert {"events_kind", "batches_time"} <= indexes
    with StateStore.open(path) as again:  # nothing left to migrate
        assert again.schema_version == 2
        assert again.alert("disk_low") is not None


def test_a_version_1_file_is_refused_by_the_read_only_open(tmp_path: Path) -> None:
    path = tmp_path / "old.sqlite"
    StateStore.open(path, migrations=MIGRATIONS[:1]).close()
    with pytest.raises(Exception, match="schema version 1; this build reads 2"):
        StateStore.open_readonly(path)


def test_alerts_fire_once_acknowledge_once_and_resolve(store: StateStore) -> None:
    store.fire_alert("provider_missing:cnn", "provider_missing", EventLevel.WARNING, "d1", at(1))
    store.fire_alert("provider_missing:cnn", "provider_missing", EventLevel.ERROR, "d2", at(2))
    (alert,) = store.alerts()  # the second fire changed nothing
    assert (alert.rule, alert.level, alert.detail, alert.fired_at) == (
        "provider_missing",
        EventLevel.WARNING,
        "d1",
        at(1),
    )
    assert not alert.acknowledged
    assert store.acknowledge_alert("provider_missing:cnn", at(3)) is True
    assert store.acknowledge_alert("provider_missing:cnn", at(4)) is True  # idempotent
    acked = store.alert("provider_missing:cnn")
    assert acked is not None
    assert acked.acknowledged_at == at(3)  # the first time counts
    assert store.acknowledge_alert("nope", at(3)) is False
    assert store.resolve_alert("provider_missing:cnn") is True
    assert store.resolve_alert("provider_missing:cnn") is False
    assert store.alerts() == []
    assert store.acknowledge_alert("provider_missing:cnn", at(5)) is False


def test_an_acknowledgement_survives_reopening_the_file(tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite"
    with StateStore.open(path) as first:
        first.fire_alert("disk_low", "disk_low", EventLevel.WARNING, "low", at(0))
        first.acknowledge_alert("disk_low", at(1))
    with StateStore.open(path) as second:
        alert = second.alert("disk_low")
        assert alert is not None
        assert alert.acknowledged_at == at(1)


def test_recent_batch_statuses_are_newest_first(store: StateStore) -> None:
    for k, status in enumerate(
        [BatchStatus.ON_TIME, BatchStatus.MISSED, BatchStatus.LATE, BatchStatus.MISSED]
    ):
        add_batch(store, "alpha", at(k), status, lateness=None if status.value == "missed" else 1.0)
    assert store.recent_batch_statuses("alpha", 3) == [
        BatchStatus.MISSED,
        BatchStatus.LATE,
        BatchStatus.MISSED,
    ]
    assert store.recent_batch_statuses("beta", 3) == []


def test_batch_stats_count_a_window_and_find_the_median(store: StateStore) -> None:
    add_batch(store, "alpha", at(0), BatchStatus.ON_TIME, lateness=1.0)  # before the window
    add_batch(store, "alpha", at(10), BatchStatus.ON_TIME, lateness=2.0)
    add_batch(store, "alpha", at(11), BatchStatus.ON_TIME, lateness=4.0)
    add_batch(store, "alpha", at(12), BatchStatus.LATE, lateness=100.0)
    add_batch(store, "alpha", at(13), BatchStatus.MISSED)
    stats = {s.strategy: s for s in store.batch_stats(at(10))}
    alpha = stats["alpha"]
    assert (alpha.on_time, alpha.late, alpha.missed) == (2, 1, 1)
    assert alpha.median_lateness == 4.0
    assert alpha.worst_lateness == 100.0
    assert alpha.last_batch_time == at(12)  # the latest one that was received
    beta = stats["beta"]
    assert (beta.on_time, beta.late, beta.missed) == (0, 0, 0)
    assert beta.median_lateness is None
    assert beta.worst_lateness is None
    assert beta.last_batch_time is None
    add_batch(store, "alpha", at(14), BatchStatus.ON_TIME, lateness=10.0)
    even = {s.strategy: s for s in store.batch_stats(at(10))}["alpha"]
    assert even.median_lateness == 7.0  # (4 + 10) / 2 over 2, 4, 10, 100


def test_recent_signals_join_their_batch_and_filter(store: StateStore) -> None:
    rows = [SignalRow("AAAUSDT", priority=1.0), SignalRow("BBBUSDT", action="exit")]
    add_batch(store, "alpha", at(1), BatchStatus.ON_TIME, lateness=1.0, signals=rows)
    add_batch(
        store, "beta", at(1), BatchStatus.ON_TIME, lateness=1.0, signals=[SignalRow("CCCUSDT")]
    )
    batch = store.batch_for("alpha", Interval.H1, at(1))
    assert batch is not None
    store.set_signal_outcome(batch.signals[0].id, SignalOutcome.REJECTED, "slots_full")
    everything = store.recent_signals(10)
    assert [s.signal.symbol for s in everything] == ["CCCUSDT", "BBBUSDT", "AAAUSDT"]
    assert store.recent_signals(2) == everything[:2]
    only = store.recent_signals(10, "alpha")
    assert [s.signal.symbol for s in only] == ["BBBUSDT", "AAAUSDT"]
    assert only[1].signal.outcome is SignalOutcome.REJECTED
    assert only[1].signal.reason == "slots_full"
    assert (only[1].strategy, only[1].candle_time, only[1].batch_status) == (
        "alpha",
        at(1),
        BatchStatus.ON_TIME,
    )


def test_trades_and_equity_points(store: StateStore) -> None:
    add_position(store, "alpha", "AAAUSDT", at(0), pnl=10.0, exit_time=at(2))
    add_position(store, "alpha", "BBBUSDT", at(1), pnl=-4.0, exit_time=at(3))
    add_position(store, "alpha", "CCCUSDT", at(1), pnl=1.5, exit_time=at(5))
    add_position(store, "beta", "DDDUSDT", at(1), pnl=7.0, exit_time=at(4))
    add_position(store, "alpha", "EEEUSDT", at(6), exit_time=at(6))  # open: not a trade
    trades = store.recent_closed_positions(10)
    assert [p.symbol for p in trades] == ["CCCUSDT", "DDDUSDT", "BBBUSDT", "AAAUSDT"]
    assert [p.symbol for p in store.recent_closed_positions(2, "alpha")] == ["CCCUSDT", "BBBUSDT"]
    assert store.equity_points("alpha", 1000.0, 10) == [
        (at(2), 1010.0),
        (at(3), 1006.0),
        (at(5), 1007.5),
    ]
    # the latest points keep their cumulative values
    assert store.equity_points("alpha", 1000.0, 2) == [(at(3), 1006.0), (at(5), 1007.5)]
    assert store.realised_pnl("alpha") == pytest.approx(7.5)
    assert store.equity_points("beta", 500.0, 10) == [(at(4), 507.0)]


def test_event_queries(store: StateStore) -> None:
    store.add_event(at(0), EventLevel.INFO, "alert", "one")
    store.add_event(at(1), EventLevel.INFO, "other", "x")
    store.add_event(at(2), EventLevel.WARNING, "alert", "two")
    store.add_event(at(3), EventLevel.WARNING, "seed_failed", "s")
    assert [e.message for e in store.recent_events_of_kind("alert", 10)] == ["two", "one"]
    assert [e.message for e in store.recent_events_of_kind("alert", 1)] == ["two"]
    assert store.count_events(("seed_failed", "backfill_failed"), at(0)) == 1
    assert store.count_events(("seed_failed",), at(4)) == 0
    assert store.count_events(("alert", "other"), at(1)) == 2
