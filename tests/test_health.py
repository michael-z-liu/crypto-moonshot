"""The health rules: each fires and resolves with exactly one event per transition."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from engine_fakes import FakeClock, flat_bars
from ops_fakes import HealthFeed, add_batch, add_position

from crypto_moonshot.live.config import RunConfig, config_from_dict
from crypto_moonshot.live.engine import Engine, build_specs
from crypto_moonshot.live.health import GB, Finding, HealthMonitor
from crypto_moonshot.live.state import BatchStatus, EventLevel, StateStore
from crypto_moonshot.models import Interval

H1 = Interval.H1
HOUR = timedelta(hours=1)
T0 = datetime(2024, 3, 1, tzinfo=UTC)
SYMBOLS = ["AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT", "EEEUSDT"]


def at(k: int) -> datetime:
    return T0 + k * HOUR


def make_config(path: Path, health: dict[str, Any] | None = None) -> RunConfig:
    return config_from_dict(
        {
            "state": str(path),
            "account": {"initial_cash": 10_000.0},
            "engine": {"tick_timeout": "10s"},
            "health": health or {},
            "strategies": [
                {
                    "name": "ext",
                    "source": "external",
                    "interval": "1h",
                    "warmup": 3,
                    "symbols": SYMBOLS,
                    "capital": 1000.0,
                    "sizing": {"stake": 100.0},
                    "exit": {"max_hold": "10h"},
                },
                {
                    "name": "inproc",
                    "source": "tests:not_loaded",
                    "symbols": ["ZZZUSDT"],
                    "capital": 1000.0,
                    "sizing": {"stake": 100.0},
                    "risk": {"daily_loss_limit": 0.1},
                },
            ],
        }
    )


class Rig:
    def __init__(self, tmp_path: Path, health: dict[str, Any] | None = None) -> None:
        from engine_fakes import Scheduled

        # the time-based rules are off unless a test turns them on
        health = {"startup_grace": "365d", "no_ticks_intervals": 10_000, **(health or {})}
        self.config = make_config(tmp_path / "state.sqlite", health)
        self.clock = FakeClock(at(0))
        self.feed = HealthFeed(H1, T0, {s: flat_bars(40) for s in (*SYMBOLS, "ZZZUSDT")})
        self.store = StateStore.open(self.config.state)
        specs = build_specs(self.config, None, plugins={"inproc": Scheduled({}, name="inproc")})
        self.engine = Engine(self.config, specs, self.store, clock=self.clock)
        self.engine.start()
        self.free = 50 * GB
        self.backup_error: str | None = None
        self.monitor = HealthMonitor(
            self.engine,
            self.feed,
            self.config.health,
            disk_free=lambda _path: self.free,
            backup_error=lambda: self.backup_error,
        )

    def check(self, *, advance: timedelta = timedelta(0)) -> list[Finding]:
        self.clock.now += advance
        return self.monitor.check()

    def alerts(self) -> set[str]:
        return {a.key for a in self.store.alerts()}

    def events(self) -> list[tuple[str, str, str]]:
        """(key, state, level) of every alert event, oldest first."""
        return [
            (e.data["key"], e.data["state"], e.level.value)
            for e in self.store.events_since()
            if e.kind == "alert" and e.data is not None
        ]


@pytest.fixture
def rig(tmp_path: Path) -> Rig:
    return Rig(tmp_path)


def test_a_healthy_engine_raises_nothing(rig: Rig) -> None:
    assert rig.check(advance=timedelta(seconds=30)) == []
    assert rig.alerts() == set()
    assert rig.events() == []


def test_feed_disconnected_fires_after_the_grace_and_resolves_once(rig: Rig) -> None:
    rig.feed.connected = False
    assert rig.check() == []  # first seen down: the grace starts now
    assert rig.check(advance=timedelta(seconds=60)) == []  # exactly the grace
    (finding,) = rig.check(advance=timedelta(seconds=1))
    assert finding.key == "feed_disconnected"
    for _ in range(3):  # still down: no further events
        rig.check(advance=timedelta(seconds=15))
    assert rig.events() == [("feed_disconnected", "firing", "error")]
    assert rig.alerts() == {"feed_disconnected"}
    rig.feed.connected = True
    assert rig.check(advance=timedelta(seconds=15)) == []
    rig.check(advance=timedelta(seconds=15))
    assert rig.events() == [
        ("feed_disconnected", "firing", "error"),
        ("feed_disconnected", "resolved", "info"),
    ]
    assert rig.alerts() == set()


def test_a_reconnect_inside_the_grace_never_alerts(rig: Rig) -> None:
    rig.feed.connected = False
    rig.check()
    rig.check(advance=timedelta(seconds=50))
    rig.feed.connected = True
    rig.check(advance=timedelta(seconds=5))
    rig.feed.connected = False
    rig.check(advance=timedelta(seconds=5))
    rig.check(advance=timedelta(seconds=55))  # 55 s down since the second drop
    assert rig.events() == []


def test_thresholds_come_from_the_config(tmp_path: Path) -> None:
    custom = Rig(tmp_path, {"feed_grace": "5s"})
    custom.feed.connected = False
    custom.check()
    assert [f.key for f in custom.check(advance=timedelta(seconds=6))] == ["feed_disconnected"]


def test_no_ticks_waits_two_intervals_plus_the_tick_timeout(tmp_path: Path) -> None:
    rig = Rig(tmp_path, {"no_ticks_intervals": 2})
    limit = 2 * HOUR + timedelta(seconds=10)
    rig.check(advance=limit)
    assert rig.alerts() == set()  # exactly the limit is still fine
    (finding,) = rig.check(advance=timedelta(seconds=1))
    assert finding.key == "no_ticks"
    assert finding.level is EventLevel.ERROR
    rig.check(advance=timedelta(seconds=15))
    assert rig.events() == [("no_ticks", "firing", "error")]
    # a processed tick resolves it
    rig.engine._tick_done_at[H1] = rig.clock.now
    rig.check()
    assert rig.events()[-1] == ("no_ticks", "resolved", "info")
    assert len(rig.events()) == 2


def test_provider_missing_fires_on_three_missed_or_late_batches(rig: Rig) -> None:
    store = rig.store
    add_batch(store, "ext", at(0), BatchStatus.ON_TIME, lateness=1.0)
    add_batch(store, "ext", at(1), BatchStatus.MISSED)
    add_batch(store, "ext", at(2), BatchStatus.LATE, lateness=100.0)
    assert rig.check() == []  # two bad ones, the third from the last is on time
    add_batch(store, "ext", at(3), BatchStatus.MISSED)
    (finding,) = rig.check()
    assert finding.key == "provider_missing:ext"
    assert finding.level is EventLevel.WARNING
    rig.check()
    assert rig.events() == [("provider_missing:ext", "firing", "warning")]
    add_batch(store, "ext", at(4), BatchStatus.ON_TIME, lateness=2.0)
    assert rig.check() == []
    assert rig.events()[-1] == ("provider_missing:ext", "resolved", "info")
    assert len(rig.events()) == 2


def test_plugin_failing_counts_only_missed_batches_of_in_process_plugins(rig: Rig) -> None:
    store = rig.store
    for k in range(3):
        add_batch(store, "inproc", at(k), BatchStatus.MISSED)
    (finding,) = rig.check()
    assert finding.key == "plugin_failing:inproc"
    add_batch(store, "inproc", at(3), BatchStatus.ON_TIME, lateness=0.5)
    rig.check()
    assert rig.events() == [
        ("plugin_failing:inproc", "firing", "warning"),
        ("plugin_failing:inproc", "resolved", "info"),
    ]


def test_the_batch_rules_use_the_configured_count(tmp_path: Path) -> None:
    custom = Rig(tmp_path, {"provider_misses": 1})
    add_batch(custom.store, "ext", at(0), BatchStatus.MISSED)
    assert [f.key for f in custom.check()] == ["provider_missing:ext"]


def test_daily_loss_limit_fires_for_the_day_and_resolves_at_midnight(rig: Rig) -> None:
    rig.clock.now = at(30)  # 2024-03-02 06:00
    add_position(rig.store, "inproc", "ZZZUSDT", at(25), pnl=-150.0, exit_time=at(26))
    (finding,) = rig.check()
    assert finding.key == "daily_loss_limit:inproc"
    rig.check(advance=timedelta(minutes=15))
    assert rig.events() == [("daily_loss_limit:inproc", "firing", "warning")]
    rig.clock.now = datetime(2024, 3, 3, 0, 0, tzinfo=UTC)
    assert rig.check() == []
    assert rig.events()[-1] == ("daily_loss_limit:inproc", "resolved", "info")
    assert len(rig.events()) == 2


def test_a_loss_within_the_limit_does_not_fire(rig: Rig) -> None:
    rig.clock.now = at(30)
    add_position(rig.store, "inproc", "ZZZUSDT", at(25), pnl=-90.0, exit_time=at(26))
    assert rig.check() == []


def test_backfill_failed_while_a_failure_is_within_the_window(rig: Rig) -> None:
    rig.clock.now = at(5)
    rig.engine.record_event(EventLevel.WARNING, "backfill_failed", "x", {})
    (finding,) = rig.check()
    assert finding.key == "backfill_failed"
    rig.check(advance=timedelta(minutes=59))
    assert rig.alerts() == {"backfill_failed"}
    rig.check(advance=timedelta(minutes=2))  # the failure is now more than an hour old
    assert rig.events() == [
        ("backfill_failed", "firing", "warning"),
        ("backfill_failed", "resolved", "info"),
    ]


def test_a_failed_seed_counts_too(rig: Rig) -> None:
    rig.engine.record_event(EventLevel.WARNING, "seed_failed", "x", {})
    assert [f.key for f in rig.check()] == ["backfill_failed"]


def fresh_feed(rig: Rig) -> None:
    """Every traded symbol has the candle before `at(60)`."""
    rig.feed.bars = {s: flat_bars(80) for s in rig.feed.bars}
    for symbol in rig.feed.bars:
        rig.feed.late(symbol, 59)
    rig.clock.now = at(60) + timedelta(minutes=5)


def test_stale_symbols_fires_above_the_fraction_and_resolves(tmp_path: Path) -> None:
    rig = Rig(tmp_path, {"startup_grace": "1m"})
    fresh_feed(rig)
    assert rig.check() == []
    held = rig.feed.known.pop("AAAUSDT")  # 1 of the 6 traded symbols: 16.7%, not above 20%
    assert rig.check() == []
    rig.feed.known.pop("BBBUSDT")  # 2 of 6
    (finding,) = rig.check()
    assert finding.key == "stale_symbols:1h"
    assert "2 of 6" in finding.detail
    rig.check(advance=timedelta(minutes=1))
    assert rig.events() == [("stale_symbols:1h", "firing", "warning")]
    rig.feed.known["AAAUSDT"] = held
    rig.check()
    assert rig.events()[-1] == ("stale_symbols:1h", "resolved", "info")
    assert len(rig.events()) == 2


def test_the_stale_fraction_is_configurable(tmp_path: Path) -> None:
    rig = Rig(tmp_path, {"startup_grace": "1m", "stale_fraction": 0.0})
    fresh_feed(rig)
    rig.feed.known.pop("AAAUSDT")
    assert [f.key for f in rig.check()] == ["stale_symbols:1h"]


def test_stale_symbols_waits_out_the_start_up_grace(tmp_path: Path) -> None:
    rig = Rig(tmp_path, {"startup_grace": "2m"})
    rig.clock.now = at(0) + timedelta(seconds=30)  # the feed is still seeding: windows empty
    assert rig.check() == []
    assert [f.key for f in rig.check(advance=timedelta(minutes=2))] == ["stale_symbols:1h"]


def test_backup_failed_follows_the_schedulers_error(rig: Rig) -> None:
    rig.backup_error = "OSError: disk full"
    (finding,) = rig.check()
    assert finding.key == "backup_failed"
    assert "disk full" in finding.detail
    rig.check()
    rig.backup_error = None
    rig.check()
    assert rig.events() == [
        ("backup_failed", "firing", "error"),
        ("backup_failed", "resolved", "info"),
    ]


def test_disk_low_uses_the_injected_disk_usage(rig: Rig, tmp_path: Path) -> None:
    seen: list[Path] = []

    def disk(path: Path) -> int:
        seen.append(path)
        return rig.free

    rig.monitor._disk_free = disk
    rig.free = int(0.5 * GB)
    (finding,) = rig.check()
    assert finding.key == "disk_low"
    assert "0.50 GB" in finding.detail
    assert seen[0] == tmp_path  # where the state file lives
    rig.free = 5 * GB
    rig.check()
    assert rig.events() == [("disk_low", "firing", "warning"), ("disk_low", "resolved", "info")]


def test_the_disk_threshold_is_configurable(tmp_path: Path) -> None:
    custom = Rig(tmp_path, {"min_free_gb": 100})
    assert [f.key for f in custom.check()] == ["disk_low"]  # 50 GB free


def test_a_replay_skips_the_wall_clock_rules(tmp_path: Path) -> None:
    quiet = Rig(tmp_path, {"no_ticks_intervals": 2})
    quiet.monitor._replay = True
    quiet.feed.connected = False
    quiet.check()
    keys = {f.key for f in quiet.check(advance=timedelta(days=3))}
    assert keys.isdisjoint({"feed_disconnected", "no_ticks"})


def test_an_alert_that_still_fires_after_a_restart_stays_and_keeps_its_ack(tmp_path: Path) -> None:
    first = Rig(tmp_path)
    first.free = 0
    first.check()
    assert first.store.acknowledge_alert("disk_low", first.clock.now)
    first.store.close()
    second = Rig(tmp_path)
    second.free = 0
    second.clock.now = at(1)
    second.check()
    (alert,) = second.store.alerts()
    assert alert.key == "disk_low"
    assert alert.acknowledged
    assert [e for e in second.events() if e[1] == "firing"] == [("disk_low", "firing", "warning")]
    second.free = 50 * GB  # resolved while it was down is found out at the next check
    second.check()
    assert second.alerts() == set()


def test_a_rule_that_raises_leaves_its_alerts_alone(
    rig: Rig, caplog: pytest.LogCaptureFixture
) -> None:
    rig.free = 0
    rig.check()
    assert rig.alerts() == {"disk_low"}

    def broken(_path: Path) -> int:
        raise OSError("no such device")

    rig.monitor._disk_free = broken
    rig.check()
    assert rig.alerts() == {"disk_low"}  # not resolved by a failed check
    assert "health rule disk_low failed" in caplog.text


def test_acknowledgement_is_cleared_when_the_alert_resolves_and_fires_again(rig: Rig) -> None:
    rig.free = 0
    rig.check()
    assert rig.store.acknowledge_alert("disk_low", rig.clock.now) is True
    rig.check()  # still firing: stays acknowledged, no new event
    assert rig.store.alert("disk_low") is not None
    assert rig.store.alert("disk_low").acknowledged  # type: ignore[union-attr]
    rig.free = 50 * GB
    rig.check()
    assert rig.store.alert("disk_low") is None
    rig.free = 0
    rig.check()
    fresh = rig.store.alert("disk_low")
    assert fresh is not None
    assert not fresh.acknowledged
    assert [e[1] for e in rig.events()] == ["firing", "resolved", "firing"]
