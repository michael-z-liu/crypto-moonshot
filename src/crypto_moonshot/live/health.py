"""Health rules: the monitor that turns conditions into alerts.

`HealthMonitor.check()` evaluates a fixed set of rules against the engine, the feed and the state
file at the engine's clock time. Each rule yields zero or more `Finding`s with a key (`feed_
disconnected`, `provider_missing:cnn`, ...). A key that fires while it is not active becomes an
active alert (a row in the state file's `alerts` table) and writes one `alert` event
(`state: "firing"`); an active key that no longer fires is removed and writes one
(`state: "resolved"`). A condition that lasts therefore writes two events, not one per check.
Active alerts live in the state file, so a restart keeps them, with their acknowledgement, and
resolves the ones that ended meanwhile.

Rules (thresholds: `live.config.HealthConfig`):

- `feed_disconnected`: a feed connection has been down for longer than `feed_grace`.
- `no_ticks`: an interval had no processed tick for `no_ticks_intervals` intervals plus the tick
  timeout.
- `provider_missing:<strategy>`: an external strategy missed (or sent late) its last
  `provider_misses` batches.
- `plugin_failing:<strategy>`: an in-process plugin gave no signals for its last
  `provider_misses` candles.
- `daily_loss_limit:<strategy>`: the strategy's daily loss limit refuses entries today.
- `backfill_failed`: a seed or backfill request failed within `backfill_window`.
- `stale_symbols:<interval>`: more than `stale_fraction` of an interval's traded symbols is stale.
- `backup_failed`: the last automatic backup failed.
- `disk_low`: less than `min_free_gb` GB free where the state file lives.
- `orphan_order` and `balance_mismatch`: the latest reconciliation with the broker found open
  orders the state file does not know (warning), or balances that do not cover the open
  positions (error; the engine paused entries).
- `unprotected_position:<symbol>`: an open position whose exit plan needs a bracket the broker
  does not hold (exchange brokers: placing it failed).
- `broker_halted`: the broker stopped placing orders (the exchange refused the keys or banned
  the IP); it lasts until the daemon is restarted.

In a replay the clock is simulated, so the rules about wall-clock time (`feed_disconnected`,
`no_ticks`, the start-up grace of `stale_symbols`) are off; the others run.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from crypto_moonshot.live.config import HealthConfig
from crypto_moonshot.live.engine import Engine
from crypto_moonshot.live.feed import Feed
from crypto_moonshot.live.state import AlertRecord, BatchStatus, EventLevel
from crypto_moonshot.models import Interval
from crypto_moonshot.risk import DailyLossLimit

logger = logging.getLogger(__name__)

ALERT_KIND = "alert"
"""The `kind` of the events that record an alert firing or resolving."""
BACKFILL_KINDS = ("seed_failed", "backfill_failed")
GB = 1_000_000_000

DiskFree = Callable[[Path], int]


def disk_free_bytes(path: Path) -> int:
    """Free bytes on the file system holding `path`."""
    return shutil.disk_usage(path).free


@dataclass(frozen=True, slots=True)
class Finding:
    """A rule's result: `key` is the alert's identity, `rule` the rule's name."""

    key: str
    rule: str
    level: EventLevel
    detail: str


def _fmt(delta: timedelta) -> str:
    seconds = int(delta.total_seconds())
    if seconds < 120:
        return f"{seconds}s"
    if seconds < 7200:
        return f"{seconds // 60}m"
    return f"{seconds / 3600:.1f}h"


class HealthMonitor:
    """Evaluates the rules and records alert transitions; see the module docstring.

    Args:
        replay: simulated time: skip the wall-clock rules (default: `engine.replay`).
        disk_free: free bytes of a path (tests inject a fake).
        backup_error: the last automatic backup's error, or `None` when it succeeded (or none
            ran yet).
    """

    def __init__(
        self,
        engine: Engine,
        feed: Feed,
        config: HealthConfig,
        *,
        replay: bool | None = None,
        disk_free: DiskFree = disk_free_bytes,
        backup_error: Callable[[], str | None] = lambda: None,
    ) -> None:
        self._engine = engine
        self._feed = feed
        self._config = config
        self._replay = engine.replay if replay is None else replay
        self._disk_free = disk_free
        self._backup_error = backup_error
        self._started = engine.clock()
        self._down_since: datetime | None = None

    # -- evaluation -------------------------------------------------------------------------

    def check(self) -> list[Finding]:
        """Evaluate every rule once, record the transitions and return what fires now. A rule
        that raises is logged and keeps its alerts as they are."""
        engine = self._engine
        now = engine.clock()
        rules: list[tuple[tuple[str, ...], Callable[[datetime], Iterator[Finding]]]] = [
            (("feed_disconnected",), self._feed_disconnected),
            (("no_ticks",), self._no_ticks),
            (("provider_missing", "plugin_failing"), self._batches),
            (("daily_loss_limit",), self._daily_loss_limit),
            (("backfill_failed",), self._backfill_failed),
            (("stale_symbols",), self._stale_symbols),
            (("backup_failed",), self._backup_failed),
            (("disk_low",), self._disk_low),
            (("orphan_order", "balance_mismatch"), self._reconciliation),
            (("unprotected_position",), self._unprotected),
            (("broker_halted",), self._broker_halted),
        ]
        findings: dict[str, Finding] = {}
        failed: set[str] = set()
        for names, rule in rules:
            try:
                for finding in rule(now):
                    findings[finding.key] = finding
            except Exception:
                logger.exception("health rule %s failed", names[0])
                failed.update(names)
        self._apply(findings, failed)
        return list(findings.values())

    def _apply(self, findings: dict[str, Finding], failed: set[str]) -> None:
        store = self._engine.store
        active = {a.key: a for a in store.alerts()}
        with store.transaction():
            for key, finding in findings.items():
                if key not in active:
                    store.fire_alert(
                        key, finding.rule, finding.level, finding.detail, self._engine.clock()
                    )
                    self._write(finding.level, finding, "firing", finding.detail)
            for key, alert in active.items():
                if key in findings or alert.rule in failed:
                    continue
                store.resolve_alert(key)
                self._write(EventLevel.INFO, alert, "resolved", f"{key} is back to normal")

    def _write(
        self, level: EventLevel, alert: Finding | AlertRecord, state: str, detail: str
    ) -> None:
        key = alert.key
        message = f"{key}: {detail}" if state == "firing" else detail
        self._engine.record_event(
            level,
            ALERT_KIND,
            message,
            {"rule": alert.rule, "key": key, "state": state, "detail": detail},
        )

    # -- rules ------------------------------------------------------------------------------

    def _feed_disconnected(self, now: datetime) -> Iterator[Finding]:
        if self._replay:
            return
        info = self._feed.health()
        if info.connected:
            self._down_since = None
            return
        if self._down_since is None:
            self._down_since = now
        down = now - self._down_since
        if down > self._config.feed_grace:
            yield Finding(
                "feed_disconnected",
                "feed_disconnected",
                EventLevel.ERROR,
                f"the market data feed has been down for {_fmt(down)}",
            )

    def _no_ticks(self, now: datetime) -> Iterator[Finding]:
        if self._replay:
            return
        engine = self._engine
        timeout = engine.config.engine.tick_timeout
        done = engine.tick_times()
        late = []
        for interval in sorted({s.interval for s in engine.specs.values()}, key=_length):
            limit = self._config.no_ticks_intervals * interval.length + timeout
            since = max(self._started, done.get(interval, self._started))
            if now - since > limit:
                late.append(f"{interval.value} ({_fmt(now - since)})")
        if late:
            yield Finding(
                "no_ticks",
                "no_ticks",
                EventLevel.ERROR,
                "no tick processed for " + ", ".join(late),
            )

    def _batches(self, now: datetime) -> Iterator[Finding]:
        engine = self._engine
        n = self._config.provider_misses
        for spec in engine.specs.values():
            statuses = engine.store.recent_batch_statuses(spec.name, n)
            if len(statuses) < n:
                continue
            if spec.external and all(s is not BatchStatus.ON_TIME for s in statuses):
                late = sum(1 for s in statuses if s is BatchStatus.LATE)
                yield Finding(
                    f"provider_missing:{spec.name}",
                    "provider_missing",
                    EventLevel.WARNING,
                    f"the provider of {spec.name} missed or was late for its last {n} batches"
                    f" ({late} late)",
                )
            elif not spec.external and all(s is BatchStatus.MISSED for s in statuses):
                yield Finding(
                    f"plugin_failing:{spec.name}",
                    "plugin_failing",
                    EventLevel.WARNING,
                    f"the plugin of {spec.name} gave no signals for its last {n} candles",
                )

    def _daily_loss_limit(self, now: datetime) -> Iterator[Finding]:
        engine = self._engine
        for spec in engine.specs.values():
            limit = spec.config.risk.daily_loss_limit
            if limit is None:
                continue
            book = engine.store.derive_book(spec.name, spec.capital, now)
            if not DailyLossLimit(limit).allows(book, "", 0.0):
                yield Finding(
                    f"daily_loss_limit:{spec.name}",
                    "daily_loss_limit",
                    EventLevel.WARNING,
                    f"{spec.name} lost {-book.day_worst_pnl:,.2f} today, more than "
                    f"{limit:.1%} of its {book.day_start_equity:,.2f}: "
                    "no new entries until 00:00 UTC",
                )

    def _backfill_failed(self, now: datetime) -> Iterator[Finding]:
        window = self._config.backfill_window
        failures = self._engine.store.count_events(BACKFILL_KINDS, now - window)
        if failures:
            yield Finding(
                "backfill_failed",
                "backfill_failed",
                EventLevel.WARNING,
                f"{failures} seed or backfill request(s) failed in the last {_fmt(window)}",
            )

    def _stale_symbols(self, now: datetime) -> Iterator[Finding]:
        engine = self._engine
        if not self._replay and now - self._started < self._config.startup_grace:
            return  # the feed is still seeding its windows
        traded: dict[Interval, set[str]] = {}
        for spec in engine.specs.values():
            traded.setdefault(spec.interval, set()).update(spec.symbols)
        for interval in sorted(traded, key=_length):
            symbols = traded[interval]
            if not symbols:
                continue
            stale = self._feed.stale_symbols(interval, now, engine.config.account.stale_after)
            count = len(stale & symbols)
            if count / len(symbols) > self._config.stale_fraction:
                yield Finding(
                    f"stale_symbols:{interval.value}",
                    "stale_symbols",
                    EventLevel.WARNING,
                    f"{count} of {len(symbols)} {interval.value} symbols have stale data",
                )

    def _backup_failed(self, now: datetime) -> Iterator[Finding]:
        error = self._backup_error()
        if error is not None:
            yield Finding(
                "backup_failed",
                "backup_failed",
                EventLevel.ERROR,
                f"the last automatic backup failed: {error}",
            )

    def _disk_low(self, now: datetime) -> Iterator[Finding]:
        free = self._disk_free(self._engine.store.path.parent)
        if free < self._config.min_free_gb * GB:
            yield Finding(
                "disk_low",
                "disk_low",
                EventLevel.WARNING,
                f"only {free / GB:.2f} GB free where the state file lives",
            )

    def _reconciliation(self, now: datetime) -> Iterator[Finding]:
        report = self._engine.reconciliation
        if report is None:
            return
        if report.orphans:
            ids = ", ".join(o.order_id for o in report.orphans)
            yield Finding(
                "orphan_order",
                "orphan_order",
                EventLevel.WARNING,
                f"{len(report.orphans)} open order(s) on the exchange not in the state file: {ids}",
            )
        if report.mismatches:
            assets = ", ".join(
                f"{m.asset} (holds {m.actual:g}, needs {m.expected:g})" for m in report.mismatches
            )
            yield Finding(
                "balance_mismatch",
                "balance_mismatch",
                EventLevel.ERROR,
                f"balances do not cover the open positions: {assets}; entries paused",
            )

    def _unprotected(self, now: datetime) -> Iterator[Finding]:
        for position in self._engine.unprotected_positions():
            yield Finding(
                f"unprotected_position:{position.symbol}",
                "unprotected_position",
                EventLevel.ERROR,
                f"{position.symbol} ({position.strategy}) has no bracket at the broker",
            )

    def _broker_halted(self, now: datetime) -> Iterator[Finding]:
        halt = self._engine.broker_halt
        if halt is not None:
            yield Finding(
                "broker_halted",
                "broker_halted",
                EventLevel.ERROR,
                f"the broker stopped placing orders ({halt.reason}); restart once fixed",
            )


def _length(interval: Interval) -> timedelta:
    return interval.length


async def supervise(
    monitor: HealthMonitor,
    every: float,
    extra: Callable[[], None] | None = None,
) -> None:
    """Run `monitor.check()` (and `extra`, e.g. the stall watchdog's check) every `every`
    seconds until cancelled. Failures are logged, never raised."""
    while True:
        for step in (monitor.check, extra):
            if step is None:
                continue
            try:
                step()
            except Exception:
                logger.exception("health check failed")
        await asyncio.sleep(every)
