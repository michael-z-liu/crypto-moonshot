"""Watchdogs: exit (code 3) when the daemon is stuck, so Docker's restart policy starts it again
and restart recovery (the state file) takes over.

- `StallWatchdog`: the engine processed no tick for `watchdog_after` although the feed keeps
  delivering messages. Checked with the health rules (`check` returns the reason or `None`); the
  daemon then writes a `watchdog` error event, stops gracefully and exits with code 3. A feed
  that is down or silent is not the engine's fault (`feed_disconnected` covers it), so it never
  triggers this.
- `LoopWatchdog`: a thread that notices the event loop did not run for `threshold` seconds (a
  coroutine refreshes a timestamp). A blocked loop cannot shut down cleanly, so the thread
  logs and calls `os._exit(3)`.

Both take their thresholds, clocks and exit function as arguments, so tests drive them.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections.abc import Callable

from crypto_moonshot.live.config import HealthConfig
from crypto_moonshot.live.engine import Engine
from crypto_moonshot.live.feed import Feed

logger = logging.getLogger(__name__)

EXIT_WATCHDOG = 3
"""The exit code of a watchdog exit."""


class StallWatchdog:
    """Detects an engine that stopped processing ticks while the feed is alive."""

    def __init__(self, engine: Engine, feed: Feed, config: HealthConfig) -> None:
        self._engine = engine
        self._feed = feed
        self._config = config
        self._started = engine.clock()
        shortest = min(spec.interval.length for spec in engine.specs.values())
        self.threshold = config.watchdog_threshold(shortest)

    def check(self) -> str | None:
        """Why the engine counts as stalled now, or `None`."""
        engine = self._engine
        now = engine.clock()
        last = engine.last_tick_at
        since = self._started if last is None else max(self._started, last)
        stalled = now - since
        if stalled <= self.threshold:
            return None
        seen = self._feed.health().last_message_at
        if seen is None or now - seen > self._config.feed_grace:
            return None  # the feed itself is down or silent
        return (
            f"no tick was processed for {int(stalled.total_seconds())}s (limit "
            f"{int(self.threshold.total_seconds())}s) although the feed is delivering messages"
        )


class LoopWatchdog:
    """Exits the process when the event loop stops running for more than `threshold` seconds.

    `await run()` in the loop to be watched: it refreshes a timestamp every `beat` seconds and
    runs the checking thread while it lives.
    """

    def __init__(
        self,
        threshold: float,
        *,
        exit_fn: Callable[[int], object] = os._exit,
        monotonic: Callable[[], float] = time.monotonic,
        poll: float | None = None,
    ) -> None:
        if threshold <= 0:
            msg = "the loop watchdog's threshold must be positive"
            raise ValueError(msg)
        self.threshold = threshold
        self._exit = exit_fn
        self._monotonic = monotonic
        self._poll = poll if poll is not None else min(1.0, threshold / 4)
        self._beat_every = min(1.0, threshold / 8)
        self._last_beat = monotonic()
        self._stop = threading.Event()
        self.fired = threading.Event()

    def _watch(self) -> None:
        while not self._stop.wait(self._poll):
            blocked = self._monotonic() - self._last_beat
            if blocked > self.threshold:
                self.fired.set()
                logger.critical(
                    "the event loop has been blocked for %.0fs (limit %.0fs): exiting with code %d",
                    blocked,
                    self.threshold,
                    EXIT_WATCHDOG,
                )
                self._exit(EXIT_WATCHDOG)
                return

    async def run(self) -> None:
        self._last_beat = self._monotonic()
        thread = threading.Thread(target=self._watch, name="loop-watchdog", daemon=True)
        thread.start()
        try:
            while True:
                self._last_beat = self._monotonic()
                await asyncio.sleep(self._beat_every)
        finally:
            self._stop.set()
            thread.join(timeout=2.0)
