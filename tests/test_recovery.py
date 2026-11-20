"""Crash recovery: a replay interrupted anywhere and resumed ends exactly like one that was not."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from engine_fakes import dump_state
from polars.testing import assert_frame_equal
from replay_fakes import END, H1, START, at, rule_strategy, run_config, strategy_dict

from crypto_moonshot.live.config import RunConfig
from crypto_moonshot.live.engine import Step
from crypto_moonshot.live.replay import ReplayResult, StoreHistory, run_replay
from crypto_moonshot.live.state import StateStore
from crypto_moonshot.portfolio import Book
from crypto_moonshot.strategy.base import Signals

TESTS = Path(__file__).parent


class Crash(Exception):
    """An injected failure."""


def crash_at(point: str, n: int) -> Callable[[str, Any], None]:
    """A step hook that raises at the `n`-th time `point` is reached."""
    seen = [0]

    def hook(step: str, tick: Any) -> None:
        if step == point:
            seen[0] += 1
            if seen[0] == n:
                raise Crash(f"{point} #{n}")

    return hook


def config_for(path: Path) -> RunConfig:
    # Equal slots compound from realised equity, so a restart must re-derive the books to the
    # last bit; the slot and cash limits make the entries compete.
    entry = strategy_dict(capital=1000.0, sizing={"slots": 3}, risk={"max_positions": 3})
    return run_config(path, [entry])


def replay(
    path: Path,
    root: Path,
    signals: Signals,
    hook: Callable[[str, Any], None] | None = None,
    **kw: Any,
) -> ReplayResult:
    return run_replay(
        config_for(path),
        at(START),
        at(END),
        root,
        plugins={"cap": rule_strategy()},
        precomputed={"cap": signals},
        step_hook=hook,
        **kw,
    )


@dataclass(frozen=True)
class Final:
    tables: dict[str, list[tuple[object, ...]]]
    book: Book
    result: ReplayResult


def final(path: Path, result: ReplayResult) -> Final:
    with StateStore.open(path) as store:
        return Final(dump_state(store), store.derive_book("cap", 1000.0, at(END)), result)


@pytest.fixture(scope="module")
def baseline(
    replay_root: Path, replay_signals: Signals, tmp_path_factory: pytest.TempPathFactory
) -> Final:
    path = tmp_path_factory.mktemp("baseline") / "state.sqlite"
    return final(path, replay(path, replay_root, replay_signals))


def assert_same(got: Final, want: Final) -> None:
    assert got.tables.keys() == want.tables.keys()
    for table in want.tables:
        assert got.tables[table] == want.tables[table], table
    assert got.book == want.book
    assert_frame_equal(got.result.trades, want.result.trades)
    assert_frame_equal(got.result.equity, want.result.equity)


def test_the_baseline_trades_and_competes(baseline: Final) -> None:
    assert baseline.result.trades.height > 30
    signals = baseline.result.document["signals"]
    assert signals["rejected:slots_full"] > 10
    assert len(baseline.tables["fills"]) == 2 * baseline.result.trades.height


@pytest.mark.parametrize(
    ("point", "n"),
    [
        (Step.AFTER_BROKER, 1),
        (Step.AFTER_BROKER, 57),
        (Step.AFTER_BROKER, 251),
        (Step.AFTER_SIGNALS, 9),
        (Step.AFTER_SIGNALS, 140),
        (Step.BEFORE_BROKER_CALL, 1),
        (Step.BEFORE_BROKER_CALL, 17),
        (Step.BEFORE_BROKER_CALL, 40),
        (Step.BEFORE_TICK_MARKER, 6),
        (Step.BEFORE_TICK_MARKER, 190),
        (Step.TICK_DONE, 100),
    ],
)
def test_a_crash_at_any_step_then_a_resume_matches_an_uninterrupted_run(
    replay_root: Path, replay_signals: Signals, tmp_path: Path, baseline: Final, point: str, n: int
) -> None:
    path = tmp_path / "state.sqlite"
    with pytest.raises(Crash):
        replay(path, replay_root, replay_signals, crash_at(point, n))
    resumed = replay(path, replay_root, replay_signals)
    assert_same(final(path, resumed), baseline)


def test_crash_after_crash_still_matches(
    replay_root: Path, replay_signals: Signals, tmp_path: Path, baseline: Final
) -> None:
    path = tmp_path / "state.sqlite"
    for point, n in [
        (Step.BEFORE_BROKER_CALL, 5),
        (Step.AFTER_BROKER, 30),
        (Step.BEFORE_TICK_MARKER, 1),
        (Step.BEFORE_BROKER_CALL, 9),
        (Step.AFTER_SIGNALS, 60),
    ]:
        with pytest.raises(Crash):
            replay(path, replay_root, replay_signals, crash_at(point, n))
    resumed = replay(path, replay_root, replay_signals)
    assert_same(final(path, resumed), baseline)
    with StateStore.open(path) as store:
        assert [e.kind for e in store.events_since()].count("restart") == 5


def test_a_resume_fetches_old_positions_candles_from_the_store(
    replay_root: Path,
    replay_signals: Signals,
    tmp_path: Path,
    baseline: Final,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, Any]] = []
    fetch = StoreHistory.fetch_klines

    async def spy(self: StoreHistory, symbol: str, interval: Any, start: Any, end: Any) -> Any:
        calls.append((symbol, start))
        return await fetch(self, symbol, interval, start, end)

    monkeypatch.setattr(StoreHistory, "fetch_klines", spy)
    path = tmp_path / "state.sqlite"
    with pytest.raises(Crash):
        replay(path, replay_root, replay_signals, crash_at(Step.TICK_DONE, 150))
    # The resumed feed's windows reach back two candles only: older positions' candles come
    # from the history fetcher.
    resumed = replay(path, replay_root, replay_signals, lookback={H1: 2})
    assert {symbol for symbol, _ in calls} == {"AAAUSDT", "EEEUSDT", "FFFUSDT"}
    assert_same(final(path, resumed), baseline)


SCRIPT = """
import os
import sys
from pathlib import Path

sys.path.insert(0, {tests!r})
from replay_fakes import END, START, at, rule_strategy, run_config, strategy_dict
from crypto_moonshot.live.replay import run_replay

entry = strategy_dict(capital=1000.0, sizing={{"slots": 3}}, risk={{"max_positions": 3}})
config = run_config(Path({state!r}), [entry])
seen = [0]


def hook(step, tick):
    if step == {point!r}:
        seen[0] += 1
        if seen[0] == {n}:
            os._exit(17)


run_replay(
    config,
    at(START),
    at(END),
    Path({root!r}),
    plugins={{"cap": rule_strategy()}},
    signals="backtest",
    step_hook=hook,
)
sys.exit(0)
"""


@pytest.mark.parametrize(
    ("point", "n"), [(Step.BEFORE_TICK_MARKER, 150), (Step.BEFORE_BROKER_CALL, 20)]
)
def test_a_killed_process_resumes_to_the_same_result(
    replay_root: Path, replay_signals: Signals, tmp_path: Path, baseline: Final, point: str, n: int
) -> None:
    path = tmp_path / "state.sqlite"
    code = SCRIPT.format(
        tests=str(TESTS), state=str(path), root=str(replay_root), point=str(point), n=n
    )
    script = tmp_path / "kill.py"
    script.write_text(textwrap.dedent(code))
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST")}
    done = subprocess.run(
        [sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 17, done.stderr
    with StateStore.open(path) as store:
        assert store.last_tick(H1) is not None  # it had committed some ticks
        assert store.last_tick(H1) < at(END - 1)  # type: ignore[operator]
    resumed = replay(path, replay_root, replay_signals)
    assert_same(final(path, resumed), baseline)
