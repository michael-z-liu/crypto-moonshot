from pathlib import Path

import pytest
import typer.rich_utils
from replay_fakes import COINS, END, H1, START, at, rule_strategy, write_replay_store

from crypto_moonshot.data.store import KlineStore
from crypto_moonshot.live.plugin import rolling_signals
from crypto_moonshot.strategy.base import Signals


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def _plain_typer_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Render CLI output as on a local non-terminal run, wherever tests run.

    Typer forces terminal output (colours, panel borders wrapped through the message) when
    GITHUB_ACTIONS, FORCE_COLOR or PY_COLORS is set, and rich treats FORCE_COLOR as a terminal,
    which breaks assertions on output text in CI. A wide `COLUMNS` keeps error panels from
    wrapping a message through long temporary paths, which differ between machines.
    """
    monkeypatch.setattr(typer.rich_utils, "FORCE_TERMINAL", False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setenv("COLUMNS", "200")


@pytest.fixture(scope="session")
def replay_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The hourly capitulation store of `replay_fakes` (shared, read only)."""
    path = tmp_path_factory.mktemp("replay")
    write_replay_store(path)
    return path


@pytest.fixture(scope="session")
def replay_signals(replay_root: Path) -> Signals:
    """The rolling (live) signals of `replay_fakes.RULE` over the replay range."""
    store = KlineStore(replay_root)
    series = {s: store.read_series(s, H1) for s in COINS}
    context = {"BTCUSDT": store.read_series("BTCUSDT", H1)}
    return rolling_signals(rule_strategy(), series, context, at(START), at(END))


@pytest.fixture(autouse=True)
def _no_testnet_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's own testnet keys or URLs in the shell never reach a test."""
    for name in (
        "MOONSHOT_BINANCE_TESTNET_API_KEY",
        "MOONSHOT_BINANCE_TESTNET_API_SECRET",
        "MOONSHOT_BINANCE_TESTNET_REST_URL",
        "MOONSHOT_BINANCE_TESTNET_WS_API_URL",
        "MOONSHOT_BINANCE_API_KEY",
        "MOONSHOT_BINANCE_API_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)
