import pytest
import typer.rich_utils


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def _plain_typer_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Render CLI output as on a local non-terminal run, wherever tests run.

    Typer forces terminal output (colours, panel borders wrapped through the message) when
    GITHUB_ACTIONS, FORCE_COLOR or PY_COLORS is set, and rich treats FORCE_COLOR as a terminal,
    which breaks assertions on output text in CI.
    """
    monkeypatch.setattr(typer.rich_utils, "FORCE_TERMINAL", False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
