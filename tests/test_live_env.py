"""Where `moonshot live run` reads keys from: paper mode never reads an env file; testnet mode
reads `--env-file`, else `./.env`; real environment variables win; secrets are masked in logs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fake_exchange import API_KEY, API_SECRET
from typer.testing import CliRunner, Result

from crypto_moonshot import live_cli
from crypto_moonshot.cli import app

TOKEN = "tkn-env-test-Pw3Mx8"
KEYS = (
    f"MOONSHOT_BINANCE_TESTNET_API_KEY={API_KEY}\n"
    f"MOONSHOT_BINANCE_TESTNET_API_SECRET={API_SECRET}\n"
)
STRATEGY = """
[[strategies]]
name = "ext"
source = "external"
interval = "1h"
warmup = 5
symbols = ["BTCUSDT"]
capital = 100
sizing = { stake = 10 }
"""


def config(tmp_path: Path, mode: str) -> Path:
    path = tmp_path / "run.toml"
    cash = "initial_cash = 1000\n" if mode == "paper" else ""
    path.write_text(f'mode = "{mode}"\nstate = "s.sqlite"\n[account]\n{cash}{STRATEGY}')
    return path


@pytest.fixture
def seen(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[Any]:
    """Replace the daemon: record the settings `live run` hands it."""
    monkeypatch.chdir(tmp_path)
    calls: list[Any] = []

    async def fake(run_config: Any, settings: Any, **kwargs: Any) -> None:
        calls.append(settings)

    monkeypatch.setattr(live_cli, "run_daemon", fake)
    return calls


def run(*args: str, env: dict[str, str] | None = None) -> Result:
    return CliRunner().invoke(app, ["live", "run", *args], env=env)


def test_paper_mode_never_reads_a_dot_env(tmp_path: Path, seen: list[Any]) -> None:
    (tmp_path / ".env").write_text(f"MOONSHOT_API_TOKEN={TOKEN}\n\x00 garbage = = =\n{KEYS}")
    result = run("--config", str(config(tmp_path, "paper")))
    assert result.exit_code == 2  # the token in .env was not picked up
    assert "MOONSHOT_API_TOKEN is not set" in result.stderr
    assert seen == []
    # with the token in the environment it runs, and still holds no testnet keys
    ok = run("--config", str(config(tmp_path, "paper")), env={"MOONSHOT_API_TOKEN": TOKEN})
    assert ok.exit_code == 0, ok.output
    assert seen[0].binance_testnet_api_key is None


def test_paper_mode_ignores_an_env_file_option_with_a_warning(
    tmp_path: Path, seen: list[Any]
) -> None:
    keys = tmp_path / "keys.env"
    keys.write_text(KEYS)
    result = run(
        "--config",
        str(config(tmp_path, "paper")),
        "--env-file",
        str(keys),
        env={"MOONSHOT_API_TOKEN": TOKEN},
    )
    assert result.exit_code == 0, result.output
    assert "--env-file is ignored" in result.stderr
    assert seen[0].binance_testnet_api_key is None


def test_testnet_reads_the_env_file_option(tmp_path: Path, seen: list[Any]) -> None:
    keys = tmp_path / "keys.env"
    keys.write_text(KEYS + f"MOONSHOT_API_TOKEN={TOKEN}\n")
    result = run("--config", str(config(tmp_path, "testnet")), "--env-file", str(keys))
    assert result.exit_code == 0, result.output
    settings = seen[0]
    assert settings.binance_testnet_api_key.get_secret_value() == API_KEY
    assert settings.binance_testnet_api_secret.get_secret_value() == API_SECRET
    assert settings.api_token.get_secret_value() == TOKEN
    assert API_KEY not in result.output


def test_testnet_falls_back_to_dot_env_in_the_current_folder(
    tmp_path: Path, seen: list[Any]
) -> None:
    (tmp_path / ".env").write_text(KEYS + f"MOONSHOT_API_TOKEN={TOKEN}\n")
    result = run("--config", str(config(tmp_path, "testnet")))
    assert result.exit_code == 0, result.output
    assert seen[0].binance_testnet_api_key.get_secret_value() == API_KEY


def test_the_real_environment_wins_over_the_file(tmp_path: Path, seen: list[Any]) -> None:
    keys = tmp_path / "keys.env"
    keys.write_text(KEYS + f"MOONSHOT_API_TOKEN={TOKEN}\n")
    result = run(
        "--config",
        str(config(tmp_path, "testnet")),
        "--env-file",
        str(keys),
        env={"MOONSHOT_BINANCE_TESTNET_API_KEY": "from-the-environment"},
    )
    assert result.exit_code == 0, result.output
    assert seen[0].binance_testnet_api_key.get_secret_value() == "from-the-environment"


def test_a_missing_env_file_is_a_usage_error(tmp_path: Path, seen: list[Any]) -> None:
    result = run(
        "--config", str(config(tmp_path, "testnet")), "--env-file", str(tmp_path / "nope.env")
    )
    assert result.exit_code == 2
    assert "does not exist" in result.stderr
    assert seen == []


def test_the_logs_mask_the_keys(
    tmp_path: Path, seen: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    masked: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        live_cli, "configure_logging", lambda *a, secrets=(), **k: masked.append(tuple(secrets))
    )
    keys = tmp_path / "keys.env"
    keys.write_text(KEYS + f"MOONSHOT_API_TOKEN={TOKEN}\n")
    assert run("--config", str(config(tmp_path, "testnet")), "--env-file", str(keys)).exit_code == 0
    assert set(masked[0]) == {TOKEN, API_KEY, API_SECRET}


def test_the_example_testnet_config_loads_and_its_exit_plans_fit_the_exchange() -> None:
    from crypto_moonshot.broker.binance import check_exit_plan
    from crypto_moonshot.live.config import load_config
    from crypto_moonshot.live.engine import build_specs

    root = Path(__file__).resolve().parent.parent
    run = load_config(root / "examples" / "testnet.toml")
    run.require_runnable()
    assert run.mode == "testnet"
    assert run.account.initial_cash is None
    specs = build_specs(run, None)
    for spec in specs:
        check_exit_plan(spec.exit)
    assert sum(s.capital for s in specs) <= 100


def test_the_env_example_names_the_two_testnet_variables() -> None:
    text = (Path(__file__).resolve().parent.parent / ".env.example").read_text()
    assert "MOONSHOT_BINANCE_TESTNET_API_KEY=" in text
    assert "MOONSHOT_BINANCE_TESTNET_API_SECRET=" in text
    assert "testnet.binance.vision" in text
