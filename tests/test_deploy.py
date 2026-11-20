"""The Docker deployment files: the example container config and the read-only runtime."""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

from crypto_moonshot.live.config import EXTERNAL, load_config
from crypto_moonshot.live.plugin import load_strategy
from crypto_moonshot.live.state import StateStore

ROOT = Path(__file__).resolve().parent.parent
CONTAINER_CONFIG = ROOT / "deploy" / "config" / "paper.toml"
PLUGIN = ROOT / "examples" / "strategies" / "ma_cross.py"


def test_container_config_loads_with_absolute_container_paths() -> None:
    config = load_config(CONTAINER_CONFIG)
    config.require_runnable()
    assert config.mode == "paper"
    assert config.state == Path("/state/paper.sqlite")
    for strategy in config.strategies:
        assert strategy.source == EXTERNAL or strategy.source.startswith("/strategies/")


def test_container_config_plugins_load_from_the_examples_folder() -> None:
    raw = tomllib.loads(CONTAINER_CONFIG.read_text())
    for entry in raw["strategies"]:
        source: str = entry["source"].replace("/strategies/", f"{PLUGIN.parent}/")
        strategy = load_strategy(source, name=entry["name"], params=entry.get("params"))
        assert strategy.name == entry["name"]


def test_container_config_holds_no_secret() -> None:
    text = CONTAINER_CONFIG.read_text().lower()
    assert "token =" not in text
    assert "secret" not in text.replace("no secrets", "")


def test_cli_needs_neither_a_home_nor_a_writable_working_directory(tmp_path: Path) -> None:
    """The container's root filesystem is read-only and the user has no home folder."""
    state = tmp_path / "state" / "paper.sqlite"
    state.parent.mkdir()
    with StateStore.open(state):
        pass
    config = tmp_path / "paper.toml"
    config.write_text(
        CONTAINER_CONFIG.read_text()
        .replace("/state/", f"{state.parent}/")
        .replace("/strategies/", f"{PLUGIN.parent}/")
    )
    cwd = tmp_path / "readonly"
    cwd.mkdir()
    cwd.chmod(0o500)
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"HOME", "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "MOONSHOT_API_TOKEN"}
    }
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        for args in (["live", "run", "--help"], ["live", "status", "--config", str(config)]):
            result = subprocess.run(
                [sys.executable, "-m", "crypto_moonshot", *args],
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            assert result.returncode == 0, result.stderr
        assert list(cwd.iterdir()) == []
    finally:
        cwd.chmod(0o700)
