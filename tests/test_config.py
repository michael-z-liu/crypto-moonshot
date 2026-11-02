from pathlib import Path

import pytest

from crypto_moonshot.config import Settings


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MOONSHOT_DATA_DIR", raising=False)
    monkeypatch.delenv("MOONSHOT_BINANCE_BASE_URL", raising=False)
    settings = Settings()
    assert settings.data_dir == Path("data")
    assert settings.binance_base_url == "https://data-api.binance.vision"


def test_environment_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOONSHOT_DATA_DIR", "/tmp/elsewhere")
    monkeypatch.setenv("MOONSHOT_BINANCE_BASE_URL", "http://localhost:9999")
    settings = Settings()
    assert settings.data_dir == Path("/tmp/elsewhere")
    assert settings.binance_base_url == "http://localhost:9999"


def test_dotenv_file_is_not_read(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("MOONSHOT_DATA_DIR", raising=False)
    (tmp_path / ".env").write_text("MOONSHOT_DATA_DIR=from-dotenv\n")
    monkeypatch.chdir(tmp_path)
    assert Settings().data_dir == Path("data")
