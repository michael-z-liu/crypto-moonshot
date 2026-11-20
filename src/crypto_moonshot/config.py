"""Runtime settings, read from `MOONSHOT_*` environment variables only."""

from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings from the environment (`MOONSHOT_DATA_DIR`, `MOONSHOT_BINANCE_BASE_URL`,
    `MOONSHOT_STREAM_URL`, `MOONSHOT_API_TOKEN`).

    No `.env` file is read: keys and `.env` loading arrive with the broker work. The API token
    is a `SecretStr`: it is never shown in a repr, log line or error message.
    """

    model_config = SettingsConfigDict(env_prefix="MOONSHOT_", env_file=None, extra="ignore")

    data_dir: Path = Path("data")
    binance_base_url: str = "https://data-api.binance.vision"
    stream_url: str = "wss://data-stream.binance.vision"
    """Base URL of the Binance market-data WebSocket host (the client adds `/stream`)."""

    api_token: SecretStr | None = None
    """Bearer token of the daemon's HTTP API (`MOONSHOT_API_TOKEN`); the daemon refuses to start
    without one, and providers and the `live` commands send it."""
