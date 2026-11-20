"""Runtime settings, read from `MOONSHOT_*` environment variables only."""

from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings from the environment (`MOONSHOT_DATA_DIR`, `MOONSHOT_BINANCE_BASE_URL`,
    `MOONSHOT_STREAM_URL`, `MOONSHOT_API_TOKEN`, `MOONSHOT_BINANCE_TESTNET_API_KEY`, `_SECRET`,
    `MOONSHOT_BINANCE_TESTNET_REST_URL`).

    No `.env` file is read (testnet mode decides how to load one). The API token and the
    testnet keys are `SecretStr`s: they are never shown in a repr, log line or error message.
    """

    model_config = SettingsConfigDict(env_prefix="MOONSHOT_", env_file=None, extra="ignore")

    data_dir: Path = Path("data")
    binance_base_url: str = "https://data-api.binance.vision"
    stream_url: str = "wss://data-stream.binance.vision"
    """Base URL of the Binance market-data WebSocket host (the client adds `/stream`)."""

    api_token: SecretStr | None = None
    """Bearer token of the daemon's HTTP API (`MOONSHOT_API_TOKEN`); the daemon refuses to start
    without one, and providers and the `live` commands send it."""

    binance_testnet_api_key: SecretStr | None = None
    """Binance spot testnet API key (`MOONSHOT_BINANCE_TESTNET_API_KEY`). Mainnet keys will use
    other names (`MOONSHOT_BINANCE_API_KEY`), so the two cannot be mixed up."""
    binance_testnet_api_secret: SecretStr | None = None
    """Binance spot testnet API secret (`MOONSHOT_BINANCE_TESTNET_API_SECRET`)."""
    binance_testnet_rest_url: str = "https://testnet.binance.vision"
    """Base URL of the Binance spot testnet REST API (`MOONSHOT_BINANCE_TESTNET_REST_URL`)."""
