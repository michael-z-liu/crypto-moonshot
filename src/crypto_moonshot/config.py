"""Runtime settings, read from `MOONSHOT_*` environment variables only."""

from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings from the environment (`MOONSHOT_DATA_DIR`, `MOONSHOT_BINANCE_BASE_URL`,
    `MOONSHOT_STREAM_URL`, `MOONSHOT_API_TOKEN`, `MOONSHOT_BINANCE_TESTNET_API_KEY`, `_SECRET`,
    `MOONSHOT_BINANCE_TESTNET_REST_URL`, `MOONSHOT_BINANCE_TESTNET_WS_API_URL`).

    No `.env` file is read by default: only the testnet code paths load one, through
    `load_settings(env_file=...)` (paper mode never does). The API token and the testnet keys
    are `SecretStr`s: they are never shown in a repr, log line or error message.
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
    binance_testnet_ws_api_url: str = "wss://ws-api.testnet.binance.vision/ws-api/v3"
    """Spot testnet WebSocket API (user data stream; `MOONSHOT_BINANCE_TESTNET_WS_API_URL`)."""


class EnvFileError(Exception):
    """The `--env-file` cannot be used; the message says why (never a value from the file)."""


def load_settings(*, env_file: Path | None = None, search_cwd: bool = False) -> Settings:
    """`Settings`, with the variables of an env file added for this process only (real
    environment variables win over the file's).

    `env_file` is read when given (`EnvFileError` if it is missing or not a file); otherwise
    `./.env` is read when `search_cwd` is set and the file exists. With neither, no file is read.
    Only testnet code paths should pass either: paper mode never reads a `.env`.
    """
    if env_file is not None:
        if not env_file.is_file():
            msg = f"the env file {env_file} does not exist or is not a file"
            raise EnvFileError(msg)
        return Settings(_env_file=env_file)  # type: ignore[call-arg]
    if search_cwd:
        default = Path.cwd() / ".env"
        if default.is_file():
            return Settings(_env_file=default)  # type: ignore[call-arg]
    return Settings()
