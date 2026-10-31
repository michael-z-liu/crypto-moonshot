"""Runtime settings, read from `MOONSHOT_*` environment variables only."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings from the environment (`MOONSHOT_DATA_DIR`, `MOONSHOT_BINANCE_BASE_URL`).

    No `.env` file is read: keys and `.env` loading arrive with the broker work.
    """

    model_config = SettingsConfigDict(env_prefix="MOONSHOT_", env_file=None, extra="ignore")

    data_dir: Path = Path("data")
    binance_base_url: str = "https://data-api.binance.vision"
