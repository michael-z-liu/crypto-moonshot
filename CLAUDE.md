# crypto-moonshot

Personal Binance research, backtesting and live-trading toolkit. Python 3.13, managed with uv.

## Commands

- Install: `uv sync`
- Run CLI: `uv run moonshot`
- Check everything (same as CI): `uv run ruff format --check . && uv run ruff check . && uv run mypy src tests && uv run pytest`

## Conventions

- Source lives in `src/crypto_moonshot/`, tests in `tests/`. CLI entry point is `crypto_moonshot.cli:app` (Typer).
- Manage dependencies only through `uv add` / `uv remove`; never edit `uv.lock` by hand.
- mypy runs in strict mode; type every function.
- Datetimes are always timezone-aware UTC (ruff `DTZ` rules enforce this).
- Secrets come from environment variables or `.env` (gitignored). The repo is public: never commit keys, and never log them.
- Default to the Binance spot testnet for anything that places orders.
