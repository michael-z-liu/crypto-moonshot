# crypto-moonshot

Personal toolkit for researching, backtesting and trading crypto on Binance.

Planned scope:

- **Research**: download historical data from Binance and analyse strategies.
- **Trading**: a small CLI that runs those strategies live against Binance.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pre-commit install
cp .env.example .env   # add Binance API keys when needed
uv run moonshot --help
```

## Development

```bash
uv run ruff format .     # format
uv run ruff check .      # lint
uv run mypy src tests    # type check
uv run pytest            # test
```

Add dependencies with `uv add <pkg>` (or `uv add --dev <pkg>` for tooling) so `uv.lock` stays in sync.

## Safety

This repository is public. API keys go in `.env`, which is gitignored. Use the Binance spot testnet until a strategy has been proven.
