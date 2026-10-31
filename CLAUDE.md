# crypto-moonshot

Personal Binance research, backtesting and live-trading toolkit. Python 3.13, managed with uv.

## Commands

- Install: `uv sync`
- Run CLI: `uv run moonshot`
- Check everything (same as CI): `uv run ruff format --check . && uv run ruff check . && uv run mypy src tests && uv run pytest`

## Conventions

- Source lives in `src/crypto_moonshot/`, tests in `tests/`. CLI entry point is `crypto_moonshot.cli:app` (Typer).
- `docs/architecture.md` describes the current design. It is a starting point, not a fixed spec: when needs change, change the design and update the doc in the same PR.
- Manage dependencies only through `uv add` / `uv remove`; never edit `uv.lock` by hand.
- mypy runs in strict mode; type every function.
- Datetimes are always timezone-aware UTC (ruff `DTZ` rules enforce this).
- Secrets come from environment variables or `.env` (gitignored). The repo is public: never commit keys, and never log them.
- Default to the Binance spot testnet for anything that places orders.
- Commits and PR titles use Conventional Commits (`type(scope): summary`); see `CONTRIBUTING.md` for types, scopes and branch naming. Squash-merge PRs.

## Agentic workflow

- `/project <what to build>` scopes the work with the user, asks for one final confirmation, then builds it unattended milestone by milestone and opens a PR. The master's instructions are in `.claude/skills/project/SKILL.md`; the worker subagent is `.claude/agents/implementer.md`.
- Start that session on Opus and in auto mode, so the build doesn't stall on permission prompts. `.claude/settings.json` blocks reading `.env`, mainnet runs, force pushes, pushes to `main` and merging PRs.
- Each project's plan and progress live in `docs/plans/<name>.md`, committed on its branch. A heartbeat in the session resumes the build after a usage limit; if the session itself is gone, run `/project resume` in a new session on the same branch.
