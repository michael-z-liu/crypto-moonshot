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

## Downloading data

`moonshot data sync` downloads Binance spot candles into the local Parquet store (`./data`, or
`$MOONSHOT_DATA_DIR`, or `--data-dir`). It fetches only what is missing, in parallel across
symbols and months, so re-running it resumes an interrupted download. Public market data needs no
API keys. The stored format is described in [docs/data-format.md](docs/data-format.md).

```bash
uv run moonshot data sync BTCUSDT ETHUSDT                      # 5m candles since 2020-01-01
uv run moonshot data sync BTCUSDT SOLUSDT -i 1h -i 1d --start 2022-01-01 --concurrency 4
uv run moonshot data status                                    # what is stored, gaps, bad files
uv run moonshot data status BTCUSDT -i 1h --json
```

Intervals are case sensitive (`1m` is a minute, `1M` a month). Dates are `YYYY-MM-DD` or an ISO
datetime, in UTC unless an offset is given. Progress goes to stderr: live bars with a row per
download when stderr is a terminal, otherwise one plain line per event plus a snapshot every
5 seconds (`--progress bars|lines|none`, `-q` for none, `-v` for debug logs). The summary goes to
stdout.

For scripts and AI agents, ask for machine-readable output and plain progress:

```bash
uv run moonshot data sync BTCUSDT ETHUSDT -i 1h --json --progress lines 2>sync.log
```

stdout is then a single JSON object (`outcome`, `candles_added`, `requests`, and one entry per
symbol and interval with its `status`, `error` and stored range); `sync.log` has the progress.
Exit codes: 0 ok, 1 some downloads failed, 2 usage error, 3 aborted because Binance rate-limited
or banned the IP (the client stays under half of Binance's limit and stops after
`--max-rate-limit-hits` 429 responses), 130 interrupted. Months already written are kept.

To reuse data downloaded in another folder instead of fetching it again, import it:

```bash
uv run moonshot data import ../other-checkout/data --dry-run   # report only, writes nothing
uv run moonshot data import ../other-checkout/data
```

Each month file is checked against the current stored format (path, schema version, columns and
types, UTC, aligned, no duplicates, closed candles) and imported, merged into an existing month
(stored candles win) or skipped with its reasons; nothing is upgraded, so run `data sync` to
download whatever was skipped. The source is never modified. `--json` prints the report; the exit
code is 0 even when files were skipped.

## Backtesting

`moonshot backtest` runs a strategy over the stored candles and writes the results to
`results/<strategy>-<UTC timestamp>/` (gitignored): `trades.parquet` and `trades.csv`,
`equity.csv`, `yearly.csv` and `metrics.json` (metrics, skipped signals, the full configuration
and the data end used). It prints the overall metrics and a table per calendar year. Signals are
cached in `.cache/`, never in the data folder.

```bash
uv run moonshot backtest f2 --until 2026-11-02          # F2: pool, 5 equal slots, 1 hour hold
uv run moonshot backtest capitulation                   # R2: all coins, $50 stakes, +20% or 3 days
uv run moonshot backtest capitulation --fine --tp-fill close   # fill exits on 1m candles
uv run moonshot backtest f2 --slots 10 --hold 90m --stop 0.05 --fee 0.002 --out results/mine
uv run moonshot backtest convert-scores wf/base.npz seq2_meta.parquet scores/base.parquet
uv run moonshot backtest scores --scores scores/base.parquet --threshold 0.98 --json
```

`--slots N` compounds (stake = equity / N), `--stake USDT` keeps it fixed; `--hold` takes `1h`,
`90m`, `3d`, `2w`; `--take-profit`, `--stop` and `--trailing` are fractions (0 switches a preset's
level off); `--until` is the data end (later candles are ignored), `--start`/`--end` limit the
signal range; `--json` prints the `metrics.json` document on stdout. The presets and their
validation against the research results are in [docs/backtest-validation.md](docs/backtest-validation.md);
the design is in [docs/architecture.md](docs/architecture.md).

## Development

```bash
uv run ruff format .     # format
uv run ruff check .      # lint
uv run mypy src tests    # type check
uv run pytest            # test
```

Add dependencies with `uv add <pkg>` (or `uv add --dev <pkg>` for tooling) so `uv.lock` stays in sync.

Commits follow Conventional Commits. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Safety

This repository is public. API keys go in `.env`, which is gitignored. Use the Binance spot testnet until a strategy has been proven.
