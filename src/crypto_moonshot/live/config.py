"""The run config of the trading daemon: a TOML file, validated with pydantic.

It names the mode, the state file, the account (paper cash, account-wide limits, trading
costs), the engine's timing and each strategy instance (see `docs/architecture.md`, "Run
config"). Secrets never go in it. Durations are strings (`10s`, `90m`, `3d`;
`models.parse_duration`). A relative `state` path is relative to the config file's folder.

```toml
mode = "paper"
state = "state/paper.sqlite"

[account]
initial_cash = 10000
max_order_usdt = 500
max_total_exposure = 1.0
stale_after = 2
costs = { fee = 0.001, slippage = 0.0005 }

[engine]
tick_timeout = "10s"
signal_deadline = "30s"
entry_grace = "5m"

[[strategies]]
name = "r2"
source = "crypto_moonshot.strategy.capitulation:R2"
symbols = "all"
capital = 2000
sizing = { stake = 50 }
risk = { max_positions = 200 }
exit = { take_profit = 0.2, max_hold = "3d" }
```

Only `paper` runs in this build: `testnet` and `live` parse, and `RunConfig.require_runnable`
refuses them with a clear message. Every validation error names its field
(`strategies[1] (cnn).warmup: ...`).
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from datetime import timedelta
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from crypto_moonshot.broker.base import Costs, ExitPlan
from crypto_moonshot.data.universe import Universe, universe_symbols
from crypto_moonshot.live.contract import ExitPlanIn
from crypto_moonshot.models import Interval, parse_duration
from crypto_moonshot.portfolio import EqualSlots, FixedStake, Sizing, WeightedSlots
from crypto_moonshot.risk import RiskCheck, standard_checks

EXTERNAL = "external"
"""The `source` of a strategy whose batches an external provider posts."""

Mode = Literal["paper", "testnet", "live"]
RUNNABLE_MODES: tuple[Mode, ...] = ("paper",)


class ConfigError(ValueError):
    """The run config cannot be used; the message names the field."""


def _duration(value: object) -> object:
    if isinstance(value, timedelta):
        return value
    if not isinstance(value, str):
        msg = "a duration is a string such as '10s', '90m' or '3d'"
        raise ValueError(msg)
    parsed = parse_duration(value)
    if parsed is None:
        msg = "a duration is required here ('none' is not allowed)"
        raise ValueError(msg)
    return parsed


Duration = Annotated[timedelta, BeforeValidator(_duration)]


def _optional_duration(value: object) -> object:
    if isinstance(value, str) and parse_duration(value) is None:
        return None
    return _duration(value)


OptionalDuration = Annotated[timedelta | None, BeforeValidator(_optional_duration)]
"""A duration, or `'none'` (None)."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CostsConfig(_Model):
    """Trading costs of the simulated exchange (one fee schedule for the whole account)."""

    fee: float = Field(default=0.001, ge=0, lt=1)
    slippage: float = Field(default=0.0005, ge=0, lt=1)
    latency: float = Field(default=0.0, ge=0, lt=1)

    def to_costs(self) -> Costs:
        return Costs(self.fee, self.slippage, self.latency)


class AccountConfig(_Model):
    """The account: paper cash and the limits checked across every strategy's book.

    - `max_order_usdt`: an entry's stake above this is refused.
    - `max_total_exposure`: the open cost basis of every book, the new stake included, at most
      this fraction of the books' total realised equity.
    - `stale_after`: no entries in a symbol whose latest candle is more than this many
      intervals old.
    """

    initial_cash: float = Field(gt=0)
    max_order_usdt: float | None = Field(default=None, gt=0)
    max_total_exposure: float | None = Field(default=None, gt=0)
    stale_after: int = Field(default=2, ge=1)
    costs: CostsConfig = CostsConfig()


class EngineConfig(_Model):
    """The engine's timing.

    - `tick_timeout`: the feed waits this long after a close for every symbol's candle.
    - `signal_deadline`: after a candle closes, wait this long for every strategy's batch.
    - `entry_grace`: a pending buy whose entry candle has not arrived this long after that
      candle should have closed is cancelled (a late candle may still fill it before then).
    - `replay_timeout`: in replay, the longest real time to wait for a candle's batches (replay
      has no wall-clock deadline: it waits for every strategy).
    - `drain_interval`: exchange brokers only: between ticks, record the broker's fills this
      often (`'none'` turns it off; the paper broker is never drained between ticks).
    """

    tick_timeout: Duration = timedelta(seconds=10)
    signal_deadline: Duration = timedelta(seconds=30)
    entry_grace: Duration = timedelta(minutes=5)
    replay_timeout: Duration = timedelta(hours=1)
    drain_interval: OptionalDuration = timedelta(seconds=5)


class HealthConfig(_Model):
    """Thresholds of the health rules, the watchdog and the backups (`live/health.py`).

    - `health_every`: how often the rules are evaluated.
    - `feed_grace`: a feed connection down longer than this raises `feed_disconnected`.
    - `no_ticks_intervals`: `no_ticks` fires after this many intervals plus `tick_timeout`
      without a processed tick.
    - `provider_misses`: `provider_missing` / `plugin_failing` fire after this many batches in a
      row that were missed (or, for an external provider, late).
    - `backfill_window`: `backfill_failed` fires while a seed or backfill request failed within it.
    - `stale_fraction`: `stale_symbols` fires when more than this share of an interval's traded
      symbols is stale (after `startup_grace` from the start, while the feed seeds).
    - `min_free_gb`: `disk_low` fires below this much free disk where the state file lives.
    - `watchdog_after`: exit with code 3 when no tick was processed for this long while the feed
      reports messages (default: 10 x the shortest interval, at least 15 minutes).
    - `loop_blocked_after`: exit with code 3 (no clean shutdown) when the event loop did not run
      for this long.
    - `backup_every`, `backup_keep`: the automatic state backups (at start, then every
      `backup_every`; the newest `backup_keep` are kept).
    """

    health_every: Duration = timedelta(seconds=15)
    feed_grace: Duration = timedelta(seconds=60)
    no_ticks_intervals: int = Field(default=2, ge=1)
    provider_misses: int = Field(default=3, ge=1)
    backfill_window: Duration = timedelta(hours=1)
    stale_fraction: float = Field(default=0.2, ge=0, le=1)
    startup_grace: Duration = timedelta(minutes=2)
    min_free_gb: float = Field(default=1.0, ge=0)
    watchdog_after: Duration | None = None
    loop_blocked_after: Duration = timedelta(seconds=60)
    backup_every: Duration = timedelta(hours=24)
    backup_keep: int = Field(default=14, ge=1)

    def watchdog_threshold(self, shortest: timedelta) -> timedelta:
        """`watchdog_after`, or 10 x the shortest interval (at least 15 minutes)."""
        if self.watchdog_after is not None:
            return self.watchdog_after
        return max(timedelta(minutes=15), 10 * shortest)


class SizingConfig(_Model):
    """Exactly one of `stake` (fixed USDT), `slots` (equal slots, compounding) or
    `weighted_slots` (equal slots scaled by each signal's weight, capped at the cash)."""

    stake: float | None = Field(default=None, gt=0)
    slots: int | None = Field(default=None, ge=1)
    weighted_slots: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _exactly_one(self) -> Self:
        given = [n for n in ("stake", "slots", "weighted_slots") if getattr(self, n) is not None]
        if len(given) != 1:
            msg = "give exactly one of stake, slots or weighted_slots"
            raise ValueError(msg)
        return self

    def to_sizing(self) -> Sizing:
        if self.stake is not None:
            return FixedStake(self.stake)
        if self.slots is not None:
            return EqualSlots(self.slots)
        assert self.weighted_slots is not None
        return WeightedSlots(self.weighted_slots)


class RiskConfig(_Model):
    """The strategy book's risk checks (`risk.py`); unset ones are off. One position per
    symbol always applies."""

    max_positions: int | None = Field(default=None, ge=1)
    max_exposure: float | None = Field(default=None, gt=0)
    daily_loss_limit: float | None = Field(default=None, gt=0)
    max_entries_per_day: int | None = Field(default=None, ge=1)

    def checks(self) -> list[RiskCheck]:
        return standard_checks(
            max_positions=self.max_positions,
            max_exposure=self.max_exposure,
            daily_loss_limit=self.daily_loss_limit,
            max_entries_per_day=self.max_entries_per_day,
        )


class StrategyConfig(_Model):
    """One strategy instance.

    `source` is `package.module:attr`, `path/to/file.py:attr` or `"external"` (an external
    provider posts its batches; then `interval` and `warmup` are required, and
    `context_symbols` names the extra symbols the provider reads). `symbols` is a
    universe name (`pool`, `all`) or a list. `exit` overrides fields of the plugin's exit plan.
    For an in-process plugin, `interval` must match the plugin's if given and `warmup`
    replaces the plugin's window length.
    """

    name: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,64}$")
    source: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)
    symbols: str | list[str]
    capital: float = Field(gt=0)
    sizing: SizingConfig
    risk: RiskConfig = RiskConfig()
    exit: ExitPlanIn | None = None
    interval: Interval | None = None
    warmup: int | None = Field(default=None, ge=1)
    context_symbols: list[str] = Field(default_factory=list)
    """External strategies only: extra symbols whose windows the provider reads (e.g. BTCUSDT);
    the daemon subscribes them. An in-process plugin declares its own."""

    @property
    def external(self) -> bool:
        return self.source == EXTERNAL

    @field_validator("symbols")
    @classmethod
    def _symbols(cls, value: str | list[str]) -> str | list[str]:
        if isinstance(value, str):
            if value.lower() not in (Universe.POOL.value, Universe.ALL.value):
                msg = "symbols must be 'pool', 'all' or a list of symbols"
                raise ValueError(msg)
            return value.lower()
        symbols = list(dict.fromkeys(s.strip().upper() for s in value if s.strip()))
        if not symbols:
            msg = "the symbol list is empty"
            raise ValueError(msg)
        bad = [s for s in symbols if not s.endswith("USDT") or s == "USDT"]
        if bad:
            msg = f"only USDT-quoted symbols are supported, got {bad[0]}"
            raise ValueError(msg)
        return symbols

    @field_validator("context_symbols")
    @classmethod
    def _context(cls, value: list[str]) -> list[str]:
        symbols = list(dict.fromkeys(s.strip().upper() for s in value if s.strip()))
        bad = [s for s in symbols if not s.endswith("USDT") or s == "USDT"]
        if bad:
            msg = f"only USDT-quoted symbols are supported, got {bad[0]}"
            raise ValueError(msg)
        return symbols

    @field_validator("interval")
    @classmethod
    def _fixed_length(cls, value: Interval | None) -> Interval | None:
        if value is not None and value.is_monthly:
            msg = "1M candles are not supported"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _external(self) -> Self:
        if self.external:
            missing = [n for n in ("interval", "warmup") if getattr(self, n) is None]
            if missing:
                msg = f"an external strategy needs {' and '.join(missing)}"
                raise ValueError(msg)
            if self.params:
                msg = "params are for in-process plugins; an external strategy takes none"
                raise ValueError(msg)
        elif self.context_symbols:
            msg = "context_symbols are for external strategies; a plugin declares its own"
            raise ValueError(msg)
        return self

    def exit_plan(self, base: ExitPlan | None) -> ExitPlan | None:
        """The plugin's plan `base` with this config's overrides; `None` when no exit is set."""
        plan = base if self.exit is None else self.exit.to_plan(base)
        if plan is None or plan == ExitPlan(tp_fill=plan.tp_fill):
            return None
        return plan

    def resolve_symbols(self, root: Path | None, interval: Interval) -> list[str]:
        """The traded symbols: the list, or the universe's symbols stored at `interval` in the
        data root (`ConfigError` without one, or when a universe list is missing)."""
        if isinstance(self.symbols, list):
            return list(self.symbols)
        if root is None:
            msg = f"strategy {self.name}: symbols = {self.symbols!r} needs a data folder"
            raise ConfigError(msg)
        try:
            symbols = universe_symbols(Universe(self.symbols), root, interval=interval)
        except FileNotFoundError as exc:
            msg = f"strategy {self.name}: {exc.filename} not found (the {self.symbols} universe)"
            raise ConfigError(msg) from None
        if not symbols:
            msg = f"strategy {self.name}: no {interval.value} candles in {root} for {self.symbols}"
            raise ConfigError(msg)
        return symbols


class RunConfig(_Model):
    """A whole run config; see the module docstring."""

    mode: Mode = "paper"
    state: Path
    account: AccountConfig
    engine: EngineConfig = EngineConfig()
    health: HealthConfig = HealthConfig()
    strategies: list[StrategyConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        seen: set[str] = set()
        for strategy in self.strategies:
            if strategy.name in seen:
                msg = f"strategy name {strategy.name!r} is used twice; names must be unique"
                raise ValueError(msg)
            seen.add(strategy.name)
        total = sum(s.capital for s in self.strategies)
        if total > self.account.initial_cash * (1 + 1e-12):
            msg = (
                f"the strategies' capital ({total:g}) exceeds the account's initial_cash "
                f"({self.account.initial_cash:g})"
            )
            raise ValueError(msg)
        return self

    def require_runnable(self) -> None:
        """Raise `ConfigError` unless this build can run the config's mode (paper only)."""
        if self.mode not in RUNNABLE_MODES:
            msg = (
                f"mode {self.mode!r} is not available in this build: only 'paper' runs "
                "(no exchange orders are placed). Testnet and live trading come later."
            )
            raise ConfigError(msg)

    def strategy(self, name: str) -> StrategyConfig:
        for strategy in self.strategies:
            if strategy.name == name:
                return strategy
        raise KeyError(name)


def _where(loc: tuple[int | str, ...], data: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for i, part in enumerate(loc):
        if isinstance(part, int):
            parts[-1] = f"{parts[-1]}[{part}]"
            if loc[:i] == ("strategies",):
                strategies = data.get("strategies")
                if isinstance(strategies, list) and part < len(strategies):
                    entry = strategies[part]
                    if isinstance(entry, dict) and isinstance(entry.get("name"), str):
                        parts[-1] += f" ({entry['name']})"
        elif part not in ("function-after", "str", "list[str]") and not part.startswith(
            ("function-", "constrained-")
        ):
            parts.append(part)
    return ".".join(parts) or "config"


def config_from_dict(data: Mapping[str, Any], *, base_dir: Path | None = None) -> RunConfig:
    """Validate a parsed config; a relative `state` is taken relative to `base_dir`."""
    try:
        config = RunConfig.model_validate(data)
    except ValidationError as exc:
        lines = [
            f"{_where(e['loc'], data)}: {e['msg'].removeprefix('Value error, ')}"
            for e in exc.errors()
        ]
        raise ConfigError("invalid run config:\n  " + "\n  ".join(lines)) from None
    if base_dir is not None and not config.state.is_absolute():
        config = config.model_copy(update={"state": base_dir / config.state})
    return config


def parse_config(text: str, *, base_dir: Path | None = None) -> RunConfig:
    """Parse and validate TOML text (see `config_from_dict`)."""
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        msg = f"invalid TOML: {exc}"
        raise ConfigError(msg) from None
    return config_from_dict(data, base_dir=base_dir)


def load_config(path: Path) -> RunConfig:
    """Read a run config file; a relative `state` path is relative to the file's folder."""
    try:
        text = path.read_text("utf-8")
    except OSError as exc:
        msg = f"cannot read the run config {path}: {exc.strerror or exc}"
        raise ConfigError(msg) from None
    return parse_config(text, base_dir=path.parent)
