"""Testnet mode: the daemon's exchange wiring and its start-up checks.

`check_testnet_config` (pure, before anything is opened) refuses a missing key, a REST or
WebSocket URL that is not the spot testnet (`exchange.guard`) and an exit plan the exchange
cannot hold. `open_exchange` (after the state file is open) builds the signed client, syncs the
clock, starts a `BinanceBroker` with its user data stream and checks the account: the strategies'
symbols must be listed and trading (the others are dropped with a warning; a strategy left with
none refuses the start), the key must be valid and may trade, and the free USDT must cover the
books' cash. A refusal is a `DaemonError`; a message never contains a key, a secret or a
signature.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime
from decimal import Decimal
from typing import Any

import httpx
from pydantic import SecretStr

from crypto_moonshot.broker.binance import BinanceBroker, ExitPlanError, check_exit_plan
from crypto_moonshot.config import Settings
from crypto_moonshot.exchange.client import SignedClient
from crypto_moonshot.exchange.errors import AuthError, ExchangeError
from crypto_moonshot.exchange.guard import MainnetRefused, require_testnet_url
from crypto_moonshot.exchange.rules import ExchangeRules
from crypto_moonshot.exchange.ws_api import StreamMessage, UserDataStream
from crypto_moonshot.live.config import RunConfig
from crypto_moonshot.live.engine import StrategySpec
from crypto_moonshot.live.state import StateStore

logger = logging.getLogger(__name__)

KEY_VARS = ("MOONSHOT_BINANCE_TESTNET_API_KEY", "MOONSHOT_BINANCE_TESTNET_API_SECRET")
KEYS_HELP = (
    "testnet mode needs the Binance spot testnet keys in the environment variables "
    f"{KEY_VARS[0]} and {KEY_VARS[1]} (or in an env file: `--env-file PATH`, "
    "see docs/runbook.md). Create them at https://testnet.binance.vision/"
)


class TestnetError(Exception):
    """A testnet start-up check failed; `usage` marks a configuration problem (exit 2)."""

    __test__ = False  # not a pytest class

    def __init__(self, message: str, *, usage: bool = False) -> None:
        super().__init__(message)
        self.usage = usage


@dataclass
class ExchangeOptions:
    """Test hooks of the exchange wiring: an HTTP client (a mock transport), extra keyword
    arguments of the REST client, the user data stream and the broker."""

    http: httpx.AsyncClient | None = None
    client: dict[str, Any] = field(default_factory=dict)
    stream: dict[str, Any] = field(default_factory=dict)
    broker: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExchangeSession:
    """The open exchange connection of a testnet daemon."""

    client: SignedClient
    broker: BinanceBroker
    specs: list[StrategySpec]
    """The strategies' specs, without the symbols the exchange does not trade."""
    warnings: list[tuple[str, str]] = field(default_factory=list)
    """(event kind, message) to record as warning events once the engine is up."""

    async def aclose(self) -> None:
        await self.broker.aclose()
        await self.client.aclose()


def require_keys(settings: Settings) -> tuple[SecretStr, SecretStr]:
    """The testnet key and secret, or a usage `TestnetError` naming the two variables."""
    key, secret = settings.binance_testnet_api_key, settings.binance_testnet_api_secret
    if (
        key is None
        or secret is None
        or not key.get_secret_value().strip()
        or not secret.get_secret_value().strip()
    ):
        raise TestnetError(KEYS_HELP, usage=True)
    return key, secret


def check_testnet_config(
    config: RunConfig, settings: Settings, specs: list[StrategySpec]
) -> tuple[SecretStr, SecretStr]:
    """Refuse what can be seen without the network; returns the keys."""
    keys = require_keys(settings)
    for url, what in (
        (settings.binance_testnet_rest_url, "MOONSHOT_BINANCE_TESTNET_REST_URL"),
        (settings.binance_testnet_ws_api_url, "MOONSHOT_BINANCE_TESTNET_WS_API_URL"),
    ):
        try:
            require_testnet_url(url, what)
        except MainnetRefused as exc:
            raise TestnetError(str(exc), usage=True) from None
    for spec in specs:
        try:
            check_exit_plan(spec.exit, name=f"strategy {spec.name}: exit plan")
        except ExitPlanError as exc:
            raise TestnetError(str(exc), usage=True) from None
    return keys


def cash_needed(store: StateStore, specs: list[StrategySpec], now: datetime) -> Decimal:
    """The USDT the books hold: each strategy's capital, less what its open positions cost
    (already spent on the exchange), plus its realised result."""
    known = {s.name for s in store.strategies()}
    total = Decimal(0)
    for spec in specs:
        cash = (
            store.derive_book(spec.name, spec.capital, now).cash
            if spec.name in known
            else spec.capital
        )
        total += Decimal(repr(max(cash, 0.0)))
    return total


async def open_exchange(
    config: RunConfig,
    settings: Settings,
    specs: list[StrategySpec],
    store: StateStore,
    *,
    clock: Callable[[], datetime],
    options: ExchangeOptions | None = None,
) -> ExchangeSession:
    """Build the client and the broker, start the stream and run the account checks. Closes
    everything it opened when a check refuses."""
    options = options or ExchangeOptions()
    key, secret = check_testnet_config(config, settings, specs)
    try:
        client = SignedClient.from_settings(
            settings, http=options.http, now=clock, **options.client
        )
    except ValueError as exc:
        raise TestnetError(str(exc), usage=True) from None
    broker: BinanceBroker | None = None
    try:
        await client.sync_time()
        rules = ExchangeRules(client, now=clock)

        def stream(sink: Callable[[StreamMessage], None]) -> UserDataStream:
            return UserDataStream(
                key,
                secret,
                sink,
                url=settings.binance_testnet_ws_api_url,
                now=lambda: clock() + client.time_offset,
                **options.stream,
            )

        broker = BinanceBroker(
            client,
            rules=rules,
            stream=stream,
            clock=clock,
            plans=[s.exit for s in specs],
            **options.broker,
        )
        await rules.refresh()
        kept, warnings = _tradable(specs, rules)
        await _check_account(client, cash_needed(store, kept, clock()))
        await broker.start()
    except ExitPlanError as exc:
        await _close(client, broker)
        raise TestnetError(str(exc), usage=True) from None
    except AuthError:
        await _close(client, broker)
        msg = (
            "the testnet rejected the API key (invalid key, IP restriction or missing "
            f"permission); check {KEY_VARS[0]} and {KEY_VARS[1]}"
        )
        raise TestnetError(msg) from None
    except ExchangeError as exc:
        await _close(client, broker)
        raise TestnetError(f"cannot reach the Binance spot testnet: {exc}") from None
    except BaseException:
        await _close(client, broker)
        raise
    return ExchangeSession(client, broker, kept, warnings)


async def _close(client: SignedClient, broker: BinanceBroker | None) -> None:
    if broker is not None:
        await broker.aclose()
    await client.aclose()


def _tradable(
    specs: list[StrategySpec], rules: ExchangeRules
) -> tuple[list[StrategySpec], list[tuple[str, str]]]:
    """Drop the symbols that are not listed and trading; a strategy left empty is an error."""
    kept: list[StrategySpec] = []
    warnings: list[tuple[str, str]] = []
    for spec in specs:
        usable = [s for s in spec.symbols if (r := rules.peek(s)) is not None and r.is_trading]
        gone = [s for s in spec.symbols if s not in usable]
        if not usable:
            msg = (
                f"strategy {spec.name}: none of its symbols ({', '.join(spec.symbols[:5])}"
                f"{', ...' if len(spec.symbols) > 5 else ''}) is trading on the testnet"
            )
            raise TestnetError(msg, usage=True)
        if gone:
            shown = ", ".join(gone[:10]) + (f" and {len(gone) - 10} more" if len(gone) > 10 else "")
            message = (
                f"strategy {spec.name}: dropped {len(gone)} symbol(s) not trading on the "
                f"testnet: {shown}"
            )
            logger.warning(message)
            warnings.append(("symbols_dropped", message))
            spec = replace(spec, symbols=tuple(usable))
        kept.append(spec)
    return kept, warnings


async def _check_account(client: SignedClient, needed: Decimal) -> None:
    account = await client.account()
    if not account.can_trade:
        msg = "the testnet account is not allowed to trade (canTrade is false)"
        raise TestnetError(msg, usage=True)
    free = account.balance("USDT").free
    if free < needed:
        msg = (
            f"not enough free USDT on the testnet: {free:f} available, the strategies' "
            f"books need {needed:f} (lower `capital` in the config, or get testnet funds)"
        )
        raise TestnetError(msg, usage=True)


__all__ = [
    "KEYS_HELP",
    "KEY_VARS",
    "ExchangeOptions",
    "ExchangeSession",
    "TestnetError",
    "check_testnet_config",
    "open_exchange",
    "require_keys",
]
