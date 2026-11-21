"""`moonshot exchange check`: a read-only connectivity check of the Binance spot testnet.

It reads the testnet keys (`MOONSHOT_BINANCE_TESTNET_API_KEY` / `_SECRET`, from the environment
or `--env-file`), then asks the testnet for its time (and measures the clock offset), the
account (can it trade, the USDT and base-asset balances), the symbol's trading rules and the
symbol's open orders. It places, cancels and changes nothing.

Exit codes: 0 ok, 1 failure (the key is rejected, the testnet cannot be reached), 2 usage (no
keys, a mainnet URL, an unreadable env file). With `--json` the result (or `{"ok": false,
"error": ...}`) is printed on stdout. No key, secret or signature is ever printed.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlsplit

import httpx
import typer

from crypto_moonshot.config import EnvFileError, Settings, load_settings
from crypto_moonshot.exchange.client import SignedClient
from crypto_moonshot.exchange.errors import AuthError, ExchangeError
from crypto_moonshot.exchange.guard import MainnetRefused, require_testnet_url
from crypto_moonshot.live.testnet import KEY_VARS, TestnetError, require_keys

exchange_app = typer.Typer(
    no_args_is_help=True, help="Check the Binance spot testnet connection (read-only)."
)

EXIT_FAILURE = 1
EXIT_USAGE = 2
OFFSET_WARN = timedelta(seconds=1)
"""A clock offset beyond this is reported as a warning (requests allow 5 s)."""


class CheckFailed(Exception):
    """The check could not finish: `usage` for a configuration problem."""

    def __init__(self, message: str, *, usage: bool = False) -> None:
        super().__init__(message)
        self.usage = usage


def _text(value: Any) -> str:
    return f"{value:f}"


async def run_check(
    settings: Settings,
    symbol: str,
    *,
    http: httpx.AsyncClient | None = None,
    **client_kwargs: Any,
) -> dict[str, Any]:
    """Run the read-only check; the JSON-ready result. Raises `CheckFailed`."""
    try:
        require_keys(settings)
    except TestnetError as exc:
        raise CheckFailed(str(exc), usage=True) from None
    try:
        require_testnet_url(settings.binance_testnet_rest_url, "MOONSHOT_BINANCE_TESTNET_REST_URL")
    except MainnetRefused as exc:
        raise CheckFailed(str(exc), usage=True) from None
    try:
        client = SignedClient.from_settings(settings, http=http, **client_kwargs)
    except ValueError as exc:
        raise CheckFailed(str(exc), usage=True) from None
    try:
        async with client:
            local = datetime.now(UTC)
            offset = await client.sync_time()
            info = await client.exchange_info()
            rules = info.symbols.get(symbol)
            if rules is None:
                msg = f"{symbol} is not listed on the testnet"
                raise CheckFailed(msg)
            account = await client.account()
            open_orders = await client.open_orders(symbol)
    except AuthError:
        msg = (
            "the testnet rejected the API key (invalid key or secret, IP restriction or missing "
            f"permission); check {KEY_VARS[0]} and {KEY_VARS[1]}"
        )
        raise CheckFailed(msg) from None
    except ExchangeError as exc:
        msg = f"cannot reach the Binance spot testnet: {exc}"
        raise CheckFailed(msg) from None
    except httpx.HTTPError as exc:
        msg = f"cannot reach the Binance spot testnet: {type(exc).__name__}"
        raise CheckFailed(msg) from None
    base = account.balance(rules.base_asset)
    usdt = account.balance("USDT")
    host = urlsplit(settings.binance_testnet_rest_url).hostname
    warnings: list[str] = []
    if abs(offset) > OFFSET_WARN:
        warnings.append(
            f"the local clock is {offset.total_seconds():+.1f} s from the exchange's; "
            "fix it (NTP) before trading"
        )
    if not account.can_trade:
        warnings.append("the account cannot trade")
    return {
        "ok": True,
        "host": host,
        "local_time": local.isoformat(),
        "server_time": (local + offset).isoformat(),
        "clock_offset_ms": round(offset / timedelta(milliseconds=1)),
        "account": {
            "can_trade": account.can_trade,
            "usdt_free": _text(usdt.free),
            "usdt_locked": _text(usdt.locked),
            "base_asset": rules.base_asset,
            "base_free": _text(base.free),
            "base_locked": _text(base.locked),
        },
        "symbol": {
            "symbol": rules.symbol,
            "status": rules.status,
            "tick_size": _text(rules.tick_size),
            "lot_step": _text(rules.lot_step),
            "min_qty": _text(rules.min_qty),
            "min_notional": _text(rules.min_notional),
        },
        "open_orders": len(open_orders),
        "warnings": warnings,
    }


def _print(result: dict[str, Any]) -> None:
    account, rules = result["account"], result["symbol"]
    lines = [
        f"testnet {result['host']}: ok (read-only, no orders placed)",
        f"server time   {result['server_time']}  (clock offset {result['clock_offset_ms']:+d} ms)",
        f"account       can trade: {'yes' if account['can_trade'] else 'NO'}",
        f"  USDT        free {account['usdt_free']}  locked {account['usdt_locked']}",
        f"  {account['base_asset']:<11} free {account['base_free']}  "
        f"locked {account['base_locked']}",
        f"{rules['symbol']:<13} {rules['status']}: tick {rules['tick_size']}, "
        f"step {rules['lot_step']}, min qty {rules['min_qty']}, "
        f"min notional {rules['min_notional']}",
        f"open orders   {result['open_orders']} in {rules['symbol']}",
    ]
    typer.echo("\n".join(lines))
    for warning in result["warnings"]:
        typer.echo(f"warning: {warning}", err=True)


@exchange_app.command("check")
def check_command(
    env_file: Annotated[
        Path | None,
        typer.Option(
            "--env-file",
            help="Read the testnet keys from this env file, e.g. "
            "/Users/mike/code/crypto-moonshot/.env (default: ./.env if it exists).",
        ),
    ] = None,
    symbol: Annotated[str, typer.Option(help="The symbol whose rules and orders to check.")] = (
        "BTCUSDT"
    ),
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON on stdout.")] = False,
) -> None:
    """Check the testnet connection and keys without placing any order."""

    def fail(message: str, code: int) -> typer.Exit:
        typer.echo(f"error: {message}", err=True)
        if as_json:
            typer.echo(json.dumps({"ok": False, "error": message}))
        return typer.Exit(code)

    try:
        settings = load_settings(env_file=env_file, search_cwd=True)
    except EnvFileError as exc:
        raise fail(str(exc), EXIT_USAGE) from None
    try:
        result = asyncio.run(run_check(settings, symbol.strip().upper()))
    except CheckFailed as exc:
        raise fail(str(exc), EXIT_USAGE if exc.usage else EXIT_FAILURE) from None
    if as_json:
        typer.echo(json.dumps(result, indent=2))
    else:
        _print(result)
