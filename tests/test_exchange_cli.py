"""`moonshot exchange check` against the fake exchange: read-only, exit codes, JSON."""

from __future__ import annotations

import json
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest
from binance_fakes import REST_URL, Clock, fast_sleep
from fake_exchange import API_KEY, API_SECRET, FakeExchange
from pydantic import SecretStr
from typer.testing import CliRunner, Result

from crypto_moonshot import exchange_cli
from crypto_moonshot.cli import app
from crypto_moonshot.config import Settings
from crypto_moonshot.exchange_cli import CheckFailed, run_check

pytestmark = pytest.mark.anyio

KEY_VARS = {
    "MOONSHOT_BINANCE_TESTNET_API_KEY": API_KEY,
    "MOONSHOT_BINANCE_TESTNET_API_SECRET": API_SECRET,
}


def settings(**extra: Any) -> Settings:
    return Settings(
        binance_testnet_api_key=SecretStr(API_KEY),
        binance_testnet_api_secret=SecretStr(API_SECRET),
        binance_testnet_rest_url=REST_URL,
        **extra,
    )


def client_options(fake: FakeExchange, clock: Clock) -> dict[str, Any]:
    return {
        "http": httpx.AsyncClient(transport=httpx.MockTransport(fake)),
        "now": clock,
        "sleep": fast_sleep,
        "backoff_base": 0.0,
    }


async def test_the_check_reads_time_account_rules_and_orders_and_places_nothing() -> None:
    clock = Clock()
    fake = FakeExchange(clock, usdt=Decimal(5000))
    result = await run_check(settings(), "BTCUSDT", **client_options(fake, clock))
    assert result["ok"] is True
    assert result["host"] == "testnet.invalid"
    assert result["clock_offset_ms"] == 0
    assert result["account"]["can_trade"] is True
    assert Decimal(result["account"]["usdt_free"]) == 5000
    assert result["account"]["base_asset"] == "BTC"
    assert result["symbol"]["symbol"] == "BTCUSDT"
    assert result["symbol"]["status"] == "TRADING"
    assert result["symbol"]["tick_size"] == "0.01000000"
    assert result["symbol"]["lot_step"] == "0.00001000"
    assert result["symbol"]["min_notional"] != "0"
    assert result["open_orders"] == 0
    assert result["warnings"] == []
    # read-only: only GETs, and no order endpoint at all
    assert {r.method for r in fake.requests} == {"GET"}
    assert not any(
        "order" in r.url.path.lower() and "open" not in r.url.path for r in fake.requests
    )
    assert fake.orders == {}
    # the key travels in a header only, never printed
    assert API_SECRET not in json.dumps(result)
    assert API_KEY not in json.dumps(result)


async def test_a_clock_offset_is_reported_with_a_warning() -> None:
    clock = Clock()
    fake = FakeExchange(clock)
    options = client_options(fake, clock)
    server = Clock()
    server.advance(seconds=3)
    fake.clock = server  # the exchange's clock runs 3 s ahead of the client's (within recvWindow)
    result = await run_check(settings(), "BTCUSDT", **options)
    assert result["clock_offset_ms"] == 3000
    assert "clock" in result["warnings"][0]


async def test_a_rejected_key_is_a_failure_that_names_the_variables_not_the_key() -> None:
    clock = Clock()
    fake = FakeExchange(clock, api_key="some-other-fake-key")
    with pytest.raises(CheckFailed, match="rejected the API key") as caught:
        await run_check(settings(), "BTCUSDT", **client_options(fake, clock))
    assert not caught.value.usage
    assert "MOONSHOT_BINANCE_TESTNET_API_KEY" in str(caught.value)
    assert API_KEY not in str(caught.value)
    assert API_SECRET not in str(caught.value)


async def test_an_unlisted_symbol_fails() -> None:
    clock = Clock()
    fake = FakeExchange(clock)
    with pytest.raises(CheckFailed, match="NOPEUSDT is not listed"):
        await run_check(settings(), "NOPEUSDT", **client_options(fake, clock))


async def test_a_server_that_is_down_is_a_failure() -> None:
    clock = Clock()
    fake = FakeExchange(clock)
    fake.fail("/api/v3/time", status=503, times=100)
    with pytest.raises(CheckFailed, match="cannot reach the Binance spot testnet"):
        await run_check(settings(), "BTCUSDT", **client_options(fake, clock), max_attempts=2)


async def test_no_keys_and_a_mainnet_url_are_usage_errors() -> None:
    with pytest.raises(CheckFailed, match="MOONSHOT_BINANCE_TESTNET_API_KEY") as caught:
        await run_check(Settings(), "BTCUSDT")
    assert caught.value.usage
    mainnet = settings().model_copy(update={"binance_testnet_rest_url": "https://api.binance.com"})
    with pytest.raises(CheckFailed, match="refusing the MOONSHOT_BINANCE_TESTNET_REST_URL") as c2:
        await run_check(mainnet, "BTCUSDT")
    assert c2.value.usage


# -- the command --------------------------------------------------------------------------------


def invoke(*args: str, env: dict[str, str] | None = None) -> Result:
    return CliRunner().invoke(app, list(args), env=env)


@pytest.fixture
def fake_testnet(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Callable[[], FakeExchange]:
    """Point the command at a fake exchange (no network) and run it in an empty folder."""
    monkeypatch.chdir(tmp_path)
    clock = Clock()
    fake = FakeExchange(clock)
    real = run_check

    async def patched(settings: Settings, symbol: str) -> dict[str, Any]:
        return await real(settings, symbol, **client_options(fake, clock))

    monkeypatch.setattr(exchange_cli, "run_check", patched)
    return lambda: fake


def env_file(path: Path, **override: str) -> Path:
    values = {**KEY_VARS, **override}
    path.write_text("".join(f"{k}={v}\n" for k, v in values.items()))
    return path


def test_the_command_prints_a_summary_and_exits_0(
    fake_testnet: Callable[[], FakeExchange], tmp_path: Path
) -> None:
    keys = env_file(tmp_path / "keys.env")
    result = invoke("exchange", "check", "--env-file", str(keys))
    assert result.exit_code == 0, result.output
    assert "ok (read-only, no orders placed)" in result.stdout
    assert "min notional" in result.stdout
    assert "BTCUSDT" in result.stdout
    assert API_KEY not in result.output
    assert API_SECRET not in result.output
    assert {r.method for r in fake_testnet().requests} == {"GET"}


def test_json_output_and_a_symbol_option(
    fake_testnet: Callable[[], FakeExchange], tmp_path: Path
) -> None:
    keys = env_file(tmp_path / "keys.env")
    result = invoke("exchange", "check", "--env-file", str(keys), "--symbol", "ethusdt", "--json")
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["ok"] is True
    assert data["symbol"]["symbol"] == "ETHUSDT"
    assert data["account"]["base_asset"] == "ETH"


def test_the_default_env_file_is_dot_env_in_the_current_folder(
    fake_testnet: Callable[[], FakeExchange], tmp_path: Path
) -> None:
    env_file(tmp_path / ".env")
    assert invoke("exchange", "check").exit_code == 0


def test_a_rejected_key_exits_1_with_json_error(
    fake_testnet: Callable[[], FakeExchange], tmp_path: Path
) -> None:
    keys = env_file(tmp_path / "keys.env", MOONSHOT_BINANCE_TESTNET_API_KEY="wrong-fake-key")
    result = invoke("exchange", "check", "--env-file", str(keys), "--json")
    assert result.exit_code == 1
    assert json.loads(result.stdout) == {
        "ok": False,
        "error": result.stderr.split(": ", 1)[1].strip(),
    }
    assert "rejected the API key" in result.stderr
    assert "wrong-fake-key" not in result.output


def test_usage_errors_exit_2(fake_testnet: Callable[[], FakeExchange], tmp_path: Path) -> None:
    no_keys = invoke("exchange", "check")
    assert no_keys.exit_code == 2
    assert "MOONSHOT_BINANCE_TESTNET_API_KEY" in no_keys.stderr
    assert "MOONSHOT_BINANCE_TESTNET_API_SECRET" in no_keys.stderr
    missing = invoke("exchange", "check", "--env-file", str(tmp_path / "nope.env"))
    assert missing.exit_code == 2
    assert "does not exist" in missing.stderr
    # keys under the mainnet variable names do not count
    wrong = invoke(
        "exchange",
        "check",
        env={"MOONSHOT_BINANCE_API_KEY": API_KEY, "MOONSHOT_BINANCE_API_SECRET": API_SECRET},
    )
    assert wrong.exit_code == 2
    assert API_KEY not in wrong.output
    bad_option = invoke("exchange", "check", "--bogus")
    assert bad_option.exit_code == 2


def test_real_environment_variables_win_over_the_env_file(
    fake_testnet: Callable[[], FakeExchange], tmp_path: Path
) -> None:
    keys = env_file(tmp_path / "keys.env", MOONSHOT_BINANCE_TESTNET_API_KEY="from-the-file")
    result = invoke("exchange", "check", "--env-file", str(keys), env=KEY_VARS)
    assert result.exit_code == 0, result.output


def test_the_command_is_wired_into_the_cli() -> None:
    result = invoke("exchange", "--help")
    assert result.exit_code == 0
    assert "check" in result.stdout
