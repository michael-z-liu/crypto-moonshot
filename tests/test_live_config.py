"""The TOML run config: parsing, validation messages and the paper-only mode rule."""

from __future__ import annotations

import re
from datetime import timedelta
from pathlib import Path

import pytest
from broker_fakes import T0
from store_fakes import kline_frame, write_series

from crypto_moonshot.broker.base import Costs, ExitPlan
from crypto_moonshot.live.config import ConfigError, load_config, parse_config
from crypto_moonshot.models import Interval
from crypto_moonshot.portfolio import EqualSlots, FixedStake, WeightedSlots
from crypto_moonshot.risk import DailyLossLimit, MaxPositions, OnePerSymbol

GOOD = """
mode = "paper"
state = "state/paper.sqlite"

[account]
initial_cash = 10000
max_order_usdt = 500
max_total_exposure = 0.9
stale_after = 3
costs = { fee = 0.002 }

[engine]
tick_timeout = "5s"
signal_deadline = "20s"
entry_grace = "10m"

[[strategies]]
name = "r2"
source = "crypto_moonshot.strategy.capitulation:R2"
params = {}
symbols = "all"
capital = 2000
sizing = { stake = 50 }
risk = { max_positions = 200, daily_loss_limit = 0.1 }
exit = { take_profit = 0.3, max_hold = "2d" }

[[strategies]]
name = "cnn"
source = "external"
interval = "5m"
symbols = ["solusdt", "BTCUSDT", "SOLUSDT"]
warmup = 2016
capital = 3000
sizing = { weighted_slots = 5 }
exit = { max_hold = "2h" }
"""


def test_a_good_file_parses_with_everything_typed(tmp_path: Path) -> None:
    path = tmp_path / "run.toml"
    path.write_text(GOOD)
    config = load_config(path)
    assert config.mode == "paper"
    config.require_runnable()
    assert config.state == tmp_path / "state/paper.sqlite"
    assert config.account.initial_cash == 10_000
    assert config.account.max_order_usdt == 500
    assert config.account.stale_after == 3
    assert config.account.costs.to_costs() == Costs(fee=0.002)
    assert config.engine.tick_timeout == timedelta(seconds=5)
    assert config.engine.signal_deadline == timedelta(seconds=20)
    assert config.engine.entry_grace == timedelta(minutes=10)
    assert config.engine.replay_timeout == timedelta(hours=1)

    r2, cnn = config.strategies
    assert not r2.external
    assert r2.symbols == "all"
    assert r2.sizing.to_sizing() == FixedStake(50)
    checks = r2.risk.checks()
    assert [type(c) for c in checks] == [DailyLossLimit, MaxPositions, OnePerSymbol]
    base = ExitPlan(take_profit=0.2, max_hold=timedelta(days=3))
    assert r2.exit_plan(base) == ExitPlan(take_profit=0.3, max_hold=timedelta(days=2))

    assert cnn.external
    assert cnn.interval is Interval.M5
    assert cnn.warmup == 2016
    assert cnn.symbols == ["SOLUSDT", "BTCUSDT"]
    assert cnn.sizing.to_sizing() == WeightedSlots(5)
    assert cnn.exit_plan(None) == ExitPlan(max_hold=timedelta(hours=2))
    assert config.strategy("cnn") is cnn


def test_defaults_and_an_absolute_state_path(tmp_path: Path) -> None:
    config = parse_config(
        f"""
state = "{tmp_path / "s.sqlite"}"
account = {{ initial_cash = 100 }}
[[strategies]]
name = "a"
source = "x:Y"
symbols = ["AAAUSDT"]
capital = 100
sizing = {{ slots = 2 }}
""",
        base_dir=Path("/elsewhere"),
    )
    assert config.state == tmp_path / "s.sqlite"
    assert config.mode == "paper"
    assert config.account.max_order_usdt is None
    assert config.account.stale_after == 2
    assert config.engine.signal_deadline == timedelta(seconds=30)
    strategy = config.strategies[0]
    assert strategy.sizing.to_sizing() == EqualSlots(2)
    assert [type(c) for c in strategy.risk.checks()] == [OnePerSymbol]
    assert strategy.exit_plan(None) is None
    assert strategy.exit_plan(ExitPlan()) is None


def test_live_parses_but_is_refused_and_testnet_runs_but_not_as_a_replay() -> None:
    live = parse_config(GOOD.replace('mode = "paper"', 'mode = "live"'))
    assert live.mode == "live"
    with pytest.raises(ConfigError, match="mode 'live' is not available.*only 'paper' and"):
        live.require_runnable()
    testnet = parse_config(GOOD.replace('mode = "paper"', 'mode = "testnet"'))
    testnet.require_runnable()
    with pytest.raises(ConfigError, match="a replay is paper-only"):
        testnet.require_runnable(replay=True)
    parse_config(GOOD).require_runnable(replay=True)


def test_initial_cash_is_required_for_paper_and_ignored_for_testnet() -> None:
    no_cash = GOOD.replace("initial_cash = 10000\n", "")
    assert no_cash != GOOD
    with pytest.raises(ConfigError, match="account.initial_cash is required"):
        parse_config(no_cash)
    config = parse_config(no_cash.replace('mode = "paper"', 'mode = "testnet"'))
    assert config.account.initial_cash is None
    # the paper check of the capital against the cash does not apply on an exchange
    big = GOOD.replace('mode = "paper"', 'mode = "testnet"').replace(
        "initial_cash = 10000", "initial_cash = 1"
    )
    assert parse_config(big).mode == "testnet"


def _swap(old: str, new: str) -> str:
    assert old in GOOD
    return GOOD.replace(old, new, 1)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (_swap('mode = "paper"', 'mode = "demo"'), r"mode: Input should be 'paper'"),
        (_swap('state = "state/paper.sqlite"\n', ""), r"state: Field required"),
        (_swap("initial_cash = 10000", "initial_cash = 0"), r"account.initial_cash: .*greater"),
        (_swap("max_order_usdt = 500", "max_order_usdt = -1"), r"account.max_order_usdt"),
        (_swap("stale_after = 3", "stale_after = 0"), r"account.stale_after"),
        (_swap("costs = { fee = 0.002 }", "costs = { fees = 1 }"), r"account.costs.fees: Extra"),
        (_swap('tick_timeout = "5s"', 'tick_timeout = "soon"'), r"engine.tick_timeout: .*dur"),
        (_swap('signal_deadline = "20s"', "signal_deadline = 20"), r"engine.signal_deadline"),
        (_swap('entry_grace = "10m"', 'entry_grace = "none"'), r"engine.entry_grace: .*requi"),
        (_swap('name = "cnn"', 'name = "r2"'), r"used twice"),
        (_swap('name = "cnn"', 'name = "c n n"'), r"strategies\[1\] \(c n n\).name"),
        (_swap('symbols = "all"', 'symbols = "most"'), r"strategies\[0\] \(r2\).symbols"),
        (_swap('symbols = "all"', "symbols = []"), r"strategies\[0\] \(r2\).symbols: .*empty"),
        (_swap('symbols = "all"', 'symbols = ["BTCEUR"]'), r"USDT-quoted"),
        (_swap("sizing = { stake = 50 }", "sizing = {}"), r"\(r2\).sizing: .*exactly one"),
        (
            _swap("sizing = { stake = 50 }", "sizing = { stake = 50, slots = 2 }"),
            r"\(r2\).sizing: .*exactly one",
        ),
        (_swap("capital = 2000", "capital = 9000"), r"exceeds the account's initial_cash"),
        (_swap("max_positions = 200", "max_positions = 0"), r"\(r2\).risk.max_positions"),
        (_swap('max_hold = "2d"', 'max_hold = "2 days"'), r"\(r2\).exit"),
        (_swap("take_profit = 0.3", "take_profit = -1"), r"\(r2\).exit"),
        (_swap("warmup = 2016\n", ""), r"\(cnn\): .*external strategy needs warmup"),
        (_swap('interval = "5m"\n', ""), r"external strategy needs interval"),
        (_swap('interval = "5m"', 'interval = "7m"'), r"\(cnn\).interval"),
        (_swap('interval = "5m"', 'interval = "1M"'), r"1M candles are not supported"),
        (_swap('source = "external"', 'source = "external"\nparams = {a = 1}'), r"takes none"),
        (_swap("capital = 3000", "capital = 3000\nleverage = 2"), r"\(cnn\).leverage: Extra"),
        ("state = ", r"invalid TOML"),
    ],
)
def test_each_error_names_its_field(text: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        parse_config(text)


def test_a_missing_file_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read the run config"):
        load_config(tmp_path / "nope.toml")


def test_universes_resolve_against_the_data_root(tmp_path: Path) -> None:
    config = parse_config(GOOD)
    r2, cnn = config.strategies
    assert cnn.resolve_symbols(None, Interval.M5) == ["SOLUSDT", "BTCUSDT"]
    with pytest.raises(ConfigError, match="needs a data folder"):
        r2.resolve_symbols(None, Interval.M5)
    with pytest.raises(ConfigError, match=r"top200.txt not found"):
        r2.resolve_symbols(tmp_path, Interval.M5)
    for name in ("top200.txt", "delisted.txt", "outside_pool.txt"):
        (tmp_path / name).write_text("")
    with pytest.raises(ConfigError, match="no 5m candles"):
        r2.resolve_symbols(tmp_path, Interval.M5)
    (tmp_path / "top200.txt").write_text("AAAUSDT USDCUSDT")
    (tmp_path / "delisted.txt").write_text("OLDUSDT")
    for symbol in ("AAAUSDT", "OLDUSDT", "USDCUSDT"):
        write_series(tmp_path, symbol, Interval.M5, kline_frame([T0], [1.0]))
    assert r2.resolve_symbols(tmp_path, Interval.M5) == ["AAAUSDT", "OLDUSDT"]


EXTERNAL = """
state = "s.sqlite"
[account]
initial_cash = 1000
[[strategies]]
name = "cnn"
source = "external"
interval = "5m"
warmup = 100
symbols = ["AAAUSDT"]
capital = 500
sizing = {{ slots = 2 }}
{extra}
"""


def test_external_strategies_name_their_context_symbols() -> None:
    config = parse_config(
        EXTERNAL.format(extra='context_symbols = ["btcusdt", "ETHUSDT", "BTCUSDT"]')
    )
    assert config.strategy("cnn").context_symbols == ["BTCUSDT", "ETHUSDT"]
    assert parse_config(EXTERNAL.format(extra="")).strategy("cnn").context_symbols == []


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ('context_symbols = ["BTCUSD"]', "strategies[0] (cnn).context_symbols: only USDT-quoted"),
    ],
)
def test_bad_context_symbols_name_their_field(extra: str, message: str) -> None:
    with pytest.raises(ConfigError, match=re.escape(message)):
        parse_config(EXTERNAL.format(extra=extra))


def test_an_in_process_plugin_declares_its_own_context_symbols() -> None:
    text = GOOD.replace("params = {}", 'params = {}\ncontext_symbols = ["BTCUSDT"]')
    with pytest.raises(ConfigError, match="context_symbols are for external strategies"):
        parse_config(text)
