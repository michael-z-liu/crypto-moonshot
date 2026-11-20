"""End to end: the daemon replays while a plugin in a separate process trades over HTTP, and the
result equals the same plugin run in process. Also the example plugin and config."""

from __future__ import annotations

import asyncio
import os
import sys
import textwrap
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from pydantic import SecretStr
from replay_fakes import COINS, END, H1, START, at, run_config, strategy_dict, trades_core

from crypto_moonshot.config import Settings
from crypto_moonshot.live.config import load_config
from crypto_moonshot.live.daemon import serve_replay
from crypto_moonshot.live.engine import build_specs
from crypto_moonshot.live.plugin import load_strategy
from crypto_moonshot.live.replay import ReplayResult, prepare_replay, run_replay
from crypto_moonshot.provider import Provider

TOKEN = "tkn-e2e-0b6fK2sWq9"
ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "examples" / "strategies" / "ma_cross.py"

PLUGIN = textwrap.dedent(
    """
    from datetime import timedelta

    from crypto_moonshot.broker.base import ExitPlan
    from crypto_moonshot.live.plugin import RuleStrategy
    from crypto_moonshot.models import Interval
    from crypto_moonshot.strategy.capitulation import CapitulationRule


    def make() -> RuleStrategy:
        rule = CapitulationRule(
            "cap",
            min_qv_base=1.0,
            exit=ExitPlan(take_profit=0.03, stop=0.04, max_hold=timedelta(hours=12)),
            max_ret_4h=-0.03,
            min_btc_vol_mult=3.0,
            spike_mult=4.0,
            interval=Interval.H1,
        )
        return RuleStrategy(rule, workers=1)
    """
)
EXTERNAL: dict[str, Any] = {
    "source": "external",
    "interval": "1h",
    "context_symbols": ["BTCUSDT"],
    "exit": {"take_profit": 0.03, "stop": 0.04, "max_hold": "12h"},
}


def configs(tmp_path: Path, source: str) -> tuple[object, object]:
    """(in-process config, external config) of the same strategy on the replay store."""
    inproc = run_config(tmp_path / "inproc.sqlite", [strategy_dict(source=source, capital=350.0)])
    warmup = load_strategy(source, name="cap").warmup
    external = run_config(
        tmp_path / "external.sqlite",
        [strategy_dict(**EXTERNAL, warmup=warmup, capital=350.0)],
    )
    return inproc, external


def settings() -> Settings:
    return Settings(api_token=SecretStr(TOKEN))


def test_a_plugin_in_another_process_trades_exactly_like_one_in_process(
    replay_root: Path, tmp_path: Path
) -> None:
    plugin = tmp_path / "plugin.py"
    plugin.write_text(PLUGIN)
    source = f"{plugin}:make"
    inproc, external = configs(tmp_path, source)
    expected = run_replay(inproc, at(START), at(END), replay_root, tmp_path / "out-inproc")  # type: ignore[arg-type]
    assert expected.trades.height > 20  # the fixture trades a lot

    async def scenario() -> tuple[ReplayResult, int, str, str]:
        prepared = prepare_replay(external, at(START), at(END), replay_root, tmp_path / "out-ext")  # type: ignore[arg-type]
        ready: asyncio.Future[int] = asyncio.get_running_loop().create_future()
        server = asyncio.create_task(
            serve_replay(
                prepared,
                settings(),
                port=0,
                provider_timeout=60,
                install_signals=False,
                on_ready=ready.set_result,
            )
        )
        try:
            port = await asyncio.wait_for(ready, 30)
            child = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "crypto_moonshot",
                "provider",
                "run",
                source,
                "--name",
                "cap",
                "--url",
                f"http://127.0.0.1:{port}",
                env={**os.environ, "MOONSHOT_API_TOKEN": TOKEN},
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            out, err = await asyncio.wait_for(child.communicate(), 180)
            result = await asyncio.wait_for(server, 60)
        finally:
            if not server.done():
                server.cancel()
            prepared.close()
        assert child.returncode is not None
        return result, child.returncode, out.decode(), err.decode()

    result, code, out, err = asyncio.run(scenario())
    assert code == 0, err
    assert "REFUSED" not in err
    assert TOKEN not in out + err
    assert_frame_equal(trades_core(result.trades), trades_core(expected.trades))
    assert result.document["signals"] == expected.document["signals"]
    assert result.metrics == expected.metrics
    assert result.open_positions == 0


def test_the_example_plugin_replays_the_same_in_and_out_of_process(
    replay_root: Path, tmp_path: Path
) -> None:
    source = f"{EXAMPLE}:MaCross"
    params = {"fast": 6, "slow": 24}
    plugin = load_strategy(source, name="cross", params=params)
    assert plugin.interval is H1
    common = {"symbols": list(COINS), "capital": 1000.0, "sizing": {"slots": 4}}
    inproc = run_config(
        tmp_path / "inproc.sqlite",
        [{"name": "cross", "source": source, "params": params, **common}],
    )
    expected = run_replay(inproc, at(START), at(END), replay_root, tmp_path / "out-inproc")
    assert expected.trades.height > 5

    external = run_config(
        tmp_path / "external.sqlite",
        [
            {
                "name": "cross",
                "source": "external",
                "interval": "1h",
                "warmup": plugin.warmup,
                "exit": {"take_profit": 0.05, "stop": 0.03, "max_hold": "48h"},
                **common,
            }
        ],
    )

    async def scenario() -> ReplayResult:
        prepared = prepare_replay(external, at(START), at(END), replay_root)
        ready: asyncio.Future[int] = asyncio.get_running_loop().create_future()
        server = asyncio.create_task(
            serve_replay(
                prepared, settings(), port=0, install_signals=False, on_ready=ready.set_result
            )
        )
        port = await asyncio.wait_for(ready, 30)
        provider = Provider(plugin, url=f"http://127.0.0.1:{port}", token=TOKEN, name="cross")
        try:
            await asyncio.wait_for(provider.run(), 120)
            return await asyncio.wait_for(server, 60)
        finally:
            prepared.close()

    result = asyncio.run(scenario())
    assert_frame_equal(trades_core(result.trades), trades_core(expected.trades))


def test_the_example_config_loads_and_its_plugin_satisfies_the_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(ROOT)
    config = load_config(ROOT / "examples" / "paper.toml")
    config.require_runnable()
    (spec,) = build_specs(config, None)
    assert spec.name == "cross"
    assert spec.interval is H1
    assert spec.warmup == 50
    assert spec.exit is not None
    assert spec.symbols[0] == "BTCUSDT"


def test_the_example_plugin_signals_a_cross_and_nothing_else() -> None:
    plugin = load_strategy(f"{EXAMPLE}:MaCross", name="x", params={"fast": 2, "slow": 4})
    rising = pl.DataFrame(
        {"open_time": [at(i) for i in range(6)], "close": [10.0, 10.0, 10.0, 10.0, 10.0, 14.0]}
    )
    flat = rising.with_columns(close=pl.lit(10.0))
    frame = plugin.on_candles({"UPUSDT": rising, "FLATUSDT": flat}, at(5), {})
    assert frame["symbol"].to_list() == ["UPUSDT"]
    assert plugin.on_candles({"UPUSDT": rising.head(4)}, at(5), {}).height == 0
