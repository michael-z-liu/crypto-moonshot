"""`moonshot live ...` and `moonshot provider run`: happy paths and exit codes."""

from __future__ import annotations

import asyncio
import json
import socket
import threading
from pathlib import Path
from typing import Any

import pytest
from api_fakes import TOKEN, ApiRig, at
from engine_fakes import until
from replay_fakes import COINS, END, RULE_PLUGIN_SOURCE, START
from replay_fakes import at as replay_at
from typer.testing import CliRunner, Result

from crypto_moonshot import live_cli
from crypto_moonshot.cli import app
from crypto_moonshot.live.config import ConfigError, load_config
from crypto_moonshot.live.daemon import DaemonError, DaemonInterrupted
from crypto_moonshot.live.plugin import load_strategy
from crypto_moonshot.live.replay import run_replay
from crypto_moonshot.live.state import Control, StateStore
from crypto_moonshot.provider import Provider

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "examples" / "strategies" / "ma_cross.py"
ENV = {"MOONSHOT_API_TOKEN": TOKEN}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in ("MOONSHOT_API_TOKEN", "MOONSHOT_URL", "MOONSHOT_DATA_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)


def invoke(*args: str, env: dict[str, str] | None = None, input: str | None = None) -> Result:
    return CliRunner().invoke(app, list(args), env=env, input=input)


def config_text(
    tmp_path: Path, strategies: str, mode: str = "paper", name: str = "run.toml"
) -> Path:
    path = tmp_path / name
    path.write_text(
        f'mode = "{mode}"\nstate = "state/run.sqlite"\n\n[account]\ninitial_cash = 1000\n\n'
        + strategies
    )
    return path


RULE_STRATEGY = """[[strategies]]
name = "cap"
source = "{plugin}:make"
symbols = {symbols}
capital = 350
sizing = {{ stake = 100 }}
"""


def rule_config(tmp_path: Path) -> Path:
    plugin = tmp_path / "plugin.py"
    plugin.write_text(RULE_PLUGIN_SOURCE)
    return config_text(
        tmp_path, RULE_STRATEGY.format(plugin=plugin, symbols=json.dumps(list(COINS)))
    )


async def in_thread(
    *args: str, env: dict[str, str] | None = None, input: str | None = None
) -> Result:
    """Run the CLI in a worker thread so the event loop that serves the API keeps running."""
    return await asyncio.to_thread(invoke, *args, env=env, input=input)


# -- live run -----------------------------------------------------------------------------------


def test_run_needs_a_token_and_a_runnable_config(tmp_path: Path) -> None:
    config = rule_config(tmp_path)
    result = invoke("live", "run", "--config", str(config))
    assert result.exit_code == 2
    assert "MOONSHOT_API_TOKEN is not set" in result.stderr
    testnet = config_text(tmp_path, "", mode="testnet")
    result = invoke("live", "run", "--config", str(testnet), env=ENV)
    assert result.exit_code == 2  # no strategies: invalid
    broken = tmp_path / "broken.toml"
    broken.write_text("mode = 'paper'\n")
    result = invoke("live", "run", "--config", str(broken), env=ENV)
    assert result.exit_code == 2
    assert "invalid run config" in result.stderr
    assert TOKEN not in result.output + result.stderr
    missing = invoke("live", "run", "--config", str(tmp_path / "nope.toml"), env=ENV)
    assert missing.exit_code == 2
    assert "cannot read the run config" in missing.stderr


def test_run_refuses_the_mainnet_mode(tmp_path: Path) -> None:
    config = config_text(
        tmp_path, RULE_STRATEGY.format(plugin="x.py", symbols='["AAAUSDT"]'), mode="live"
    )
    result = invoke("live", "run", "--config", str(config), env=ENV)
    assert result.exit_code == 2
    assert "not available in this build" in result.stderr


def test_run_passes_its_options_and_maps_failures_to_exit_codes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = rule_config(tmp_path)
    calls: list[dict[str, Any]] = []
    outcome: list[BaseException | None] = [None]

    async def fake(run_config: Any, settings: Any, *, host: str, port: int) -> None:
        calls.append({"host": host, "port": port, "data": settings.data_dir, "cash": run_config})
        if outcome[0] is not None:
            raise outcome[0]

    monkeypatch.setattr(live_cli, "run_daemon", fake)
    base = ["live", "run", "--config", str(config), "--host", "0.0.0.0", "--port", "9999"]
    ok = invoke(*base, "--data-dir", str(tmp_path / "d"), env=ENV)
    assert ok.exit_code == 0, ok.output
    assert calls[0]["host"] == "0.0.0.0"
    assert calls[0]["port"] == 9999
    assert calls[0]["data"] == tmp_path / "d"
    cases: list[tuple[BaseException, int, str]] = [
        (DaemonError("cannot listen on x"), 1, "cannot listen on x"),
        (DaemonError("bad setting", usage=True), 2, "bad setting"),
        (ConfigError("the state contradicts"), 2, "the state contradicts"),
        (KeyboardInterrupt(), 130, ""),
        (RuntimeError("boom"), 1, "RuntimeError: boom"),
    ]
    for exc, code, text in cases:
        outcome[0] = exc
        result = invoke(*base, env=ENV)
        assert result.exit_code == code, (exc, result.output)
        assert text in result.stderr


# -- live replay --------------------------------------------------------------------------------


def test_replay_prints_the_metrics_and_writes_the_results(
    replay_root: Path, tmp_path: Path
) -> None:
    config = rule_config(tmp_path)
    out = tmp_path / "out"
    result = invoke(
        "live", "replay", "--config", str(config), "--data-dir", str(replay_root),
        "--start", replay_at(START).isoformat(), "--end", replay_at(END).isoformat(),
        "--out", str(out),
    )  # fmt: skip
    assert result.exit_code == 0, result.output + result.stderr
    assert "trades" in result.stdout
    assert "final equity" in result.stdout
    assert "results written to" in result.stderr
    assert {p.name for p in out.iterdir()} >= {"trades.parquet", "metrics.json", "equity.csv"}
    document = json.loads((out / "metrics.json").read_text())
    assert document["metrics"]["trades"] > 20
    assert (tmp_path / "state" / "run.sqlite").exists()  # the config's state file


def test_replay_json_is_one_document_on_stdout(replay_root: Path, tmp_path: Path) -> None:
    config = rule_config(tmp_path)
    result = invoke(
        "live", "replay", "--config", str(config), "--data-dir", str(replay_root),
        "--start", "2024-01-09", "--end", "2024-01-21", "--out", str(tmp_path / "out"),
        "--state", str(tmp_path / "other.sqlite"), "--json",
    )  # fmt: skip
    assert result.exit_code == 0, result.output + result.stderr
    document = json.loads(result.stdout)
    assert document["mode"] == "replay"
    assert document["out"] == str(tmp_path / "out")
    assert document["metrics"]["trades"] > 0
    assert document["start"].startswith("2024-01-09")
    assert set(document["strategies"]) == {"cap"}
    assert (tmp_path / "other.sqlite").exists()


def test_replay_errors(replay_root: Path, tmp_path: Path) -> None:
    config = rule_config(tmp_path)
    common = ["--start", "2024-01-09", "--end", "2024-01-21"]
    backwards = invoke("live", "replay", "--config", str(config), "--start", "2024-02-01",
                       "--end", "2024-01-01", "--data-dir", str(replay_root))  # fmt: skip
    assert backwards.exit_code == 2
    bad_date = invoke("live", "replay", "--config", str(config), "--start", "soon", "--end",
                      "2024-01-21", "--data-dir", str(replay_root))  # fmt: skip
    assert bad_date.exit_code == 2
    no_data = invoke("live", "replay", "--config", str(config), *common,
                     "--data-dir", str(tmp_path / "empty"))  # fmt: skip
    assert no_data.exit_code == 1
    assert "no candles stored" in no_data.stderr
    no_config = invoke("live", "replay", "--config", str(tmp_path / "x.toml"), *common)
    assert no_config.exit_code == 2
    no_plugin = tmp_path / "np.toml"
    no_plugin.write_text(
        config.read_text().replace(f"{tmp_path / 'plugin.py'}:make", "missing_plugin.py:make")
    )
    result = invoke("live", "replay", "--config", str(no_plugin), *common,
                    "--data-dir", str(replay_root))  # fmt: skip
    assert result.exit_code == 2
    assert "missing_plugin.py" in result.stderr
    serve = invoke("live", "replay", "--config", str(config), *common, "--serve",
                   "--data-dir", str(replay_root))  # fmt: skip
    assert serve.exit_code == 2
    assert "MOONSHOT_API_TOKEN is not set" in serve.stderr


def test_replay_can_be_interrupted(
    replay_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def interrupted(*args: object, **kwargs: object) -> None:
        raise DaemonInterrupted("the replay was interrupted; run it again to resume")

    monkeypatch.setattr(live_cli, "serve_replay", interrupted)
    result = invoke(
        "live", "replay", "--config", str(rule_config(tmp_path)), "--serve",
        "--start", "2024-01-09", "--end", "2024-01-21", "--data-dir", str(replay_root),
        env=ENV,
    )  # fmt: skip
    assert result.exit_code == 130
    assert "interrupted" in result.stderr


def free_port() -> int:
    with socket.create_server(("127.0.0.1", 0)) as sock:
        port: int = sock.getsockname()[1]
        return port


def test_replay_serve_trades_with_a_provider_over_the_api(
    replay_root: Path, tmp_path: Path
) -> None:
    """`live replay --serve`: the CLI serves the API and an external provider takes part."""
    plugin = f"{EXAMPLE}:MaCross"
    params = {"fast": 6, "slow": 24}
    inproc = config_text(
        tmp_path,
        f'[[strategies]]\nname = "cross"\nsource = "{plugin}"\nparams = {{ fast = 6, slow = 24 }}\n'
        f"symbols = {json.dumps(list(COINS))}\ncapital = 1000\nsizing = {{ slots = 4 }}\n",
        name="inproc.toml",
    )
    expected = invoke(
        "live", "replay", "--config", str(inproc), "--data-dir", str(replay_root),
        "--start", replay_at(START).isoformat(), "--end", replay_at(END).isoformat(),
        "--out", str(tmp_path / "a"), "--state", str(tmp_path / "a.sqlite"), "--json",
    )  # fmt: skip
    assert expected.exit_code == 0, expected.stderr
    strategy = load_strategy(plugin, name="cross", params=params)
    external = config_text(
        tmp_path,
        '[[strategies]]\nname = "cross"\nsource = "external"\ninterval = "1h"\n'
        f"warmup = {strategy.warmup}\nsymbols = {json.dumps(list(COINS))}\ncapital = 1000\n"
        "sizing = { slots = 4 }\n"
        'exit = { take_profit = 0.05, stop = 0.03, max_hold = "48h" }\n',
        name="external.toml",
    )
    port = free_port()
    failures: list[BaseException] = []

    def provide() -> None:
        try:
            asyncio.run(
                Provider(
                    strategy,
                    url=f"http://127.0.0.1:{port}",
                    token=TOKEN,
                    name="cross",
                    backoff_start=0.05,
                    backoff_max=0.2,
                ).run()
            )
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=provide, daemon=True)
    thread.start()
    served = invoke(
        "live", "replay", "--config", str(external), "--data-dir", str(replay_root),
        "--start", replay_at(START).isoformat(), "--end", replay_at(END).isoformat(),
        "--out", str(tmp_path / "b"), "--state", str(tmp_path / "b.sqlite"), "--json",
        "--serve", "--port", str(port),
        env=ENV,
    )  # fmt: skip
    thread.join(30)
    assert not thread.is_alive()
    assert failures == []
    assert served.exit_code == 0, served.stderr
    a, b = json.loads(expected.stdout), json.loads(served.stdout)
    assert a["metrics"]["trades"] > 5
    assert b["metrics"] == a["metrics"]
    assert b["signals"] == a["signals"]


# -- status and controls --------------------------------------------------------------------------


@pytest.mark.anyio
async def test_status_pause_resume_and_flatten_through_the_api(tmp_path: Path) -> None:
    async with ApiRig(tmp_path) as rig:
        await rig.tick(0)
        status = await in_thread("live", "status", "--url", rig.url, "--json", env=ENV)
        assert status.exit_code == 0, status.stderr
        document = json.loads(status.stdout)
        assert document["source"] == "daemon"
        assert document["mode"] == "paper"
        assert set(document["books"]) == {"ext", "inproc"}
        assert document["last_ticks"] == {"1h": at(0).isoformat()}
        assert isinstance(document["events"], list)
        human = await in_thread("live", "status", "--url", rig.url, env=ENV)
        assert human.exit_code == 0
        assert "paper (daemon): running" in human.stdout
        assert "STRATEGY" in human.stdout
        assert TOKEN not in human.output + human.stderr

        paused = await in_thread("live", "pause", "--url", rig.url, env=ENV)
        assert paused.exit_code == 0
        assert "paused" in paused.stdout
        assert rig.store.controls().paused
        resumed = await in_thread("live", "resume", "--url", rig.url, env=ENV)
        assert resumed.exit_code == 0
        assert not rig.store.controls().paused

        declined = await in_thread("live", "flatten", "--url", rig.url, env=ENV, input="n\n")
        assert declined.exit_code == 1
        assert not rig.store.controls().flatten_requested
        confirmed = await in_thread("live", "flatten", "--url", rig.url, env=ENV, input="y\n")
        assert confirmed.exit_code == 0
        assert rig.store.controls().flatten_requested
        assert "flatten requested" in confirmed.stdout
        rig.store.set_control(Control.FLATTEN_REQUESTED, False)
        forced = await in_thread("live", "flatten", "--url", rig.url, "--yes", env=ENV)
        assert forced.exit_code == 0
        assert rig.store.controls().flatten_requested


@pytest.mark.anyio
async def test_status_and_controls_report_api_failures(tmp_path: Path) -> None:
    async with ApiRig(tmp_path) as rig:
        wrong = await in_thread("live", "status", "--url", rig.url, env={"MOONSHOT_API_TOKEN": "x"})
        assert wrong.exit_code == 1
        assert "refused the API token" in wrong.stderr
        none = await in_thread("live", "pause", "--url", rig.url)
        assert none.exit_code == 2
        assert "MOONSHOT_API_TOKEN is not set" in none.stderr
        no_token_status = await in_thread("live", "status", "--url", rig.url)
        assert no_token_status.exit_code == 2
    down = await in_thread("live", "pause", "--url", "http://127.0.0.1:1", env=ENV)
    assert down.exit_code == 1
    assert "cannot reach the daemon" in down.stderr
    assert TOKEN not in down.output + down.stderr
    down_status = await in_thread("live", "status", "--url", "http://127.0.0.1:1", env=ENV)
    assert down_status.exit_code == 1


def test_status_reads_the_state_file_when_the_daemon_is_down(
    replay_root: Path, tmp_path: Path
) -> None:
    config = rule_config(tmp_path)
    run_replay(load_config(config), replay_at(START), replay_at(END), replay_root)
    for env in (ENV, None):  # a token and no daemon, or no token at all
        result = invoke(
            "live", "status", "--config", str(config), "--url", "http://127.0.0.1:1", "--json",
            env=env,
        )  # fmt: skip
        assert result.exit_code == 0, result.stderr
        assert "reading the state file" in result.stderr
        document = json.loads(result.stdout)
        assert document["source"] == "state file"
        assert document["running"] is False
        assert document["books"]["cap"]["capital"] == 350.0
        assert document["last_ticks"]["1h"] == replay_at(END - 1).isoformat()
        assert document["events"]
        assert document["health"]["source"] == "state_file"
    human = invoke(
        "live", "status", "--config", str(config), "--url", "http://127.0.0.1:1", env=ENV
    )
    assert human.exit_code == 0
    assert "not running" in human.stdout
    assert "cap" in human.stdout
    # read-only: the file is untouched and still opens as usual
    with StateStore.open(load_config(config).state) as store:
        assert store.last_ticks()


def test_status_without_a_state_file_is_an_error(tmp_path: Path) -> None:
    config = rule_config(tmp_path)
    result = invoke("live", "status", "--config", str(config), "--url", "http://127.0.0.1:1",
                    env=ENV)  # fmt: skip
    assert result.exit_code == 1
    assert "no state file" in result.stderr
    bad = invoke("live", "status", "--config", str(tmp_path / "nope.toml"), env=ENV)
    assert bad.exit_code == 2


# -- provider -------------------------------------------------------------------------------------


def test_provider_run_errors(tmp_path: Path) -> None:
    source = f"{EXAMPLE}:MaCross"
    no_token = invoke("provider", "run", source, "--name", "x", "--url", "http://127.0.0.1:1")
    assert no_token.exit_code == 2
    assert "MOONSHOT_API_TOKEN is not set" in no_token.stderr
    missing = invoke("provider", "run", f"{tmp_path}/nope.py:X", "--name", "x", env=ENV)
    assert missing.exit_code == 2
    assert "nope.py" in missing.stderr
    bad_param = invoke("provider", "run", source, "--name", "x", "--param", "fast", env=ENV)
    assert bad_param.exit_code == 2
    bad_value = invoke(
        "provider", "run", source, "--name", "x", "--param", "fast=60", env=ENV
    )  # fmt: skip (fast >= slow)
    assert bad_value.exit_code == 2
    assert "fast" in bad_value.stderr


@pytest.mark.anyio
async def test_provider_run_fatal_problems_exit_1_and_the_end_of_a_run_exits_0(
    tmp_path: Path,
) -> None:
    source = f"{EXAMPLE}:MaCross"
    async with ApiRig(tmp_path) as rig:
        unknown = await in_thread("provider", "run", source, "--name", "nobody", "--url", rig.url,
                                  env=ENV)  # fmt: skip
        assert unknown.exit_code == 1
        assert "no strategy named" in unknown.stderr
        wrong = await in_thread("provider", "run", source, "--name", "ext", "--url", rig.url,
                                env={"MOONSHOT_API_TOKEN": "nope"})  # fmt: skip
        assert wrong.exit_code == 1
        assert "refused the API token" in wrong.stderr
        inproc = await in_thread("provider", "run", source, "--name", "inproc", "--url", rig.url,
                                 env=ENV)  # fmt: skip
        assert inproc.exit_code == 1
        assert "runs inside the daemon" in inproc.stderr

        # a provider that runs until the daemon says the run is over
        running = asyncio.create_task(
            in_thread(
                "provider",
                "run",
                source,
                "--name",
                "ext",
                "--url",
                rig.url,
                "--param",
                "fast=2",
                "--param",
                "slow=4",
                env=ENV,
            )
        )
        await until(lambda: rig.app.state.hub.client_count == 1, timeout=10)
        rig.app.state.hub.finish("replay finished")
        result = await asyncio.wait_for(running, 10)
        assert result.exit_code == 0, result.stderr
        assert TOKEN not in result.output + result.stderr


def test_the_commands_are_registered() -> None:
    result = invoke("--help")
    assert "live" in result.stdout
    assert "provider" in result.stdout
    assert "run" in invoke("live", "--help").stdout
