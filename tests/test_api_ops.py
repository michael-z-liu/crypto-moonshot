"""The dashboard's endpoints: trades, equity, signals, providers and alerts."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import httpx
import pytest
from api_fakes import ApiRig, at
from engine_fakes import flat_bars
from ops_fakes import add_batch, add_position

from crypto_moonshot.live.health import GB, HealthMonitor
from crypto_moonshot.live.state import BatchStatus, EventLevel, SignalOutcome, SignalRow
from crypto_moonshot.models import Interval

pytestmark = pytest.mark.anyio

NEW_ENDPOINTS = ["/trades", "/equity", "/signals", "/providers", "/alerts"]


async def test_the_new_endpoints_need_the_token(tmp_path: Path) -> None:
    async with ApiRig(tmp_path) as rig:
        bare = httpx.AsyncClient(base_url=rig.url + "/api/v1")
        for path in NEW_ENDPOINTS:
            assert (await bare.get(path)).status_code == 401, path
            assert (await bare.get(path, headers={"Authorization": "Bearer x"})).status_code == 401
            assert (await rig.http.get(path)).status_code == 200, path
        assert (await bare.post("/alerts/x/ack")).status_code == 401
        await bare.aclose()


# -- trades ---------------------------------------------------------------------------------------


def three_trades(rig: ApiRig) -> None:
    store = rig.store
    add_position(store, "ext", "AAAUSDT", at(1), pnl=10.0, exit_time=at(3), quantity=1.0)
    add_position(store, "ext", "BBBUSDT", at(1), pnl=-4.0, exit_time=at(5))
    add_position(store, "inproc", "DDDUSDT", at(2), pnl=2.5, exit_time=at(4))
    add_position(store, "inproc", "EEEUSDT", at(6), exit_time=at(6))  # open: not a trade


async def test_trades_are_newest_first_with_stable_fields(tmp_path: Path) -> None:
    async with ApiRig(tmp_path) as rig:
        three_trades(rig)
        body = (await rig.http.get("/trades")).json()
        assert body["strategy"] is None
        assert body["limit"] == 100
        assert body["count"] == 3
        assert [t["symbol"] for t in body["trades"]] == ["BBBUSDT", "DDDUSDT", "AAAUSDT"]
        first = body["trades"][2]
        assert set(first) == {
            "id",
            "strategy",
            "symbol",
            "interval",
            "entry_time",
            "entry_price",
            "exit_time",
            "exit_price",
            "exit_reason",
            "quantity",
            "cost",
            "proceeds",
            "pnl",
            "return_fraction",
            "hold_seconds",
            "signal_info",
        }
        assert first["strategy"] == "ext"
        assert first["interval"] == "1h"
        assert first["entry_time"] == at(1).isoformat()
        assert first["exit_time"] == at(3).isoformat()
        assert first["pnl"] == 10.0
        assert first["cost"] == 100.0
        assert first["proceeds"] == 110.0
        assert first["return_fraction"] == pytest.approx(0.1)
        assert first["hold_seconds"] == 2 * 3600
        assert first["exit_reason"] == "take_profit"
        assert first["signal_info"] == {"score": 1}


async def test_trades_filter_and_limit(tmp_path: Path) -> None:
    async with ApiRig(tmp_path) as rig:
        three_trades(rig)
        ext = (await rig.http.get("/trades", params={"strategy": "ext"})).json()
        assert [t["symbol"] for t in ext["trades"]] == ["BBBUSDT", "AAAUSDT"]
        assert ext["strategy"] == "ext"
        limited = (await rig.http.get("/trades", params={"limit": 1})).json()
        assert [t["symbol"] for t in limited["trades"]] == ["BBBUSDT"]
        assert limited["limit"] == 1
        assert (await rig.http.get("/trades", params={"strategy": "nope"})).status_code == 404
        for bad in (0, 1001, -1, "x"):
            response = await rig.http.get("/trades", params={"limit": bad})
            assert response.status_code == 422, bad
        assert (await rig.http.get("/trades", params={"limit": 1000})).status_code == 200


async def test_trades_with_none_yet(tmp_path: Path) -> None:
    async with ApiRig(tmp_path) as rig:
        assert (await rig.http.get("/trades")).json()["trades"] == []


# -- equity ---------------------------------------------------------------------------------------


async def test_equity_has_realised_points_and_a_mark_to_market(tmp_path: Path) -> None:
    async with ApiRig(tmp_path) as rig:
        three_trades(rig)
        rig.feed.bars["EEEUSDT"] = flat_bars(40, 110.0)
        rig.feed.bars["DDDUSDT"] = flat_bars(40, 100.0)
        rig.feed.late("EEEUSDT", 3)  # the feed's last close of the open position: 110
        # an open position, 2 units bought at 100 for 200: worth 220 at the last close
        add_position(rig.store, "ext", "CCCUSDT", at(7), exit_time=at(7), cost=200.0, quantity=2.0)
        rig.feed.bars["CCCUSDT"] = flat_bars(40, 110.0)
        rig.feed.late("CCCUSDT", 3)
        rig.clock.now = at(8)
        body = (await rig.http.get("/equity")).json()
        assert body["time"] == at(8).isoformat()
        assert set(body["strategies"]) == {"ext", "inproc"}
        ext = body["strategies"]["ext"]
        assert ext["capital"] == 1000.0
        assert ext["realised_pnl"] == pytest.approx(6.0)
        assert ext["realised_equity"] == pytest.approx(1006.0)
        assert ext["unrealised_pnl"] == pytest.approx(20.0)
        assert ext["equity"] == pytest.approx(1026.0)
        assert ext["open_positions"] == 1
        assert ext["unpriced_positions"] == 0
        assert ext["points"] == [
            {"time": at(3).isoformat(), "equity": 1010.0},
            {"time": at(5).isoformat(), "equity": 1006.0},
        ]
        inproc = body["strategies"]["inproc"]
        assert inproc["realised_pnl"] == 2.5
        # EEE is open in inproc, but its candle (price 110, entry 100, qty 1) is held: +10
        assert inproc["unrealised_pnl"] == pytest.approx(10.0)
        assert inproc["equity"] == pytest.approx(1012.5)
        assert inproc["points"] == [{"time": at(4).isoformat(), "equity": 1002.5}]


async def test_equity_prices_a_position_without_candles_at_its_entry(tmp_path: Path) -> None:
    async with ApiRig(tmp_path) as rig:
        add_position(
            rig.store, "inproc", "EEEUSDT", at(1), exit_time=at(1), entry_price=50.0, quantity=2.0
        )
        body = (await rig.http.get("/equity", params={"strategy": "inproc"})).json()
        assert list(body["strategies"]) == ["inproc"]
        row = body["strategies"]["inproc"]
        assert row["unrealised_pnl"] == pytest.approx(0.0)  # 2 x 50 against a cost of 100
        assert row["unpriced_positions"] == 1
        assert row["points"] == []


async def test_equity_limit_keeps_the_latest_points_and_validates(tmp_path: Path) -> None:
    async with ApiRig(tmp_path) as rig:
        three_trades(rig)
        body = (await rig.http.get("/equity", params={"strategy": "ext", "limit": 1})).json()
        assert body["strategies"]["ext"]["points"] == [
            {"time": at(5).isoformat(), "equity": 1006.0}
        ]
        assert (await rig.http.get("/equity", params={"strategy": "nope"})).status_code == 404
        assert (await rig.http.get("/equity", params={"limit": 0})).status_code == 422
        assert (await rig.http.get("/equity", params={"limit": 10_001})).status_code == 422


# -- signals --------------------------------------------------------------------------------------


async def test_signals_list_the_outcome_and_reason(tmp_path: Path) -> None:
    async with ApiRig(tmp_path) as rig:
        rows = [SignalRow("AAAUSDT", priority=2.0, info={"z": 3}), SignalRow("BBBUSDT", "exit")]
        add_batch(rig.store, "ext", at(1), BatchStatus.ON_TIME, lateness=1.0, signals=rows)
        add_batch(
            rig.store,
            "inproc",
            at(2),
            BatchStatus.ON_TIME,
            lateness=1.0,
            signals=[SignalRow("DDDUSDT")],
        )
        batch = rig.store.batch_for("ext", Interval.H1, at(1))
        assert batch is not None
        rig.store.set_signal_outcome(batch.signals[0].id, SignalOutcome.REJECTED, "slots_full")
        body = (await rig.http.get("/signals")).json()
        assert body["count"] == 3
        assert [s["symbol"] for s in body["signals"]] == ["DDDUSDT", "BBBUSDT", "AAAUSDT"]
        entered = body["signals"][2]
        assert set(entered) == {
            "id",
            "strategy",
            "interval",
            "candle_time",
            "batch_status",
            "symbol",
            "action",
            "priority",
            "weight",
            "outcome",
            "reason",
            "info",
        }
        assert entered["strategy"] == "ext"
        assert entered["interval"] == "1h"
        assert entered["candle_time"] == at(1).isoformat()
        assert entered["batch_status"] == "on_time"
        assert entered["action"] == "enter"
        assert entered["priority"] == 2.0
        assert entered["outcome"] == "rejected"
        assert entered["reason"] == "slots_full"
        assert entered["info"] == {"z": 3}
        assert body["signals"][1]["outcome"] is None  # not decided yet


async def test_signals_filter_limit_and_validation(tmp_path: Path) -> None:
    async with ApiRig(tmp_path) as rig:
        for k in range(5):
            add_batch(
                rig.store,
                "ext",
                at(k),
                BatchStatus.ON_TIME,
                lateness=1.0,
                signals=[SignalRow("AAAUSDT")],
            )
        add_batch(
            rig.store,
            "inproc",
            at(0),
            BatchStatus.ON_TIME,
            lateness=1.0,
            signals=[SignalRow("DDDUSDT")],
        )
        two = (await rig.http.get("/signals", params={"strategy": "ext", "limit": 2})).json()
        assert [s["candle_time"] for s in two["signals"]] == [at(4).isoformat(), at(3).isoformat()]
        assert all(s["strategy"] == "ext" for s in two["signals"])
        assert two["limit"] == 2
        assert (await rig.http.get("/signals", params={"strategy": "nope"})).status_code == 404
        assert (await rig.http.get("/signals", params={"limit": 1001})).status_code == 422
        assert (await rig.http.get("/signals", params={"limit": 0})).status_code == 422


# -- providers ------------------------------------------------------------------------------------


async def test_providers_summarise_the_last_24_hours(tmp_path: Path) -> None:
    async with ApiRig(tmp_path) as rig:
        rig.clock.now = at(40)
        add_batch(rig.store, "ext", at(10), BatchStatus.ON_TIME, lateness=1.0)  # older than 24 h
        add_batch(rig.store, "ext", at(17), BatchStatus.ON_TIME, lateness=2.0)
        add_batch(rig.store, "ext", at(18), BatchStatus.ON_TIME, lateness=4.0)
        add_batch(rig.store, "ext", at(19), BatchStatus.LATE, lateness=50.0)
        add_batch(rig.store, "ext", at(20), BatchStatus.MISSED)
        add_batch(rig.store, "inproc", at(30), BatchStatus.ON_TIME, lateness=0.5)
        rig.store.provider_seen("ext", at(39), at(19))
        body = (await rig.http.get("/providers")).json()
        assert body["window_hours"] == 24
        assert body["since"] == at(16).isoformat()
        by_name = {p["strategy"]: p for p in body["providers"]}
        assert list(by_name) == ["ext", "inproc"]
        ext = by_name["ext"]
        assert set(ext) == {
            "strategy",
            "external",
            "interval",
            "on_time",
            "late",
            "missed",
            "median_lateness_seconds",
            "worst_lateness_seconds",
            "last_batch_time",
            "last_seen",
        }
        assert (ext["external"], ext["interval"]) == (True, "1h")
        assert (ext["on_time"], ext["late"], ext["missed"]) == (2, 1, 1)
        assert ext["median_lateness_seconds"] == 4.0
        assert ext["worst_lateness_seconds"] == 50.0
        assert ext["last_batch_time"] == at(19).isoformat()
        assert ext["last_seen"] == at(39).isoformat()
        inproc = by_name["inproc"]
        assert inproc["external"] is False
        assert (inproc["on_time"], inproc["late"], inproc["missed"]) == (1, 0, 0)
        assert inproc["last_seen"] is None


async def test_providers_without_batches(tmp_path: Path) -> None:
    async with ApiRig(tmp_path) as rig:
        body = (await rig.http.get("/providers")).json()
        for row in body["providers"]:
            assert (row["on_time"], row["late"], row["missed"]) == (0, 0, 0)
            assert row["median_lateness_seconds"] is None
            assert row["worst_lateness_seconds"] is None
            assert row["last_batch_time"] is None


# -- alerts ---------------------------------------------------------------------------------------


def fire(rig: ApiRig, key: str = "disk_low", level: EventLevel = EventLevel.WARNING) -> None:
    rule = key.split(":")[0]
    with rig.store.transaction():
        rig.store.fire_alert(key, rule, level, f"detail of {key}", rig.clock.now)
        rig.engine.record_event(
            level,
            "alert",
            f"{key}: x",
            {"rule": rule, "key": key, "state": "firing", "detail": "x"},
        )


async def test_alerts_list_active_with_the_acknowledged_flag_and_the_history(
    tmp_path: Path,
) -> None:
    async with ApiRig(tmp_path) as rig:
        assert (await rig.http.get("/alerts")).json() == {
            "active": [],
            "unacknowledged": 0,
            "history": [],
        }
        rig.clock.now = at(1)
        fire(rig, "disk_low")
        rig.clock.now = at(2)
        fire(rig, "provider_missing:ext", EventLevel.ERROR)
        body = (await rig.http.get("/alerts")).json()
        assert body["unacknowledged"] == 2
        first, second = body["active"]
        assert first == {
            "key": "disk_low",
            "rule": "disk_low",
            "level": "warning",
            "detail": "detail of disk_low",
            "fired_at": at(1).isoformat(),
            "acknowledged": False,
            "acknowledged_at": None,
        }
        assert second["key"] == "provider_missing:ext"
        assert [h["key"] for h in body["history"]] == ["provider_missing:ext", "disk_low"]
        assert set(body["history"][0]) == {
            "id",
            "time",
            "level",
            "key",
            "rule",
            "state",
            "detail",
        }
        assert body["history"][0]["state"] == "firing"
        assert body["history"][0]["level"] == "error"
        one = (await rig.http.get("/alerts", params={"history": 1})).json()
        assert [h["key"] for h in one["history"]] == ["provider_missing:ext"]
        none = (await rig.http.get("/alerts", params={"history": 0})).json()
        assert none["history"] == []
        assert (await rig.http.get("/alerts", params={"history": 501})).status_code == 422


async def test_acknowledging_hides_it_from_the_count_and_is_idempotent(tmp_path: Path) -> None:
    async with ApiRig(tmp_path) as rig:
        rig.clock.now = at(1)
        fire(rig, "disk_low")
        fire(rig, "provider_missing:ext")
        rig.clock.now = at(2)
        response = await rig.http.post("/alerts/provider_missing:ext/ack")
        assert response.status_code == 200
        alert = response.json()["alert"]
        assert response.json()["status"] == "acknowledged"
        assert alert["acknowledged"] is True
        assert alert["acknowledged_at"] == at(2).isoformat()
        body = (await rig.http.get("/alerts")).json()
        assert body["unacknowledged"] == 1
        assert [a["acknowledged"] for a in body["active"]] == [False, True]  # still listed
        rig.clock.now = at(3)
        again = (await rig.http.post("/alerts/provider_missing:ext/ack")).json()["alert"]
        assert again["acknowledged_at"] == at(2).isoformat()  # the first time counts
        kinds = [e.kind for e in rig.store.events_since()]
        assert kinds.count("alert_acknowledged") == 2  # an audit trail, not an alert


async def test_acknowledging_an_unknown_or_inactive_alert_is_404(tmp_path: Path) -> None:
    async with ApiRig(tmp_path) as rig:
        assert (await rig.http.post("/alerts/nope/ack")).status_code == 404
        fire(rig, "disk_low")
        assert (await rig.http.post("/alerts/disk_low/ack")).status_code == 200
        rig.store.resolve_alert("disk_low")
        response = await rig.http.post("/alerts/disk_low/ack")
        assert response.status_code == 404
        assert "disk_low" in response.json()["detail"]


async def test_an_acknowledgement_is_cleared_when_the_alert_resolves_and_fires_again(
    tmp_path: Path,
) -> None:
    async with ApiRig(tmp_path) as rig:
        free = [0]
        monitor = HealthMonitor(
            rig.engine,
            rig.feed,
            rig.config.health.model_copy(
                update={"startup_grace": timedelta(days=365), "no_ticks_intervals": 10_000}
            ),
            disk_free=lambda _path: free[0],
        )
        monitor.check()  # disk_low fires
        assert (await rig.http.post("/alerts/disk_low/ack")).status_code == 200
        monitor.check()
        body = (await rig.http.get("/alerts")).json()
        assert [(a["key"], a["acknowledged"]) for a in body["active"]] == [("disk_low", True)]
        free[0] = 50 * GB
        monitor.check()
        assert (await rig.http.get("/alerts")).json()["active"] == []
        assert (await rig.http.post("/alerts/disk_low/ack")).status_code == 404
        free[0] = 0
        monitor.check()
        body = (await rig.http.get("/alerts")).json()
        assert [(a["key"], a["acknowledged"]) for a in body["active"]] == [("disk_low", False)]
        assert body["unacknowledged"] == 1
        states = [h["state"] for h in body["history"]]
        assert states == ["firing", "resolved", "firing"]  # newest first, one event each


async def test_an_acknowledgement_survives_a_restart(tmp_path: Path) -> None:
    async with ApiRig(tmp_path) as rig:
        fire(rig, "disk_low")
        assert (await rig.http.post("/alerts/disk_low/ack")).status_code == 200
    async with ApiRig(tmp_path) as again:  # the same state file, a new process in effect
        body = (await again.http.get("/alerts")).json()
        assert [(a["key"], a["acknowledged"]) for a in body["active"]] == [("disk_low", True)]
        assert body["unacknowledged"] == 0
