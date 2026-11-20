"""The dashboard in a real browser (Chromium via Playwright) against a served API.

Opt in: `uv run pytest -m browser --trace off` (needs `uv run playwright install chromium`).
The API runs in a background thread on its own event loop: an engine on a hand-driven feed with
known trades, an open position, a pending order, signals, provider batches and two active
alerts. Everything is fake data and a fake token; nothing leaves the machine.
"""

from __future__ import annotations

import asyncio
import re
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from api_fakes import TOKEN, ApiRig, at
from engine_fakes import flat_bars
from ops_fakes import add_batch, add_position
from playwright.sync_api import BrowserContext, ConsoleMessage, Page, Request, Route, expect

from crypto_moonshot.live.state import BatchStatus, Control, EventLevel, SignalOutcome, SignalRow

pytestmark = pytest.mark.browser


def fire(rig: ApiRig, key: str, level: EventLevel, detail: str) -> None:
    rule = key.split(":")[0]
    with rig.store.transaction():
        rig.store.fire_alert(key, rule, level, detail, rig.clock.now)
        rig.engine.record_event(
            level,
            "alert",
            f"{key}: {detail}",
            {"rule": rule, "key": key, "state": "firing", "detail": detail},
        )


def seed(rig: ApiRig) -> None:
    store = rig.store
    rig.clock.now = at(12)
    add_position(store, "ext", "AAAUSDT", at(1), pnl=10.0, exit_time=at(3), quantity=1.0)
    add_position(store, "ext", "BBBUSDT", at(1), pnl=-4.0, exit_time=at(5))
    add_position(store, "inproc", "DDDUSDT", at(2), pnl=2.5, exit_time=at(4))
    # open: 2 units bought at 100, last close 110 -> +20.00 unrealised
    add_position(store, "ext", "CCCUSDT", at(7), exit_time=at(7), cost=200.0, quantity=2.0)
    rig.feed.bars["CCCUSDT"] = flat_bars(40, 110.0)
    rig.feed.late("CCCUSDT", 3)
    add_position(store, "inproc", "EEEUSDT", at(8))  # pending entry: an open order
    add_batch(
        store,
        "ext",
        at(9),
        BatchStatus.ON_TIME,
        lateness=0.5,
        signals=[
            SignalRow("AAAUSDT", outcome=SignalOutcome.PLACED, reason=None),
            SignalRow("BBBUSDT", outcome=SignalOutcome.REJECTED, reason="max_positions reached"),
        ],
    )
    add_batch(store, "ext", at(10), BatchStatus.LATE, lateness=20.0)
    add_batch(store, "ext", at(11), BatchStatus.MISSED)
    fire(rig, "feed_down", EventLevel.ERROR, "the feed has been down for 5 minutes")
    fire(rig, "disk_low", EventLevel.WARNING, "2.0 GB free on the state disk")


class Served:
    """The API on a background event loop, with `call` to run a function against the rig."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.loop: asyncio.AbstractEventLoop | None = None
        self.rig: ApiRig | None = None
        self.ready = threading.Event()
        self.error: BaseException | None = None
        self._stop: asyncio.Event | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        try:
            asyncio.run(self._main())
        except BaseException as exc:
            self.error = exc
            self.ready.set()

    async def _main(self) -> None:
        self.loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        async with ApiRig(self.tmp_path) as rig:
            self.rig = rig
            seed(rig)
            self.ready.set()
            await self._stop.wait()

    def start(self) -> None:
        self._thread.start()
        assert self.ready.wait(30), "the API did not start"
        assert self.error is None, self.error

    def stop(self) -> None:
        assert self.loop is not None
        assert self._stop is not None
        self.loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(15)

    def call(self, fn: Callable[[ApiRig], Any]) -> Any:
        assert self.loop is not None
        assert self.rig is not None
        rig = self.rig

        async def run() -> Any:
            return fn(rig)

        return asyncio.run_coroutine_threadsafe(run(), self.loop).result(10)

    @property
    def url(self) -> str:
        assert self.rig is not None
        return self.rig.url + "/"


@pytest.fixture(scope="module")
def served(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Served]:
    server = Served(tmp_path_factory.mktemp("browser"))
    server.start()
    yield server
    server.stop()


@pytest.fixture(scope="session")
def browser_context_args() -> dict[str, Any]:
    return {
        "locale": "en-US",
        "timezone_id": "America/New_York",
        "viewport": {"width": 1280, "height": 900},
    }


@pytest.fixture
def errors(page: Page) -> list[str]:
    """Console errors and uncaught exceptions of the page."""
    found: list[str] = []

    def on_console(message: ConsoleMessage) -> None:
        if message.type == "error":
            found.append(message.text)

    page.on("console", on_console)
    page.on("pageerror", lambda exc: found.append(str(exc)))
    return found


@pytest.fixture
def dash(page: Page, served: Served, errors: list[str]) -> Page:
    """The overview, logged in."""
    page.goto(served.url)
    login(page)
    return page


def login(page: Page, token: str = TOKEN) -> None:
    page.get_by_label("API token").fill(token)
    page.get_by_role("button", name="Log in").click()
    expect(page.get_by_role("heading", name="Overview")).to_be_visible()


def table_text(page: Page, name: str) -> str:
    return page.get_by_role("table", name=name).inner_text()


# -- login ----------------------------------------------------------------------------------------


def test_a_wrong_token_shows_a_message_and_the_right_one_opens_the_overview(
    page: Page, served: Served, errors: list[str]
) -> None:
    response = page.goto(served.url)
    assert response is not None
    assert response.headers["content-security-policy"].startswith("default-src 'self'")
    expect(page.get_by_label("API token")).to_be_visible()
    page.get_by_label("API token").fill("not-the-token")
    page.get_by_role("button", name="Log in").click()
    expect(page.get_by_text("That token was not accepted.")).to_be_visible()
    assert page.evaluate("sessionStorage.length") == 0
    login(page)
    assert page.evaluate("sessionStorage.getItem('moonshot.token')") == TOKEN
    assert page.evaluate("localStorage.length") == 0
    assert TOKEN not in page.url
    # a 401 to the wrong token is the only console noise allowed here
    assert [e for e in errors if "401" not in e] == []


# -- the pages ------------------------------------------------------------------------------------


def test_the_overview_shows_the_daemon_alerts_and_one_row_per_strategy(dash: Page) -> None:
    facts = dash.get_by_label("Daemon state")
    expect(facts).to_contain_text("paper")
    expect(facts).to_contain_text("running")
    expect(facts).to_contain_text("trading")
    expect(dash.locator("#banner")).to_contain_text("2 unacknowledged alerts")
    expect(dash.locator("#banner")).to_have_class("error")
    rows = dash.get_by_role("table", name="Strategies").locator("tbody tr")
    expect(rows).to_have_count(2)
    ext = rows.filter(has_text="ext").first
    expect(ext).to_contain_text("1,000.00")  # capital
    expect(ext).to_contain_text("1,026.00")  # equity: 1000 + 6 realised + 20 unrealised
    expect(ext).to_contain_text("+6.00")
    expect(ext).to_contain_text("+20.00")
    expect(ext).to_contain_text("1 on time, 1 late, 1 missed")
    expect(dash.locator("#conn")).to_contain_text("Updated")
    # local time with UTC in the tooltip
    stamp = dash.locator("time").first
    assert stamp.get_attribute("title", timeout=2000) is not None
    assert str(stamp.get_attribute("title")).endswith(" UTC")


def test_a_strategy_page_has_the_chart_positions_trades_and_signals(dash: Page) -> None:
    dash.get_by_role("link", name="ext", exact=True).first.click()
    expect(dash.get_by_role("heading", name="ext", exact=True)).to_be_visible()
    chart = dash.get_by_role("img", name=re.compile("^Equity of the strategy"))
    expect(chart).to_have_attribute("aria-label", re.compile("2 realised points, now 1,026.00"))
    assert dash.locator("svg.equity path.line").count() >= 1
    expect(dash.locator("svg.equity circle.dot.now")).to_have_count(1)
    dash.locator("svg.equity rect.hit").hover(position={"x": 300, "y": 100})
    expect(dash.locator(".tooltip")).to_be_visible()
    dash.get_by_text("Table view").click()
    expect(dash.get_by_role("table", name="Equity values")).to_contain_text("1,010.00")
    expect(dash.get_by_role("table", name="Equity values")).to_contain_text("mark to market")
    assert "CCCUSDT" in table_text(dash, "Open positions")
    trades = table_text(dash, "Closed trades")
    assert "AAAUSDT" in trades
    assert "BBBUSDT" in trades
    assert "DDDUSDT" not in trades
    assert "max_positions reached" in table_text(dash, "Recent signals")


def test_positions_and_orders(dash: Page) -> None:
    dash.get_by_role("link", name="Positions").click()
    expect(dash.get_by_role("heading", name="Positions and orders")).to_be_visible()
    positions = dash.get_by_role("table", name="Open positions")
    expect(positions).to_contain_text("CCCUSDT")
    expect(positions).to_contain_text("110")
    expect(positions).to_contain_text("+20.00")
    expect(dash.get_by_role("table", name="Open orders")).to_contain_text("EEEUSDT")


def test_trades_are_filtered_by_strategy_and_pnl_is_not_colour_only(dash: Page) -> None:
    dash.get_by_role("link", name="Trades").click()
    table = dash.get_by_role("table", name="Closed trades")
    expect(table.locator("tbody tr")).to_have_count(3)
    expect(table).to_contain_text("+10.00")
    expect(table).to_contain_text("\u22124.00")  # a real minus sign
    expect(table.locator(".pnl-loss").first).to_contain_text("loss")
    expect(table.locator(".pnl-gain").first).to_contain_text("gain")
    dash.get_by_label("Strategy").select_option("inproc")
    expect(table.locator("tbody tr")).to_have_count(1)
    expect(table).to_contain_text("DDDUSDT")


def test_signals_and_providers(dash: Page) -> None:
    dash.get_by_role("link", name="Signals").click()
    providers = dash.get_by_role("table", name="Provider lateness")
    expect(providers).to_contain_text("external")
    row = providers.locator("tbody tr").filter(has_text="ext").first
    expect(row).to_contain_text("20 s")  # worst lateness
    signals = dash.get_by_role("table", name="Recent signals")
    expect(signals).to_contain_text("rejected")
    expect(signals).to_contain_text("max_positions reached")
    expect(signals).to_contain_text("placed")


def test_alerts_are_listed_and_can_be_acknowledged(dash: Page, served: Served) -> None:
    served.call(lambda rig: fire(rig, "backup_failed", EventLevel.WARNING, "no space left"))
    dash.get_by_role("link", name="Alerts").first.click()
    active = dash.get_by_role("table", name="Active alerts")
    expect(active).to_contain_text("feed_down")
    expect(active).to_contain_text("the feed has been down for 5 minutes")
    expect(active.get_by_role("button", name="Acknowledge backup_failed")).to_be_visible()
    before = dash.locator("#banner").inner_text()
    dash.get_by_role("button", name="Acknowledge backup_failed").click()
    expect(active.locator("tr", has_text="backup_failed")).to_contain_text("acknowledged")
    assert dash.locator("#banner").inner_text() != before
    expect(dash.get_by_role("table", name="Alert history")).to_contain_text("firing")
    feed = dash.get_by_role("table", name="Events")
    expect(feed).to_contain_text("alert_acknowledged")
    dash.get_by_label("Level").select_option("error")
    expect(feed).not_to_contain_text("alert_acknowledged")
    expect(feed).to_contain_text("feed_down")


# -- controls -------------------------------------------------------------------------------------


def test_pause_and_resume_ask_for_confirmation(dash: Page) -> None:
    dialog = dash.get_by_role("dialog")
    expect(dash.locator("#resume")).to_be_disabled()  # only the control that changes something
    dash.get_by_role("button", name="Pause").click()
    expect(dialog).to_be_visible()
    dialog.get_by_role("button", name="Cancel").click()
    expect(dash.locator("#controls-state")).to_contain_text("Trading is running")
    dash.get_by_role("button", name="Pause").click()
    dialog.get_by_role("button", name="Pause").click()
    expect(dash.locator("#controls-state")).to_contain_text("Trading is paused")
    expect(dash.get_by_label("Daemon state")).to_contain_text("paused")
    expect(dash.locator("#pause")).to_be_disabled()
    dash.get_by_role("button", name="Resume").click()
    dialog.get_by_role("button", name="Resume").click()
    expect(dash.locator("#controls-state")).to_contain_text("Trading is running")


def test_flatten_needs_the_word_typed(dash: Page, served: Served) -> None:
    dash.get_by_role("button", name="Flatten").click()
    dialog = dash.get_by_role("dialog")
    confirm = dialog.get_by_role("button", name="Flatten")
    expect(confirm).to_be_disabled()
    dialog.get_by_label("Type flatten to confirm").fill("flat")
    expect(confirm).to_be_disabled()
    dialog.get_by_label("Type flatten to confirm").fill("flatten")
    expect(confirm).to_be_enabled()
    confirm.click()
    expect(dash.locator("#controls-state")).to_contain_text("Flatten is requested")
    expect(dash.get_by_label("Daemon state")).to_contain_text("flatten requested")
    served.call(lambda rig: rig.store.set_control(Control.FLATTEN_REQUESTED, False))


# -- session --------------------------------------------------------------------------------------


def test_logout_clears_the_token(dash: Page) -> None:
    dash.get_by_role("button", name="Log out").click()
    expect(dash.get_by_label("API token")).to_be_visible()
    assert dash.evaluate("sessionStorage.getItem('moonshot.token')") is None
    dash.reload()
    expect(dash.get_by_label("API token")).to_be_visible()


def test_a_401_returns_to_the_login_with_a_message(dash: Page) -> None:
    def reject(route: Route, request: Request) -> None:
        route.fulfill(status=401, json={"detail": "missing or wrong API token"})

    dash.route("**/api/v1/status", reject)
    dash.get_by_role("link", name="Trades").click()
    expect(dash.get_by_text("The API rejected the token. Log in again.")).to_be_visible()
    expect(dash.get_by_label("API token")).to_be_visible()
    assert dash.evaluate("sessionStorage.getItem('moonshot.token')") is None


def test_the_page_says_when_the_connection_is_lost(dash: Page, page: Page) -> None:
    context: BrowserContext = page.context
    context.set_offline(True)
    dash.get_by_role("link", name="Trades").click()
    expect(dash.locator("#conn")).to_contain_text("Connection lost")
    expect(dash.locator("#conn")).to_contain_text("Last updated")
    context.set_offline(False)
    dash.get_by_role("link", name="Overview").click()
    expect(dash.locator("#conn")).to_contain_text("Updated")


def test_every_page_loads_without_console_errors(dash: Page, errors: list[str]) -> None:
    for name, heading in [
        ("Positions", "Positions and orders"),
        ("Trades", "Trades"),
        ("Signals", "Signals and providers"),
        ("Alerts", "Alerts and events"),
        ("Overview", "Overview"),
    ]:
        dash.get_by_role("link", name=name, exact=True).first.click()
        expect(dash.get_by_role("heading", name=heading, exact=True)).to_be_visible()
        dash.wait_for_load_state("networkidle")
    dash.set_viewport_size({"width": 390, "height": 800})  # a phone
    expect(dash.get_by_role("heading", name="Overview")).to_be_visible()
    assert errors == []
