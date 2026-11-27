"""Forward paper read-out of the pre-registered LMC book (docs/plans/capitulation-long-short.md).

Signals after 2026-11-02:
- L, C from `forward_signals.parquet` (run forward_signals.py first, after syncing spot candles);
- M from Binance's public spot exchangeInfo (new USDT listings) and the public futures API.
Outcomes: L on the shared spot 5m store; C and M on perp klines and funding from the public
futures API (fapi.binance.com, no keys). Open positions are marked at the last price.
Writes results/forward.md. Places no orders.

Run: uv run --group research python lmc_forward.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
from common_ls import CACHE_ALL, OWN

os.environ["MOONSHOT_UNIVERSE"] = "all"
sys.path.insert(0, str(CACHE_ALL.parent))
from common import load  # noqa: E402

START = datetime(2026, 11, 2, tzinfo=UTC)
SLOTS, EQ0 = 10, 10_000.0
HERE = Path(__file__).resolve().parent
FAPI, SAPI = "https://fapi.binance.com", "https://api.binance.com"


def get(url: str) -> object:
    for i in range(4):
        try:
            return json.loads(urllib.request.urlopen(url, timeout=30).read())
        except Exception:  # noqa: BLE001
            time.sleep(1 + i)
    raise RuntimeError(url)


def fklines(sym: str, interval: str, start_ms: int, limit: int = 1500) -> np.ndarray:
    """[n, 6] open time (s), open, high, low, close, quote volume from the futures API."""
    rows = get(f"{FAPI}/fapi/v1/klines?symbol={sym}&interval={interval}&startTime={start_ms}&limit={limit}")
    a = np.array([[r[0] / 1000, float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[7])] for r in rows])  # type: ignore[union-attr]
    return a.reshape(-1, 6)


def funding(sym: str, t0: int, t1: int) -> float:
    rows = get(f"{FAPI}/fapi/v1/fundingRate?symbol={sym}&startTime={t0 * 1000}&endTime={t1 * 1000}&limit=1000")
    return float(sum(float(r["fundingRate"]) for r in rows))  # type: ignore[union-attr]


_PERPS: set[str] | None = None


def perp_of(spot: str) -> str | None:
    global _PERPS
    if _PERPS is None:
        info = get(f"{FAPI}/fapi/v1/exchangeInfo")
        _PERPS = {s["symbol"] for s in info["symbols"] if s["contractType"] == "PERPETUAL"}  # type: ignore[index]
    base = spot.removesuffix("USDT")
    for p in (spot, f"1000{base}USDT", f"1000000{base}USDT", f"1M{base}USDT"):
        if p in _PERPS:
            return p
    return None


def long_trade(r: dict) -> dict | None:
    df = load(r["symbol"]).filter(pl.col("open_time") > r["open_time"]).head(72 * 12)
    if df.height == 0:
        return None
    e = float(df["open"][0])
    hi, cl = df["high"].to_numpy() / e - 1, df["close"].to_numpy() / e - 1
    k = np.flatnonzero(hi >= 0.2)
    status = "closed" if (len(k) or df.height == 72 * 12) else "open"
    ret = 0.2 if len(k) else float(cl[-1])
    t_out = df["open_time"][int(k[0]) if len(k) else df.height - 1]
    return {"sleeve": "L", "symbol": r["symbol"], "t_in": df["open_time"][0], "t_out": t_out, "net": ret - 0.004, "status": status}


def short_trade(sleeve: str, perp: str, t_entry_s: int, hold_h: int, tp: float | None, sl: float, interval: str) -> dict | None:
    k = fklines(perp, interval, t_entry_s * 1000, limit=1500)
    step = 300 if interval == "5m" else 3600
    k = k[k[:, 0] >= t_entry_s][: hold_h * 3600 // step]
    if len(k) == 0:
        return None
    e = k[0, 1]
    up, dn, cl = k[:, 2] / e - 1, k[:, 3] / e - 1, k[:, 4] / e - 1
    ks, kt = np.flatnonzero(up >= sl), (np.flatnonzero(-dn >= tp) if tp else np.array([], int))
    i_s, i_t = (ks[0] if len(ks) else 10**9), (kt[0] if len(kt) else 10**9)
    if i_s <= i_t and i_s < 10**9:
        ret, j = -sl - (0.0 if sl >= 1.0 else 0.02), i_s
    elif i_t < 10**9:
        ret, j = tp, i_t
    else:
        ret, j = -float(cl[-1]), len(k) - 1
    done = (i_s < 10**9 or i_t < 10**9) or len(k) >= hold_h * 3600 // step
    t_out = int(k[j, 0]) + step
    f = funding(perp, t_entry_s, t_out)
    return {"sleeve": sleeve, "symbol": perp, "t_in": datetime.fromtimestamp(t_entry_s, UTC), "t_out": datetime.fromtimestamp(t_out, UTC),
            "net": ret - 0.003 + f, "status": "closed" if done else "open"}  # fmt: skip


def listings() -> list[dict]:
    """New USDT spot listings whose 72h entry falls after START (first spot 1h candle = listing)."""
    info = get(f"{SAPI}/api/v3/exchangeInfo?permissions=SPOT")
    syms = [s["symbol"] for s in info["symbols"] if s["quoteAsset"] == "USDT" and s["status"] == "TRADING"]  # type: ignore[index]
    since = int(datetime(2026, 10, 25, tzinfo=UTC).timestamp() * 1000)
    out = []
    for s in syms:
        rows = get(f"{SAPI}/api/v3/klines?symbol={s}&interval=1h&startTime={since}&limit=2")
        if not rows or rows[0][0] <= since:  # type: ignore[index]
            continue  # listed before the window
        t0 = int(rows[0][0] / 1000)  # type: ignore[index]
        out.append({"symbol": s, "listed": t0, "p_list": float(rows[0][1])})  # type: ignore[index]
    return out


def main() -> None:
    f = pl.read_parquet(OWN / "forward_signals.parquet").filter(pl.col("open_time") >= START).sort("open_time")
    trades = []
    for r in f.filter(pl.col("pct") >= 0.995).iter_rows(named=True):
        if (t := long_trade(r)) is not None:
            trades.append(t)
    for r in f.filter((pl.col("ret_4h") <= -0.15) & (pl.col("btc_ret_4h").abs() <= 0.01) & (pl.col("pct") <= 0.5) & (pl.col("qv_base") >= 20_000)).iter_rows(named=True):
        p = perp_of(r["symbol"])
        if p is not None and (t := short_trade("C", p, int(r["open_time"].timestamp()) + 300, 24, None, 1.0, "5m")) is not None:
            trades.append(t)
    for li in listings():
        p = perp_of(li["symbol"])
        t_e = (li["listed"] // 3600 + 72) * 3600
        if p is None or t_e < START.timestamp() or t_e > time.time() - 3600:
            continue
        k = fklines(p, "1h", (t_e - 3600) * 1000, limit=2)
        if len(k) < 2 or k[0, 5] < 50_000 or k[1, 1] > li["p_list"]:
            continue
        if (t := short_trade("M", p, t_e, 30 * 24, 0.5, 0.5, "1h")) is not None:
            trades.append(t)
    d = pl.DataFrame(trades) if trades else pl.DataFrame({"sleeve": [], "symbol": [], "t_in": [], "t_out": [], "net": [], "status": []})
    prio = {"L": 0, "C": 1, "M": 2}
    d = d.with_columns(p=pl.col("sleeve").replace_strict(prio, default=3)).sort("t_in", "p") if d.height else d
    eq, cash, pos, taken = EQ0, EQ0, [], []
    for r in d.iter_rows(named=True):
        for q in [q for q in pos if q[0] <= r["t_in"]]:
            cash += q[1] * (1 + q[2]); eq += q[1] * q[2]; pos.remove(q)
        if len(pos) >= SLOTS or any(q[3] == r["symbol"] for q in pos):
            continue
        stake = min(eq / SLOTS, cash)
        cash -= stake
        pos.append((r["t_out"], stake, r["net"], r["symbol"]))
        taken.append({**r, "stake": stake})
    for q in pos:
        eq += q[1] * q[2]
    t = pl.DataFrame(taken) if taken else d
    closed = t.filter(pl.col("status") == "closed") if t.height else t
    lines = ["# Forward read-out: LMC (pre-registered)", "", f"Signals from {START:%Y-%m-%d} to the scored data end; updated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC.", "",
             f"- Trades taken: {t.height} (closed {closed.height}); mean net per closed trade: " + (f"{closed['net'].mean() * 1e4:+.0f} bp" if closed.height else "n/a"),
             f"- Book: ${eq:,.0f} from ${EQ0:,.0f} ({eq / EQ0 - 1:+.1%}), open positions marked at the last price.", "", "| Sleeve | Symbol | Entry | Exit | Net | Status |", "|---|---|---|---|---|---|"]  # fmt: skip
    for r in t.iter_rows(named=True) if t.height else []:
        lines.append(f"| {r['sleeve']} | {r['symbol']} | {r['t_in']:%m-%d %H:%M} | {r['t_out']:%m-%d %H:%M} | {r['net'] * 100:+.1f}% | {r['status']} |")
    (HERE / "results" / "forward.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:7]))


if __name__ == "__main__":
    main()
