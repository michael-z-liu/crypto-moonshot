"""X10: book-level sizing rules on LMC / LMCF (in-sample exploration).

Pre-declared rules, each using only trades closed before the new entry:

- momentum: stake weight by the sleeve's trailing 180-day mean net (>= 5 closed trades):
  > +5% -> 1.25, < 0 -> 0.5, else 1.0;
- loss brake: weight 0.5 for 7 days after a same-sleeve trade closed at <= -30%;
- book brake: weight 0.5 for 7 days after any trade closed at <= -30%;
- inverse momentum (control): the opposite of momentum.

Trades: the cached sleeve frame from x8_lib (L, C, F, M; every candidate, no slot limit).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from common_ls import OWN
from e25_combined import book

OUT = Path(__file__).resolve().parent / "results" / "explore"
COLS = ["symbol", "adv", "t_in", "t_out", "net", "qv", "sleeve", "prio", "w"]
DAY = 86400


def weights(t: pl.DataFrame, rule: str) -> np.ndarray:
    t_in, t_out = t["t_in"].to_numpy(), t["t_out"].to_numpy()
    net, sl = t["net"].to_numpy(), t["sleeve"].to_numpy()
    w = np.ones(t.height)
    for i in range(t.height):
        done = t_out < t_in[i]
        if rule in ("momentum", "inverse"):
            m = done & (sl == sl[i]) & (t_out >= t_in[i] - 180 * DAY)
            if m.sum() >= 5:
                mu = net[m].mean()
                hi, lo = (1.25, 0.5) if rule == "momentum" else (0.5, 1.25)
                w[i] = hi if mu > 0.05 else lo if mu < 0 else 1.0
        elif rule in ("loss_brake", "book_brake"):
            m = done & (net <= -0.30) & (t_out >= t_in[i] - 7 * DAY)
            if rule == "loss_brake":
                m &= sl == sl[i]
            if m.any():
                w[i] = 0.5
    return w


def run(t: pl.DataFrame, rule: str) -> dict[str, object]:
    w = weights(t, rule) if rule != "base" else np.ones(t.height)
    b = book(t.with_columns(w=pl.Series(w)).select(COLS), 10)
    n = int(b["trades"].sum())
    return {
        "bp": float((b["bp"] * b["trades"]).sum() / n),
        "n": n,
        "min": float(b["ret"].min()),
        "dd": float(b["dd"].min()),
        "years": "/".join(f"{r['ret'] * 100:+.0f}" for r in b.iter_rows(named=True)),
        "w_mean": float(w.mean()),
    }


def predeclared() -> None:
    s = pl.read_parquet(OWN / "explore" / "x8_sleeves.parquet").sort("t_in")
    lines = [
        "# X10: book-level sizing rules (10 slots, $10k a year, in-sample)",
        "",
        "| book | rule | trades | bp/trade | min year % | worst DD % | years 22..26 % | mean w |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for combo in ("LMC", "LMCF"):
        t = s.filter(pl.col("sleeve").is_in(list(combo)))
        for rule in ("base", "momentum", "inverse", "loss_brake", "book_brake"):
            r = run(t, rule)
            row = (f"| {combo} | {rule} | {r['n']} | {r['bp']:+.0f} | {r['min'] * 100:+.0f} | "
                   f"{r['dd'] * 100:+.0f} | {r['years']} | {r['w_mean']:.2f} |")  # fmt: skip
            lines.append(row)
            print(row, flush=True)
    (OUT / "meta_sizing.md").write_text("\n".join(lines) + "\n")


def brake_w(t: pl.DataFrame, thr: float, days: int, wt: float) -> np.ndarray:
    t_in, t_out = t["t_in"].to_numpy(), t["t_out"].to_numpy()
    net, sl = t["net"].to_numpy(), t["sleeve"].to_numpy()
    w = np.ones(t.height)
    for i in range(t.height):
        m = (t_out < t_in[i]) & (net <= thr) & (t_out >= t_in[i] - days * DAY) & (sl == sl[i])
        if m.any():
            w[i] = wt
    return w


def neighbours() -> None:
    """Loss-brake neighbourhood (run after the pre-declared table)."""
    s = pl.read_parquet(OWN / "explore" / "x8_sleeves.parquet").sort("t_in")
    lines = ["", "## Loss-brake neighbourhood (same-sleeve trade closed <= thr in the last N days)", "",  # noqa: E501
             "| book | thr | days | weight | bp/trade | min year % | worst DD % | years 22..26 % | braked trades |",  # noqa: E501
             "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for combo in ("LMC", "LMCF"):
        t = s.filter(pl.col("sleeve").is_in(list(combo)))
        for thr in (-0.2, -0.3, -0.5):
            for days in (3, 7, 14):
                for wt in (0.0, 0.5):
                    w = brake_w(t, thr, days, wt)
                    tw = t.with_columns(w=pl.Series(w))
                    if wt == 0.0:  # book() reads w=0 as 1, so a skip drops the trade instead
                        tw = tw.filter(pl.col("w") > 0)
                    b = book(tw.select(COLS), 10)
                    n = int(b["trades"].sum())
                    row = (f"| {combo} | {thr:+.0%} | {days} | {wt} | {(b['bp'] * b['trades']).sum() / n:+.0f} | "  # noqa: E501
                           f"{b['ret'].min() * 100:+.0f} | {b['dd'].min() * 100:+.0f} | "
                           + "/".join(f"{r['ret'] * 100:+.0f}" for r in b.iter_rows(named=True))
                           + f" | {int((w < 1).sum())} |")  # fmt: skip
                    lines.append(row)
                    print(row, flush=True)
    with (OUT / "meta_sizing.md").open("a") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    predeclared()
    neighbours()
