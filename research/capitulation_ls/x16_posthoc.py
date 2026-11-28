# ruff: noqa: E501
"""X16 post-hoc checks (written AFTER seeing the x16_listing_longs.py grid; nothing here is a
pre-declared test, every number is labelled post-hoc in the note).

1. Concentration and a coin-level bootstrap of the two picks (A: 1h entry, r1h > 15%, 24 h,
   TP 50%, stop 30%; B: 72 h check, above the reference, 30 d, no exits) and of the B cell that
   stayed positive in 2025-26 (72 h, above 0-20%, 30 d, stop 50%).
2. First-hour quote volume (qv_1h, bars 0-11, known at a 1 h entry) as a split for the 1 h entry:
   the A calibration showed the top first-DAY volume decile doing well, but first-day volume is
   only known at 24 h. Terciles with cut points from 2021-24 listings; hold {24 h, 72 h} x TP
   {50%, none} x stop {30%, none}.
3. M's squeezes from the long side: the e27 strong complement split by how its long trade ended.

Writes results/explore/listing_longs_posthoc.md.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import polars as pl
from common_ls import OWN
from x16_listing_longs import fmt_tp, listings, paths, pct, subset, summary, trades, yr_cells

OUT = Path(__file__).resolve().parent / "results" / "explore"


def boot(t: pl.DataFrame, n: int = 5000, seed: int = 0) -> tuple[float, float]:
    """95% interval of the mean net, coins resampled (one trade per coin here)."""
    x = t["net"].to_numpy()
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def conc(name: str, t: pl.DataFrame) -> str:
    s = t.sort("net", descending=True)
    tot = float(s["net"].sum())
    lo, hi = boot(t)
    a = t.filter(pl.col("year") <= 2024)
    lo2, hi2 = boot(a)
    top = ", ".join(
        f"{r['symbol'].removesuffix('USDT')} {r['year']} {r['net'] * 100:+.0f}%"
        for r in s.head(3).iter_rows(named=True)
    )
    return (
        f"| {name} | {t.height} | {pct(t['net'].mean())} [{pct(lo)}, {pct(hi)}] | {pct(a['net'].mean())} [{pct(lo2)}, {pct(hi2)}] | {pct(t['net'].median())} | "
        f"{top} | {s['net'].head(3).sum() / tot:.2f} | {pct(s['net'].tail(t.height - 3).mean())} | {pct(s['net'].tail(t.height - 5).mean())} |"
    )


def main() -> None:
    L = listings()
    P = paths()
    allx = {int(i): 0 for i in L["i"]}
    md = [
        "# X16 post-hoc checks (labelled post-hoc: chosen after seeing the grid)",
        "",
        "Script `x16_posthoc.py`.",
        "",
    ]

    md += ["## 1. Concentration and coin bootstrap", "",
           "| config | n | mean [95% coin bootstrap] | 2021-24 mean [95%] | median | top 3 trades | top-3 share of total P&L | mean without top 3 | without top 5 |",
           "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    ta = subset(trades(dict.fromkeys(allx, 12), 288, 0.5, 0.3), "r1h", 0.15, 9e9)
    tb_full = trades(dict.fromkeys(allx, 864), 8640, None, None)
    tb = subset(tb_full, "since", 0.0, 9e9)
    tc = subset(trades(dict.fromkeys(allx, 864), 8640, None, 0.5), "since", 0.0, 0.2)
    md += [conc("A pick: 1h, r1h>15%, 24h, TP50 stop30", ta), conc("B pick: 72h, above>0, 30d, no exits", tb),
           conc("B cell: 72h, above 0-20%, 30d, stop 50%", tc)]  # fmt: skip

    # 2. first-hour volume split for the 1 h entry
    qv1h = np.nansum(P["qv"][:, :12], axis=1)
    Lq = L.with_columns(qv_1h=pl.Series(qv1h))
    c1, c2 = np.quantile(Lq.filter(pl.col("year") <= 2024)["qv_1h"].to_numpy(), [1 / 3, 2 / 3])
    md += ["", "## 2. First-hour volume split, 1 h entry (post-hoc)", "",
           f"Tercile cut points from 2021-24 listings: qv_1h {c1 / 1e6:.1f}M and {c2 / 1e6:.1f}M USDT.", "",
           "| hold | TP | stop | qv_1h tercile | n | mean | median | stopped | e21_24 (n) | e25_26 (n) | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |",
           "|" + "---|" * 16]  # fmt: skip
    for (hn, h), tp, sl in itertools.product(
        {"24h": 288, "72h": 864}.items(), (0.5, None), (0.3, None)
    ):
        t = trades(dict.fromkeys(allx, 12), h, tp, sl).join(Lq.select("i", "qv_1h"), on="i")
        for nm, lo, hi in (("low", -1.0, c1), ("mid", c1, c2), ("high", c2, 1e15)):
            s = summary(subset(t, "qv_1h", lo, hi))
            md.append(f"| {hn} | {fmt_tp(tp)} | {fmt_tp(sl)} | {nm} | {s['n']} | {pct(s['mean'])} | {pct(s['median'])} | {float(s['stopped']):.2f} | "
                      f"{pct(s['e21_24'])} ({s['n21_24']}) | {pct(s['e25_26'])} ({s['n25_26']}) | {yr_cells(s)} |")  # fmt: skip

    # 3. M's strong complement: what a long sees
    idx = {s: int(i) for s, i in zip(L["symbol"], L["i"], strict=True)}
    e27 = pl.read_parquet(OWN / "e27_listings.parquet")
    st = e27.filter(pl.col("since_list") > 0)
    e_of = {
        idx[r["symbol"]]: (int(r["t_in"]) - int(P["listed"][idx[r["symbol"]]])) // 300
        for r in st.iter_rows(named=True)
        if r["symbol"] in idx
    }
    lt = trades(e_of, 8640, 0.5, 0.5).join(L.select("i", "symbol"), on=["i", "symbol"])
    j = lt.join(st.select("symbol", short_net="net", since_list="since_list"), on="symbol")
    md += ["", "## 3. M's strong complement (58 listings above the perp listing price at +72 h), long vs short", "",
           "Long = spot at M's entry time, 30 d, TP 50%, stop 50%; short = M's own e27 trade (perp, 30 d, TP/stop 50%, funding).", "",
           "| long exit | n | long mean | short mean (same coins) | sum long + short (per pair, 2 stakes) | median since_list |", "|---|---|---|---|---|---|"]  # fmt: skip
    for k in ("tp", "stop", "time"):
        g = j.filter(pl.col("kind") == k)
        if g.height:
            md.append(
                f"| {k} | {g.height} | {pct(g['net'].mean())} | {pct(g['short_net'].mean())} | {pct((g['net'] + g['short_net']).mean() / 2)} | {pct(g['since_list'].median())} |"
            )
    md.append(
        f"| all | {j.height} | {pct(j['net'].mean())} | {pct(j['short_net'].mean())} | {pct((j['net'] + j['short_net']).mean() / 2)} | {pct(j['since_list'].median())} |"
    )
    corr = float(np.corrcoef(j["net"].to_numpy(), j["short_net"].to_numpy())[0, 1])
    md += ["", f"Correlation of the long and short net on the same coin: {corr:+.2f} (n={j.height}). "
           f"Both stopped/TP at 50% in opposite directions: TP hits of the long = {int((j['kind'] == 'tp').sum())}, short stops in e27 = {int((j['short_net'] <= -0.45).sum())}."]  # fmt: skip
    (OUT / "listing_longs_posthoc.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
