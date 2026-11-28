# ruff: noqa: E501
"""X8 families B and C: daily baskets (in-sample exploration; every configuration is listed).

B. Basket short of the day's pumps: each day at a fixed hour H (00 or 12 UTC), short the top-K
   24h gainers among active USD-M perps (K = 3, 5, 10), with or without the funding gate (last
   settled rate >= 0), equal weight, hold 1/3/7 days, isolated 1x (liquidation at +100% = stake
   lost), 0.3% round trip, funding paid; optionally with a BTC perp long of equal notional
   (capital split 1x/1x, so the hedged trade's return is (short + BTC long) / 2).
C. Basket long of the biggest losers on BTC flush days (BTC 24h <= -8%): top-K spot coins by
   24h loss (K = 3, 5, 10), hold 1/2/3 days, time exit or take profit 20% (as L), 0.4% round
   trip. Trigger: the first hour BTC's 24h return is <= -8% (then 24h lock), or 00 UTC daily.

Signals use closed hourly bars only; entry = the open of the next hour. Gates: perp quote volume
>= $50k (moving price) in each of the two prior hours and >= 24h of history; spot quote volume
>= $10k in each of the two prior hours and >= $1M in the prior 24h. Universe = every perp / spot
pair in the caches, delisted included; a coin delisted during a hold exits at its last close.
Books: e25 `book`, each year from $10k, stake <= 0.5% of the coin's prior 24h quote volume.

Run: uv run --group research python x8_baskets.py  -> results/explore/baskets.md
"""

from __future__ import annotations

import itertools

import numpy as np
import polars as pl
from diag import table
from e25_combined import book
from x8_lib import (
    NH,
    OUT,
    T0,
    as_book,
    book_stats,
    card,
    funding_matrix,
    perp_h,
    run_book,
    sleeves,
    spot_h,
)

VERDICT = """## Verdicts

| family | configs tried | best E[o] per trade | +10%/trade target | best LMC + addition (2022-26, 10 slots) | verdict |
|---|---|---|---|---|---|
| B pump basket short | 72 (2 hours x gate x 3 holds x K 3/5/10 x BTC hedge) | **-0.14%** (H00, funding gate, 1d, K3) | no: all 72 negative | worst year +5% (LMC alone +51%), 64 bp/trade | **fails** |
| B, 24h gain >= 40% only | 8 (2 hours x gate x 1/3d, K3) | -0.05% | no: all 8 negative | worst year -8% | **fails** |
| C flush-loser basket long | 36 (2 triggers x 3 holds x time/TP20 x K 3/5/10) | +5.2% (event, 3d, TP 20%, K3) | no | worst year +48%, 1082 bp (LMC +51%, 1158 bp) | **fails**: positive, but below target and it dilutes LMC |

No configuration improves LMC or LMCF on bp per trade or on worst year. The best C addition is nearly neutral
(LMCF +99% vs +101% worst year), and every B addition harms the book badly.

## Why B failed

* **The daily top gainers do not mean-revert enough to pay for costs and the squeeze tail.** The median name earns
  +1..+4%, but the mean is negative in all 80 configurations. Decomposition (H00, naked): the short's gross price move is
  +0.1% at 1 day and -0.3..-1.4% at 3-7 days. Liquidations (+100%) hit 0.5-5.7% of names and cost 0.5-5.7 points per
  trade, while the names that are not liquidated net only +0.2..+3.8%. Diag: 37% `never_went`, median giveback 0.79.
* **Funding.** Without the gate, the top gainers' crowded shorts pay 0.1-0.7% per trade. The funding gate turns that into
  +0.04..+0.26% received and halves the loss (H00 1d K3: -0.53% -> -0.14%), but no config turns positive.
* **The BTC hedge removes market beta, not the idiosyncratic squeeze.** It halves the variance and the loss per unit of
  capital, but E[o] stays at -0.2..-1.1%.
* **Diversification does not create an edge.** Larger K (5, 10) only moves toward the average of all gainers (K10 naked: -0.2..-1.8%). The basket
  diversifies the tail, as intended, but averages a negative edge. F's +9.5% comes from the *event*: entering within
  minutes of a >= 40% 4h spike, which is the 5m-resolution reversal. Even restricted to 24h gains >= 40% (802-1276 trades),
  a fixed-hour basket stays at -0.05..-3.2% with 3.4-9.5% liquidations.
* **In the book, a daily basket fills slots every day.** It displaces L, M and C entries. LMC + B: worst year
  +51% -> +5% at best and -99% at worst; bp/trade 1158 -> -168..+79. With the >= 40% subset: worst year -8%..-80%.

## Why C failed

* **The CNN has already taken the good flush days.** On the 9 flush signals in 2022-26 that L also traded, the loser basket
  earns +7..+11% per name (L earns +10% on those days). On the other 27 flush signals it earns -0.2..+2.7%. BTC -8% days
  without a capitulation signature keep falling (LUNA 2022-05-09: K5 basket -57%; FTX 2022-11-08: -18%; 2022-01-21: -22%).
  The 2020-21 bull-market dips pay +3.7..+6.6%, so the result also depends on the regime.
* **Too few events for a sleeve, and 2022 loses.** There are 78 event signals over 6.8 years (27 at the 00 UTC check);
  2023 has 2 and 2026 has 1. The best-E[o] config (event, 3d, TP 20%, K3), as a standalone 10-slot book: 2020 +17%,
  2021 +122%, 2022 -27%, 2023 +4%, 2024 +29%, 2025 +10%, 2026 +0%.
* **In the LMC book it competes with L for the same days.** The basket triggers on the first -8% hour, often before L's
  CNN signals, and takes slots that L fills with better trades. LMC worst year: +51% -> +48% at best and -68% at worst
  (K10, 3d hold); bp/trade: 1158 -> 643..1085.

"""

PERP_COST, SPOT_COST = 0.003, 0.004
EXCLUDE = {"BLUEBIRDUSDT", "BTCDOMUSDT", "DEFIUSDT", "FOOTBALLUSDT", "USDCUSDT"}
YEARS = range(2021, 2027)


def _last_finite(c: np.ndarray, a: int, b: int) -> int:
    """Index of the last finite close in [a, b] (a if none)."""
    seg = np.isfinite(c[a : b + 1])
    k = np.flatnonzero(seg)
    return a + int(k[-1]) if len(k) else a


# ---------------------------------------------------------------- B: pump basket shorts --------
def pump_trades(
    hour: int, hold_d: int, gate: bool, kmax: int = 10, min_gain: float | None = None
) -> pl.DataFrame:
    """Top-kmax 24h gainers each day at `hour` (rank kept), simulated as 1x perp shorts."""
    P = perp_h()
    o, hi, lo, c, qv = P["open"], P["high"], P["low"], P["close"], P["qv"]
    syms, spots = P["sym"], P["spot"]
    ok_col = np.array([s not in EXCLUDE and s != "BTCUSDT" for s in syms])
    jb = int(np.flatnonzero(syms == "BTCUSDT")[0])
    hours = np.arange(24 + hour + 24, NH - 1, 24)  # signal hours (entry bar index)
    cum, last = funding_matrix(spots, hours)
    hold = hold_d * 24
    ends = np.minimum(hours + hold - 1, NH - 1)
    cum_end = funding_matrix(spots, ends + 1)[0]  # settlements in (t_in, t_out] are paid
    rows = []
    for n, i in enumerate(hours):
        r24 = c[i - 1] / c[i - 25] - 1
        act = (qv[i - 1] >= 50_000) & (qv[i - 2] >= 50_000) & np.isfinite(o[i]) & ok_col
        act &= np.isfinite(r24) & np.isfinite(c[i - 25])
        if min_gain is not None:
            act &= np.nan_to_num(r24, nan=-1.0) >= min_gain
        if gate:
            act &= np.nan_to_num(last[n], nan=0.0) >= 0
        idx = np.flatnonzero(act)
        if len(idx) == 0:
            continue
        top = idx[np.argsort(-r24[idx])][:kmax]
        end = int(ends[n])
        t_in = T0 + int(i) * 3600
        cend = cum_end[n]
        # BTC long hedge leg
        kb = _last_finite(c[:, jb], i, end)
        btc = c[kb, jb] / o[i, jb] - 1 - PERP_COST - (cend[jb] - cum[n, jb])
        for rank, j in enumerate(top):
            e0 = float(o[i, j])
            k_end = _last_finite(c[:, j], i, end)
            h = hi[i : k_end + 1, j]
            lw = lo[i : k_end + 1, j]
            up = np.nanmax(h) / e0 - 1
            mfe = float(1 - np.nanmin(lw) / e0)
            if up >= 1.0:  # liquidation at +100% (isolated 1x): the stake is lost
                kl = i + int(np.flatnonzero(h / e0 - 1 >= 1.0)[0])
                net, kind, xo, gross, f = -1.0, "liq", kl, -1.0, 0.0
                mfe = float(1 - np.nanmin(lo[i : kl + 1, j]) / e0)
            else:
                f = (cend[j] - cum[n, j]) if np.isfinite(cum[n, j]) else 0.0
                f = 0.0 if not np.isfinite(f) else f
                gross = -(c[k_end, j] / e0 - 1)
                net = gross - PERP_COST + f  # short receives +rate
                net, kind, xo = max(net, -1.0), "time", k_end
            rows.append({
                "symbol": str(spots[j]), "t_in": t_in, "t_out": T0 + int(xo + 1) * 3600,
                "rank": rank, "r24": float(r24[j]), "net": net, "btc": float(btc),
                "mfe": mfe, "mae": float(min(up, 1.0)), "exit_kind": kind,
                "qv": float(qv[i - 24 : i, j].sum()), "fund_last": float(last[n, j]),
                "gross": float(gross), "fund": float(f),
            })  # fmt: skip
    d = pl.DataFrame(rows)
    return d.with_columns(year=pl.from_epoch("t_in").dt.year(), day=pl.from_epoch("t_in").dt.date())


# ---------------------------------------------------------------- C: flush loser baskets -------
def flush_trades(mode: str, hold_d: int, tp: float | None, kmax: int = 10) -> pl.DataFrame:
    S = spot_h()
    o, hi, c, qv, syms = S["open"], S["high"], S["close"], S["qv"], S["syms"]
    jb = int(np.flatnonzero(syms == "BTCUSDT")[0])
    btc24 = c[:, jb] / np.roll(c[:, jb], 24) - 1
    btc24 = np.roll(btc24, 1)  # value known at the start of hour i (closes up to i-1)
    btc24[:26] = np.nan
    if mode == "event":
        sig, lock = [], -(10**9)
        for i in np.flatnonzero(np.nan_to_num(btc24, nan=0.0) <= -0.08):
            if i - lock >= 24:
                sig.append(i)
                lock = i
    else:  # 00 UTC daily check
        cand = np.arange(48, len(btc24) - 1, 24)
        sig = [int(i) for i in cand if np.nan_to_num(btc24[i], nan=0.0) <= -0.08]
    hold = hold_d * 24
    rows = []
    nh = c.shape[0]
    for i in sig:
        if i + 1 >= nh:
            continue
        r24 = c[i - 1] / c[i - 25] - 1
        q24 = np.nansum(qv[i - 24 : i], axis=0)
        act = (np.nan_to_num(qv[i - 1]) >= 10_000) & (np.nan_to_num(qv[i - 2]) >= 10_000)
        act &= (q24 >= 1e6) & np.isfinite(r24) & np.isfinite(o[i])
        act[jb] = False
        idx = np.flatnonzero(act)
        top = idx[np.argsort(r24[idx])][:kmax]
        end = min(i + hold - 1, nh - 1)
        for rank, j in enumerate(top):
            e0 = float(o[i, j])
            k_end = _last_finite(c[:, j], i, end)
            h = hi[i : k_end + 1, j] / e0 - 1
            lw = S["low"][i : k_end + 1, j] / e0 - 1
            kind, xo = "time", k_end
            ret = float(c[k_end, j] / e0 - 1)
            if tp is not None and np.nanmax(h) >= tp:
                kt = int(np.flatnonzero(h >= tp)[0])
                ret, kind, xo = tp, "tp", i + kt
                h, lw = h[: kt + 1], lw[: kt + 1]
            rows.append({
                "symbol": str(syms[j]), "t_in": T0 + int(i) * 3600, "t_out": T0 + int(xo + 1) * 3600,
                "rank": rank, "r24": float(r24[j]), "btc24": float(btc24[i]),
                "net": ret - SPOT_COST, "mfe": float(np.nanmax(h)), "mae": float(max(-np.nanmin(lw), 0.0)),
                "exit_kind": kind, "qv": float(q24[j]),
            })  # fmt: skip
    d = pl.DataFrame(rows)
    return d.with_columns(year=pl.from_epoch("t_in").dt.year(), day=pl.from_epoch("t_in").dt.date())


# ---------------------------------------------------------------- reporting --------------------
def diag_frame(t: pl.DataFrame, tp: float | None, stop: float | None) -> pl.DataFrame:
    return t.select(
        signal=pl.lit(1.0), threshold=pl.lit(1.0), net="net", mfe="mfe", mae="mae",
        tp=pl.lit(tp, dtype=pl.Float64), stop=pl.lit(stop, dtype=pl.Float64),
        exit_kind="exit_kind", year="year",
    )  # fmt: skip


def bookable(t: pl.DataFrame, sleeve: str, prio: float) -> pl.DataFrame:
    return t.select(
        "symbol", adv="mae", t_in="t_in", t_out="t_out", net="net", qv="qv",
        sleeve=pl.lit(sleeve), prio=pl.lit(prio) - pl.col("rank") * 0.001,
    )  # fmt: skip


def year_cells(t: pl.DataFrame) -> str:
    g = dict(t.group_by("year").agg(e=pl.col("net").clip(-1, 1).mean()).iter_rows())
    return " ".join(f"{g[y] * 100:+.1f}" if y in g else "-" for y in YEARS)


def main() -> None:
    S = sleeves()
    lmc = [as_book(S.filter(pl.col("sleeve") == k)) for k in "LMC"]
    fsl = as_book(S.filter(pl.col("sleeve") == "F"))
    base_lmc = run_book(lmc, 10)
    base_lmcf = run_book([*lmc, fsl], 10)
    md = ["# X8 B and C: daily baskets (pump shorts, flush-loser longs)", "",
          "In-sample exploration (`x8_baskets.py`, helpers `x8_lib.py`); every configuration tried is listed.", "", VERDICT]  # fmt: skip

    # ---------------- B
    md += ["## B. Basket short of the day's top-K 24h gainers (perp, 1x isolated)", "",
           "Trade = one basket name. `o` = net on the stake, clipped. Hedged = with a BTC perp long of equal notional,",
           "capital split 1x/1x (o = (short + long) / 2). Standalone book: 10 slots (one position per coin), each",
           "year from $10k. `+LMC` = the basket (2022-26 only, as LMC) added to the LMC book at the lowest priority, 10 slots",
           f"(LMC alone: {base_lmc['bp']:+.0f} bp/trade, worst year {base_lmc['min'] * 100:+.0f}%, {base_lmc['years']}).", "",  # type: ignore[operator]
           "| hour | gate | hold d | K | hedge | trades | E[o] % | median % | win | liq % | E[o] by year 2021..2026 (%) | book bp | book worst yr | book geo | book years | +LMC bp | +LMC worst yr |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    best: dict[str, tuple[float, str, pl.DataFrame]] = {}
    add: dict[str, tuple[float, str, pl.DataFrame]] = {}  # best +LMC worst year per family

    def keep_add(fam: str, lmin: float, name: str, t: pl.DataFrame) -> None:
        if fam not in add or lmin > add[fam][0]:
            add[fam] = (lmin, name, t)

    grid = []
    for hour, gate, hold in itertools.product((0, 12), (False, True), (1, 3, 7)):
        tr = pump_trades(hour, hold, gate)
        for k, hedge in itertools.product((3, 5, 10), (False, True)):
            t = tr.filter(pl.col("rank") < k)
            if hedge:
                t = t.with_columns(net=((pl.col("net") + pl.col("btc")) / 2).clip(-1.0, None))
            o = t["net"].clip(-1, 1)
            sb = book_stats(book(bookable(t, "B", 0.5), 10))
            lb = run_book([*lmc, bookable(t.filter(pl.col("year") >= 2022), "B", 0.5)], 10)
            name = f"H{hour:02d} {'fund>=0' if gate else 'no gate'} {hold}d K{k} {'hedged' if hedge else 'naked'}"
            row = (f"| {hour:02d} | {'fund>=0' if gate else '-'} | {hold} | {k} | {'BTC' if hedge else '-'} | {t.height} | "
                   f"{o.mean() * 100:+.2f} | {o.median() * 100:+.2f} | {(o > 0).mean():.2f} | {(t['exit_kind'] == 'liq').mean() * 100:.1f} | "
                   f"{year_cells(t)} | {sb['bp']:+.0f} | {sb['min'] * 100:+.0f}% | {sb['geo'] * 100:+.0f}% | {sb['years']} | "  # type: ignore[operator]
                   f"{lb['bp']:+.0f} | {lb['min'] * 100:+.0f}% |")  # type: ignore[operator]  # fmt: skip
            md.append(row)
            print(row, flush=True)
            keep_add("B", float(lb["min"]), name, t)  # type: ignore[arg-type]
            grid.append({"family": "B", "config": name, "trades": t.height, "E_o": float(o.mean()),
                         "book_bp": sb["bp"], "book_min": sb["min"], "lmc_bp": lb["bp"], "lmc_min": lb["min"]})  # fmt: skip
            if name.endswith("naked") and (("B" not in best) or o.mean() > best["B"][0]):
                best["B"] = (float(o.mean()), name, t)
            if hedge and (("Bh" not in best) or o.mean() > best["Bh"][0]):
                best["Bh"] = (float(o.mean()), name, t)

    # Single-name F vs basket: per-name outcome by gain rank (top-3, no gate vs gate, 3d, H00)
    md += ["", "### B by rank within the basket (H00, 3 days, naked)", "",
           "| gate | rank | n | E[o] % | liq % | mean 24h gain % |", "|---|---|---|---|---|---|"]  # fmt: skip
    for gate in (False, True):
        tr = pump_trades(0, 3, gate)
        for rk in (0, 1, 2, 4, 9):
            t = tr.filter(pl.col("rank") == rk)
            md.append(f"| {'fund>=0' if gate else '-'} | {rk + 1} | {t.height} | {t['net'].clip(-1, 1).mean() * 100:+.2f} | "
                      f"{(t['exit_kind'] == 'liq').mean() * 100:.1f} | {t['r24'].mean() * 100:.0f} |")  # fmt: skip

    md += ["", "### B where the per-name return goes (naked, H00)", "",
           "Mean per name, %: gross = price move of the short (liquidated names count -100%), funding received,",
           "cost 0.30; liq contribution = liquidated share x -100%; `ex-liq` = mean net of the names not liquidated.", "",
           "| gate | hold d | K | n | gross % | funding % | cost % | net % | liq share % | liq contribution % | net ex-liq % |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for gate, hold, k in itertools.product((False, True), (1, 3, 7), (3, 10)):
        t = pump_trades(0, hold, gate).filter(pl.col("rank") < k)
        lq = t["exit_kind"] == "liq"
        md.append(f"| {'fund>=0' if gate else '-'} | {hold} | {k} | {t.height} | {t['gross'].mean() * 100:+.2f} | "
                  f"{t['fund'].mean() * 100:+.2f} | -0.30 | {t['net'].mean() * 100:+.2f} | {lq.mean() * 100:.1f} | "
                  f"{-lq.mean() * 100:+.2f} | {t.filter(~lq)['net'].mean() * 100:+.2f} |")  # fmt: skip
    md += ["", "### B restricted to 24h gains >= 40% (closest to F, but at a fixed hour instead of the 4h-spike event)", "",
           "| hour | gate | hold d | K | trades | days | E[o] % | liq % | E[o] by year 2021..2026 (%) | book bp | book worst yr | +LMC bp | +LMC worst yr |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for hour, gate, hold in itertools.product((0, 12), (False, True), (1, 3)):
        t = pump_trades(hour, hold, gate, kmax=3, min_gain=0.4)
        o = t["net"].clip(-1, 1)
        sb = book_stats(book(bookable(t, "B", 0.5), 10))
        lb = run_book([*lmc, bookable(t.filter(pl.col("year") >= 2022), "B", 0.5)], 10)
        md.append(f"| {hour:02d} | {'fund>=0' if gate else '-'} | {hold} | 3 | {t.height} | {t['day'].n_unique()} | {o.mean() * 100:+.2f} | "
                  f"{(t['exit_kind'] == 'liq').mean() * 100:.1f} | {year_cells(t)} | {sb['bp']:+.0f} | {sb['min'] * 100:+.0f}% | "  # type: ignore[operator]
                  f"{lb['bp']:+.0f} | {lb['min'] * 100:+.0f}% |")  # type: ignore[operator]  # fmt: skip
        keep_add(
            "B40",
            float(lb["min"]),
            f"H{hour:02d} {'fund>=0' if gate else 'no gate'} {hold}d K3 r24>=40%",
            t,
        )  # type: ignore[arg-type]
        grid.append({"family": "B40", "config": f"H{hour:02d} {'fund>=0' if gate else 'no gate'} {hold}d K3 r24>=40%",
                     "trades": t.height, "E_o": float(o.mean()), "book_bp": sb["bp"], "book_min": sb["min"],
                     "lmc_bp": lb["bp"], "lmc_min": lb["min"]})  # fmt: skip

    # ---------------- C
    md += ["", "## C. Basket long of the biggest 24h losers on BTC flush days (spot)", "",
           "BTC 24h <= -8% at the signal hour. `event` = first such hour, then a 24h lock; `00UTC` = checked daily at 00 UTC.",
           "Standalone book: 10 slots. `+LMC` = added (2022-26) to LMC at the lowest priority, 10 slots.", "",
           "| trigger | hold d | exit | K | signals | trades | E[o] % | median % | win | TP share | E[o] by year 2021..2026 (%) | book bp | book worst yr | book years | +LMC bp | +LMC worst yr |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for mode, hold, tp in itertools.product(("event", "00UTC"), (1, 2, 3), (None, 0.2)):
        tr = flush_trades(mode, hold, tp)
        for k in (3, 5, 10):
            t = tr.filter(pl.col("rank") < k)
            o = t["net"].clip(-1, 1)
            sb = book_stats(book(bookable(t, "D", 0.5), 10))
            lb = run_book([*lmc, bookable(t.filter(pl.col("year") >= 2022), "D", 0.5)], 10)
            name = f"{mode} {hold}d {'tp20' if tp else 'time'} K{k}"
            row = (f"| {mode} | {hold} | {'TP 20%' if tp else 'time'} | {k} | {t['t_in'].n_unique()} | {t.height} | "
                   f"{o.mean() * 100:+.2f} | {o.median() * 100:+.2f} | {(o > 0).mean():.2f} | {(t['exit_kind'] == 'tp').mean():.2f} | "
                   f"{year_cells(t)} | {sb['bp']:+.0f} | {sb['min'] * 100:+.0f}% | {sb['years']} | {lb['bp']:+.0f} | {lb['min'] * 100:+.0f}% |")  # type: ignore[operator]  # fmt: skip
            md.append(row)
            print(row, flush=True)
            keep_add("C", float(lb["min"]), name, t)  # type: ignore[arg-type]
            grid.append({"family": "C", "config": name, "trades": t.height, "E_o": float(o.mean()),
                         "book_bp": sb["bp"], "book_min": sb["min"], "lmc_bp": lb["bp"], "lmc_min": lb["min"]})  # fmt: skip
            if ("C" not in best) or o.mean() > best["C"][0]:
                best["C"] = (float(o.mean()), name, t)

    # Flush days: the basket vs L on the same days
    tr = flush_trades("event", 3, 0.2)
    Ld = S.filter(pl.col("sleeve") == "L").with_columns(day=pl.from_epoch("t_in").dt.date())
    md += ["", "### C signals vs L (event trigger, 3 days, TP 20%, K=5)", "",
           "| signal time (UTC) | BTC 24h % | basket E[o] % (K=5) | L trades within 24h after | L E[o] % |", "|---|---|---|---|---|"]  # fmt: skip
    t5 = tr.filter(pl.col("rank") < 5)
    for (ti,), g in t5.group_by(["t_in"], maintain_order=True):
        lt = Ld.filter((pl.col("t_in") >= ti - 6 * 3600) & (pl.col("t_in") < ti + 86400))
        md.append(f"| {pl.from_epoch(pl.Series([ti])).dt.strftime('%Y-%m-%d %H:%M')[0]} | {g['btc24'][0] * 100:+.1f} | "
                  f"{g['net'].mean() * 100:+.1f} | {lt.height if ti >= 1640995200 else '- (no L before 2022)'} | {lt['net'].mean() * 100 if lt.height else float('nan'):+.1f} |")  # fmt: skip

    md += ["", "### C on L days vs other flush days (2022-26; L day = an L entry within 24h after the signal)", "",
           "| config | L-day signals | basket E[o] % on L days | L E[o] % on those days | other signals | basket E[o] % on other days | 2020-21 signals | basket E[o] % 2020-21 |",
           "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for mode, hold, tp, k in (
        ("event", 3, 0.2, 5),
        ("event", 1, 0.2, 5),
        ("event", 3, None, 10),
        ("00UTC", 3, 0.2, 5),
    ):
        t = flush_trades(mode, hold, tp).filter(pl.col("rank") < k)
        sig = t.group_by("t_in").agg(e=pl.col("net").mean(), y=pl.col("year").first())
        lday, lnet = [], []
        for ti in sig["t_in"].to_list():
            lt = Ld.filter((pl.col("t_in") >= ti - 6 * 3600) & (pl.col("t_in") < ti + 86400))
            lday.append(lt.height > 0)
            lnet.append(float(lt["net"].mean()) if lt.height else float("nan"))
        sig = sig.with_columns(lday=pl.Series(lday), lnet=pl.Series(lnet))
        a = sig.filter((pl.col("y") >= 2022) & pl.col("lday"))
        b = sig.filter((pl.col("y") >= 2022) & ~pl.col("lday"))
        c0 = sig.filter(pl.col("y") <= 2021)
        md.append(f"| {mode} {hold}d {'tp20' if tp else 'time'} K{k} | {a.height} | {a['e'].mean() * 100:+.1f} | {a['lnet'].mean() * 100:+.1f} | "
                  f"{b.height} | {b['e'].mean() * 100:+.1f} | {c0.height} | {c0['e'].mean() * 100:+.1f} |")  # fmt: skip

    md += ["", "## LMC and LMCF with the best addition of each family (chosen by the +LMC worst year, in-sample; 2022-26 as LMC)", "",
           "| book | addition | trades | bp/trade | worst year | geo yearly | max DD | years |", "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for nm, r in (("LMC", base_lmc), ("LMCF", base_lmcf)):
        md.append(
            f"| {nm} | none | {r['trades']} | {r['bp']:+.0f} | {r['min'] * 100:+.0f}% | {r['geo'] * 100:+.0f}% | {r['dd'] * 100:+.0f}% | {r['years']} |"
        )  # type: ignore[operator]
    for fam, (_, name, t) in add.items():
        bt = bookable(t.filter(pl.col("year") >= 2022), "D" if fam == "C" else "B", 0.5)
        for nm, parts in (("LMC", lmc), ("LMCF", [*lmc, fsl])):
            r = run_book([*parts, bt], 10)
            md.append(
                f"| {nm} | {fam}: {name} | {r['trades']} | {r['bp']:+.0f} | {r['min'] * 100:+.0f}% | {r['geo'] * 100:+.0f}% | {r['dd'] * 100:+.0f}% | {r['years']} |"
            )  # type: ignore[operator]

    # ---------------- scorecards
    cards: dict[str, dict[str, float]] = {}
    for key, (_, name, t) in best.items():
        tp = 0.2 if "tp20" in name else None
        stop = None if key == "C" else 1.0
        cards[f"{key}: {name}"] = card(diag_frame(t, tp, stop))
    cards["F (single-name pump shorts, reference)"] = card(S.filter(pl.col("sleeve") == "F"))
    cards["L (capitulation longs, reference)"] = card(S.filter(pl.col("sleeve") == "L"))
    md += [
        "",
        "## Diag scorecards (best E[o] configuration of each family; 2021-26 for baskets, 2022-26 for sleeves)",
        "",
        table(cards),
        "",
    ]
    md += [f"Reference books: LMC {base_lmc['bp']:+.0f} bp, worst {base_lmc['min'] * 100:+.0f}%, geo {base_lmc['geo'] * 100:+.0f}% ({base_lmc['years']}); "  # type: ignore[operator]
           f"LMCF {base_lmcf['bp']:+.0f} bp, worst {base_lmcf['min'] * 100:+.0f}%, geo {base_lmcf['geo'] * 100:+.0f}% ({base_lmcf['years']}).", ""]  # type: ignore[operator]  # fmt: skip
    pl.DataFrame(grid).write_csv(OUT / "x8_baskets_grid.csv")
    (OUT / "baskets.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
