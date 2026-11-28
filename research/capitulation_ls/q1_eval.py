"""Q1 (illiquid spikes): the pre-registered long sleeve L on the spikes that seq2 drops for
liquidity, per band (b1k: 1k <= qv_base < 5k USDT per 5m, b500: 500 <= qv_base < 1k), 2022-26.

- L: spot long when the fold's day5 pct >= 0.995, entry next 5m open, TP +20%, 72h, cost 0.4%,
  one position per coin within the band (h1_eval.trades on the q1 caches: sim's meta / fwd2 /
  fwd72 loaders swapped, as h1_eval does for the holdout); perp-only coins (if q1_build.py perp
  ran): p1_eval.sim on perp paths with p1's activity gate, cost 0.3% + funding.
- diag.py scorecard, decile calibration (pool pct >= 0.95) and p1's fixed bins per band;
- flush days a band adds against spot L and perp-only L (UTC days with a pct >= 0.995 candidate);
- books: e25 `book` (10 slots, $10k a year, stake <= 0.5% of qv_base x 288) for LMC, LMC+
  (LMC + p1's L-perp-only trades) and LMC+ with each band, with the band trades at 0.4%, 1.0% and
  2.0% round trip. An instrumented copy of `book` (checked equal to e25's) logs each band trade's
  stake, whether the liquidity cap set it, and the entry candle's quote volume.

Writes results/explore/illiquid_tables.md and OWN/q1/trades_*.parquet.
Run: uv run --group research python q1_eval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import h1_eval as H
import numpy as np
import polars as pl
import sim
from common_ls import OWN, meta, scores
from diag import annotate as dannotate
from diag import calibration, scorecard, table, threshold_reach

Q1 = OWN / "q1"
P1 = OWN / "p1"
OUT = Path(__file__).resolve().parent / "results" / "explore"
YEARS = (2022, 2023, 2024, 2025, 2026)
COSTS = (0.004, 0.010, 0.020)
COLS = ["symbol", "adv", "t_in", "t_out", "net", "qv", "sleeve", "prio", "w", "src"]
EDGES = [0.95, 0.976, 0.984, 0.9955, 1.0001]


def md(df: pl.DataFrame, digits: int = 3) -> str:
    cols = df.columns
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.iter_rows():
        out.append(
            "| "
            + " | ".join(f"{v:.{digits}f}" if isinstance(v, float) else str(v) for v in r)
            + " |"
        )
    return "\n".join(out)


def book_log(
    tr: pl.DataFrame, slots: int = 10, start: float = 10_000.0
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """e25_combined.book (gross 1, no caps / weights) that also logs every taken trade."""
    tr = tr.with_columns(year=pl.from_epoch("t_in").dt.year()).sort(
        ["t_in", "prio"], descending=[False, True]
    )
    res, log = [], []
    for (y,), g in tr.group_by(["year"], maintain_order=True):
        eq, cash, pos, nets, peak, dd, held = start, start, [], [], start, 0.0, set()
        by: dict[str, list[float]] = {"L": [], "P": [], "N": [], "C": []}
        for r in g.iter_rows(named=True):
            keep = []
            for p in pos:
                if p[0] <= r["t_in"]:
                    cash += p[1] * (1 + p[2])
                    eq += p[1] * p[2]
                    held.discard(p[3])
                    peak = max(peak, eq)
                    dd = min(dd, eq / peak - 1)
                else:
                    keep.append(p)
            pos = keep
            if len(pos) >= slots or r["symbol"] in held:
                continue
            rw = r.get("w")
            w = 1.0 if rw is None else rw
            slot = w * eq / slots
            cap = r["qv"] * 0.005
            stake = min(slot, cash, cap)
            if stake < 1:
                continue
            cash -= stake
            pos.append((r["t_out"], stake, r["net"], r["symbol"], r["adv"], r["sleeve"]))
            held.add(r["symbol"])
            log.append({"year": y, "src": r["src"], "symbol": r["symbol"], "t_in": r["t_in"],
                        "stake": stake, "slot": min(slot, cash + stake), "cap": cap,
                        "capped": cap < min(slot, cash + stake), "net": r["net"]})  # fmt: skip
            nets.append(r["net"])
            by.setdefault(r["sleeve"], []).append(r["net"])
        for p in sorted(pos):
            eq += p[1] * p[2]
            peak = max(peak, eq)
            dd = min(dd, eq / peak - 1)
        res.append({"year": y, "trades": len(nets), "ret": eq / start - 1, "dd": dd,
                    **{f"n{k}": len(v) for k, v in by.items()}})  # fmt: skip
    return pl.DataFrame(res).sort("year"), pl.DataFrame(log)


def base_parts() -> dict[str, pl.DataFrame]:
    from e25_combined import sleeve_crash_short, sleeve_listing_weak, sleeve_long

    parts = {"L": sleeve_long(), "M": sleeve_listing_weak(), "C": sleeve_crash_short()}
    out = {k: v.with_columns(w=pl.lit(1.0), src=pl.lit(k)).select(COLS) for k, v in parts.items()}
    po = pl.read_parquet(P1 / "trades_L.parquet")
    out["Lpo"] = add(po, "Lpo")
    return out


def add(t: pl.DataFrame, src: str) -> pl.DataFrame:
    return t.select(
        "symbol", adv=pl.col("adv").clip(0, 1), t_in="t_in", t_out="t_out", net="net",
        qv=pl.col("qv_base") * 288, sleeve=pl.lit("L"), prio=3.0 + pl.col("pct"), w=pl.lit(1.0),
        src=pl.lit(src),
    ).select(COLS)  # fmt: skip


def by_year(t: pl.DataFrame) -> pl.DataFrame:
    return (
        t.group_by("year")
        .agg(
            n=pl.len(),
            coins=pl.col("symbol").n_unique(),
            days=pl.col("day").n_unique(),
            mean_net=pl.col("net").mean(),
            mean_1pct=pl.col("net").mean() - 0.006,
            mean_2pct=pl.col("net").mean() - 0.016,
            median_net=pl.col("net").median(),
            win=(pl.col("net") > 0).mean(),
            tp_rate=(pl.col("exit_kind") == "tp").mean(),
            qv_entry_med=pl.col("qv_entry").median(),
            tiny_entry=(pl.col("qv_entry").fill_null(0) < 500).mean(),
        )
        .sort("year")
    )


def bins(t: pl.DataFrame) -> pl.DataFrame:
    import itertools

    lab = [f"{a:.4f}-{b:.4f}" for a, b in itertools.pairwise(EDGES)]
    a = dannotate(t).with_columns(
        bin=pl.col("signal").cut(EDGES[1:-1], labels=lab, left_closed=True)
    )
    return (
        a.group_by("bin")
        .agg(
            n=pl.len(),
            E_o=pl.col("o").mean(),
            win=(pl.col("o") > 0).mean(),
            p_tp=(pl.col("mode") == "tp_win").mean(),
        )
        .sort("bin")
    )


def spot_band_trades(cm: pl.DataFrame, band: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Entries (pct >= 0.995) and non-entries (0.95 <= pct < 0.995) of one band, as h1_eval."""
    pct = cm["pct"].to_numpy().astype(np.float64)
    if band != "both":
        pct = np.where((cm["band"] == band).to_numpy(), pct, np.nan)
    extra = cm.select(pl.int_range(pl.len()).alias("row"), "band", "qv_entry", "n_entry",
                      "qv_spike", "pct")  # fmt: skip
    ent = H.trades(cm, pct, YEARS, 0.995).join(extra, on="row", how="left")
    ent = ent.with_columns(t_in=pl.col("t") + 300)
    non = H.trades(cm, pct, YEARS, 0.95, 0.995).join(extra, on="row", how="left")
    return ent, non


def perp_band_trades(band: str) -> tuple[pl.DataFrame, pl.DataFrame] | None:
    p = Q1 / "perp_spikes.parquet"
    if not p.exists():
        return None
    from p1_eval import RL
    from p1_eval import sim as psim

    d = pl.read_parquet(p).filter(pl.col("perp") & pl.col("pct").is_not_nan())
    if band != "both":
        d = d.filter(pl.col("band") == band)
    d = d.with_columns(prio=pl.col("pct").cast(pl.Float64), pct=pl.col("pct").cast(pl.Float64))
    # cost 0.4% as the spot bands (p1 used 0.3% for liquid perps)
    rule = sim.Rule(hold_min=RL.hold_min, tp=RL.tp, slots=RL.slots, cost=0.004, liq_frac=None)
    ent = psim(d.filter(pl.col("pct") >= 0.995), 1, rule)
    non = psim(d.filter((pl.col("pct") >= 0.95) & (pl.col("pct") < 0.995)), 1, rule)
    return ent, non


def tt_row(name: str, base: pl.DataFrame, tt: pl.DataFrame) -> dict[str, object]:
    """Mean net with the usual +20% take profit vs one that needs a +22% high (filled at +20%)."""
    net_tt = tt.with_columns(net=pl.when(pl.col("kind") == 2).then(0.2 - 0.004).otherwise("net"))
    return {"trades": name, "n": base.height, "mean_net": float(base["net"].mean()),
            "tp_rate": float((base["kind"] == 2).mean()), "mean_net_tt": float(net_tt["net"].mean()),  # noqa: E501
            "tp_rate_tt": float((tt["kind"] == 2).mean())}  # fmt: skip


def frame(t: pl.DataFrame, sig: str) -> pl.DataFrame:
    return t.select(
        signal=pl.col(sig).cast(pl.Float64), threshold=pl.lit(0.995), net="net", mfe="mfe",
        mae="adv", tp=pl.lit(0.2, dtype=pl.Float64), stop=pl.lit(None, dtype=pl.Float64),
        exit_kind="exit_kind", year="year",
    )  # fmt: skip


def main() -> None:
    pl.Config.set_tbl_rows(100)
    lines: list[str] = []

    def emit(*xs: str) -> None:
        lines.extend(xs)
        print(*xs, sep="\n", flush=True)

    # -------------------------------------------------------------- reference parts (shared)
    base = base_parts()
    m = meta()
    pct_s = scores("day5")
    sel = (m["year"].to_numpy() >= 2022) & np.isfinite(pct_s) & (pct_s >= 0.995)
    spot_days = set(m.filter(pl.Series(sel))["day"].to_list())
    p1s = pl.read_parquet(P1 / "spikes.parquet")
    p1c = pl.read_parquet(P1 / "scores.parquet").sort("row")
    po = p1s.with_columns(pct=p1c["pct"]).filter((pl.col("pct") >= 0.995) & pl.col("perp"))
    po_days = set(po["open_time"].dt.date().to_list())
    sm = H.use_shared()
    tt_rows = [tt_row("spot liquid L (seq2)", H.trades(sm, pct_s, YEARS, 0.995),
                      H.trades(sm, pct_s, YEARS, 0.995, tp=0.22))]  # fmt: skip

    # -------------------------------------------------------------- coverage
    sp = pl.read_parquet(Q1 / "spot_spikes.parquet")
    cov = (
        sp.group_by("year", "band")
        .agg(
            spikes=pl.len(), coins=pl.col("symbol").n_unique(),
            nan=pl.col("pct").is_nan().sum(), q95=(pl.col("pct") >= 0.95).sum(),
            q995=(pl.col("pct") >= 0.995).sum(),
            q995_days=pl.col("day").filter(pl.col("pct") >= 0.995).n_unique(),
            share_q995=(pl.col("pct") >= 0.995).mean(),
        )
        .sort("band", "year")
    )  # fmt: skip
    emit("## Coverage (spot, sub-5k spikes, 2022-01..2026-10)", "",
         f"{sp['symbol'].n_unique()} coins, {sp.height:,} spikes below qv_base 5k "
         f"(b1k {int((sp['band'] == 'b1k').sum()):,}, b500 {int((sp['band'] == 'b500').sum()):,}).",
         "", md(cov), "")  # fmt: skip

    # -------------------------------------------------------------- spot bands
    cm = pl.read_parquet(Q1 / "cand.parquet")
    f2 = np.load(Q1 / "fwd2.npy", mmap_mode="r")
    f72 = np.load(Q1 / "fwd72.npy", mmap_mode="r")
    sim.meta, sim.fwd2, sim.fwd72 = (lambda: cm), (lambda: f2), (lambda: f72)  # type: ignore[assignment]
    H.sim.meta = sim.meta  # same module object; explicit for readers
    cards: dict[str, dict[str, float]] = {}
    band_tr: dict[str, pl.DataFrame] = {}
    flush_rows = []
    for band in ("b1k", "b500", "both"):
        ent, non = spot_band_trades(cm, band)
        ent = ent.with_columns(
            new_day=pl.Series([x not in (spot_days | po_days) for x in ent["day"].to_list()])
        )
        band_tr[band] = ent
        ent.write_parquet(Q1 / f"trades_spot_{band}.parquet")
        fe, fn = frame(ent, "signal"), frame(non, "signal")
        cards[f"L-spot-{band}"] = scorecard(dannotate(fe))
        tr = threshold_reach(fe, fn)
        emit(f"## L on spot {band} spikes (cost 0.4%; mean_1pct / mean_2pct = at 1.0% / 2.0%)", "",
             md(by_year(ent)), "",
             f"All years: n {ent.height}, mean net {ent['net'].mean():+.4f} (1%: "
             f"{ent['net'].mean() - 0.006:+.4f}, 2%: {ent['net'].mean() - 0.016:+.4f}), "
             f"TP rate {(ent['exit_kind'] == 'tp').mean():.2f}, entry candle quote volume < 500 "
             f"USDT: {int((ent['qv_entry'].fill_null(0) < 500).sum())} trades, zero: "
             f"{int((ent['qv_entry'].fill_null(0) == 0).sum())}.", "",
             H.clustered(ent, f"{band} day-clustered") if ent.height else "", "",
             "Entry days (n, mean net, TP rate, median quote volume of the entry candle):", "",
             md(ent.group_by("day").agg(
                 n=pl.len(), b500=(pl.col("band") == "b500").sum(), mean_net=pl.col("net").mean(),
                 tp_rate=(pl.col("exit_kind") == "tp").mean(),
                 qv_entry_med=pl.col("qv_entry").median(), qv_base_med=pl.col("qv").median() / 288,
             ).sort("day"), 3), "",
             f"Entry candle quote volume over all {band} trades: 5th pct "
             f"{ent['qv_entry'].quantile(0.05):,.0f} USDT, min {ent['qv_entry'].min():,.0f}; trade "
             f"count 5th pct {ent['n_entry'].quantile(0.05):,.0f}, min {ent['n_entry'].min()}.", "",
             f"Threshold 0.995: {tr['share_reaching']:.1%} of the pct >= 0.95 pool reaches it; "
             f"Spearman(signal, outcome) {tr['spearman_signal_outcome']:+.3f}.", "",
             "Calibration by signal decile (pool pct >= 0.95):", "", md(calibration(fe, fn)), "",
             "Fixed bins (spot liquid L: ~0 / +0.026 / +0.059 / +0.132; perp-only: +0.009 / +0.002 / +0.043 / +0.137):",  # noqa: E501
             "", md(bins(pl.concat([fe, fn]))), "")  # fmt: skip
        cand_days = set(
            cm.filter(
                (pl.col("pct") >= 0.995) & ((pl.col("band") == band) | pl.lit(band == "both"))
            )["day"].to_list()
        )
        for y in YEARS:
            cy = {d for d in cand_days if d.year == y}
            flush_rows.append({
                "band": band, "year": y, "spot_L_days": len({d for d in spot_days if d.year == y}),
                "perp_only_days": len({d for d in po_days if d.year == y}),
                "band_days": len(cy), "new_vs_spot": len(cy - spot_days),
                "new_vs_both": len(cy - spot_days - po_days),
                "new_day_trades": int(ent.filter(pl.col("new_day") & (pl.col("year") == y)).height),
            })  # fmt: skip
        nd = ent.group_by("new_day").agg(
            n=pl.len(), days=pl.col("day").n_unique(), mean_net=pl.col("net").mean(),
            tp_rate=(pl.col("exit_kind") == "tp").mean(),
        ).sort("new_day")  # fmt: skip
        emit("Trades on new flush days (no spot-L, no perp-only-L candidate that day) vs shared:",
             "", md(nd), "")  # fmt: skip
    emit("## New flush days", "", md(pl.DataFrame(flush_rows)), "")
    lq = base["L"].select("symbol", t_l="t_in")
    ov_rows = []
    for band in ("b1k", "b500", "both"):
        e = band_tr[band].with_row_index("k")
        j = e.select("k", "symbol", "t_in").join(lq, on="symbol", how="inner")
        hit = set(j.filter((pl.col("t_l") - pl.col("t_in")).abs() <= 72 * 3600)["k"].to_list())
        e = e.with_columns(overlap=pl.Series([k in hit for k in e["k"].to_list()]))
        ov_rows.append({"band": band, "n": e.height, "same_coin_liquid_L_72h": len(hit),
                        "mean_net_overlap": e.filter("overlap")["net"].mean(),
                        "mean_net_rest": e.filter(~pl.col("overlap"))["net"].mean()})  # fmt: skip
        pct = cm["pct"].to_numpy().astype(np.float64)
        if band != "both":
            pct = np.where((cm["band"] == band).to_numpy(), pct, np.nan)
        tt_rows.append(
            tt_row(f"spot {band}", band_tr[band], H.trades(cm, pct, YEARS, 0.995, tp=0.22))
        )
    emit("Same coin: band trades whose coin also has a liquid spot-L trade within 72h (the book's"
         " one-position-per-coin rule keeps only one of them):", "", md(pl.DataFrame(ov_rows)), "",
         "Trade-through take profit (the high must reach +22% for the +20% fill; a wick that only"
         " touches +20% does not count), cost 0.4%:", "", md(pl.DataFrame(tt_rows)), "")  # fmt: skip  # noqa: E501

    # -------------------------------------------------------------- perp-only bands
    perp_tr: dict[str, pl.DataFrame] = {}
    for band in ("b1k", "b500"):
        got = perp_band_trades(band)
        if got is None:
            emit("Perp-only bands: not built (q1_build.py perp not run).", "")
            break
        ent, non = got
        if ent.height == 0:
            emit(f"## L on perp-only {band}: no entries", "")
            continue
        ent = ent.with_columns(
            new_day=pl.Series([x not in (spot_days | po_days) for x in ent["day"].to_list()])
        )
        perp_tr[band] = ent
        ent.write_parquet(Q1 / f"trades_perp_{band}.parquet")
        fe, fn = frame(ent, "pct"), frame(non, "pct")
        cards[f"L-perp-only-{band}"] = scorecard(dannotate(fe))
        tr = threshold_reach(fe, fn)
        emit(f"## L on perp-only {band} spikes (perp, gate on, cost 0.4% + funding)", "",
             md(by_year(ent)), "",
             f"All years: n {ent.height}, mean net {ent['net'].mean():+.4f}; Spearman "
             f"{tr['spearman_signal_outcome']:+.3f}; new-day trades "
             f"{int(ent['new_day'].sum())} (mean {ent.filter('new_day')['net'].mean() if ent['new_day'].any() else float('nan'):+.4f}).",  # noqa: E501
             "", "Fixed bins:", "", md(bins(pl.concat([fe, fn]))), "")  # fmt: skip
    emit("## Scorecards (diag.py)", "", table(cards), "")

    # -------------------------------------------------------------- books
    from e25_combined import book

    lmc = [base["L"], base["M"], base["C"]]
    lmcp = [*lmc, base["Lpo"]]
    chk, _ = book_log(pl.concat(lmcp, how="vertical_relaxed"))
    ref = book(pl.concat(lmcp, how="vertical_relaxed").drop("src"), 10)
    assert np.allclose(chk["ret"].to_numpy(), ref["ret"].to_numpy()), "book_log parity"
    emit("book_log parity with e25 `book` on LMC+: equal yearly returns.", "")

    def bt(t: pl.DataFrame, src: str, cost: float) -> pl.DataFrame:
        if t.height == 0:
            return pl.DataFrame(schema={c: pl.Float64 for c in COLS})
        base_cost = 0.004
        return add(t.with_columns(net=pl.col("net") - (cost - base_cost),
                                  qv_base=pl.col("qv") / 288 if "qv_base" not in t.columns
                                  else pl.col("qv_base")), src)  # fmt: skip

    rows, caprows = [], []
    variants: dict[str, list[pl.DataFrame] | None] = {"LMC": lmc, "LMC+": lmcp}
    for cost in COSTS:
        c = f"{cost * 100:.1f}%"
        for name, parts in {
            **variants,
            f"LMC+ + b1k @{c}": [*lmcp, bt(band_tr["b1k"], "b1k", cost)],
            f"LMC+ + b500 @{c}": [*lmcp, bt(band_tr["b500"], "b500", cost)],
            f"LMC+ + both @{c}": [*lmcp, bt(band_tr["both"], "both", cost)],
            f"LMC+ + both as filler (prio below liquid L) @{c}": [
                *lmcp, bt(band_tr["both"], "both", cost).with_columns(prio=pl.col("prio") - 0.5)],
            **(
                {f"LMC+ + spot b1k + perp-only b1k @{c}": [
                    *lmcp, bt(band_tr["b1k"], "b1k", cost), bt(perp_tr["b1k"], "pb1k", cost)]}
                if "b1k" in perp_tr else {}
            ),
        }.items():  # fmt: skip
            if parts is None:
                continue
            b, log = book_log(pl.concat([p for p in parts if p.height], how="vertical_relaxed"))
            yrs = {int(r["year"]): r for r in b.iter_rows(named=True)}
            row: dict[str, object] = {"book": name, "trades": int(b["trades"].sum())}
            for y in YEARS:
                r = yrs.get(y)
                row[str(y)] = (
                    f"{r['ret'] * 100:+.0f}% (dd {r['dd'] * 100:.0f}, L{r['nL']})" if r else "-"
                )
            row["min yr"] = f"{min(float(r['ret']) for r in yrs.values()) * 100:+.0f}%"
            rows.append(row)
            lb = log.filter(~pl.col("src").is_in(["L", "M", "C", "Lpo"])) if log.height else log
            if lb.height:
                caprows.append({
                    "book": name, "band_trades": lb.height, "capped": int(lb["capped"].sum()),
                    "avg_stake": float(lb["stake"].mean()), "avg_slot": float(lb["slot"].mean()),
                    "band_pnl": float((lb["stake"] * lb["net"]).sum()),
                    "band_mean_net": float(lb["net"].mean()),
                })  # fmt: skip
        variants = {}
    emit("## 10-slot books (e25 `book`, $10k each year; band trades at the given cost, the rest"
         " at their own costs)", "", md(pl.DataFrame(rows)), "",
         "Band trades taken by the book: how many had their stake set by the 0.5%-of-24h cap,"
         " average stake and slot (USDT), total band P&L (USDT, summed over years):", "",
         md(pl.DataFrame(caprows), 1), "")  # fmt: skip

    # entry-candle volume vs stake for the band trades taken (cost 0.4%, both bands)
    b, log = book_log(
        pl.concat([*lmcp, bt(band_tr["both"], "both", 0.004)], how="vertical_relaxed")
    )
    lb = log.filter(pl.col("src") == "both").join(
        band_tr["both"].select("symbol", t_in=pl.col("t") + 300, qv_entry="qv_entry",
                               band="band"), on=["symbol", "t_in"], how="left")  # fmt: skip
    if lb.height:
        emit("Band trades taken (both bands, 0.4%): stake vs the entry candle's quote volume:", "",
             md(lb.group_by("year").agg(
                 n=pl.len(), b500=(pl.col("band") == "b500").sum(), capped=pl.col("capped").sum(),
                 stake_med=pl.col("stake").median(), qv_entry_med=pl.col("qv_entry").median(),
                 stake_gt_entry_bar=(pl.col("stake") > pl.col("qv_entry").fill_null(0)).sum(),
                 mean_net=pl.col("net").mean(),
             ).sort("year"), 1), "")  # fmt: skip
    (OUT / "illiquid_tables.md").write_text("\n".join(lines) + "\n")
    print("written", file=sys.stderr)


if __name__ == "__main__":
    main()
