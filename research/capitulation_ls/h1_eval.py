"""H1 (holdout): the pre-registered long sleeve L on 2017-08 .. 2019-12 (never used by any study),
with the same simulator, book and diagnostics as the 2022-26 work:

- L: spot long when the day5 CNN pct >= 0.995; entry at the next 5m open; take profit +20%;
  time exit 72h; cost 0.4% round trip; one position per coin (sim.simulate, no slot limit, as
  e25_combined.sleeve_long);
- book: e25_combined.book, 10 equal slots, a fresh $10k each entry year, stake <= 0.5% of the
  coin's 24h quote volume;
- diag.py scorecard and calibration (pool pct >= 0.95, below-threshold candidates simulated as if
  entered, as d01_scorecards.py);
- neighbours (pct 0.99 / 0.998, take profit 15% / 25%) and the 2022 fold of day5 as a second scorer;
- the same numbers for 2022-26 from the shared caches as the reference.

The simulator reads the holdout caches by swapping sim's `meta` / `fwd2` / `fwd72` loaders after the
reference is computed. Writes holdout/eval_tables.md, holdout/l_trades.parquet and prints both.
Run: uv run --group research python h1_eval.py
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import polars as pl
import sim
from common_ls import scores
from diag import annotate, calibration, scorecard, table, threshold_reach
from e25_combined import book
from sim import Rule, simulate

HOLDOUT = Path("/Users/mike/code/crypto-moonshot/data/cache/capitulation_ls/holdout")
KIND = {0: "time", 1: "stop", 2: "tp", 3: "liq", 4: "stop"}
HY = (2017, 2018, 2019)
RY = (2022, 2023, 2024, 2025, 2026)
ORIG = (sim.meta, sim.fwd2, sim.fwd72)
out: list[str] = []


def emit(s: str = "") -> None:
    out.append(s)
    print(s)


def use_holdout() -> pl.DataFrame:
    hm = pl.read_parquet(HOLDOUT / "seq2_meta.parquet").with_columns(
        year=pl.col("open_time").dt.year(), day=pl.col("open_time").dt.date()
    )
    f2 = np.load(HOLDOUT / "fwd2.npy", mmap_mode="r")
    f72 = np.load(HOLDOUT / "fwd72.npy", mmap_mode="r")
    sim.meta, sim.fwd2, sim.fwd72 = (lambda: hm), (lambda: f2), (lambda: f72)  # type: ignore[assignment]
    return hm


def use_shared() -> pl.DataFrame:
    sim.meta, sim.fwd2, sim.fwd72 = ORIG  # type: ignore[assignment]
    return sim.meta()


def trades(
    m: pl.DataFrame,
    pct: np.ndarray,
    years: tuple[int, ...],
    lo: float,
    hi: float = 2.0,
    tp: float = 0.2,
    hold: int = 4320,
) -> pl.DataFrame:
    """Every candidate with lo <= pct < hi in `years`, one position per coin, no slot limit."""
    yr = m["year"].to_numpy()
    ok = np.isin(yr, years) & np.isfinite(pct) & (pct >= lo) & (pct < hi)
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        return pl.DataFrame()
    c = pl.DataFrame({"row": idx, "side": 1, "prio": pct[idx]})
    rule = Rule(hold_min=hold, tp=tp, slots=100_000, cost=0.004, liq_frac=None)
    t = simulate(c, rule, start_equity=1e12).trades
    qv = m["qv_base"].to_numpy()
    return t.with_columns(
        year=pl.col("open_time").dt.year(),
        day=pl.col("open_time").dt.date(),
        exit_kind=pl.col("kind").replace_strict(KIND, default="time"),
        signal=pl.Series(pct[t["row"].to_numpy()].astype(np.float64)),
        qv=pl.Series(qv[t["row"].to_numpy()] * 288.0),
        tp=pl.lit(tp),
    )


def sleeve(t: pl.DataFrame) -> pl.DataFrame:
    return t.select(
        "symbol", "adv", t_in=pl.col("t") + 300, t_out="t_out", net="net", qv="qv",
        sleeve=pl.lit("L"), prio=pl.lit(3.0) + pl.col("signal"),
    )  # fmt: skip


def by_year(t: pl.DataFrame) -> pl.DataFrame:
    return (
        t.group_by("year")
        .agg(
            n=pl.len(),
            mean_bp=pl.col("net").mean() * 1e4,
            median_bp=pl.col("net").median() * 1e4,
            tp_rate=(pl.col("kind") == 2).mean(),
            win=(pl.col("net") > 0).mean(),
            days=pl.col("day").n_unique(),
            coins=pl.col("symbol").n_unique(),
        )
        .sort("year")
    )


def md(df: pl.DataFrame, fmt: dict[str, str] | None = None) -> str:
    fmt = fmt or {}
    cols = df.columns
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.iter_rows():
        cells = []
        for c, v in zip(cols, r, strict=True):
            if isinstance(v, float):
                cells.append(format(v, fmt.get(c, ".3f")))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


YFMT = {"mean_bp": "+.0f", "median_bp": "+.0f", "tp_rate": ".2f", "win": ".2f"}
BFMT = {"bp": "+.0f", "ret": "+.3f", "dd": "+.3f", "worst_mtm": ".2f"}


def book_table(t: pl.DataFrame) -> pl.DataFrame:
    if t.height == 0:
        return pl.DataFrame()
    b = book(sleeve(t), 10)
    return b.select("year", "trades", "bp", "ret", "dd", "worst_mtm")


def diag_frame(t: pl.DataFrame, theta: float = 0.995) -> pl.DataFrame:
    return t.select(
        "signal", threshold=pl.lit(theta), net="net", mfe="mfe", mae="adv",
        tp=pl.lit(0.2, dtype=pl.Float64), stop=pl.lit(None, dtype=pl.Float64),
        exit_kind="exit_kind", year="year",
    )  # fmt: skip


def per_spike(m: pl.DataFrame, pct: np.ndarray, years: tuple[int, ...], lo: float) -> pl.DataFrame:
    """Every spike with pct >= lo as its own trade (no one-per-coin lock): L's exit rule."""
    yr = m["year"].to_numpy()
    idx = np.flatnonzero(np.isin(yr, years) & np.isfinite(pct) & (pct >= lo))
    p = sim.paths(idx)
    ret, _, kind = sim.exits(p, np.ones(len(idx)), Rule(hold_min=4320, tp=0.2))
    return pl.DataFrame({"pct": pct[idx], "net": ret - 0.004, "tp": kind == 2})


EDGES = [0.9, 0.95, 0.976, 0.984, 0.9955, 1.01]


def fixed_bins(ps: pl.DataFrame, n_all: int) -> pl.DataFrame:
    labels = [f"{a:.4g}-{b:.4g}" for a, b in itertools.pairwise(EDGES)]
    k = np.searchsorted(EDGES, ps["pct"].to_numpy(), side="right") - 1
    d = ps.with_columns(bin=pl.Series([labels[i] for i in k]))
    return (
        d.group_by("bin")
        .agg(n=pl.len(), mean_bp=pl.col("net").mean() * 1e4, tp_rate=pl.col("tp").mean())
        .with_columns(share=pl.col("n") / n_all)
        .sort("bin")
    )


def clustered(t: pl.DataFrame, label: str, n_boot: int = 10_000) -> str:
    """Day-clustered view of a trade list: trades on one flush day are not independent."""
    d = t.group_by("day").agg(n=pl.len(), s=pl.col("net").sum(), e=pl.col("net").mean()).sort("day")
    n, s = d["n"].to_numpy(), d["s"].to_numpy()
    rng = np.random.default_rng(0)
    k = rng.integers(0, len(d), (n_boot, len(d)))
    boot = s[k].sum(1) / n[k].sum(1) * 1e4
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return (
        f"{label}: {t.height} trades on {len(d)} days; trade mean {t['net'].mean() * 1e4:+.0f} bp,"
        f" mean of day means {d['e'].mean() * 1e4:+.0f} bp, positive days"
        f" {(d['e'] > 0).mean():.0%}, largest day {n.max()} trades; day-bootstrap 95% CI of the"
        f" trade mean [{lo:+.0f}, {hi:+.0f}] bp, P(mean <= 0) {(boot <= 0).mean():.4f}"
    )


def btc_year(path: Path, years: tuple[int, ...]) -> dict[int, float]:
    b = pl.read_parquet(path / "*.parquet", columns=["open_time", "close"]).sort("open_time")
    g = b.group_by(pl.col("open_time").dt.year().alias("y")).agg(
        first=pl.col("close").first(), last=pl.col("close").last(), lo=pl.col("close").min(),
        hi=pl.col("close").max(),
    )  # fmt: skip
    return {r["y"]: r["last"] / r["first"] - 1 for r in g.iter_rows(named=True) if r["y"] in years}


def baselines(m: pl.DataFrame, years: tuple[int, ...], label: str) -> list[str]:
    """L's exit rule on simpler entries: every liquid spike, and spikes after a 20% 24h fall."""
    lines = []
    r24 = m["ret_24h"].fill_null(0).to_numpy()
    for name, sig in (("every liquid spike", np.ones(m.height)),
                      ("spike after a >= 20% 24h fall", np.where(r24 <= -0.2, 1.0, 0.0))):  # fmt: skip  # noqa: E501
        t = trades(m, sig, years, 0.5)
        lines.append(
            f"- {label}, {name}: n {t.height}, mean {t['net'].mean() * 1e4:+.0f} bp, TP rate"
            f" {(t['kind'] == 2).mean():.2f}, days {t['day'].n_unique()}"
        )
    return lines


def main() -> None:
    pl.Config.set_tbl_rows(100)
    # ---------------------------------------------------------------- reference 2022-26
    sm = use_shared()
    sp = scores("day5")
    ref = trades(sm, sp, RY, 0.995)
    emit("## Reference: L on 2022-26 (shared caches, walk-forward day5)\n")
    emit(md(by_year(ref), YFMT))
    emit("")
    emit(md(book_table(ref), BFMT))
    emit("")
    yr_s = sm["year"].to_numpy()
    ref_ps = per_spike(sm, sp, RY, 0.9)
    n_ref = int((np.isin(yr_s, RY) & np.isfinite(sp)).sum())
    emit("Per-spike outcome by fixed pct bins (no per-coin lock), 2022-26:\n")
    emit(md(fixed_bins(ref_ps, n_ref), {"mean_bp": "+.0f", "tp_rate": ".2f", "share": ".4f"}))
    emit("")
    emit(clustered(ref, "reference 2022-26") + "\n")
    base_lines = baselines(sm.with_columns(pl.Series("ret_24h", sm["ret_24h"])), RY, "2022-26")
    btc = btc_year(
        Path("/Users/mike/code/crypto-moonshot/data/klines/spot/interval=5m/symbol=BTCUSDT"), RY
    )
    ref_neigh = {}
    for q in (0.99, 0.995, 0.998):
        for tp in (0.15, 0.2, 0.25):
            t = trades(sm, sp, RY, q, tp=tp)
            ref_neigh[(q, tp)] = (t.height, float(t["net"].mean()) * 1e4, t["day"].n_unique())

    # ---------------------------------------------------------------- holdout
    hm = use_holdout()
    sc = pl.read_parquet(HOLDOUT / "scores.parquet")
    yr = hm["year"].to_numpy()
    hy = np.isin(yr, HY)
    base_lines += baselines(hm, HY, "2017-19")
    btc |= btc_year(HOLDOUT / "klines/spot/interval=5m/symbol=BTCUSDT", HY)
    emit("## Baselines (L's exit, one position per coin, no slot limit)\n")
    emit("\n".join(base_lines) + "\n")
    emit(
        "BTC return by calendar year: "
        + ", ".join(f"{y} {v:+.0%}" for y, v in sorted(btc.items()))
        + "\n"
    )
    emit("## Holdout data\n")
    cov = (
        hm.filter(pl.col("year").is_in(HY))
        .group_by("year")
        .agg(spikes=pl.len(), coins=pl.col("symbol").n_unique(),
             med_qv_base=pl.col("qv_base").median(),
             first=pl.col("open_time").min(), last=pl.col("open_time").max())
        .sort("year")
    )  # fmt: skip
    emit(md(cov, {"med_qv_base": ".0f"}))
    emit("")
    models = [c.removeprefix("pct_") for c in sc.columns if c.startswith("pct_")]
    all_l: dict[str, pl.DataFrame] = {}
    for name in models:
        pct = sc[f"pct_{name}"].to_numpy()
        emit(f"## Model {name}\n")
        sh = (
            pl.DataFrame({"year": yr[hy], "pct": pct[hy]})
            .group_by("year")
            .agg(
                n=pl.len(),
                ge_095=(pl.col("pct") >= 0.95).mean(),
                ge_099=(pl.col("pct") >= 0.99).mean(),
                ge_0995=(pl.col("pct") >= 0.995).mean(),
                ge_0998=(pl.col("pct") >= 0.998).mean(),
            )
            .sort("year")
        )
        emit("Share of liquid spikes at each pct threshold:\n")
        emit(md(sh, {k: ".4f" for k in sh.columns if k.startswith("ge")}))
        emit("")
        lt = trades(hm, pct, HY, 0.995)
        all_l[name] = lt
        emit("### L trades by year (pre-registered: pct >= 0.995, TP 20%, 72h)\n")
        emit(md(by_year(lt), YFMT) if lt.height else "no trades")
        emit("")
        if lt.height:
            emit(
                f"All years: n {lt.height}, mean {lt['net'].mean() * 1e4:+.0f} bp, median"
                f" {lt['net'].median() * 1e4:+.0f} bp, TP rate {(lt['kind'] == 2).mean():.2f},"
                f" days {lt['day'].n_unique()}\n"
            )
            emit(clustered(lt, f"holdout 2017-19 ({name})") + "\n")
            emit("### 10-slot book ($10k fresh each year)\n")
            emit(md(book_table(lt), BFMT))
            emit("")
        # diag
        pool = trades(hm, pct, HY, 0.95, 0.995)
        if lt.height:
            fe, fn = diag_frame(lt), diag_frame(pool) if pool.height else None
            card = scorecard(annotate(fe))
            emit("### diag scorecard\n")
            emit(table({f"L holdout ({name})": card}))
            tr = threshold_reach(fe, fn)
            emit(
                f"threshold {tr['threshold']:.3f}: {tr['share_reaching']:.1%} of the pool reach"
                f" it; Spearman(signal, outcome) {tr['spearman_signal_outcome']:+.3f}\n"
            )
            cal = calibration(fe, fn)
            emit("### calibration by signal decile (pool pct >= 0.95)\n")
            emit(md(cal, {"bin": ".4f", "E_o": "+.3f"}))
            emit("")
        ps = per_spike(hm, pct, HY, 0.9)
        emit("Per-spike outcome by fixed pct bins (no per-coin lock), 2017-19:\n")
        emit(md(fixed_bins(ps, int((hy & np.isfinite(pct)).sum())),
                {"mean_bp": "+.0f", "tp_rate": ".2f", "share": ".4f"}))  # fmt: skip
        emit("")
        # neighbours
        rows = []
        for q in (0.99, 0.995, 0.998):
            for tp in (0.15, 0.2, 0.25):
                t = trades(hm, pct, HY, q, tp=tp)
                bt = book_table(t)
                rn = ref_neigh[(q, tp)]
                rows.append({
                    "rule": f"pct>={q} TP{int(tp * 100)}" + (" (TEST)" if (q, tp) == (0.995, 0.2) else ""),  # noqa: E501
                    "n": t.height, "mean_bp": float(t["net"].mean() or 0) * 1e4 if t.height else 0.0,  # noqa: E501
                    "tp_rate": float((t["kind"] == 2).mean()) if t.height else 0.0,
                    "days": t["day"].n_unique() if t.height else 0,
                    **{f"book{y}": float(bt.filter(pl.col("year") == y)["ret"].sum()) if bt.height else 0.0 for y in HY},  # noqa: E501
                    "ref_n": rn[0], "ref_bp": rn[1], "ref_days": rn[2],
                })  # fmt: skip
        emit("### neighbours (holdout 2017-19 vs reference 2022-26)\n")
        emit(md(pl.DataFrame(rows), {"mean_bp": "+.0f", "tp_rate": ".2f", "ref_bp": "+.0f",
                                     **{f"book{y}": "+.3f" for y in HY}}))  # fmt: skip
        emit("")
        if lt.height:
            fd = (
                lt.join(hm.with_row_index("row").select(pl.col("row").cast(pl.Int64),
                        "btc_ret_4h", "btc_ret_24h", "ret_4h", "ret_24h"), on="row", how="left")
                .group_by("day")
                .agg(n=pl.len(), coins=pl.col("symbol").unique().sort().str.join(","),
                     mean_bp=pl.col("net").mean() * 1e4, tp=(pl.col("kind") == 2).sum(),
                     coin_ret_24h=pl.col("ret_24h").median(),
                     btc_ret_24h=pl.col("btc_ret_24h").first(), pct_max=pl.col("signal").max())
                .sort("day")
            )  # fmt: skip
            emit("### flush days\n")
            emit(md(fd, {"mean_bp": "+.0f", "coin_ret_24h": "+.3f", "btc_ret_24h": "+.3f",
                         "pct_max": ".4f"}))  # fmt: skip
            emit("")
    pl.concat(
        [t.with_columns(model=pl.lit(k)) for k, t in all_l.items() if t.height],
        how="diagonal_relaxed",
    ).write_parquet(HOLDOUT / "l_trades.parquet")
    (HOLDOUT / "eval_tables.md").write_text("\n".join(out) + "\n")
    print("written", HOLDOUT / "eval_tables.md", file=sys.stderr)


if __name__ == "__main__":
    main()
