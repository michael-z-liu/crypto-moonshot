"""S2 of the CNN robustness study (docs/plans/cnn-robustness.md): regime and execution.

    cd research/volume_spike
    export MOONSHOT_UNIVERSE=all
    uv run --group research python s2_regime.py score          # every fixed rule x {path_all, ens3}
    uv run --group research python s2_regime.py timing         # entry-timing diagnostic (1m)
    uv run --group research python s2_regime.py report         # markdown tables from the CSVs

`score` runs the 14 pre-registered rules on both score sets through the scorecard (ledger rows
named `S2:<scores>:<rule>`), writes the cards to results/robustness/cards/ and a summary with the
fragile flag to results/robustness/S2_rules.csv. `timing` writes S2_timing.csv (not in the
ledger, not an engine rule).

All filters use only information available at the close of the signal candle:

- BTC flush: `btc_ret_4h`, `btc_vol_mult` of the signal candle (seq2_meta).
- Breadth: `spike_share` of market.parquet at the signal candle (share of the universe whose
  volume is >= 5x its 24h median at that candle); thresholds are the median / 75th percentile of
  the spike rows of 2020-21 (the walk-forward scores start in 2022, so the 2020-21 rows of
  seq2_meta are the population); and the number of candidates (`score >= q`) on the same UTC day
  up to and including this candle's open time.
- Own performance: two passes. Pass 1 is the ungated book (same scores and rule); a candidate is
  allowed when the mean net return of the 20 ungated trades closed (exit time <= entry time of
  the candidate) most recently is >= 0, or fewer than 20 have closed.
- Liquidity / tick: `qv_base` (seq2_meta), `tick_rel` (events.parquet, joined on event).
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import scorecard as sc
from common import CACHE
from s0_baseline import scores as s0_scores

from crypto_moonshot.broker.base import Costs
from crypto_moonshot.models import Interval

WORKERS = 4
T0 = datetime(2020, 1, 1, tzinfo=UTC)
STEP_US = 300_000_000
FLUSH_RET = (-0.02, -0.04)
FLUSH_VOL = (4.0, 6.0)
OWN_N = 20
BASE_COSTS = sc.COSTS


@dataclass(frozen=True)
class Features:
    """Per-row features aligned with seq2_meta, all known at the close of the signal candle."""

    qv_base: np.ndarray
    btc_ret_4h: np.ndarray
    btc_vol_mult: np.ndarray
    tick_rel: np.ndarray
    spike_share: np.ndarray
    share_p50: float
    share_p75: float


def features() -> Features:
    m = pl.read_parquet(
        CACHE / "seq2_meta.parquet",
        columns=["event", "open_time", "qv_base", "btc_ret_4h", "btc_vol_mult"],
    )
    ev = pl.read_parquet(CACHE / "events.parquet", columns=["row", "tick_rel"])
    tick = m.select("event").join(ev, left_on="event", right_on="row", how="left")["tick_rel"]
    market = pl.read_parquet(CACHE / "market.parquet")
    share_all = market["spike_share"].to_numpy()
    t0_us = int((T0 - datetime(1970, 1, 1, tzinfo=UTC)) / timedelta(microseconds=1))
    gidx = (sc.meta().t_us - t0_us) // STEP_US
    share = share_all[gidx]
    train = sc.meta().t_us < sc._us(sc.START)
    return Features(
        qv_base=m["qv_base"].to_numpy(),
        btc_ret_4h=m["btc_ret_4h"].to_numpy(),
        btc_vol_mult=m["btc_vol_mult"].to_numpy(),
        tick_rel=tick.to_numpy(),
        spike_share=share,
        share_p50=float(np.median(share[train])),
        share_p75=float(np.percentile(share[train], 75)),
    )


def candidates_today(score: np.ndarray, q: float) -> np.ndarray:
    """Per row: how many candidates (`score >= q`) share its UTC day and opened at or before it
    (this candle's own co-candidates included). 0 for non-candidates."""
    m = sc.meta()
    with np.errstate(invalid="ignore"):
        cand = np.isfinite(score) & (score >= q)
    k = np.flatnonzero(cand)
    f = pl.DataFrame({"k": k, "day": m.day[k], "t": m.t_us[k]})
    f = f.with_columns(n=pl.col("t").rank("max").over("day").cast(pl.Int64))
    out = np.zeros(len(score), dtype=np.int64)
    out[f["k"].to_numpy()] = f["n"].to_numpy()
    return out


def own_gate(score: np.ndarray, rule: sc.Rule) -> np.ndarray:
    """Allow a row if the 20 most recently closed ungated trades (exit <= its entry time) have a
    mean net return >= 0, or fewer than 20 have closed. Closed trades come from the ungated book.
    """
    res = sc.book(sc.signals(score, rule), rule)
    tr = res.trades.sort("exit_time")
    exit_us = tr["exit_time"].dt.epoch("us").to_numpy().astype(np.int64)
    cs = np.concatenate(([0.0], np.cumsum(tr["net_return"].to_numpy())))
    entry_us = sc.meta().t_us + STEP_US * (1 + rule.entry_delay)
    n = np.searchsorted(exit_us, entry_us, side="right")
    lo = np.maximum(n - OWN_N, 0)
    mean = (cs[n] - cs[lo]) / OWN_N
    return (n < OWN_N) | (mean >= 0)


def rule_set(f: Features, score: np.ndarray) -> dict[str, tuple[np.ndarray | None, sc.Rule, str]]:
    """name -> (row filter, rule, description), the S2 list in the plan's order."""
    base = sc.Rule()
    day_n = candidates_today(score, base.q)

    def lat(x: float) -> sc.Rule:
        return replace(
            base, costs=Costs(fee=BASE_COSTS.fee, slippage=BASE_COSTS.slippage, latency=x)
        )

    with np.errstate(invalid="ignore"):
        rules: dict[str, tuple[np.ndarray | None, sc.Rule, str]] = {
            "btc4h<=-2%": (f.btc_ret_4h <= -0.02, base, "btc_ret_4h <= -2%"),
            "btc4h<=-4%": (f.btc_ret_4h <= -0.04, base, "btc_ret_4h <= -4%"),
            "btcvol>=4": (f.btc_vol_mult >= 4, base, "btc_vol_mult >= 4"),
            "btcvol>=6": (f.btc_vol_mult >= 6, base, "btc_vol_mult >= 6"),
            "share>=p50": (
                f.spike_share >= f.share_p50,
                base,
                f"spike_share >= 2020-21 median {f.share_p50:.4f}",
            ),
            "share>=p75": (
                f.spike_share >= f.share_p75,
                base,
                f"spike_share >= 2020-21 p75 {f.share_p75:.4f}",
            ),
            "day>=3": (day_n >= 3, base, "candidates so far today (incl. this) >= 3"),
            "own20>=0": (own_gate(score, base), base, "mean of last 20 closed trades >= 0"),
            "qv>=20k": (f.qv_base >= 20_000, base, "qv_base >= 20k"),
            "qv>=50k": (f.qv_base >= 50_000, base, "qv_base >= 50k"),
            "tick<=0.1%": (f.tick_rel <= 0.001, base, "tick_rel <= 0.1%"),
            "lat0.05%": (None, lat(0.0005), "latency 0.05% per side"),
            "lat0.1%": (None, lat(0.001), "latency 0.1% per side"),
            "delay2": (None, replace(base, entry_delay=2), "entry delay 2 candles"),
        }
    return rules


def fragile(trades: pl.DataFrame) -> bool:
    """Some 2022-24 year with positive bp is <= 0 without its single best trade."""
    for y in sc.SELECT_YEARS:
        r = trades.filter(pl.col("entry_time").dt.year() == y)["net_return"].sort(descending=True)
        if r.len() > 1 and r.mean() > 0 and r[1:].mean() <= 0:  # type: ignore[operator]
            return True
    return False


def score_all() -> None:
    f = features()
    print(f"share p50 {f.share_p50:.4f} p75 {f.share_p75:.4f}", flush=True)
    cards = sc.OUT / "cards"
    cards.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in ("path_all", "ens3"):
        score, source = s0_scores(name)
        for rname, (mask, rule, desc) in rule_set(f, score).items():
            variant = f"S2:{name}:{rname}"
            src = f"{source}; S2 {desc}"
            card = sc.scorecard(
                variant, score, rule, stage="S2", source=src, row_filter=mask, workers=WORKERS
            )
            text = sc.show(card)
            print(text, flush=True)
            (
                cards / f"{variant.replace(':', '_').replace('>', 'ge').replace('<', 'le')}.txt"
            ).write_text(text + "\n")
            run = sc.run_book(sc.signals(score, rule, mask), rule, sc.SELECT_END)
            y = card.years
            rows.append(
                {
                    "scores": name,
                    "rule": rname,
                    "bp_2022": y[2022]["bp"],
                    "bp_2023": y[2023]["bp"],
                    "bp_2024": y[2024]["bp"],
                    "trades_2022": y[2022]["trades"],
                    "trades_2023": y[2023]["trades"],
                    "trades_2024": y[2024]["trades"],
                    "mean_bp": card.mean_bp,
                    "worst_bp": card.verdict.rank_key[1],
                    "cost06_bp": sc.mean_bp(card.cost06),
                    "late_bp": sc.mean_bp(card.late),
                    "ex5_bp": card.ex5_bp,
                    "placebo_p95": card.placebo_p95,
                    "ci_lo": card.ci[0],
                    "ci_hi": card.ci[1],
                    "fragile": fragile(run.trades),
                    "qualifies": card.verdict.qualifies,
                    "why": "; ".join(card.verdict.reasons),
                }
            )
    # the unfiltered baselines are in the S0 cards; add them for the table
    pl.DataFrame(rows).write_csv(sc.OUT / "S2_rules.csv")


def baseline_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name in ("path_all", "ens3"):
        score, _ = s0_scores(name)
        rule = sc.Rule()
        sig = sc.signals(score, rule)
        card = sc.scorecard(name, score, rule, ledger=False, workers=WORKERS)
        run = sc.run_book(sig, rule, sc.SELECT_END)
        y = card.years
        rows.append(
            {
                "scores": name,
                "rule": "baseline",
                "bp_2022": y[2022]["bp"],
                "bp_2023": y[2023]["bp"],
                "bp_2024": y[2024]["bp"],
                "trades_2022": y[2022]["trades"],
                "trades_2023": y[2023]["trades"],
                "trades_2024": y[2024]["trades"],
                "mean_bp": card.mean_bp,
                "worst_bp": card.verdict.rank_key[1],
                "cost06_bp": sc.mean_bp(card.cost06),
                "late_bp": sc.mean_bp(card.late),
                "ex5_bp": card.ex5_bp,
                "placebo_p95": card.placebo_p95,
                "ci_lo": card.ci[0],
                "ci_hi": card.ci[1],
                "fragile": fragile(run.trades),
                "qualifies": card.verdict.qualifies,
                "why": "; ".join(card.verdict.reasons),
            }
        )
    return rows


# ----------------------------------------------------------------------------- timing


def timing() -> None:
    """The baseline path_all book's 2022-24 trades, re-priced on 1m candles: enter at the open
    of the 5m entry (t+5m, as the engine), at minute 3 (t+3m) and minute 4 (t+4m) of the spike
    candle; exit 60 minutes after each entry (the close of the last 1m candle before). Costs
    0.1% fee + 0.05% slippage per side, as the book."""
    score, _ = s0_scores("path_all")
    rule = sc.Rule()
    res = sc.book(sc.signals(score, rule, end=sc.SELECT_END), rule, sc.SELECT_END)
    tr = res.trades.filter(pl.col("entry_time").dt.year().is_in(sc.SELECT_YEARS))
    src = sc.source()
    costs = BASE_COSTS
    offsets = {"t+5m (engine entry)": 5, "t+4m": 4, "t+3m": 3}
    out = []
    for r in tr.iter_rows(named=True):
        t = r["signal_time"]
        c = src.window(r["symbol"], Interval.M1, t, t + timedelta(minutes=70))
        if c is None or len(c) == 0:
            continue
        ot = c.open_time
        rec: dict[str, object] = {"symbol": r["symbol"], "signal_time": t, "year": t.year}
        ok = True
        for label, off in offsets.items():
            e_us = (
                int((t - datetime(1970, 1, 1, tzinfo=UTC)) / timedelta(microseconds=1))
                + off * 60_000_000
            )
            x_us = (
                e_us + 59 * 60_000_000
            )  # the 1m candle opening 59 minutes later; exit at its close
            i, j = np.searchsorted(ot, [e_us, x_us])
            if i >= len(ot) or j >= len(ot) or ot[i] != e_us or ot[j] != x_us:
                ok = False
                break
            buy = costs.buy_price(float(c.open[i]))
            sell = costs.sell_price(float(c.close[j]), market=True)
            rec[label] = (sell / buy) * (1 - costs.fee) / (1 + costs.fee) - 1
        if ok:
            out.append(rec)
    df = pl.DataFrame(out)
    labels = list(offsets)
    by_year = (
        df.group_by("year")
        .agg(n=pl.len(), **{f"bp {k}": pl.col(k).mean() * 1e4 for k in labels})
        .sort("year")
    )
    allm = df.select(
        n=pl.len(), **{f"bp {k}": pl.col(k).mean() * 1e4 for k in labels}
    ).with_columns(year=pl.lit(0, dtype=pl.Int64))
    table = pl.concat([by_year, allm.select(by_year.columns)])
    # the same trades, paired differences vs the engine entry, with a day-bootstrap-free t stat
    for k in labels[1:]:
        d = (df[k] - df[labels[0]]).to_numpy() * 1e4
        print(
            f"{k} minus engine entry: mean {d.mean():+.1f} bp,"
            f" t {d.mean() / (d.std(ddof=1) / np.sqrt(len(d))):+.2f}"
        )
    table.write_csv(sc.OUT / "S2_timing.csv")
    print(f"trades re-priced {df.height} of {tr.height}")
    with pl.Config(tbl_rows=20, tbl_width_chars=200):
        print(table)


# ----------------------------------------------------------------------------- report


def report() -> None:
    t = pl.read_csv(sc.OUT / "S2_rules.csv")
    base = pl.DataFrame(baseline_rows())
    t = pl.concat([base, t.select(base.columns)])
    t = t.with_columns(pl.col("fragile").cast(pl.Boolean), pl.col("qualifies").cast(pl.Boolean))
    for name in ("path_all", "ens3"):
        print(f"\n### {name}\n")
        print(
            "| Rule | bp 2022 | bp 2023 | bp 2024 | Mean | 0.6% | Late | w/o 5 days"
            " | Trades/yr | Verdict | Fragile |"
        )
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in t.filter(pl.col("scores") == name).iter_rows(named=True):
            n = (r["trades_2022"] + r["trades_2023"] + r["trades_2024"]) / 3
            print(
                f"| {r['rule']} | {r['bp_2022']:+.0f} | {r['bp_2023']:+.0f} | {r['bp_2024']:+.0f} |"
                f" {r['mean_bp']:+.0f} | {r['cost06_bp']:+.0f} | {r['late_bp']:+.0f} |"
                f" {r['ex5_bp']:+.0f} | {n:.0f} | {'qualifies' if r['qualifies'] else 'fails'} |"
                f" {'yes' if r['fragile'] else 'no'} |"
            )


CMDS: dict[str, Callable[[], None]] = {"score": score_all, "timing": timing, "report": report}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in CMDS:
        raise SystemExit(__doc__)
    CMDS[sys.argv[1]]()
