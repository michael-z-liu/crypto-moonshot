"""M6b: the one reveal of 2025-26 (docs/plans/ma-rebound.md, "M6 design" and "M6b"), idempotent.

    uv run --group research python m6_reveal.py            # compute what is missing, write all
    uv run --group research python m6_reveal.py --report   # only rewrite the tables and M6.md

Run once, after `results/M6_pick.md` was committed and pushed (there is no pick). Steps:

1. **Comparators.** Every distinct rule of a scored ledger row gets its *every candidate* row (the
   same rule without scores). Those that do not exist yet are scored through `scorecard` as stage
   `M6` rows named `every-...` (2022-24 to the ledger, all years to the sealed file).
2. **Every ledger row** is rebuilt as (rule, score table) from the item lists of `m4_score.py`,
   `m5_screen.py` and `m6_finalists.py`; the rebuild must reproduce the row's config hash and
   scores digest. Its 2025 and 2026 numbers come from its sealed file (name and hash checked).
   One full-range book per row (the book is causal, so its 2022-24 part is the ledger's) gives the
   pooled 2025-26 numbers, the daily mark-to-market curve for the Sharpe ratios, and checks
   against the sealed file and the ledger. Per-row results are cached in `cache/m6/reveal/`.
3. Tables in `results/m6/`: `reveal.csv` (every row), `comparators.csv`, `named.csv` (the three
   rows named in the plan), `named_seeds.csv`, `rank_corr.csv`, `dsr_trials.csv`, `dsr.json`;
   and `results/M6.md` (prose from `results/m6_summary_text.md`, written after reading the tables).

2026 ends at the data end, 2026-11-02 (exclusive). The 2025-26 summary number of a row is the mean
of its 2025 and 2026 book returns, next to the pooled 2025-26 figures. Revealed cards never go to
the ledger; the comparators' ledger rows are 2022-24 only, as every row's.
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from statistics import NormalDist
from typing import Any

import book as bk
import m5_screen as m5
import m6_finalists as m6
import numpy as np
import numpy.typing as npt
import polars as pl
import train as tr
from common import CACHE, INITIAL_CAPITAL, RESULTS, SEALED, TIMEFRAMES
from scorecard import (
    LEDGER,
    Rule,
    config_hash,
    score_array,
    scorecard,
    scores_digest,
    selected,
)

type F64 = npt.NDArray[np.float64]

STAGE = "M6"
OUT = RESULTS / "m6"
ROWS = CACHE / "m6" / "reveal"
REPORT = RESULTS / "M6.md"
SUMMARY = RESULTS / "m6_summary_text.md"
COMP_SOURCE = "every candidate (no model), M6 comparator"
BP = 1e4
ANNUAL = 365
BOOT_DRAWS = 2000
BOOT_SEED = 0
EULER_GAMMA = 0.5772156649015329
ALL_YEARS = (2022, 2023, 2024, 2025, 2026)
SEL_YEARS = (2022, 2023, 2024)
REV_YEARS = (2025, 2026)
TFS = tuple(TIMEFRAMES)
D2022 = datetime(2022, 1, 1, tzinfo=UTC)
D2025 = datetime(2025, 1, 1, tzinfo=UTC)
D2026 = datetime(2026, 1, 1, tzinfo=UTC)
DEND = datetime(2026, 11, 2, tzinfo=UTC)
PRIMARY = "m6-base5-base"
NAMED = (PRIMARY, "m6-breadth5-base", "m6-otherma5-pct0.9")
BASE_RULE = Rule(pct=0.80)


# --- trials: every ledger row as (rule, scores) ----------------------------------------------


def every_name(rule: Rule) -> str:
    """The *every candidate* row name of a rule (the M3 names for the M3 rules)."""
    r = replace(rule, pct=None)
    name = "every-touch" if r.family == "touch" else f"every-L{r.buffer:g}"
    name += f"-{r.exit}"
    if r.tfs is not None:
        name += "-tf" + "".join(r.tfs)
    if r.lengths is not None:
        name += "-n" + "-".join(str(n) for n in r.lengths)
    if r.majors:
        name += "-majors"
    if r.slots != 10:
        name += f"-slots{r.slots}"
    return name


def scored_bases() -> dict[str, tuple[Rule, str, str | None]]:
    """Base row name -> (rule with its percentile, score folder, fixed table or None)."""
    out: dict[str, tuple[Rule, str, str | None]] = {
        "cnn-base": (BASE_RULE, "base", None),
        "gbm-base": (BASE_RULE, "gbm", "seed0"),
    }
    for it in m5.items():
        assert it.variant is not None
        out[it.key] = (it.rule, it.variant, None)
    for it6 in m6.items():
        out[it6.key] = (it6.spec.rule, it6.model.out, None)
    return out


@dataclass(frozen=True)
class Trial:
    stage: str
    name: str
    chash: str
    source: str
    rule: Rule
    out: str | None  # score folder (None: every candidate)
    which: str | None  # table in it
    base: str  # name without the seed suffix
    seed: int | None

    @property
    def scored(self) -> bool:
        return self.out is not None

    def scores(self) -> pl.DataFrame | None:
        if self.out is None or self.which is None:
            return None
        return pl.read_parquet(tr.SCORES / self.out / f"{self.which}.parquet").select(
            "cid", "score"
        )


def ledger() -> pl.DataFrame:
    return pl.read_csv(LEDGER, infer_schema_length=None)


def trials() -> list[Trial]:
    bases = scored_bases()
    every: dict[str, Rule] = {every_name(r): replace(r, pct=None) for r, _, _ in bases.values()}
    every |= {f"every-L0.05-{e}": Rule(exit=e) for e in ("E1", "E2", "E3", "E4", "E5", "E6")}
    every["every-touch-E1"] = Rule(family="touch")
    out: list[Trial] = []
    for r in ledger().iter_rows(named=True):
        name, source = str(r["name"]), str(r["source"])
        m = re.fullmatch(r"(.*)-seed(\d+)", name)
        base, seed = (m.group(1), int(m.group(2))) if m else (name, None)
        if base in every:
            rule, sc_out, which = every[base], None, None
        else:
            rule, sc_out, fixed = bases[base]
            which = fixed or (f"seed{seed}" if seed is not None else "mean")
        t = Trial(
            str(r["stage"]), name, str(r["config_hash"]), source, rule, sc_out, which, base, seed
        )
        assert config_hash(rule, source) == t.chash, (
            f"{name}: rebuilt rule does not match the ledger hash"
        )
        out.append(t)
    return out


def ensure_comparators() -> list[str]:
    """Score the missing *every candidate* rows of the scored rules. Returns the new names."""
    lg = ledger()
    have = set(lg["name"].to_list())
    rules = {every_name(r): replace(r, pct=None) for r, _, _ in scored_bases().values()}
    new = []
    for name, rule in sorted(rules.items()):
        if name in have:
            continue
        t0 = time.perf_counter()
        scorecard(name, rule, None, stage=STAGE, source=COMP_SOURCE)
        print(f"comparator {name}: scored in {time.perf_counter() - t0:.0f}s", flush=True)
        new.append(name)
    return new


# --- one row ----------------------------------------------------------------------------------


def grid_equity(eq: pl.DataFrame, a: datetime, b: datetime) -> F64:
    """Equity at every midnight from a to b inclusive (the last mark at or before; the initial
    capital before the first)."""
    grid = pl.datetime_range(a, b, "1d", time_unit="us", time_zone="UTC", eager=True)
    j = (
        pl.DataFrame({"time": grid})
        .join_asof(eq.select("time", "equity").sort("time"), on="time", strategy="backward")
        .with_columns(pl.col("equity").fill_null(INITIAL_CAPITAL))
    )
    return j["equity"].to_numpy().astype(np.float64)


def sharpe_stats(e: F64) -> dict[str, float]:
    r = e[1:] / e[:-1] - 1
    sd = float(r.std(ddof=1)) if len(r) > 1 else 0.0
    sr = float(r.mean()) / sd if sd > 0 else 0.0
    s0 = float(r.std(ddof=0))
    z = (r - r.mean()) / (s0 if s0 > 0 else 1.0)
    return {
        "sr_daily": sr,
        "sr_annual": sr * math.sqrt(ANNUAL),
        "skew": float((z**3).mean()),
        "kurt": float((z**4).mean()),  # not excess: a normal has 3
        "n_days": float(len(r)),
    }


def tf_text(net: F64, tf: npt.NDArray[np.str_]) -> str:
    return "; ".join(
        f"{t} {int((tf == t).sum())}/{net[tf == t].mean() * BP:+.0f}"
        for t in TFS
        if (tf == t).any()
    )


def _year(t: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    return t.astype("datetime64[us]").astype("datetime64[Y]").astype(np.int64) + 1970


def _f(v: object) -> float:
    return float("nan") if v is None else float(v)  # type: ignore[arg-type]


def rebuild(t: Trial) -> dict[str, Any]:
    """The row's full-range book, its checks and its 2025-26 numbers (cached)."""
    path = ROWS / f"{t.name}-{t.chash}.json"
    if path.exists():
        return dict(json.loads(path.read_text()))
    t0 = time.perf_counter()
    sealed_path = SEALED / f"{t.name}-{t.chash}.json"
    sealed = json.loads(sealed_path.read_text())
    assert sealed["name"] == t.name, f"{t.name}: sealed file name"
    assert sealed["config_hash"] == t.chash, f"{t.name}: sealed file hash"
    lrow = (
        ledger()
        .filter((pl.col("name") == t.name) & (pl.col("config_hash") == t.chash))
        .row(0, named=True)
    )
    sc = t.scores()
    score = None if sc is None else score_array(sc)
    assert scores_digest(score) == lrow["scores_sha"], (
        f"{t.name}: scores digest differs from the ledger"
    )
    rule = t.rule
    assert (sc is None) == (rule.pct is None), f"{t.name}: scores and threshold go together"
    sel = selected(rule, score)
    c = bk.cands()
    lg = bk.legs(rule.family, rule.label_buffer, "base", rule.exit)
    rows = bk.order_rows(lg, sel[lg.cand] & bk.in_book(lg, bk.DATA_END_US), score)
    res = bk.book(lg, rows, slots=rule.slots)
    br = res.run
    net = br.net_return
    fy = _year(br.entry_t)
    tf = c.tf[lg.cand[br.row]]
    vs = lg.gross[br.row] - lg.btc[br.row]
    eq = res.equity.filter(pl.col("time") <= DEND)
    e_sel = grid_equity(eq, D2022, D2025)
    e_rev = grid_equity(eq, D2025, DEND)
    i26 = (D2026 - D2025).days
    # stress: the same selection on the stress legs (its own book)
    lgs = bk.legs(rule.family, rule.label_buffer, "stress", rule.exit)
    brs = bk.run(
        lgs,
        bk.order_rows(lgs, sel[lgs.cand] & bk.in_book(lgs, bk.DATA_END_US), score),
        slots=rule.slots,
    )
    s_net, s_fy = brs.net_return, _year(brs.entry_t)

    def bp(x: F64) -> float:
        return float(x.mean() * BP) if len(x) else float("nan")

    sy = sealed["years"]
    ss = sealed["stress"]
    mine_ret = {2025: e_rev[i26] / e_rev[0] - 1, 2026: e_rev[-1] / e_rev[i26] - 1}
    m = np.isin(fy, REV_YEARS)
    ms = np.isin(s_fy, REV_YEARS)
    seg = e_rev / np.maximum.accumulate(e_rev) - 1
    out: dict[str, Any] = {
        "stage": t.stage,
        "name": t.name,
        "config_hash": t.chash,
        "kind": "scored" if t.scored else "every",
        "base": t.base,
        "seed": t.seed,
        "rule": rule.describe(),
        "source": t.source,
        "comparator": every_name(rule) if t.scored else "",
    }
    for y in SEL_YEARS:
        out |= {
            f"trades_{y}": _f(lrow[f"trades_{y}"]),
            f"bp_{y}": _f(lrow[f"bp_{y}"]),
            f"ret_{y}": _f(lrow[f"ret_{y}"]),
        }
    out |= {
        "rank_value_2224": _f(lrow["rank_value"]),
        "qualifies_2224": bool(lrow["qualifies"]),
        "fragile_2224": bool(lrow["fragile"]),
        "stress_bp_2224": _f(lrow["stress_bp"]),
    }
    for y in REV_YEARS:
        d, st = sy[str(y)], ss[str(y)]
        out |= {
            f"trades_{y}": _f(d["trades"]),
            f"bp_{y}": _f(d["bp"]),
            f"win_{y}": _f(d["win"]),
            f"ret_{y}": _f(d["ret"]),
            f"dd_{y}": _f(d["dd"]),
            f"vs_btc_bp_{y}": _f(d["vs_btc_bp"]),
            f"stress_bp_{y}": _f(st["bp"]),
            f"fill_{y}": _f(d.get("fill_rate")),
            f"per_tf_{y}": "; ".join(
                f"{k} {v['trades']:.0f}/{v['bp']:+.0f}"
                for k, v in sealed["per_tf"][str(y)].items()
                if v["trades"]
            ),
        }
    out |= {
        "trades_2526": float(m.sum()),
        "bp_2526": bp(net[m]),
        "win_2526": float((net[m] > 0).mean()) if m.any() else float("nan"),
        "ret_2526": float(e_rev[-1] / e_rev[0] - 1),
        "dd_2526": float(seg.min()),
        "vs_btc_bp_2526": float(np.nanmean(vs[m]) * BP) if m.any() else float("nan"),
        "stress_bp_2526": bp(s_net[ms]),
        "per_tf_2526": tf_text(net[m], tf[m]),
        "summary_2526": (_f(sy["2025"]["ret"]) + _f(sy["2026"]["ret"])) / 2,
    }
    out |= {f"{k}_2224": v for k, v in sharpe_stats(e_sel).items()}
    out |= {f"{k}_2526": v for k, v in sharpe_stats(e_rev).items()}
    # checks: rebuilt book against the sealed file (all years) and the ledger (2022-24)
    chk_trades = all(int((fy == y).sum()) == int(sy[str(y)]["trades"]) for y in ALL_YEARS)
    chk_strades = all(int((s_fy == y).sum()) == int(ss[str(y)]["trades"]) for y in ALL_YEARS)
    bp_diff = max(
        (abs(bp(net[fy == y]) - _f(sy[str(y)]["bp"])) for y in ALL_YEARS if (fy == y).any()),
        default=0.0,
    )
    out |= {
        "chk_trades_vs_sealed": chk_trades and chk_strades,
        "chk_trades_vs_ledger": all(
            int((fy == y).sum()) == int(_f(lrow[f"trades_{y}"])) for y in SEL_YEARS
        ),
        "chk_bp_maxdiff": bp_diff,
        "chk_ret_maxdiff": max(abs(mine_ret[y] - _f(sy[str(y)]["ret"])) for y in REV_YEARS),
        "chk_ret2224_vs_ledger": abs((e_sel[-1] / INITIAL_CAPITAL - 1) - _f(lrow["ret_2224"])),
        "seconds": time.perf_counter() - t0,
    }
    ROWS.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"entry_t": br.entry_t, "net": net, "tf": tf, "vs": vs, "year": fy}).write_parquet(
        ROWS / f"{t.name}-{t.chash}_trades.parquet"
    )
    path.write_text(json.dumps(out, indent=1))
    return out


def trades_of(name: str, chash: str) -> pl.DataFrame:
    return pl.read_parquet(ROWS / f"{name}-{chash}_trades.parquet")


# --- statistics -------------------------------------------------------------------------------


def day_bootstrap(tr_: pl.DataFrame, years: tuple[int, ...] = REV_YEARS) -> tuple[float, float]:
    """95% CI of the pooled mean net bp per trade over `years`, fill days resampled within each
    year (2,000 draws, seed 0)."""
    d = (
        tr_.filter(pl.col("year").is_in(years))
        .with_columns(day=pl.col("entry_t") // bk.DAY)
        .group_by("day")
        .agg(year=pl.col("year").first(), s=pl.col("net").sum(), n=pl.len())
        .sort("day")
    )
    rng = np.random.default_rng(BOOT_SEED)
    s_tot = np.zeros(BOOT_DRAWS)
    n_tot = np.zeros(BOOT_DRAWS)
    for y in years:
        dy = d.filter(pl.col("year") == y)
        s, n = dy["s"].to_numpy(), dy["n"].to_numpy().astype(np.float64)
        if len(s) == 0:
            continue
        idx = rng.integers(0, len(s), (BOOT_DRAWS, len(s)))
        s_tot += s[idx].sum(1)
        n_tot += n[idx].sum(1)
    lo, hi = np.percentile(s_tot / n_tot * BP, [2.5, 97.5])
    return float(lo), float(hi)


def without_best_days(tr_: pl.DataFrame, k: int = 5) -> float:
    d = tr_.filter(pl.col("year").is_in(REV_YEARS)).with_columns(day=pl.col("entry_t") // bk.DAY)
    best = d.group_by("day").agg(s=pl.col("net").sum()).sort("s", descending=True).head(k)["day"]
    rest = d.filter(~pl.col("day").is_in(best.implode()))
    return float(rest["net"].mean() * BP)  # type: ignore[operator]


def spearman(a: list[float], b: list[float]) -> float:
    ra = pl.Series(a).rank("average").to_numpy()
    rb = pl.Series(b).rank("average").to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 2 else float("nan")


def expected_max_sr(var: float, n: int) -> float:
    """E[max SR] of n trials without edge, SR variance `var` (Bailey and Lopez de Prado, 2014)."""
    nd = NormalDist()
    return math.sqrt(var) * (
        (1 - EULER_GAMMA) * nd.inv_cdf(1 - 1 / n) + EULER_GAMMA * nd.inv_cdf(1 - 1 / (n * math.e))
    )


def psr(sr: float, sr0: float, t: float, skew: float, kurt: float) -> float:
    """Probabilistic Sharpe ratio P(true SR > sr0); SRs per observation."""
    den = math.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2)
    return NormalDist().cdf((sr - sr0) * math.sqrt(t - 1) / den)


# --- report -----------------------------------------------------------------------------------


def pc(v: object) -> str:
    x = _f(v)
    return f"{x * 100:+.1f}%" if x == x else "-"


def bpf(v: object) -> str:
    x = _f(v)
    return f"{x:+.0f}" if x == x else "-"


def n0(v: object) -> str:
    x = _f(v)
    return f"{x:.0f}" if x == x else "-"


def yn(v: object) -> str:
    return "yes" if bool(v) else "no"


def md(header: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        *["| " + " | ".join(r) + " |" for r in rows],
    ]


def write(rows: list[dict[str, Any]], new_comps: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(rows, infer_schema_length=None)
    df.write_csv(OUT / "reveal.csv")
    by = {r["name"]: r for r in rows}
    scored_ns = [r for r in rows if r["kind"] == "scored" and r["seed"] is None]
    every = [r for r in rows if r["kind"] == "every"]

    # comparators
    comp_rows = []
    for r in scored_ns:
        c = by[r["comparator"]]
        d: dict[str, Any] = {"name": r["name"], "stage": r["stage"], "comparator": c["name"]}
        for y, sfx in ((2025, "2025"), (2026, "2026"), ("2526", "2526")):
            d |= {
                f"ret_{sfx}": r[f"ret_{y}"],
                f"comp_ret_{sfx}": c[f"ret_{y}"],
                f"beats_ret_{sfx}": _f(r[f"ret_{y}"]) > _f(c[f"ret_{y}"]),
                f"bp_{sfx}": r[f"bp_{y}"],
                f"comp_bp_{sfx}": c[f"bp_{y}"],
                f"beats_bp_{sfx}": _f(r[f"bp_{y}"]) > _f(c[f"bp_{y}"]),
            }
        comp_rows.append(d)
    comp = pl.DataFrame(comp_rows, infer_schema_length=None)
    comp.write_csv(OUT / "comparators.csv")

    # named rows
    named_rows = []
    seed_rows = []
    for name in NAMED:
        r = by[name]
        c = by[r["comparator"]]
        trd = trades_of(name, r["config_hash"])
        ctr = trades_of(c["name"], c["config_hash"])
        lo, hi = day_bootstrap(trd)
        clo, chi = day_bootstrap(ctr)
        named_rows.append(
            {
                "name": name,
                "trades_2526": r["trades_2526"],
                "bp_2526": r["bp_2526"],
                "ci_lo_bp": lo,
                "ci_hi_bp": hi,
                "ex5_bp": without_best_days(trd),
                "ret_2025": r["ret_2025"],
                "ret_2026": r["ret_2026"],
                "ret_2526": r["ret_2526"],
                "summary_2526": r["summary_2526"],
                "dd_2526": r["dd_2526"],
                "vs_btc_bp_2526": r["vs_btc_bp_2526"],
                "stress_bp_2526": r["stress_bp_2526"],
                "per_tf_2025": r["per_tf_2025"],
                "per_tf_2026": r["per_tf_2026"],
                "per_tf_2526": r["per_tf_2526"],
                "comparator": c["name"],
                "comp_trades_2526": c["trades_2526"],
                "comp_bp_2526": c["bp_2526"],
                "comp_ci_lo_bp": clo,
                "comp_ci_hi_bp": chi,
                "comp_ret_2025": c["ret_2025"],
                "comp_ret_2026": c["ret_2026"],
                "comp_ret_2526": c["ret_2526"],
                "comp_summary_2526": c["summary_2526"],
                "comp_per_tf_2526": c["per_tf_2526"],
            }
        )
        for s in sorted(
            (x for x in rows if x["base"] == name and x["seed"] is not None),
            key=lambda x: x["seed"],
        ):
            seed_rows.append(
                {
                    "row": name,
                    "seed": s["seed"],
                    "rank_value_2224": s["rank_value_2224"],
                    **{
                        k: s[k]
                        for k in (
                            "trades_2025",
                            "bp_2025",
                            "ret_2025",
                            "trades_2026",
                            "bp_2026",
                            "ret_2026",
                            "trades_2526",
                            "bp_2526",
                            "ret_2526",
                            "summary_2526",
                        )
                    },
                }
            )
    named = pl.DataFrame(named_rows, infer_schema_length=None)
    named.write_csv(OUT / "named.csv")
    nseeds = pl.DataFrame(seed_rows, infer_schema_length=None)
    nseeds.write_csv(OUT / "named_seeds.csv")

    # rank correlation
    groups = {
        "all non-seed scored rows": scored_ns,
        "M5 rows": [r for r in scored_ns if r["stage"] == "M5"],
        "M6 rows": [r for r in scored_ns if r["stage"] == "M6"],
    }
    rc = []
    for g, rs in groups.items():
        rv = [_f(r["rank_value_2224"]) for r in rs]
        rc.append(
            {
                "group": g,
                "n": len(rs),
                "vs_summary_2526": spearman(rv, [_f(r["summary_2526"]) for r in rs]),
                "vs_ret_2526_pooled": spearman(rv, [_f(r["ret_2526"]) for r in rs]),
                "vs_bp_2526_pooled": spearman(rv, [_f(r["bp_2526"]) for r in rs]),
            }
        )
    rcdf = pl.DataFrame(rc)
    rcdf.write_csv(OUT / "rank_corr.csv")

    def positives(rs: list[dict[str, Any]]) -> tuple[int, int, int, int]:
        p25 = sum(_f(r["ret_2025"]) > 0 for r in rs)
        p26 = sum(_f(r["ret_2026"]) > 0 for r in rs)
        both = sum(_f(r["ret_2025"]) > 0 and _f(r["ret_2026"]) > 0 for r in rs)
        return len(rs), p25, p26, both

    pos_ns = positives(scored_ns)
    pos_seed = positives([r for r in rows if r["kind"] == "scored" and r["seed"] is not None])
    pos_every = positives(every)

    # deflated Sharpe ratio over the non-seed scored rows
    tri = pl.DataFrame(
        [
            {
                "name": r["name"],
                "stage": r["stage"],
                **{k: r[f"{k}_2224"] for k in ("sr_daily", "sr_annual", "skew", "kurt", "n_days")},
                "sr_annual_2526": r["sr_annual_2526"],
            }
            for r in scored_ns
        ]
    )
    tri.write_csv(OUT / "dsr_trials.csv")
    n_tr = tri.height
    var_d = float(tri["sr_daily"].var())  # type: ignore[arg-type]
    sr0 = expected_max_sr(var_d, n_tr)
    top_sr = str(tri.sort("sr_annual", descending=True)["name"][0])
    dsr: dict[str, Any] = {
        "n_trials": n_tr,
        "var_sr_daily": var_d,
        "sd_sr_annual": math.sqrt(float(tri["sr_annual"].var())),  # type: ignore[arg-type]
        "expected_max_sr_annual": sr0 * math.sqrt(ANNUAL),
        "trials_sr_annual_mean": float(tri["sr_annual"].mean()),  # type: ignore[arg-type]
        "trials_sr_annual_max": float(tri["sr_annual"].max()),  # type: ignore[arg-type]
        "rows": {},
    }
    for name in dict.fromkeys([*NAMED, top_sr]):
        p = tri.filter(pl.col("name") == name).row(0, named=True)
        dsr["rows"][name] = {
            **{k: p[k] for k in ("sr_daily", "sr_annual", "skew", "kurt", "n_days")},
            "rank_of_sr": int((tri["sr_annual"] > p["sr_annual"]).sum()) + 1,
            "dsr": psr(p["sr_daily"], sr0, p["n_days"], p["skew"], p["kurt"]),
            "dsr_normal": psr(p["sr_daily"], sr0, p["n_days"], 0.0, 3.0),
            "psr_vs_0": psr(p["sr_daily"], 0.0, p["n_days"], p["skew"], p["kurt"]),
            "sr_annual_2526": p["sr_annual_2526"],
            "top_2224_sharpe": name == top_sr,
        }
    (OUT / "dsr.json").write_text(json.dumps(dsr, indent=1))

    # checks
    bad = df.filter(
        ~pl.col("chk_trades_vs_sealed")
        | ~pl.col("chk_trades_vs_ledger")
        | (pl.col("chk_bp_maxdiff") > 1e-6)
        | (pl.col("chk_ret_maxdiff") > 1e-6)
        | (pl.col("chk_ret2224_vs_ledger") > 1e-3)
    )
    if bad.height:
        print(
            "WARNING: rows whose rebuild differs:",
            bad.select(
                "name",
                "chk_trades_vs_sealed",
                "chk_trades_vs_ledger",
                "chk_bp_maxdiff",
                "chk_ret_maxdiff",
                "chk_ret2224_vs_ledger",
            ),
            flush=True,
        )
    report(
        df,
        by,
        scored_ns,
        every,
        comp,
        named,
        nseeds,
        rcdf,
        dsr,
        (pos_ns, pos_seed, pos_every),
        bad,
        new_comps,
    )


def report(
    df: pl.DataFrame,
    by: dict[str, dict[str, Any]],
    scored_ns: list[dict[str, Any]],
    every: list[dict[str, Any]],
    comp: pl.DataFrame,
    named: pl.DataFrame,
    nseeds: pl.DataFrame,
    rcdf: pl.DataFrame,
    dsr: dict[str, Any],
    pos: tuple[tuple[int, int, int, int], ...],
    bad: pl.DataFrame,
    new_comps: list[str],
) -> None:
    hdr = [
        "row",
        "stage",
        "rank 2022-24",
        "trades 2025 / 2026",
        "bp 2025 / 2026 / pooled",
        "return 2025 / 2026",
        "summary (mean)",
        "pooled return",
        "max DD 2025-26",
        "stress bp pooled",
        "comparator: summary / pooled bp",
    ]

    def line(r: dict[str, Any]) -> list[str]:
        c = by.get(r["comparator"]) if r["comparator"] else None
        return [
            r["name"],
            r["stage"],
            pc(r["rank_value_2224"]),
            f"{n0(r['trades_2025'])} / {n0(r['trades_2026'])}",
            f"{bpf(r['bp_2025'])} / {bpf(r['bp_2026'])} / {bpf(r['bp_2526'])}",
            f"{pc(r['ret_2025'])} / {pc(r['ret_2026'])}",
            pc(r["summary_2526"]),
            pc(r["ret_2526"]),
            pc(r["dd_2526"]),
            bpf(r["stress_bp_2526"]),
            f"{pc(c['summary_2526'])} / {bpf(c['bp_2526'])}" if c else "-",
        ]

    order = {"M3": 0, "M4": 1, "M5": 2, "M6": 3}
    all_tab = md(hdr, [line(r) for r in sorted(scored_ns, key=lambda r: order[r["stage"]])])
    ev_tab = md(
        [
            "comparator",
            "stage",
            "rank 2022-24",
            "trades 2025 / 2026",
            "bp 2025 / 2026 / pooled",
            "return 2025 / 2026",
            "summary (mean)",
            "pooled return",
            "max DD 2025-26",
        ],
        [line(r)[:9] for r in sorted(every, key=lambda r: r["name"])],
    )
    nm = named.to_dicts()
    named_tab = md(
        [
            "row",
            "trades 2025-26",
            "bp pooled",
            "day-bootstrap 95% CI",
            "without 5 best days",
            "return 2025 / 2026 / pooled",
            "max DD",
            "vs BTC bp",
            "stress bp",
        ],
        [
            [
                x["name"],
                n0(x["trades_2526"]),
                bpf(x["bp_2526"]),
                f"[{bpf(x['ci_lo_bp'])}, {bpf(x['ci_hi_bp'])}]",
                bpf(x["ex5_bp"]),
                f"{pc(x['ret_2025'])} / {pc(x['ret_2026'])} / {pc(x['ret_2526'])}",
                pc(x["dd_2526"]),
                bpf(x["vs_btc_bp_2526"]),
                bpf(x["stress_bp_2526"]),
            ]
            for x in nm
        ],
    )
    named_comp = md(
        [
            "row",
            "comparator",
            "comparator trades",
            "comparator bp [CI]",
            "comparator return 2025 / 2026 / pooled",
            "row beats it: return 2025 / 2026 / pooled",
            "row beats it: bp pooled",
        ],
        [
            [
                x["name"],
                x["comparator"],
                n0(x["comp_trades_2526"]),
                f"{bpf(x['comp_bp_2526'])} [{bpf(x['comp_ci_lo_bp'])}, {bpf(x['comp_ci_hi_bp'])}]",
                f"{pc(x['comp_ret_2025'])} / {pc(x['comp_ret_2026'])} / {pc(x['comp_ret_2526'])}",
                " / ".join(
                    yn(_f(x[f"ret_{k}"]) > _f(x[f"comp_ret_{k}"])) for k in ("2025", "2026", "2526")
                ),
                yn(_f(x["bp_2526"]) > _f(x["comp_bp_2526"])),
            ]
            for x in nm
        ],
    )
    tf_tab = md(
        ["row", "2025 (trades/bp)", "2026 (trades/bp)", "2025-26 pooled (trades/bp)"],
        [[x["name"], x["per_tf_2025"], x["per_tf_2026"], x["per_tf_2526"]] for x in nm]
        + [
            [
                f"{x['comparator']} (no model)",
                by[x["comparator"]]["per_tf_2025"],
                by[x["comparator"]]["per_tf_2026"],
                x["comp_per_tf_2526"],
            ]
            for x in nm[:1]
        ],
    )
    seeds_tab = md(
        [
            "row",
            "seed",
            "rank 2022-24",
            "trades 2025-26",
            "bp 2025 / 2026 / pooled",
            "return 2025 / 2026",
            "summary",
            "pooled return",
        ],
        [
            [
                s["row"],
                str(s["seed"]),
                pc(s["rank_value_2224"]),
                n0(s["trades_2526"]),
                f"{bpf(s['bp_2025'])} / {bpf(s['bp_2026'])} / {bpf(s['bp_2526'])}",
                f"{pc(s['ret_2025'])} / {pc(s['ret_2026'])}",
                pc(s["summary_2526"]),
                pc(s["ret_2526"]),
            ]
            for s in nseeds.to_dicts()
        ],
    )
    cmp = comp.to_dicts()

    def count(k: str, rs: list[dict[str, Any]]) -> str:
        return f"{sum(bool(x[k]) for x in rs)} of {len(rs)}"

    comp_lines = []
    for label, rs in (
        ("all", cmp),
        ("M5", [x for x in cmp if x["stage"] == "M5"]),
        ("M6", [x for x in cmp if x["stage"] == "M6"]),
    ):
        comp_lines.append(
            [
                label,
                *[count(f"beats_ret_{k}", rs) for k in ("2025", "2026", "2526")],
                *[count(f"beats_bp_{k}", rs) for k in ("2025", "2026", "2526")],
            ]
        )
    comp_tab = md(
        [
            "rows",
            "return 2025",
            "return 2026",
            "return 2025-26",
            "bp 2025",
            "bp 2026",
            "bp 2025-26",
        ],
        comp_lines,
    )
    rc_tab = md(
        [
            "rows",
            "n",
            "vs summary (mean of 2025, 2026 returns)",
            "vs pooled 2025-26 return",
            "vs pooled 2025-26 bp per trade",
        ],
        [
            [
                x["group"],
                str(x["n"]),
                f"{x['vs_summary_2526']:+.2f}",
                f"{x['vs_ret_2526_pooled']:+.2f}",
                f"{x['vs_bp_2526_pooled']:+.2f}",
            ]
            for x in rcdf.to_dicts()
        ],
    )
    dsr_tab = md(
        [
            "row",
            "Sharpe 2022-24 (annual)",
            "rank among trials",
            "skew",
            "kurtosis",
            "DSR",
            "DSR (normal returns)",
            "PSR vs 0",
            "Sharpe 2025-26 (annual)",
        ],
        [
            [
                k + (" (highest 2022-24 Sharpe)" if v["top_2224_sharpe"] else ""),
                f"{v['sr_annual']:+.2f}",
                str(v["rank_of_sr"]),
                f"{v['skew']:+.2f}",
                f"{v['kurt']:.1f}",
                f"{v['dsr']:.3f}",
                f"{v['dsr_normal']:.3f}",
                f"{v['psr_vs_0']:.3f}",
                f"{v['sr_annual_2526']:+.2f}",
            ]
            for k, v in dsr["rows"].items()
        ],
    )
    pos_tab = md(
        ["rows", "n", "positive 2025", "positive 2026", "positive both"],
        [
            [label, str(p[0]), str(p[1]), str(p[2]), str(p[3])]
            for label, p in zip(
                ("non-seed scored", "seed rows", "every candidate"), pos, strict=True
            )
        ],
    )
    # M2 context
    t2 = pl.read_csv(OUT / "m2_t2_timeframe_year.csv", infer_schema_length=None)
    t3 = pl.read_csv(OUT / "m2_t3_study_vs_placebo.csv", infer_schema_length=None)
    m2_rows = []
    for x in t2.filter(pl.col("year_g").is_in(["2025", "2026", "all"])).to_dicts():
        m2_rows.append(
            [
                x["tf_g"],
                x["year_g"],
                x["entry"],
                x["exit"],
                n0(x["trades"]),
                bpf(x["mean_bp"]),
                f"[{bpf(x['ci_lo_bp'])}, {bpf(x['ci_hi_bp'])}]",
                pc(x["win_rate"]).lstrip("+"),
                pc(x["fill_rate"]).lstrip("+"),
            ]
        )
    m2_tab = md(
        [
            "timeframe",
            "fill year",
            "entry",
            "exit",
            "trades",
            "mean bp",
            "day-bootstrap CI",
            "win rate",
            "fill rate (trigger year)",
        ],
        m2_rows,
    )
    m3_tab = md(
        [
            "timeframe",
            "entry",
            "exit",
            "study trades",
            "study bp",
            "placebo trades",
            "placebo bp",
            "study - placebo bp",
            "CI",
        ],
        [
            [
                x["tf_g"],
                x["entry"],
                x["exit"],
                n0(x["study_trades"]),
                bpf(x["study_bp"]),
                n0(x["placebo_trades"]),
                bpf(x["placebo_bp"]),
                bpf(x["diff_bp"]),
                f"[{bpf(x['diff_lo_bp'])}, {bpf(x['diff_hi_bp'])}]",
            ]
            for x in t3.to_dicts()
        ],
    )
    chk = (
        f"All {df.height} rows rebuilt: trades per year match the sealed files (base and stress books, 2022-26) and the ledger (2022-24); "
        f"largest bp difference to the sealed file {df['chk_bp_maxdiff'].max():.2e}, largest 2025/2026 return difference {df['chk_ret_maxdiff'].max():.2e}, "
        f"largest 2022-24 return difference to the ledger {df['chk_ret2224_vs_ledger'].max():.2e}."
        if bad.height == 0
        else f"{bad.height} rows whose rebuild differs: {', '.join(bad['name'].to_list())}."
    )
    summary = SUMMARY.read_text().strip() if SUMMARY.exists() else "(summary not written yet)"
    body = [
        "# M6: the reveal of 2025-26",
        "",
        summary,
        "",
        "## How this was computed",
        "",
        "- One reveal, after `results/M6_pick.md` was committed and pushed (commits 849cab2, 5cf2ca2): every ledger",
        f"  row ({df.height} including the {len(new_comps)} new comparators) from its sealed file, plus one full-range",
        "  book per row for the pooled 2025-26 figures and the daily curves. Script: `m6_reveal.py`.",
        "- 2025 is a full year; **2026 runs to the data end, 2026-11-02 (exclusive), about 10 months.** The summary",
        "  number of a row is the mean of its 2025 and 2026 book returns; *pooled* figures treat 2025-01-01 to the",
        "  data end as one period (book return from the equity on 2025-01-01; the book itself is continuous from",
        "  2022). Positions still open at the data end close at the coin's last 5m close (book.py).",
        f"- Checks: {chk}",
        f"- New comparator rows (stage M6, 2022-24 in the ledger, benchmarks, not trials): {', '.join(new_comps) or 'none'}.",
        "- All rows, seed rows included: `results/m6/reveal.csv`. Comparisons: `comparators.csv`; the three named",
        "  rows: `named.csv`, `named_seeds.csv`; `rank_corr.csv`; `dsr_trials.csv`, `dsr.json`; M2 context: `m2_*.csv`.",
        "",
        "## All scored rows (non-seed), 2025-26",
        "",
        "Rank 2022-24 is the ledger's rank value (mean yearly book return 2022-24). Comparator: *every candidate* under",
        "the same rule (no scores). Book returns are of the 10-slot book unless the rule says otherwise.",
        "",
        *all_tab,
        "",
        "## The three rows named before the reveal",
        "",
        f"`{PRIMARY}` is the protocol's finalist (the primary row). `m6-breadth5-base` and `m6-otherma5-pct0.9` were",
        "named as comparisons because they ranked first on 2022-24 (base-rule rows, all 25 rows), so their 2022-24",
        "numbers are biased upward by that selection. None of the three can be confirmed: there was no pick.",
        "",
        "The CI resamples fill days within each year (2,000 draws, seed 0) for the pooled 2025-26 mean bp per trade.",
        "",
        *named_tab,
        "",
        "Against *every candidate* under the same rule (for the percentile-0.9 row: the base rule without scores):",
        "",
        *named_comp,
        "",
        "Per timeframe of the MA (trades / mean bp):",
        "",
        *tf_tab,
        "",
        "Seed by seed (each seed's own percentile table under the row's rule):",
        "",
        *seeds_tab,
        "",
        "## Comparators: every scored row against *every candidate* under its rule",
        "",
        "How many non-seed scored rows beat their no-model comparator (higher book return, higher bp per trade):",
        "",
        *comp_tab,
        "",
        "The comparators themselves (no model; the M3 rows and the M6 additions):",
        "",
        *ev_tab,
        "",
        "## Rank correlation",
        "",
        "Spearman correlation between the 2022-24 rank value and 2025-26 outcomes, non-seed scored rows:",
        "",
        *rc_tab,
        "",
        "Rows with a positive book return:",
        "",
        *pos_tab,
        "",
        "## Deflated Sharpe ratio",
        "",
        f"Trials: the {dsr['n_trials']} non-seed scored ledger rows (M4, M5, M6; seed rows and *every candidate* rows",
        "excluded). Each trial's Sharpe ratio comes from its book's daily mark-to-market returns from 2022-01-01 to",
        f"2025-01-01 (annualised with sqrt(365)). Their annual Sharpe ratios average {dsr['trials_sr_annual_mean']:+.2f}",
        f"(sd {dsr['sd_sr_annual']:.2f}, max {dsr['trials_sr_annual_max']:+.2f}); the expected maximum of {dsr['n_trials']} trials without",
        f"edge is {dsr['expected_max_sr_annual']:+.2f}. DSR = P(true Sharpe > that maximum) by Bailey and Lopez de Prado (2014),",
        "with the trial's skew and kurtosis, and with normal returns (skew 0, kurtosis 3).",
        "",
        *dsr_tab,
        "",
        "## Context without a model: M2's tables on 2025-26",
        "",
        "Every study candidate's trade (deduplicated as in M2), by fill year; `all` pools 2025 and 2026. Labels still",
        "open at the data end (14,400 label rows over all buffers, scenarios and exits) are left out, as M2 leaves out",
        "unfinished labels. `m2_baseline.py --holdout`; the 2020-24 outputs are unchanged.",
        "",
        *m2_tab,
        "",
        "Study MAs against placebo MAs (lengths 37, 73, 137, 163), fills 2025-01-01 to the data end:",
        "",
        *m3_tab,
        "",
    ]
    REPORT.write_text("\n".join(body))
    print(f"wrote {REPORT} and {OUT}/", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--report", action="store_true", help="only rewrite the tables and M6.md")
    a = p.parse_args()
    new = [] if a.report else ensure_comparators()
    comps_file = ROWS / "new_comparators.json"
    ROWS.mkdir(parents=True, exist_ok=True)
    if new:
        old = json.loads(comps_file.read_text()) if comps_file.exists() else []
        comps_file.write_text(json.dumps(sorted(set(old) | set(new))))
    new_all: list[str] = json.loads(comps_file.read_text()) if comps_file.exists() else []
    ts = trials()
    t0 = time.perf_counter()
    rows = []
    for k, t in enumerate(ts):
        rows.append(rebuild(t))
        if (k + 1) % 20 == 0:
            print(f"rows {k + 1}/{len(ts)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    write(rows, new_all)


if __name__ == "__main__":
    main()
