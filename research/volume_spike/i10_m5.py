"""Issue #10, M5: combine, pick, reveal and the deflated Sharpe ratio.

    cd research/volume_spike
    export MOONSHOT_UNIVERSE=all
    uv run --group research python i10_m5.py score [volscale5 day5 volday5] [--only esc,...]
    uv run --group research python i10_m5.py sanity        # seeds 0-2 vs the 3-seed runs
    uv run --group research python i10_m5.py pick          # 2022-24 table + pick -> M5_pick.md
    # commit M5_pick.md and the ledger and PUSH before the next step (the proof of order)
    uv run --group research python i10_m5.py reveal        # 2025-26 of every #10 row and seed
    uv run --group research python i10_m5.py dsr           # deflated Sharpe over all trials
    uv run --group research python i10_m5.py summary       # the reveal tables (markdown)

The plan's Combination rule (docs/plans/cnn-improvements.md): the two training variants that
beat base3 (M4 `volscale`, M3 `day`) retrained with 5 seeds (`volscale5`, `day5`) and both
changes together (`volday5`), each scored with six rules, all `CashCapped(5)`, pct >= 0.98,
highest score first, fee 0.1% + slippage 0.05% + latency 0.05% per side (0.4% round trip):

    a_h1_1h   h1 head (1h median), 1h hold                          eligible for the pick
    b_h1_2h   h1 head, 2h hold                                      eligible
    c_h2_2h   h2 head (2h median), 2h hold (M2 `h2_2h`)             eligible
    d_h2_1h   h2 head, 1h hold (M2 `h2_1h`)                         eligible
    esc       h1, 1h, escalating threshold per cascade (M2 `esc`)  comparison row only
    esc_2h    `esc` with a 2h hold                                  comparison row only

Ledger rows `M5:<scores>:<rule>` (20 placebos) and per-seed rows in `seeds.csv`. The pick is the
top-ranked eligible run that qualifies and is not fragile (seed-mean 2022-24 mean bp, worst year
as the tiebreak); the runner-up is the next.

`reveal` reads the sealed 2025-26 numbers (`results/issue10/sealed/`, checked against the
ledger's / seeds.csv's config hash) of every #10 ledger row and every per-seed row, re-runs each
book to the data end (its trade counts must equal the sealed book's) for the 2025-26 drawdown,
daily equity and the 2025-10-10 trades, and writes `M5_reveal.csv` / `M5_reveal_seeds.csv`.
`dsr` rebuilds every trial's 2022-24 book (no placebos): the robustness study's 115 ledger rows
(`s5_confirm.ledger_trials`, checked against `results/robustness/S5_trials.csv`) and every #10
ledger row, takes daily mark-to-market returns (annualised with sqrt(365)) and computes the
deflated Sharpe ratio (Bailey & Lopez de Prado, 2014) of the pick and the runner-up, as
`s5_confirm.dsr` does. Nothing here writes to the robustness ledger.
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any

import i10_m2
import i10_score as sc10
import i10_train
import numpy as np
import polars as pl
import s5_confirm as s5
import scorecard as sc

NAMES = ("volscale5", "day5", "volday5")
REF3 = {"volscale5": "volscale", "day5": "day"}  # the 3-seed runs (sanity check)
STAGE = "M5"
WORKERS = 8
PICK_CSV = sc10.OUT / "M5_table.csv"
EQUITY = i10_train.WF_I10 / "m5_equity"  # local, gitignored
ROBUST = sc10.HERE / "results" / "robustness"
CRASH_DAY = date(2025, 10, 10)
ColsFn = Callable[[np.ndarray], dict[str, np.ndarray]]
FilterFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class M5Rule:
    name: str
    head: str
    rule: sc.Rule
    eligible: bool
    cols: ColsFn | None = None
    note: str = ""


def rules() -> dict[str, M5Rule]:
    one, two = s5.rules()["a_1h"][1], s5.rules()["d_hold2h"][1]
    esc = i10_m2.ESC_TIERS
    out = [
        M5Rule("a_h1_1h", "h1", one, True, note="(a) h1, 1h"),
        M5Rule("b_h1_2h", "h1", two, True, note="(b) h1, 2h"),
        M5Rule("c_h2_2h", "h2", two, True, note="(c) h2, 2h (M2 h2_2h)"),
        M5Rule("d_h2_1h", "h2", one, True, note="(d) h2, 1h (M2 h2_1h)"),
        M5Rule(
            "esc",
            "h1",
            replace(one, group_tiers=esc),
            False,
            i10_m2.group_cols,
            "esc: h1, 1h, escalating threshold (comparison)",
        ),
        M5Rule(
            "esc_2h",
            "h1",
            replace(two, group_tiers=esc),
            False,
            i10_m2.group_cols,
            "esc_2h: esc with a 2h hold (comparison)",
        ),
    ]
    return {r.name: r for r in out}


def variant(name: str, rule: str) -> str:
    return f"{STAGE}:{name}:{rule}"


def m5_variants() -> list[str]:
    return [variant(n, r) for n in NAMES for r in rules()]


# ----------------------------------------------------------------------------- scoring


def score(names: list[str], only: set[str] | None) -> None:
    for name in names:
        for r in rules().values():
            if only and r.name not in only:
                continue
            kw: dict[str, Any] = dict(
                stage=STAGE,
                rule=r.rule,
                variant=variant(name, r.name),
                extra=f"M5 rule {r.name}",
                signal_cols=r.cols,
            )
            sc10.score_variant(name, r.head, workers=WORKERS, **kw)
            sc10.score_seeds(name, r.head, workers=5, **kw)


def sanity() -> None:
    """Seeds 0-2 of the 5-seed runs against the 3-seed runs (pct per seed, every head)."""
    for name, ref in REF3.items():
        if not sc10.npz(name).exists():
            print(f"{name}: not trained yet")
            continue
        with np.load(sc10.npz(name)) as a, np.load(sc10.npz(ref)) as b:
            for key in ("pct_seed", "pct_seed_h2"):
                for j in range(len(b[key])):
                    x, y = a[key][j], b[key][j]
                    ok = np.isfinite(x) & np.isfinite(y)
                    same_nan = bool((np.isfinite(x) == np.isfinite(y)).all())
                    d = np.abs(x[ok] - y[ok])
                    print(
                        f"{name} vs {ref} {key}[{j}]: {int(ok.sum())} rows, max abs diff"
                        f" {d.max():.3g}, rows differing {int((d > 0).sum())},"
                        f" same NaN pattern {same_nan}"
                    )


# ----------------------------------------------------------------------------- the pick


def seed_stats(variants: list[str]) -> pl.DataFrame:
    s = pl.read_csv(sc10.SEEDS_CSV, infer_schema_length=None)
    s = s.filter(pl.col("variant").is_in(variants)).sort("variant", "seed")
    return s.group_by("variant", maintain_order=True).agg(
        seed_means=pl.col("mean_bp"),
        seed_avg=pl.col("mean_bp").mean(),
        seed_spread=pl.col("mean_bp").max() - pl.col("mean_bp").min(),
        seeds_fragile=pl.col("fragile").sum(),
        seeds_c1=pl.col("c1_every_year").sum(),
        n_seeds=pl.len(),
    )


def m5_rows() -> pl.DataFrame:
    """The 18 M5 ledger rows (2022-24), ranked among the eligible runs by the Selection rule."""
    led = s5.latest(pl.read_csv(sc.LEDGER, infer_schema_length=None))
    names = m5_variants()
    order = {v: i for i, v in enumerate(names)}
    el = {r.name: r.eligible for r in rules().values()}
    t = led.filter(pl.col("variant").is_in(names)).with_columns(
        scores=pl.col("variant").str.split(":").list.get(1),
        rule_name=pl.col("variant").str.split(":").list.get(2),
        order=pl.col("variant").replace_strict(order, return_dtype=pl.Int64),
        trades_yr=(pl.col("trades_2022") + pl.col("trades_2023") + pl.col("trades_2024")) / 3,
    )
    t = t.with_columns(
        rule_eligible=pl.col("rule_name").replace_strict(el, return_dtype=pl.Boolean),
    ).with_columns(eligible=pl.col("rule_eligible") & pl.col("qualifies") & ~pl.col("fragile"))
    ranked = (
        t.filter("eligible")
        .sort(["mean_bp", "worst_bp"], descending=True)
        .with_row_index("rank", offset=1)
        .select("variant", "rank")
    )
    t = t.join(ranked, on="variant", how="left").join(seed_stats(names), on="variant", how="left")
    return t.sort("order")


def _s(v: object, nd: int = 0) -> str:
    return s5._s(v, nd=nd)


def pick_table(t: pl.DataFrame) -> list[str]:
    out = [
        "| Run | 2022 | 2023 | 2024 | Mean bp | Worst | 0.6% | Late | Placebo p95 | w/o 5 days"
        " | 95% CI | Trades/yr | Verdict | Fragile (w/o best 22 / 23 / 24) | Seed means"
        " | Seed avg | Spread | Rank |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in t.iter_rows(named=True):
        ex = " / ".join(_s(r[f"exbest_{y}"]) for y in sc.SELECT_YEARS)
        seeds = " / ".join(_s(v) for v in r["seed_means"])
        rank = r["rank"] if r["rank"] is not None else ("-" if r["rule_eligible"] else "n/e")
        out.append(
            f"| {r['scores']}:{r['rule_name']} | {_s(r['bp_2022'])} | {_s(r['bp_2023'])}"
            f" | {_s(r['bp_2024'])} | **{_s(r['mean_bp'], 1)}** | {_s(r['worst_bp'])}"
            f" | {_s(r['cost06_bp'])} | {_s(r['late_bp'])} | {_s(r['placebo_p95_bp'])}"
            f" | {_s(r['ex5_bp'])} | [{_s(r['ci_lo_bp'])}, {_s(r['ci_hi_bp'])}]"
            f" | {r['trades_yr']:.0f} | {'qualifies' if r['qualifies'] else 'fails'}"
            f" | {'**yes**' if r['fragile'] else 'no'} ({ex}) | {seeds}"
            f" | {_s(r['seed_avg'], 1)} | {r['seed_spread']:.1f} | {rank} |"
        )
    return out


def pick_rows() -> tuple[dict[str, Any], dict[str, Any]]:
    """The pick and the runner-up, from the committed table (M5_table.csv)."""
    t = pl.read_csv(PICK_CSV).filter(pl.col("rank").is_not_null()).sort("rank")
    return t.row(0, named=True), t.row(1, named=True)


def pick() -> None:
    t = m5_rows()
    if t.height != len(m5_variants()):
        msg = f"expected {len(m5_variants())} M5 rows, found {t.height}"
        raise SystemExit(msg)
    t.drop("reasons").with_columns(
        seed_means=pl.col("seed_means").cast(pl.List(pl.String)).list.join(" ")
    ).write_csv(PICK_CSV)
    ranked = t.filter(pl.col("rank").is_not_null()).sort("rank")
    top, second = ranked.row(0, named=True), ranked.row(1, named=True)
    band, base_seeds = sc10.noise_band()
    led = s5.latest(pl.read_csv(sc.LEDGER, infer_schema_length=None))
    base = led.filter(pl.col("variant") == sc10.BASE_VARIANT).row(0, named=True)
    bar = base["mean_bp"] + band
    above = t.filter(~pl.col("eligible") & (pl.col("mean_bp") >= top["mean_bp"]))
    el = t.filter("rule_eligible")
    n_q = int(el["qualifies"].sum())
    n_f = int((el["qualifies"] & el["fragile"]).sum())

    def line(label: str, r: dict[str, Any]) -> str:
        seeds = " / ".join(_s(v, 1) for v in r["seed_means"])
        return (
            f"- **{label}: `{r['variant']}`** (rank {r['rank']}): mean {_s(r['mean_bp'], 1)} bp"
            f" per trade 2022-24 ({_s(r['bp_2022'])} / {_s(r['bp_2023'])} /"
            f" {_s(r['bp_2024'])}), worst year {_s(r['worst_bp'], 1)}, 0.6%"
            f" {_s(r['cost06_bp'], 1)}, late {_s(r['late_bp'], 1)}, placebo p95"
            f" {_s(r['placebo_p95_bp'], 1)}, w/o 5 best days {_s(r['ex5_bp'], 1)}, CI"
            f" [{_s(r['ci_lo_bp'])}, {_s(r['ci_hi_bp'])}], {r['trades_yr']:.0f} trades a year;"
            f" per seed {seeds} (avg {_s(r['seed_avg'], 1)}, spread {r['seed_spread']:.1f},"
            f" {r['seeds_fragile']} of {r['n_seeds']} seeds fragile). Beats base3"
            f" (> {bar:+.1f}): {'yes' if r['mean_bp'] > bar else 'no'}."
        )

    lines = [
        "# M5 pick (written before any 2025-26 number of issue #10 was opened)",
        "",
        "Plan: `docs/plans/cnn-improvements.md`, Combination rule. Runner: `i10_m5.py`. Every"
        " run: `CashCapped(5)` equal slots, pct >= 0.98, highest score first, fee 0.1% +"
        " slippage 0.05% + latency 0.05% per side (0.4% round trip), one continuous"
        " production-engine book; stresses keep the latency (0.6% = slippage 0.2% per side;"
        " late = one candle later); 20 within-day placebos. 2022-24 only, seed-mean scores"
        " (5 seeds); per-seed rows use each seed's own percentiles.",
        "",
        "Score sets: `volscale5` (M4 volscale, 5 seeds), `day5` (M3 day, 5 seeds), `volday5`"
        " (volscale + day weights, 5 seeds). Rules: (a) `a_h1_1h` h1 head, 1h hold; (b)"
        " `b_h1_2h` h1, 2h; (c) `c_h2_2h` h2 (2h median) head, 2h; (d) `d_h2_1h` h2, 1h;"
        " comparison rows, not eligible (n/e): `esc` (h1, 1h, escalating threshold per"
        " cascade) and `esc_2h` (esc, 2h hold).",
        "",
        f"{el.height} eligible-rule runs; {n_q} qualify; fragile among those: {n_f};"
        f" {ranked.height} are eligible (qualified and not fragile). Rank = the Selection rule"
        " among them (seed-mean mean bp per trade 2022-24, worst year as the tiebreak)."
        f" Reference: base3 {base['mean_bp']:+.1f} bp (per seed"
        f" {' / '.join(f'{v:+.1f}' for v in base_seeds)}); noise band {band:.1f} bp; the"
        f" beats-the-baseline bar is {bar:+.1f}.",
        "",
        *pick_table(t),
        "",
        "## Pick",
        "",
        line("Pick", top),
        line("Runner-up", second),
    ]
    if ranked.height > 2:
        third = ranked.row(2, named=True)
        lines.append(
            f"- Rank 3: `{third['variant']}`, mean {_s(third['mean_bp'], 2)} against the"
            f" runner-up's {_s(second['mean_bp'], 2)} (worst year {_s(third['worst_bp'], 1)})."
        )
    lines.append(
        "- Why: the plan's rule picks the top-ranked qualified, non-fragile eligible run,"
        " nothing else."
        + (
            " Ranked at or above it by mean bp but not eligible: "
            + ", ".join(
                f"`{r['variant']}` ({_s(r['mean_bp'], 1)} bp; "
                + (
                    "comparison row"
                    if not r["rule_eligible"]
                    else ("fragile" if r["qualifies"] else "fails")
                )
                + ")"
                for r in above.sort("mean_bp", descending=True).iter_rows(named=True)
            )
            + "."
            if above.height
            else " No ineligible run ranks above it."
        )
    )
    sanity_txt = i10_train.WF_I10 / "m5_sanity.txt"
    if sanity_txt.exists():
        lines += [
            "",
            "## Sanity: seeds 0-2 of the 5-seed runs (`i10_m5.py sanity`)",
            "",
            "Per-seed percentiles (h1 and h2 heads) of `volscale5` / `day5` seeds 0-2 against"
            " the 3-seed `volscale` / `day` runs, every scored row 2022-26 (scores only):",
            "",
            "```",
            sanity_txt.read_text().strip(),
            "```",
        ]
    (sc10.OUT / "M5_pick.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ----------------------------------------------------------------------------- trials


@dataclass(frozen=True)
class Trial:
    """A #10 book as its stage scored it: the ledger variant (`seed` None) or one seed, with
    its engine signal frame (the scores themselves are not kept: 200 arrays would not fit)."""

    variant: str
    seed: int | None
    rule: sc.Rule
    sig: pl.DataFrame

    @property
    def key(self) -> str:
        return self.variant if self.seed is None else f"{self.variant}:seed{self.seed}"


def spec_of(v: str) -> tuple[str, str, sc.Rule, FilterFn | None, ColsFn | None]:
    """(npz name, head, rule, filter_fn, cols) of a #10 ledger variant."""
    stage, name, rest = v.split(":", 2)
    if stage in ("M1", "M3", "M4"):
        return name, rest, sc10.rule_of(), None, None
    if stage == "M2":
        r2 = i10_m2.rules()[rest]
        return name, r2.head, r2.rule, r2.filter_fn, r2.cols
    if stage == "M5":
        r5 = rules()[rest]
        return name, r5.head, r5.rule, None, r5.cols
    msg = f"cannot rebuild {v}"
    raise ValueError(msg)


def i10_trials(seeds: bool, end: datetime = sc.UNTIL) -> list[Trial]:
    """Every #10 ledger row (and, with `seeds`, each of its seeds) with signals up to `end`.
    Score arrays are loaded once per (npz, head) and dropped when the npz changes."""
    led = s5.latest(pl.read_csv(sc.LEDGER, infer_schema_length=None))
    specs = [(v, *spec_of(v)) for v in led["variant"]]
    out = []
    for name in dict.fromkeys(sp[1] for sp in specs):
        cache: dict[str, list[tuple[int | None, np.ndarray]]] = {}
        for v, _, head, rule, filter_fn, cols in (sp for sp in specs if sp[1] == name):
            if head not in cache:
                cache[head] = [(None, sc10.scores(name, head))]
                if seeds:
                    cache[head] += list(enumerate(sc10.seed_scores(name, head)))
            for j, s in cache[head]:
                rf = None if filter_fn is None else filter_fn(s)
                out.append(Trial(v, j, rule, sc.signals(s, rule, rf, end=end, cols=cols)))
    return out


def _pool() -> ProcessPoolExecutor:
    return s5._pool()


def books(trials: list[Trial], end: datetime | None) -> dict[str, dict[str, Any]]:
    with _pool() as pool:
        futs: dict[str, Future[dict[str, Any]]] = {
            tr.key: pool.submit(s5.equity_book, tr.sig, tr.rule, end) for tr in trials
        }
        return {k: f.result() for k, f in futs.items()}


# ----------------------------------------------------------------------------- reveal


def sealed(key: str) -> dict[str, Any]:
    d: dict[str, Any] = json.loads((sc.SEALED / f"{key}.json").read_text())
    return d


def _ret(d: dict[str, dict[str, float]], years: tuple[int, ...]) -> float:
    return float(np.prod([1 + d[str(y)]["ret"] for y in years if str(y) in d]) - 1)


def reveal() -> None:
    """2025, 2026 and 2025-26 for every #10 ledger row and every per-seed row."""
    if not PICK_CSV.exists():
        raise SystemExit("run `pick` (and commit and push M5_pick.md) first")
    led = s5.latest(pl.read_csv(sc.LEDGER, infer_schema_length=None))
    seeds = pl.read_csv(sc10.SEEDS_CSV, infer_schema_length=None)
    trials = i10_trials(seeds=True)
    EQUITY.mkdir(parents=True, exist_ok=True)
    got = books(trials, None)
    rows = []
    for tr in trials:
        d = sealed(tr.key)
        lrow = led.filter(pl.col("variant") == tr.variant).row(0, named=True)
        if tr.seed is None:
            ref_hash, mean_2224 = lrow["config_hash"], lrow["mean_bp"]
        else:
            srow = seeds.filter((pl.col("variant") == tr.variant) & (pl.col("seed") == tr.seed))
            ref_hash, mean_2224 = srow["config_hash"][0], srow["mean_bp"][0]
        if d["config_hash"] != ref_hash:
            msg = f"{tr.key}: sealed hash {d['config_hash']} != ledger {ref_hash}"
            raise SystemExit(msg)
        if tr.rule.describe() != lrow["rule"]:
            raise SystemExit(f"{tr.key}: rule differs from the ledger's")
        b = got[tr.key]
        stem = tr.key.replace(":", "_")
        b["equity"].write_parquet(EQUITY / f"{stem}.parquet")
        b["trades"].write_parquet(EQUITY / f"{stem}_trades.parquet")
        for y in sc.YEARS:
            yr = b["yearly"].filter(pl.col("year") == y)
            n = float(yr["trades"][0]) if yr.height else 0.0
            if n != d["years"].get(str(y), {}).get("trades", 0.0):
                raise SystemExit(f"{tr.key} {y}: re-run trades differ from the sealed book")
        crash = b["trades"].filter(pl.col("entry_time").dt.date() == CRASH_DAY)
        row: dict[str, Any] = {
            "variant": tr.variant,
            "seed": -1 if tr.seed is None else tr.seed,
            "mean_bp_2224": mean_2224,
        }
        for label, years in (("2025", (2025,)), ("2026", (2026,)), ("2526", s5.REVEAL_YEARS)):
            n, bp = s5._pooled(d["years"], years)
            row |= {
                f"n_{label}": n,
                f"bp_{label}": bp,
                f"bp06_{label}": s5._pooled(d["cost06"], years)[1],
                f"late_{label}": s5._pooled(d["late"], years)[1],
                f"pnl_{label}": sum(d["years"].get(str(y), {}).get("pnl", 0.0) for y in years),
                f"ret_{label}": _ret(d["years"], years),
            }
        row |= {
            "dd_2526": s5.max_dd(b["equity"], s5.REVEAL_START, sc.UNTIL),
            "oct10_n": crash.height,
            "oct10_bp_sum": float(crash["net_return"].sum()) * 1e4,
        }
        rows.append(row)
    df = pl.DataFrame(rows)
    df.filter(pl.col("seed") < 0).drop("seed").write_csv(sc10.OUT / "M5_reveal.csv")
    df.filter(pl.col("seed") >= 0).write_csv(sc10.OUT / "M5_reveal_seeds.csv")
    print(f"revealed {df.height} books ({df.filter(pl.col('seed') < 0).height} ledger rows)")


# ----------------------------------------------------------------------------- summary


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = pl.Series(x).rank("average").to_numpy()
    ry = pl.Series(y).rank("average").to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def summary() -> str:
    """Markdown tables of the reveal (all #10 rows, seeds, 1h vs 2h, Spearman)."""
    rv = pl.read_csv(sc10.OUT / "M5_reveal.csv")
    rs = pl.read_csv(sc10.OUT / "M5_reveal_seeds.csv")
    top, second = pick_rows()
    out = [
        "| Run | Mean bp 22-24 | 2025 bp (n) | 2026 bp (n) | **2025-26 bp** (n) | 0.6% 25-26"
        " | Late 25-26 | $ P&L 25-26 | Return 2025 / 2026 / 25-26 | Max DD 25-26"
        " | 2025-10-10 bp sum (n) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rv.iter_rows(named=True):
        mark = (
            " (pick)"
            if r["variant"] == top["variant"]
            else " (runner-up)"
            if r["variant"] == second["variant"]
            else ""
        )
        out.append(
            f"| {r['variant']}{mark} | {_s(r['mean_bp_2224'], 1)}"
            f" | {_s(r['bp_2025'])} ({r['n_2025']:.0f}) | {_s(r['bp_2026'])} ({r['n_2026']:.0f})"
            f" | **{_s(r['bp_2526'])}** ({r['n_2526']:.0f}) | {_s(r['bp06_2526'])}"
            f" | {_s(r['late_2526'])} | {_s(r['pnl_2526'])}"
            f" | {r['ret_2025'] * 100:+.0f}% / {r['ret_2026'] * 100:+.0f}% /"
            f" {r['ret_2526'] * 100:+.0f}% | {r['dd_2526'] * 100:.0f}%"
            f" | {_s(r['oct10_bp_sum'])} ({r['oct10_n']}) |"
        )
    # per seed
    out += [
        "",
        "| Run | Seed-mean 25-26 bp | Per-seed 25-26 bp (n) | Seed avg | Seeds > 0"
        " | Per-seed 2025 bp | Per-seed 2025-10-10 bp sum |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rv.iter_rows(named=True):
        g = rs.filter(pl.col("variant") == r["variant"]).sort("seed")
        cells = " / ".join(
            f"{_s(x)} ({n:.0f})" for x, n in zip(g["bp_2526"], g["n_2526"], strict=True)
        )
        out.append(
            f"| {r['variant']} | {_s(r['bp_2526'])} | {cells} | {_s(g['bp_2526'].mean(), 1)}"
            f" | {int((g['bp_2526'] > 0).sum())} of {g.height}"
            f" | {' / '.join(_s(x) for x in g['bp_2025'])}"
            f" | {' / '.join(_s(x) for x in g['oct10_bp_sum'])} |"
        )
    # 1h vs 2h
    m5 = rv.filter(pl.col("variant").str.starts_with("M5:")).with_columns(
        rule=pl.col("variant").str.split(":").list.get(2)
    )

    def avg(rule: str, col: str) -> float:
        return float(m5.filter(pl.col("rule") == rule)[col].mean())  # type: ignore[arg-type]

    out += [
        "",
        "| Pair (mean over the 3 score sets) | 1h: 2025 bp | 2h: 2025 bp | 1h: 2025-26 bp"
        " | 2h: 2025-26 bp |",
        "|---|---|---|---|---|",
    ]
    for label, one, two in (
        ("h1 head: (a) vs (b)", "a_h1_1h", "b_h1_2h"),
        ("h2 head: (d) vs (c)", "d_h2_1h", "c_h2_2h"),
        ("esc vs esc_2h", "esc", "esc_2h"),
    ):
        out.append(
            f"| {label} | {_s(avg(one, 'bp_2025'), 1)} | {_s(avg(two, 'bp_2025'), 1)}"
            f" | {_s(avg(one, 'bp_2526'), 1)} | {_s(avg(two, 'bp_2526'), 1)} |"
        )
    one_h = [(avg("a_h1_1h", c) + avg("d_h2_1h", c)) / 2 for c in ("bp_2025", "bp_2526")]
    two_h = [(avg("b_h1_2h", c) + avg("c_h2_2h", c)) / 2 for c in ("bp_2025", "bp_2526")]
    out.append(
        f"| (a)+(d) 1h vs (b)+(c) 2h | {_s(one_h[0], 1)} | {_s(two_h[0], 1)}"
        f" | {_s(one_h[1], 1)} | {_s(two_h[1], 1)} |"
    )
    ab = (avg("a_h1_1h", "bp_2025") + avg("b_h1_2h", "bp_2025")) / 2
    cd = (avg("c_h2_2h", "bp_2025") + avg("d_h2_1h", "bp_2025")) / 2
    out += ["", f"Average 2025 bp of rules (a) and (b): {ab:+.1f}; of (c) and (d): {cd:+.1f}."]
    # Spearman
    out.append("")
    for label, t in (("all #10 ledger rows", rv), ("the 18 M5 rows", m5)):
        x, y = t["mean_bp_2224"].to_numpy(), t["bp_2025"].to_numpy()
        y26 = t["bp_2526"].to_numpy()
        out.append(
            f"- Spearman(2022-24 mean bp, 2025 bp) over {label} (n = {t.height}):"
            f" {spearman(x, y):+.3f}; with 2025-26 pooled bp: {spearman(x, y26):+.3f}."
        )
    xs, ys = rs["mean_bp_2224"].to_numpy(), rs["bp_2025"].to_numpy()
    out.append(f"- The same over every per-seed row (n = {rs.height}): {spearman(xs, ys):+.3f}.")
    text = "\n".join(out)
    print(text)
    return text


# ----------------------------------------------------------------------------- deflated Sharpe


def robust_trials() -> list[s5.Trial]:
    """The robustness study's 115 ledger trials, rebuilt as S5 did (its ledger, read only)."""
    keep = sc.LEDGER
    sc.LEDGER = ROBUST / "ledger.csv"
    try:
        return s5.ledger_trials()
    finally:
        sc.LEDGER = keep


def dsr() -> None:
    top, second = pick_rows()
    led = s5.latest(pl.read_csv(sc.LEDGER, infer_schema_length=None))
    ref = pl.read_csv(ROBUST / "S5_trials.csv")
    rob = robust_trials()
    mine = i10_trials(seeds=False, end=sc.SELECT_END)
    with _pool() as pool:
        futs: dict[str, Future[dict[str, Any]]] = {
            tr.variant: pool.submit(
                s5.equity_book,
                sc.signals(tr.score, tr.rule, tr.row_filter, end=sc.SELECT_END),
                tr.rule,
                sc.SELECT_END,
            )
            for tr in rob
        }
        rob_books = {k: f.result() for k, f in futs.items()}
    my_books = books(mine, sc.SELECT_END)
    rows = []
    for tr in rob:
        r = s5.daily_returns(rob_books[tr.variant]["equity"], sc.START, sc.SELECT_END)
        st = s5.sharpe_stats(r)
        stored = ref.filter(pl.col("variant") == tr.variant).row(0, named=True)
        rows.append(
            {
                "variant": tr.variant,
                "study": "robustness",
                "matches": abs(st["sr_daily"] - stored["sr_daily"]) < 1e-9,
                "mean_bp": stored["mean_bp"],
                **st,
            }
        )
    for tr in mine:
        b = my_books[tr.variant]
        lrow = led.filter(pl.col("variant") == tr.variant).row(0, named=True)
        check = []
        for yr in sc.SELECT_YEARS:
            got = b["yearly"].filter(pl.col("year") == yr)
            n = float(got["trades"][0]) if got.height else 0.0
            check.append(n == float(lrow[f"trades_{yr}"]))
        r = s5.daily_returns(b["equity"], sc.START, sc.SELECT_END)
        rows.append(
            {
                "variant": tr.variant,
                "study": "issue10",
                "matches": all(check),
                "mean_bp": lrow["mean_bp"],
                **s5.sharpe_stats(r),
            }
        )
    t = pl.DataFrame(rows)
    t.write_csv(sc10.OUT / "M5_trials.csv")
    bad = t.filter(~pl.col("matches"))
    if bad.height:
        print("WARNING: rebuilt trials that differ (robustness: Sharpe vs S5_trials.csv;"
              " issue10: trades vs the ledger):")  # fmt: skip
        print(bad)
    n = t.height
    var_daily = float(t["sr_daily"].var())  # type: ignore[arg-type]
    var_annual = float(t["sr_annual"].var())  # type: ignore[arg-type]
    sr0 = s5.expected_max_sr(var_daily, n)
    out: dict[str, Any] = {
        "n_trials": n,
        "n_robustness": len(rob),
        "n_issue10": len(mine),
        "n_trials_mismatched": bad.height,
        "var_sr_daily": var_daily,
        "var_sr_annual": var_annual,
        "sd_sr_annual": math.sqrt(var_annual),
        "expected_max_sr_annual": sr0 * math.sqrt(s5.ANNUAL),
        "trials_sr_annual_mean": float(t["sr_annual"].mean()),  # type: ignore[arg-type]
        "trials_sr_annual_max": float(t["sr_annual"].max()),  # type: ignore[arg-type]
        "issue10_sr_annual_mean": float(
            t.filter(pl.col("study") == "issue10")["sr_annual"].mean()  # type: ignore[arg-type]
        ),
    }
    for label, row in (("pick", top), ("runner_up", second)):
        p = t.filter(pl.col("variant") == row["variant"]).row(0, named=True)
        out[label] = {
            "variant": row["variant"],
            **{k: p[k] for k in ("sr_daily", "sr_annual", "skew", "kurt", "n_days")},
            "rank_of_sr_among_trials": int((t["sr_annual"] > p["sr_annual"]).sum()) + 1,
            "dsr": s5.psr(p["sr_daily"], sr0, p["n_days"], p["skew"], p["kurt"]),
            "dsr_normal": s5.psr(p["sr_daily"], sr0, p["n_days"], 0.0, 3.0),
            "psr_vs_0": s5.psr(p["sr_daily"], 0.0, p["n_days"], p["skew"], p["kurt"]),
        }
        stem = row["variant"].replace(":", "_")
        eq = pl.read_parquet(EQUITY / f"{stem}.parquet")
        trades = pl.read_parquet(EQUITY / f"{stem}_trades.parquet")
        oos = s5.sharpe_stats(s5.daily_returns(eq, s5.REVEAL_START, sc.UNTIL))
        out[label]["oos_2025_26"] = {
            **oos,
            "psr_vs_0": s5.psr(oos["sr_daily"], 0.0, oos["n_days"], oos["skew"], oos["kurt"]),
            "bp_ci95": s5.day_bootstrap_bp(trades, s5.REVEAL_YEARS),
        }
    (sc10.OUT / "M5_dsr.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def _check_paths() -> None:
    """The robustness ledger is read, never written: the scorecard must point at issue10."""
    if Path(sc.LEDGER).resolve() != (sc10.OUT / "ledger.csv").resolve():
        raise SystemExit(f"scorecard ledger is {sc.LEDGER}, expected results/issue10")


if __name__ == "__main__":
    _check_paths()
    args = sys.argv[1:]
    cmd = args[0] if args else ""
    if cmd == "score":
        only = set(sc10._opt(args, "--only", "").split(",")) - {""}
        names = [a for a in args[1:] if a in NAMES]
        score(names or list(NAMES), only or None)
    elif cmd == "sanity":
        sanity()
    elif cmd == "pick":
        pick()
    elif cmd == "reveal":
        reveal()
    elif cmd == "dsr":
        dsr()
    elif cmd == "summary":
        summary()
    else:
        raise SystemExit(__doc__)
