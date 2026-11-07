"""S5 of the CNN robustness study (docs/plans/cnn-robustness.md): combine and confirm.

    cd research/volume_spike
    export MOONSHOT_UNIVERSE=all
    uv run --group research python s5_confirm.py score [path_all ctx path_all5] [--only a_1h,...]
    uv run --group research python s5_confirm.py pick      # 2022-24 table + pick -> S5_pick.md
    # commit S5_pick.md BEFORE the next step: the commit is the proof that the pick came first
    uv run --group research python s5_confirm.py reveal    # 2025-26 of the 24 runs + S0 baselines
    uv run --group research python s5_confirm.py fine      # 1m fills for the pick and runner-up
    uv run --group research python s5_confirm.py dsr       # deflated Sharpe over the ledger
    uv run --group research python s5_confirm.py report    # results/robustness/S5.md

The plan's Combination rule (written before any S3/S4 result and before any 2025-26 number):
three score sets (path_all, and the top two qualified, non-fragile S3/S4 score sets at the
default rule: S4:ctx and S3:path_all5) x eight rules, each with `CashCapped` equal slots and
fee 0.1% + slippage 0.05% + latency 0.05% per side:

    a_1h               default: pct >= 0.98, 5 slots, 1h hold
    b_btcvol6          BTC volume multiple >= 6 at the signal candle
    c_btc4h-2          BTC 4h return <= -2%
    d_hold2h           2h hold
    e_s10q99           10 slots, pct >= 0.99
    f_btcvol6_hold2h   b + 2h hold
    g_btcvol6_tp5_2h   b + take profit +5%, 2h time exit, no stop
    h_share75_hold2h   spike share >= its 2020-21 p75 (0.1692) + 2h hold

The gates are S2's (`s2_regime.features`). Ledger rows `S5:<scores>:<rule>`; cards in
`results/robustness/cards/S5_*`. The pick is the top-ranked qualified, non-fragile run by the
Selection rule (mean bp per trade 2022-24, worst year as the tiebreak); the runner-up is next.

`reveal` reads the sealed 2025-26 numbers (`results/robustness/sealed/`, checked against the
ledger's config hash) of the 24 runs and the S0 baselines, and re-runs each base book to the data
end for its 2025-26 drawdown and daily equity. `fine` re-runs the pick and the runner-up with exits
filled on 1m candles (take profit `touch`, and `close` when the rule has one); these are a
fill-model check of runs already in the ledger, so they are not new ledger trials. `dsr`
rebuilds every ledger trial's 2022-24 book (no placebos), takes its daily mark-to-market
returns, and computes the deflated Sharpe ratio (Bailey & Lopez de Prado, 2014) of the pick.
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import sys
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from statistics import NormalDist
from typing import Any

import numpy as np
import polars as pl
import scorecard as sc
from common import CACHE

from crypto_moonshot.broker.base import Costs, ExitPlan
from crypto_moonshot.portfolio import CashCapped

WORKERS = 8
SCORE_SETS = ("path_all", "ctx", "path_all5")
S0_BASELINES = ("base", "market", "path_all", "ens3")
LAT_COSTS = Costs(fee=0.001, slippage=0.0005, latency=0.0005)
REVEAL_YEARS = (2025, 2026)
REVEAL_START = datetime(2025, 1, 1, tzinfo=UTC)
EQUITY = CACHE / "s5_equity"  # local, gitignored
PICK_CSV = sc.OUT / "S5_table.csv"
DAY = timedelta(days=1)
EULER_GAMMA = 0.5772156649015329
ANNUAL = 365  # crypto trades every day


# ----------------------------------------------------------------------------- the runs


def score_set(name: str) -> tuple[np.ndarray, str]:
    """A score set's per-row pct array and its source description, loaded as its stage did."""
    if name == "path_all":
        from s0_baseline import scores

        return scores("path_all")
    if name == "ctx":
        import s4_inputs

        return sc.wf_scores("s4_ctx"), s4_inputs.source_of("s4_ctx")
    if name == "path_all5":
        import s3_train

        return sc.wf_scores("path_all5"), s3_train.source_of("path_all5")
    msg = f"unknown score set {name}"
    raise ValueError(msg)


def _rule(*, slots: int = 5, q: float = 0.98, hours: int = 1, tp: float | None = None) -> sc.Rule:
    return sc.Rule(
        q=q,
        slots=slots,
        exit=ExitPlan(max_hold=timedelta(hours=hours), take_profit=tp),
        sizing=CashCapped(slots),
        costs=LAT_COSTS,
    )


def rules() -> dict[str, tuple[str | None, sc.Rule]]:
    """Rule name -> (gate name or None, rule), in the plan's order (a..h)."""
    return {
        "a_1h": (None, _rule()),
        "b_btcvol6": ("btcvol6", _rule()),
        "c_btc4h-2": ("btc4h-2", _rule()),
        "d_hold2h": (None, _rule(hours=2)),
        "e_s10q99": (None, _rule(slots=10, q=0.99)),
        "f_btcvol6_hold2h": ("btcvol6", _rule(hours=2)),
        "g_btcvol6_tp5_2h": ("btcvol6", _rule(hours=2, tp=0.05)),
        "h_share75_hold2h": ("share75", _rule(hours=2)),
    }


def gates() -> dict[str, tuple[np.ndarray, str]]:
    """S2's gates (row masks aligned with seq2_meta) and their descriptions."""
    from s2_regime import FLUSH_RET, FLUSH_VOL, features

    f = features()
    with np.errstate(invalid="ignore"):
        return {
            "btcvol6": (f.btc_vol_mult >= FLUSH_VOL[1], f"btc_vol_mult >= {FLUSH_VOL[1]:g}"),
            "btc4h-2": (f.btc_ret_4h <= FLUSH_RET[0], f"btc_ret_4h <= {FLUSH_RET[0]:.0%}"),
            "share75": (
                f.spike_share >= f.share_p75,
                f"spike_share >= 2020-21 p75 {f.share_p75:.4f}",
            ),
        }


def variant_of(scores: str, rule: str) -> str:
    return f"S5:{scores}:{rule}"


def card_file(variant: str) -> str:
    return variant.replace(":", "_") + ".txt"


def score(names: list[str], only: set[str] | None) -> None:
    g = gates()
    cards = sc.OUT / "cards"
    cards.mkdir(parents=True, exist_ok=True)
    for name in names:
        s, source = score_set(name)
        for rname, (gate, rule) in rules().items():
            if only and rname not in only:
                continue
            mask, desc = (None, "no gate") if gate is None else g[gate]
            variant = variant_of(name, rname)
            card = sc.scorecard(
                variant,
                s,
                rule,
                stage="S5",
                source=f"{source}; S5 {desc}",
                row_filter=mask,
                workers=WORKERS,
                reveal=False,
            )
            text = sc.show(card) + f"\n  gate: {desc}"
            print(text, flush=True)
            (cards / card_file(variant)).write_text(text + "\n")


# ----------------------------------------------------------------------------- the pick


def latest(led: pl.DataFrame) -> pl.DataFrame:
    return led.sort("timestamp").unique("variant", keep="last", maintain_order=True)


def s5_rows() -> pl.DataFrame:
    """The 24 S5 ledger rows (2022-24 only), ranked by the Selection rule."""
    led = latest(pl.read_csv(sc.LEDGER, infer_schema_length=None))
    names = [variant_of(s, r) for s in SCORE_SETS for r in rules()]
    order = {v: i for i, v in enumerate(names)}
    t = led.filter(pl.col("variant").is_in(list(order)))
    t = t.with_columns(
        scores=pl.col("variant").str.split(":").list.get(1),
        rule_name=pl.col("variant").str.split(":").list.get(2),
        order=pl.col("variant").replace_strict(order, return_dtype=pl.Int64),
        trades_yr=(pl.col("trades_2022") + pl.col("trades_2023") + pl.col("trades_2024")) / 3,
        eligible=pl.col("qualifies") & ~pl.col("fragile"),
    )
    ranked = (
        t.filter("eligible")
        .sort(["mean_bp", "worst_bp"], descending=True)
        .with_row_index("rank", offset=1)
        .select("variant", "rank")
    )
    return t.join(ranked, on="variant", how="left").sort("order")


def _s(v: object, plus: bool = True, nd: int = 0) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{float(v):+.{nd}f}" if plus else f"{float(v):.{nd}f}"  # type: ignore[arg-type]


def table_2224(t: pl.DataFrame) -> list[str]:
    out = [
        "| Run | 2022 | 2023 | 2024 | Mean bp | Worst | 0.6% | Late | w/o 5 days | 95% CI"
        " | Trades/yr | DD 22 / 23 / 24 | Verdict | Fragile (w/o best 22 / 23 / 24) | Rank |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in t.iter_rows(named=True):
        ex = " / ".join(_s(r[f"exbest_{y}"]) for y in sc.SELECT_YEARS)
        dd = " / ".join(f"{r[f'dd_{y}'] * 100:.0f}%" for y in sc.SELECT_YEARS)
        out.append(
            f"| {r['scores']}:{r['rule_name']} | {_s(r['bp_2022'])} | {_s(r['bp_2023'])}"
            f" | {_s(r['bp_2024'])} | **{_s(r['mean_bp'])}** | {_s(r['worst_bp'])}"
            f" | {_s(r['cost06_bp'])} | {_s(r['late_bp'])} | {_s(r['ex5_bp'])}"
            f" | [{_s(r['ci_lo_bp'])}, {_s(r['ci_hi_bp'])}] | {r['trades_yr']:.0f} | {dd}"
            f" | {'qualifies' if r['qualifies'] else 'fails'}"
            f" | {'**yes**' if r['fragile'] else 'no'} ({ex})"
            f" | {r['rank'] if r['rank'] is not None else '-'} |"
        )
    return out


def pick_rows() -> tuple[dict[str, Any], dict[str, Any]]:
    """The pick and the runner-up, from the committed table (S5_table.csv)."""
    t = pl.read_csv(PICK_CSV).filter(pl.col("rank").is_not_null()).sort("rank")
    return t.row(0, named=True), t.row(1, named=True)


def pick() -> None:
    t = s5_rows()
    if t.height != len(SCORE_SETS) * len(rules()):
        msg = f"expected 24 S5 rows, found {t.height}"
        raise SystemExit(msg)
    t.drop("reasons").write_csv(PICK_CSV)
    ranked = t.filter(pl.col("rank").is_not_null()).sort("rank")
    top, second = ranked.row(0, named=True), ranked.row(1, named=True)
    above = t.filter(
        ~pl.col("eligible")
        & (
            (pl.col("mean_bp") > top["mean_bp"])
            | ((pl.col("mean_bp") == top["mean_bp"]) & (pl.col("worst_bp") > top["worst_bp"]))
        )
    )
    ref = t.filter(pl.col("variant") == variant_of("path_all", "a_1h")).row(0, named=True)
    third = ranked.row(2, named=True)
    n_q = int(t["qualifies"].sum())
    n_f = int((t["qualifies"] & t["fragile"]).sum())
    lines = [
        "# S5 pick (written before any 2025-26 number of S5 was opened)",
        "",
        "Plan: `docs/plans/cnn-robustness.md`, S5 Combination rule. Runner: `s5_confirm.py`."
        " Every run: `CashCapped` equal slots, fee 0.1% + slippage 0.05% + latency 0.05% per"
        " side, continuous production-engine book; stresses keep the latency (0.6% = slippage"
        " 0.2% per side + latency 0.05%; late = one candle later). 2022-24 only.",
        "",
        f"{t.height} runs; {n_q} qualify; {n_f} of those are fragile; {ranked.height} are"
        " eligible (qualified and not fragile). Rank = the Selection rule ranking among the"
        " eligible runs (mean bp per trade 2022-24, worst year as the tiebreak).",
        "",
        *table_2224(t),
        "",
        "## Pick",
        "",
        f"- **Pick: `{top['variant']}`** (rank 1): mean {_s(top['mean_bp'], nd=1)} bp per trade"
        f" 2022-24 ({_s(top['bp_2022'])} / {_s(top['bp_2023'])} / {_s(top['bp_2024'])}), worst"
        f" year {_s(top['worst_bp'], nd=1)}, 0.6% {_s(top['cost06_bp'], nd=1)}, late"
        f" {_s(top['late_bp'], nd=1)}, w/o 5 best days {_s(top['ex5_bp'], nd=1)}, CI"
        f" [{_s(top['ci_lo_bp'])}, {_s(top['ci_hi_bp'])}], {top['trades_yr']:.0f} trades a year.",
        f"- **Runner-up: `{second['variant']}`** (rank 2): mean {_s(second['mean_bp'], nd=1)}"
        f" ({_s(second['bp_2022'])} / {_s(second['bp_2023'])} / {_s(second['bp_2024'])}),"
        f" worst year {_s(second['worst_bp'], nd=1)}, 0.6% {_s(second['cost06_bp'], nd=1)},"
        f" late {_s(second['late_bp'], nd=1)}, w/o 5 best days {_s(second['ex5_bp'], nd=1)},"
        f" {second['trades_yr']:.0f} trades a year.",
        f"- Rank 3: `{third['variant']}`, mean {_s(third['mean_bp'], nd=2)} against the"
        f" runner-up's {_s(second['mean_bp'], nd=2)} (worst year {_s(third['worst_bp'], nd=1)}):"
        " the rule ranks by the mean first, so it is third however small the gap.",
        f"- Reference (`{ref['variant']}`, kept as a comparison by the rule): mean"
        f" {_s(ref['mean_bp'], nd=1)}, worst year {_s(ref['worst_bp'], nd=1)}.",
        "- Why: the plan's rule picks the top-ranked qualified, non-fragile run, nothing else."
        + (
            " Ranked higher by mean bp but not eligible: "
            + ", ".join(
                f"`{r['variant']}` ({_s(r['mean_bp'])} bp; "
                + ("fragile" if r["qualifies"] else "fails")
                + ")"
                for r in above.sort("mean_bp", descending=True).iter_rows(named=True)
            )
            + "."
            if above.height
            else " No ineligible run ranks above it."
        ),
    ]
    (sc.OUT / "S5_pick.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ----------------------------------------------------------------------------- books


@dataclass(frozen=True)
class Trial:
    variant: str
    score: np.ndarray
    rule: sc.Rule
    row_filter: np.ndarray | None


def equity_book(sig: pl.DataFrame, rule: sc.Rule, end: datetime | None) -> dict[str, Any]:
    """One engine book (no placebos): its yearly table, trades and mark-to-market equity."""
    res = sc.book(sig, rule, end)
    return {
        "yearly": res.yearly,
        "trades": res.trades.select("entry_time", "exit_time", "symbol", "net_return", "pnl"),
        "equity": res.equity.select("time", "equity"),
    }


def daily_returns(equity: pl.DataFrame, start: datetime, end: datetime) -> np.ndarray:
    """Daily returns of the mark-to-market equity on the UTC midnights in [start, end] (the
    initial capital before the curve starts; the last point carries forward after it ends)."""
    t = equity["time"].dt.epoch("us").to_numpy().astype(np.int64)
    e = equity["equity"].to_numpy().astype(np.float64)
    grid = np.arange(sc._us(start), sc._us(end) + 1, 86_400_000_000, dtype=np.int64)
    i = np.searchsorted(t, grid, side="right") - 1
    level = np.where(i >= 0, e[np.maximum(i, 0)], 10_000.0)
    return np.asarray(level[1:] / level[:-1] - 1, dtype=np.float64)


def max_dd(equity: pl.DataFrame, start: datetime, end: datetime) -> float:
    """Max drawdown of the curve in (start, end], the equity at `start` counting as a peak."""
    t = equity["time"].dt.epoch("us").to_numpy().astype(np.int64)
    e = equity["equity"].to_numpy().astype(np.float64)
    a, b = sc._us(start), sc._us(end)
    i0 = int(np.searchsorted(t, a, side="right")) - 1
    seg = np.concatenate(([10_000.0 if i0 < 0 else float(e[i0])], e[(t > a) & (t <= b)]))
    return float((seg / np.maximum.accumulate(seg) - 1).min())


def _pool() -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=WORKERS, mp_context=mp.get_context("spawn"))


def s5_trials() -> list[Trial]:
    g = gates()
    out = []
    for name in SCORE_SETS:
        s, _ = score_set(name)
        for rname, (gate, rule) in rules().items():
            out.append(
                Trial(variant_of(name, rname), s, rule, None if gate is None else g[gate][0])
            )
    return out


def s0_trials() -> list[Trial]:
    from s0_baseline import scores

    return [Trial(n, scores(n)[0], sc.Rule(), None) for n in S0_BASELINES]


# ----------------------------------------------------------------------------- reveal


def sealed(variant: str) -> dict[str, Any]:
    d: dict[str, Any] = json.loads((sc.SEALED / f"{variant}.json").read_text())
    return d


def _pooled(d: dict[str, dict[str, float]], years: tuple[int, ...]) -> tuple[float, float]:
    n = sum(d[str(y)]["trades"] for y in years if str(y) in d)
    s = sum(d[str(y)]["trades"] * d[str(y)]["bp"] for y in years if str(y) in d)
    return n, (s / n if n else float("nan"))


def reveal() -> None:
    """2025, 2026 and 2025-26 for the 24 S5 runs and the S0 baselines (all rows, no hiding)."""
    if not PICK_CSV.exists():
        raise SystemExit("run `pick` (and commit S5_pick.md) first")
    led = latest(pl.read_csv(sc.LEDGER, infer_schema_length=None))
    trials = s0_trials() + s5_trials()
    EQUITY.mkdir(parents=True, exist_ok=True)
    with _pool() as pool:
        futs = {
            tr.variant: pool.submit(
                equity_book, sc.signals(tr.score, tr.rule, tr.row_filter), tr.rule, None
            )
            for tr in trials
        }
        books = {k: f.result() for k, f in futs.items()}
    rows = []
    for tr in trials:
        d = sealed(tr.variant)
        lrow = led.filter(pl.col("variant") == tr.variant).row(0, named=True)
        if d["config_hash"] != lrow["config_hash"]:
            msg = f"{tr.variant}: sealed hash {d['config_hash']} != ledger {lrow['config_hash']}"
            raise SystemExit(msg)
        b = books[tr.variant]
        b["equity"].write_parquet(EQUITY / f"{tr.variant.replace(':', '_')}.parquet")
        b["trades"].write_parquet(EQUITY / f"{tr.variant.replace(':', '_')}_trades.parquet")
        # the re-run book must equal the sealed one
        for y in (*sc.SELECT_YEARS, *REVEAL_YEARS):
            yr = b["yearly"].filter(pl.col("year") == y)
            if yr.height and abs(float(yr["trades"][0]) - d["years"][str(y)]["trades"]) > 0:
                msg = f"{tr.variant} {y}: re-run trades differ from the sealed book"
                raise SystemExit(msg)
        for label, years in (("2025", (2025,)), ("2026", (2026,)), ("2025-26", REVEAL_YEARS)):
            n, bp = _pooled(d["years"], years)
            _, bp06 = _pooled(d["cost06"], years)
            _, bpl = _pooled(d["late"], years)
            pnl = sum(d["years"][str(y)]["pnl"] for y in years if str(y) in d["years"])
            ret = float(np.prod([1 + d["years"][str(y)]["ret"] for y in years]) - 1)
            a = datetime(years[0], 1, 1, tzinfo=UTC)
            z = min(datetime(years[-1] + 1, 1, 1, tzinfo=UTC), sc.UNTIL)
            rows.append(
                {
                    "variant": tr.variant,
                    "period": label,
                    "trades": n,
                    "bp": bp,
                    "bp_06": bp06,
                    "bp_late": bpl,
                    "pnl": pnl,
                    "ret": ret,
                    "dd": max_dd(b["equity"], a, z),
                    "mean_bp_2224": lrow["mean_bp"],
                }
            )
    df = pl.DataFrame(rows)
    df.write_csv(sc.OUT / "S5_reveal.csv")
    with pl.Config(tbl_rows=200, tbl_cols=20, tbl_width_chars=250, float_precision=1):
        print(df)


# ----------------------------------------------------------------------------- 1m fills


def fine() -> None:
    """The pick and the runner-up with exits filled on 1m candles (all years revealed)."""
    top, second = pick_rows()
    by = {tr.variant: tr for tr in s5_trials()}
    rows = []
    for r in (top, second):
        tr = by[r["variant"]]
        fills = ["touch"] + (["close"] if tr.rule.exit.take_profit is not None else [])
        gate, _ = rules()[r["rule_name"]]
        desc = "no gate" if gate is None else gates()[gate][1]
        for fill in fills:
            rule = replace(tr.rule, use_fine=True, exit=replace(tr.rule.exit, tp_fill=fill))
            variant = f"{tr.variant}:1m-{fill}"
            card = sc.scorecard(
                variant,
                tr.score,
                rule,
                stage="S5",
                source=f"{score_set(r['scores'])[1]}; S5 {desc}",
                row_filter=tr.row_filter,
                workers=WORKERS,
                reveal=True,
                ledger=False,
            )
            text = sc.show(card) + f"\n  gate: {desc}"
            print(text, flush=True)
            (sc.OUT / "cards" / card_file(variant)).write_text(text + "\n")
            for y, s in card.years.items():
                rows.append(
                    {
                        "variant": tr.variant,
                        "fills": f"1m, tp {fill}",
                        "year": y,
                        "trades": s["trades"],
                        "bp": s["bp"],
                        "bp_06": card.cost06[y]["bp"],
                        "bp_late": card.late[y]["bp"],
                        "ret": s["ret"],
                        "dd": s["dd"],
                        "mean_bp_2224": card.mean_bp,
                        "qualifies": card.verdict.qualifies,
                        "fragile": card.fragile,
                    }
                )
        d = sealed(tr.variant)
        for y in sc.YEARS:
            s = d["years"][str(y)]
            rows.append(
                {
                    "variant": tr.variant,
                    "fills": "5m (engine default)",
                    "year": y,
                    "trades": s["trades"],
                    "bp": s["bp"],
                    "bp_06": d["cost06"][str(y)]["bp"],
                    "bp_late": d["late"][str(y)]["bp"],
                    "ret": s["ret"],
                    "dd": s["dd"],
                    "mean_bp_2224": r["mean_bp"],
                    "qualifies": r["qualifies"],
                    "fragile": r["fragile"],
                }
            )
    df = pl.DataFrame(rows)
    df.write_csv(sc.OUT / "S5_fine.csv")
    with pl.Config(tbl_rows=100, tbl_cols=20, tbl_width_chars=250, float_precision=1):
        print(df)


# ----------------------------------------------------------------------------- deflated Sharpe


def ledger_trials() -> list[Trial]:
    """Every ledger row rebuilt as (scores, rule, row filter), as its stage scored it."""
    import s1_exits
    import s3_train
    import s4_inputs
    from s0_baseline import scores
    from s2_regime import features, rule_set

    led = latest(pl.read_csv(sc.LEDGER, infer_schema_length=None))
    f = features()
    s2_cache: dict[str, dict[str, tuple[np.ndarray | None, sc.Rule, str]]] = {}
    s5 = {tr.variant: tr for tr in s5_trials()}
    out = []
    for v in led["variant"]:
        parts = v.split(":")
        if parts[0] == "S5":
            out.append(s5[v])
        elif len(parts) == 1:
            out.append(Trial(v, scores(v)[0], sc.Rule(), None))
        elif parts[0] == "S1":
            out.append(Trial(v, scores(parts[1])[0], s1_exits.rules()[parts[2]], None))
        elif parts[0] == "S2":
            s = scores(parts[1])[0]
            if parts[1] not in s2_cache:
                s2_cache[parts[1]] = rule_set(f, s)
            mask, rule, _ = s2_cache[parts[1]][parts[2]]
            out.append(Trial(v, s, rule, mask))
        elif parts[0] == "S3":
            s, rule = s3_train.scores_rule(parts[1])
            out.append(Trial(v, s, rule, None))
        elif parts[0] == "S4":
            name = f"s4_{parts[1]}"
            assert name in s4_inputs.S4_CNN
            out.append(Trial(v, sc.wf_scores(name), sc.Rule(), None))
        else:
            msg = f"cannot rebuild ledger trial {v}"
            raise ValueError(msg)
    return out


def sharpe_stats(r: np.ndarray) -> dict[str, float]:
    sd = float(r.std(ddof=1))
    sr = float(r.mean()) / sd if sd > 0 else 0.0
    z = (r - r.mean()) / (r.std(ddof=0) if r.std(ddof=0) > 0 else 1.0)
    return {
        "sr_daily": sr,
        "sr_annual": sr * math.sqrt(ANNUAL),
        "skew": float((z**3).mean()),
        "kurt": float((z**4).mean()),  # not excess: a normal has 3
        "n_days": float(len(r)),
    }


def expected_max_sr(var: float, n: int) -> float:
    """E[max SR] of n trials with no edge and SR variance `var` (Bailey & Lopez de Prado)."""
    nd = NormalDist()
    return math.sqrt(var) * (
        (1 - EULER_GAMMA) * nd.inv_cdf(1 - 1 / n) + EULER_GAMMA * nd.inv_cdf(1 - 1 / (n * math.e))
    )


def psr(sr: float, sr0: float, t: float, skew: float, kurt: float) -> float:
    """Probabilistic Sharpe ratio P(true SR > sr0), all SRs per observation."""
    den = math.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2)
    return NormalDist().cdf((sr - sr0) * math.sqrt(t - 1) / den)


def day_bootstrap_bp(trades: pl.DataFrame, years: tuple[int, ...], seed: int = 0) -> list[float]:
    """95% CI of the pooled bp per trade over `years`, resampling entry days within each year."""
    d = (
        trades.filter(pl.col("entry_time").dt.year().is_in(years))
        .group_by(day=pl.col("entry_time").dt.date())
        .agg(year=pl.col("entry_time").dt.year().first(), s=pl.col("net_return").sum(), n=pl.len())
        .sort("day")  # group_by order is not stable; the draws must be reproducible
    )
    rng = np.random.default_rng(seed)
    s_tot = np.zeros(sc.BOOT_DRAWS)
    n_tot = np.zeros(sc.BOOT_DRAWS)
    for y in years:
        dy = d.filter(pl.col("year") == y)
        s, n = dy["s"].to_numpy(), dy["n"].to_numpy().astype(np.float64)
        idx = rng.integers(0, len(s), (sc.BOOT_DRAWS, len(s)))
        s_tot += s[idx].sum(1)
        n_tot += n[idx].sum(1)
    lo, hi = np.percentile(s_tot / n_tot * 1e4, [2.5, 97.5])
    return [float(lo), float(hi)]


def dsr() -> None:
    top, second = pick_rows()
    led = latest(pl.read_csv(sc.LEDGER, infer_schema_length=None))
    trials = ledger_trials()
    with _pool() as pool:
        futs = {
            tr.variant: pool.submit(
                equity_book,
                sc.signals(tr.score, tr.rule, tr.row_filter, end=sc.SELECT_END),
                tr.rule,
                sc.SELECT_END,
            )
            for tr in trials
        }
        books = {k: f.result() for k, f in futs.items()}
    rows = []
    for tr in trials:
        b = books[tr.variant]
        lrow = led.filter(pl.col("variant") == tr.variant).row(0, named=True)
        y = b["yearly"]
        check = []
        for yr in sc.SELECT_YEARS:
            got = y.filter(pl.col("year") == yr)
            n = float(got["trades"][0]) if got.height else 0.0
            check.append(n == float(lrow[f"trades_{yr}"]))
        r = daily_returns(b["equity"], sc.START, sc.SELECT_END)
        rows.append(
            {
                "variant": tr.variant,
                "stage": lrow["stage"],
                "rule_matches": tr.rule.describe() == lrow["rule"],
                "trades_match": all(check),
                "mean_bp": lrow["mean_bp"],
                **sharpe_stats(r),
            }
        )
    t = pl.DataFrame(rows)
    t.write_csv(sc.OUT / "S5_trials.csv")
    bad = t.filter(~pl.col("rule_matches") | ~pl.col("trades_match"))
    if bad.height:
        print("WARNING: rebuilt trials that differ from the ledger:")
        print(bad)
    n = t.height
    var_daily = float(t["sr_daily"].var())  # type: ignore[arg-type]
    var_annual = float(t["sr_annual"].var())  # type: ignore[arg-type]
    sr0 = expected_max_sr(var_daily, n)
    out: dict[str, Any] = {
        "n_trials": n,
        "n_trials_mismatched": bad.height,
        "var_sr_daily": var_daily,
        "var_sr_annual": var_annual,
        "sd_sr_annual": math.sqrt(var_annual),
        "expected_max_sr_annual": sr0 * math.sqrt(ANNUAL),
        "trials_sr_annual_mean": float(t["sr_annual"].mean()),  # type: ignore[arg-type]
        "trials_sr_annual_max": float(t["sr_annual"].max()),  # type: ignore[arg-type]
    }
    for label, row in (("pick", top), ("runner_up", second)):
        p = t.filter(pl.col("variant") == row["variant"]).row(0, named=True)
        out[label] = {
            "variant": row["variant"],
            **{k: p[k] for k in ("sr_daily", "sr_annual", "skew", "kurt", "n_days")},
            "rank_of_sr_among_trials": int((t["sr_annual"] > p["sr_annual"]).sum()) + 1,
            "dsr": psr(p["sr_daily"], sr0, p["n_days"], p["skew"], p["kurt"]),
            # sensitivity: the same with normal returns (skew 0, kurtosis 3)
            "dsr_normal": psr(p["sr_daily"], sr0, p["n_days"], 0.0, 3.0),
            "psr_vs_0": psr(p["sr_daily"], 0.0, p["n_days"], p["skew"], p["kurt"]),
        }
        # out of sample: the full book (2025-26 entries and equity), as `reveal` saved it
        eq = pl.read_parquet(EQUITY / f"{row['variant'].replace(':', '_')}.parquet")
        tr_ = pl.read_parquet(EQUITY / f"{row['variant'].replace(':', '_')}_trades.parquet")
        oos = sharpe_stats(daily_returns(eq, REVEAL_START, sc.UNTIL))
        out[label]["oos_2025_26"] = {
            **oos,
            "psr_vs_0": psr(oos["sr_daily"], 0.0, oos["n_days"], oos["skew"], oos["kurt"]),
            "bp_ci95": day_bootstrap_bp(tr_, REVEAL_YEARS),
        }
    (sc.OUT / "S5_dsr.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


# ----------------------------------------------------------------------------- report


def report() -> None:
    """results/robustness/S5.md from S5_table.csv, S5_reveal.csv, S5_fine.csv, S5_dsr.json."""
    t = pl.read_csv(PICK_CSV)
    top, second = pick_rows()
    rv = pl.read_csv(sc.OUT / "S5_reveal.csv")
    fn = pl.read_csv(sc.OUT / "S5_fine.csv")
    ds = json.loads((sc.OUT / "S5_dsr.json").read_text())
    led = latest(pl.read_csv(sc.LEDGER, infer_schema_length=None))
    lines = [
        "# S5: combine and confirm (CNN robustness study)",
        "",
        "Plan: `docs/plans/cnn-robustness.md` (S5 Combination rule). Runner: `s5_confirm.py`."
        " Ledger rows `S5:<scores>:<rule>`; cards `cards/S5_*`; tables `S5_table.csv`"
        " (2022-24), `S5_reveal.csv`, `S5_fine.csv`, `S5_trials.csv`, `S5_dsr.json`. The pick"
        " was committed in `S5_pick.md` before any S5 2025-26 number was opened.",
        "",
        "Every run: `CashCapped` equal slots, fee 0.1% + slippage 0.05% + latency 0.05% per"
        " side (0.4% round trip), one continuous production-engine book from 2022-01-01 to"
        " 2026-11-02. 0.6% = slippage 0.2% per side with the latency kept (0.7% round trip);"
        " late = one candle later. Gates as S2 (`btc_vol_mult >= 6`, `btc_ret_4h <= -2%`,"
        " `spike_share >= 0.1692`, the 2020-21 p75).",
        "",
        "## 2022-24 (selection)",
        "",
        *table_2224(t),
        "",
        f"Pick: **`{top['variant']}`**; runner-up: **`{second['variant']}`**.",
        "",
        "## 2025-26 reveal (all 24 runs and the S0 baselines)",
        "",
        "bp per trade, net. 2025-26 bp is pooled over all trades of both years. Return and"
        " drawdown are of the continuous book (2026 to 2026-11-02); the 2025-26 return compounds"
        " the two years and its drawdown resets the peak on 2025-01-01. S0 baselines are at their"
        " S0 rule (`EqualSlots(5)`, no latency).",
        "",
        "| Run | Mean bp 22-24 | 2025 bp (n) | 2026 bp (n) | **2025-26 bp** (n) | 0.6% 25-26"
        " | Late 25-26 | $ P&L 25-26 | Return 2025 / 2026 / 25-26 | Max DD 25-26 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    variants = [*S0_BASELINES, *(variant_of(s, r) for s in SCORE_SETS for r in rules())]
    for v in variants:
        g = {r["period"]: r for r in rv.filter(pl.col("variant") == v).iter_rows(named=True)}
        a, b, c = g["2025"], g["2026"], g["2025-26"]
        mark = (
            " (pick)" if v == top["variant"] else " (runner-up)" if v == second["variant"] else ""
        )
        name = v.removeprefix("S5:") + mark
        if v == top["variant"]:
            name = f"**{name}**"
        lines.append(
            f"| {name} | {_s(c['mean_bp_2224'])} | {_s(a['bp'])} ({a['trades']:.0f})"
            f" | {_s(b['bp'])} ({b['trades']:.0f}) | **{_s(c['bp'])}** ({c['trades']:.0f})"
            f" | {_s(c['bp_06'])} | {_s(c['bp_late'])} | {_s(c['pnl'])}"
            f" | {a['ret'] * 100:+.0f}% / {b['ret'] * 100:+.0f}% / {c['ret'] * 100:+.0f}%"
            f" | {c['dd'] * 100:.0f}% |"
        )
    lines += [
        "",
        "## 1m fills (pick and runner-up)",
        "",
        "The same books with exits filled on 1m candles where they cover a 5m candle"
        " (`use_fine`); take profit `touch` (high reaches the level) and, for a rule with a take"
        " profit, `close` (a 1m close at or above it). Not ledger trials (a fill-model check).",
        "",
        "| Run | Fills | 2022 | 2023 | 2024 | 2025 | 2026 | Mean 22-24 | 0.6% 22-24 | Late 22-24"
        " | 0.6% 25 / 26 | Late 25 / 26 | Verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (v, fill), grp in fn.group_by(["variant", "fills"], maintain_order=True):
        by = {r["year"]: r for r in grp.iter_rows(named=True)}
        m06 = float(np.mean([by[y]["bp_06"] for y in sc.SELECT_YEARS]))
        mlt = float(np.mean([by[y]["bp_late"] for y in sc.SELECT_YEARS]))
        r0 = by[2022]
        verdict = ("qualifies" if r0["qualifies"] else "fails") + (
            ", fragile" if r0["fragile"] else ""
        )
        lines.append(
            f"| {str(v).removeprefix('S5:')} | {fill} | "
            + " | ".join(f"{_s(by[y]['bp'])} ({by[y]['trades']:.0f})" for y in sc.YEARS)
            + f" | {_s(r0['mean_bp_2224'])} | {_s(m06)} | {_s(mlt)}"
            f" | {_s(by[2025]['bp_06'])} / {_s(by[2026]['bp_06'])}"
            f" | {_s(by[2025]['bp_late'])} / {_s(by[2026]['bp_late'])} | {verdict} |"
        )
    p = ds["pick"]
    o = p["oos_2025_26"]
    lines += [
        "",
        "## Deflated Sharpe ratio (pick)",
        "",
        f"- Trials N = {ds['n_trials']} (every ledger row, S0-S5; {ds['n_trials_mismatched']}"
        " rebuilt books differ from their ledger row). Each trial's 2022-24 book was rebuilt"
        " (no placebos) and its daily mark-to-market returns taken on the UTC midnights"
        " 2022-01-01..2025-01-01 (1,096 days; annualised with sqrt(365)).",
        f"- Trials' annualised Sharpe: mean {ds['trials_sr_annual_mean']:.2f}, max"
        f" {ds['trials_sr_annual_max']:.2f}, variance {ds['var_sr_annual']:.3f}"
        f" (sd {ds['sd_sr_annual']:.2f}).",
        f"- Expected maximum annualised Sharpe of {ds['n_trials']} trials with no edge:"
        f" {ds['expected_max_sr_annual']:.2f}.",
        f"- Pick `{p['variant']}`: 2022-24 annualised Sharpe {p['sr_annual']:.2f} (rank"
        f" {p['rank_of_sr_among_trials']} of {ds['n_trials']} by Sharpe), skew {p['skew']:.2f},"
        f" kurtosis {p['kurt']:.1f}.",
        f"- **Deflated Sharpe ratio: P(true Sharpe > {ds['expected_max_sr_annual']:.2f}) ="
        f" {p['dsr']:.3f}** (PSR against 0: {p['psr_vs_0']:.3f}). With normal returns (skew 0,"
        f" kurtosis 3) instead of the pick's own: {p['dsr_normal']:.3f}. The large positive skew"
        " comes from a few crash days and raises the DSR, so the normal figure is the safer one.",
        f"- Out of sample 2025-26 (2025-01-01..2026-11-02, {o['n_days']:.0f} days): annualised"
        f" Sharpe {o['sr_annual']:.2f} (PSR against 0: {o['psr_vs_0']:.3f}); bp per trade"
        f" day-bootstrap 95% CI [{o['bp_ci95'][0]:+.0f}, {o['bp_ci95'][1]:+.0f}].",
    ]
    r2 = ds["runner_up"]
    lines.append(
        f"- Runner-up `{r2['variant']}`: 2022-24 Sharpe {r2['sr_annual']:.2f}, DSR"
        f" {r2['dsr']:.3f} (normal returns {r2['dsr_normal']:.3f}); 2025-26 Sharpe"
        f" {r2['oos_2025_26']['sr_annual']:.2f}, bp CI"
        f" [{r2['oos_2025_26']['bp_ci95'][0]:+.0f}, {r2['oos_2025_26']['bp_ci95'][1]:+.0f}]."
    )
    del led
    conclusion = sc.OUT / "S5_conclusion.md"
    if conclusion.exists():
        lines += ["", conclusion.read_text().rstrip()]
    (sc.OUT / "S5.md").write_text("\n".join(lines) + "\n")
    print("wrote", sc.OUT / "S5.md")


CMDS: dict[str, Callable[[], None]] = {
    "pick": pick,
    "reveal": reveal,
    "fine": fine,
    "dsr": dsr,
    "report": report,
}

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    if args[0] == "score":
        only = None
        if "--only" in args:
            i = args.index("--only")
            only = set(args[i + 1].split(","))
            args = args[:i] + args[i + 2 :]
        score(args[1:] or list(SCORE_SETS), only)
    elif args[0] in CMDS:
        CMDS[args[0]]()
    else:
        raise SystemExit(__doc__)
