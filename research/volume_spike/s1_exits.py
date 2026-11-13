"""S1 of the CNN robustness study (docs/plans/cnn-robustness.md): exits and sizing.

    cd research/volume_spike
    export MOONSHOT_UNIVERSE=all
    uv run --group research python s1_exits.py run [path_all ens3] [--only tp5_1h,cap2]
    uv run --group research python s1_exits.py report

`run` scores every rule below on each score set (default path_all and ens3, the S0 scores, no
training) and upserts it into the ledger as `S1:<scores>:<rule>`, with the 2022-24 card in
results/robustness/cards/. The list is fixed (written before any result); `--only` just
restricts a run to some rule names. `report` writes results/robustness/S1.md from the ledger.
Each card uses at most 4 worker processes.
"""

from __future__ import annotations

import sys
from datetime import timedelta

import polars as pl
import scorecard as sc
from s0_baseline import scores

from crypto_moonshot.broker.base import ExitPlan
from crypto_moonshot.portfolio import CashCapped

WORKERS = 4
SCORE_SETS = ("path_all", "ens3")
H = {"30m": timedelta(minutes=30), "1h": timedelta(hours=1), "2h": timedelta(hours=2),
     "4h": timedelta(hours=4)}  # fmt: skip


def _exit(hold: str, **kw: float) -> ExitPlan:
    return ExitPlan(max_hold=H[hold], **kw)


def rules() -> dict[str, sc.Rule]:
    """Rule name -> rule, in table order (group, then name). Anything not listed stays at the
    baseline: q 0.98, 5 slots, EqualSlots, 1h hold, 0.1% + 0.05% per side."""
    r: dict[str, sc.Rule] = {}
    for h in ("30m", "2h", "4h"):
        r[f"hold{h}"] = sc.Rule(exit=_exit(h))
    for tp, h in ((0.03, "1h"), (0.05, "1h"), (0.08, "1h"), (0.05, "2h")):
        r[f"tp{tp * 100:.0f}_{h}"] = sc.Rule(exit=_exit(h, take_profit=tp))
    for tr in (0.03, 0.05):
        r[f"trail{tr * 100:.0f}_1h"] = sc.Rule(exit=_exit("1h", trailing=tr))
    r["stop10_1h"] = sc.Rule(exit=_exit("1h", stop=0.10))
    r["tp5_stop10_1h"] = sc.Rule(exit=_exit("1h", take_profit=0.05, stop=0.10))
    r["cashcap"] = sc.Rule(sizing=CashCapped(5))
    r["weighted"] = sc.Rule(weights=(0.5, 1.5))
    for slots in (3, 5, 10):
        for q in (0.97, 0.98, 0.99):
            if (slots, q) != (5, 0.98):
                r[f"s{slots}_q{q * 100:.0f}"] = sc.Rule(q=q, slots=slots)
    r["cap2"] = sc.Rule(max_per_day=2)
    r["cap3"] = sc.Rule(max_per_day=3)
    return r


def run(names: list[str], only: set[str] | None) -> None:
    cards = sc.OUT / "cards"
    cards.mkdir(parents=True, exist_ok=True)
    for name in names:
        s, source = scores(name)
        for rname, rule in rules().items():
            if only and rname not in only:
                continue
            variant = f"S1:{name}:{rname}"
            card = sc.scorecard(variant, s, rule, stage="S1", source=source, workers=WORKERS)
            text = sc.show(card)
            print(text, flush=True)
            (cards / f"{variant.replace(':', '_')}.txt").write_text(text + "\n")


def _fmt(v: object, plus: bool = True) -> str:
    return "n/a" if v is None else f"{float(v):+.0f}" if plus else f"{float(v):.0f}"  # type: ignore[arg-type]


def report() -> None:
    led = pl.read_csv(sc.LEDGER, infer_schema_length=None)
    order = list(rules())
    out = [
        "# S1: exits and sizing (CNN robustness study)",
        "",
        "Plan: `docs/plans/cnn-robustness.md`. Runner: `s1_exits.py` (re-runnable). Ledger:"
        " `ledger.csv` (`S1:<scores>:<rule>`). 2022-24 only; 2025-26 stays sealed. Every rule"
        " changes one thing from the baseline (q 0.98, 5 slots, `EqualSlots`, highest score"
        " first, 1h hold, 0.1% + 0.05% per side) and the list was fixed before any result."
        " Bp are net per trade, entry-year buckets of one continuous book.",
        "",
        "Columns: bp by year, mean of the three years, mean at a 0.6% round trip, mean entering"
        " one candle late, mean without the 5 best days, trades a year (mean), qualification"
        " verdict, and the fragile flag (some year turns negative without that year's single"
        " best trade; the three numbers are the yearly bp without it).",
        "",
    ]
    for name in SCORE_SETS:
        out += [f"## {name}", ""]
        out += [
            "| Rule | 2022 | 2023 | 2024 | Mean | 0.6% | Late | ex-5 days | Trades/yr |"
            " Verdict | Fragile (ex-best 22/23/24) |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        rows = []
        base = led.filter(pl.col("variant") == name)
        if base.height:
            rows.append(("baseline", base.row(0, named=True)))
        for rname in order:
            m = led.filter(pl.col("variant") == f"S1:{name}:{rname}")
            if m.height:
                rows.append((rname, m.row(0, named=True)))
        for rname, r in rows:
            trades = sum(r[f"trades_{y}"] for y in sc.SELECT_YEARS) / 3
            frag = r.get("fragile")
            ex = "/".join(_fmt(r.get(f"exbest_{y}")) for y in sc.SELECT_YEARS)
            out.append(
                f"| {rname} | {_fmt(r['bp_2022'])} | {_fmt(r['bp_2023'])} | {_fmt(r['bp_2024'])}"
                f" | **{_fmt(r['mean_bp'])}** | {_fmt(r['cost06_bp'])} | {_fmt(r['late_bp'])}"
                f" | {_fmt(r['ex5_bp'])} | {trades:.0f} |"
                f" {'qualifies' if r['qualifies'] else 'fails'} |"
                f" {'yes' if frag else 'no' if frag is not None else 'n/a'} ({ex}) |"
            )
        out.append("")
    conclusion = sc.OUT / "S1_conclusion.md"
    if conclusion.exists():
        out += [conclusion.read_text()]
    (sc.OUT / "S1.md").write_text("\n".join(out))
    print("wrote", sc.OUT / "S1.md")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    only = None
    if "--only" in args:
        i = args.index("--only")
        only = set(args[i + 1].split(","))
        args = args[:i] + args[i + 2 :]
    if args[0] == "run":
        run(args[1:] or list(SCORE_SETS), only)
    elif args[0] == "report":
        report()
    else:
        raise SystemExit(__doc__)
