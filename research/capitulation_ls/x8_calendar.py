# ruff: noqa: E501
"""X8 family A: calendar effects on the existing sleeves L, C, M, F (in-sample exploration).

Do entries differ by weekday, weekend, hour of day (UTC, 4h buckets), turn of month, month end,
or the phase of the 8h funding cycle (settlements at 00/08/16 UTC)? Outcome = diag o (net on the
stake, clipped). For every sleeve x bucket: mean o and the bucket-minus-rest difference with a
day-clustered bootstrap 95% CI (calendar days resampled; the sleeves rest on few days), separately
in 2022-24 and 2025-26. An effect needs the difference CI to exclude 0 with the same sign in both.
Then the best exclusion per sleeve chosen on 2022-24 is read on 2025-26 and in the 10-slot books.

Run: uv run --group research python x8_calendar.py  -> results/explore/calendar.md
"""

from __future__ import annotations

import zlib

import numpy as np
import polars as pl
from diag import table
from x8_lib import OUT, as_book, boot_diff, card, run_book, sleeves

VERDICT = """## Verdict: no calendar effect; nothing to adopt

* 88 sleeve x bucket tests (weekday, weekend, 4h hour-of-day, turn of month, month end, funding-cycle phase) per period.
  **One** passes the both-periods rule: C entries in the hour before a 00/08/16 funding settlement are worse
  (diff -0.14 [-0.25, -0.01] in 2022-24 on 5 trades, -0.19 [-0.39, -0.01] in 2025-26 on 12 trades). About 0.1 such hits
  are expected by chance, and the hit fails the check further down: on the wider crash pools (no CNN filter, 400 and 1,073
  trades, 2020-26) the difference never excludes 0 in both periods, and it reverses in 2020-21. 25 more buckets hit in one
  period only (about 9 expected), and none repeats in the other period.
* Part of the C pattern has a mechanism, but a small one. A short opened just before a settlement after a crash pays the
  first, deeply negative rate: funding paid is +1.2-3.2% vs +0.4-1.2% for other entries. That is worth at most an
  execution rule ("open C shorts just after a settlement"), worth about 1-2% on the ~10% of C trades affected. It is not
  a filter.
* Choosing each sleeve's best calendar exclusion on 2022-24 and reading it on 2025-26: L +0.167 -> +0.168, C +0.041 -> +0.067,
  M +0.227 -> +0.195, F +0.110 -> +0.101. One of four holds, which is what chance gives. The LMC book with the C filter
  looks better (worst year +51% -> +68%), but that comes from 34 dropped C trades whose bucket was chosen in-sample.
  With all four filters, LMCF's worst year falls from +101% to +81%.

## Why it failed

The sleeves rest on few independent days. L's 1,883 trades come from about 25 flush days, so each hour or weekday bucket of L
is 1-8 days, and a single day decides it: L's "funding phase 2-4h" is 295 trades on 8 days in 2022-24 (diff -0.20) and 51
trades on 2 days in 2025-26 (diff -0.01). C, M and F have 5-30 trades per bucket and period. With day-clustered CIs of
+-0.1 to +-0.3 on E[o], only an effect of more than 15-20 points per trade could show up. The sleeves' edges are event-driven
(flushes, listings, pumps), and the calendar does not schedule those events.

"""

PERIODS = {"22-24": (2022, 2024), "25-26": (2025, 2026)}
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def with_calendar(s: pl.DataFrame) -> pl.DataFrame:
    dt = pl.from_epoch("t_in")
    s = s.with_columns(
        dow=dt.dt.weekday() - 1,
        hour=dt.dt.hour(),
        dom=dt.dt.day(),
        dim=dt.dt.month_end().dt.day(),
        phase=(pl.col("t_in") % 28800) / 60.0,  # minutes since the last 00/08/16 settlement
        day=dt.dt.date(),
        o=pl.col("net").clip(-1.0, 1.0),
    )
    return s


def buckets(s: pl.DataFrame) -> dict[str, np.ndarray]:
    dow, hour = s["dow"].to_numpy(), s["hour"].to_numpy()
    dom, dim, ph = s["dom"].to_numpy(), s["dim"].to_numpy(), s["phase"].to_numpy()
    b: dict[str, np.ndarray] = {f"dow={d}": dow == i for i, d in enumerate(DOW)}
    b["weekend (Sat+Sun)"] = dow >= 5
    for h in range(0, 24, 4):
        b[f"hour {h:02d}-{h + 4:02d}"] = (hour >= h) & (hour < h + 4)
    b["turn of month (last 3 + first 2 days)"] = (dom >= dim - 2) | (dom <= 2)
    b["month end (last 3 days)"] = dom >= dim - 2
    b["<=1h before funding settlement"] = ph >= 420
    b["<1h after funding settlement"] = ph < 60
    for a in range(0, 8, 2):
        b[f"funding phase {a}-{a + 2}h"] = (ph >= a * 60) & (ph < (a + 2) * 60)
    return b


def main() -> None:
    S = with_calendar(sleeves())
    md = ["# X8 A: calendar effects on the sleeves L, C, M, F", "",
          "In-sample exploration (`x8_calendar.py`, helpers `x8_lib.py`). Trades: every candidate of each sleeve, no",
          "slot limit (L 1883, C 161, F 143, M 136; 2022-26, the sleeves need the walk-forward CNN, so 2021 has none).",
          "Entry time = the actual entry (next bar open after the signal), UTC. o = net on the stake, clipped to [-1, 1].",
          "CI = 95% bootstrap with calendar days resampled (2000 draws); `diff` = bucket mean minus rest mean.",
          "An **effect** needs the diff CI to exclude 0 with the same sign in 2022-24 and in 2025-26.", "", VERDICT]  # fmt: skip
    n_tests, hits, near = 0, [], []
    rows_csv = []
    for sl in "LCMF":
        s = S.filter(pl.col("sleeve") == sl)
        md += [f"## {sl}", "",
               "| bucket | n 22-24 (days) | E[o] 22-24 [CI] | diff 22-24 [CI] | n 25-26 (days) | E[o] 25-26 [CI] | diff 25-26 [CI] | effect |",
               "|---|---|---|---|---|---|---|---|"]  # fmt: skip
        per = {k: s.filter(pl.col("year").is_between(*v)) for k, v in PERIODS.items()}
        bks = {k: buckets(v) for k, v in per.items()}
        for name in bks["22-24"]:
            cells, sig = [], []
            for k, v in per.items():
                msk = bks[k][name]
                o, day = v["o"].to_numpy(), v["day"].to_numpy()
                n, nd = int(msk.sum()), len(np.unique(day[msk]))
                if n < 3 or n > len(o) - 3:
                    cells += [f"{n} ({nd})", "-", "-"]
                    sig.append(0)
                    continue
                lo, hi, a, b, _ = boot_diff(o, day, msk, seed=zlib.crc32(f"{sl}{name}{k}".encode()))
                mi, mo = float(o[msk].mean()), float(o[~msk].mean())
                cells += [f"{n} ({nd})", f"{mi:+.3f} [{a:+.2f},{b:+.2f}]",
                          f"{mi - mo:+.3f} [{lo:+.2f},{hi:+.2f}]"]  # fmt: skip
                sig.append(1 if lo > 0 else -1 if hi < 0 else 0)
                rows_csv.append({"sleeve": sl, "bucket": name, "period": k, "n": n, "days": nd,
                                 "E_o": mi, "diff": mi - mo, "diff_lo": lo, "diff_hi": hi})  # fmt: skip
            n_tests += 1
            eff = "**yes**" if sig[0] != 0 and sig[0] == sig[1] else ""
            if sig[0] != 0 and sig[0] == sig[1]:
                hits.append(f"{sl} {name}")
            elif sig[0] != 0 or sig[1] != 0:
                near.append(
                    f"{sl} {name} ({'22-24' if sig[0] else '25-26'} only, {'+' if max(sig) > 0 else '-'})"
                )
                eff = "one period only"
            md.append(f"| {name} | " + " | ".join(cells) + f" | {eff} |")
        md.append("")
    pl.DataFrame(rows_csv).write_csv(OUT / "x8_calendar_buckets.csv")

    # Expected false positives: with independent tests at 5% two-sided, P(same-sign hit in both)
    # = 2 * 0.025^2 = 0.00125 per test.
    md += ["## Multiple testing", "",
           f"{n_tests} sleeve x bucket tests per period. Under no effect, a test hits one period at about 5% (expected "
           f"{n_tests * 0.05:.1f} one-period hits per period) and both periods with the same sign at 0.125% (expected "
           f"{n_tests * 0.00125:.2f}). Buckets are nested and correlated, so these are rough.", "",
           f"* Effects in both periods: **{len(hits)}** {'(' + '; '.join(hits) + ')' if hits else ''}",
           f"* One-period-only hits: {len(near)}: " + ("; ".join(near) if near else "none"), ""]  # fmt: skip

    # Choose-on-2022-24, read-on-2025-26: per sleeve the single bucket exclusion that most
    # raises the 2022-24 mean o while keeping >= 60% of trades; then books.
    md += ["## Best exclusion chosen on 2022-24, read on 2025-26", "",
           "Per sleeve: drop the one bucket (from the table above) whose removal most raises the 2022-24 mean o while",
           "keeping >= 60% of the 2022-24 trades. Then the same rule on 2025-26 (out of the choosing window, but all",
           "of this is in-sample research; the sleeves themselves were built on 2022-26).", "",
           "| sleeve | dropped bucket | n kept 22-24 | E[o] base -> kept 22-24 | n kept 25-26 | E[o] base -> kept 25-26 |",
           "|---|---|---|---|---|---|"]  # fmt: skip
    keep_masks: dict[str, np.ndarray] = {}
    cards: dict[str, dict[str, float]] = {}
    for sl in "LCMF":
        s = S.filter(pl.col("sleeve") == sl)
        a = s.filter(pl.col("year") <= 2024)
        ba = buckets(a)
        oa = a["o"].to_numpy()
        best, bv = None, -9.0
        for name, msk in ba.items():
            if (~msk).sum() < 0.6 * len(oa) or (~msk).sum() == len(oa):
                continue
            v = float(oa[~msk].mean())
            if v > bv:
                best, bv = name, v
        assert best is not None
        bs = buckets(s)[best]
        keep_masks[sl] = ~bs
        b = s.filter(pl.col("year") >= 2025)
        bb = buckets(b)[best]
        ob = b["o"].to_numpy()
        md.append(f"| {sl} | {best} | {int((~ba[best]).sum())} | {oa.mean():+.3f} -> {bv:+.3f} | "
                  f"{int((~bb).sum())} | {ob.mean():+.3f} -> {ob[~bb].mean():+.3f} |")  # fmt: skip
        cards[f"{sl} base"] = card(s)
        cards[f"{sl} minus {best}"] = card(s.filter(pl.Series(~bs)))
    md += ["", "## Diag scorecards (every candidate, 2022-26)", "", table(cards), ""]

    # Per-year E[o] of each sleeve, base vs filtered (2021 has no sleeve trades).
    md += [
        "## Per-year mean o (base -> filtered)",
        "",
        "| sleeve | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |",
        "|---|---|---|---|---|---|---|",
    ]
    for sl in "LCMF":
        s = S.filter(pl.col("sleeve") == sl).with_columns(k=pl.Series(keep_masks[sl]))
        cells = []
        for y in range(2021, 2027):
            sy = s.filter(pl.col("year") == y)
            if sy.height == 0:
                cells.append("-")
                continue
            sk = sy.filter(pl.col("k"))
            cells.append(
                f"{sy['o'].mean():+.3f} -> {sk['o'].mean() if sk.height else float('nan'):+.3f} ({sy.height}/{sk.height})"
            )
        md.append(f"| {sl} | " + " | ".join(cells) + " |")

    # Books: 10 slots, e25 book (per-year $10k), LMC and LMCF with each filter and all filters.
    def fr(sl: str, filt: bool) -> pl.DataFrame:
        s = S.filter(pl.col("sleeve") == sl)
        return as_book(s.filter(pl.Series(keep_masks[sl])) if filt else s)

    variants: dict[str, dict[str, bool]] = {"base": {}}
    for sl in "LCMF":
        variants[f"filter {sl}"] = {sl: True}
    variants["filter all four"] = dict.fromkeys("LCMF", True)
    md += ["", "## 10-slot books (e25 `book`, each year from $10k)", "",
           "| book | variant | trades | bp/trade | worst year | geo yearly | max DD | years |", "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for combo in ("LMC", "LMCF"):
        for vn, v in variants.items():
            if any(k not in combo for k in v):
                continue
            r = run_book([fr(sl, v.get(sl, False)) for sl in combo], 10)
            md.append(f"| {combo} | {vn} | {r['trades']} | {r['bp']:+.0f} | {r['min'] * 100:+.0f}% | "  # type: ignore[operator]
                      f"{r['geo'] * 100:+.0f}% | {r['dd'] * 100:+.0f}% | {r['years']} |")  # type: ignore[operator]  # fmt: skip
            print(md[-1], flush=True)
    md += c_probe()
    (OUT / "calendar.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def c_probe() -> list[str]:
    """Does C's 'just before a funding settlement' effect hold on the wider crash pool (no CNN
    filter; r4 <= -15% and <= -10%; 2020-26) and on each coin's actual settlement clock?"""
    from common_ls import OWN
    from futures_ls import funding_cum
    from sim import Rule
    from x8_lib import _sim

    feats = pl.read_parquet(OWN / "feats.parquet")
    fc = funding_cum()

    def nxt(sym: str, t: int) -> float:
        c = fc.get(sym)
        if c is None:
            return float("nan")
        k = int(np.searchsorted(c[0], t, side="right"))
        return float((c[0][k] - t) / 60) if k < len(c[0]) else float("nan")

    md = ["", "## Check of the one hit: C entries just before a funding settlement, on wider crash pools", "",
          "Pools: perp, qv_base >= 20k, BTC 4h calm (|r| <= 1%), r4 <= -15% or <= -10%, **no CNN filter** (C uses CNN <= 0.5),",
          "2020-26, same 24h perp short. `actual` = minutes to the coin's next settlement in the funding history (4h-interval",
          "coins included); `fixed` = the 00/08/16 grid. Funding = mean funding paid by the short over the hold.", "",
          "| pool | bucket | period | n in bucket / all | E[o] bucket | E[o] rest | diff CI (day bootstrap) | funding paid bucket / rest (bp) |",
          "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for r4 in (-0.15, -0.10):
        d = feats.filter(pl.col("perp") & (pl.col("qv_base") >= 20_000) & (pl.col("ret_4h") <= r4)
                         & (pl.col("btc_ret_4h").abs() <= 0.01))  # fmt: skip
        t = _sim(
            d.select("row", side=pl.lit(-1), prio=-pl.col("ret_4h")),
            Rule(hold_min=1440, slots=100_000, cost=0.003, sl=1.0, liq_frac=None),
            perp_funding=True,
            source="perp",
        )
        t = t.with_columns(
            ph=(pl.col("t_in") % 28800) / 60.0, day=pl.from_epoch("t_in").dt.date(),
            o=pl.col("net").clip(-1, 1),
            nx=pl.Series([nxt(s, int(x)) for s, x in zip(t["symbol"], t["t_in"], strict=True)]),
        )  # fmt: skip
        for nm, ex in (("fixed <=1h before", pl.col("ph") >= 420), ("fixed last 2h (phase 6-8h)", pl.col("ph") >= 360),
                       ("actual <=60 min before", pl.col("nx") <= 60), ("actual <=120 min before", pl.col("nx") <= 120)):  # fmt: skip
            for per in ((2020, 2021), (2022, 2024), (2025, 2026)):
                v = t.filter(pl.col("year").is_between(*per))
                msk = v.select(ex.fill_null(False)).to_series().to_numpy()
                if msk.sum() < 3:
                    continue
                lo, hi, _, _, _ = boot_diff(v["o"].to_numpy(), v["day"].to_numpy(), msk)
                a, b = v.filter(pl.Series(msk)), v.filter(pl.Series(~msk))
                md.append(f"| r4 <= {r4:.0%} | {nm} | {per[0]}-{str(per[1])[2:]} | {int(msk.sum())} / {v.height} | {a['o'].mean():+.3f} | "
                          f"{b['o'].mean():+.3f} | [{lo:+.3f}, {hi:+.3f}] | {a['fund'].mean() * 1e4:+.0f} / {b['fund'].mean() * 1e4:+.0f} |")  # fmt: skip
    return md


if __name__ == "__main__":
    main()
