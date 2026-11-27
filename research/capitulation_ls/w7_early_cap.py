"""w7: why capping early-episode long trades at a few slots fails.
Own booking loop (copy of e25.book logic)
that logs every accepted trade, so displaced and replacement trades can be compared."""

import numpy as np
import polars as pl
from e25_combined import book, sleeve_long

SLOTS = 10
L = sleeve_long().sort("t_in")
t = L["t_in"].to_numpy()
ep = np.concatenate([[0], np.cumsum(np.diff(t) >= 7200)])
L = L.with_columns(
    ep=pl.Series(ep), ep_start=pl.Series(t[np.r_[0, np.flatnonzero(np.diff(t) >= 7200) + 1]][ep])
)
L = L.with_columns(
    off=(pl.col("t_in") - pl.col("ep_start")) / 60, year=pl.from_epoch("t_in").dt.year()
)
print("candidates", L.height, "episodes", L["ep"].n_unique())


def run(tr, caps_win=None, cap=None, slots=SLOTS, start=10_000.0):
    """returns accepted trades with stake and pnl; early = off < caps_win capped at `cap` concurrent early."""  # noqa: E501
    tr = tr.sort(["t_in", "prio"], descending=[False, True])
    out = []
    for (y,), g in tr.group_by(["year"], maintain_order=True):
        eq, cash, pos, held = start, start, [], set()
        for r in g.iter_rows(named=True):
            keep = []
            for p in pos:
                if p["t_out"] <= r["t_in"]:
                    cash += p["stake"] * (1 + p["net"])
                    eq += p["stake"] * p["net"]
                    held.discard(p["symbol"])
                else:
                    keep.append(p)
            pos = keep
            early = caps_win is not None and r["off"] < caps_win
            r["early"] = early
            if len(pos) >= slots or r["symbol"] in held:
                continue
            if early and sum(1 for p in pos if p["early"]) >= cap:
                continue
            stake = min(eq / slots, cash, r["qv"] * 0.005)
            if stake < 1:
                continue
            cash -= stake
            r["stake"] = stake
            pos.append(r)
            held.add(r["symbol"])
            out.append(r)
        for p in pos:
            eq += p["stake"] * p["net"]
        out_eq = eq
        for o in out:
            if o["year"] == y:
                o["eq_end"] = out_eq
    return pl.DataFrame(out)


def yr_ret(a, start=10_000.0):
    return {y: g["eq_end"][0] / start - 1 for (y,), g in a.group_by(["year"])}


# 1. candidates: share of trades and P&L (sum net) in first 15/30 min
print(
    "\n== 1a. unconstrained candidates by year: n, share of n and of sum(net) in first 15 / 30 min, mean bp early/late"  # noqa: E501
)
for W in (15, 30):
    print(f"-- W={W}")
    for y in (2022, 2023, 2024, 2025, 2026, None):
        g = L if y is None else L.filter(pl.col("year") == y)
        e = g.filter(pl.col("off") < W)
        late = g.filter(pl.col("off") >= W)
        print(
            y,
            "n",
            g.height,
            "early n",
            e.height,
            f"({e.height / max(g.height, 1):.0%})",
            f"sum-net share {e['net'].sum() / g['net'].sum():.0%}",
            f"bp early {e['net'].mean() * 1e4 if e.height else 0:+.0f} late {late['net'].mean() * 1e4 if late.height else 0:+.0f} n_late {late.height}",  # noqa: E501
        )

# 2. uncapped booked
U = run(L)
print(
    "\n== 2. booked uncapped (n, years):",
    U.height,
    {k: round(v, 3) for k, v in sorted(yr_ret(U).items())},
    f"mean bp {U['net'].mean() * 1e4:+.0f}",
)
ref = book(L, SLOTS)
print("check vs e25.book:", ref["trades"].to_list(), [round(x, 3) for x in ref["ret"].to_list()])
U = U.with_columns(pnl=pl.col("stake") * pl.col("net"))
for W in (15, 30):
    print(f"-- booked, share in first {W} min by year (n, pnl$, mean bp early vs late)")
    for y in (2022, 2023, 2024, 2025, 2026):
        g = U.filter(pl.col("year") == y)
        e = g.filter(pl.col("off") < W)
        late = g.filter(pl.col("off") >= W)
        print(
            y,
            "n",
            g.height,
            "early n",
            e.height,
            f"pnl share {e['pnl'].sum() / g['pnl'].sum():.0%}",
            f"early bp {e['net'].mean() * 1e4 if e.height else 0:+.0f} late bp {late['net'].mean() * 1e4 if late.height else 0:+.0f}",  # noqa: E501
            f"early pnl {e['pnl'].sum():+.0f} late pnl {late['pnl'].sum():+.0f}",
        )

# 3. capped vs uncapped: displaced vs replacements
for W, cap in ((15, 3), (15, 5), (30, 3)):
    C = run(L, W, cap).with_columns(pnl=pl.col("stake") * pl.col("net"))
    print(
        f"\n== 3. cap early<{W}min at {cap}: n {C.height} bp {C['net'].mean() * 1e4:+.0f}",
        {k: round(v, 3) for k, v in sorted(yr_ret(C).items())},
    )
    ku = set(zip(U["symbol"], U["t_in"], strict=False))
    kc = set(zip(C["symbol"], C["t_in"], strict=False))
    disp = U.filter(
        pl.struct("symbol", "t_in").map_elements(
            lambda s, kc=kc: (s["symbol"], s["t_in"]) not in kc, return_dtype=pl.Boolean
        )
    )
    repl = C.filter(
        pl.struct("symbol", "t_in").map_elements(
            lambda s, ku=ku: (s["symbol"], s["t_in"]) not in ku, return_dtype=pl.Boolean
        )
    )
    print(
        f"displaced {disp.height} mean bp {disp['net'].mean() * 1e4:+.0f}; replacements {repl.height} mean bp {repl['net'].mean() * 1e4 if repl.height else 0:+.0f}"  # noqa: E501
    )
    print(
        " displaced all early? ",
        disp.filter(pl.col("off") < W).height,
        "| replacement off median min",
        repl["off"].median() if repl.height else None,
    )
    for y in (2022, 2023, 2024, 2025, 2026):
        d = disp.filter(pl.col("year") == y)
        r = repl.filter(pl.col("year") == y)
        print(
            y,
            f"displaced n {d.height} bp {d['net'].mean() * 1e4 if d.height else 0:+.0f} sum {d['net'].sum() * 1e4:+.0f}",  # noqa: E501
            f"| repl n {r.height} bp {r['net'].mean() * 1e4 if r.height else 0:+.0f} sum {r['net'].sum() * 1e4:+.0f}",  # noqa: E501
        )
    # per episode, those with displacement
    de = (
        disp.group_by("ep")
        .agg(nd=pl.len(), bpd=pl.col("net").mean() * 1e4, day=pl.col("t_in").min())
        .sort("day")
    )
    re_ = repl.group_by("ep").agg(nr=pl.len(), bpr=pl.col("net").mean() * 1e4)
    cand_late = (
        L.filter(pl.col("off") >= W)
        .group_by("ep")
        .agg(late_cands=pl.len(), late_coins=pl.col("symbol").n_unique())
    )
    x = (
        de.join(re_, on="ep", how="left")
        .join(cand_late, on="ep", how="left")
        .fill_null(0)
        .with_columns(day=pl.from_epoch("day").dt.strftime("%Y-%m-%d %H:%M"))
    )
    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_width_chars(200)
    print(x)
    print(
        "episodes with displacement:",
        x.height,
        "of those with >=1 replacement:",
        x.filter(pl.col("nr") > 0).height,
        "no late candidates at all:",
        x.filter(pl.col("late_cands") == 0).height,
    )
