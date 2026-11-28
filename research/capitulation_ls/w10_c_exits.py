"""W10: why exit rules fail on sleeve C (cap-crash perp shorts)."""

from __future__ import annotations

import numpy as np
import polars as pl
from common_ls import OWN, meta, scores
from sim import PERP_STEP_END_MIN, Rule, perp_paths, simulate, simulate_yearly

m = meta()
d = pl.read_parquet(OWN / "feats.parquet").filter(
    pl.col("perp")
    & (pl.col("qv_base") >= 20_000)
    & (pl.col("open_time").dt.year() >= 2022)
    & (pl.col("ret_4h") <= -0.15)
    & (pl.col("btc_ret_4h").abs() <= 0.01)
)
pct = scores("day5")
d = d.filter(pl.Series(pct[d["row"].to_numpy()] <= 0.5))
cand = d.select("row", side=pl.lit(-1), prio=-pl.col("ret_4h"))
print("pool", cand.height)


def run(rule, label, slots_list=(5, 10, 100000)):
    for s in slots_list:
        r = Rule(**{**rule.__dict__, "slots": s})
        b = simulate_yearly(
            cand, r, perp_funding=True, source="perp", years=(2022, 2023, 2024, 2025, 2026)
        )
        y = {int(a["year"]): (a["trades"], round(a["bp"])) for a in b.years.iter_rows(named=True)}
        print(f"{label:28s} slots={s:6d} n={b.summary['trades']:4d} bp={b.summary['bp']:+6.0f} {y}")
    return None


base = dict(cost=0.003, sl=1.0, liq_frac=None)
run(Rule(hold_min=1440, **base), "24h", (5,))
run(Rule(hold_min=480, **base), "8h", (5,))
run(Rule(hold_min=720, **base), "12h", (5,))
run(Rule(hold_min=240, **base), "4h", (5,))
run(Rule(hold_min=1440, tp=0.10, **base), "TP10 24h", (5,))
run(Rule(hold_min=1440, tp=0.15, **base), "TP15 24h", (5,))
run(Rule(hold_min=1440, trail=0.05, trail_arm=0.05, **base), "trail5 24h", (5,))


base_r = dict(cost=0.003, sl=1.0, liq_frac=None)


def book(rule):
    return simulate_yearly(
        cand,
        Rule(**{**rule.__dict__, "slots": 5}),
        perp_funding=True,
        source="perp",
        years=(2022, 2023, 2024, 2025, 2026),
    ).trades


def net_of(rule, rowset):
    """unconstrained per-row net under a rule"""
    sub = cand.filter(pl.col("row").is_in(list(rowset)))
    for _y in (2022, 2023, 2024, 2025, 2026):
        pass
    t = simulate(
        sub,
        Rule(**{**rule.__dict__, "slots": 100000}),
        start_equity=1e15,
        perp_funding=True,
        source="perp",
    ).trades
    return dict(zip(t["row"].to_list(), t["net"].to_list(), strict=False))


b24 = book(Rule(hold_min=1440, **base_r))
rows24 = set(b24["row"].to_list())
print("book24 rows", len(rows24))
allrows = cand["row"].to_list()
N24 = dict(zip(b24["row"].to_list(), b24["net"].to_list(), strict=False))
print(
    "unconstrained n",
    len(N24),
    "mean",
    np.mean(list(N24.values())) * 1e4,
    "in-book mean",
    np.mean([N24[r] for r in rows24 if r in N24]) * 1e4,
    "out-of-book mean",
    np.mean([v for r, v in N24.items() if r not in rows24]) * 1e4,
)
# freed-slot analysis (book rows: kept = same entry row as the 24h book; added = extra entries)
for lab, rule in {
    "TP10 24h": Rule(hold_min=1440, tp=0.10, **base_r),
    "TP15 24h": Rule(hold_min=1440, tp=0.15, **base_r),
    "trail5 24h": Rule(hold_min=1440, trail=0.05, trail_arm=0.05, **base_r),
    "8h": Rule(hold_min=480, **base_r),
    "4h": Rule(hold_min=240, **base_r),
}.items():
    bt = book(rule)
    isk = bt["row"].is_in(list(rows24))
    k = bt.filter(isk)
    a = bt.filter(~isk)
    print(
        f"{lab}: n={bt.height} kept {k.height} mean {k['net'].mean() * 1e4:+.0f}bp (24h same rows {np.mean([N24[r] for r in k['row'].to_list() if r in N24]) * 1e4:+.0f}) ; added {a.height} mean {a['net'].mean() * 1e4 if a.height else float('nan'):+.0f}bp win {(a['net'] > 0).mean() if a.height else float('nan'):.0%}; book mean {bt['net'].mean() * 1e4:+.0f}"  # noqa: E501
    )
    if a.height:
        print(
            "   added by year",
            a.group_by("year").agg(n=pl.len(), bp=pl.col("net").mean() * 1e4).sort("year").rows(),
        )
# chain analysis: re-entries in the same coin within 24h of the event's first entry
print("--- chains (event = first entry + re-entries of the same coin inside 24h of it)")
first24 = {
    (a, r): n for a, r, n in zip(b24["symbol"].to_list(), b24["t"].to_list(), b24["net"].to_list(), strict=False)  # noqa: E501
}
for lab, rule in {
    "TP10 24h": Rule(hold_min=1440, tp=0.10, **base_r),
    "TP15 24h": Rule(hold_min=1440, tp=0.15, **base_r),
    "trail5 24h": Rule(hold_min=1440, trail=0.05, trail_arm=0.05, **base_r),
}.items():
    bt = book(rule).sort(["symbol", "t"])
    ev = []
    cur = None
    for r in bt.iter_rows(named=True):
        if cur and cur["symbol"] == r["symbol"] and r["t"] - cur["t0"] < 86400 * 1.0:
            cur["nets"].append(r["net"])
            cur["gaps"].append((r["t"] - cur["t_last_out"]) / 3600)
            cur["t_last_out"] = r["t_out"]
        else:
            if cur:
                ev.append(cur)
            cur = {
                "symbol": r["symbol"],
                "t0": r["t"],
                "nets": [r["net"]],
                "gaps": [],
                "t_last_out": r["t_out"],
                "year": r["year"],
            }
    ev.append(cur)
    multi = [e for e in ev if len(e["nets"]) > 1]
    first_only = np.mean([e["nets"][0] for e in ev]) * 1e4
    chain_sum = np.mean([sum(e["nets"]) for e in ev]) * 1e4
    print(
        f"{lab}: events {len(ev)} ; with re-entries {len(multi)} ({sum(len(e['nets']) - 1 for e in multi)} extra trades); first-trade mean {first_only:+.0f}bp; chain-sum mean per event {chain_sum:+.0f}bp (24h plain +436)"  # noqa: E501
    )
    ex = [x for e in multi for x in e["nets"][1:]]
    print(
        f"   re-entry trades mean {np.mean(ex) * 1e4:+.0f}bp, win {np.mean(np.array(ex) > 0):.0%}; median gap exit->re-entry {np.median([g for e in multi for g in e['gaps']]):.1f}h"  # noqa: E501
    )
    # equal stake per trade in the book; per-event stake-weighted
    for y in (2022,):
        e22 = [e for e in ev if e["year"] == y]
        print("   2022 events", len(e22), "chain-sum", np.mean([sum(e["nets"]) for e in e22]) * 1e4)


print("--- per-trade decomposition on the 162 shared entries: rule net minus 24h net")
for lab, rule in {
    "TP10 24h": Rule(hold_min=1440, tp=0.10, **base_r),
    "TP15 24h": Rule(hold_min=1440, tp=0.15, **base_r),
    "trail5 24h": Rule(hold_min=1440, trail=0.05, trail_arm=0.05, **base_r),
    "8h": Rule(hold_min=480, **base_r),
    "4h": Rule(hold_min=240, **base_r),
    "12h": Rule(hold_min=720, **base_r),
}.items():
    bt = book(rule)
    k = bt.filter(bt["row"].is_in(list(rows24))).select("row", nr="net", kr="kind", hm="hold_min")
    j = b24.select("row", n24="net", k24="kind").join(k, on="row")
    d = (j["nr"] - j["n24"]).to_numpy()
    n = len(d)
    print(
        f"{lab}: n={n} total diff {d.sum() / n * 1e4:+.0f}bp/trade ; trades where rule better: {(d > 0).sum()} sum {d[d > 0].sum() / n * 1e4:+.0f}bp ; worse: {(d < 0).sum()} sum {d[d < 0].sum() / n * 1e4:+.0f}bp ; of worse, 24h net>+20%: {((d < 0) & (j['n24'].to_numpy() > 0.2)).sum()} costing {d[(d < 0) & (j['n24'].to_numpy() > 0.2)].sum() / n * 1e4:+.0f}bp; early exits {(j['kr'].to_numpy() != 0).sum()}"  # noqa: E501
    )
    top = np.argsort(d)[:5]
    print(
        "   5 worst diffs bp",
        np.round(d[top] * 1e4).astype(int),
        "their 24h nets",
        np.round(j["n24"].to_numpy()[top] * 1e4).astype(int),
    )
# stops in the 24h book
print(
    "24h book stops:",
    b24.filter(pl.col("kind") == 1).select("symbol", "open_time", "net", "hold_min").rows(),
)
print("24h mean without the 3 stops:", b24.filter(pl.col("kind") != 1)["net"].mean() * 1e4)

# path analysis on the unconstrained 922 and on 162 book
rows = np.array(allrows)
p, ok = perp_paths(rows)
rows_ok = rows[ok]


def sub_analysis(label, sel_rows):
    sel = np.isin(rows_ok, list(sel_rows))
    P = p[sel]
    rr = rows_ok[sel]
    net = np.array([N24[r] for r in rr])
    n = len(rr)
    fav_close = -P[:, 2, :]
    fav_hi = -P[:, 1, :]
    adv_hi = P[:, 0, :]
    idx = np.array([h * 12 - 1 if h <= 8 else 96 + (h - 9) for h in range(1, 73)])
    path = fav_close[:, idx]
    print(f"=== {label} n={n} 24h mean {net.mean() * 1e4:+.0f}bp")

    def pr(lab, mask):
        pp = path[mask]
        print(f"  {lab} n={mask.sum()}  (short favourable close, %)")
        print("   hr   " + " ".join(f"{h:6d}" for h in (1, 2, 4, 6, 8, 12, 18, 24, 36, 48, 60, 72)))
        print(
            "   mean "
            + " ".join(
                f"{np.nanmean(pp[:, h - 1]) * 100:+6.1f}"
                for h in (1, 2, 4, 6, 8, 12, 18, 24, 36, 48, 60, 72)
            )
        )
        print(
            "   med  "
            + " ".join(
                f"{np.nanmedian(pp[:, h - 1]) * 100:+6.1f}"
                for h in (1, 2, 4, 6, 8, 12, 18, 24, 36, 48, 60, 72)
            )
        )

    pr("all", np.ones(n, bool))
    pr("winners(24h net>0)", net > 0)
    pr("losers", net <= 0)
    i24 = idx[23]
    fin24 = fav_close[:, i24]
    mfe24 = np.nanmax(fav_hi[:, : i24 + 1], axis=1)
    mae24 = np.nanmax(adv_hi[:, : i24 + 1], axis=1)
    print(
        "  mfe24 p10/25/50/75/90",
        np.percentile(mfe24, [10, 25, 50, 75, 90]).round(3),
        " mae24",
        np.percentile(mae24, [10, 25, 50, 75, 90]).round(3),
    )
    for L in (0.10, 0.15):
        hit = mfe24 >= L
        first = np.array([np.argmax(fav_hi[i, : i24 + 1] >= L) if hit[i] else -1 for i in range(n)])
        tth = np.array([PERP_STEP_END_MIN[f] / 60 for f in first if f >= 0])
        fin = fin24[hit]
        np.array([0.0] * n)
        print(
            f"  L={L}: hit {hit.sum()} ({hit.mean():.0%}), median t-to-hit {np.median(tth):.1f}h; close24 of hitters mean {fin.mean() * 100:+.1f}% med {np.median(fin) * 100:+.1f}%; finish>L {np.mean(fin > L):.0%}; finish in 0..L {np.mean((fin > 0) & (fin <= L)):.0%}; finish<0 {np.mean(fin < 0):.0%}"  # noqa: E501
        )
        up = np.clip(fin - L, 0, None).sum() / n * 1e4
        dn = np.clip(L - fin, 0, None).sum() / n * 1e4
        print(
            f"     TP effect vs hold (per trade over all trades, before funding): lose upside beyond L {-up:.0f}bp, save fall-backs {dn:.0f}bp, net {dn - up:+.0f}bp"  # noqa: E501
        )
        rt = 0
        rt_hi = 0
        for i in np.flatnonzero(hit):
            f = first[i]
            seg = fav_close[i, f : i24 + 1]
            if np.nanmin(seg) <= L - 0.05:
                rt += 1
                if fin24[i] > L:
                    rt_hi += 1
        print(
            f"     hit then closed >=5pp below L at some hour: {rt}; of those finishing above L at 24h: {rt_hi}; above L at 72h: {sum(1 for i in np.flatnonzero(hit) if path[i, 71] > L and np.nanmin(fav_close[i, first[i] : i24 + 1]) <= L - 0.05)}"  # noqa: E501
        )
    s = np.sort(net)[::-1]
    print(
        "  top10% share of total:",
        round(s[: max(1, n // 10)].sum() / s.sum(), 2),
        "top20%:",
        round(s[: max(1, n // 5)].sum() / s.sum(), 2),
        "; net pcts 5/25/50/75/95",
        np.percentile(net, [5, 25, 50, 75, 95]).round(3),
    )
    return rows_ok[sel], net, path, mfe24, fin24, idx


r162, net162, path162, mfe162, fin162, idx = sub_analysis("24h BOOK (162)", rows24)
# beyond 24h: drift
print(
    "book mean fav close at 24/36/48/72:",
    [round(float(np.nanmean(path162[:, h - 1]) * 100), 2) for h in (24, 36, 48, 72)],
)
# 2022
t22 = b24.filter(pl.col("year") == 2022).sort("open_time")
sel = np.array([list(r162).index(r) for r in t22["row"].to_list()])
print("2022 book trades")
for k, i in enumerate(sel):
    print(
        t22["open_time"][k],
        t22["symbol"][k],
        f"net24 {net162[i] * 1e4:+.0f}  f4 {path162[i, 3] * 100:+.1f} f8 {path162[i, 7] * 100:+.1f} f12 {path162[i, 11] * 100:+.1f} f24 {path162[i, 23] * 100:+.1f} mfe24 {mfe162[i] * 100:.1f}",  # noqa: E501
    )
for y in (2022, 2023, 2024, 2025, 2026):
    ty = b24.filter(pl.col("year") == y)
    ii = np.array([list(r162).index(r) for r in ty["row"].to_list()])
    print(
        y,
        len(ii),
        "f4/f8/f12/f24 mean %",
        [round(float(np.nanmean(path162[ii, h - 1]) * 100), 2) for h in (4, 8, 12, 24)],
    )
