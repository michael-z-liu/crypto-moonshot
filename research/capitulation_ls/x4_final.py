"""X4 step 3: best/baseline configs, trade + non-entry files, diag scorecards, tables for the note."""  # noqa: E501

import diag
import numpy as np
import polars as pl
from x4_lib import EX, KINDS, D, exits, run

g = pl.read_parquet(EX / "contagion_grid.parquet").with_columns(
    my=pl.mean_horizontal([f"a{y}" for y in range(2022, 2027)])
)
g = g.filter(pl.col("n") >= 100)
cfgs = {}
for mode in ("short", "long", "neutral"):
    cfgs[mode] = g.filter(pl.col("mode") == mode).sort("my", descending=True).row(0, named=True)
    c = cfgs[mode]
    print(
        "best",
        mode,
        {k: c[k] for k in ("kind", "k", "fell", "thr", "hold", "tp", "stop", "n", "avg", "my")},
    )
print(
    "grid stats: configs",
    g.height,
    "| avg trade > 0:",
    {m: round(float((g.filter(pl.col("mode") == m)["avg"] > 0).mean()), 3) for m in cfgs},
    "| max avg trade %:",
    {m: round(float(g.filter(pl.col("mode") == m)["avg"].max()) * 100, 2) for m in cfgs},
)
for m in cfgs:
    s = g.filter(pl.col("mode") == m)
    print(
        m,
        "by hold avg %:",
        {
            h: round(float(x) * 100, 2)
            for h, x in s.group_by("hold").agg(pl.col("avg").mean()).sort("hold").rows()
        },
    )
    print(
        m,
        "by kind avg %:",
        s.group_by("kind").agg(pl.col("avg").mean()).sort("kind").rows(),
        "by k:",
        s.group_by("k").agg(pl.col("avg").mean()).sort("k").rows(),
    )
    print(
        m,
        "by fell:",
        s.group_by("fell").agg(pl.col("avg").mean()).sort("fell").rows(),
        "by thr:",
        s.group_by("thr").agg(pl.col("avg").mean()).sort("thr").rows(),
    )

rows, nrows, cards = [], [], {}
KIND = KINDS


def frame(label, cname, r, side):
    idx, net, kd, mfe, mae, hh, _prio, taken = r["_taken"]
    t = taken
    tin = D["t"].to_numpy()[idx][t]
    stop = r["stop"] if r["stop"] is not None else (1.0 if side < 0 else np.nan)
    return pl.DataFrame(dict(
        strategy=label, config=cname, symbol=D["symbol"].to_numpy()[idx][t], side=np.full(t.sum(), side),  # noqa: E501
        t_in=tin, t_out=tin + hh[t] * 3600, signal=D["signal"].to_numpy()[idx][t], threshold=np.full(t.sum(), float(r["thr"])),  # noqa: E501
        net=net[t], mfe=mfe[t], mae=mae[t], tp=np.full(t.sum(), r["tp"] if r["tp"] is not None else np.nan),  # noqa: E501
        stop=np.full(t.sum(), stop), exit_kind=KIND[kd[t]], year=D["year"].to_numpy()[idx][t],
    ))  # fmt: skip


res = {}
for mode, side in (("short", -1), ("long", 1), ("neutral", -1)):
    c = cfgs[mode]
    args = (mode, c["kind"], c["k"], c["fell"], c["thr"], c["hold"], c["tp"], c["stop"])
    best = run(*args)
    base = run(mode, c["kind"], 10, None, 0.0, c["hold"], c["tp"], c["stop"], random_prio=True)
    for lab, r in ((f"contagion_{mode}", best), (f"contagion_{mode}_baseline", base)):
        res[lab] = r
        cname = f"{mode}_kind{r['kind']}_k{r['k']}_fell{r['fell']}_thr{r['thr']}_h{r['hold']}_tp{r['tp']}_s{r['stop']}"  # noqa: E501
        df = frame(lab, cname, r, side)
        rows.append(df)
        ys = " ".join(f"{y}:{r[f'y{y}'] * 100:+.0f}%({r[f'n{y}']})" for y in range(2021, 2027))
        print(
            lab,
            cname,
            "| book n",
            r["book_n"],
            "avg trade %.2f%%" % (r["book_avg"] * 100),
            "|",
            ys,
            "| worst {:.0f}% maxDD {:.0f}%".format(r["worst"] * 100, r["maxdd"] * 100),
        )
        print(
            f"   all-candidate avg (no slot limit) {r['avg'] * 100:.2f}% n={r['n']}; per-year all-cand avg %:",  # noqa: E501
            {y: round(r[f"a{y}"] * 100, 2) for y in range(2021, 2027)},
        )
    # non-entries of the best config: same kind / fell, lower-ranked candidates (as if entered)
    idx_in, _ = __import__("x4_lib").select(c["kind"], c["k"], c["fell"], c["thr"])
    m = (D["kind"].to_numpy() == c["kind"]) & (D["year"].to_numpy() >= 2021)
    if c["fell"] is not None:
        m &= D["peer_r4"].to_numpy() > c["fell"]
    m[idx_in] = False
    idx = np.where(m)[0]
    idx = np.sort(np.random.default_rng(1).choice(idx, min(20000, len(idx)), replace=False))
    net, kd, mfe, mae, hh = exits(idx, mode, c["hold"], c["tp"], c["stop"])
    tin = D["t"].to_numpy()[idx]
    ent = res[f"contagion_{mode}"]["_taken"]
    thr_sig = float(np.min(D["signal"].to_numpy()[ent[0]]))
    nrows.append(pl.DataFrame(dict(
        strategy=f"contagion_{mode}", config="nonentry", symbol=D["symbol"].to_numpy()[idx], side=np.full(len(idx), side),  # noqa: E501
        t_in=tin, t_out=tin + hh * 3600, signal=D["signal"].to_numpy()[idx], threshold=np.full(len(idx), thr_sig), net=net, mfe=mfe, mae=mae,  # noqa: E501
        tp=np.full(len(idx), c["tp"] if c["tp"] is not None else np.nan), stop=np.full(len(idx), c["stop"] if c["stop"] is not None else (1.0 if side < 0 else np.nan)),  # noqa: E501
        exit_kind=KIND[kd], year=D["year"].to_numpy()[idx],
    )))  # fmt: skip
T = pl.concat(rows)
T.write_parquet(EX / "contagion_trades.parquet")
N = pl.concat(nrows)
N.write_parquet(EX / "contagion_nonentries.parquet")
print("trades", T.height, "nonentries", N.height)
# diag
for lab in [r for r in res if not r.endswith("baseline")] + [
    r for r in res if r.endswith("baseline")
]:
    t = T.filter(pl.col("strategy") == lab)
    cards[lab] = diag.scorecard(t)
for lab in cards:
    print(lab, {k: round(v, 3) for k, v in cards[lab].items()})
for mode in ("short", "long", "neutral"):
    t = T.filter(pl.col("strategy") == f"contagion_{mode}")
    n = N.filter(pl.col("strategy") == f"contagion_{mode}")
    print(mode, "threshold_reach", {k: round(v, 3) for k, v in diag.threshold_reach(t, n).items()})
    print(diag.calibration(t, n))
# signal lift: avg 24h net by signal quintile over all top-30 candidates (kind 0, short and long)
for mode in ("short", "long"):
    for kind in (0, 1):
        idx = np.where(D["kind"].to_numpy() == kind)[0]
        net, *_ = exits(idx, mode, 24, None, None)
        b = np.minimum((D["signal"].to_numpy()[idx] * 5).astype(int), 4)
        cr = np.minimum((D["crash_pct"].to_numpy()[idx] * 3).astype(int), 2)
        rk = D["rank"].to_numpy()[idx]
        print(
            mode,
            "kind",
            kind,
            "24h avg net bp by signal quintile",
            [int(net[b == q].mean() * 1e4) for q in range(5)],
            "| by crash-size tercile",
            [int(net[cr == q].mean() * 1e4) for q in range(3)],
            "| by rank 0-4/5-14/15-29",
            [
                int(net[rk < 5].mean() * 1e4),
                int(net[(rk >= 5) & (rk < 15)].mean() * 1e4),
                int(net[rk >= 15].mean() * 1e4),
            ],
        )
(EX / "x4_scorecards.md").write_text(diag.table(cards))
print(diag.table(cards))
