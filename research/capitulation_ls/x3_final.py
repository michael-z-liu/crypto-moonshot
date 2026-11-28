"""X3 step 6: pick best configs, write trade files and the tables for the note."""

import numpy as np
import polars as pl
from x3_eval import EX, MODES, D, exits, run_config, select

bk = pl.read_parquet(EX / "lag_books.parquet").filter(pl.col("book_n") >= 60)
bk = bk.with_columns(mean_year=pl.mean_horizontal([f"y{y}" for y in range(2021, 2027)]))
hyp = bk.filter(pl.col("mode").is_in(["up_long", "down_short"]) & (pl.col("thr") > 0))
best = hyp.sort("mean_year", descending=True).row(0, named=True)
bnc = bk.filter(pl.col("mode") == "bounce_fell").sort("mean_year", descending=True)
bnc = bnc.filter(pl.col("thr") > 0).row(0, named=True)
print("best hyp", {k: best[k] for k in ("mode", "combo", "thr", "hold", "tp", "stop")})
print("best bounce", {k: bnc[k] for k in ("mode", "combo", "thr", "hold", "tp", "stop")})
rows = []


def add(label, cfg, thr_for_row, rp):
    r = run_config(
        cfg["mode"],
        cfg["combo"],
        cfg["thr"] if not rp else 0.0,
        cfg["hold"],
        cfg["tp"],
        cfg["stop"],
        random_prio=rp,
    )
    idx, net, kind, mfe, mae, hh, _prio, taken, _stake = r["_taken"]
    _ev, side, _venue, sg = MODES[cfg["mode"]]
    sig = D["signal"].to_numpy()[idx]
    sig = sig if sg > 0 else 1 - sig
    cname = f"{cfg['mode']}_c{cfg['combo']}_h{cfg['hold']}_tp{cfg['tp']}_s{cfg['stop']}_thr{thr_for_row}"  # noqa: E501
    t = taken
    stop_col = cfg["stop"] if cfg["stop"] is not None else (1.0 if side < 0 else None)
    tin = D["t"].to_numpy()[idx][t]
    df = pl.DataFrame(dict(
        strategy=label, config=cname, symbol=D["symbol"].to_numpy()[idx][t], side=np.full(t.sum(), side),  # noqa: E501
        t_in=tin, t_out=tin + hh[t] * 3600, signal=sig[t], threshold=np.full(t.sum(), float(thr_for_row)),  # noqa: E501
        net=net[t], mfe=mfe[t], mae=mae[t], tp=np.full(t.sum(), cfg["tp"], dtype=float),
        stop=np.full(t.sum(), stop_col, dtype=float), exit_kind=KIND[kind[t]], year=D["year"].to_numpy()[idx][t],  # noqa: E501
    ))  # fmt: skip
    rows.append(df)
    return r, df


KIND = np.array(["time", "tp", "stop", "liq"])
out = {}
for label, cfg, thr, rp in (
    ("lag_best", best, best["thr"], False),
    ("lag_baseline", best, 0.0, True),
    ("bounce_best", bnc, bnc["thr"], False),
    ("bounce_baseline", bnc, 0.0, True),
):
    r, df = add(label, cfg, thr, rp)
    out[label] = (r, df)
    ys = " ".join(f"{y}:{r[f'y{y}'] * 100:+.0f}%({r[f'n{y}']})" for y in range(2021, 2027))
    print(
        label,
        "trades",
        df.height,
        "avg net %.2f%%" % (df["net"].mean() * 100),
        "|",
        ys,
        "| worst {:.0f}% maxDD {:.0f}%".format(r["worst"] * 100, r["maxdd"] * 100),
    )
    print(
        "   per-year avg trade %:",
        {
            y: round(float(g["net"].mean()) * 100, 1)
            for (y,), g in df.group_by(["year"]).agg(pl.col("net")).iter_rows() and []
        }
        if False
        else df.group_by("year")
        .agg(a=pl.col("net").mean() * 100, n=pl.len(), win=(pl.col("net") > 0).mean())
        .sort("year")
        .rows(),
    )
pl.concat(rows).write_parquet(EX / "lag_trades.parquet")

# non-entries: candidates of the best config's mode/combo below the threshold
ev, side, venue, sg = MODES[best["mode"]]
_, _, sig_all = select(best["mode"], best["combo"], 0.0)
sig_all = sig_all
m = (
    (D["combo"].to_numpy() == best["combo"])
    & (D["side"].to_numpy() == ev)
    & (D["year"].to_numpy() >= 2021)
    & (sig_all < best["thr"])
)
if venue == "perp":
    m &= D["perp_ok"].to_numpy()
idx = np.where(m)[0]
idx = np.sort(np.random.default_rng(1).choice(idx, min(20000, len(idx)), replace=False))
net, kind, mfe, mae, hh = exits(idx, side, venue, best["hold"], best["tp"], best["stop"])
tin = D["t"].to_numpy()[idx]
pl.DataFrame(dict(
    strategy="lag_best", config="nonentry", symbol=D["symbol"].to_numpy()[idx], side=np.full(len(idx), side),  # noqa: E501
    t_in=tin, t_out=tin + hh * 3600, signal=sig_all[idx], threshold=np.full(len(idx), best["thr"]), net=net,  # noqa: E501
    mfe=mfe, mae=mae, tp=np.full(len(idx), best["tp"], dtype=float),
    stop=np.full(len(idx), best["stop"] if best["stop"] is not None else (1.0 if side < 0 else np.nan), dtype=float),  # noqa: E501
    exit_kind=KIND[kind], year=D["year"].to_numpy()[idx],
)).write_parquet(EX / "lag_nonentries.parquet")  # fmt: skip
print("nonentries", len(idx))

# signal lift: avg net by signal decile, 24h hold, no tp/stop, per mode, years 2021+
for mode in ("up_long", "down_short", "bounce_long"):
    ev, side, venue, sg = MODES[mode]
    for combo in (0, 2, 4):
        idx, prio, sig = select(mode, combo, 0.0)
        net, *_ = exits(idx, side, venue, 24, None, None)
        b = np.minimum((sig[idx] * 5).astype(int), 4)
        print(
            mode,
            "combo",
            combo,
            "n",
            len(idx),
            "quintile avg net bp (24h):",
            [int(net[b == k].mean() * 1e4) for k in range(5)],
        )
# event-level view: independent events (one equal-weight basket per event) for the up/down modes
for mode in ("up_long", "down_short", "bounce_long"):
    ev, side, venue, sg = MODES[mode]
    for combo in (0, 2, 4, 5):
        idx, prio, sig = select(mode, combo, 0.0)
        net, *_ = exits(idx, side, venue, 24, None, None)
        e = D["i"].to_numpy()[idx]
        df = pl.DataFrame(dict(e=e, net=net, hi=sig[idx] >= 0.9))
        ev_all = df.group_by("e").agg(a=pl.col("net").mean())
        ev_top = df.filter(pl.col("hi")).group_by("e").agg(a=pl.col("net").mean())
        print(
            mode,
            combo,
            "events",
            ev_all.height,
            "basket avg {:.2f}% (median {:.2f}%) | top10% lag basket {:.2f}%".format(
                ev_all["a"].mean() * 100, ev_all["a"].median() * 100, ev_top["a"].mean() * 100
            ),
        )
