"""Data for the visual report (M8): every method recomputed per year on one basis, plus chart data.

Every method is scored the same way here: every qualifying trade, at most one open trade per coin,
entry at the next candle's open, 0.3% round-trip cost. That differs from the slot-limited
portfolio numbers (portfolio.py), which take at most 5 positions at a time.

Run after build_events.py, build_seq.py and m11_cnn.py:
    uv run python research/volume_spike/report_data.py   # writes report/data.json
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Any

import numpy as np
import polars as pl
from common import CACHE, HERE, RESULTS, load
from evalx import COST, load_events, triple_barrier
from m3_rules import non_overlapping

OUT = HERE / "report" / "data.json"
MAJORS = {
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "TRXUSDT",
    "LINKUSDT", "AVAXUSDT", "DOTUSDT", "LTCUSDT", "BCHUSDT", "XLMUSDT", "ATOMUSDT",
}  # fmt: skip
FLUSH = (pl.col("btc_vol_mult") >= 6) | (pl.col("btc_ret_4h") <= -0.02)


def rows(df: pl.DataFrame) -> list[dict[str, Any]]:
    return [{k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in r.items()}
            for r in df.iter_rows(named=True)]  # fmt: skip


def csv(name: str) -> list[dict[str, Any]]:
    path = RESULTS / name
    return rows(pl.read_csv(path)) if path.exists() else []


def main() -> None:
    ev, fwd = load_events()
    n = ev.height
    print("returns")
    close = np.asarray(fwd[:, 2, :], dtype=np.float32)
    r1h = close[:, 11].astype(np.float64) - COST
    r4h = close[:, 47].astype(np.float64) - COST
    r24h = ev["fwd_24h"].fill_null(np.nan).to_numpy() - COST
    r2019 = triple_barrier(fwd, 0.01, 0.05, 48)
    spike = pl.col("is_spike")
    liq5 = pl.col("qv_base") >= 5_000

    preds: dict[str, np.ndarray] = {}
    meta = pl.read_parquet(CACHE / "seq_meta.parquet")
    spike_rows = meta["event"].to_numpy()
    tr = meta["split"].to_numpy() == "train"
    for name, file in (("gbm", "gbm_pred.npy"), ("cnn", "cnn_pred_chosen.npy")):
        if (CACHE / file).exists():
            p = np.full(n, np.nan)
            pp = np.load(CACHE / file)
            p[spike_rows] = pp
            preds[name] = p
            ev = ev.with_columns(pl.Series(f"pred_{name}", p).fill_nan(None))
            ev = ev.with_columns(pl.lit(float(np.quantile(pp[tr], 0.995))).alias(f"th_{name}"))

    # name -> (description, mask, returns, hold bars, category)
    methods: dict[str, tuple[str, pl.Expr, np.ndarray, int, str]] = {
        "random": ("Random 5m candle, hold 1h", pl.col("is_random"), r1h, 12, "baseline"),
        "all_spikes": ("Every volume spike (>= 10x), hold 1h", spike, r1h, 12, "baseline"),
        "rule2019": ("2019 rule: up candle, taker-buy > 60%, flat last hour; +1% / -5% / 4h",
                     spike & (pl.col("ret_bar") > 0) & (pl.col("tbr") > 0.6)
                     & (pl.col("runup_1h").abs() < 0.01), r2019, 48, "original"),
        "pump_long": ("Spike after a pump (coin +10% in 4h), buy, hold 4h",
                      spike & liq5 & (pl.col("ret_4h") >= 0.10), r4h, 48, "pump"),
        "pump_short": ("Same pump spikes, short (hypothetical on spot), hold 4h",
                       spike & liq5 & (pl.col("ret_4h") >= 0.10), -(r4h + COST) - COST, 48,
                       "pump"),
        "breakout": ("Breakout: up-candle spike at a new 7-day high, hold 24h",
                     spike & liq5 & (pl.col("ret_bar") > 0) & (pl.col("dist_hi_7d") >= 0),
                     r24h, 288, "trend"),
        "pullback": ("Uptrend pullback: coin +20% in 7d, -3% in 4h, spike, hold 24h",
                     spike & liq5 & (pl.col("ret_7d") >= 0.20) & (pl.col("ret_4h") <= -0.03),
                     r24h, 288, "trend"),
        "crash_alone": ("Coin crash with calm BTC: -10% in 4h, >= 20k USDT/bar, hold 1h",
                        spike & (pl.col("ret_4h") <= -0.10) & (pl.col("qv_base") >= 20_000)
                        & ~FLUSH.fill_null(False), r1h, 12, "capitulation"),
        "F1": ("F1: coin -8% in 4h, >= 5k USDT/bar, BTC volume >= 6x, hold 1h",
               spike & (pl.col("ret_4h") <= -0.08) & liq5 & (pl.col("btc_vol_mult") >= 6), r1h, 12,
               "capitulation"),
        "F2": ("F2: coin -10% in 4h, >= 20k USDT/bar, BTC volume >= 6x, hold 1h",
               spike & (pl.col("ret_4h") <= -0.10) & (pl.col("qv_base") >= 20_000)
               & (pl.col("btc_vol_mult") >= 6), r1h, 12, "capitulation"),
        "F3": ("F3: coin -15% in 4h, >= 5k USDT/bar, no BTC filter, hold 4h",
               spike & (pl.col("ret_4h") <= -0.15) & liq5, r4h, 48, "capitulation"),
        "majors_flush": ("Majors only (15 large coins): -5% in 4h during a BTC flush, hold 1h",
                         spike & pl.col("symbol").is_in(MAJORS) & (pl.col("ret_4h") <= -0.05)
                         & FLUSH.fill_null(False), r1h, 12, "capitulation"),
    }  # fmt: skip
    for name in preds:
        methods[f"{name}_top"] = (
            f"{name.upper()} model: top 0.5% of predicted 1h return (2020-23 threshold), hold 1h",
            pl.col(f"pred_{name}") >= pl.col(f"th_{name}"), r1h, 12, "model",
        )  # fmt: skip

    print("methods")
    yearly, cum, quarterly, summary = [], {}, [], []
    base = ev.select("symbol", "open_time", "split").with_columns(
        year=pl.col("open_time").dt.year(), day=pl.col("open_time").dt.date()
    )
    for name, (desc, cond, ret, hold, cat) in methods.items():
        m = ev.select(cond.fill_null(False).alias("m"))["m"].to_numpy() & np.isfinite(ret)
        idx = np.flatnonzero(m)
        d = base[idx].with_columns(ret=pl.Series(ret[idx]))
        d = d.filter(
            pl.Series(non_overlapping(d["symbol"].to_numpy(), d["open_time"].to_numpy(), hold))
        )
        days = d.group_by("day").agg(pl.col("ret").mean(), pl.len().alias("k")).sort("day")
        y = (
            d.group_by("year").agg(trades=pl.len(), mean_bp=pl.col("ret").mean() * 1e4,
                                   win=(pl.col("ret") > 0).mean())
            .join(days.group_by(pl.col("day").dt.year().alias("year"))
                  .agg(days=pl.len(), day_bp=pl.col("ret").mean() * 1e4,
                       win_days=(pl.col("ret") > 0).mean()), on="year")
            .sort("year")
        )  # fmt: skip
        yearly += [{"method": name, **r} for r in rows(y)]
        cum[name] = {"day": [str(x) for x in days["day"]],
                     "cum": (days["ret"].cum_sum() * 100).round(2).to_list()}  # fmt: skip
        q = days.group_by(pl.col("day").dt.truncate("1q").alias("q")).agg(
            day_bp=pl.col("ret").mean() * 1e4, days=pl.len()
        ).sort("q")  # fmt: skip
        quarterly += [{"method": name, "q": str(r["q"]), "day_bp": r["day_bp"], "days": r["days"]}
                      for r in q.iter_rows(named=True)]  # fmt: skip
        per = d.group_by("split").agg(trades=pl.len(), mean_bp=pl.col("ret").mean() * 1e4,
                                      win=(pl.col("ret") > 0).mean())  # fmt: skip
        summary.append({"method": name, "desc": desc, "category": cat, "hold_bars": hold,
                        **{f"{r['split']}_{k}": r[k] for r in per.iter_rows(named=True)
                           for k in ("trades", "mean_bp", "win")}})  # fmt: skip

    print("paths")
    sp = ev["is_spike"].to_numpy()
    r4 = ev["ret_4h"].fill_null(0).to_numpy()
    q5 = (ev["qv_base"].fill_null(0) >= 5_000).to_numpy()
    flush = ev.select(FLUSH.fill_null(False).alias("f"))["f"].to_numpy()
    late = (ev["split"] == "test").to_numpy()
    # 2025-10-10 (80-99% wicks on some alts) swamps any average; the shape charts leave it out
    cascade = (ev["open_time"].dt.date() == date(2025, 10, 10)).to_numpy()
    rule = methods["rule2019"][1].fill_null(False)
    groups = {
        "random": ev["is_random"].to_numpy(),
        "all spikes": sp,
        "2019 rule entries": ev.select(rule.alias("m"))["m"].to_numpy(),
        "after a pump (+10% in 4h)": sp & q5 & (r4 >= 0.10),
        "crash, BTC calm": sp & q5 & (r4 <= -0.10) & ~flush,
        "crash during a BTC flush": sp & q5 & (r4 <= -0.10) & flush,
    }  # fmt: skip
    paths = []
    for g, m in groups.items():
        for period, pm in (("2020-24", ~late), ("2025-26", late)):
            k = np.flatnonzero(m & pm & ~cascade)
            avg = np.clip(close[k], -0.2, 0.2).mean(0) * 1e4
            paths.append({"group": g, "period": period, "n": len(k),
                          "bp": [0.0, *avg.round(1).tolist()]})  # fmt: skip

    print("depth")
    buckets = [-1, -0.2, -0.15, -0.1, -0.08, -0.05, -0.03, -0.01, 0.01, 0.03, 0.05, 0.1, 0.15, 10]
    labels = ["< -20%", "-20..-15", "-15..-10", "-10..-8", "-8..-5", "-5..-3", "-3..-1", "-1..+1",
              "+1..+3", "+3..+5", "+5..+10", "+10..+15", "> +15%"]  # fmt: skip
    dd = ev.filter(spike & liq5).with_columns(
        r=pl.Series(np.clip(r1h[np.flatnonzero(sp & q5)], -0.2, 0.2)),
        b=pl.col("ret_4h").cut(buckets[1:-1], labels=labels),
        flush=FLUSH.fill_null(False),
        period=pl.when(pl.col("split") == "test").then(pl.lit("2025-26"))
        .otherwise(pl.lit("2020-24")),
    )  # fmt: skip
    depth = rows(
        dd.filter(pl.col("open_time").dt.date() != date(2025, 10, 10))
        .group_by("b", "flush", "period").agg(n=pl.len(), mean_bp=pl.col("r").mean() * 1e4)
        .with_columns(pl.col("b").cast(pl.String)).sort("b", "flush", "period")
    )  # fmt: skip

    print("examples")
    examples = []
    pick = {
        "2019 rule entry that hit the -5% stop": ev.with_columns(r=pl.Series(r2019)).filter(
            methods["rule2019"][1].fill_null(False) & (pl.col("r") < -0.05)
            & (pl.col("qv_base") >= 20_000) & (pl.col("split") == "valid")).sort("open_time")[60],
        "2019 rule entry that took the +1%": ev.with_columns(r=pl.Series(r2019)).filter(
            methods["rule2019"][1].fill_null(False) & (pl.col("r") > 0)
            & (pl.col("qv_base") >= 20_000) & (pl.col("split") == "valid")).sort("open_time")[60],
        "F2 flush bounce, 2021-05-19": ev.with_columns(r=pl.Series(r1h)).filter(
            methods["F2"][1].fill_null(False)
            & (pl.col("open_time").dt.date() == pl.date(2021, 5, 19))).sort("r")[-1],
        "F2 signal that kept falling, 2025": ev.with_columns(r=pl.Series(r1h)).filter(
            methods["F2"][1].fill_null(False) & (pl.col("open_time").dt.year() == 2025)
            & (pl.col("qv_base") >= 50_000)).sort("r")[0],
    }  # fmt: skip
    for title, r in pick.items():
        sym, t = r["symbol"][0], r["open_time"][0]
        raw = load(sym).filter(
            (pl.col("open_time") > t - timedelta(minutes=5 * 96))
            & (pl.col("open_time") <= t + timedelta(minutes=5 * 48))
        )  # fmt: skip
        examples.append({
            "title": title, "symbol": sym, "time": t.isoformat(), "ret_bp": r["r"][0] * 1e4,
            "ret_4h": r["ret_4h"][0], "vol_mult": r["vol_mult"][0], "tbr": r["tbr"][0],
            "btc_vol_mult": r["btc_vol_mult"][0],
            "t": [x.isoformat() for x in raw["open_time"]],
            **{c: raw[c].to_list() for c in ("open", "high", "low", "close", "volume")},
            "taker_buy": raw["taker_buy_base_volume"].to_list(),
        })  # fmt: skip

    print("model extras")
    extras: dict[str, Any] = {}
    if "cnn" in preds:
        seq = np.load(CACHE / "seq.npy", mmap_mode="r")
        pc = np.load(CACHE / "cnn_pred_chosen.npy")
        top = np.flatnonzero(pc >= np.quantile(pc[tr], 0.995))
        bot = np.flatnonzero(pc <= np.quantile(pc[tr], 0.5))
        rng = np.random.default_rng(0)
        prof = {}
        for g, k in (
            ("top 0.5% CNN score", top),
            ("below-median CNN score", rng.choice(bot, 20_000, replace=False)),
        ):
            s = np.asarray(seq[np.sort(k)], dtype=np.float32)
            prof[g] = {"close_bp": (s[:, 3].mean(0) * 1e4).round(1).tolist(),
                       "volume": s[:, 4].mean(0).round(3).tolist(),
                       "taker": (s[:, 5].mean(0) + 0.5).round(4).tolist(),
                       "btc_bp": (s[:, 6].mean(0) * 1e4).round(1).tolist(),
                       "n": len(k)}  # fmt: skip
        extras["cnn_profiles"] = prof
        if "gbm" in preds:
            pg = np.load(CACHE / "gbm_pred.npy")
            extras["cnn_gbm_corr"] = float(np.corrcoef(pc, pg)[0, 1])
            tg = pg >= np.quantile(pg[tr], 0.995)
            tc = pc >= np.quantile(pc[tr], 0.995)
            extras["top_overlap"] = float((tg & tc).sum() / max(tc.sum(), 1))
        cm = meta.with_columns(pred=pl.Series(pc))
        top_m = cm.filter(pl.col("pred") >= np.quantile(pc[tr], 0.995))
        extras["cnn_top_traits"] = rows(top_m.group_by("split").agg(
            n=pl.len(), median_ret_4h=pl.col("ret_4h").median(),
            share_btc_vol6=(pl.col("btc_vol_mult") >= 6).mean(),
            median_qv=pl.col("qv_base").median()).sort("split"))  # fmt: skip

    universe = {
        "coins": ev["symbol"].n_unique(), "spikes": int(sp.sum()),
        "random": int(ev["is_random"].sum()),
        "first": str(ev["open_time"].min()), "last": str(ev["open_time"].max()),
        "spikes_by_split": rows(ev.filter(spike).group_by("split").len().sort("split")),
    }  # fmt: skip
    data = {
        "universe": universe, "summary": summary, "yearly": yearly, "cum": cum,
        "quarterly": quarterly, "paths": paths, "depth": depth, "examples": examples,
        "extras": extras,
        "exit_grid": csv("m2_baseline_grid.csv"), "deciles_spike": csv("m3_deciles_spike.csv"),
        "deciles_random": csv("m3_deciles_random.csv"), "robust": csv("m4_robust.csv"),
        "finalists": csv("m6_test.csv"), "walkforward": csv("m7_walkforward_t_30.csv"),
        "limit": csv("m8_limit_entry.csv"), "daily": csv("m9_daily.csv"),
        "gbm": csv("m10_model.csv"), "cnn_training": csv("m11_cnn_training.csv"),
        "cnn_selection": csv("m11_cnn_selection.csv"), "cnn_f1": csv("m11_cnn_f1_filter.csv"),
        "cnn_portfolio": csv("m11_cnn_portfolio.csv"), "cnn_stress": csv("m11_cnn_stress.csv"),
        "equity": {},
    }  # fmt: skip
    print("model portfolios")
    from portfolio import Exit, simulate

    for name in preds:
        k = np.flatnonzero(ev.select((pl.col(f"pred_{name}") >= pl.col(f"th_{name}"))
                                     .fill_null(False).alias("m"))["m"].to_numpy())  # fmt: skip
        e2 = ev[k].with_columns(neg=-pl.col(f"pred_{name}"))
        res = simulate(e2, np.asarray(fwd[k]), Exit(bars=12), slots=5, priority="neg")
        data["equity"][f"{name.upper()} top 0.5%"] = {
            "day": [str(x) for x in res.equity["day"]],
            "equity": res.equity["equity"].round(4).to_list(),
        }
    for f in ("F1", "F2", "F3"):
        t = pl.read_csv(RESULTS / f"trades_{f}.csv").with_columns(
            d=pl.col("exit_time").str.slice(0, 10)
        )
        e = t.group_by("d").agg((pl.col("stake") * pl.col("ret")).sum()).sort("d")
        data["equity"][f] = {"day": e["d"].to_list(),
                             "equity": (1 + e["stake"].cum_sum()).round(4).to_list()}  # fmt: skip
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(data, default=str, separators=(",", ":")))
    print(OUT, OUT.stat().st_size // 1024, "KB")
    build_html()


def build_html() -> None:
    """Inline the data and the CNN section into report/index.html (one self-contained page)."""
    d = OUT.parent
    cnn = "\n".join(
        (d / f).read_text()
        for f in ("cnn_text.js", "cnn_section.js", "wf_section.js", "surv_section.js")
        if (d / f).exists()
    )
    html = (d / "template.html").read_text()
    html = html.replace("__DATA__", OUT.read_text()).replace("/*CNN_SECTION*/", cnn)
    (d / "index.html").write_text(html)
    print(d / "index.html", len(html) // 1024, "KB")


if __name__ == "__main__":
    main()
