"""X5: reduce sleeve M's stop/squeeze tail without selling the retraces (in-sample exploration).

Re-simulates the weak listing shorts (e27, since_list <= 0) on perp 1h candles with variants of the
exit/entry/size rules, prints a diag scorecard row per variant (share_stopped, contrib_stopped,
E[o], years) and the LMC book (L + M variant + C, 10 slots) with the variant. Every variant is
tried and listed; nothing here is out-of-sample.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from common_ls import OWN
from diag import annotate, scorecard
from e13_listing_book import COST, perp
from e25_combined import book, sleeve_crash_short, sleeve_long
from futures_ls import funding_cum

X = pl.read_parquet(OWN / "e27_listings.parquet").filter(pl.col("since_list") <= 0)
FC = funding_cum()
OUT = Path(__file__).resolve().parent / "results" / "explore"


def fpaid(sym: str, a_t: int, b_t: int) -> float:
    c = FC.get(sym)
    if c is None:
        return 0.0
    t, cs = c
    a = np.searchsorted(t, a_t, side="right") - 1
    b = np.searchsorted(t, b_t, side="right") - 1
    return -((cs[b] if b >= 0 else 0.0) - (cs[a] if a >= 0 else 0.0))  # short pays -rate


def rate3(sym: str, t_end: int) -> float:
    """Mean of the last 3 funding rates settled at or before t_end (nan if unknown)."""
    c = FC.get(sym)
    if c is None:
        return np.nan
    t, cs = c
    j = np.searchsorted(t, t_end, side="right") - 1
    return float((cs[j] - cs[j - 3]) / 3) if j >= 3 else np.nan


def sim(r: dict, cfg: dict) -> dict | None:
    sym = r["symbol"]
    h, o, hi, lo, c, _qv = perp[sym]
    i = int(np.searchsorted(h, r["t_in"]))
    k0 = int(np.searchsorted(h, r["t_in"] - 72 * 3600))
    e0 = o[i]
    j_end = min(int(np.searchsorted(h, h[i] + 30 * 86400)), len(h) - 1)
    pre_lo = lo[k0:i].min() if i > k0 else e0
    if cfg.get("skip_fund_neg") and r["fund"] == r["fund"] and r["fund"] < 0:
        return None
    if cfg.get("skip_bounce") and e0 / pre_lo - 1 >= cfg["skip_bounce"]:
        return None
    w = 1.0
    if cfg.get("volw"):
        lr = np.diff(np.log(c[k0 : i + 1]))
        v = float(lr.std()) if len(lr) > 5 else np.nan
        w = float(np.clip(cfg["vol_med"] / v, 0.5, 1.5)) if v == v and v > 0 else 1.0
    tp, stop = cfg.get("tp", 0.5), cfg.get("stop", 0.5)
    add = cfg.get("add")  # ("dn", X) or ("up", Y)
    legs = [[e0, 0.5 if add else 1.0]]
    added = False
    minlow_e, armed_k, k_exit, kind, xp = e0, None, j_end, "time", c[j_end]
    mfe, mae = 0.0, 0.0
    for k in range(i, j_end + 1):
        up, dn = hi[k] / e0 - 1, lo[k] / e0 - 1
        mae = max(mae, up)
        if up >= stop:  # adverse first
            k_exit, kind, xp = k, "stop" if stop < 1 else "liq", e0 * (1 + stop)
            break
        mfe = max(mfe, -dn)
        if -dn >= tp:
            k_exit, kind, xp = k, "tp", e0 * (1 - tp)
            break
        if add and not added:
            lvl = -add[1] if add[0] == "dn" else add[1]
            if (add[0] == "dn" and dn <= lvl) or (add[0] == "up" and up >= lvl):
                legs.append([e0 * (1 + lvl), 0.5])
                added = True
        tr = cfg.get("trail")  # (kind, param) armed after +25% favourable
        if tr:
            if armed_k is not None:
                if tr[0] == "be" and hi[k] >= e0 * (1 + tr[1]):
                    pass  # handled below (stop level), kept for clarity
                if tr[0] == "trail" and hi[k] >= minlow_e * (1 + tr[1]):
                    k_exit, kind, xp = k, "trail", minlow_e * (1 + tr[1])
                    break
                if tr[0] == "be" and hi[k] >= e0 * (1 + tr[1]):
                    k_exit, kind, xp = k, "trail", e0 * (1 + tr[1])
                    break
                if tr[0] == "time" and k - armed_k >= tr[1]:
                    k_exit, kind, xp = k, "trail", c[k]
                    break
            if -dn >= cfg.get("arm", 0.25) and armed_k is None:
                armed_k = k
            minlow_e = min(minlow_e, lo[k])
        fx = cfg.get("fund_exit")
        if fx is not None:
            rr = rate3(sym, int(h[k]) + 3600)
            if rr == rr and rr < -fx and k > i:
                k_exit, kind, xp = k, "fund", c[k]
                break
    t_out = int(h[k_exit]) + 3600
    ret = sum(wl * -(xp / pl_ - 1) for pl_, wl in legs)
    ncost = sum(wl for _, wl in legs) * COST
    # Both legs are charged funding from the first entry (slightly conservative for the add-on leg).
    fund = sum(wl * fpaid(sym, int(h[i]), t_out) for _, wl in legs)
    net = ret - ncost - fund
    net = max(net, -1.0 - 0.0)
    return {"symbol": sym, "t_in": int(h[i]), "t_out": t_out, "net": net * w, "raw": net, "w": w,
            "adv": min(mae, 1.0), "mfe": mfe, "qv": r["qv"], "year": r["year"],
            "kind": kind, "stop": stop, "full": float(sum(wl for _, wl in legs))}  # fmt: skip


def run(cfg: dict) -> pl.DataFrame:
    rows = [sim(r, cfg) for r in X.iter_rows(named=True)]
    return pl.DataFrame([r for r in rows if r is not None])


def card(t: pl.DataFrame, stop: float | None) -> dict:
    f = t.select(
        signal=pl.lit(1.0), threshold=pl.lit(0.0), net="net", mfe="mfe", mae="adv",
        tp=pl.lit(0.5), stop=pl.lit(stop), exit_kind="kind", year="year",
    )  # fmt: skip
    s = scorecard(annotate(f))
    s["E_all"] = float(f["net"].clip(-1, 1).sum()) / X.height
    return s


def lmc(t: pl.DataFrame, parts: tuple) -> tuple[float, float, str]:
    m = (
        t.filter(pl.col("year") >= 2022)
        .select("symbol", "adv", "t_in", "t_out", net="raw", qv="qv", w="w",
                sleeve=pl.lit("N"), prio=pl.lit(1.0))
        .with_columns(w=pl.col("w"))
    )  # fmt: skip
    cols = ["symbol", "adv", "t_in", "t_out", "net", "qv", "sleeve", "prio", "w"]
    fr = [p.with_columns(w=pl.lit(1.0)).select(cols) for p in parts]
    b = book(pl.concat([fr[0], m.select(cols), fr[1]], how="vertical_relaxed"), 10)
    n = int(b["trades"].sum())
    bp = float((b["bp"] * b["trades"]).sum() / n)
    yrs = "/".join(f"{r['ret'] * 100:+.0f}" for r in b.iter_rows(named=True))
    return bp, float(b["ret"].min()), yrs


if __name__ == "__main__":
    Lg, Cs = sleeve_long(), sleeve_crash_short()
    base0 = run({})
    chk = pl.read_parquet(OWN / "e27_listings.parquet").filter(pl.col("since_list") <= 0)
    print(
        "base check: sim mean net",
        base0["raw"].mean(),
        "e27 mean net",
        chk["net"].mean(),
        base0.height,
    )
    # vol median of the base set
    vols = []
    for r in X.iter_rows(named=True):
        h, o, hi, lo, c, qv = perp[r["symbol"]]
        i = int(np.searchsorted(h, r["t_in"]))
        k0 = int(np.searchsorted(h, r["t_in"] - 72 * 3600))
        vols.append(np.diff(np.log(c[k0 : i + 1])).std())
    vm = float(np.nanmedian(vols))
    V: dict[str, dict] = {
        "V0 base (stop 50)": {},
        "stop 75": {"stop": 0.75},
        "stop 100 (liq only)": {"stop": 1.0},
        "add dn10 (half+half)": {"add": ("dn", 0.10)},
        "add dn20": {"add": ("dn", 0.20)},
        "add up10": {"add": ("up", 0.10)},
        "add up20": {"add": ("up", 0.20)},
        "fund exit <-0.02%": {"fund_exit": 0.0002},
        "fund exit <-0.05%": {"fund_exit": 0.0005},
        "fund exit <-0.10%": {"fund_exit": 0.0010},
        "skip fund<0 at entry": {"skip_fund_neg": True},
        "skip bounce>=20%": {"skip_bounce": 0.20},
        "skip bounce>=30%": {"skip_bounce": 0.30},
        "skip bounce>=50%": {"skip_bounce": 0.50},
        "trail: BE stop after +25%": {"trail": ("be", 0.0)},
        "trail: stop at -10% after +25%": {"trail": ("be", -0.10)},
        "trail: 25% above low after +25%": {"trail": ("trail", 0.25)},
        "trail: 40% above low after +25%": {"trail": ("trail", 0.40)},
        "trail: exit 72h after +25%": {"trail": ("time", 72)},
        "trail: exit 168h after +25%": {"trail": ("time", 168)},
        "vol-scaled size": {"volw": True, "vol_med": vm},
        "vol-scaled + stop 75": {"volw": True, "vol_med": vm, "stop": 0.75},
        "stop75 + skip bounce30": {"stop": 0.75, "skip_bounce": 0.30},
        "stop75 + fund exit 0.05%": {"stop": 0.75, "fund_exit": 0.0005},
        "stop75 + skip fund<0": {"stop": 0.75, "skip_fund_neg": True},
    }
    lines = [
        "| variant | n | E_o | E_all (per 143) | win | share_stopped | contrib_stopped | mean w | years>0 | worst yr E_o | LMC bp | LMC min yr % | LMC years % 22..26 |",  # noqa: E501
        "|" + "---|" * 13,
    ]
    for name, cfg in V.items():
        t = run(cfg)
        s = card(t, cfg.get("stop", 0.5) if cfg.get("stop", 0.5) < 1 else None)
        bp, mn, yrs = lmc(t, (Lg, Cs))
        row = (
            f"| {name} | {s['n']:.0f} | {s['E_o']:+.3f} | {s['E_all']:+.3f} | {s['win']:.2f} | {s['share_stopped']:.2f} | "  # noqa: E501
            f"{s['contrib_stopped']:+.3f} | {t['w'].mean():.2f} | {s['years_positive']:.2f} | {s['worst_year_E_o']:+.3f} | {bp:+.0f} | {mn * 100:+.0f} | {yrs} |"  # noqa: E501
        )
        lines.append(row)
        print(row, flush=True)
    (OUT / "x5_m_tail.md").write_text("\n".join(lines) + "\n")
