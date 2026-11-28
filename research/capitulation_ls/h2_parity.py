"""H2 (holdout, BTC pairs): how much does the BTC-to-USDT conversion change L's outcome?

Takes h1's USDT-pair spikes of 2017-19 whose coin also traded against BTC at the time and whose
day5 pct >= 0.976 (the upper calibration bins, which contain L's entries), converts the coin's
BTC pair for the spike month and the next one exactly as h2_fetch.py does (`h2_fetch.convert`,
into the separate store holdout/klines_btc_overlap/, zips in the new folder raw_btc_overlap/
and deleted), rebuilds the same forward paths from the converted series (`build_seq2.fwd_job`,
`build_fwd72.job`, same entry time: the next 5m open), and runs L's exit (`sim.exits`, TP 20%,
72h) on both the native USDT path and the converted one.

Writes holdout/btc/parity.parquet and parity.txt.
Run: uv run --group research python h2_parity.py
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor

import h1_build
from h1_build import HOLDOUT, T0_H

LO = 0.976

import build_fwd72  # noqa: E402
import build_seq2  # noqa: E402
import common  # noqa: E402
import h2_fetch  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from sim import Rule, exits  # noqa: E402

OVL = HOLDOUT / "klines_btc_overlap"
OUT = HOLDOUT / "btc"


def paths(f2: np.ndarray, f72: np.ndarray) -> np.ndarray:
    return np.concatenate([f2, f72[:, :, 8:]], axis=2).astype(np.float32)


def main() -> None:
    meta = pl.read_parquet(HOLDOUT / "seq2_meta.parquet").with_row_index("row")
    sc = pl.read_parquet(HOLDOUT / "scores.parquet")
    btc_months: dict[str, list[str]] = json.loads((HOLDOUT / "symbols_btc.json").read_text())
    m = meta.with_columns(pct=sc["pct_day5"], month=pl.col("open_time").dt.strftime("%Y-%m"))
    m = m.filter(
        (pl.col("open_time").dt.year() < 2020) & pl.col("pct").is_finite() & (pl.col("pct") >= LO)
    ).with_columns(bsym=pl.col("symbol").str.replace(r"USDT$", "BTC"))
    m = m.filter(
        pl.struct("bsym", "month").map_elements(
            lambda r: r["month"] in btc_months.get(r["bsym"], []), return_dtype=pl.Boolean
        )
    )
    print(m.height, "spikes with a BTC pair in the same month", file=sys.stderr)
    jobs = sorted(
        {(s, mo) for s, mo0 in zip(m["bsym"], m["month"], strict=True)
         for mo in (mo0, h2_fetch.next_month(mo0)) if mo in btc_months[s]}
    )  # fmt: skip
    print(len(jobs), "BTC-pair months to fetch", file=sys.stderr)
    raw = HOLDOUT / "raw_btc_overlap"
    raw.mkdir(parents=True, exist_ok=True)
    assert not any(raw.iterdir()), "raw_btc_overlap must start empty"
    h2_fetch.STORE, h2_fetch.RAW, h2_fetch.BTC = OVL, raw, h2_fetch.btcusdt()
    with ThreadPoolExecutor(12) as pool:
        res = list(pool.map(h2_fetch.fetch_month, jobs))
    bad = [r for r in res if isinstance(r[2], str) and r[2] != "exists"]
    print(len(bad), "fetch problems", bad[:5], file=sys.stderr)
    assert not any(raw.iterdir())
    raw.rmdir()

    common.ARCHIVE_5M = OVL / "klines" / "spot" / "interval=5m"
    have = {p.name.removeprefix("symbol=") for p in common.ARCHIVE_5M.glob("symbol=*")}
    m = m.filter(pl.col("bsym").is_in(list(have)))
    n = m.height
    gidx = h1_build.gidx_of(m)
    assert h1_build.T0_H == T0_H
    f2p, f72p = OUT / "parity_fwd2.npy", OUT / "parity_fwd72.npy"
    del_ = [np.lib.format.open_memmap(f2p, mode="w+", dtype=np.float32, shape=(n, 3, 96)),
            np.lib.format.open_memmap(f72p, mode="w+", dtype=np.float32, shape=(n, 3, 72))]  # fmt: skip  # noqa: E501
    del del_
    syms = m["bsym"].to_numpy()
    for s in np.unique(syms):
        k = np.flatnonzero(syms == s)
        build_seq2.fwd_job((s, gidx[k], k, str(f2p)))
        build_fwd72.job((s, gidx[k], k, str(f72p)))
    conv = paths(np.load(f2p), np.load(f72p))
    rows = m["row"].to_numpy()
    nat = paths(np.load(HOLDOUT / "fwd2.npy", mmap_mode="r")[rows],
                np.load(HOLDOUT / "fwd72.npy", mmap_mode="r")[rows])  # fmt: skip
    ok = np.isfinite(conv).all(axis=(1, 2)) & np.isfinite(nat).all(axis=(1, 2))
    rule = Rule(hold_min=4320, tp=0.2)
    side = np.ones(n)
    r_n, _, k_n = exits(nat, side, rule)
    r_c, _, k_c = exits(conv, side, rule)
    out = m.select("symbol", "open_time", "pct", "qv_base").with_columns(
        net_native=pl.Series(r_n - 0.004), net_conv=pl.Series(r_c - 0.004),
        tp_native=pl.Series(k_n == 2), tp_conv=pl.Series(k_c == 2),
        mfe_native=pl.Series(np.nanmax(nat[:, 0, :], axis=1)),
        mfe_conv=pl.Series(np.nanmax(conv[:, 0, :], axis=1)), ok=pl.Series(ok),
    )  # fmt: skip
    out.write_parquet(OUT / "parity.parquet")
    lines = [f"spikes compared {int(ok.sum())} of {n} (pct >= {LO}, coin also on a BTC pair)"]
    o = out.filter(pl.col("ok"))
    for lab, d in (("pct >= 0.976", o), ("pct >= 0.995 (L)", o.filter(pl.col("pct") >= 0.995))):
        if d.height == 0:
            continue
        dn = (d["net_conv"] - d["net_native"]).to_numpy()
        lines.append(
            f"{lab}: n {d.height}, mean net native {d['net_native'].mean() * 1e4:+.0f} bp,"
            f" converted {d['net_conv'].mean() * 1e4:+.0f} bp; TP rate native"
            f" {d['tp_native'].mean():.3f}, converted {d['tp_conv'].mean():.3f}; TP agreement"
            f" {(d['tp_native'] == d['tp_conv']).mean():.3f}; per-spike net diff mean"
            f" {dn.mean() * 1e4:+.0f} bp, median |diff| {np.median(np.abs(dn)) * 1e4:.0f} bp, p95"
            f" |diff| {np.percentile(np.abs(dn), 95) * 1e4:.0f} bp; mfe conv - native median"
            f" {(d['mfe_conv'] - d['mfe_native']).median() * 1e4:+.0f} bp, mean"
            f" {(d['mfe_conv'] - d['mfe_native']).mean() * 1e4:+.0f} bp"
        )
    (OUT / "parity.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
