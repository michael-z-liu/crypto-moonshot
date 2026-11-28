"""w4 C: is the 72h entry-delay peak noise? Coin-level paired bootstrap, per-year dispersion, E36 waiting."""  # noqa: E501

from __future__ import annotations

import numpy as np
import polars as pl
from e13_listing_book import L, perp, trades

rng = np.random.default_rng(7)
DEL = [12, 24, 36, 48, 60, 66, 72, 78, 84, 96, 120, 144, 168]
tab: dict[int, dict[str, float]] = {}
yr: dict[str, int] = {}
for dly in DEL:
    d = trades(dly, 30, 0.5, 0.5)
    since = []
    for r in d.iter_rows(named=True):
        h, o = perp[r["symbol"]][0], perp[r["symbol"]][1]
        lst = int(L.filter(pl.col("symbol") == r["symbol"])["listed"][0].timestamp())
        k0 = np.searchsorted(h, lst)
        i = np.searchsorted(h, r["t_in"])
        since.append(o[i] / o[k0] - 1)
        yr[r["symbol"]] = r["year"]
    d = d.with_columns(since=pl.Series(since)).with_columns(weak=pl.col("since") <= 0)
    tab[dly] = {
        "all": dict(zip(d["symbol"], d["net"], strict=False)),
        "weak": dict(zip(d.filter("weak")["symbol"], d.filter("weak")["net"], strict=False)),
    }  # type: ignore[dict-item]

syms = sorted(yr)
print("delay | ALL n mean (se) win | WEAK n mean (se) med win | SL-share(weak)")
for dly in DEL:
    a = np.array(list(tab[dly]["all"].values()))
    w = np.array(list(tab[dly]["weak"].values()))
    print(
        f"{dly:4d}  all n={len(a)} {a.mean():+.3f} (se {a.std() / np.sqrt(len(a)):.3f}) win {np.mean(a > 0):.2f} | weak n={len(w)} {w.mean():+.3f} (se {w.std() / np.sqrt(len(w)):.3f}) med {np.median(w):+.3f} win {np.mean(w > 0):.2f} sl {np.mean(w <= -0.5):.2f}"  # noqa: E501
    )


def coin_mat(key: str, delays: list[int]) -> np.ndarray:
    m = np.full((len(syms), len(delays)), np.nan)
    for j, dly in enumerate(delays):
        for i, s in enumerate(syms):
            v = tab[dly][key].get(s)
            if v is not None:
                m[i, j] = v
    return m


for key in ("all", "weak"):
    core = [24, 48, 72, 96, 120]
    M = coin_mat(key, core)
    # coin-level bootstrap: mean per delay over resampled coins (each delay uses its own available coins)  # noqa: E501
    arg = np.zeros(len(core))
    diffs = {c: [] for c in core}
    B = 5000
    for _ in range(B):
        ix = rng.integers(0, len(syms), len(syms))
        mu = np.nanmean(M[ix], axis=0)
        arg[np.nanargmax(mu)] += 1
        for k, c in enumerate(core):
            diffs[c].append(mu[2] - mu[k])
    print(
        f"\n[{key}] bootstrap over coins: P(argmax delay) =",
        dict(zip(core, np.round(arg / B, 3), strict=False)),
    )
    for c in core:
        if c == 72:
            continue
        x = np.array(diffs[c])
        print(
            f"   mean(72h) - mean({c}h): point {np.nanmean(M[:, 2]) - np.nanmean(M[:, core.index(c)]):+.3f}  boot 5-95% [{np.percentile(x, 5):+.3f}, {np.percentile(x, 95):+.3f}]  P(>0)={np.mean(x > 0):.2f}"  # noqa: E501
        )
    # paired on the common coin set only
    com = np.all(np.isfinite(M), axis=1)
    print(
        f"   common-coin set n={com.sum()}: means by delay",
        dict(zip(core, np.round(M[com].mean(0), 3), strict=False)),
    )
    # per year 72h minus mean of neighbours (48,96)
    ys = np.array([yr[s] for s in syms])
    for y in sorted(set(ys)):
        mm = M[(ys == y)]
        print(
            f"   {y}: n72={np.isfinite(mm[:, 2]).sum():3d} "
            + " ".join(f"d{c}:{np.nanmean(mm[:, k]):+.2f}" for k, c in enumerate(core))
        )

# local smoothness: neighbours 66/72/78 on weak
print(
    "\nfine grid (weak): ",
    {d_: round(float(np.mean(list(tab[d_]["weak"].values()))), 3) for d_ in DEL},
)
print(
    "fine grid (all):  ",
    {d_: round(float(np.mean(list(tab[d_]["all"].values()))), 3) for d_ in DEL},
)

# randomisation check: how large is the max-min spread across 5 delays if delay were irrelevant?
M = coin_mat("weak", [24, 48, 72, 96, 120])
obs = np.nanmean(M, axis=0)
print(
    "\nobserved weak means [24,48,72,96,120]:",
    np.round(obs, 3),
    "spread (max - mean of others):",
    round(obs.max() - np.delete(obs, obs.argmax()).mean(), 3),
)
# permute delay labels within coin (keeps coin effect, removes delay effect)
sp = []
Mf = M.copy()
for _ in range(5000):
    P = Mf.copy()
    for i in range(P.shape[0]):
        ok = np.flatnonzero(np.isfinite(P[i]))
        P[i, ok] = rng.permutation(P[i, ok])
    mu = np.nanmean(P, axis=0)
    sp.append(mu.max() - np.delete(mu, mu.argmax()).mean())
sp = np.array(sp)
print(
    f"permutation null: spread (max - mean others) median {np.median(sp):.3f} p95 {np.percentile(sp, 95):.3f}; P(null >= observed) = {np.mean(sp >= obs.max() - np.delete(obs, obs.argmax()).mean()):.3f}"  # noqa: E501
)
