"""Bracket CNN (B1): learn which spikes reach +5% before -2%, walk-forward 2022-2026.

Same data and folds as the walk-forward study (`m12_cnn_wf.py`): for test year Y the model trains
on liquid spikes before Y-1 July, stops early on Y-1 July-December and scores year Y. The target
is the net return of a bracket trade entered at the next candle's open: take profit +5%, stop -2%,
time stop H (stop first when both are touched in one 5m candle), minus 0.3% costs. Protocol:
docs/plans/cnn-replication.md, milestone B1 (pre-registered before training).

Run:
  uv run python research/volume_spike/m17_bracket.py run br1h br1h_w2h ...
  uv run python research/volume_spike/m17_bracket.py eval      # frontier grid, all variants
  uv run python research/volume_spike/m17_bracket.py stress br1h  # costs, stop slip, delay, placebo
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime

import numpy as np
import polars as pl
import torch
from common import RESULTS
from m12_cnn_wf import BATCH, BTC, COIN, DEVICE, MARKET, WF, YEARS, Cfg, Data, make, predict
from portfolio import Exit, exit_path, simulate
from torch import nn

TP, SL = 0.05, 0.02
COST = 0.003
SELECT_YEARS = (2022, 2023, 2024)
HOLD = {"br30m": 6, "br1h": 12, "br2h": 24}


@dataclass(frozen=True)
class BCfg:
    cfg: Cfg
    hold: int  # time stop in 5m candles
    bce: bool = False  # classify "take profit first" instead of regressing the net return
    note: str = ""


_B = Cfg(channels=COIN + BTC, epochs=6)
VARIANTS: dict[str, BCfg] = {
    "br1h": BCfg(_B, 12, note="primary: coin + BTC, 8h window, max hold 1h"),
    "br1h_w2h": BCfg(replace(_B, seq=24), 12, note="2h window"),
    "br1h_mkt": BCfg(replace(_B, channels=COIN + BTC + MARKET), 12, note="+ market channels"),
    "br1h_bce": BCfg(_B, 12, bce=True, note="classify take-profit first"),
    "br2h": BCfg(_B, 24, note="max hold 2h"),
    "br30m": BCfg(_B, 6, note="max hold 30m"),
}


def bracket(fwd: np.ndarray, hold: int, cost: float = COST) -> np.ndarray:
    ret, _ = exit_path(np.nan_to_num(fwd), Exit(tp=TP, sl=SL, bars=hold), cost)
    return ret


def fit_fold(
    data: Data, b: BCfg, y: np.ndarray, year: int, seed: int
) -> tuple[nn.Module, list[dict[str, float]]]:
    """As m12's fit_fold, but early stopping follows the bracket target itself."""
    cfg = b.cfg
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    t = data.meta["open_time"]
    cut_tr = datetime(year - 1, 7, 1, tzinfo=UTC)
    cut_va = datetime(year, 1, 1, tzinfo=UTC)
    tr = np.flatnonzero((t < cut_tr).to_numpy())
    va = np.flatnonzero(((t >= cut_tr) & (t < cut_va)).to_numpy())
    net = make(cfg, sum(data.scal[k].shape[1] for k in cfg.scalars), 1).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    yt = torch.from_numpy(y)
    tp_hit = y > (TP - COST) * 100 - 1e-3
    best, best_state, log, bad = -np.inf, None, [], 0
    for epoch in range(cfg.epochs):
        net.train()
        t0 = time.time()
        perm = rng.permutation(tr)
        for k in range(0, len(perm), BATCH):
            bb = np.sort(perm[k : k + BATCH])
            if len(bb) < 32:
                continue
            x, s = data.batch(bb, cfg)
            o = net(x, s).squeeze(1)
            yb = yt[bb].to(DEVICE)
            if b.bce:
                loss = nn.functional.binary_cross_entropy_with_logits(
                    o, torch.from_numpy(tp_hit[bb]).float().to(DEVICE)
                )
            else:
                loss = nn.functional.mse_loss(o, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sv = predict(net, data, cfg, va)[:, 0]
        corr = float(np.corrcoef(sv, y[va])[0, 1])
        log.append({"year": year, "seed": seed, "epoch": epoch, "valid_corr": corr,
                    "secs": time.time() - t0})  # fmt: skip
        if corr > best:
            best, bad = corr, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 2:
                break
    assert best_state is not None
    net.load_state_dict(best_state)
    return net, log


def run(name: str, data: Data) -> None:
    b = VARIANTS[name]
    y = (bracket(np.asarray(data.fwd), b.hold) * 100).astype(np.float32)
    n = len(y)
    years = data.meta["year"].to_numpy()
    t = data.meta["open_time"]
    pct = np.full(n, np.nan, dtype=np.float32)
    raw = np.full(n, np.nan, dtype=np.float32)
    logs = []
    t_start = time.time()
    for year in YEARS:
        test = np.flatnonzero(years == year)
        train_rows = np.flatnonzero((t < datetime(year - 1, 7, 1, tzinfo=UTC)).to_numpy())
        ref = np.sort(train_rows[:: max(1, len(train_rows) // 50_000)])
        o_test, o_ref = [], []
        for seed in range(b.cfg.seeds):
            net, log = fit_fold(data, b, y, year, seed)
            logs += log
            o_test.append(predict(net, data, b.cfg, test)[:, 0])
            o_ref.append(predict(net, data, b.cfg, ref)[:, 0])
        s_test, s_ref = np.mean(o_test, 0), np.mean(o_ref, 0)
        raw[test] = s_test
        pct[test] = np.searchsorted(np.sort(s_ref), s_test) / len(s_ref)
        print(f"{name} {year}: {len(test)} scored, {time.time() - t_start:.0f}s", flush=True)
    WF.mkdir(exist_ok=True)
    np.savez(WF / f"{name}.npz", pct=pct, raw=raw)
    pl.DataFrame(logs).with_columns(variant=pl.lit(name)).write_csv(WF / f"{name}_log.csv")


# ----------------------------------------------------------------------------- evaluation
def year_stats(
    data: Data, pct: np.ndarray, hold: int, q: float, slots: int, cost: float = COST,
    fwd: np.ndarray | None = None, sl: float = SL,
) -> list[dict[str, object]]:  # fmt: skip
    """5-slot (or `slots`) portfolio per test year, bracket exit, highest score first."""
    out = []
    years = data.meta["year"].to_numpy()
    f_all = data.fwd if fwd is None else fwd
    for year in YEARS:
        k = np.flatnonzero((years == year) & (pct >= q))
        f = np.nan_to_num(np.asarray(f_all[k]))
        sig = data.meta[k].with_columns(neg=pl.Series(-pct[k]))
        res = simulate(sig, f, Exit(tp=TP, sl=sl, bars=hold), slots=slots, cost=cost,
                       priority="neg")  # fmt: skip
        st = res.stats()
        tr = res.trades
        n = st["trades"]
        out.append({
            "year": year, "trades": n, "bp": st.get("mean_bp"),
            "ret": float((tr["stake"] * tr["ret"]).sum()) if n else 0.0,
            "dd": st.get("max_dd"),
            "tp_rate": float((tr["ret"] > TP - cost - 1e-9).mean()) if n else None,
            "sl_rate": float((tr["ret"] < -sl - cost + 1e-9).mean()) if n else None,
            "win": st.get("win"),
        })  # fmt: skip
    return out


def scores() -> dict[str, tuple[np.ndarray, int]]:
    """Percentile scores and max hold of every finished variant, plus the comparisons."""
    out = {}
    for name, b in VARIANTS.items():
        f = WF / f"{name}.npz"
        if f.exists():
            out[name] = (np.nan_to_num(np.load(f)["pct"], nan=-1), b.hold)
    done = [n for n in VARIANTS if n in out]
    if len(done) >= 2:
        ens = np.mean([out[n][0] for n in done], 0)
        out["br_ens"] = (ens, 12)
    for old in ("base", "path_all"):  # PR 8 models, traded with the bracket (no retraining)
        f = WF / f"{old}.npz"
        if f.exists():
            out[f"{old}+bracket"] = (np.nan_to_num(np.load(f)["pct"], nan=-1), 12)
    return out


def evaluate() -> None:
    data = Data()
    rows = []
    for name, (pct, hold) in scores().items():
        for q in (0.9, 0.95, 0.98, 0.99, 0.995):
            for slots in (5, 10, 20):
                for r in year_stats(data, pct, hold, q, slots):
                    rows.append({"variant": name, "q": q, "slots": slots, **r})
        print(name, "done", flush=True)
    df = pl.DataFrame(rows)
    df.write_csv(RESULTS / "m17_bracket_grid.csv")
    d = df.filter((pl.col("q") == 0.98) & (pl.col("slots") == 5))
    wide = d.pivot(on="year", index="variant", values="bp", aggregate_function="first")
    with pl.Config(tbl_rows=40, tbl_cols=20, tbl_width_chars=200, float_precision=0):
        print("bp per trade, top 2%, 5 slots")
        print(wide)
        print(d.pivot(on="year", index="variant", values="trades", aggregate_function="first"))


def delayed(fwd: np.ndarray) -> np.ndarray:
    """The forward path re-based to an entry one candle later (at the first candle's close)."""
    f = np.asarray(fwd, dtype=np.float32)
    c0 = 1 + f[:, 2, :1]
    out = np.full_like(f, np.nan)
    out[:, :, :-1] = (1 + f[:, :, 1:]) / c0[:, :, None] - 1
    return out


def stress(names: list[str], q: float = 0.98, slots: int = 5) -> None:
    data = Data()
    sc = scores()
    late = delayed(data.fwd)
    rng = np.random.default_rng(0)
    day = data.meta["day"].to_physical().to_numpy()
    rows = []
    for name in names:
        pct, hold = sc[name]
        shuffled = pct.copy()
        for d in np.unique(day):  # placebo: same days and counts, scores shuffled within a day
            k = np.flatnonzero(day == d)
            shuffled[k] = rng.permutation(pct[k])
        cases = {
            "default": {},
            "cost 0.6%": {"cost": 0.006},
            "one candle late": {"fwd": late},
        }
        for case, kw in cases.items():
            for r in year_stats(data, pct, hold, q, slots, **kw):  # type: ignore[arg-type]
                rows.append({"variant": name, "case": case, **r})
                if case == "default" and r["trades"]:
                    # the stop triggers at -2% but fills 0.2% lower: same trades, stops cost more
                    slip = {**r, "bp": r["bp"] - 20 * r["sl_rate"]}  # type: ignore[operator]
                    rows.append({"variant": name, "case": "stop fills 0.2% lower", **slip})
        for r in year_stats(data, shuffled, hold, q, slots):
            rows.append({"variant": name, "case": "placebo", **r})
    df = pl.DataFrame(rows)
    df.write_csv(RESULTS / "m17_bracket_stress.csv")
    with pl.Config(tbl_rows=60, tbl_cols=20, tbl_width_chars=200, float_precision=0):
        print(df.pivot(on="year", index=["variant", "case"], values="bp",
                       aggregate_function="first"))  # fmt: skip


if __name__ == "__main__":
    if sys.argv[1] == "run":
        d = Data()
        for v in sys.argv[2:]:
            run(v, d)
    elif sys.argv[1] == "stress":
        stress(sys.argv[2:])
    else:
        evaluate()
