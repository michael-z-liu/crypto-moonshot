"""M4: robustness of the capitulation candidates: costs, late entry, slots, years, coin age."""

from __future__ import annotations

import numpy as np
import polars as pl
from common import RESULTS
from portfolio import Exit, load_long, simulate

CANDIDATES = {
    "C1": (pl.col("ret_4h") <= -0.15) & (pl.col("qv_base") >= 5_000),
    "C2": (pl.col("ret_4h") <= -0.10) & (pl.col("qv_base") >= 20_000),
}
EXIT = Exit(bars=48)


def delayed(fwd: np.ndarray, bars: int) -> np.ndarray:
    """Paths as if entering at the close of the first bar after the signal instead of its open."""
    base = 1 + fwd[:, 2, :1]
    out = (1 + fwd[:, :, 1:]) / base[:, None, :] - 1
    return np.concatenate([out, out[:, :, -1:]], axis=2)[:, :, : max(bars, 1) + 288]


def run(ev: pl.DataFrame, fwd: np.ndarray, mask: np.ndarray, **kw: object) -> dict[str, float]:
    idx = np.flatnonzero(mask)
    return simulate(ev[idx], fwd[idx], kw.pop("ex", EXIT), **kw).stats()  # type: ignore[arg-type]


def main() -> None:
    ev, fwd = load_long()
    early = ev.group_by("symbol").agg(pl.col("open_time").min().alias("first"))
    early_syms = early.filter(pl.col("first") < pl.datetime(2020, 2, 1, time_zone="UTC"))["symbol"]
    late_fwd = delayed(fwd, 48)
    rows = []
    for name, cond in CANDIDATES.items():
        m = ev.select((pl.col("is_spike") & cond).fill_null(False))["is_spike"].to_numpy()
        not_test = (ev["split"] != "test").to_numpy()
        base = m & not_test
        variants: dict[str, tuple[np.ndarray, dict[str, object]]] = {
            "base": (base, {}),
            "cost 0.6%": (base, {"cost": 0.006}),
            "cost 1.0%": (base, {"cost": 0.010}),
            "late entry": (base, {"fwd": late_fwd}),
            "slots 3": (base, {"slots": 3}),
            "slots 10": (base, {"slots": 10}),
            "coins listed by 2020-01": (base & ev["symbol"].is_in(early_syms).to_numpy(), {}),
            "coins listed later": (base & ~ev["symbol"].is_in(early_syms).to_numpy(), {}),
        }
        for y in range(2020, 2025):
            variants[f"year {y}"] = (m & (ev["open_time"].dt.year() == y).to_numpy(), {})
        for label, (mask, kw) in variants.items():
            f = kw.pop("fwd", fwd)
            rows.append({"cand": name, "variant": label, **run(ev, f, mask, **kw)})  # type: ignore[arg-type]
    df = pl.DataFrame(rows)
    df.write_csv(RESULTS / "m4_robust.csv")
    with pl.Config(tbl_rows=60, tbl_cols=12, tbl_width_chars=220, float_precision=3):
        print(df.drop("days_in_market"))


if __name__ == "__main__":
    main()
