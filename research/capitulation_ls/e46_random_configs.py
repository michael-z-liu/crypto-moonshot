"""E46: how much of the LMC configuration space meets both targets? 400 random configurations
from plausible ranges per sleeve and slots; share with avg trade >= 10% and every year >= +50%."""

from __future__ import annotations

import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl


def run(seed: int) -> dict:
    from e25_combined import book, sleeve_crash_short, sleeve_listing_weak, sleeve_long

    rnd = random.Random(seed)
    cfg = {
        "q": rnd.choice([0.99, 0.993, 0.995, 0.997]), "tp": rnd.choice([0.1, 0.15, 0.2, 0.25, 0.3]), "hold": rnd.choice([1440, 2880, 4320]),  # noqa: E501
        "r4": rnd.choice([-0.1, -0.15, -0.2]), "calm": rnd.choice([0.01, 0.015, 0.02]), "cnn": rnd.choice([0.3, 0.5, 0.7]), "chold": rnd.choice([1440, 4320]),  # noqa: E501
        "slots": rnd.choice([6, 8, 10, 12, 15]),
    }  # fmt: skip
    parts = pl.concat([sleeve_long(q=cfg["q"], tp=cfg["tp"], hold=cfg["hold"]), sleeve_listing_weak(),  # noqa: E501
                       sleeve_crash_short(r4=cfg["r4"], calm=cfg["calm"], cnn=cfg["cnn"], hold=cfg["chold"])], how="vertical_relaxed")  # fmt: skip  # noqa: E501
    b = book(parts, cfg["slots"])
    n = int(b["trades"].sum())
    return {
        **cfg,
        "bp": float((b["bp"] * b["trades"]).sum() / n),
        "min_ret": float(b["ret"].min()),
        "med_ret": float(b["ret"].median()),
        "dd": float(b["dd"].min()),
    }


if __name__ == "__main__":
    with ProcessPoolExecutor(8) as ex:
        rows = list(ex.map(run, range(400)))
    d = pl.DataFrame(rows)
    d.write_csv("results/e46_random_configs.csv")
    both = d.filter((pl.col("bp") >= 1000) & (pl.col("min_ret") >= 0.5))
    print(
        f"{d.height} configs: avg trade >= 10%: {(d['bp'] >= 1000).mean():.0%}; every year >= +50%: {(d['min_ret'] >= 0.5).mean():.0%}; both: {both.height / d.height:.0%}; every year > 0: {(d['min_ret'] > 0).mean():.0%}"  # noqa: E501
    )
    print(
        "quantiles of avg trade (bp):",
        np.round(np.quantile(d["bp"], [0.1, 0.5, 0.9])).tolist(),
        "min-year return:",
        np.round(np.quantile(d["min_ret"], [0.1, 0.5, 0.9]), 2).tolist(),
        "max DD:",
        np.round(np.quantile(d["dd"], [0.1, 0.5, 0.9]), 2).tolist(),
    )
    for col in ("q", "tp", "hold", "slots", "cnn"):
        print(
            col,
            d.group_by(col)
            .agg(
                both=((pl.col("bp") >= 1000) & (pl.col("min_ret") >= 0.5)).mean().round(2),
                bp=pl.col("bp").mean().round(),
            )
            .sort(col)
            .to_dicts(),
        )
