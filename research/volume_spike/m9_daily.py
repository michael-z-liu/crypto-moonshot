"""M9: the same idea at a daily scale: abnormal daily volume and the next days' returns."""

from __future__ import annotations

import polars as pl
from common import CACHE, KLINES_5M, RESULTS, split_of, universe
from evalx import COST


def daily_bars() -> pl.DataFrame:
    path = CACHE / "daily.parquet"
    if path.exists():
        return pl.read_parquet(path)
    frames = []
    for s in universe():
        d = (
            pl.scan_parquet(KLINES_5M / f"symbol={s}" / "*.parquet")
            .group_by_dynamic("open_time", every="1d")
            .agg(
                pl.col("open").first(), pl.col("high").max(), pl.col("low").min(),
                pl.col("close").last(), pl.col("quote_volume").sum().alias("qv"),
                pl.col("taker_buy_quote_volume").sum().alias("tbq"), pl.len().alias("bars"),
            )
            .collect()
            .filter(pl.col("bars") >= 280)
            .with_columns(symbol=pl.lit(s))
        )  # fmt: skip
        frames.append(d.slice(7))
    df = pl.concat(frames).sort("symbol", "open_time")
    df.write_parquet(path)
    return df


def main() -> None:
    d = daily_bars()
    c = pl.col("close")
    d = d.with_columns(
        vmult=pl.col("qv")
        / pl.col("qv").shift(1).rolling_median(30, min_samples=20).over("symbol"),
        ret_d=c / pl.col("open") - 1,
        ret_7d=c / c.shift(7).over("symbol") - 1,
        tbr=pl.col("tbq") / pl.col("qv"),
        # enter at the next day's open (~ this close), hold n days
        f1=c.shift(-1).over("symbol") / c - 1,
        f3=c.shift(-3).over("symbol") / c - 1,
        f7=c.shift(-7).over("symbol") / c - 1,
    ).with_columns(split_of())
    base = d.filter(pl.col("vmult").is_finite() & pl.col("f7").is_not_null())
    btc = base.filter(pl.col("symbol") == "BTCUSDT").select(
        "open_time", btc_ret_d=pl.col("ret_d"), btc_vmult=pl.col("vmult")
    )
    base = base.join(btc, on="open_time", how="left")
    groups = {
        "all coin-days": pl.lit(True),
        "vol>=3x": pl.col("vmult") >= 3,
        "vol>=3x & up day >5%": (pl.col("vmult") >= 3) & (pl.col("ret_d") > 0.05),
        "vol>=3x & down day <-5%": (pl.col("vmult") >= 3) & (pl.col("ret_d") < -0.05),
        "vol>=5x & up day >10%": (pl.col("vmult") >= 5) & (pl.col("ret_d") > 0.10),
        "vol>=5x & down day <-10%": (pl.col("vmult") >= 5) & (pl.col("ret_d") < -0.10),
        "down <-10%, BTC down <-5%": (pl.col("ret_d") < -0.10) & (pl.col("btc_ret_d") < -0.05),
        "vol>=3x & flat day |r|<2%": (pl.col("vmult") >= 3) & (pl.col("ret_d").abs() < 0.02),
        "vol>=3x & flat & tbr>0.55": (pl.col("vmult") >= 3) & (pl.col("ret_d").abs() < 0.02)
        & (pl.col("tbr") > 0.55),
    }  # fmt: skip
    rows = []
    for name, cond in groups.items():
        for split in ("train", "valid", "test"):
            s = base.filter(cond & (pl.col("split") == split))
            rows.append({
                "group": name, "split": split, "n": s.height,
                **{h: (s[h].mean() - COST) * 1e4 for h in ("f1", "f3", "f7")},
            })  # fmt: skip
    df = pl.DataFrame(rows)
    df.write_csv(RESULTS / "m9_daily.csv")
    wide = df.pivot(on="split", index="group", values=["n", "f1", "f3", "f7"])
    with pl.Config(tbl_rows=20, tbl_cols=14, tbl_width_chars=230, float_precision=0):
        print(wide)


if __name__ == "__main__":
    main()
