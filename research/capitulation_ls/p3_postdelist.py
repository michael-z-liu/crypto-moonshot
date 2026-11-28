"""P3 (perp-only follow-ups): the post-delisting window. Coins whose USD-M perp kept trading for at
least 3 days after their last Binance spot 5m candle (spot delisting). The spot backtest stops at
the spot delisting and p1 excluded these coins (they have spot klines), so this window was
uncovered.

Coins: a perp in OWN/futures/perp_1h.parquet (every USDT perp of the archive, delisted ones
included) whose spot name has spot 5m klines in the shared store, and whose last *active* perp
hour (quote volume > 0 and high > low; the archive keeps printing flat zero-volume candles after a
perp is delisted) is >= 3 days after the last spot 5m candle.

Fetches the perps' monthly 5m klines from the month before the spot delisting (lookback for the
spike features, max window 7 days, and the 1-day CNN sequence) to the perp's last active month + 1,
into OWN/futures/perp5m_postdelist/ (raw zips in OWN/futures/raw_postdelist/, deleted after
parsing). Flat zero-volume candles after the last traded candle are dropped, so a trade still open
when the perp stops trading is settled at the last traded price (`perp_paths` forward-fills).

Spikes after the last spot candle are scored with the day5 walk-forward fold of their year (2022+)
and traded as L on the perp (as p1 / p2_prespot: 1x, TP 20%, 72h, cost 0.3% + funding, perp gate,
one position per coin). Writes OWN/p3/postdelist_spikes.parquet, OWN/p3/trades_L_postdelist.parquet
and results/explore/post_delist.md.
Run: uv run --group research python p3_postdelist.py
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import p1_lib
import polars as pl
from common_ls import meta, scores
from e25_combined import book
from fetch_futures import list_prefix
from futures_ls import spot_name
from p1_lib import END, START, Context
from p2_lib import (
    BOOK_COLS,
    RL,
    book_row,
    fetch_csv,
    finish,
    lmc,
    lperp,
    md,
    parse_klines,
    perp_only_spikes,
    score_coins,
    sim_perp,
    to_book,
)

FUT = p1_lib.FUT
RAW = FUT / "raw_postdelist"
PD5 = FUT / "perp5m_postdelist"
P3 = p1_lib.P1.parent / "p3"
SPOT_5M = Path("/Users/mike/code/crypto-moonshot/data/klines/spot/interval=5m")
DELISTED = Path("/Users/mike/code/crypto-moonshot/data/delisted.txt")
OUTMD = Path(__file__).resolve().parent / "results" / "explore" / "post_delist.md"
MIN_DAYS = 3.0
HOLD_S = RL.hold_min * 60


def spot_last() -> dict[str, datetime]:
    last: dict[str, datetime] = {}
    for p in SPOT_5M.glob("symbol=*"):
        fs = sorted(p.glob("*.parquet"))
        if not fs:
            continue
        t = pl.read_parquet(fs[-1], columns=["open_time"])["open_time"].max()
        assert isinstance(t, datetime)
        last[p.name.removeprefix("symbol=")] = t
    return last


def coins() -> pl.DataFrame:
    """Perps active >= MIN_DAYS after their coin's last spot 5m candle."""
    sl = spot_last()
    p = (
        pl.scan_parquet(FUT / "perp_1h.parquet")
        .group_by("symbol")
        .agg(
            perp_first=pl.col("open_time").min(),
            perp_last=pl.col("open_time")
            .filter((pl.col("quote_volume") > 0) & (pl.col("high") > pl.col("low")))
            .max(),
        )
        .collect()
        .drop_nulls()
    )
    delisted = set(DELISTED.read_text().split())
    rows = []
    for s, a, b in p.iter_rows():
        sp = spot_name(s)
        if sp not in sl:
            continue
        after = (b - sl[sp]).total_seconds() / 86400
        if after >= MIN_DAYS:
            rows.append((s, sp, a, sl[sp], b, after, sp in delisted))
    schema = ["perp", "spot", "perp_first", "spot_last", "perp_last", "days_after", "in_delisted"]
    return pl.DataFrame(rows, schema=schema, orient="row").sort("spot_last")


def months(a: datetime, b: datetime) -> list[str]:
    """Months from the one before `a` to the one after `b`."""
    y, m = (a.year - 1, 12) if a.month == 1 else (a.year, a.month - 1)
    out = []
    while (y, m) <= (b.year + (b.month == 12), b.month % 12 + 1):
        out.append(f"{y}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def fetch(r: dict[str, object]) -> int:
    sym = str(r["perp"])
    out = PD5 / f"{sym}.parquet"
    if out.exists():
        return 0
    have = set(list_prefix(f"data/futures/um/monthly/klines/{sym}/5m/", False))
    parts = []
    for mo in months(r["spot_last"], r["perp_last"]):  # type: ignore[arg-type]
        key = f"data/futures/um/monthly/klines/{sym}/5m/{sym}-5m-{mo}.zip"
        if key not in have:
            continue
        raw = fetch_csv(key, RAW)
        k = parse_klines(raw) if raw is not None else None
        if k is not None:
            parts.append(k)
    if not parts:
        return 0
    df = finish(parts)
    traded = df.filter(pl.col("qv") > 0)["ms"]
    if traded.len() == 0:
        return 0
    df = df.filter(pl.col("ms") <= traded.max())  # drop the flat candles after the perp's end
    df.write_parquet(out)
    return df.height


def last_traded() -> dict[str, int]:
    """perp -> open time (epoch s) of its last traded 5m candle in the fetched window."""
    out = {}
    for f in PD5.glob("*.parquet"):
        ms = pl.read_parquet(f, columns=["ms"])["ms"].max()
        assert isinstance(ms, int)
        out[f.stem] = ms // 1000
    return out


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    PD5.mkdir(parents=True, exist_ok=True)
    P3.mkdir(parents=True, exist_ok=True)
    c = coins()
    print(c.height, "post-delisting coins", file=sys.stderr)
    with ThreadPoolExecutor(12) as ex:
        n = sum(ex.map(fetch, c.iter_rows(named=True)))
    print("rows", n, file=sys.stderr)
    lt = last_traded()
    p5_orig = p1_lib.P5
    p1_lib.P5 = PD5  # load_perp / perp_paths read the post-delisting store
    raws, st, en = {}, {}, {}
    for r in c.iter_rows(named=True):
        f = PD5 / f"{r['perp']}.parquet"
        if f.exists():
            raws[r["perp"]] = p1_lib.load_perp(r["perp"])
            st[r["perp"]] = max(r["spot_last"] + timedelta(minutes=5), START)
            en[r["perp"]] = END
    sp = score_coins(raws, st, en, Context())
    sp.write_parquet(P3 / "postdelist_spikes.parquet")

    lines = ["# The post-delisting window: perps that kept trading after the spot delisting", ""]
    win = c.with_columns(
        perp_last_traded=pl.col("perp").replace_strict(lt, default=None).cast(pl.Int64),
    ).with_columns(
        perp_last_traded=pl.from_epoch("perp_last_traded", "s").dt.replace_time_zone("UTC"),
        window_days=pl.col("days_after").round(1),
        days_in_2022_26=(
            (pl.min_horizontal(pl.col("perp_last"), pl.lit(END)) - pl.max_horizontal(pl.col("spot_last"), pl.lit(START)))  # noqa: E501
            .dt.total_hours() / 24
        ).clip(0, None).round(1),
    )  # fmt: skip
    yc = (
        sp.group_by("year")
        .agg(
            spikes=pl.len(), coins=pl.col("symbol").n_unique(), perp_active=pl.col("perp").mean(),
            q995=(pl.col("pct") >= 0.995).sum(),
            q995_active=((pl.col("pct") >= 0.995) & pl.col("perp")).sum(),
            q995_days=pl.col("day").filter((pl.col("pct") >= 0.995) & pl.col("perp")).n_unique(),
        )
        .sort("year")
    ) if sp.height else pl.DataFrame()  # fmt: skip
    lines += [
        "## Coverage", "",
        f"{c.height} perps traded at least {MIN_DAYS:.0f} days after their coin's last spot 5m candle "  # noqa: E501
        f"(all {int(c['in_delisted'].sum())} in data/delisted.txt); "
        f"{int(win['days_in_2022_26'].sum()):,} coin-days in 2022-01..2026-10:", "",
        md(win.select("perp", "spot_last", "perp_last_traded", "window_days", "days_in_2022_26"), 1), "",  # noqa: E501
        f"{sp.height:,} liquid perp spikes after the spot delisting (scored by the walk-forward fold of their year):", "",  # noqa: E501
        md(yc) if yc.height else "(none)", "",
    ]  # fmt: skip

    sig = (
        sp.filter(pl.col("perp") & (pl.col("pct") >= 0.995)).with_columns(prio=pl.col("pct"))
        if sp.height
        else pl.DataFrame()
    )
    t = sim_perp(sig, 1, RL) if sig.height else pl.DataFrame()
    p1_lib.P5 = p5_orig
    if t.height:
        t = t.with_columns(
            perp_end=pl.col("symbol").replace_strict(lt, default=None).cast(pl.Int64)
        ).with_columns(
            path_short=pl.col("t_in") + HOLD_S > pl.col("perp_end"),
            settled_at_end=pl.col("t_out") > pl.col("perp_end"),
        )
        t.write_parquet(P3 / "trades_L_postdelist.parquet")
    # flush days vs spot L and perp-only L
    m = meta()
    pct = scores("day5")
    sel = (m["year"].to_numpy() >= 2022) & np.isfinite(pct) & (pct >= 0.995)
    spot_days = set(m.filter(pl.Series(sel))["day"].to_list())
    po = perp_only_spikes().filter(pl.col("perp") & (pl.col("pct") >= 0.995))
    po_days = set(po["open_time"].dt.date().to_list())
    sig_days = set(sig["day"].to_list()) if sig.height else set()
    new = sorted(sig_days - spot_days - po_days)
    lines += ["## L trades", ""]
    if t.height:
        yt = t.group_by("year").agg(
            n=pl.len(), coins=pl.col("symbol").n_unique(), days=pl.col("day").n_unique(),
            mean_net=pl.col("net").mean(), tp_rate=(pl.col("exit_kind") == "tp").mean(),
        ).sort("year")  # fmt: skip
        t_new = t.filter(pl.col("day").is_in(pl.Series(new, dtype=pl.Date)))
        lines += [
            f"L signals (pct >= 0.995 with an active perp): {sig.height} spikes on "
            f"{sig['symbol'].n_unique()} coins and {len(sig_days)} days. Trades (perp long, 1x, TP 20%, "  # noqa: E501
            "72h, cost 0.3% + funding, one per coin):",
            "",
            md(yt),
            "",
            f"All: n {t.height}, mean net {t['net'].mean():+.4f}, TP rate {(t['exit_kind'] == 'tp').mean():.2f}, "  # noqa: E501
            f"{t['day'].n_unique()} days. The 72h path runs past the perp's last traded candle for "
            f"{int(t['path_short'].sum())} trades; {int(t['settled_at_end'].sum())} were still open "  # noqa: E501
            "then and are settled at the last traded price. "
            f"{len(new)} of the {len(sig_days)} signal days are not spot-L or perp-only-L days "
            f"({', '.join(str(x) for x in new) or '-'}); trades on them: n {t_new.height}, mean net "  # noqa: E501
            f"{t_new['net'].mean() if t_new.height else float('nan'):+.4f}.",
            "",
            md(
                t.select(
                    "symbol", "open_time", "pct", "ret_4h", "net", "exit_kind", "settled_at_end"
                ).sort("open_time"),
                3,
            ),
            "",
        ]
        base = lmc()
        plus = [*base, to_book(lperp(), "L").select(BOOK_COLS)]
        rows = [
            book_row("LMC+", book(pl.concat(plus, how="vertical_relaxed"), 10)),
            book_row(
                "LMC+ + post-delisting L",
                book(
                    pl.concat([*plus, to_book(t, "L").select(BOOK_COLS)], how="vertical_relaxed"),
                    10,
                ),
            ),
        ]
        lines += ["## Book (10 slots, $10k each year)", "", md(pl.DataFrame(rows)), ""]
    else:
        lines += [
            "No L signal (pct >= 0.995 with an active perp) in any post-delisting window.",
            "",
        ]
        # for information: the best-scored spikes
        if sp.height:
            top = sp.sort("pct", descending=True).head(10)
            lines += [
                "Top 10 post-delisting spikes by pct (for information):",
                "",
                md(top.select("symbol", "open_time", "pct", "perp", "qv_base", "ret_4h"), 3),
                "",
            ]
    out = P3 / "post_delist_auto.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
