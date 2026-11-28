"""P2.3 (perp-only follow-ups): the pre-spot window. Coins whose USD-M perp traded at least 8 days
before their Binance spot listing (spot = the first 5m candle in the shared store or the archive
store, after 2020-01-15). Their perp spikes from the perp listing to the spot listing are scored
with the day5 walk-forward fold of their year (2022+) and traded as L on the perp (as p1).

Fetches the perps' monthly 5m klines (perp listing month .. spot listing month + 1) into
OWN/futures/perp5m_prespot/ (raw zips in OWN/futures/raw_p2pre/, deleted after parsing).
Writes OWN/p2/prespot_spikes.parquet, OWN/p2/trades_L_prespot.parquet and
results/explore/p2_prespot.md.
Run: uv run --group research python p2_prespot.py
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
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
    P2,
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
RAW = FUT / "raw_p2pre"
PRE5 = FUT / "perp5m_prespot"
SPOT_DIRS = (
    Path("/Users/mike/code/crypto-moonshot/data/klines/spot/interval=5m"),
    Path(__file__).resolve().parent.parent / "volume_spike/cache/archive/klines/spot/interval=5m",
)
OUTMD = Path(__file__).resolve().parent / "results" / "explore" / "p2_prespot.md"


def spot_first() -> dict[str, datetime]:
    first: dict[str, datetime] = {}
    for d in SPOT_DIRS:
        for p in d.glob("symbol=*"):
            fs = sorted(p.glob("*.parquet"))
            if not fs:
                continue
            t = pl.read_parquet(fs[0], columns=["open_time"])["open_time"].min()
            s = p.name.removeprefix("symbol=")
            assert isinstance(t, datetime)
            first[s] = min(first.get(s, t), t)
    return first


def coins() -> pl.DataFrame:
    sf = spot_first()
    p = pl.read_parquet(FUT / "perp_1h.parquet", columns=["symbol", "open_time"])
    g = p.group_by("symbol").agg(a=pl.col("open_time").min())
    rows = []
    for s, a in g.iter_rows():
        sp = spot_name(s)
        if sp in sf and sf[sp] > datetime(2020, 1, 15, tzinfo=UTC):
            lead = (sf[sp] - a).total_seconds() / 86400
            if lead >= 8:
                rows.append((s, sp, a, sf[sp], lead))
    return pl.DataFrame(
        rows, schema=["perp", "spot", "perp_first", "spot_first", "lead_days"], orient="row"
    ).sort("perp_first")


def months(a: datetime, b: datetime) -> list[str]:
    out, y, m = [], a.year, a.month
    while (y, m) <= (b.year + (b.month == 12), b.month % 12 + 1):
        out.append(f"{y}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def fetch(r: dict[str, object]) -> int:
    sym = str(r["perp"])
    out = PRE5 / f"{sym}.parquet"
    if out.exists():
        return 0
    have = set(list_prefix(f"data/futures/um/monthly/klines/{sym}/5m/", False))
    parts = []
    for mo in months(r["perp_first"], r["spot_first"]):  # type: ignore[arg-type]
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
    df.write_parquet(out)
    return df.height


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    PRE5.mkdir(parents=True, exist_ok=True)
    c = coins()
    print(c.height, "pre-spot coins", file=sys.stderr)
    with ThreadPoolExecutor(16) as ex:
        n = sum(ex.map(fetch, c.iter_rows(named=True)))
    print("rows", n, file=sys.stderr)
    P5_ORIG = p1_lib.P5
    p1_lib.P5 = PRE5  # load_perp / perp_paths read the pre-spot store
    raws, st, en = {}, {}, {}
    for r in c.iter_rows(named=True):
        f = PRE5 / f"{r['perp']}.parquet"
        if f.exists():
            raws[r["perp"]] = p1_lib.load_perp(r["perp"])
            st[r["perp"]] = START
            en[r["perp"]] = min(r["spot_first"], END)
    sp = score_coins(raws, st, en, Context())
    sp.write_parquet(P2 / "prespot_spikes.parquet")
    lines = ["### F3. The pre-spot window (perp listed before spot)", ""]
    cov = c.with_columns(y=pl.col("spot_first").dt.year()).group_by("y").agg(
        coins=pl.len(), median_lead_days=pl.col("lead_days").median(),
        coin_days=pl.col("lead_days").sum().round(0),
    ).sort("y")  # fmt: skip
    yc = sp.group_by("year").agg(
        spikes=pl.len(), coins=pl.col("symbol").n_unique(), active=pl.col("perp").mean(),
        q995=((pl.col("pct") >= 0.995) & pl.col("perp")).sum(),
        q995_days=pl.col("day").filter((pl.col("pct") >= 0.995) & pl.col("perp")).n_unique(),
    ).sort("year")  # fmt: skip
    lines += [f"{c.height} coins had a perp at least 8 days before their spot listing (by spot listing year):", "",  # noqa: E501
              md(cov, 1), "",
              f"{sp.height:,} liquid perp spikes in their pre-spot windows from 2022 (scored by the walk-forward fold of their year):", "",  # noqa: E501
              md(yc), ""]  # fmt: skip

    sig = sp.filter(pl.col("perp") & (pl.col("pct") >= 0.995)).with_columns(prio=pl.col("pct"))
    t = sim_perp(sig, 1, RL)
    t.write_parquet(P2 / "trades_L_prespot.parquet")
    p1_lib.P5 = P5_ORIG
    # flush days vs spot L and perp-only L
    m = meta()
    pct = scores("day5")
    sel = (m["year"].to_numpy() >= 2022) & np.isfinite(pct) & (pct >= 0.995)
    spot_days = set(m.filter(pl.Series(sel))["day"].to_list())
    po = perp_only_spikes().filter(pl.col("perp") & (pl.col("pct") >= 0.995))
    po_days = set(po["open_time"].dt.date().to_list())
    pre_days = set(sig["day"].to_list())
    new = sorted(pre_days - spot_days - po_days)
    if t.height:
        yt = t.group_by("year").agg(
            n=pl.len(), coins=pl.col("symbol").n_unique(), days=pl.col("day").n_unique(),
            mean_net=pl.col("net").mean(), tp_rate=(pl.col("exit_kind") == "tp").mean(),
        ).sort("year")  # fmt: skip
        t_new = t.filter(pl.col("day").is_in(pl.Series(new, dtype=pl.Date)))
        lines += [
            "L trades in the pre-spot window (perp long, 1x, TP 20%, 72h, cost 0.3% + funding, one per coin):",  # noqa: E501
            "",
            md(yt),
            "",
            f"All: n {t.height}, mean net {t['net'].mean():+.4f}, TP rate {(t['exit_kind'] == 'tp').mean():.2f}, "  # noqa: E501
            f"{t['day'].n_unique()} days; {len(new)} of the {len(pre_days)} signal days are not spot-L or perp-only-L days "  # noqa: E501
            f"({', '.join(str(x) for x in new)}); trades on them: n {t_new.height}, mean net "
            f"{t_new['net'].mean() if t_new.height else float('nan'):+.4f}.",
            "",
            md(
                t.select("symbol", "open_time", "pct", "ret_4h", "net", "exit_kind").sort(
                    "open_time"
                ),
                3,
            ),
            "",
        ]
        base = lmc()
        plus = [*base, to_book(lperp(), "L").select(BOOK_COLS)]
        rows = [
            book_row("LMC+", book(pl.concat(plus, how="vertical_relaxed"), 10)),
            book_row(
                "LMC+ + pre-spot L",
                book(
                    pl.concat([*plus, to_book(t, "L").select(BOOK_COLS)], how="vertical_relaxed"),
                    10,
                ),
            ),
        ]
        lines += ["Book (10 slots, $10k each year):", "", md(pl.DataFrame(rows)), ""]
    else:
        lines += ["No L signal (pct >= 0.995 with an active perp) in any pre-spot window.", ""]
    OUTMD.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
