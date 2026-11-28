"""X6: crowding / positioning gates on sleeve F (pre-declared grid, every variant reported).

Base F = pump shorts r4 >= 0.4, perp, qv_base >= 20k, 2022+, last funding >= 0, 72h, isolated 1x.
Gates use only entry-time columns (feats / metric_feats). Gate levels were fixed from the pool
quartiles of the ENTRY variables only (not outcomes). A gate filters candidate rows before the
simulation (one position per coin while held), so which row triggers a coin may change.
Selection split: choose on 2022-24, read 2025-26. Writes results/explore/x6_*.csv."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
from x6_lib import card, pool, run_book, trades

OUT = Path(__file__).resolve().parent / "results" / "explore"
p = pool()
tl_med = float(p["taker_ls"].median())
C = pl.col
G: dict[str, pl.Expr] = {
    "base": pl.lit(True),
    "ls>=0.75": C("ls_acc") >= 0.75,
    "ls>=1.0": C("ls_acc") >= 1.0,
    "oi4h<=0.34": C("oi_ch4h") <= 0.34,
    "oi4h<=0.12": C("oi_ch4h") <= 0.12,
    "oi4h>0.34": C("oi_ch4h") > 0.34,
    "rate>=1e-4": C("rate") >= 1e-4,
    "rate24>=1e-4": C("rate24") >= 1e-4,
    "taker<=med": C("taker_ls") <= tl_med,
    "taker>med": C("taker_ls") > tl_med,
    "age>=180d": C("listed_days") >= 180,
    "age>=365d": C("listed_days") >= 365,
    "ls>=1.0&rate>=1e-4": (C("ls_acc") >= 1.0) & (C("rate") >= 1e-4),
    "ls>=1.0&oi4h<=0.34": (C("ls_acc") >= 1.0) & (C("oi_ch4h") <= 0.34),
    "ls>=1.0&age>=180d": (C("ls_acc") >= 1.0) & (C("listed_days") >= 180),
    "ls>=1.0&rate>=1e-4&oi4h<=0.34": (C("ls_acc") >= 1.0) & (C("rate") >= 1e-4) & (C("oi_ch4h") <= 0.34),  # noqa: E501
}  # fmt: skip


def row(name: str, t: pl.DataFrame, per: str) -> dict[str, object]:
    c = card(t)
    return {"gate": name, "period": per, "n": int(c["n"]), "E_o": c["E_o"], "stopped": c["share_stopped"],  # noqa: E501
            "contrib_stopped": c["contrib_stopped"], "yrs_pos": c["years_positive"],
            "bp": float(t["net"].mean()) * 1e4}  # fmt: skip


rows, books = [], []
for name, g in G.items():
    t = trades(p.filter(g))
    rows += [row(name, t, "all"), row(name, t.filter(C("year") <= 2024), "22-24"),
             row(name, t.filter(C("year") >= 2025), "25-26")]  # fmt: skip
    b = run_book(t)
    for per, ys in (
        ("all", range(2022, 2027)),
        ("22-24", range(2022, 2025)),
        ("25-26", range(2025, 2027)),
    ):
        s = b.filter(C("year").is_in(list(ys)))
        n = int(s["trades"].sum())
        books.append({"gate": name, "period": per, "trades": n,
                      "bp": float((s["bp"] * s["trades"]).sum() / n),
                      "geo_ret": float(np.prod(1 + s["ret"].to_numpy()) ** (1 / len(s)) - 1),
                      "min_ret": float(s["ret"].min()), "worst_dd": float(s["dd"].min()),
                      "bpP": float((s["bpP"] * s["nP"]).sum() / max(1, s["nP"].sum())), "nP": int(s["nP"].sum()),  # noqa: E501
                      "yrs": " ".join(f"{r:+.0%}" for r in s["ret"].to_list())})  # fmt: skip
S = pl.DataFrame(rows)
B = pl.DataFrame(books)
S.write_csv(OUT / "x6_scorecards.csv")
B.write_csv(OUT / "x6_books.csv")
pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_width_chars(250)
pl.Config.set_tbl_cols(20)
print(S.filter(C("period") == "all"))
print(S.filter(C("period") == "22-24"))
print(S.filter(C("period") == "25-26"))
print(B.filter(C("period") == "all"))
print(
    B.filter(C("period") == "22-24").select(
        "gate", "trades", "bp", "geo_ret", "min_ret", "bpP", "nP"
    )
)
print(
    B.filter(C("period") == "25-26").select(
        "gate", "trades", "bp", "geo_ret", "min_ret", "bpP", "nP"
    )
)

# diagnostics: why gates fail. Outcome by tercile of each gate variable on base-F trades, and AUC for  # noqa: E501
# squeeze (adv >= 0.5) and stopped. Also on the ungated P pool (r4 >= 0.4, any funding) for power.
pa = pl.read_parquet(__import__("common_ls").OWN / "feats.parquet").filter(
    C("perp") & (C("qv_base") >= 20_000) & (C("open_time").dt.year() >= 2022) & (C("ret_4h") >= 0.4)
).join(pl.read_parquet(__import__("common_ls").OWN / "metric_feats.parquet"), on="row", how="left")  # fmt: skip  # noqa: E501
tb = trades(p).join(
    p.select("row", "ls_acc", "oi_ch4h", "oi_ch1h", "taker_ls", "rate", "rate24", "listed_days"),
    on="row",
)
ta = trades(pa).join(
    pa.select("row", "ls_acc", "oi_ch4h", "oi_ch1h", "taker_ls", "rate", "rate24", "listed_days"),
    on="row",
)
V = ["ls_acc", "oi_ch4h", "oi_ch1h", "taker_ls", "rate", "rate24", "listed_days"]
diag = []
for nm, t in (("F", tb), ("P-any-funding", ta)):
    t = t.with_columns(
        sq50=C("adv") >= 0.5, st=C("exit_kind").is_in(["stop", "liq"]) | (C("net") <= -0.95)
    )
    for v in V:
        x = t.drop_nulls(v).drop_nans(v)
        auc50 = roc_auc_score(x["sq50"], x[v]) if x["sq50"].sum() > 0 else float("nan")
        auc_st = roc_auc_score(x["st"], x[v]) if x["st"].sum() > 0 else float("nan")
        q = x.with_columns(q=(C(v).rank("ordinal") - 1) * 3 // pl.len()).group_by("q").agg(
            n=pl.len(), E_o=C("net").clip(-1, 1).mean(), sq50=C("sq50").mean(), stopped=C("st").mean(),  # noqa: E501
            fund=C("fund").mean()).sort("q")  # fmt: skip
        diag.append({"pool": nm, "var": v, "n": x.height, "n_sq50": int(x["sq50"].sum()), "n_stopped": int(x["st"].sum()),  # noqa: E501
                     "auc_sq50": auc50, "auc_stopped": auc_st,
                     "E_o_by_tercile": " / ".join(f"{a:+.3f}" for a in q["E_o"]),
                     "stopped_by_tercile": " / ".join(f"{a:.1%}" for a in q["stopped"]),
                     "sq50_by_tercile": " / ".join(f"{a:.1%}" for a in q["sq50"])})  # fmt: skip
D = pl.DataFrame(diag)
D.write_csv(OUT / "x6_diag.csv")
print(D)
# the stopped trades of F
print(
    tb.filter(C("exit_kind").is_in(["stop", "liq"]) | (C("net") <= -0.95)).select(
        "symbol",
        "open_time",
        "ret",
        "net",
        "ls_acc",
        "oi_ch4h",
        "taker_ls",
        "rate",
        "rate24",
        "listed_days",
    )
)
print(tb.select(C(v).median().alias(v) for v in V))
