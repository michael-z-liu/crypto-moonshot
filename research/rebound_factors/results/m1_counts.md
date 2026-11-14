# M1 entry counts and discovery base rates

Entries are coin-days at UTC midnight `D` (protocol in `docs/plans/rebound-factors.md`;
definitions in the docstrings of `build_panel.py` and `build_labels.py`). Counts of
entries cover every year; label base rates are for **discovery** entries only
(D 2020-01-31 .. 2023-12-01). 2024-26 is sealed: counts only.

| year | entries | coins | median liq ($M) | discovery entries | hit10 | hit20 | hit_ns |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2020 | 21769 | 162 | 4.8 | 21769 | 27.5% | 38.7% | 46.9% |
| 2021 | 79240 | 290 | 15.0 | 79240 | 31.1% | 45.8% | 55.5% |
| 2022 | 100646 | 338 | 4.8 | 100646 | 22.3% | 30.7% | 34.2% |
| 2023 | 90766 | 362 | 3.9 | 81268 | 24.5% | 29.9% | 31.3% |
| 2024 | 123101 | 408 | 5.6 |  |  |  |  |
| 2025 | 119371 | 472 | 3.9 |  |  |  |  |
| 2026 | 57454 | 399 | 2.6 |  |  |  |  |
| **all** | 592347 | 576 | | 282923 | | | |

## By split

| split | D range | entries | coins |
| --- | --- | --- | --- |
| discovery | 2020-01-31 .. 2023-12-01 | 282923 | 387 |
| gap | 2023-12-02 .. 2023-12-31 (unused) | 9498 | 342 |
| confirm | 2024-01-01 .. 2026-10-02 (sealed) | 299711 | 532 |
| unused | 2026-10-03 (complete window, after the confirm range; sealed) | 215 | 215 |

## Gaps and delistings (discovery entries)

Missing 5m candles are simply absent. If a coin's candles stop more than a day before
the window end (a delisting), the entry is kept, the trade exits at the last close in
the window (reason `series_end`) and `stopped_early` is set. An entry with no candle in
the window would have reason `no_entry` and null outcomes. An entry needs a candle in
the day before `D` for its price `P`, so a coin delisted days earlier is not an entry.
These flags are not reported for 2024-26 (they come from the sealed labels).

| year | stopped early | no entry | entry 1 h or more after D |
| --- | --- | --- | --- |
| 2020 | 60 | 3 | 0 |
| 2021 | 92 | 3 | 4 |
| 2022 | 549 | 22 | 2 |
| 2023 | 225 | 10 | 3 |
