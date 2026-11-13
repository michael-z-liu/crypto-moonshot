# Research

One-off research scripts, their results and the reports written from them. Nothing here places
orders, and nothing in `src/` imports from here. Reusable code lives in `src/crypto_moonshot/`;
when a research idea is worth keeping, it is ported there with tests (as the backtest engine and
the capitulation rules were).

All research so far is one line of work in `volume_spike/`: does a sudden 5-minute volume spike
predict a price move, and can a strategy be built on it?

## Studies

In the order they were done. Each later study builds on the earlier ones' caches and scripts.

| Study | Question | Scripts (`volume_spike/`) | Report | Plan and run log |
| --- | --- | --- | --- | --- |
| Volume spike and capitulation rules | Does buying a spike pay? Which crash conditions do? | `build_events.py`, `build_long.py`, `m2_`..`m9_`, `m10_model.py`, `evalx.py`, `portfolio.py` | `docs/research/volume-spike.md` | `docs/plans/volume-spike-research.md` |
| CNN, walk-forward and survivorship | Can a 1D CNN on the raw candles rank the spikes? Does it survive delisted coins? | `build_seq.py`, `build_seq2.py`, `m11_cnn.py`..`m16_survivorship.py`, `fetch_archive.py` | `docs/research/volume-spike.md` | `docs/plans/volume-spike-research.md` |
| Replication and side experiments | Bracket exits (B1), $50 long shots (B2), quiet coins (B3), 1-minute triggers (B4), 3-day path forecasts (B5) | `m17_`..`m25_`, `build_quiet*.py`, `build_long30.py`, `build_fwd72.py`, `fetch_1m.py` | none; the conclusions are in the plan's log | `docs/plans/cnn-replication.md` |
| CNN robustness | What makes the walk-forward CNN more robust? | `scorecard.py`, `s0_baseline.py`..`s5_confirm.py`, `build_s4_inputs.py` | `docs/research/cnn-robustness.md` | `docs/plans/cnn-robustness.md` |
| Issue #10 improvements | Do the improvement ideas from issue #10 hold up? | `i10_*.py` | `docs/research/cnn-issue10.md` | `docs/plans/cnn-improvements.md` |
| Forward paper test | How does the 2h CNN rule do on candles after every study's data end? | `paper.py` | `volume_spike/results/paper/summary.md` | `docs/plans/cnn-robustness.md` (P1) |

Reporting: `report_data.py`, `report_wf.py`, `report_examples.py` and `report/` build the visual
casebook (the built page and its data are gitignored); `plots.py` draws the summary charts in
`results/`.

## Running the scripts

```bash
uv sync --group research        # numpy, matplotlib, scikit-learn, torch
cd research/volume_spike
export MOONSHOT_UNIVERSE=all    # pool (default) | poolA (+ delisted) | all (+ every archived USDT pair)
uv run --group research python m2_baseline.py
```

- The scripts read candles from `MOONSHOT_DATA_DIR` (see `common.py` for the default) and expect
  the data that `moonshot data sync` and `fetch_archive.py` / `fetch_1m.py` put there.
- Large intermediate tables go to `cache/` or `cache_<universe>/`, which are gitignored. Small
  result tables, notes and charts are committed under `results/`.
- From the robustness study on, books are run by the production engine
  (`crypto_moonshot.backtest`) through `scorecard.py`. Earlier studies use the simulators in
  `evalx.py` and `portfolio.py`; `docs/backtest-validation.md` measures the difference.
- Each script's docstring says which study and step it belongs to and how to run it.

## Reading the results

- 2025-26 was the held-out test period. It has now been opened by three studies, so treat any new
  result on it as seen data.
- The plans record what was decided before each test period was opened. The reports cite those
  decisions by commit hash; the original branch histories are kept under the `archive/*` tags.
