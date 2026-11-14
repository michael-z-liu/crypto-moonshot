# Rebound factors: what tells a +30% move apart before it happens?

- **Status:** building <!-- building | blocked | review | done -->
- **Branch:** `feat/rebound-factors` (stacked on PR #14's head; the study code it reuses is not on `main` yet)
- **PR:** <!-- link, once opened -->
- **Updated:** 2026-11-14T12:15Z
- **Last report to the user:** 2026-11-14T11:20Z (build started)

## Goal

The frequency scan after the MA-rebound study (PR #14) showed that +30% moves within 30 days are
common (thousands of episodes a year) but follow a random hour about as often as a touch of a
major moving average (24.7% against 23.6%). The user wants to know which factors, known at the
time of entry, do separate the entries that go on to such a move. This project is a research
study on spot candles: build a coin-day panel, test factors one at a time and combined on
2020-23, freeze the choice, and score it once on 2024-26.

## Out of scope

- Futures data (funding, open interest), news and listing announcements.
- CNNs or other image models. Factor tables and gradient-boosted trees only, so results explain.
- A strategy in `src/`, engine changes, paper or live trading.
- Reruns of the MA-rebound or volume-spike studies, or writes into their caches.

## Ground rules

- Code in `research/rebound_factors/`, run from there with `uv run --group research python ...`.
  Small result tables and notes are committed under `research/rebound_factors/results/`.
- Market data: `/Users/mike/code/crypto-moonshot/data` (`MOONSHOT_DATA_DIR`), read only. 5m candles
  for the "all" universe (582 USDT pairs incl. delisted, minus pegged assets and tokenised stocks,
  as `research/volume_spike/common.py` and `research/ma_rebound/common.py` define it).
- Reusable, read only: `research/ma_rebound/cache/parts/bars_{1h,4h,1d,1w,1M}/<SYM>.parquet`
  (bars with SMA levels `L{n}`, `atr_d`, `liq`), and `research/ma_rebound/{common,events,book}.py`.
  In this worktree `research/ma_rebound/cache` is a symlink to
  `/Users/mike/code/crypto-moonshot/data/cache/ma_rebound`.
- Cache: `research/rebound_factors/cache` is a gitignored symlink to
  `/Users/mike/code/crypto-moonshot/data/cache/rebound_factors/`.
- Data end for everything: 2026-11-02 00:00 UTC (exclusive), as in the earlier studies.
- Tests: checks that need the real data are `check_*.py` scripts (as in `ma_rebound`); pure logic
  (label paths, look-ahead on synthetic candles) gets pytest tests under `tests/research/` if the
  repo's test layout allows importing it, otherwise in the `check_*.py` scripts. The check command
  in `CLAUDE.md` must pass.
- **Sealed period:** entries from 2024-01-01 on. Until the M4 spec is committed and pushed, no
  outcome of such an entry (label, peak, return, hit rate, book) may be printed, plotted,
  committed or read by anyone, master or worker. Counts and factor values (which use only the
  past) are allowed. Scripts write these outcomes only to
  `research/rebound_factors/cache/sealed/`, and discovery scripts never open that folder.
  Known caveat, to be stated in the report: yearly base rates for 2024 and 2025 (MA touches and
  random hours) were seen in the earlier frequency scan, and the MA study's 2025-26 results are
  known. No factor relationship in 2024-26 has been looked at.

## Protocol (fixed 2026-11-14T11:20Z, before any result)

**Entries.** Every coin at every UTC midnight `D` (the daily close) from 2020-01-31 while its mean
daily quote volume over the previous 30 complete UTC days is at least $1,000,000 and it has at
least 30 complete days of candles. Reference price `P`: the close of the last 5m candle before
`D`. Entries whose 30-day window is not complete by the data end are dropped.

**Labels**, from 5m candles in `[D, D + 30 days)`; when one candle touches target and stop, the
stop counts first:
- `hit10` (primary): high reaches `1.30 P` before low reaches `0.90 P`.
- `hit20`: the same with the stop at `0.80 P`.
- `hit_ns`: high reaches `1.30 P` at any time in the window.
- Also stored: peak and trough over 7, 30 days, time to target, time to each stop.
- Trade result per stop variant (10%, 20%, none): market buy at the first 5m open at or after `D`;
  limit sell at `1.30 P` (no slippage); stop as a market sell at the stop price; else market sell
  at the window end. Fee 0.1% a side, 0.1% slippage on market fills.

**Split.** Discovery: `D` from 2020-01-31 to 2023-12-01 (every window ends before 2024).
Confirm: `D` from 2024-01-01 to 2026-10-02. Entries between are unused.

**Factors.** Each uses only candles that closed at or before `D`. Two kinds, reported apart:
*market* factors (same value for every coin on a day: BTC trend, volatility, drawdown, breadth
across coins) answer "when"; *coin* factors answer "which coin", and are tested both raw and as
the rank within the day (which removes the market). Families: market regime; coin momentum and
position against its SMAs and its range; volatility and range compression; volume, trade count
and taker-buy share; relative strength, size rank and listing age. Reference flags: touched a
major SMA from above in the last day (the old thesis).

**Single-factor test (M3).** On discovery: hit rate by decile for each label, lift of the top and
bottom decile over the base rate, the same within each year 2020..2023, a rank correlation with
the peak, and a 95% interval from a bootstrap over calendar months (windows overlap for 30 days,
so days are not independent). A factor is *stable* when its best decile beats the base rate in at
least three of the four years with the same direction, and for coin factors when that holds for
the within-day rank too.

**Combined model (M4).** Gradient-boosted trees on the factors, target `hit10`, walk-forward
inside discovery: train on entries whose window ends before year Y, score year Y, for Y = 2021,
2022, 2023 (at least 3 seeds). Compared with a transparent rule of two or three stable factors.
The frozen spec names: the factor list, the model settings, the rule for the confirm years
(model trained on all of discovery, then retrained each January on everything whose window has
ended), the pick rule (which entries are taken each day and how many), and the book rules.

**Book.** $10,000, 10 equal slots of realised equity, one position per coin, at most the pick
rule's top entries each day in score order, no free slot: skipped. Exits as in the trade result.

**Confirm (M5), once.** On 2024-26: hit rate of the picks against the base rate and against
random eligible entries on the same days (100 draws); the same per year; trade result and book
after costs for each stop variant, against the same-day random book. **Confirmed** means: the
picks' `hit10` rate exceeds the same-day random draws' 95th percentile overall and in at least two
of the three years, and the 10%-stop book's return is positive and above the random books' median.

## Reporting to the user

- `PushNotification` at once on a stop condition, when the PR is ready, and for the M5 reveal.
- Otherwise at most one every 3-4 hours: on an accepted milestone or a heartbeat, send a one-line
  status only if `Last report to the user` is over 3 hours old, and update it.
- Never report a 2024-26 outcome before the reveal.

## Milestones

### M1: Coin-day panel and labels (done)

- **Delivers:** `research/rebound_factors/` with `common.py`, the cache link and `.gitignore`
  entries; `build_panel.py` writing the eligible entries with `P`; `build_labels.py` writing
  labels and trade results, discovery to `cache/labels.parquet` and 2024-26 to `cache/sealed/`;
  `check_panel.py`; a counts table in `results/`.
- **Done when:** labels match hand-computed paths on synthetic candles (target first, stop first,
  both in one candle, neither, each stop variant, costs); changing candles after the window
  changes no label and changing candles after `D` changes no panel row; a spot check of 20 random
  discovery entries against the raw 5m candles agrees; counts per year are committed (counts only
  for 2024-26); the check command passes.
- **Worker:** opus
- **Commits:** ca71278, 6c4debe, 7db045d, cbacb74

### M2: Factors (done)

- **Delivers:** `build_factors.py` writing about 50 factors per entry for all years to
  `cache/factors.parquet`, a factor dictionary in `results/factors.md` (name, family, kind,
  formula), `check_factors.py`.
- **Done when:** for every factor, truncating the candles at `D` gives the same value (look-ahead
  check on real data for a sample of coins and days); coverage and basic distribution per factor
  are committed; the check command passes.
- **Worker:** opus
- **Commits:** a04da83, cf30df9, aabed04, 82ebcf6 (exclusions, master)

### M3: Single-factor results on 2020-23 (in progress)

- **Delivers:** `m3_single.py` and `results/m3_single.csv` / `.md` per the protocol, with the list
  of stable factors.
- **Done when:** the table covers every factor and the three labels; the sealed folder is never
  opened; the check command passes.
- **Worker:** sonnet
- **Commits:**

### M4: Combined model and frozen spec (todo)

- **Delivers:** `m4_model.py` (walk-forward GBM, seeds, the simple rule, pick rule and book on
  discovery), `results/m4_*.md`, and `results/spec.json` with the frozen choice.
- **Done when:** walk-forward results by year and seed are committed; the spec is committed and
  pushed before any sealed outcome is read; the check command passes.
- **Worker:** opus
- **Commits:**

### M5: Reveal on 2024-26 (todo)

- **Delivers:** `m5_confirm.py` reading the spec and the sealed labels once; `results/m5_*.md`.
- **Done when:** results are committed exactly as they come out; the verdict against the
  protocol's *Confirmed* rule is stated; the check command passes.
- **Worker:** sonnet
- **Commits:**

### M6: Report and dashboard (todo)

- **Delivers:** `docs/research/rebound-factors.md`, a row in `research/README.md`, a "Factors" tab
  on the session dashboard, the PR.
- **Done when:** the check command passes and the PR is open.
- **Worker:** master
- **Commits:**

## Decisions

- 2026-11-14: user chose the target (+30% / 30 d with 10% stop, also scored at 20% and no stop),
  spot candles only, find on 2020-23 and confirm on 2024-26, success as hit rate and net profit.
- 2026-11-14: entries are coin-days at UTC midnight rather than MA touches or random hours: the
  scan showed the touch carries no information, and daily entries make market and coin factors
  easy to separate.
- 2026-11-14: branch stacked on PR #14's head because `research/ma_rebound/` is not on `main`.

- 2026-11-14 (M1): 592,347 entries, 576 coins; discovery 282,923, confirm 299,711. Discovery base
  rates of hit10 by year: 27.5%, 31.1%, 22.3%, 24.5%.
- 2026-11-14 (M1): `P` must come from a candle in the 24 h before `D` (a coin delisted days earlier
  is not an entry). Stops fill at the lower of the stop price and the hitting candle's open
  (conservative on gaps). A delisting inside the window exits at the last close (`series_end`).
- 2026-11-14 (M1): the logic tests are asserts in `check_panel.py` (`--logic-only`), not pytest:
  each study has a top-level `common` module that `mypy src tests` cannot resolve.
- 2026-11-14 (M1): `labels.parquet` also holds the December 2023 gap entries, whose windows reach
  into January 2024. M3 and M4 must filter on `split == "discovery"`.

- 2026-11-14 (M2): 65 factors (20 market, 45 coin), all from daily bars built from the 5m candles;
  every one passes the truncate-at-D check. Habit and `m_moon30` are no-stop analogs.
- 2026-11-14 (M2 review): nine redenominations / token swaps (COCOS, DREP, SUN, BTCST, LUNA,
  VIDT, BNX, QUICK, and STRAX in 2024) make false hits and false factor values. Mechanical rule
  for all years in `build_exclusions.py`: a consecutive-candle open/close ratio beyond 3x is a
  break; entries with a break within 200 days before `D` or inside the window are excluded
  (1,661 discovery, 249 confirm). M3-M5 apply `cache/exclusions.parquet`.

## Blocked on

## Follow-ups

- Futures funding and open interest, and listing/news events, as further factor families.
- If confirmed: a forward paper test, then porting the rule to `src/`.
