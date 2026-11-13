"use strict";
/* Text for the CNN section and the closing section, written after the one-time test (M7). */
window.CNN_TEXT = {
  chip: "Most promising so far",
  chipClass: "best",
  trainingNote: `Both variants overfit quickly: the 2024 correlation peaks after 1–3 epochs while the
    training error keeps falling, so each run keeps its best-epoch weights. The coin + BTC variant was
    better on 2024 at every threshold, so it was chosen before the test, at the top 0.5% of scores (the
    best 2024 per-day result among thresholds with at least 20 signal days). That choice was committed
    before 2025–26 was opened.`,
  portfolioIntro: `<p>The pre-registered measure (every qualifying trade) gave a mixed answer. On
    2025–26 the chosen CNN lost 110 bp per trade but made +177 bp <i>per signal day</i>, with 60% of
    101 days positive. The losses pile up on a few days with dozens of signals at once. A real account
    can only hold a few positions, so I ran the 5-slot portfolio, taking the highest scores first.
    That check was decided <i>after</i> seeing the test, so treat it as post-hoc.</p>`,
  extra: `<h3>Is it just buying the deepest crashes?</h3>
    <p>Partly. Its top picks in 2025–26 had fallen a median 29% in 4 hours, with BTC's volume
    spiking in 99% of cases. But simple rules that buy the deepest flushes don't reproduce it in the
    same portfolio: a −15% threshold makes +27 to +85 bp per trade, −20% loses, and −25% looks huge only
    because of 2025-10-10. Leave that day out and the −25% rules make −9 to +139 bp on 10–22 trades. The CNN
    keeps +114 bp on 235 trades without that day (<code>results/m11_rule_compare.csv</code>). Its
    edge is in <b>ranking signals within a crash day</b>: which coins to buy when 30 of them spike
    at once. Boosted trees on the 36 hand-made features had the same information and did not manage this.</p>`,
  verdict: `<b>In the 5-slot portfolio, the CNN is the only method that stays clearly positive in 2025–26:</b>
    +114 bp per trade, +33% a year, −11% worst drawdown, positive in both 2025 and 2026. It holds with
    2025-10-10 left out, with 3 or 10 slots, at 1% costs (+44 bp) and entering a candle late (+79 bp).
    The coin-only variant fails (−21 bp), so the market context matters more than the coin's own candles.
    It doesn't work as an enter-or-skip filter on F1. Caveats: the portfolio view is post-hoc,
    it's one 22-month test with 254 trades, and the survivorship bias is at its worst here, since it buys the coins that crashed hardest.`,
  next: `The walk-forward study below retrains it every year and tests 27 variants of it.`,
};

window.EVOLVE = `
  <p>What the evidence says to keep, drop and try next, ordered by how much each could change the conclusion.</p>
  <h3>Keep</h3>
  <ul class="plain">
    <li><b>The trigger is the spike, the edge is the context.</b> Every method that worked buys a volume spike <i>after a market-wide crash</i>. Nothing that ignores BTC worked.</li>
    <li><b>A CNN to rank spikes within a crash day.</b> The walk-forward study confirms the edge in all five test years, and shuffling the scores within a day turns it negative.</li>
    <li><b>Simple settings, averaged.</b> The base CNN, the path forecaster and an equal-weight ensemble with a plain 1-hour hold all made money every year from 2022 to 2026.</li>
  </ul>
  <h3>Drop</h3>
  <ul class="plain">
    <li>Picking the "best" variant or trading rule on past years. Across 27 variants, 2022–24 results did not predict 2025–26 results (rank correlation −0.06).</li>
    <li>Tight targets with wide stops (+1% / −5%), limit entries below the spike, breakouts, uptrend pullbacks, daily-scale volume, and 4-hour targets for the CNN.</li>
  </ul>
  <h3>Try next</h3>
  <ul class="plain">
    <li><b>Speed.</b> Entering one candle late halves the ensemble's edge. Get 1-minute candles (or live trades) on flush days, and act within seconds of the candle close.</li>
    <li><b>Costs.</b> At a 1% round trip the 2025–26 edge is gone. Measure real spreads and slippage on flush days in paper mode before sizing anything.</li>
    <li><b>Delisted coins.</b> The CNN buys the coins that crashed hardest, and coins that died are missing from the data, so survivorship bias could be large. Binance's public archive (data.binance.vision) should have them.</li>
    <li><b>The forecast as a risk input.</b> The path forecaster's shape is right but its size is optimistic. Shrink it toward its historical accuracy, then use the band width for position sizing rather than for targets.</li>
    <li><b>Paper-trade the ensemble</b> next to F2 on the live stream. Only live data shows real fills on crash days.</li>
  </ul>`;

window.SURV_TEXT = {
  chip: "Edge smaller, still there",
  chipClass: "mixed",
  extra: `<h3>What changes</h3>
    <ul class="plain">
      <li><b>Rules without the BTC filter break.</b> F3 (a deep crash, any market) goes from +42 to −139 bp per trade in 2025–26 with the
        delisted coins (−46% a year, −75% drawdown), and to −148 bp with every coin. Coins crashing on their own, while the market is calm, are
        often dying. F3's worst real trade bought LUNA on 12 May 2022 and lost 99.8% in four hours.</li>
      <li><b>F2 (deep, liquid, market-wide flush) holds up.</b> It made money every year from 2020 to 2024 in every universe (+75 to +319 bp per trade),
        lost a little in 2025 (−7 to −44 bp) and made +187 to +313 bp in 2026.</li>
      <li><b>The CNNs keep a smaller edge.</b> With every USDT pair added, the ensemble makes +105 bp per trade over 2022–26 (95% interval +41 to +173),
        against +138 on the pool, and the path forecaster +119 bp. Shuffling the scores within each day still turns both negative (about −78 bp).
        2025 is the weak year: +20 bp for the ensemble, which turns negative at 0.6% costs or one candle late.</li>
      <li><b>The added coins are most of the trading.</b> On the full universe, 50–87% of the CNN trades in 2022–25 are in coins outside today's top 200,
        so the pool-only results described a different strategy from the one that would really have run.</li>
    </ul>`,
  verdict: `<b>Survivorship bias was real but didn't create the edge.</b> Adding 430 delisted and smaller coins killed the rules that buy any deep crash,
    left the market-wide flush rule (F2) and the CNN ranking positive in most years, and made 2025 close to break-even for most of them.
    What decides whether this is tradable is now costs and speed: at a 1% round trip, or one candle late, the 2025 edge is gone.`,
  next: `Next: measure real fills and latency in paper mode (the trading-engine session), and get 1m candles for flush days.`,
};
