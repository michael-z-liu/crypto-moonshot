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
  next: `Next: retrain it walk-forward (each year on all earlier years), paper-trade it next to F2, and test it on delisted coins.`,
};

window.EVOLVE = `
  <p>What the evidence says to keep, drop and try next, ordered by how much each could change the conclusion.</p>
  <h3>Keep</h3>
  <ul class="plain">
    <li><b>The trigger is the spike, the edge is the context.</b> Every method that worked buys a volume spike <i>after a market-wide crash</i>. Nothing that ignores BTC worked.</li>
    <li><b>Short time exits.</b> A 1-hour hold beat every target, stop and trailing stop that was tried.</li>
    <li><b>Ranking within a crash day.</b> The CNN's edge, and the reason the 5-slot portfolio beats the every-trade view.</li>
  </ul>
  <h3>Drop</h3>
  <ul class="plain">
    <li>Tight targets with wide stops (+1% / −5%), limit entries below the spike, breakouts, uptrend pullbacks, daily-scale volume.</li>
  </ul>
  <h3>Try next</h3>
  <ul class="plain">
    <li><b>Walk-forward CNN.</b> Retrain each year on all earlier data and test the next year. This turns one lucky-or-not test into four or five, and it's the cheapest next step (no new data).</li>
    <li><b>Delisted coins.</b> The CNN buys the coins that crashed hardest, and coins that died are missing from the data, so survivorship bias could be large here. Binance's public archive (data.binance.vision) should have them.</li>
    <li><b>1-minute candles on flush days.</b> Entering 5 minutes late costs the CNN about a third of its edge. 1m data would show how much is lost inside the spike candle and whether a 1m CNN does better.</li>
    <li><b>More market channels.</b> The BTC channels made the difference, so add ETH, an altcoin index and, from futures, funding, open interest and liquidations, which measure forced selling directly.</li>
    <li><b>Paper-trade it.</b> Run the CNN and F2 side by side in paper mode on the live stream. The biggest unknown is the fill price on flush days, and only live data answers it.</li>
  </ul>`;
