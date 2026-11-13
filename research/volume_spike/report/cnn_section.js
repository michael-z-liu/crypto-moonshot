"use strict";
/* The CNN section (M7). Text that depends on the test result comes from window.CNN_TEXT, set in
   report/cnn_text.js, which is written after the one-time test. */
method("m-cnn", "Convolutional network on the raw candles", (window.CNN_TEXT || {}).chip || "Pending", (window.CNN_TEXT || {}).chipClass || "base", s => {
  const T = window.CNN_TEXT || {};
  s.body(`<p>Your suggestion: instead of hand-made features, let a convolutional network look at the
    raw candles before each spike and learn the pattern itself. A 1D CNN slides small filters along
    the time axis, so it can pick up shapes like "a slow bleed, then a capitulation candle with heavy
    selling", whatever their exact size.</p>
    <p>Each spike becomes a picture of the 8 hours before it (96 candles), with one row per input
    channel. The network is trained on 2020–23 to predict the 1h net return. Its settings and
    epochs were chosen on 2024, and it was tested once on 2025–26. It was scored like the boosted
    trees: trade only the spikes it scores highest.</p>`);
  plate(s.main, "What the network sees and how it decides", "96 candles × 6 coin channels (+2 BTC channels in the second variant), 54k weights, an average of 3 training runs", p => {
    p.insertAdjacentHTML("beforeend", `
    <svg class="diagram" viewBox="0 0 900 300" role="img" aria-label="CNN architecture">
      <defs><marker id="ah3" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="var(--muted)"/></marker></defs>
      <text class="t" x="10" y="18">Input: the last 96 candles (8h), one row per channel</text>
      ${["open vs. spike close", "high", "low", "close", "volume ÷ day's median", "taker-buy share", "BTC close", "BTC volume"].map((c, i) => `
        <rect x="10" y="${30 + i * 26}" width="250" height="20" rx="3" fill="color-mix(in oklab, ${i < 6 ? "var(--s1)" : "var(--s6)"} ${14 + (i % 3) * 6}%, var(--surface))" stroke="${i < 6 ? "var(--s1)" : "var(--s6)"}" stroke-opacity=".5"/>
        <text x="18" y="${44 + i * 26}" style="fill:var(--ink)">${c}</text>`).join("")}
      <rect x="236" y="30" width="22" height="204" rx="3" fill="none" stroke="var(--s2)" stroke-width="2"/>
      <text x="236" y="250" style="fill:var(--s2)">spike candle</text>
      <text x="10" y="270">+ the coin's liquidity (median USDT per candle) as one extra number</text>
      <text x="10" y="288">rows 7–8 only in the "coin + BTC" variant</text>
      <path class="arrow" d="M270,130 H306" marker-end="url(#ah3)" style="marker-end:url(#ah3)"/>
      <rect class="box" x="310" y="60" width="120" height="140" rx="8"/><text class="t" x="322" y="84">Conv × 2</text><text x="322" y="104">32 filters,</text><text x="322" y="120">5 candles wide</text><text x="322" y="146">pool ÷2 → 48</text>
      <path class="arrow" d="M430,130 H456" style="marker-end:url(#ah3)"/>
      <rect class="box" x="460" y="60" width="120" height="140" rx="8"/><text class="t" x="472" y="84">Conv × 3</text><text x="472" y="104">64 filters,</text><text x="472" y="120">5 then 3 wide</text><text x="472" y="146">pool ÷2 → 24</text>
      <path class="arrow" d="M580,130 H606" style="marker-end:url(#ah3)"/>
      <rect class="box" x="610" y="60" width="130" height="140" rx="8"/><text class="t" x="622" y="84">Summarise</text><text x="622" y="104">average over time</text><text x="622" y="120">max over time</text><text x="622" y="136">the last step</text><text x="622" y="156">+ liquidity</text>
      <path class="arrow" d="M740,130 H766" style="marker-end:url(#ah3)"/>
      <rect class="hl" x="770" y="80" width="120" height="100" rx="8"/><text class="t" x="782" y="106">Predicted</text><text class="t" x="782" y="124">1h net return</text><text x="782" y="150">trade the top</text><text x="782" y="166">0.1–1% of spikes</text>
    </svg>`);
  });
  const g = s.grid();
  for (const v of ["coin", "coin_btc"]) plate(g, v === "coin" ? "Training, coin channels only" : "Training, coin + BTC channels", "validation (2024) correlation between prediction and result, per epoch; each line is one training run", p => {
    const rows = D.cnn_training.filter(r => r.variant === v);
    lineChart(p, { hgt: 190, yFmt: x => x.toFixed(2), xFmt: x => "epoch " + (x + 1), xTicks: [0, 2, 4, 7],
      series: [0, 1, 2].map(sd => ({ name: "run " + (sd + 1), color: `var(--s${sd + 1})`, x: rows.filter(r => r.seed === sd).map(r => r.epoch), y: rows.filter(r => r.seed === sd).map(r => r.valid_corr) })) });
  });
  if (T.trainingNote) s.body(`<p class="note">${T.trainingNote}</p>`);
  plate(s.main, "Net per trade of the top-scored spikes: CNN vs. boosted trees", "same spikes, same target, same thresholds (from the 2020–23 scores); one trade per coin at a time. Hover for the result without 2025-10-10", p => {
    const models = ["cnn_coin", "cnn_coin_btc", "gbm"], names = { cnn_coin: "CNN, coin only", cnn_coin_btc: "CNN, coin + BTC", gbm: "boosted trees" };
    const g2 = h("div", { class: "grid3" }); p.appendChild(g2);
    const qs = models.map(m => { const q = h("div"); q.appendChild(h("h4", {}, names[m])); g2.appendChild(q); return q; });
    models.forEach((m, i) => modelChart(qs[i], D.cnn_selection.filter(r => r.model === m), "quantile", "split"));
  });
  plate(s.main, "Selection results in full", "results/m11_cnn_selection.csv", p => tableFrom(p,
    [["Model", r => r.model], ["Top", r => (100 - r.quantile * 100).toFixed(1) + "%"], ["Period", r => r.split], ["Trades", r => num(r.trades)], ["Net / trade", r => bp(r.mean_bp)],
     ["Net / day", r => bp(r.day_bp)], ["Days +", r => pct(r.win_days)], ["Without 10-10", r => bp(r.mean_bp_ex1010)]], D.cnn_selection));
  plate(s.main, "As an enter-or-skip filter on F1", "F1 trades split by the model's score (cut at the 2020–23 median of F1 scores); net bp per trade", p => {
    const groups = ["all F1", "enter (score above)", "skip (score below)"];
    const g3 = h("div", { class: "grid3" }); p.appendChild(g3);
    const ms = ["cnn_coin", "cnn_coin_btc", "gbm"];
    const qs = ms.map(m => { const q = h("div"); q.appendChild(h("h4", {}, { cnn_coin: "CNN, coin only", cnn_coin_btc: "CNN, coin + BTC", gbm: "boosted trees" }[m])); g3.appendChild(q); return q; });
    for (const [mi, m] of ms.entries()) { const q = qs[mi];
      const rows = D.cnn_f1.filter(r => r.model === m);
      barChart(q, { cats: ["2020–23", "2024", "2025–26"], hgt: 210, signColor: false,
        series: groups.map((gname, i) => ({ name: gname, color: ["var(--muted)", "var(--s1)", "var(--s2)"][i], vals: ["train", "valid", "test"].map(sp => (rows.find(r => r.split === sp && r.group === gname) || {}).mean_bp) })),
        tipExtra: (i, k) => { const r = rows.find(x => x.split === ["train", "valid", "test"][i] && x.group === groups[k]) || {}; return `${num(r.trades)} trades · per day ${bp(r.day_bp)}`; } }); }
  });
  const prof = (D.extras || {}).cnn_profiles;
  if (prof) {
    const g4 = s.grid();
    const keys = Object.keys(prof);
    plate(g4, "What the top-scored spikes look like: price", "average close over the 8h before the spike, relative to the spike close", p => lineChart(p, { hgt: 210, yFmt: v => bp(v), xFmt: x => (x - 95) * 5 + " min", xTicks: [0, 24, 48, 72, 95],
      series: keys.map((k, i) => ({ name: `${k} (${num(prof[k].n)})`, color: i ? "var(--muted)" : "var(--s1)", x: prof[k].close_bp.map((_, j) => j), y: prof[k].close_bp })) }));
    plate(g4, "… and volume", "average log(1 + volume ÷ day's median)", p => lineChart(p, { hgt: 210, yFmt: v => v.toFixed(1), xFmt: x => (x - 95) * 5 + " min", xTicks: [0, 24, 48, 72, 95],
      series: keys.map((k, i) => ({ name: k, color: i ? "var(--muted)" : "var(--s1)", x: prof[k].volume.map((_, j) => j), y: prof[k].volume })) }));
    plate(g4, "… and BTC", "average BTC close over the same 8h", p => lineChart(p, { hgt: 210, yFmt: v => bp(v), xFmt: x => (x - 95) * 5 + " min", xTicks: [0, 24, 48, 72, 95],
      series: keys.map((k, i) => ({ name: k, color: i ? "var(--muted)" : "var(--s1)", x: prof[k].btc_bp.map((_, j) => j), y: prof[k].btc_bp })) }));
    plate(g4, "… and taker-buy share", "average share of volume bought by takers", p => lineChart(p, { hgt: 210, yFmt: v => (v * 100).toFixed(0) + "%", xFmt: x => (x - 95) * 5 + " min", xTicks: [0, 24, 48, 72, 95],
      series: keys.map((k, i) => ({ name: k, color: i ? "var(--muted)" : "var(--s1)", x: prof[k].taker.map((_, j) => j), y: prof[k].taker })) }));
  }
  if (T.portfolioIntro) s.body(T.portfolioIntro);
  const rebased = (k, from) => { const e = D.equity[k], i0 = e.day.findIndex(d => d >= from), b = e.equity[Math.max(0, i0 - 1)];
    return { x: e.day.slice(i0), y: e.equity.slice(i0).map(v => +(v / b).toFixed(4)) }; };
  const trio = [["CNN top 0.5%", "CNN, coin + BTC", "var(--s1)"], ["GBM top 0.5%", "boosted trees", "var(--s2)"], ["F2", "F2 rule", "var(--s3)"]].filter(([k]) => D.equity[k]);
  plate(s.main, "5-slot portfolio from 2024: CNN vs. boosted trees vs. F2", "equity rebased to 1 on 2024-01-01; highest score first when signals coincide. 2024 picked the variant, 2025–26 is the test", p => lineChart(p, { x: "time", yFmt: v => v.toFixed(2) + "×", hgt: 290,
    series: trio.map(([k, n, c]) => ({ name: n, color: c, ...rebased(k, "2024-01-01") })),
    xTicks: ["2024-01-01", "2024-07-01", "2025-01-01", "2025-07-01", "2026-01-01", "2026-07-01"].map(d => +new Date(d)), xFmt: t => new Date(t).toISOString().slice(0, 7),
    marks: [{ x: "2025-01-01", label: "test" }] }));
  plate(s.main, "The same portfolios over the whole period (log scale)", "2020–23 is in-sample for both models: they were fitted on it", p => lineChart(p, { x: "time", log: true, yFmt: v => v + "×", hgt: 240,
    series: trio.map(([k, n, c]) => ({ name: n, color: c, x: D.equity[k].day, y: D.equity[k].equity })),
    marks: [{ x: "2024-01-01", label: "valid" }, { x: "2025-01-01", label: "test" }] }));
  plate(s.main, "Portfolio results by period", "results/m11_cnn_portfolio.csv (post-hoc: the portfolio view was not the pre-registered measure)", p => tableFrom(p,
    [["Strategy", r => ({ cnn_coin_btc: "CNN, coin + BTC", gbm: "boosted trees", "F2 rule": "F2 rule" })[r.model] || r.model], ["Period", r => r.split], ["Trades", r => num(r.trades)], ["Per week", r => r.trades_per_week.toFixed(1)],
     ["Net / trade", r => bp(r.mean_bp)], ["Win", r => pct(r.win)], ["CAGR", r => pct(r.cagr)], ["Max DD", r => pct(r.max_dd)], ["By year", r => r.by_year]], D.cnn_portfolio));
  plate(s.main, "Stress tests of the CNN portfolio, 2025–26", "net bp per trade; results/m11_cnn_stress.csv", p => {
    const r = D.cnn_stress;
    barChart(p, { cats: r.map(x => (x.variant === "coin" ? "coin-only CNN" : x.case)), rotate: true, hgt: 270, series: [{ name: "net / trade", vals: r.map(x => x.mean_bp) }],
      tipExtra: i => `CAGR ${pct(r[i].cagr)} · max DD ${pct(r[i].max_dd)}<br>${num(r[i].trades)} trades · ${pct(r[i].win)} win` });
  });
  if (D.summary.some(r => r.method === "cnn_top")) plate(s.main, "CNN top 0.5%, every trade (no slot limit), net per trade by year", "the measure that looked bad: losses pile up on days with dozens of signals", p => yearlyChart(p, "cnn_top"));
  if (T.extra) s.body(T.extra);
  s.verdict(T.verdict || "Pending the one-time test.", T.next);
});
