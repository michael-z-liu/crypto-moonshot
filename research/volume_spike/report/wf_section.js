"use strict";
/* Sections for the walk-forward CNN study (M9-M11) and the forecasting-engine idea. Data: D.wf. */
const WF = D.wf || null;
const WY = [2022, 2023, 2024, 2025, 2026];

function scatter(target, { pts, xTitle, yTitle, hgt = 320 }) {
  const f = frame(target, { hgt, m: { l: 58, b: 40 } });
  const xs = pts.map(p => p.x), ys = pts.map(p => p.y);
  const x0 = Math.min(0, ...xs), x1 = Math.max(...xs) * 1.08, y0 = Math.min(0, ...ys) * 1.1, y1 = Math.max(...ys) * 1.1;
  const X = lin(x0, x1, f.M.l, f.w - f.M.r), Y = lin(y0, y1, f.h - f.M.b, f.M.t);
  yAxis(f, Y, y0, y1, v => bp(v), yTitle);
  for (const t of ticks(x0, x1, 6)) el("text", { x: X(t), y: f.h - f.M.b + 16, "text-anchor": "middle" }, f.svg, bp(t));
  el("line", { class: "zero", x1: X(0), x2: X(0), y1: f.M.t, y2: f.h - f.M.b }, f.svg);
  el("text", { x: f.w - f.M.r, y: f.h - 4, "text-anchor": "end" }, f.svg, xTitle);
  for (const p of pts) {
    const c = el("circle", { cx: X(p.x), cy: Y(p.y), r: 5, fill: p.color || "var(--s1)", stroke: "var(--surface)", "stroke-width": 2 }, f.svg);
    if (p.label) el("text", { x: X(p.x) + 8, y: Y(p.y) + 4, style: "fill:var(--ink2)" }, f.svg, p.label);
    c.addEventListener("pointermove", e => showTip(e, p.tip));
    c.addEventListener("pointerleave", hideTip);
  }
}

function fan(target, ex) {
  const xs = ex.p50.map((_, i) => (i + 1) * 15);
  const f = frame(target, { hgt: 190, m: { l: 48, b: 26 } });
  const all = [...ex.p10, ...ex.p90, ...ex.real, 0];
  const y0 = Math.min(...all), y1 = Math.max(...all);
  const X = lin(0, 480, f.M.l, f.w - f.M.r), Y = lin(y0 - .3, y1 + .3, f.h - f.M.b, f.M.t);
  yAxis(f, Y, y0 - .3, y1 + .3, v => (v > 0 ? "+" : "") + v.toFixed(1) + "%");
  for (const t of [0, 120, 240, 480]) el("text", { x: X(t), y: f.h - f.M.b + 15, "text-anchor": "middle" }, f.svg, t ? t / 60 + "h" : "entry");
  const band = [...xs.map((x, i) => `${X(x)},${Y(ex.p90[i])}`), ...xs.slice().reverse().map((x, i) => `${X(x)},${Y(ex.p10[xs.length - 1 - i])}`)];
  el("polygon", { points: `${X(0)},${Y(0)} ${band.join(" ")}`, fill: "color-mix(in oklab, var(--s1) 18%, transparent)" }, f.svg);
  el("polyline", { points: [`${X(0)},${Y(0)}`, ...xs.map((x, i) => `${X(x)},${Y(ex.p50[i])}`)].join(" "), fill: "none", stroke: "var(--s1)", "stroke-width": 2 }, f.svg);
  el("polyline", { points: [`${X(0)},${Y(0)}`, ...xs.map((x, i) => `${X(x)},${Y(ex.real[i])}`)].join(" "), fill: "none", stroke: "var(--s2)", "stroke-width": 2 }, f.svg);
}

if (WF) {
  method("m-cnnwf", "Walk-forward CNN study: 27 variants, five test years", "Edge confirmed, picking fails", "best", s => {
    s.body(`<p>The first CNN result rested on one 22-month test, which I had already looked at. So I rebuilt the study
      to test every idea on <b>five</b> out-of-sample years. For each year from 2022 to 2026, a model is trained only on
      data up to the middle of the year before. It's stopped on the second half of that year, then it scores the test
      year. Scores are compared with the model's own training scores, so a threshold never uses the future.</p>
      <p>The richer dataset covers every spike on a liquid coin (≥ 5k USDT per candle; 632k spikes). It has 24 hours of
      candles and 13 channels: the coin's OHLC, volume, taker share and trade count; BTC and ETH price and volume; and
      two market-wide channels, the average move of all coins and the share of coins spiking at that moment. All
      results below use the 5-slot portfolio at 0.3% costs.</p>`);
    plate(s.main, "How the walk-forward folds work", "train → stop early → test, re-done for every year", p => {
      const rows = WY.map((y, i) => `<g transform="translate(0,${20 + i * 30})">
        <text x="0" y="14" style="fill:var(--ink)">test ${y}</text>
        <rect x="80" y="2" width="${(y - 1 - 2020 + 0.5) * 120}" height="16" rx="3" fill="color-mix(in oklab, var(--s1) 30%, var(--surface))" stroke="var(--s1)"/>
        <rect x="${80 + (y - 1 - 2020 + 0.5) * 120}" y="2" width="60" height="16" rx="3" fill="color-mix(in oklab, var(--s3) 30%, var(--surface))" stroke="var(--s3)"/>
        <rect x="${80 + (y - 2020) * 120}" y="2" width="120" height="16" rx="3" fill="color-mix(in oklab, var(--s2) 30%, var(--surface))" stroke="var(--s2)"/></g>`).join("");
      const yrs = [2020, 2021, 2022, 2023, 2024, 2025, 2026].map((y, i) => `<text x="${80 + i * 120 + 4}" y="180">${y}</text>`).join("");
      p.insertAdjacentHTML("beforeend", `<svg class="diagram" viewBox="0 0 940 190" role="img" aria-label="Walk-forward folds">${rows}${yrs}</svg>`);
      legend(p, [{ name: "train", color: "var(--s1)" }, { name: "stop early (pick epochs)", color: "var(--s3)" }, { name: "test year", color: "var(--s2)" }], true);
    });
    s.body(`<h3>What was varied</h3><p>Each variant changes one thing from the base (the first CNN: coin + BTC, 8h,
      predict 1h), trained twice with different seeds. The selection rule was written down before the screen. It judges
      2022–24 only, at the top 2% of scores (I moved the threshold from 0.5%, which gave only 12–41 trades a year,
      before looking at any results). The top four by that rule and three combinations were re-run with 5 seeds.</p>`);
    plate(s.main, "Every variant, net bp per trade by test year", "5-slot portfolio, top 2% of scores, hold = the variant's target (1h unless stated); hover for trade counts", p => {
      const v = WF.variants;
      const t = h("table");
      t.innerHTML = `<thead><tr><th>Variant</th><th>What changes</th>${WY.map(y => `<th>${y}</th>`).join("")}<th>2022–24</th><th>2025–26</th></tr></thead>`;
      const tb = h("tbody");
      for (const r of v) {
        const tr = h("tr");
        tr.innerHTML = `<td class="mono">${r.variant}</td><td style="text-align:left;white-space:normal;min-width:200px">${r.note}</td>` +
          WY.map(y => `<td class="heat" style="background:${heatColor(r["bp_" + y], 300)}" title="${num(r["n_" + y])} trades">${bp(r["bp_" + y])}</td>`).join("") +
          `<td><b>${bp(r.select_mean_bp)}</b></td><td>${bp(r.late_mean_bp)}</td>`;
        tb.appendChild(tr);
      }
      t.appendChild(tb); const w = h("div", { class: "scroll" }); w.appendChild(t); p.appendChild(w);
    });
    plate(s.main, "Did doing well in 2022–24 predict doing well in 2025–26?", "each dot is a variant; rank correlation −0.06", p => {
      const v = WF.variants;
      scatter(p, { xTitle: "2022–24 mean, net bp per trade", yTitle: "2025–26 mean",
        pts: v.map(r => ({ x: r.select_mean_bp, y: r.late_mean_bp, label: ["crash5", "combo_c", "t4h5", "market", "path_all5", "base", "combo_a"].includes(r.variant) ? r.variant : "",
          color: r.variant.startsWith("combo") || r.variant.endsWith("5") ? "var(--s2)" : "var(--s1)",
          tip: `<b>${r.variant}</b>: ${r.note}<br>2022–24 ${bp(r.select_mean_bp)} · 2025–26 ${bp(r.late_mean_bp)}` })) });
      legend(p, [{ name: "screened (2 seeds)", color: "var(--s1)" }, { name: "finalists and combos (5 seeds)", color: "var(--s2)" }], true);
    });
    s.body(`<p><b>No.</b> The variants that looked best on 2022–24 (4h targets, training only on crash-context spikes, and
      the combinations of the "best" choices) are the ones that failed in 2025–26. Across 27 variants the rank
      correlation is −0.06, and −0.43 for 2024 alone. Choosing between CNN variants on past results mostly fitted
      noise, and the 4h targets learned a bounce persistence that 2025 no longer had.</p>`);
    plate(s.main, "Four strategies, chained across the five test years", "each year is out of sample; equity starts at 1 in 2022", p => lineChart(p, { x: "time", hgt: 290, yFmt: v => v.toFixed(1) + "×",
      series: Object.entries(WF.strategies).map(([n, st], i) => ({ name: n, color: `var(--s${i + 1})`, x: st.equity.day, y: st.equity.equity })),
      marks: [{ x: "2025-01-01", label: "2025" }] }));
    plate(s.main, "The same strategies, return by year", "5-slot portfolio, 0.3% costs", p => {
      const names = Object.keys(WF.strategies);
      barChart(p, { cats: WY.map(String), hgt: 240, signColor: false, yFmt: v => (v * 100).toFixed(0) + "%",
        series: names.map((n, i) => ({ name: n, color: `var(--s${i + 1})`, vals: WY.map(y => (WF.strategies[n].yearly.find(r => r.year === y) || {}).cagr) })),
        tipExtra: (i, k) => { const r = WF.strategies[names[k]].yearly.find(x => x.year === WY[i]) || {}; return `${num(r.trades)} trades · ${bp(r.mean_bp)} per trade · max DD ${pct(r.max_dd)}`; } });
    });
    plate(s.main, "Tuning the trading rule on 2022–24 overfits as well", "average yearly return in 2025–26: the rule that was best on 2022–24 vs. the plain default (top 2%, hold 1h, 5 slots)", p => {
      const t = WF.rule_tuning.filter(r => !["path"].includes(r.variant));
      barChart(p, { cats: t.map(r => r.variant), rotate: true, hgt: 270, signColor: false, yFmt: v => (v * 100).toFixed(0) + "%",
        series: [{ name: "rule tuned on 2022–24", color: "var(--s2)", vals: t.map(r => r.best_late) }, { name: "default rule", color: "var(--s1)", vals: t.map(r => r.default_late) }],
        tipExtra: i => `tuned rule: ${t[i].best_rule}<br>2022–24: tuned ${pct(t[i].best_select)} vs default ${pct(t[i].default_select)}` });
    });
    s.body(`<p>A grid of about 200 trading rules was tried on every finalist: thresholds, slots, holding times, targets,
      stops, a BTC-flush gate, score-weighted sizing, a per-day cap and cool-downs. The rule that won on 2022–24
      (usually longer holds and fewer slots) did <i>worse</i> in 2025–26 than the default for almost every variant.</p>
      <h3>The official result and a better alternative</h3>
      <p>Following the rules I set in advance, the pick is the 4h-target CNN with a 4-hour hold and 3 slots (+72% a year on
      2022–24). It <b>lost 9% in 2025 and 32% in 2026</b>. Since picking doesn't work, the natural alternative is not to
      pick: average the scores of all 19 screened variants and use the default rule. The threshold is only matched to
      the base model's 2022–24 trade count. I decided on this after seeing that picking fails, but nothing in it is
      tuned.</p>`);
    plate(s.main, "Robustness of the ensemble, all five test years", "net bp per trade (results/m14_robust_ens_all.csv)", p => {
      const r = WF.robust.ens_all;
      barChart(p, { cats: r.map(x => x.case.replace(" (mean of 20)", "")), rotate: true, hgt: 300, series: [{ name: "all years", vals: r.map(x => x.mean_bp) }],
        tipExtra: i => WY.map(y => `${y}: ${bp(r[i]["bp_" + y])}`).join("<br>") });
    });
    const pe = WF.placebo.ens_all;
    s.facts([[bp(WF.robust.ens_all[0].mean_bp), "ensemble, average net per trade 2022–26"], [`${bp(pe.ci[0])} to ${bp(pe.ci[1])}`, "95% bootstrap interval (resampling days)"],
      [bp(pe.placebo.reduce((a, b) => a + b) / pe.placebo.length), "placebo: same days, scores shuffled within each day"],
      [pct(WF.subsets.filter(r => (r.bp_2025 + r.bp_2026) / 2 > 0).length / WF.subsets.length), "of 30 random 10-variant ensembles positive in 2025–26"]]);
    s.verdict(`<b>The CNN's ranking edge holds up out of sample, but choosing configurations on past results doesn't.</b> Simple setups
      (the base CNN, the path forecaster, or an average of all variants) traded with the default rule made money in every test year from 2022 to 2026.
      The ensemble made +192, +171, +137, +68 and +76 bp per trade, with a worst drawdown of −13%. Shuffling the scores within each day turns it negative (−77 bp),
      so the ranking itself is the edge. It shrinks a lot with time, costs and delay: 2025–26 is about +9% a year, it's about zero at a 1% round trip, and entering a
      candle late cuts the edge by half.`,
      `To evolve it: keep the model simple, average several, and spend the effort on speed (1m data, acting at the candle close) and on costs, not on tuning.`);
  });

  method("m-forecast", "Forecasting engine: predict the next 8 hours", "Shape right, size optimistic", "mixed", s => {
    s.body(`<p>Your idea: use the CNN as a forecaster of the price <i>path</i>, not a single number, and let the
      forecast set the trade. If the forecast shows a good upside, enter, and let its shape choose the target,
      the stop and when to leave.</p>
      <p>The forecasting variant predicts the net return at every 15 minutes out to 8 hours (32 points), each as a 10%,
      50% and 90% quantile, so it outputs a fan rather than a line. It's trained with a quantile (pinball) loss, which
      rewards getting the spread right as well as the middle, on all 13 channels, walk-forward like everything above.</p>`);
    const F = WF.forecast;
    plate(s.main, "Forecast fans vs. what happened", "blue: median forecast with the 10–90% band; orange: the real path. Two top-forecast spikes and one typical one per year", p => {
      const g = h("div", { class: "grid3" }); p.appendChild(g);
      const cells = F.examples.filter(e => e.year >= 2024).map(e => { const c = h("div"); c.appendChild(h("h4", {}, `${e.symbol} · ${e.time.slice(0, 16).replace("T", " ")}`));
        c.appendChild(h("p", { class: "sub" }, `${e.kind}; real at 4h ${e.real[15] > 0 ? "+" : ""}${e.real[15].toFixed(1)}%`)); g.appendChild(c); return [c, e]; });
      for (const [c, e] of cells) fan(c, e);
      legend(p, [{ name: "median forecast", color: "var(--s1)" }, { name: "real path", color: "var(--s2)" }]);
    });
    const g = s.grid();
    plate(g, "Is the band the right width?", "share of real outcomes below p10, inside the band, above p90 (ideal 10 / 80 / 10%)", p => lineChart(p, { hgt: 220, yFmt: v => (v * 100).toFixed(0) + "%", xFmt: v => v + " min",
      xTicks: [15, 120, 240, 480], series: [["inside", "inside the band", "var(--s1)"], ["below_p10", "below p10", "var(--s2)"], ["above_p90", "above p90", "var(--s3)"]]
        .map(([k, n, c]) => ({ name: n, color: c, x: F.calibration.map(r => r.minutes), y: F.calibration.map(r => r[k]) })) }));
    plate(g, "Does the forecast rank spikes correctly?", "correlation of the median forecast with the real return, by horizon", p => lineChart(p, { hgt: 220, yFmt: v => v.toFixed(2), xFmt: v => v + " min",
      xTicks: [15, 120, 240, 480], series: [{ name: "correlation", color: "var(--s1)", x: F.calibration.map(r => r.minutes), y: F.calibration.map(r => r.corr) },
        { name: "rank correlation", color: "var(--s3)", x: F.calibration.map(r => r.minutes), y: F.calibration.map(r => r.rank_corr) }] }));
    plate(s.main, "Forecast vs. real average path, by forecast-peak decile", "dashed: forecast; solid: what happened (each path clipped at ±30%)", p => {
      const pick = [[10, "var(--s1)"], [9, "var(--s3)"], [5, "var(--muted)"], [1, "var(--s2)"]];
      const xs = [0, ...F.calibration.map(r => r.minutes)];
      lineChart(p, { hgt: 260, yFmt: v => (v > 0 ? "+" : "") + v.toFixed(1) + "%", xFmt: v => v + " min", xTicks: [0, 60, 120, 240, 480],
        series: pick.flatMap(([d, c]) => { const r = F.deciles.find(x => x.decile === d);
          return [{ name: `decile ${d}, real`, color: c, x: xs, y: r.real_path }, { name: `decile ${d}, forecast`, color: c, dash: "5 4", x: xs, y: r.forecast_path }]; }) });
    });
    s.body(`<p>The forecaster is <b>well calibrated on average</b>. The band holds 73–78% of outcomes (80% intended,
      so slightly too narrow), and the average forecast matches the average outcome. It ranks spikes with a
      correlation of about 0.30 at 15 minutes, falling to 0.21 at 8 hours. The shape is right: the top decile does go
      up and the bottom decile does go down. But the top forecasts are <b>too optimistic in size</b>. The top decile
      forecasts a +2.1% peak, and the real path averages +0.4% after 1 hour and +0.7% after 4 hours.</p>`);
    plate(s.main, "Letting the forecast set the trade", "path forecaster (5 seeds), 5 slots, 0.3% costs; net return per year (hover for trades and bp per trade)", p => {
      const R = WF.forecast_rules;
      const t = h("table");
      t.innerHTML = `<thead><tr><th>Rule</th>${WY.map(y => `<th>${y}</th>`).join("")}</tr></thead>`;
      const tb = h("tbody");
      for (const r of R) { const tr = h("tr"); tr.innerHTML = `<td style="white-space:normal;min-width:220px">${r.rule}</td>` + WY.map(y => { const x = r.yearly.find(z => z.year === y) || {};
        return `<td class="heat" style="background:${heatColor((x.cagr || 0) * 1e4, 6000)}" title="${num(x.trades)} trades · ${bp(x.mean_bp)} per trade · max DD ${pct(x.max_dd)}">${pct(x.cagr)}</td>`; }).join(""); tb.appendChild(tr); }
      t.appendChild(tb); const w = h("div", { class: "scroll" }); w.appendChild(t); p.appendChild(w);
    });
    s.verdict(`<b>The forecast gets the direction and the rough shape right, but it can't yet be trusted to set targets.</b>
      Exiting at the forecast peak on its own is worse than a plain 1-hour hold. Adding a target at the optimistic band and a stop at the
      pessimistic band trades much more often. That made +211% in 2025 (+313% looking 8h ahead) but went flat or negative in 2026 and failed the 2022–24 drawdown limit.
      Requiring a +2% forecast upside was positive every year but faded to +1% in 2026. The forecast is most useful as a <i>ranking</i>,
      which is how the best strategies here use it.`,
      `To evolve it: shrink the forecast toward its historical accuracy before using it for targets, check whether the band
      widens on the right days (risk sizing), and train it on 1m data so the first 15 minutes are sharper.`);
  });
}
