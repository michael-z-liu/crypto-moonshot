"use strict";
/* Example trades (report_examples.py): real trades of the path forecaster on every USDT pair. Data: D.trade_examples. */
const EX = D.trade_examples || null;

/* candles around one trade, with the forecast made at entry drawn over the next 8 hours */
function tradeChart(target, ex) {
  const f = frame(target, { hgt: 300, m: { l: 50, b: 30, r: 12, t: 18 } });
  const n = ex.t.length, spikeI = ex.t.findIndex(t => +new Date(t) === +new Date(ex.time));
  const ei = spikeI + 1, xi = spikeI + ex.hold;   /* entry candle (buy at its open), exit candle (sell at its close) */
  const pTop = f.M.t, pBot = f.M.t + f.ih * .72, vTop = pBot + 12, vBot = f.h - f.M.b;
  const lg = v => Math.log(v / ex.entry), lgp = v => Math.log(1 + v / 100);
  const lo = Math.min(...ex.low.map(lg), ...ex.p10.map(lgp)), hi = Math.max(...ex.high.map(lg), ...ex.p90.map(lgp));
  const Y = lin(lo, hi, pBot, pTop), Yp = p => Y(lg(p));
  const cw = f.iw / n, X = i => f.M.l + (i + .5) * cw, bw = Math.max(1, cw * .7);
  /* gridlines at round real moves (the scale is logarithmic), at least 24px apart, out from 0% */
  const NICE = [1, 2, 3, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500, 1000];
  const keep = [0], room = 24;
  for (const sign of [1, -1]) {
    let last = Y(0);
    for (const v of NICE.map(x => sign * x).filter(x => x > -100)) {
      const yy = Y(Math.log(1 + v / 100));
      if (yy < pTop - 1 || yy > pBot + 1) continue;
      if (Math.abs(yy - last) >= room) { keep.push(v); last = yy; }
    }
  }
  for (const v of keep) {
    const yy = Y(Math.log(1 + v / 100));
    el("line", { class: v === 0 ? "zero" : "gl", x1: f.M.l, x2: f.w - f.M.r, y1: yy, y2: yy }, f.svg);
    el("text", { x: f.M.l - 6, y: yy + 3.5, "text-anchor": "end" }, f.svg, (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v) + "%");
  }
  el("text", { x: f.M.l, y: pTop - 5 }, f.svg, "price vs. entry");
  /* holding window */
  el("rect", { x: X(ei) - cw / 2, y: pTop, width: (xi - ei + 1) * cw, height: vBot - pTop,
    fill: "color-mix(in oklab, var(--accent) 10%, transparent)" }, f.svg);
  /* forecast band and median, from the entry open to 8h */
  const fx = ex.steps.map(s => X(ei + s - 1)), x0 = X(ei) - cw / 2;
  const band = [...fx.map((x, i) => `${x},${Y(lgp(ex.p90[i]))}`), ...fx.map((x, i) => `${x},${Y(lgp(ex.p10[i]))}`).reverse()];
  el("polygon", { points: `${x0},${Y(0)} ${band.join(" ")}`, fill: "color-mix(in oklab, var(--s1) 16%, transparent)" }, f.svg);
  for (let i = 0; i < n; i++) {
    const up = ex.close[i] >= ex.open[i], color = up ? "var(--pos)" : "var(--neg)";
    el("line", { x1: X(i), x2: X(i), y1: Yp(ex.high[i]), y2: Yp(ex.low[i]), stroke: color, "stroke-width": 1 }, f.svg);
    const a = Yp(Math.max(ex.open[i], ex.close[i])), b = Yp(Math.min(ex.open[i], ex.close[i]));
    el("rect", { x: X(i) - bw / 2, y: a, width: bw, height: Math.max(1, b - a), fill: color }, f.svg);
  }
  el("polyline", { points: [`${x0},${Y(0)}`, ...fx.map((x, i) => `${x},${Y(lgp(ex.p50[i]))}`)].join(" "),
    fill: "none", stroke: "var(--s1)", "stroke-width": 2, "stroke-linejoin": "round" }, f.svg);
  /* volume, taker buys darker */
  const V = lin(0, Math.max(...ex.volume), vBot, vTop);
  for (let i = 0; i < n; i++) {
    el("rect", { x: X(i) - bw / 2, y: V(ex.volume[i]), width: bw, height: V(0) - V(ex.volume[i]), fill: "var(--axis)" }, f.svg);
    el("rect", { x: X(i) - bw / 2, y: V(ex.taker_buy[i]), width: bw, height: V(0) - V(ex.taker_buy[i]), fill: i === spikeI ? "var(--s2)" : "var(--muted)" }, f.svg);
  }
  /* entry and exit */
  const xY = Yp(ex.close[xi]);
  for (const [cx, cy] of [[x0, Y(0)], [X(xi), xY]]) el("circle", { cx, cy, r: 4.5, fill: "var(--ink)", stroke: "var(--surface)", "stroke-width": 2 }, f.svg);
  const net = ex.ret_bp / 100;
  el("text", { x: X(xi) + 7, y: xY + (net >= 0 ? -6 : 13), style: "fill:var(--ink);font-weight:600" }, f.svg, `${net > 0 ? "+" : "−"}${Math.abs(net).toFixed(1)}%`);
  for (const [i, s] of [[spikeI - 48, "−4h"], [ei, "entry"], [ei + 47, "+4h"], [ei + 95, "+8h"]]) {
    if (i < 0 || i >= n) continue;
    el("text", { x: X(i), y: f.h - 10, "text-anchor": "middle" }, f.svg, s);
  }
  const hit = el("rect", { x: f.M.l, y: f.M.t, width: f.iw, height: f.ih, fill: "transparent" }, f.svg);
  hit.addEventListener("pointermove", e => {
    const r = f.svg.getBoundingClientRect(), px = (e.clientX - r.left) * f.w / r.width;
    const i = Math.max(0, Math.min(n - 1, Math.floor((px - f.M.l) / cw)));
    const rel = k => { const v = (ex[k][i] / ex.entry - 1) * 100; return (v > 0 ? "+" : "") + v.toFixed(2) + "%"; };
    const j = ex.steps.findIndex(s => ei + s - 1 >= i);
    const fc = i >= ei && j >= 0 ? `<br>forecast for ${ex.steps[j] * 5} min: ${ex.p50[j].toFixed(1)}% (${ex.p10[j].toFixed(1)} to ${ex.p90[j].toFixed(1)})` : "";
    const tag = i === spikeI ? " · spike" : i === ei ? " · buy at open" : i === xi ? " · sell at close" : "";
    showTip(e, `<b>${ex.t[i].slice(0, 16).replace("T", " ")} UTC</b>${tag}<br>close ${rel("close")} · high ${rel("high")} · low ${rel("low")}${fc}<br>taker buys ${pct(ex.taker_buy[i] / (ex.volume[i] || 1))}`);
  });
  hit.addEventListener("pointerleave", hideTip);
}

function exampleSub(ex) {
  const s = v => (v > 0 ? "+" : "") + (v * 100).toFixed(0) + "%";
  return `${ex.time.slice(0, 16).replace("T", " ")} UTC · ${ex.group}<br>` +
    `Coin ${s(ex.coin_ret_4h)} in 4h, ${s(ex.coin_ret_24h)} in 24h · spike ${Math.round(ex.vol_mult)}× volume · ` +
    `BTC ${s(ex.btc_ret_4h)} in 4h at ${ex.btc_vol_mult.toFixed(1)}× volume<br>` +
    `Forecast for 1h: ${ex.p50[3] > 0 ? "+" : ""}${ex.p50[3].toFixed(1)}% (10–90%: ${ex.p10[3].toFixed(1)} to ${ex.p90[3].toFixed(1)}%)`;
}

if (EX) {
  method("m-examples", "Example trades: the path forecaster", "Examples", "base", s => {
    const T = EX.trades, win = T.filter(t => t.win), loss = T.filter(t => !t.win);
    const avg = (a, k) => a.reduce((x, t) => x + t[k], 0) / a.length;
    const med = (a, k) => { const v = a.map(t => t[k]).sort((x, y) => x - y); return v[Math.floor(v.length / 2)]; };
    s.body(`<p>Real trades from the path forecaster on every USDT pair (top 200, delisted and smaller coins), with the
      default rule: buy at the next candle's open when the forecast ranks in the top 2%, sell at the close one hour
      later, at most 5 open at once, 0.3% costs. These are trades the 5-slot portfolio actually took. Two winners and
      two losers per year were drawn at random, so they're typical trades, not the best and worst.</p>
      <p><b>How to read each chart:</b> candles from 4 hours before the spike to 8 hours after, measured against the
      entry price. The blue line is the forecast's median path, made at the moment of entry; the blue band is its
      10–90% range. The shaded strip is the hour the trade is held; the dots are the buy and the sell. Volume is at the
      bottom, with taker buys darker and the spike candle in orange. Hover for the numbers.</p>`);
    s.facts([[num(T.length), "trades, 2022 to Oct 2026"], [pct(win.length / T.length), "of trades are winners"],
      ["+" + (avg(win, "ret_bp") / 100).toFixed(1) + "%", "average winner, after costs"],
      [(avg(loss, "ret_bp") / 100).toFixed(1).replace("-", "−") + "%", "average loser, after costs"],
      ["+" + (med(T, "ret_bp") / 100).toFixed(1) + "%", "median trade"], ["+" + (med(T, "p50_bp") / 100).toFixed(1) + "%", "median 1h forecast"]]);
    legend(s.main, [{ name: "forecast median", color: "var(--s1)" }, { name: "forecast 10–90% band", color: "color-mix(in oklab, var(--s1) 30%, transparent)" },
      { name: "holding hour", color: "color-mix(in oklab, var(--accent) 22%, transparent)" }, { name: "spike candle (taker buys)", color: "var(--s2)" }], true);
    for (const [w, label] of [[true, "10 winners"], [false, "10 losers"]]) {
      s.body(`<h3>${label}</h3>`);
      const g = s.grid();
      for (const ex of EX.examples.filter(e => e.win === w)) {
        const r = ex.ret_bp / 100;
        plate(g, `${ex.symbol.replace("USDT", "")} · ${r > 0 ? "+" : "−"}${Math.abs(r).toFixed(1)}%`, exampleSub(ex), p => tradeChart(p, ex));
      }
    }
    s.body(`<h3>All ${num(T.length)} trades</h3>`);
    const g2 = s.grid();
    plate(g2, "Forecast vs. what happened, every trade", "1h median forecast against the trade's net result; results beyond ±20% are pinned to the edge", p => {
      const f = frame(p, { hgt: 300, m: { l: 52, b: 36 } });
      const cl = v => Math.max(-20, Math.min(20, v / 100));
      const xs = T.map(t => t.p50_bp / 100), x1 = Math.min(12, Math.max(...xs));
      const X = lin(0, x1, f.M.l, f.w - f.M.r), Y = lin(-20, 20, f.h - f.M.b, f.M.t);
      yAxis(f, Y, -20, 20, v => (v > 0 ? "+" : "") + v + "%", "net result");
      for (const t of ticks(0, x1, 6)) el("text", { x: X(t), y: f.h - f.M.b + 15, "text-anchor": "middle" }, f.svg, "+" + t + "%");
      el("text", { x: f.w - f.M.r, y: f.h - 4, "text-anchor": "end" }, f.svg, "1h forecast (median)");
      el("line", { x1: X(0), x2: X(Math.min(x1, 20)), y1: Y(0), y2: Y(Math.min(x1, 20)), stroke: "var(--ink2)", "stroke-dasharray": "4 4" }, f.svg);
      for (const t of T) {
        const c = el("circle", { cx: X(Math.min(x1, t.p50_bp / 100)), cy: Y(cl(t.ret_bp)), r: 3, fill: "var(--s1)", "fill-opacity": .45 }, f.svg);
        c.addEventListener("pointermove", e => showTip(e, `<b>${t.symbol}</b> ${t.year}<br>forecast +${(t.p50_bp / 100).toFixed(1)}% · result ${(t.ret_bp / 100).toFixed(1)}%`));
        c.addEventListener("pointerleave", hideTip);
      }
      el("text", { x: X(Math.min(x1, 10)) - 4, y: Y(Math.min(x1, 10)) - 6, "text-anchor": "end" }, f.svg, "result = forecast");
    });
    plate(s.main, "How the results are spread", "number of trades by net result; each bar covers 1%, labelled by its lower edge", p => {
      const edges = []; for (let v = -10; v <= 10; v++) edges.push(v);
      const sg = v => (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v);
      const cats = ["<−10", ...edges.slice(0, -1).map(sg), "≥+10"];
      const vals = new Array(cats.length).fill(0);
      for (const t of T) { const r = t.ret_bp / 100; vals[r < -10 ? 0 : r >= 10 ? cats.length - 1 : Math.floor(r) + 11] += 1; }
      barChart(p, { cats, hgt: 260, signColor: false, yFmt: v => num(v), series: [{ name: "trades", color: "var(--s1)", vals }],
        tipExtra: i => i === 0 ? "lost more than 10%" : i === cats.length - 1 ? "made 10% or more" : `from ${sg(edges[i - 1])}% to ${sg(edges[i])}%` });
    });
    plate(s.main, "Does a bigger forecast mean a better trade?", "all trades split into four equal groups by the 1h forecast", p => {
      const sorted = T.slice().sort((a, b) => a.p50_bp - b.p50_bp), q = Math.ceil(sorted.length / 4), rowsT = [];
      for (let k = 0; k < 4; k++) {
        const a = sorted.slice(k * q, (k + 1) * q);
        rowsT.push({ k, a, lo: a[0].p50_bp, hi: a[a.length - 1].p50_bp });
      }
      const f1 = v => (v > 0 ? "+" : "") + (v / 100).toFixed(1) + "%";
      tableFrom(p, [["Forecast group", r => ["lowest", "second", "third", "highest"][r.k] + " quarter"], ["Forecast range", r => `${f1(r.lo)} to ${f1(r.hi)}`],
        ["Trades", r => num(r.a.length)], ["Winners", r => pct(r.a.filter(t => t.win).length / r.a.length)],
        ["Average result", r => f1(avg(r.a, "ret_bp"))], ["Median result", r => f1(med(r.a, "ret_bp"))]], rowsT);
    });
    plate(s.main, "Winners and losers looked alike at the moment of entry", "medians of each group", p => {
      const rowsT = [["1h forecast (median)", "p50_bp", v => "+" + (v / 100).toFixed(1) + "%"],
        ["1h forecast, pessimistic 10%", "p10_bp", v => (v / 100).toFixed(1) + "%"], ["1h forecast, optimistic 90%", "p90_bp", v => "+" + (v / 100).toFixed(1) + "%"],
        ["Coin's move over the last 24h", "coin_ret_24h", v => pct(v)], ["Spike volume vs. normal", "vol_mult", v => v.toFixed(0) + "×"],
        ["BTC's move over the last 4h", "btc_ret_4h", v => pct(v, 1)], ["BTC volume vs. normal", "btc_vol_mult", v => v.toFixed(1) + "×"],
        ["Best point during the hour (net)", "best_bp", v => (v > 0 ? "+" : "") + (v / 100).toFixed(1) + "%"],
        ["Worst point during the hour (net)", "worst_bp", v => (v / 100).toFixed(1) + "%"]];
      tableFrom(p, [["", r => r[0]], ["Winners", r => r[2](med(win, r[1]))], ["Losers", r => r[2](med(loss, r[1]))]], rowsT);
    });
    const lossUp1 = loss.filter(t => t.best_bp > 100).length / loss.length, lossUp2 = loss.filter(t => t.best_bp > 200).length / loss.length;
    const winDn2 = win.filter(t => t.worst_bp < -200).length / win.length;
    const top = T.slice().sort((a, b) => a.p50_bp - b.p50_bp).slice(Math.floor(T.length * .75));
    s.verdict(`<b>A coin-flip-plus trade with fat tails, not a sure thing.</b> About ${pct(win.length / T.length)} of trades win, and
      winners and losers are about the same size (${"+" + (avg(win, "ret_bp") / 100).toFixed(1)}% vs. ${(avg(loss, "ret_bp") / 100).toFixed(1)}%). The edge
      comes from slightly more winners and a few very large rebounds. At entry, winners and losers look almost the same: the
      winners came on slightly heavier BTC flushes, and that's all.
      <br><br><b>The forecast's size matters, its level doesn't.</b> The median forecast says +${(med(T, "p50_bp") / 100).toFixed(1)}% in the hour;
      the median trade makes +${(med(T, "ret_bp") / 100).toFixed(1)}%. So read it as a ranking, not a price target. But the quarter of
      trades with the biggest forecasts win ${pct(top.filter(t => t.win).length / top.length)} of the time and average
      +${(avg(top, "ret_bp") / 100).toFixed(1)}%, far better than the rest.
      <br><br><b>The fixed 1-hour exit leaves money on the table both ways.</b> ${pct(lossUp1)} of losing trades were up more than 1% at
      some point in the hour (${pct(lossUp2)} more than 2%), and ${pct(winDn2)} of winners first fell more than 2%. The ride is
      rough even when it works.`,
      "Ideas to test next: trading only the biggest forecasts (picked here after seeing the results, so it needs a fresh walk-forward test before it counts), and exits that react to the price path, such as a trailing stop. Exits set by the forecast itself were tested earlier and were erratic (see the forecasting section).");
  });
}
