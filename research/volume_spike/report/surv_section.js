"use strict";
/* Survivorship section (M13): the main strategies with delisted and fallen coins added. Data: D.surv. */
const SV = D.surv || null;
const UNIS = [["pool", "today's top ~200 (as before)", "var(--s1)"], ["poolA", "+ delisted coins", "var(--s2)"], ["all", "+ delisted + every other USDT pair", "var(--s3)"]];

if (SV) {
  const T = window.SURV_TEXT || {};
  method("m-surv", "Survivorship: adding delisted and fallen coins", T.chip || "Pending", T.chipClass || "base", s => {
    s.body(`<p>Every result so far used today's top ~200 coins, so the coins that died or shrank were missing, and
      they're exactly the ones a dip-buying rule likes to buy. Binance's live API has no history for delisted
      pairs, but its public archive (data.binance.vision) does. I fetched every USDT pair from 2020 on that
      wasn't in the data pool, checked each file's checksum and imported them into the main data folder.</p>
      <ul class="plain"><li><b>Delisted (191 coins):</b> pairs Binance has removed, for example MATIC (until it became
      POL), WAVES, SRM, ANC and MIR.</li>
      <li><b>Still listed but outside the top 200 (239 coins):</b> including the famous collapses that still trade,
      such as LUNA (with its May 2022 crash), FTT and USTC.</li></ul>
      <p>Tokenised stocks and stablecoins are left out as before. A position open when a coin is delisted or halted
      exits at its last traded price. Nothing was re-tuned: every rule, model setting and threshold is the same as
      on the pool.</p>`);
    plate(s.main, "Coins with data, by month", "the pool is today's survivors; the delisted group shrinks as coins die", p => {
      const months = [...new Set(Object.values(SV.coins_by_month).flatMap(c => Object.keys(c)))].sort().filter(m => m <= "2026-10");
      lineChart(p, { x: "time", hgt: 220, yFmt: v => Math.round(v),
        series: [["pool", "pool (today's top ~200)", "var(--s1)"], ["A delisted", "delisted", "var(--s2)"], ["B outside pool", "listed, outside the pool", "var(--s3)"]]
          .map(([k, n, c]) => ({ name: n, color: c, x: months.map(m => m + "-01"), y: months.map(m => SV.coins_by_month[k][m] || 0) })) });
    });
    const g = s.grid();
    for (const [period, label] of [["train", "2020–23"], ["test", "2025–26"]]) plate(g, `Finalists, ${label}`, "5-slot portfolio, net bp per trade, by coin universe", p => {
      const rule = { F1: "F1 flush bounce", F2: "F2 deep liquid flush", F3: "F3 deep crash, no BTC filter" };
      const get = (u, r) => (SV.finalists.find(x => x.universe === u && x.strategy === rule[r] && x.period === period) || {});
      barChart(p, { cats: ["F1", "F2", "F3"], hgt: 220, signColor: false,
        series: UNIS.map(([u, n, c]) => ({ name: n, color: c, vals: ["F1", "F2", "F3"].map(r => get(u, r).mean_bp) })),
        tipExtra: (i, k) => { const r = get(UNIS[k][0], ["F1", "F2", "F3"][i]); return `${num(r.trades)} trades · CAGR ${pct(r.cagr)} · max DD ${pct(r.max_dd)}`; } });
    });
    plate(s.main, "Walk-forward rules (re-tuned every six months), by year", "net bp per signal day, by coin universe", p => {
      const yrs = [2022, 2023, 2024, 2025, 2026];
      const val = (u, y) => { const r = SV.walkforward.filter(x => x.universe === u && x.start.startsWith(String(y)));
        const d = r.reduce((a, x) => a + x.oos_days, 0); return d ? r.reduce((a, x) => a + x.oos_bp * x.oos_days, 0) / d : null; };
      barChart(p, { cats: yrs.map(String), hgt: 220, signColor: false, series: UNIS.map(([u, n, c]) => ({ name: n, color: c, vals: yrs.map(y => val(u, y)) })) });
    });
    const strats = ["Base CNN, 1h, 5 slots", "Market CNN, 1h, 5 slots", "Path forecaster, 1h, 5 slots", "CNN ensemble, 1h, 5 slots", "F2 rule (5 slots)"];
    const g2 = s.grid();
    for (const st of strats) plate(g2, st, "walk-forward, net bp per trade by year and coin universe", p => {
      const yrs = [2022, 2023, 2024, 2025, 2026];
      const get = (u, y) => SV.strategies.find(x => x.universe === u && x.strategy === st && x.year === y) || {};
      barChart(p, { cats: yrs.map(String), hgt: 200, signColor: false,
        series: UNIS.filter(([u]) => SV.strategies.some(x => x.universe === u && x.strategy === st)).map(([u, n, c]) => ({ name: n, color: c, vals: yrs.map(y => get(u, y).mean_bp) })),
        tipExtra: (i, k) => { const u = UNIS.filter(([uu]) => SV.strategies.some(x => x.universe === uu && x.strategy === st))[k][0]; const r = get(u, yrs[i]);
          return `${num(r.trades)} trades, ${num(r.added_trades)} on added coins (${bp(r.added_mean_bp)})`; } });
    });
    plate(s.main, "How much of each strategy's trading is in the added coins", "share of trades on coins outside the pool, and how those trades did", p => {
      const rowsT = [];
      for (const st of strats) for (const [u] of UNIS.slice(1)) {
        const r = SV.strategies.filter(x => x.universe === u && x.strategy === st);
        if (!r.length) continue;
        const n = r.reduce((a, x) => a + (x.trades || 0), 0), na = r.reduce((a, x) => a + (x.added_trades || 0), 0);
        const ma = r.reduce((a, x) => a + (x.added_mean_bp || 0) * (x.added_trades || 0), 0) / (na || 1);
        const mp = r.reduce((a, x) => a + ((x.mean_bp || 0) * (x.trades || 0) - (x.added_mean_bp || 0) * (x.added_trades || 0)), 0) / ((n - na) || 1);
        rowsT.push({ st, u: UNIS.find(x => x[0] === u)[1], n, share: na / n, ma, mp });
      }
      tableFrom(p, [["Strategy", r => r.st], ["Universe", r => r.u], ["Trades", r => num(r.n)], ["On added coins", r => pct(r.share)],
        ["Added coins, net / trade", r => bp(r.ma)], ["Pool coins, net / trade", r => bp(r.mp)]], rowsT);
    });
    const g3 = s.grid();
    for (const e of SV.examples || []) plate(g3, e.title, "LUNA's collapse, 11–12 May 2022; shaded: the holding window", p => candles(p, e));
    if (T.extra) s.body(T.extra);
    s.verdict(T.verdict || "Pending the full-universe run.", T.next);
  });
}
