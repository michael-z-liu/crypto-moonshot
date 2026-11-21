// The pages. Each one is `mount(root, ctx)`: build the skeleton once, return
// `{ fetch(), update(data) }`. `fetch` loads what the page needs besides the always-polled
// /status and /alerts (in `ctx.status`, `ctx.alerts`); `update` redraws only the data regions,
// so a filter or a focused control survives every poll.

import { api } from "./api.js";
import { equityChart } from "./chart.js";
import { h, replace } from "./dom.js";
import { dash, num, pct, pnlEl, seconds, timeEl, usdt } from "./format.js";

// -- shared pieces ---------------------------------------------------------------------------

function table(caption, headers, rows, empty) {
  if (!rows.length) return h("p", { class: "empty" }, empty);
  return h("div", { class: "table-wrap" },
    h("table", null,
      h("caption", { class: "sr" }, caption),
      h("thead", null, h("tr", null, headers.map(([label, num]) =>
        h("th", { scope: "col", class: num ? "num" : null }, label)))),
      h("tbody", null, rows)));
}

function cells(values) {
  return h("tr", null, values.map((v) => {
    const isObj = v && typeof v === "object" && !(v instanceof Node);
    const node = isObj ? v.value : v;
    return h("td", { class: isObj && v.num ? "num" : null }, node);
  }));
}
const n = (value) => ({ value, num: true });

function card(title, ...body) {
  const id = "h-" + title.toLowerCase().replace(/[^a-z0-9]+/g, "-");
  return h("section", { class: "card", "aria-labelledby": id }, h("h2", { id }, title), ...body);
}

function strategyLink(name) {
  return h("a", { href: `#/strategy/${encodeURIComponent(name)}` }, name);
}

function strategySelect(ctx, onChange, value) {
  const select = h("select", { id: "f-strategy" },
    h("option", { value: "" }, "All strategies"),
    Object.keys(ctx.status?.books ?? {}).map((name) => h("option", { value: name }, name)));
  select.value = value;
  select.addEventListener("change", () => onChange(select.value));
  return h("div", { class: "filters" }, h("label", { for: "f-strategy" }, "Strategy"), select);
}

function badge(level, text) {
  const icon = level === "error" ? "✖" : level === "warning" ? "▲" : "●";
  return h("span", { class: `badge badge-${level}` }, h("span", { "aria-hidden": "true" }, icon), " ", text ?? level);
}

/** Last closes of the symbols of open positions: `{interval: {symbol: close}}` (best effort). */
async function markPrices(positions) {
  const wanted = {};
  for (const p of positions) (wanted[p.interval] ??= new Set()).add(p.symbol);
  const marks = {};
  await Promise.all(Object.entries(wanted).map(async ([interval, symbols]) => {
    try {
      const body = await api("/candles", {
        params: { interval, symbols: [...symbols].join(","), limit: 1, format: "json" },
      });
      const close = body.columns.indexOf("close");
      for (const [symbol, rows] of Object.entries(body.windows)) {
        if (rows.length) (marks[interval] ??= {})[symbol] = rows.at(-1)[close];
      }
    } catch {
      // no marks: the page shows a dash
    }
  }));
  return marks;
}

function positionsTable(positions, marks) {
  const rows = positions.map((p) => {
    const mark = marks?.[p.interval]?.[p.symbol];
    const open = p.status === "open";
    const unrealised = open && mark !== undefined && p.quantity ? p.quantity * mark - p.cost : null;
    return cells([
      strategyLink(p.strategy), p.symbol, p.status,
      p.entry_time ? timeEl(p.entry_time) : dash,
      n(num(p.entry_price)), n(mark === undefined ? dash : num(mark)),
      n(num(p.quantity)), n(usdt(p.cost)), n(pnlEl(unrealised)),
    ]);
  });
  return table("Open positions",
    [["Strategy"], ["Symbol"], ["Status"], ["Entry time"], ["Entry price", 1], ["Mark", 1],
      ["Quantity", 1], ["Cost (USDT)", 1], ["Unrealised P&L (USDT)", 1]],
    rows, "No open positions.");
}

function ordersTable(orders) {
  return table("Open orders",
    [["Strategy"], ["Symbol"], ["Side"], ["Type"], ["Placed"], ["Quote amount (USDT)", 1], ["Quantity", 1], ["Status"]],
    orders.map((o) => cells([strategyLink(o.strategy), o.symbol, o.side, o.type, timeEl(o.time),
      n(usdt(o.quote_amount)), n(num(o.quantity)), o.status])),
    "No open orders.");
}

function tradesTable(trades) {
  return table("Closed trades",
    [["Strategy"], ["Symbol"], ["Entry"], ["Exit"], ["Held"], ["Entry price", 1], ["Exit price", 1],
      ["P&L (USDT)", 1], ["Return", 1], ["Exit reason"]],
    trades.map((t) => cells([strategyLink(t.strategy), t.symbol, timeEl(t.entry_time), timeEl(t.exit_time),
      seconds(t.hold_seconds), n(num(t.entry_price)), n(num(t.exit_price)), n(pnlEl(t.pnl)),
      n(pnlEl(t.return_fraction, pct(t.return_fraction))), t.exit_reason])),
    "No closed trades yet.");
}

function signalsTable(signals) {
  return table("Recent signals",
    [["Candle"], ["Strategy"], ["Symbol"], ["Action"], ["Priority", 1], ["Outcome"], ["Reason"], ["Batch"]],
    signals.map((x) => cells([timeEl(x.candle_time), strategyLink(x.strategy), x.symbol, x.action,
      n(x.priority ?? dash), x.outcome ?? dash, x.reason ?? dash, x.batch_status])),
    "No signals yet.");
}

// -- overview --------------------------------------------------------------------------------

export function overview(root, ctx) {
  const facts = h("dl", { class: "facts", "aria-label": "Daemon state" });
  const rows = h("div");
  const exchangeFacts = h("dl", { class: "facts", "aria-label": "Exchange state" });
  const exchangeCard = card("Exchange", exchangeFacts);
  exchangeCard.hidden = true;
  root.append(
    h("h1", null, "Overview"),
    card("Daemon", facts),
    exchangeCard,
    card("Strategies", rows),
  );
  return {
    async fetch() {
      const [equity, providers] = await Promise.all([api("/equity", { params: { limit: 1 } }), api("/providers")]);
      return { equity, providers };
    },
    update({ equity, providers }) {
      const st = ctx.status;
      const health = st.health.feed;
      const lag = health?.last_message_at ? (Date.parse(equity.time) - Date.parse(health.last_message_at)) / 1000 : null;
      const fact = (label, value) => [h("dt", null, label), h("dd", null, value)];
      replace(facts,
        fact("Mode", h("strong", null, st.mode)),
        fact("Engine", st.running ? badge("info", "running") : badge("error", "stopped")),
        fact("Feed", health ? (health.connected ? badge("info", "connected") : badge("error", "disconnected")) : dash),
        fact("Feed lag", lag === null ? dash : seconds(Math.max(lag, 0))),
        fact("Last tick", Object.keys(st.last_ticks).length
          ? Object.entries(st.last_ticks).map(([i, t]) => h("span", { class: "tick" }, `${i} `, timeEl(t)))
          : dash),
        fact("Controls", [
          st.controls.paused ? badge("warning", "paused") : badge("info", "trading"),
          " ",
          st.controls.flatten_requested ? badge("warning", "flatten requested") : null,
        ]),
        fact("Unacknowledged alerts", String(ctx.alerts.unacknowledged)),
      );
      const ex = st.exchange;
      exchangeCard.hidden = !ex;
      if (ex) {
        const list = (obj) => Object.keys(obj).length
          ? Object.entries(obj).map(([k, v]) => `${k} ${v}`).join(", ") : "none";
        const rec = ex.last_reconcile;
        replace(exchangeFacts,
          fact("User data stream", ex.stream_connected ? badge("info", "connected") : badge("error", "disconnected")),
          fact("Broker", ex.halted ? badge("error", `halted: ${ex.halted}`) : badge("info", "placing orders")),
          fact("Dust", list(ex.dust)),
          fact("Other commissions", list(ex.other_commissions)),
          fact("Clock offset", `${ex.time_offset_ms} ms`),
          fact("Last reconcile", rec
            ? [timeEl(rec.time), " ", rec.clean ? badge("info", "clean") : badge("warning", `${rec.orphans} orphans, ${rec.mismatches} mismatches`)]
            : "not yet"),
        );
      }
      const stats = Object.fromEntries(providers.providers.map((p) => [p.strategy, p]));
      replace(rows, table("Strategies",
        [["Strategy"], ["Capital (USDT)", 1], ["Equity (USDT)", 1], ["Realised P&L", 1], ["Unrealised P&L", 1],
          ["Open positions", 1], ["Today P&L", 1], ["Today entries", 1], ["Last batch"]],
        Object.entries(st.books).map(([name, book]) => {
          const e = equity.strategies[name];
          const p = stats[name];
          return cells([strategyLink(name), n(usdt(book.capital)), n(usdt(e?.equity ?? book.equity)),
            n(pnlEl(e?.realised_pnl)), n(pnlEl(e?.unrealised_pnl)), n(e?.open_positions ?? book.open_positions),
            n(pnlEl(book.day_pnl)), n(book.day_entries),
            p ? h("span", null, p.last_batch_time ? timeEl(p.last_batch_time) : "none yet",
              ` (${p.on_time} on time, ${p.late} late, ${p.missed} missed)`) : dash]);
        }), "No strategies configured."));
    },
  };
}

// -- one strategy ----------------------------------------------------------------------------

export function strategy(root, ctx, name) {
  const chart = h("div");
  const positions = h("div");
  const trades = h("div");
  const signals = h("div");
  const facts = h("dl", { class: "facts" });
  root.append(
    h("p", null, h("a", { href: "#/overview" }, "← Overview")),
    h("h1", null, name),
    facts,
    card("Equity", chart),
    card("Open positions", positions),
    card("Recent trades", trades),
    card("Recent signals", signals),
  );
  return {
    async fetch() {
      const params = { strategy: name };
      const [equity, tr, sig] = await Promise.all([
        api("/equity", { params: { ...params, limit: 500 } }),
        api("/trades", { params: { ...params, limit: 20 } }),
        api("/signals", { params: { ...params, limit: 20 } }),
      ]);
      const mine = ctx.status.positions.filter((p) => p.strategy === name);
      return { equity, tr, sig, mine, marks: await markPrices(mine.filter((p) => p.status === "open")) };
    },
    update({ equity, tr, sig, mine, marks }) {
      const e = equity.strategies[name];
      if (!e) {
        replace(chart, h("p", { class: "empty" }, "Unknown strategy."));
        return;
      }
      const fact = (label, value) => [h("dt", null, label), h("dd", null, value)];
      replace(facts,
        fact("Capital", `${usdt(e.capital)} USDT`),
        fact("Equity", `${usdt(e.equity)} USDT`),
        fact("Realised P&L", pnlEl(e.realised_pnl)),
        fact("Unrealised P&L", pnlEl(e.unrealised_pnl)),
        fact("Open positions", String(e.open_positions)));
      replace(chart, equityChart(e, equity.time));
      replace(positions, positionsTable(mine, marks));
      replace(trades, tradesTable(tr.trades));
      replace(signals, signalsTable(sig.signals));
    },
  };
}

// -- positions and orders --------------------------------------------------------------------

export function positions(root, ctx) {
  const pos = h("div");
  const ord = h("div");
  root.append(h("h1", null, "Positions and orders"), card("Positions", pos), card("Open orders", ord));
  return {
    async fetch() {
      const live = ctx.status.positions;
      return { marks: await markPrices(live.filter((p) => p.status === "open")) };
    },
    update({ marks }) {
      replace(pos, positionsTable(ctx.status.positions, marks));
      replace(ord, ordersTable(ctx.status.open_orders));
    },
  };
}

// -- trades ----------------------------------------------------------------------------------

export function trades(root, ctx) {
  const body = h("div");
  let selected = "";
  const filters = h("div");
  root.append(h("h1", null, "Trades"), filters, card("Closed trades", body));
  const draw = () => replace(filters, strategySelect(ctx, (v) => { selected = v; ctx.refresh(); }, selected));
  draw();
  return {
    async fetch() {
      return api("/trades", { params: { strategy: selected, limit: 200 } });
    },
    update(data) {
      if (!filters.querySelector("option:nth-child(2)") && Object.keys(ctx.status.books).length) draw();
      replace(body, tradesTable(data.trades));
    },
  };
}

// -- signals and providers -------------------------------------------------------------------

export function signals(root, ctx) {
  const prov = h("div");
  const sig = h("div");
  let selected = "";
  const filters = h("div");
  root.append(h("h1", null, "Signals and providers"), card("Providers", prov), filters, card("Recent signals", sig));
  const draw = () => replace(filters, strategySelect(ctx, (v) => { selected = v; ctx.refresh(); }, selected));
  draw();
  return {
    async fetch() {
      const [providers, list] = await Promise.all([
        api("/providers"), api("/signals", { params: { strategy: selected, limit: 200 } }),
      ]);
      return { providers, list };
    },
    update({ providers, list }) {
      if (!filters.querySelector("option:nth-child(2)") && Object.keys(ctx.status.books).length) draw();
      replace(prov,
        h("p", { class: "muted" }, `Batches over the last ${providers.window_hours} hours, by candle time.`),
        table("Provider lateness",
          [["Strategy"], ["Kind"], ["Interval"], ["On time", 1], ["Late", 1], ["Missed", 1],
            ["Median lateness", 1], ["Worst lateness", 1], ["Last batch"], ["Last seen"]],
          providers.providers.map((p) => cells([strategyLink(p.strategy), p.external ? "external" : "in process",
            p.interval, n(p.on_time), n(p.late), n(p.missed), n(seconds(p.median_lateness_seconds)),
            n(seconds(p.worst_lateness_seconds)),
            p.last_batch_time ? timeEl(p.last_batch_time) : dash, p.last_seen ? timeEl(p.last_seen) : dash])),
          "No strategies."));
      replace(sig, signalsTable(list.signals));
    },
  };
}

// -- alerts and events -----------------------------------------------------------------------

export function alerts(root, ctx) {
  const active = h("div");
  const history = h("div");
  const feed = h("div");
  let level = "";
  const select = h("select", { id: "f-level" },
    [["", "All levels"], ["info", "Info"], ["warning", "Warning"], ["error", "Error"]]
      .map(([value, label]) => h("option", { value }, label)));
  select.addEventListener("change", () => { level = select.value; ctx.refresh(); });
  root.append(
    h("h1", null, "Alerts and events"),
    card("Active alerts", active),
    card("Alert history", history),
    card("Event feed",
      h("div", { class: "filters" }, h("label", { for: "f-level" }, "Level"), select), feed),
  );
  return {
    async fetch() {
      return api("/events", { params: { limit: 200 } });
    },
    update(data) {
      const a = ctx.alerts;
      replace(active, a.active.length
        ? table("Active alerts",
          [["Level"], ["Alert"], ["Since"], ["Detail"], ["State"]],
          a.active.map((x) => cells([badge(x.level), x.key, timeEl(x.fired_at), x.detail,
            x.acknowledged
              ? h("span", null, "acknowledged ", timeEl(x.acknowledged_at))
              : h("button", { type: "button", class: "btn", "data-ack": x.key,
                "aria-label": `Acknowledge ${x.key}`,
                onclick: (event) => ctx.acknowledge(x.key, event.currentTarget) }, "Acknowledge")])))
        : h("p", { class: "empty ok" }, "No active alerts."));
      replace(history, table("Alert history",
        [["Time"], ["Level"], ["Alert"], ["State"], ["Detail"]],
        a.history.map((x) => cells([timeEl(x.time), badge(x.level), x.key ?? dash, x.state ?? dash, x.detail ?? dash])),
        "No alert events yet."));
      const events = data.events.filter((e) => !level || e.level === level).reverse();
      replace(feed, table("Events",
        [["Time"], ["Level"], ["Kind"], ["Message"]],
        events.map((e) => cells([timeEl(e.time), badge(e.level), e.kind, e.message])),
        "No events."));
    },
  };
}

export const routes = { overview, positions, trades, signals, alerts };
