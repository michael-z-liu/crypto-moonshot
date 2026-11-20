// A hand-written SVG equity chart. One series, so no legend: the card title names it.
// Realised equity is the line (a point per closed trade); the current mark-to-market equity is
// a hollow ring at the end, joined by a fainter segment. Hover or focus shows a crosshair and a
// tooltip; the table view under the chart holds every value.

import { h, replace, s } from "./dom.js";
import { localText, timeEl, usdt } from "./format.js";

const HEIGHT = 260;
const MARGIN = { top: 16, right: 76, bottom: 28, left: 64 };

function niceTicks(lo, hi, count = 4) {
  if (hi - lo < 1e-9) {
    lo -= 1;
    hi += 1;
  }
  const raw = (hi - lo) / count;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((v) => v >= raw);
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) ticks.push(v);
  return { ticks, step };
}

/** Build the chart for one strategy's `/equity` entry. Returns the element; it draws itself at
 * the width of its container and again when that changes. */
export function equityChart(entry, now) {
  const marks = entry.points.map((p) => ({ time: Date.parse(p.time), equity: p.equity, iso: p.time }));
  const current = { time: Date.parse(now), equity: entry.equity, iso: now, current: true };
  const all = [...marks, current];

  const wrap = h("div", { class: "chart" });
  const tip = h("div", { class: "tooltip", role: "status", hidden: true });
  const host = h("div", { class: "chart-host" });
  wrap.append(host, tip);

  function draw() {
    const width = Math.max(280, host.clientWidth || 600);
    const plotW = width - MARGIN.left - MARGIN.right;
    const plotH = HEIGHT - MARGIN.top - MARGIN.bottom;
    const t0 = Math.min(...all.map((p) => p.time));
    let t1 = Math.max(...all.map((p) => p.time));
    const tmin = t1 - t0 < 1 ? t0 - 3600_000 : t0;
    if (t1 - tmin < 1) t1 = tmin + 1;
    const values = [...all.map((p) => p.equity), entry.capital];
    const { ticks, step } = niceTicks(Math.min(...values), Math.max(...values));
    const ymin = Math.min(ticks[0], ...values);
    const ymax = Math.max(ticks.at(-1), ...values);
    const x = (t) => MARGIN.left + ((t - tmin) / (t1 - tmin)) * plotW;
    const y = (v) => MARGIN.top + (1 - (v - ymin) / (ymax - ymin || 1)) * plotH;
    const decimals = step < 1 ? 2 : 0;
    const fmt = (v) => v.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals });

    const svg = s("svg", {
      class: "equity",
      width,
      height: HEIGHT,
      viewBox: `0 0 ${width} ${HEIGHT}`,
      role: "img",
      "aria-label":
        `Equity of the strategy: ${marks.length} realised points, now ${usdt(entry.equity)} USDT ` +
        `against ${usdt(entry.capital)} USDT of capital. The table below has every value.`,
    });
    for (const v of ticks) {
      svg.append(
        s("line", { class: "grid", x1: MARGIN.left, x2: width - MARGIN.right, y1: y(v), y2: y(v) }),
        s("text", { class: "axis", x: MARGIN.left - 8, y: y(v) + 4, "text-anchor": "end" }, fmt(v)),
      );
    }
    for (let i = 0; i < 4; i++) {
      const t = tmin + ((t1 - tmin) * i) / 3;
      const anchor = i === 0 ? "start" : i === 3 ? "end" : "middle";
      svg.append(
        s("text", { class: "axis", x: x(t), y: HEIGHT - 8, "text-anchor": anchor },
          new Date(t).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })),
      );
    }
    svg.append(
      s("line", { class: "baseline", x1: MARGIN.left, x2: width - MARGIN.right, y1: y(entry.capital), y2: y(entry.capital) }),
      s("text", { class: "axis", x: width - MARGIN.right + 6, y: y(entry.capital) + 4 }, "capital"),
    );
    if (marks.length) {
      const d = marks.map((p, i) => `${i ? "L" : "M"}${x(p.time).toFixed(1)},${y(p.equity).toFixed(1)}`).join(" ");
      const area = `${d} L${x(marks.at(-1).time).toFixed(1)},${y(ymin)} L${x(marks[0].time).toFixed(1)},${y(ymin)} Z`;
      svg.append(s("path", { class: "area", d: area }), s("path", { class: "line", d }));
    }
    const last = marks.at(-1);
    if (last) {
      svg.append(
        s("line", { class: "line soft", x1: x(last.time), y1: y(last.equity), x2: x(current.time), y2: y(current.equity) }),
      );
    }
    if (marks.length <= 30) {
      for (const p of marks) svg.append(s("circle", { class: "dot", cx: x(p.time), cy: y(p.equity), r: 4 }));
    }
    svg.append(
      s("circle", { class: "dot now", cx: x(current.time), cy: y(current.equity), r: 5 }),
      s("text", { class: "label", x: Math.min(x(current.time) + 10, width - 4), y: y(current.equity) + 4,
        "text-anchor": "start" },
        usdt(current.equity)),
    );

    const cross = s("line", { class: "cross", y1: MARGIN.top, y2: MARGIN.top + plotH, visibility: "hidden" });
    const ring = s("circle", { class: "dot hover", r: 5, visibility: "hidden" });
    svg.append(cross, ring);
    const hit = s("rect", { class: "hit", x: MARGIN.left, y: MARGIN.top, width: plotW, height: plotH, tabindex: 0,
      "aria-label": "Equity values: use the left and right arrow keys" });
    svg.append(hit);

    let index = all.length - 1;
    function show(i) {
      index = Math.max(0, Math.min(all.length - 1, i));
      const p = all[index];
      cross.setAttribute("x1", x(p.time));
      cross.setAttribute("x2", x(p.time));
      cross.setAttribute("visibility", "visible");
      ring.setAttribute("cx", x(p.time));
      ring.setAttribute("cy", y(p.equity));
      ring.setAttribute("visibility", "visible");
      replace(tip,
        h("strong", null, `${usdt(p.equity)} USDT`),
        h("span", { class: "muted" }, p.current ? " now (mark to market)" : " realised"),
        h("div", { class: "muted" }, localText(p.iso)),
      );
      tip.hidden = false;
      const left = Math.min(Math.max(x(p.time) + 12, 0), width - 190);
      tip.style.left = `${left}px`;
      tip.style.top = `${Math.max(y(p.equity) - 44, 0)}px`;
    }
    function hide() {
      cross.setAttribute("visibility", "hidden");
      ring.setAttribute("visibility", "hidden");
      tip.hidden = true;
    }
    hit.addEventListener("pointermove", (event) => {
      const box = svg.getBoundingClientRect();
      const px = event.clientX - box.left;
      let best = 0;
      all.forEach((p, i) => {
        if (Math.abs(x(p.time) - px) < Math.abs(x(all[best].time) - px)) best = i;
      });
      show(best);
    });
    hit.addEventListener("pointerleave", hide);
    hit.addEventListener("focus", () => show(index));
    hit.addEventListener("blur", hide);
    hit.addEventListener("keydown", (event) => {
      if (event.key === "ArrowLeft") show(index - 1);
      else if (event.key === "ArrowRight") show(index + 1);
      else return;
      event.preventDefault();
    });
    replace(host, svg);
  }

  const table = h("details", { class: "table-view" },
    h("summary", null, "Table view"),
    h("div", { class: "table-wrap" },
      h("table", null,
        h("caption", { class: "sr" }, "Equity values"),
        h("thead", null, h("tr", null, h("th", { scope: "col" }, "Time"), h("th", { scope: "col", class: "num" }, "Equity (USDT)"), h("th", { scope: "col" }, "Kind"))),
        h("tbody", null,
          all.map((p) => h("tr", null,
            h("td", null, timeEl(p.iso)),
            h("td", { class: "num" }, usdt(p.equity)),
            h("td", null, p.current ? "now (mark to market)" : "realised"))),
        ))));
  const root = h("div", null, wrap, table);
  new ResizeObserver(() => draw()).observe(host);
  draw();
  return root;
}
