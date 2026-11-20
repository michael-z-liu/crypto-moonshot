// Number and time formatting. Times show in the browser's zone with the UTC time in a tooltip.

import { h } from "./dom.js";

const money = new Intl.NumberFormat(undefined, {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});
const percent = new Intl.NumberFormat(undefined, {
  style: "percent",
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});
const price = new Intl.NumberFormat(undefined, { maximumFractionDigits: 8 });
const localTime = new Intl.DateTimeFormat(undefined, { dateStyle: "short", timeStyle: "medium" });
const MINUS = "−";

export const dash = "—";

export function usdt(value) {
  return value === null || value === undefined ? dash : money.format(value);
}

export function signedUsdt(value) {
  if (value === null || value === undefined) return dash;
  const text = money.format(Math.abs(value));
  if (value > 0) return "+" + text;
  return value < 0 ? MINUS + text : text;
}

export function pct(fraction) {
  if (fraction === null || fraction === undefined) return dash;
  const text = percent.format(Math.abs(fraction));
  if (fraction > 0) return "+" + text;
  return fraction < 0 ? MINUS + text : text;
}

export function num(value) {
  return value === null || value === undefined ? dash : price.format(value);
}

export function utcText(iso) {
  return new Date(iso).toISOString().replace("T", " ").replace(/\.\d+Z$/, " UTC");
}

export function localText(iso) {
  return localTime.format(new Date(iso));
}

/** A `<time>` element: local time as text, UTC in the tooltip. */
export function timeEl(iso) {
  if (!iso) return dash;
  return h("time", { datetime: iso, title: utcText(iso) }, localText(iso));
}

export function seconds(value) {
  if (value === null || value === undefined) return dash;
  if (value < 90) return `${value.toFixed(value < 10 ? 1 : 0)} s`;
  if (value < 5400) return `${(value / 60).toFixed(0)} min`;
  if (value < 129600) return `${(value / 3600).toFixed(1)} h`;
  return `${(value / 86400).toFixed(1)} d`;
}

/** A profit or loss: the sign and an icon say gain or loss as well as the colour. */
export function pnlEl(value, formatted = signedUsdt(value)) {
  if (value === null || value === undefined) return dash;
  const kind = value > 0 ? "gain" : value < 0 ? "loss" : "flat";
  const icon = value > 0 ? "▲" : value < 0 ? "▼" : "●";
  return h(
    "span",
    { class: `pnl pnl-${kind}` },
    h("span", { class: "icon", "aria-hidden": "true" }, icon),
    h("span", { class: "sr" }, kind + " "),
    formatted,
  );
}
