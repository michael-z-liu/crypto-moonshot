// Tiny DOM helpers. Every string goes in through textContent / setAttribute, never innerHTML:
// the data (symbols, alert details, reasons) is untrusted as far as the page is concerned.

const SVG_NS = "http://www.w3.org/2000/svg";

function fill(el, children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

function setAttrs(el, attrs) {
  for (const [key, value] of Object.entries(attrs ?? {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") el.setAttribute("class", value);
    else if (key.startsWith("on") && typeof value === "function") {
      el.addEventListener(key.slice(2), value);
    } else el.setAttribute(key, value === true ? "" : String(value));
  }
}

/** `h("div", {class: "x"}, child, "text")` builds an HTML element. */
export function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  setAttrs(el, attrs);
  return fill(el, children);
}

/** The same for an SVG element. */
export function s(tag, attrs, ...children) {
  const el = document.createElementNS(SVG_NS, tag);
  setAttrs(el, attrs);
  return fill(el, children);
}

/** Replace everything inside `el`. */
export function replace(el, ...children) {
  el.replaceChildren();
  return fill(el, children);
}
