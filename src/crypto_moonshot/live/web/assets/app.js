// The dashboard shell: login, routing, polling, the alert banner and the controls.

import { api, AuthError, NetworkError, token } from "./api.js";
import { h, replace } from "./dom.js";
import { routes, strategy } from "./pages.js";

const $ = (id) => document.getElementById(id);
const REFRESH_KEY = "moonshot.refresh";

const ctx = { status: null, alerts: null, refresh: () => poll(true), acknowledge };
let page = null; // { handlers, route }
let timer = null;
let generation = 0;
let lastOk = null;
let inFlight = false;

// -- login and logout ------------------------------------------------------------------------

function showLogin(message = "") {
  clearTimeout(timer);
  generation++;
  page = null;
  token.clear();
  ctx.status = ctx.alerts = null;
  $("bar").hidden = true;
  $("banner").hidden = true;
  $("controls").hidden = true;
  $("page").replaceChildren();
  $("login").hidden = false;
  $("token").value = "";
  $("login-msg").textContent = message;
  $("token").focus();
}

async function login(event) {
  event.preventDefault();
  const value = $("token").value.trim();
  if (!value) return;
  token.set(value);
  $("login-msg").textContent = "Checking…";
  try {
    await api("/status");
  } catch (error) {
    if (error instanceof AuthError) showLogin("That token was not accepted.");
    else {
      token.clear();
      $("login-msg").textContent = "Cannot reach the daemon. Check that it is running.";
    }
    return;
  }
  startSession();
}

function startSession() {
  $("login").hidden = true;
  $("login-msg").textContent = "";
  $("bar").hidden = false;
  $("conn").textContent = "";
  route();
}

// -- routing ---------------------------------------------------------------------------------

function parseRoute() {
  const [, name = "overview", arg] = location.hash.split("/");
  return { name: decodeURIComponent(name) || "overview", arg: arg ? decodeURIComponent(arg) : null };
}

function route() {
  if (!token.get()) return showLogin();
  const { name, arg } = parseRoute();
  const mount = name === "strategy" && arg ? (root) => strategy(root, ctx, arg) : routes[name]?.bind(null);
  const known = Boolean(mount);
  const key = known ? (name === "strategy" ? `strategy:${arg}` : name) : "overview";
  for (const link of document.querySelectorAll("nav a")) {
    const current = link.dataset.route === (name === "strategy" ? "overview" : name);
    if (current) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  const root = $("page");
  root.replaceChildren();
  $("controls").hidden = !(key === "overview");
  generation++;
  const factory = known ? mount : routes.overview;
  page = { key, handlers: name === "strategy" && arg ? factory(root) : factory(root, ctx) };
  if (!known) location.hash = "#/overview";
  poll(true);
}

// -- polling ---------------------------------------------------------------------------------

function interval() {
  return Number($("every").value) * 1000;
}

function schedule() {
  clearTimeout(timer);
  if (!token.get() || document.hidden) return;
  timer = setTimeout(() => poll(false), interval());
}

async function poll(now) {
  clearTimeout(timer);
  if (!token.get() || !page) return;
  if (inFlight && !now) return schedule();
  const mine = ++generation;
  inFlight = true;
  try {
    const [status, alerts] = await Promise.all([api("/status"), api("/alerts")]);
    ctx.status = status;
    ctx.alerts = alerts;
    const data = await page.handlers.fetch();
    if (mine !== generation) return; // the route changed while loading
    renderBanner();
    renderControls();
    page.handlers.update(data);
    lastOk = new Date();
    setConn(true);
  } catch (error) {
    if (error instanceof AuthError) return showLogin("The API rejected the token. Log in again.");
    setConn(false, error);
  } finally {
    inFlight = false;
    schedule();
  }
}

function setConn(ok, error) {
  const el = $("conn");
  el.classList.toggle("lost", !ok);
  const when = lastOk ? lastOk.toLocaleTimeString() : "never";
  if (ok) el.textContent = `Updated ${when}`;
  else {
    const why = error instanceof NetworkError ? "connection lost" : `error: ${error?.message ?? "unknown"}`;
    el.textContent = `⚠ ${why[0].toUpperCase()}${why.slice(1)}. Last updated ${when}`;
  }
}

// -- the alert banner ------------------------------------------------------------------------

function renderBanner() {
  const banner = $("banner");
  const a = ctx.alerts;
  const open = a.active.filter((x) => !x.acknowledged);
  banner.hidden = false;
  banner.className = "";
  if (!a.active.length) {
    banner.classList.add("none");
    replace(banner, "✓ No active alerts");
    return;
  }
  const worst = open.some((x) => x.level === "error") ? "error" : open.length ? "warning" : "acked";
  banner.classList.add(worst);
  const icon = worst === "error" ? "✖" : worst === "warning" ? "▲" : "✓";
  replace(banner,
    h("span", { "aria-hidden": "true" }, icon + " "),
    h("a", { href: "#/alerts" },
      open.length
        ? `${open.length} unacknowledged alert${open.length === 1 ? "" : "s"}`
        : `${a.active.length} active alert${a.active.length === 1 ? "" : "s"}, all acknowledged`),
    open[0] ? `: ${open[0].key}, ${open[0].detail}` : "");
}

async function acknowledge(key, button) {
  button.disabled = true;
  try {
    await api(`/alerts/${encodeURIComponent(key)}/ack`, { method: "POST" });
  } catch (error) {
    if (error instanceof AuthError) return showLogin("The API rejected the token. Log in again.");
  }
  poll(true);
}

// -- controls --------------------------------------------------------------------------------

function renderControls() {
  const c = ctx.status.controls;
  $("controls-state").textContent =
    `Trading is ${c.paused ? "paused" : "running"}. Flatten ${c.flatten_requested ? "is requested" : "is not requested"}.`;
  // Only the control that changes something is offered.
  $("pause").disabled = c.paused;
  $("resume").disabled = !c.paused;
}

function ask({ title, body, ok, word }) {
  const dialog = $("dialog");
  $("dialog-title").textContent = title;
  $("dialog-body").textContent = body;
  $("dialog-ok").textContent = ok;
  $("dialog-typed").hidden = !word;
  $("dialog-word").textContent = word ?? "";
  const input = $("dialog-input");
  input.value = "";
  const okButton = $("dialog-ok");
  okButton.disabled = Boolean(word);
  input.oninput = () => { okButton.disabled = input.value.trim() !== word; };
  return new Promise((resolve) => {
    dialog.addEventListener("close", () => resolve(dialog.returnValue === "ok"), { once: true });
    dialog.returnValue = "cancel";
    dialog.showModal();
    (word ? input : $("dialog-cancel")).focus();
  });
}

const ACTIONS = {
  pause: { title: "Pause trading?", body: "No new entries are placed while paused. Open positions keep their exits.", ok: "Pause" },
  resume: { title: "Resume trading?", body: "New entries are placed again.", ok: "Resume" },
  flatten: {
    title: "Flatten everything?",
    body: "This closes every open position at the next candle and cancels pending entries.",
    ok: "Flatten", word: "flatten",
  },
};

async function control(action) {
  if (!(await ask(ACTIONS[action]))) return;
  const buttons = [...document.querySelectorAll("#controls button")];
  buttons.forEach((b) => { b.disabled = true; });
  $("controls-msg").textContent = "";
  try {
    const result = await api(`/control/${action}`, { method: "POST" });
    ctx.status.controls = result.controls;
    renderControls();
  } catch (error) {
    if (error instanceof AuthError) return showLogin("The API rejected the token. Log in again.");
    $("controls-msg").textContent = `${action} failed: ${error.message}`;
  } finally {
    buttons.forEach((b) => { b.disabled = false; });
    renderControls();
  }
  poll(true);
}

// -- start -----------------------------------------------------------------------------------

$("login-form").addEventListener("submit", login);
$("logout").addEventListener("click", () => showLogin("Logged out."));
for (const action of Object.keys(ACTIONS)) $(action).addEventListener("click", () => control(action));
$("every").value = sessionStorage.getItem(REFRESH_KEY) ?? "5";
$("every").addEventListener("change", () => {
  sessionStorage.setItem(REFRESH_KEY, $("every").value);
  schedule();
});
window.addEventListener("hashchange", route);
document.addEventListener("visibilitychange", () => {
  if (document.hidden) clearTimeout(timer);
  else if (token.get()) poll(true);
});

if (token.get()) startSession();
else showLogin();
