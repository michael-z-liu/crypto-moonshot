# Operating the trading daemon: health, alerts, dashboard, Docker

- **Status:** building
- **Branch:** `feat/live-operations` (stacked on PR #18, `worktree-bridge-cse_015bssAUbtXue4wkarka78Pb`)
- **PR:**
- **Updated:** 2026-11-20T22:35Z

## Goal

P2 of the trading platform (P1 is PR #18, `docs/plans/live-paper-engine.md`). Make the paper
daemon something the user can run unattended on their Proxmox host and watch from a browser on
the LAN: health rules that raise alerts, a watchdog, automatic state backups, a web dashboard with
controls, a Docker image and a runbook.

## Out of scope

- Push alerts (alerts appear in the UI only, by the user's choice).
- TLS on the API (LAN plus token only; documented).
- Any exchange order placement (P3 testnet broker, P4 mainnet).

## Milestones

### M1: Operations design (done)

- **Delivers:** `docs/architecture.md` sections for health and alert rules, the watchdog, backups,
  the dashboard, Docker and logging.
- **Done when:** the doc is the spec for M2-M4; check command passes.
- **Worker:** master
- **Commits:** 7d667fb

### M2: Health, alerts and backups (in progress)

- **Delivers:** API endpoints for trades, equity, signals, provider lateness and alerts (with
  acknowledge); health rules raising alert events (feed down, no ticks, provider missing batches,
  daily loss limit hit, backfill failures); a watchdog that exits on a stalled engine; automatic
  online SQLite backups and `moonshot live backup`.
- **Done when:** tests cover each rule and endpoint, the watchdog exit and a backup restore.
- **Worker:** sonnet
- **Commits:**

### M3: Web dashboard (todo)

- **Delivers:** a dashboard served by the daemon at `/` (no build step): token login, overview and
  health, strategies with equity charts, positions and orders, trades, signals and providers,
  alert feed with acknowledge, pause/resume/flatten with confirmation.
- **Done when:** API tests plus browser tests (Playwright against a served replay) cover login,
  each page and the controls.
- **Worker:** sonnet
- **Commits:**

### M4: Docker and runbook (todo)

- **Delivers:** Dockerfile (non-root, healthcheck), compose file (state, data and strategies
  volumes; token from an env file; restart policy), optional JSON logs, `docs/runbook.md` for
  Proxmox.
- **Done when:** a CI job builds the image and checks a container's `/health`; runbook complete.
- **Worker:** sonnet
- **Commits:**

## Decisions

- 2026-11-20: P2 and P3 approved together ("continue until P3"); P4 needs the user's keys and
  go-ahead.
- 2026-11-20: Stacked branches: P2 on PR #18's branch, P3 on P2. When a base PR squash-merges,
  `main` is merged into the next branch (no rebase, no force push).
- 2026-11-20: Docker is not installed on the dev machine; the image is built and smoke-tested in a
  GitHub Actions job.
- 2026-11-20 (M1): added to the design beyond the plan: a single-instance lock on the state file,
  `moonshot live restore`, a `disk_low` rule, an event-loop-blocked watchdog (os._exit(3)), alert
  acknowledgement that hides but never silences a rule.

## Blocked on

## Follow-ups
