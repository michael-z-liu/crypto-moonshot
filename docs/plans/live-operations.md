# Operating the trading daemon: health, alerts, dashboard, Docker

- **Status:** review
- **Branch:** `feat/live-operations` (stacked on PR #18, `worktree-bridge-cse_015bssAUbtXue4wkarka78Pb`)
- **PR:**
- **Updated:** 2026-11-21T01:10Z

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

### M2: Health, alerts and backups (done)

- **Delivers:** API endpoints for trades, equity, signals, provider lateness and alerts (with
  acknowledge); health rules raising alert events (feed down, no ticks, provider missing batches,
  daily loss limit hit, backfill failures); a watchdog that exits on a stalled engine; automatic
  online SQLite backups and `moonshot live backup`.
- **Done when:** tests cover each rule and endpoint, the watchdog exit and a backup restore.
- **Worker:** sonnet
- **Commits:** 8ebbe52, 9a09a15, 5c44e11, a5f9ee4

### M3: Web dashboard (done)

- **Delivers:** a dashboard served by the daemon at `/` (no build step): token login, overview and
  health, strategies with equity charts, positions and orders, trades, signals and providers,
  alert feed with acknowledge, pause/resume/flatten with confirmation.
- **Done when:** API tests plus browser tests (Playwright against a served replay) cover login,
  each page and the controls.
- **Worker:** sonnet
- **Commits:** b68716c, bb5fb28, 7696354, 294ef3f, 1e94050, d9af340 (master)

### M4: Docker and runbook (done)

- **Delivers:** Dockerfile (non-root, healthcheck), compose file (state, data and strategies
  volumes; token from an env file; restart policy), optional JSON logs, `docs/runbook.md` for
  Proxmox.
- **Done when:** a CI job builds the image and checks a container's `/health`; runbook complete.
- **Worker:** sonnet
- **Commits:** 8c44d71, b6f5819, 5981a0f

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
- 2026-11-21 (M2): state schema v2 (an `alerts` table of active alerts; resolving deletes the row
  and its ack). `provider_missing` = 3 non-on-time batches in a row; `plugin_failing` = 3 missed.
  Served replays run the health monitor with time-based rules off; watchdogs are daemon-only.
  The state lock covers every replay too. Exit codes: 1 locked state, 3 watchdog. `live status
  --config` cannot read a v1 file until a daemon opens (migrates) it once.
- 2026-11-21 (M3): dashboard is static files served by the daemon (hash routes, CSP `default-src
  'self'`, no inline code, `Cache-Control: no-cache` + ETag); token in `sessionStorage` only.
  Browser tests are opt-in locally (`-m browser`, `--tracing off`; pytest's own `--trace` is pdb)
  and run in a separate CI job. Review fixes: only the applicable pause/resume button is enabled;
  overview open positions come from `/equity` like the strategy page.
- 2026-11-21 (M4): image on `python:3.13-slim-trixie` and uv 0.13.7, both pinned by digest;
  non-root uid 10001; read-only root with tmpfs `/tmp`; `PYTHONDONTWRITEBYTECODE=1`; absolute
  `/state` and `/strategies` paths in the container config; the host's `state/` folder must be
  owned by uid 10001. `restart: unless-stopped` also restarts on a bad config (exit 2), so a config
  mistake loops; documented. Built and smoke-tested only in the CI `docker` job.

## Blocked on

## Follow-ups

- Check on the Proxmox host: `chown 10001:10001 state`, LXC nesting/keyctl, the healthcheck turns
  healthy, the dashboard loads from the LAN, real RAM/disk use (`docker stats`).
- The PR screenshots predate the M3 review fixes (overview open positions).
