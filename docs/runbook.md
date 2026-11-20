# Runbook: running the paper trading daemon on a Proxmox host

How to run `moonshot live run` unattended in Docker and watch it from a browser on your LAN.
Paper mode only: no API keys, no orders on any exchange. The design is in
[architecture.md](architecture.md) ("Operations"); this page is what to type.

All commands below run in the repository folder on the Docker host, unless they say otherwise.

## Prerequisites

- A Debian or Ubuntu VM, or an LXC container, on the Proxmox host, with Docker Engine and the
  Compose plugin (`docker compose version` works). A VM is the simplest choice.
- In an LXC container, Docker needs the features **nesting** and **keyctl** (Proxmox: container,
  Options, Features), and usually an unprivileged container is fine. Restart the container after
  changing them.
- Outbound internet to Binance's public market-data hosts (`data-api.binance.vision` and
  `data-stream.binance.vision`) and, for the build, to the Docker, GitHub Container and Python
  package registries. No inbound internet access is needed, ever.
- `git`, to fetch and upgrade the code.

## First deploy

1. Get the code on the host: `git clone https://github.com/<you>/crypto-moonshot.git && cd
   crypto-moonshot` (or copy the folder over).
2. Create the token file and put a secret token in it:

   ```bash
   cp moonshot.env.example moonshot.env
   python3 -c "import secrets; print(secrets.token_urlsafe(32))"   # copy the output
   nano moonshot.env                                               # MOONSHOT_API_TOKEN=<paste>
   chmod 600 moonshot.env
   ```

   `moonshot.env` is gitignored; never commit it.
3. Create the folders the container mounts, and give the state folder to the container's user
   (uid 10001, who cannot write anywhere else):

   ```bash
   mkdir -p config strategies state data
   sudo chown 10001:10001 state
   ```
4. Put the run config and the strategy plugin files in place. The example config uses the
   example plugin:

   ```bash
   cp deploy/config/paper.toml config/paper.toml
   cp examples/strategies/ma_cross.py strategies/
   ```

   Edit `config/paper.toml` for your strategies. Inside the container the paths are fixed:
   `state = "/state/paper.sqlite"` and plugin sources like `/strategies/my_plugin.py:MyClass`
   (absolute: a relative `source` would be relative to the container's `/app`).
5. Optional: candle data. The daemon fetches the candles it needs from Binance by itself. Only
   strategies that use a universe name (`symbols = "all"` or `"pool"`) read stored candles; copy
   or sync the folder into `./data` (it is mounted read-only at `/data`).
6. Start it:

   ```bash
   docker compose up -d --build
   docker compose ps          # State "running", Status "healthy" once it has started
   docker compose logs -f     # JSON log lines; Ctrl-C leaves the container running
   ```

   The first build takes a few minutes; the container starts with `restart: unless-stopped`, so
   it also comes back after a reboot of the host.

## Opening the dashboard

Browse to `http://<host-ip>:8765/` from a machine on the LAN and paste the token from
`moonshot.env`. The token stays in that tab's session storage; Log out clears it. The pages are:
overview, strategies (equity charts), positions and orders, trades, signals and providers, and
alerts and events. The pause, resume and flatten controls are in the header and ask for
confirmation.

## Day to day

```bash
docker compose ps                                  # running and healthy?
docker compose logs --tail 100 moonshot            # recent logs (JSON, one object per line)
docker compose exec moonshot moonshot live status  # mode, controls, books, positions, events
docker compose exec moonshot moonshot live status --json
docker compose exec moonshot moonshot live pause   # stop taking new entries (exits keep working)
docker compose exec moonshot moonshot live resume
docker compose exec moonshot moonshot live flatten # sell everything at the next candle, then pause
```

Inside the container the token is already in the environment and the daemon is at
`http://127.0.0.1:8765`, so these need no options. `flatten` asks you to confirm (`--yes` skips
that; add `-T` to `docker compose exec` in scripts). The dashboard offers the same controls.

If the daemon is stopped, `docker compose run --rm moonshot live status --config
/config/paper.toml` reads the state file instead.

## Alerts

Alerts show in the dashboard (a badge and a banner on every page; red for an error, amber for a
warning); there are no push notifications, so look at it. An alert appears while its rule fires
and goes away by itself when the condition ends. Acknowledging it hides it from the badge until
it resolves and fires again; it never silences the rule.

| Alert | Meaning | What to do |
| --- | --- | --- |
| `feed_disconnected` (error) | The Binance candle stream has been down for over 60 s. | Usually transient; the daemon reconnects by itself. If it lasts: check the host's internet access and DNS, and whether Binance is reachable from your location. No new candles means no new trades. |
| `no_ticks` (error) | No candle close was processed for two intervals plus the tick timeout. | Look at `feed_disconnected` and the logs. If the feed is up and it persists, the watchdog restarts the daemon (below). |
| `provider_missing:<strategy>` (warning) | An external strategy's last 3 batches were missed or late. | Check that the provider process is running and can reach the daemon; see the Providers page. |
| `plugin_failing:<strategy>` (warning) | An in-process plugin gave no signals for its last 3 candles (it raised, timed out or broke the contract). | Look for the plugin's exception in the logs or the events page; fix the plugin file in `./strategies` and `docker compose restart`. |
| `daily_loss_limit:<strategy>` (warning) | The strategy hit its daily loss limit and takes no entries until the next UTC day. | Informational. Review the strategy. |
| `backfill_failed` (warning) | Seeding or backfilling candles failed within the last hour. | Check Binance reachability; the daemon fills the gap on its own once it works. |
| `stale_symbols:<interval>` (warning) | Over 20% of that interval's traded symbols have no recent candle. | Often a feed problem right after a start or reconnect; if it persists, check the logs for the symbols. |
| `backup_failed` (error) | The last automatic backup failed. | Check free disk space and that `./state` is writable by uid 10001. Run `live backup` by hand (below). |
| `disk_low` (warning) | Under 1 GB free where the state file lives. | Free space on the host (old backups, Docker images: `docker system prune`). |
| `orphan_order` (warning, exchange modes) | At start-up the exchange had open orders with the daemon's ID prefixes that the state file does not know. | The daemon leaves them alone. Look at them on the exchange and cancel them by hand if they are not yours to keep. |
| `balance_mismatch` (error, exchange modes) | At start-up the exchange held less of an asset than the open positions need; entries are paused. | Find out where the coins went (a manual trade, a restore of an old state file). Resume entries only once the account and the state file agree. |
| `unprotected_position:<symbol>` (error, exchange modes) | An open position has no bracket (take profit / stop) at the exchange. | The broker retries every tick. If it lasts, flatten the position or place the exits by hand. |

## The watchdog and restarts

The daemon stops itself with **exit code 3** when its engine stalls (no tick for 10 intervals, at
least 15 minutes, while the feed is delivering messages) or its event loop is blocked for 60 s.
`restart: unless-stopped` then starts it again and it recovers from the state file. Exit code 1
(a runtime failure, or the state file locked by another process) is restarted the same way; exit
code 2 (a bad config, a missing token, a plugin that does not load) is restarted too, so a config
mistake shows as a container that keeps restarting: read `docker compose logs`. `docker compose
ps` shows the restart count through `docker inspect -f '{{.RestartCount}}'
$(docker compose ps -q moonshot)`.

`docker compose stop` and `docker stop` send SIGTERM; the daemon finishes its tick, stops the API
and exits 0 (the grace period is 60 s). The container's `init: true` (tini) forwards the signal.

## Upgrades

```bash
docker compose exec moonshot moonshot live backup        # a backup first
git pull
docker compose up -d --build                             # rebuilds and recreates the container
docker compose logs --tail 50 moonshot
```

The state file's schema is migrated when the new version opens it. A backup made by an older
version restores fine (it is migrated at the next start), but an older build cannot open a file
migrated by a newer one, so keep the pre-upgrade backup if you may want to roll back.

## Backups

- The daemon backs itself up at start and every 24 h (`[health] backup_every`), keeping the
  newest 14 (`backup_keep`). They are verified SQLite files in `./state/backups/`
  (`paper-<UTC timestamp>.sqlite`).
- By hand, any time (the daemon may be running):

  ```bash
  docker compose exec moonshot moonshot live backup           # to /state/backups
  docker compose exec moonshot moonshot live backup --out /state/manual
  ```

- Copy them off the host (the state folder and its backups die with the host's disk):

  ```bash
  rsync -a ./state/backups/ backup-box:/srv/moonshot-backups/
  ```

  Or let Proxmox back up the whole VM or container on a schedule.
- Restore (stop the daemon first; the restore refuses while it runs):

  ```bash
  docker compose stop
  ls state/backups
  docker compose run --rm moonshot live restore /state/backups/paper-20261121T000000Z.sqlite --config /config/paper.toml
  docker compose start
  ```

  The file it replaces is kept as `state/paper.sqlite.before-restore-<UTC timestamp>`.

## Recovering from a crash or a corrupt state file

- A crash, power loss or `kill -9` needs no action: the state file is crash safe, and on start the
  daemon rebuilds its books and positions from the state file and carries on.
- A stale `paper.sqlite.lock` file is harmless; the lock goes away with the process. If the daemon
  refuses to start because the state file is in use, another container or process still has it:
  `docker compose ps -a`, stop it, and start again.
- If the daemon will not open the state file (a corrupt file or an error in the logs mentioning
  the database), stop it, restore the newest backup (above), and start. If you have no backup,
  move the file aside (`mv state/paper.sqlite* /somewhere/`) to start with a fresh book: that
  loses the history and open positions, which are paper positions only.

## Rotating the token

Generate a new one, put it in `moonshot.env`, and recreate the container (a plain `restart` does
not re-read the file):

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
nano moonshot.env
docker compose up -d --force-recreate
```

Then log in to the dashboard again with the new token. Do this whenever the token may have leaked.

## Security: LAN only, no TLS

The API and the dashboard are plain HTTP behind one bearer token. Anyone who can reach the port
and learns the token can pause or flatten the book (it can never place a real order in this
build). So:

- Bind the port to the host's LAN address in `compose.yaml` (`"192.168.1.50:8765:8765"`) rather
  than all interfaces, and firewall it to your own machines.
- Never forward the port from your router or expose it to the internet.
- For remote access use a VPN into your network (WireGuard, or Tailscale on the Docker host) and
  browse to the host's VPN address.
- The container runs as a non-root user with a read-only root filesystem, no capabilities and
  `no-new-privileges`; keep it that way.

## Resources

Expectations for the example config (a handful of symbols, one interval); check your own with
`docker stats --no-stream`:

- CPU: near idle; a short burst at each candle close while strategies run.
- RAM: a few hundred MB (polars and numpy), more with many symbols or heavy plugins.
- Disk: the image is a few hundred MB. The state file grows with trades and events (megabytes);
  14 daily backups multiply that. Logs are capped by the compose rotation (5 files of 10 MB).
  Keep several GB free; `disk_low` warns below 1 GB.

## Troubleshooting

- **Container keeps restarting**: `docker compose logs --tail 50 moonshot`. Exit code 2 is a
  config, plugin or token error (the message names the field); `docker inspect -f
  '{{.State.ExitCode}}' $(docker compose ps -q moonshot)` shows the last exit code.
- **"MOONSHOT_API_TOKEN is not set"**: `moonshot.env` is missing or the line is empty.
- **Permission denied on /state**: `sudo chown 10001:10001 state`.
- **"No such file" for a plugin**: use the container path (`/strategies/file.py:Class`) and put
  the file in `./strategies`.
- **Status "unhealthy"**: `docker compose exec moonshot python -c "import urllib.request as u;
  print(u.urlopen('http://127.0.0.1:8765/api/v1/health').read())"`. Startup can take up to a
  minute.
- **The dashboard does not load**: check that the port is published (`docker compose ps`), the
  host firewall, and that you use `http://`, not `https://`.
- **401 in the dashboard or CLI**: the token does not match `moonshot.env`; after editing it,
  recreate the container.
- **Feed alerts**: check DNS and outbound access from the host; `docker compose exec moonshot
  python -c "import urllib.request as u; print(u.urlopen('https://data-api.binance.vision/api/v3/ping').read())"`.
