# syntax=docker/dockerfile:1
# The paper trading daemon (`moonshot live run`). See docs/runbook.md.
#
# Volumes:  /config (ro) run config      /state (rw) state file, lock, backups
#           /data (ro) candle store, optional      /strategies (ro) plugin files
# Port:     8765 (API and dashboard; plain HTTP, LAN only)

# python:3.13-slim-trixie, pinned by digest (refresh with `docker buildx imagetools inspect`).
ARG PYTHON_IMAGE=python:3.13-slim-trixie@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

# The uv binary, pinned: ghcr.io/astral-sh/uv:0.13.7
FROM ghcr.io/astral-sh/uv:0.13.7@sha256:96f40d3904aad8d0bd010539001d36c8e2188b981e1334471a90c4a1d93bb490 AS uv

# -- build: resolve the locked, runtime-only dependencies into a venv --------------------------
FROM ${PYTHON_IMAGE} AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_NO_CACHE=1
WORKDIR /app

# Dependencies first, so editing the source does not reinstall them. `--no-dev` leaves out the
# dev group (pyproject's default group); the research group is never installed either.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

# Then the package itself, installed (not editable) into the venv.
COPY README.md ./
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

# -- runtime -----------------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS runtime

RUN groupadd --system --gid 10001 moonshot \
    && useradd --system --uid 10001 --gid 10001 --no-create-home --home-dir /nonexistent \
        --shell /usr/sbin/nologin moonshot \
    && mkdir /config /state /data /strategies \
    && chown moonshot:moonshot /state

COPY --from=build /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MOONSHOT_DATA_DIR=/data

WORKDIR /app
USER 10001:10001
EXPOSE 8765
VOLUME ["/state"]

# /health needs no token. Python is already here, so no curl.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8765/api/v1/health', timeout=4).status == 200 else 1)"]

STOPSIGNAL SIGTERM
ENTRYPOINT ["moonshot"]
CMD ["live", "run", "--config", "/config/paper.toml", "--host", "0.0.0.0", "--log-format", "json"]
