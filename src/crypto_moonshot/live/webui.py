"""The dashboard's static files, served by the daemon's FastAPI app (`live/web/`).

No build step and no data: `index.html`, one stylesheet and a few JavaScript modules, shipped in
the package. Everything the page shows comes from the token-guarded API. Only `/` and the files
under `/assets/` are served, from a listing taken when the app is created (so a request path is
only ever a lookup key, never a file path); any other path is a plain 404.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

WEB_DIR = Path(__file__).parent / "web"

CONTENT_SECURITY_POLICY = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; "
    "connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'none'; "
    "frame-ancestors 'none'"
)
"""No inline scripts or styles, nothing from another origin."""

_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
}
_HEADERS = {
    "Content-Security-Policy": CONTENT_SECURITY_POLICY,
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    # Revalidate on every load (a cheap 304 via the ETag), so an upgraded daemon never serves
    # a page whose modules are a mix of old and new files.
    "Cache-Control": "no-cache",
}


@dataclass(frozen=True, slots=True)
class _Asset:
    body: bytes
    media_type: str
    etag: str


def _load(web_dir: Path) -> dict[str, _Asset]:
    found: dict[str, _Asset] = {}
    for path in sorted(web_dir.rglob("*")):
        media = _TYPES.get(path.suffix)
        if not path.is_file() or media is None:
            continue
        body = path.read_bytes()
        url = (
            "/"
            if path.relative_to(web_dir).as_posix() == "index.html"
            else ("/" + path.relative_to(web_dir).as_posix())
        )
        found[url] = _Asset(body, media, '"' + hashlib.sha256(body).hexdigest()[:16] + '"')
    return found


def add_dashboard(app: FastAPI, web_dir: Path = WEB_DIR) -> None:
    """Serve the dashboard at `/` and its files under `/assets/`. Register after the API
    routes; nothing else is answered."""
    assets = _load(web_dir)
    if "/" not in assets:
        msg = f"the dashboard is missing: no index.html in {web_dir}"
        raise RuntimeError(msg)

    def respond(request: Request, asset: _Asset) -> Response:
        headers = {**_HEADERS, "ETag": asset.etag}
        if request.headers.get("if-none-match") == asset.etag:
            return Response(status_code=304, headers=headers)
        return Response(asset.body, media_type=asset.media_type, headers=headers)

    @app.get("/", include_in_schema=False)
    async def index(request: Request) -> Response:
        return respond(request, assets["/"])

    @app.get("/assets/{name}", include_in_schema=False)
    async def asset(request: Request, name: str) -> Response:
        found = assets.get("/assets/" + name)
        if found is None:
            raise HTTPException(status_code=404, detail="not found")
        return respond(request, found)
