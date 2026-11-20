"""The dashboard's static routes: files, headers, 404s, and that the API stays guarded."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from api_fakes import ApiRig

from crypto_moonshot.live.webui import CONTENT_SECURITY_POLICY, WEB_DIR

pytestmark = pytest.mark.anyio

SHIPPED = sorted(p.relative_to(WEB_DIR).as_posix() for p in WEB_DIR.rglob("*") if p.is_file())


def bare(rig: ApiRig) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=rig.url)


async def test_the_index_and_every_asset_are_served_without_a_token(tmp_path: Path) -> None:
    async with ApiRig(tmp_path) as rig, bare(rig) as client:
        index = await client.get("/")
        assert index.status_code == 200
        assert index.headers["content-type"].startswith("text/html")
        assert "<title>moonshot</title>" in index.text
        for name in SHIPPED:
            if name == "index.html":
                continue
            response = await client.get("/" + name)
            assert response.status_code == 200, name
            assert response.content == (WEB_DIR / name).read_bytes()
        assert (
            (await client.get("/assets/app.js"))
            .headers["content-type"]
            .startswith("text/javascript")
        )
        assert (await client.get("/assets/app.css")).headers["content-type"].startswith("text/css")


async def test_static_files_carry_security_headers(tmp_path: Path) -> None:
    async with ApiRig(tmp_path) as rig, bare(rig) as client:
        for path in ("/", "/assets/app.js", "/assets/app.css"):
            headers = (await client.get(path)).headers
            assert headers["content-security-policy"] == CONTENT_SECURITY_POLICY
            assert headers["x-content-type-options"] == "nosniff"
            assert headers["referrer-policy"] == "no-referrer"
            assert headers["cache-control"] == "no-cache"
        csp = CONTENT_SECURITY_POLICY
        assert "default-src 'self'" in csp
        assert "unsafe-inline" not in csp
        assert "unsafe-eval" not in csp


async def test_an_unchanged_asset_revalidates_with_a_304(tmp_path: Path) -> None:
    async with ApiRig(tmp_path) as rig, bare(rig) as client:
        first = await client.get("/assets/app.js")
        again = await client.get("/assets/app.js", headers={"If-None-Match": first.headers["etag"]})
        assert again.status_code == 304
        assert again.content == b""
        assert again.headers["content-security-policy"] == CONTENT_SECURITY_POLICY


async def test_unknown_paths_are_404_and_never_the_api(tmp_path: Path) -> None:
    async with ApiRig(tmp_path) as rig, bare(rig) as client:
        for path in (
            "/nope",
            "/assets/nope.js",
            "/assets/",
            "/assets/..%2Fapi.py",
            "/assets/../api.py",
            "/index.html",
            "/status",
            "/trades",
            "/web/index.html",
            "/api/nope",
        ):
            response = await client.get(path)
            assert response.status_code == 404, path
            assert "content-security-policy" not in response.headers or path == "/assets/"
        # the API paths are not shadowed by the page
        assert (await client.get("/api/v1/health")).status_code == 200


async def test_the_pages_contain_no_data_and_load_nothing_from_elsewhere(tmp_path: Path) -> None:
    async with ApiRig(tmp_path) as rig, bare(rig) as client:
        index = (await client.get("/")).text
        assert "<script>" not in index
        assert "style=" not in index
        assert "http://" not in index
        assert "https://" not in index
        assert rig.token not in index


async def test_the_api_still_needs_the_token(tmp_path: Path) -> None:
    async with ApiRig(tmp_path) as rig, bare(rig) as client:
        for path in ("/api/v1/status", "/api/v1/trades", "/api/v1/alerts"):
            assert (await client.get(path)).status_code == 401, path
            ok = await client.get(path, headers={"Authorization": f"Bearer {rig.token}"})
            assert ok.status_code == 200, path
        assert (await client.post("/api/v1/control/pause")).status_code == 401


def test_the_web_files_are_in_the_package() -> None:
    assert {"index.html", "assets/app.js", "assets/app.css"} <= set(SHIPPED)
