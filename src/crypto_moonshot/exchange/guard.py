"""The mainnet guard: testnet mode can never reach a real Binance host.

`require_testnet_url` is called wherever a signed client or user data stream is built, and again
by the daemon on the configured URLs. A host is refused when its name mentions `binance` and it is
not the spot testnet (`testnet.binance.vision` and its sub-domains); every other host (a local
fake, a proxy) is allowed, because it cannot be a Binance production endpoint by name.
"""

from __future__ import annotations

from urllib.parse import urlsplit

TESTNET_DOMAIN = "testnet.binance.vision"


class MainnetRefused(ValueError):
    """A URL that points at a Binance host other than the spot testnet."""


def is_testnet_host(host: str) -> bool:
    host = host.lower().rstrip(".")
    if host == TESTNET_DOMAIN or host.endswith("." + TESTNET_DOMAIN):
        return True
    return "binance" not in host


def require_testnet_url(url: str, what: str = "URL") -> None:
    """Raise `MainnetRefused` unless `url` can only be the testnet (or a non-Binance host)."""
    try:
        parts = urlsplit(url.strip())
        host = parts.hostname
    except ValueError:
        host = None
    if host is None:
        msg = f"the {what} {url!r} is not a valid URL"
        raise MainnetRefused(msg)
    if not is_testnet_host(host):
        msg = (
            f"refusing the {what} {url!r}: testnet mode only talks to {TESTNET_DOMAIN} "
            "(a Binance production host would trade real money)"
        )
        raise MainnetRefused(msg)
