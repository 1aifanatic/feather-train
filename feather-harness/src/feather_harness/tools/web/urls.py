"""Shared validation for public-web tool URLs."""

from __future__ import annotations

from ipaddress import ip_address
from socket import inet_aton
from urllib.parse import urlsplit

MAX_WEB_URL_CHARS = 2_000

_LOCAL_HOSTNAMES = frozenset({"localhost"})
_LOCAL_SUFFIXES = (".localhost", ".local", ".internal", ".home", ".lan")


def validate_public_web_url(url: str, *, label: str) -> str:
    """Return a safe-to-render public HTTP(S) URL or raise ``ValueError``."""
    if not isinstance(url, str):
        raise ValueError(f"{label} must be a string")
    if (
        len(url) > MAX_WEB_URL_CHARS
        or any(not character.isprintable() or character.isspace() for character in url)
        or "\\" in url
        or url != url.strip()
    ):
        raise ValueError(f"{label} must be at most {MAX_WEB_URL_CHARS} printable characters")
    try:
        parsed_url = urlsplit(url)
        hostname = parsed_url.hostname
        parsed_url.port
    except ValueError as exc:
        raise ValueError(f"{label} must be a valid HTTP(S) URL") from exc
    if (
        parsed_url.scheme not in {"http", "https"}
        or not hostname
        or "%" in hostname
        or parsed_url.username is not None
        or parsed_url.password is not None
    ):
        raise ValueError(f"{label} must be an HTTP(S) URL without userinfo")

    try:
        normalized_hostname = hostname.encode("idna").decode("ascii").rstrip(".").lower()
    except UnicodeError as exc:
        raise ValueError(f"{label} must contain a valid public hostname") from exc
    if (
        normalized_hostname in _LOCAL_HOSTNAMES
        or normalized_hostname.endswith(_LOCAL_SUFFIXES)
        or _is_non_public_or_legacy_ip(normalized_hostname)
    ):
        raise ValueError(f"{label} must target the public web")
    return url


def _is_non_public_or_legacy_ip(hostname: str) -> bool:
    try:
        address = ip_address(hostname)
    except ValueError:
        try:
            inet_aton(hostname)
        except OSError:
            return False
        return True
    return not address.is_global
