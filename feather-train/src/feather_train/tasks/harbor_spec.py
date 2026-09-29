"""World network policy and MCP descriptors from Harbor templates."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from feather_harness.tools.mcp import McpServerSpec

NetworkMode = Literal["public", "no-network"]


@dataclass(frozen=True, slots=True)
class HarborSpec:
    """Everything Train reads out of one world's Harbor template."""

    mcp_servers: tuple[McpServerSpec, ...]
    network_mode: NetworkMode


def parse_harbor_spec(template_path: str | Path, *, where: str) -> HarborSpec:
    """Parse a world's Harbor template, raising on anything Train cannot honour.

    ``where`` identifies the world in error messages, so a bad export names itself.
    """
    with open(template_path, "rb") as handle:
        harbor = tomllib.load(handle)
    environment = harbor.get("environment")
    if not isinstance(environment, dict):
        raise ValueError(f"{where} Harbor template lacks [environment]")
    return HarborSpec(
        mcp_servers=_mcp_servers(environment, where=where),
        network_mode=_network_mode(environment, where=where),
    )


def _mcp_servers(environment: dict, *, where: str) -> tuple[McpServerSpec, ...]:
    try:
        return tuple(
            McpServerSpec(
                name=entry.get("name"), transport=entry.get("transport"), url=entry.get("url")
            )
            for entry in environment.get("mcp_servers") or []
        )
    except ValueError as exc:
        raise ValueError(f"{where} has an invalid MCP server: {exc}") from exc


def _network_mode(environment: dict, *, where: str) -> NetworkMode:
    if "allow_internet" in environment:
        raise ValueError(
            f"{where} uses legacy allow_internet; declare network_mode "
            f"('public' or 'no-network') instead"
        )
    if "network_mode" not in environment:
        raise ValueError(f"{where} Harbor template must declare network_mode")
    mode = environment["network_mode"]
    if mode == "allowlist":
        raise ValueError(
            f"{where} requests Harbor network_mode='allowlist', which Train cannot enforce"
        )
    if mode not in ("public", "no-network"):
        raise ValueError(f"{where} has unsupported network_mode {mode!r}")
    return mode
