"""Caller-owned connection to the local managed-process worker."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from pydantic import JsonValue

from ..tools import ToolResult


class LocalShellRuntime:
    def __init__(self, workspace: Path, *, log_directory: Path, grants: tuple[Path, ...] = ()):
        self.workspace = Path(os.path.abspath(workspace))
        self.log_directory = Path(os.path.abspath(log_directory))
        self.grants = tuple(Path(os.path.abspath(path)) for path in grants)
        if not self.workspace.is_dir():
            raise ValueError("workspace must be a directory")
        if any(path.resolve() == path.resolve().parent for path in self.grants):
            raise ValueError("path grants cannot cover the filesystem root")

    async def __aenter__(self):
        launching = asyncio.create_task(
            asyncio.create_subprocess_exec(
                sys.executable,
                str(Path(__file__).parent.parent / "scripts" / "process_worker.py"),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                # Worker failures belong in the caller's diagnostics, not model output.
                stderr=None,
                limit=1_000_000,
            )
        )
        try:
            self.process = await asyncio.shield(launching)
        except asyncio.CancelledError:
            while not launching.done():
                try:
                    await asyncio.shield(launching)
                except asyncio.CancelledError:
                    continue
            self.process = launching.result()
            await self.__aexit__(asyncio.CancelledError, None, None)
            raise
        self.process.stdin.write(
            json.dumps(
                {
                    "workspace": str(self.workspace),
                    "log_directory": str(self.log_directory),
                    "grants": [str(path) for path in self.grants],
                }
            ).encode()
            + b"\n"
        )
        try:
            await self.process.stdin.drain()
        except BaseException:
            await self.__aexit__(*sys.exc_info())
            raise
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        self.process.stdin.close()
        # Closing stdin lets the worker stop all owned commands and drain their output.
        closing = asyncio.create_task(self.process.wait())
        cancelled = False
        while not closing.done():
            try:
                await asyncio.shield(closing)
            except asyncio.CancelledError:
                cancelled = True
        if cancelled:
            raise asyncio.CancelledError
        if closing.result() and exc_type is None:
            raise RuntimeError(f"Process worker exited with code {closing.result()}")

    async def run(self, name: str, arguments: dict[str, JsonValue]) -> ToolResult:
        self.process.stdin.write(
            json.dumps({"name": name, "arguments": arguments}).encode() + b"\n"
        )
        try:
            await self.process.stdin.drain()
            response = await self.process.stdout.readline()
        except asyncio.CancelledError:
            await self.__aexit__(*sys.exc_info())
            raise
        if not response:
            code = await self.process.wait()
            raise RuntimeError(f"Process worker exited with code {code}")
        return ToolResult.model_validate_json(response)
