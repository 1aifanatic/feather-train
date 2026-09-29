"""Execute tool workers on the local filesystem as the current user."""

from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import JsonValue

from ..scripts._file_worker import run_file_worker
from ..scripts._files import _from_msys_path
from ..tools import ToolResult

SCRIPTS = Path(__file__).parent.parent / "scripts"


class LocalFileRuntime:
    def __init__(
        self,
        workspace: Path,
        *,
        read_grants: tuple[Path, ...] = (),
        write_grants: tuple[Path, ...] = (),
    ) -> None:
        self.workspace = Path(os.path.abspath(workspace))
        self.read_grants = tuple(Path(os.path.abspath(path)) for path in read_grants)
        self.write_grants = tuple(Path(os.path.abspath(path)) for path in write_grants)
        if not self.workspace.is_dir():
            raise ValueError("workspace must be a directory")
        if any(
            path.resolve() == path.resolve().parent
            for path in (*self.read_grants, *self.write_grants)
        ):
            raise ValueError("path grants cannot cover the filesystem root")

    async def run(self, name: str, arguments: dict[str, JsonValue]) -> ToolResult:
        if name in {"edit", "write"}:
            # Revision keys use lexical workspace-relative paths, not symlink targets.
            raw = arguments["file_path"]
            if os.name == "nt":
                raw = _from_msys_path(raw)
            path = Path(
                os.path.normpath(
                    os.path.join(
                        self.workspace,
                        raw.replace("\\", "/"),
                    )
                )
            )
            key = (
                str(path.relative_to(self.workspace))
                if path.is_relative_to(self.workspace)
                else str(path)
            )
            revisions = arguments["revisions"]
            arguments = {
                **arguments,
                "revisions": {key: revisions[key]} if key in revisions else {},
            }
        payload = json.dumps(
            {
                "workspace": str(self.workspace),
                "read_grants": [str(path) for path in self.read_grants],
                "write_grants": [str(path) for path in self.write_grants],
                "arguments": arguments,
            }
        ).encode()
        returncode, stdout, stderr = await run_file_worker(name, payload)
        if returncode:
            raise RuntimeError(f"{name} worker failed: {stderr.decode(errors='replace').strip()}")
        return ToolResult.model_validate_json(stdout)
