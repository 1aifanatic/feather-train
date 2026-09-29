"""Build the tool bundle locally and install it at task startup."""

from __future__ import annotations

import argparse
import logging
import shlex
import tarfile
import uuid
from pathlib import Path

from e2b import AsyncSandbox
from feather_harness.local.filesystem import SCRIPTS
from feather_harness.production import Profile, validate_profile
from feather_harness.prompt import SKILLS_ROOT

from feather_train.sandbox.paths import WORKSPACE_DIR
from feather_train.sandbox.retries import REQUEST_TIMEOUT, write_file

logger = logging.getLogger(__name__)


def build_runtime(destination: Path, *, profile: Profile = "train") -> None:
    validate_profile(profile)
    with tarfile.open(destination, "w:gz") as tar:
        for source in sorted(SCRIPTS.glob("*.py")):
            tar.add(source, arcname=f"runtime/{source.name}")
        tar.add(Path(__file__).with_name("call.py"), arcname="runtime/call.py")
        tar.add(Path(__file__).with_name("remote.py"), arcname="host/remote.py")
        for source in sorted(SKILLS_ROOT.rglob("*")):
            if source.is_file() and "__pycache__" not in source.parts:
                tar.add(source, arcname=f"skills/{source.relative_to(SKILLS_ROOT)}")
        if profile == "prod":
            production_root = SKILLS_ROOT.parent / "production_skills"
            for source in sorted(production_root.rglob("*")):
                if source.is_file() and "__pycache__" not in source.parts:
                    tar.add(
                        source,
                        arcname=f"production_skills/{source.relative_to(production_root)}",
                    )


async def install_runtime(
    sandbox: AsyncSandbox,
    archive: Path,
    *,
    container: str = "agent",
    runtime: str = "/opt/feather",
    workspace: str = WORKSPACE_DIR,
    host_runtime: str = "/opt/feather-host",
    profile: Profile = "train",
) -> None:
    """One bulk upload and setup exec for each task sandbox."""
    validate_profile(profile)
    upload = f"/tmp/feather-{uuid.uuid4().hex}.tar.gz"
    with archive.open("rb") as source:
        await write_file(sandbox, upload, source)
    try:
        setup = (
            f"mkdir -p {shlex.quote(runtime)} {shlex.quote(workspace)} && "
            f"tar --exclude=host -xzf - -C {shlex.quote(runtime)} && "
            f"python3 {shlex.quote(runtime + '/runtime/install_skills.py')} {shlex.quote(workspace)} --overwrite --profile {profile} && "
            # Keep tool code non-writable; skills and task output have separate paths.
            f"chown root:root {shlex.quote(runtime)} && "
            f"chown -R root:root {shlex.quote(runtime + '/runtime')} && "
            f"chmod 0555 {shlex.quote(runtime)} {shlex.quote(runtime + '/runtime')} && "
            f"chmod 0444 {shlex.quote(runtime + '/runtime')}/*.py && "
            "python3 -c 'import charset_normalizer' && command -v rg >/dev/null"
        )
        command = shlex.join(["docker", "exec", "-i", "-u", "root", container, "sh", "-c", setup])
        setup_host = (
            f"mkdir -p {shlex.quote(host_runtime)} && "
            f"tar -xzf {shlex.quote(upload)} -C {shlex.quote(host_runtime)} --strip-components=1 host/remote.py && "
        )
        await sandbox.commands.run(
            setup_host + command + " < " + shlex.quote(upload),
            user="root",
            timeout=60,
            request_timeout=REQUEST_TIMEOUT,
        )
    finally:
        try:
            await sandbox.files.remove(upload, user="root", request_timeout=REQUEST_TIMEOUT)
        except Exception:
            logger.warning("Could not remove temporary runtime archive %s", upload, exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the Feather tool/skill archive for task startup."
    )
    parser.add_argument("destination", type=Path)
    build_runtime(parser.parse_args().destination)
