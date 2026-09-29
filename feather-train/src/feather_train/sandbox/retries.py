"""Retry policies for operations whose callers know are safe to repeat."""

from __future__ import annotations

import logging
from typing import BinaryIO

import httpx
from e2b import AsyncSandbox, CommandExitException
from e2b.exceptions import (
    AuthenticationException,
    InvalidArgumentException,
    NotEnoughSpaceException,
    NotFoundException,
    SandboxException,
    TemplateException,
)
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)
REQUEST_TIMEOUT = 120
UPLOAD_TIMEOUT = 600
PERMANENT_E2B_ERRORS = (
    AuthenticationException,
    InvalidArgumentException,
    NotEnoughSpaceException,
    NotFoundException,
    TemplateException,
)


def transient_error(error: BaseException) -> bool:
    if isinstance(error, (CommandExitException, *PERMANENT_E2B_ERRORS)):
        return False
    return isinstance(
        error, (ConnectionError, TimeoutError, SandboxException, httpx.TransportError)
    )


async def write_file(sandbox: AsyncSandbox, path: str, data: bytes | BinaryIO) -> None:
    position = None if isinstance(data, bytes) else data.tell()
    async for attempt in AsyncRetrying(
        retry=retry_if_exception(transient_error),
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    ):
        with attempt:
            if position is not None:
                data.seek(position)
            await sandbox.files.write(path, data, user="root", request_timeout=UPLOAD_TIMEOUT)
