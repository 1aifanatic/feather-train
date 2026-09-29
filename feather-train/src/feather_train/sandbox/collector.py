"""Launch commands through the host collector and recover their saved responses."""

from __future__ import annotations

import asyncio
import json
import logging
import shlex

from e2b import AsyncSandbox
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_incrementing

from feather_train.sandbox.retries import (
    PERMANENT_E2B_ERRORS,
    REQUEST_TIMEOUT,
    transient_error,
    write_file,
)

logger = logging.getLogger(__name__)
INLINE_COMMAND_BYTES = 32 * 1024


class Collector:
    def __init__(
        self, sandbox: AsyncSandbox, namespace: str, host_runtime: str = "/opt/feather-host"
    ):
        self.sandbox = sandbox
        self.namespace = namespace
        self.host_runtime = host_runtime

    def path(self, token: str) -> str:
        return f"/tmp/feather-{self.namespace}-{token}.json"

    async def prepare_launch(self, token: str, request: dict) -> str:
        payload = json.dumps(request).encode()
        launch_args = ["python3", f"{self.host_runtime}/remote.py", "run", self.path(token)]
        command = shlex.join(launch_args)
        # Avoid quoting large payloads that will need an upload regardless.
        inline = (
            shlex.join([*launch_args, payload.decode()])
            if len(payload) <= INLINE_COMMAND_BYTES
            else None
        )
        if inline is not None and len(inline.encode()) <= INLINE_COMMAND_BYTES:
            command = inline
        else:
            await write_file(self.sandbox, self.path(token), payload)
        return command

    async def execute(
        self,
        token: str,
        background: bool,
        deadline: float,
        command: str,
        *,
        wait_for_launch_on_cancel: bool = True,
    ):
        execution_timeout = asyncio.timeout_at(deadline)
        try:
            async with execution_timeout:
                return await self._receive(
                    token, background, deadline, command, wait_for_launch_on_cancel
                )
        except TimeoutError:
            if not execution_timeout.expired():
                raise
            # The stream can stall after publication. Inspect once using the
            # bounded read budget, without waiting for or replaying execution.
            try:
                state = await self.inspect(token, wait=False, wait_ms=0)
            except PERMANENT_E2B_ERRORS:
                raise
            except Exception as exc:
                raise TimeoutError("Tool deadline expired without a recoverable result") from exc
            if not background and not state["done"]:
                raise
            return state

    async def _receive(self, token, background, deadline, command, wait_for_launch_on_cancel):
        reply = asyncio.get_running_loop().create_future()
        chunks: list[str] = []

        def receive(text):
            if reply.done():
                return
            line, separator, _ = text.partition("\n")
            chunks.append(line)
            if separator:
                reply.set_result(json.loads("".join(chunks)))

        launch = asyncio.create_task(
            self.sandbox.commands.run(
                command,
                background=True,
                timeout=0,
                user="root",
                on_stdout=receive,
            )
        )
        handle = None
        waiter = None
        try:
            try:
                handle = await asyncio.shield(launch)
            except asyncio.CancelledError:
                if not wait_for_launch_on_cancel:
                    # Grader cleanup fences delayed starts without a launch acknowledgement.
                    # Agent tools need it to clean up their container supervisor.
                    launch.cancel()
                    await asyncio.gather(launch, return_exceptions=True)
                    if not launch.cancelled() and launch.exception() is None:
                        handle = launch.result()
                    raise
                try:
                    handle = await _drain(launch)
                except Exception:
                    logger.exception("E2B tool launch failed during cancellation")
                    raise asyncio.CancelledError from None
                raise
            waiter = asyncio.create_task(handle.wait())
            await asyncio.wait((reply, waiter), return_when=asyncio.FIRST_COMPLETED)
            if reply.done():
                return reply.result()
            waiter.result()
            raise RuntimeError("Tool stream ended without a result")
        except PERMANENT_E2B_ERRORS:
            # Preserve terminal provider errors for the rollout retry policy.
            raise
        except Exception:
            # Only inspect the existing call. Never replay a possibly executed tool.
            state = await self.inspect(
                token,
                wait=not background,
                wait_ms=max(0, int((deadline - asyncio.get_running_loop().time()) * 1000)),
            )
            if not background and not state["done"]:
                raise TimeoutError("Tool did not publish its final result")
            return state
        finally:
            if handle is not None:
                await handle.disconnect()
            if waiter is not None:
                await asyncio.gather(waiter, return_exceptions=True)

    async def inspect(self, token: str, *, wait: bool, wait_ms: int, fresh: bool = False):
        command = shlex.join(
            [
                "python3",
                f"{self.host_runtime}/remote.py",
                "inspect",
                self.path(token),
                "wait" if wait else "now",
                str(wait_ms),
                *(["fresh"] if fresh else []),
            ]
        )
        timeout = wait_ms / 1000 + 15
        # Retried reads share one budget rather than restarting a long task wait.
        async with asyncio.timeout(timeout):
            return json.loads(await self.read(command, timeout=timeout))

    async def read(self, command: str, *, timeout: float) -> str:
        # Reads may be retried: they cannot execute or duplicate a tool call.
        async for attempt in AsyncRetrying(
            retry=retry_if_exception(transient_error),
            stop=stop_after_attempt(3),
            wait=wait_incrementing(start=1, increment=1),
            reraise=True,
        ):
            with attempt:
                response = await self.sandbox.commands.run(
                    command,
                    user="root",
                    timeout=timeout,
                    request_timeout=REQUEST_TIMEOUT,
                )
                return response.stdout


async def _drain(task):
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()
