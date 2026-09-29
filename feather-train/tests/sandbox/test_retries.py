import asyncio
import io
from functools import partial
from types import SimpleNamespace

import pytest
from e2b.exceptions import AuthenticationException, NotEnoughSpaceException
from tenacity import AsyncRetrying

from feather_train.sandbox import retries
from feather_train.sandbox.retries import write_file


async def test_upload_retry_rewinds_to_original_stream_position():
    received = []

    class Files:
        async def write(self, path, data, **kwargs):
            received.append(data.read())
            assert kwargs["request_timeout"] == 600
            if len(received) == 1:
                raise ConnectionError("upload reply lost")

    stream = io.BytesIO(b"prefix-payload")
    stream.seek(7)
    await write_file(SimpleNamespace(files=Files()), "/tmp/input", stream)
    assert received == [b"payload", b"payload"]


async def test_upload_exhausts_ten_attempts_and_preserves_exception(monkeypatch):
    waits = []
    calls = []
    failure = ConnectionError("unavailable")

    async def sleep(delay):
        waits.append(delay)

    # Replace only Tenacity's clock; run the real policy and upload code.
    monkeypatch.setattr(retries, "AsyncRetrying", partial(AsyncRetrying, sleep=sleep))

    class Files:
        async def write(self, *args, **kwargs):
            calls.append(True)
            raise failure

    with pytest.raises(ConnectionError) as raised:
        await write_file(SimpleNamespace(files=Files()), "/tmp/input", b"payload")
    assert raised.value is failure
    assert len(calls) == 10
    assert waits == [1, 2, 4, 8, 16, 32, 64, 128, 256]


@pytest.mark.parametrize(
    "failure",
    [
        AuthenticationException("invalid key"),
        NotEnoughSpaceException("disk full"),
        asyncio.CancelledError(),
    ],
)
async def test_upload_does_not_retry_permanent_errors_or_cancellation(failure):
    calls = []

    class Files:
        async def write(self, *args, **kwargs):
            calls.append(True)
            raise failure

    with pytest.raises(type(failure)):
        await write_file(SimpleNamespace(files=Files()), "/tmp/input", b"payload")
    assert len(calls) == 1
