import json
import re
import shlex

import httpx
import pytest
from openai import AsyncOpenAI

from feather_harness import ChatCompletionsModel, McpServerSpec, run_local
from tests.local.test_cli import response
from tests.mcp.server import mcp_server


@pytest.mark.parametrize("profile", ["train", "prod"])
@pytest.mark.parametrize("single_line", [False, True], ids=["lines", "single-line-json"])
async def test_local_agent_can_retrieve_omitted_mcp_text_without_recalling_the_tool(tmp_path, profile, single_line):
    rows = [f"row-{index:04d}: {'é' * 80}" for index in range(1000)]
    source = json.dumps(rows, ensure_ascii=False) if single_line else "\n".join(rows) + "\n"
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 1:
            reply = response(tools=[("mcp__mail__echo", {"value": "query"})])
        elif len(requests) == 2:
            observed = payload["messages"][-1]["content"]
            preview = "".join(block["text"] for block in observed)
            assert len(preview) <= 24000
            assert "row-0500:" not in preview
            path = re.search(r"Complete returned output saved at (.+?)\.\]", preview).group(1)
            if single_line:
                assert "long single-line text/JSON, use a shell command to slice the saved file" in preview
                offset = source.encode().index(b"row-0500:")
                reply = response(tools=[("bash", {
                    "command": f"dd if={shlex.quote(path)} bs=1 skip={offset} count={len(rows[500].encode())} 2>/dev/null"
                })])
            else:
                reply = response(tools=[("read", {"file_path": path, "offset": 501, "limit": 1})])
        else:
            reply = response("Retrieved the requested row.")
        return httpx.Response(200, json=reply)

    with mcp_server(text_blocks=[source]) as (url, server):
        async with AsyncOpenAI(
            api_key="test", max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        ) as client:
            result = await run_local(
                "Find row 500", model=ChatCompletionsModel(client=client, model="test"),
                workspace=tmp_path, profile=profile,
                mcp_servers=(McpServerSpec(name="mail", transport="http", url=url + "/mcp/"),),
            )

    assert server.calls == [("echo", {"value": "query"})]
    assert rows[500] in requests[2]["messages"][-1]["content"]
    assert result.stop_reason == "completed"
