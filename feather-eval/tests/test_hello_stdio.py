import sys
from pathlib import Path

from feather_harness import McpServerSpec, run_with_mcp
from feather_harness.core.runner import RunResult

_SERVER = Path(__file__).resolve().parents[1] / "tasks/hello-stdio/environment/server.py"


async def test_hello_stdio_fixture_serves_get_token() -> None:
    async def run(tools):
        assert [tool.spec.name for tool in tools] == ["mcp__token__get_token"]
        result = await tools[0].invoke("{}")
        assert result.content == [{"text": "harbor-stdio-ok", "type": "text"}]
        return RunResult(stop_reason="completed", num_turns=0, compactions=0, segments=())

    outcome = await run_with_mcp(
        (
            McpServerSpec(
                name="token",
                transport="stdio",
                command=sys.executable,
                args=(str(_SERVER),),
            ),
        ),
        run,
    )
    assert outcome.stop_reason == "completed"
