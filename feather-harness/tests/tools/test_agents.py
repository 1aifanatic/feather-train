from __future__ import annotations

import json

import pytest

from feather_harness import AssistantMessage, FeatherAgent, RunResult
from feather_harness.subagents import AgentSession, Subagents
from feather_harness.tools.agents import subagent_tools

from ..helpers import ScriptedModel


async def test_wait_returns_latest_followup_answer_while_list_omits_answers() -> None:
    model = ScriptedModel(
        AssistantMessage(content="first answer"), AssistantMessage(content="updated answer")
    )

    async def run_child(agent_id: str, serve: AgentSession) -> RunResult:
        return await serve(FeatherAgent(model=model))

    async with Subagents(run_child, instructions="shared instructions") as agents:
        tools = {tool.spec.name: tool for tool in subagent_tools(agents)}
        spawned = await tools["spawn_agent"].invoke('{"prompt": "assignment"}')
        assert spawned.error is None
        child = json.loads(spawned.content)
        assert child["status"] == "running"
        wait_args = json.dumps({"agent_ids": [child["agent_id"]], "timeout": 1})
        first = await tools["wait_agent"].invoke(wait_args)
        assert first.error is None
        assert json.loads(first.content)[0]["answer"] == "first answer"

        followup = await tools["followup_agent"].invoke(
            json.dumps({"agent_id": child["agent_id"], "prompt": "update the answer"})
        )
        assert followup.error is None
        assert json.loads(followup.content)["status"] == "running"
        assert json.loads(followup.content)["answer"] == ""
        latest = await tools["wait_agent"].invoke(wait_args)
        assert latest.error is None
        assert json.loads(latest.content)[0]["answer"] == "updated answer"
        listed = await tools["list_agents"].invoke("{}")
        assert listed.error is None
        assert json.loads(listed.content) == [
            {"agent_id": child["agent_id"], "status": "completed", "turns": 2, "error": None}
        ]

        stopped = await tools["stop_agent"].invoke(json.dumps({"agent_id": child["agent_id"]}))
        assert stopped.error is None
        assert json.loads(stopped.content)["answer"] == "updated answer"
        rejected = await tools["followup_agent"].invoke(
            json.dumps({"agent_id": child["agent_id"], "prompt": "cannot reopen"})
        )
        assert rejected.error == "tool_error"


@pytest.mark.parametrize(
    "name, arguments, error, feedback",
    [
        ("spawn_agent", '{"prompt":"assignment", "unknown":true}', "invalid_arguments", "unknown"),
        ("wait_agent", '{"agent_ids":["unknown"], "timeout":601}', "invalid_arguments", "timeout"),
        ("spawn_agent", '{"prompt":"   "}', "tool_error", "must not be empty"),
        ("wait_agent", '{"agent_ids":["unknown"], "timeout":0}', "tool_error", "Unknown agent"),
    ],
)
async def test_invalid_tool_calls_report_errors_without_starting_children(
    name: str, arguments: str, error: str, feedback: str
) -> None:
    async def run_child(agent_id: str, serve: AgentSession) -> RunResult:
        raise AssertionError("invalid calls must not start a child")

    async with Subagents(run_child, instructions="shared instructions") as agents:
        tools = {tool.spec.name: tool for tool in subagent_tools(agents)}

        result = await tools[name].invoke(arguments)

        assert result.error == error
        assert feedback in result.content
        assert agents.list() == ()
