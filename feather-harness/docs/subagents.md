# Parallel subagents

Delegation is optional. Existing library calls and CLI configurations do not expose
agent tools or change instructions unless enabled. The local runner supports it in
`prod`; enabling it in `train` is rejected before starting tools.

```json
{
  "profile": "prod",
  "subagents": {"max_concurrent": 3, "max_agents": 8, "max_turns": 50}
}
```

These fields limit simultaneous active assignments, total children created in one
parent run, and each child's cumulative model turns across follow-ups. They are not
a dollar budget: context length, compaction calls, and provider retries also cost
tokens. The caller still owns inference timeouts and retry policy.

## Model-facing behavior

- `spawn_agent(prompt)` returns an ID immediately. The assignment must contain the
  necessary question, constraints, and source paths. Parent conversation history is
  not automatically copied.
- `wait_agent(agent_ids, timeout)` returns selected statuses and their latest answers.
  It waits for any selected child to finish; a timeout does not cancel children.
- `followup_agent(agent_id, prompt)` continues a successfully completed child's own
  context. It does not reset the lifetime turn limit. Running, failed, cancelled,
  exhausted, or explicitly stopped sessions cannot be continued.
- `stop_agent(agent_id)` stops that child and waits for resource cleanup.
- `list_agents()` lists this parent's children. IDs from another parent are not valid.

The parent remains responsible for incorporating results. Children cannot delegate.
A failed child is reported to the parent without stopping its siblings. Normal parent
completion, failure, and cancellation all close child sessions; unfinished assignments
are cancelled, not silently detached to continue spending or editing.

The existing loop still executes ordinary tools sequentially. Parallelism comes from
independent child runs, not from concurrently executing every tool in an assistant turn.

## Context and workspace ownership

Children have separate histories, todo lists, file-read revision state, shell workers,
MCP connections, and compaction transcripts. Completed children keep these resources
until closed, allowing follow-ups to use their own earlier work and background tasks.
Follow-up uses the active, possibly compacted context, not the expanded archival trace.

The filesystem is shared. Assign distinct files or responsibilities and preserve other
agents' edits. This is coordination guidance, not an atomic write lock or an OS sandbox.
For a shared workbook or document, parallel research or separate draft fragments are
usually easier to coordinate than concurrent writes to the same final file.

Local child traces live under `.feather/agents/<agent_id>/`, with a separate result for
each completed or failed activation, a compaction transcript when needed, and shell
logs. These records may contain source data and model output; treat them like the
parent trace. In-memory child sessions are not a crash-resumable job service.

## Embedding in an application

`Subagents` accepts a `run_child(agent_id, serve)` coroutine. The host creates a fresh
`FeatherAgent` and awaits `serve(agent)` inside that child's resource scopes. Serve
keeps the session alive between assignments. Keep MCP connection entry and exit in
their owning task, using `run_with_mcp` as the local runner does.

The host supplies its authenticated model adapter and sandboxed tools. Do not replace
an application's sandbox with `LocalShellRuntime`, reuse a parent's mutable tool
instances, or pass delegation tools to a child. A stateless API client may be shared;
streaming adapters must attribute each child's events to that child.

Register `subagent_tools(manager)` on the parent, append `PARENT_GUIDANCE` only when
those tools are present, and include `manager.compaction_state()` alongside existing
tool state. Close the manager before closing parent-owned clients. `save_result` can
persist each activation's `RunResult`; `results(agent_id)` exposes those same records.

This package does not implement desktop UI, approval dialogs, credential storage,
cross-process resume, peer messaging, nested teams, or automatic conflict resolution.
