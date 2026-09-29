# E2B tool execution

The remote runtime executes Feather's tools in the agent container while the loop
stays on the coordinator. The important contract is that losing a connection does
not cause a tool invocation to be repeated.

## Request flow

1. The coordinator assigns an invocation ID and launches a collector on the E2B host.
2. The collector uses `docker exec` to start a supervisor inside the agent container.
3. The supervisor runs the tool and publishes status, output, and its final result.
4. The collector saves those updates on the host and returns them to the coordinator.

File operations run together in one invocation, rather than issuing a remote call
for each read, revision check, or write. Foreground Bash returns on completion;
background Bash returns a task ID whose state can be inspected later.

| Saved data | Location and purpose |
| --- | --- |
| Invocation state and results | E2B host, for recovery and polling. |
| Command logs | Agent container, for inspection through tools. |

## Connection recovery and stopping

- A dropped reply may follow a successful command. Recovery checks the original
  invocation's saved state instead of launching another edit or command.
- Completed results remain available even if subsequent communication fails.
  A failed execution without a saved result is reported as a failure, not replayed.
- Background inspection reads state while the command continues. Heartbeats provide
  progress signals independently of the waiting client connection.
- Timeout and stop requests target the command's supervisor and owned processes
  inside the container. Stopping the host's `docker exec` process alone would not
  reliably stop its children.

## Runtime updates

Each restored world receives a runtime archive from the installed packages,
containing tool scripts and skills. Tool updates therefore do not require a world
rebake. Set `runtime_archive` to supply a prebuilt bundle.

Implementation: [transport](src/feather_train/sandbox/transport.py),
[collector](src/feather_train/sandbox/remote.py),
[supervisor](src/feather_train/sandbox/call.py).
