# Code quality and style

Optimize for code that is easy to understand and review. Prefer the smallest
implementation that clearly expresses the supported behavior.

## Readability

- Prefer early returns over nested control flow. Keep variables close to their use.
  Put public interfaces near the top and private helpers below when it aids reading.
  Use a single leading underscore for non-public functions, methods, and helpers.
- Annotate functions and methods. Annotate locals only when their type is unclear.
  Use names that explain the value or action without a walkthrough.
- Comments explain non-obvious invariants, tradeoffs, algorithms, and edge cases.
  Add public API docstrings when the contract is not clear from the signature.
  Do not narrate obvious code.
- Avoid reflection (`getattr`, `hasattr`), broad input shapes, and type-based dispatch
  unless the problem is genuinely dynamic or external data needs normalization.
- Prefer top-level imports. Local imports are for optional or unusually expensive
  dependencies, not circular-import workarounds.
- Use logging for diagnostics and reserve stdout for intentional CLI output.
  Use deferred formatting with stdlib logging: `logger.info("count=%s", count)`.

## Design

- Trace callers and data flow before changing an abstraction. Ask what behavior
  requires each field, helper, branch, and copy of state. Give each value one
  authoritative home; derive redundant data only when the source is complete.
- Use Pydantic at untrusted boundaries: config, CLI input, API payloads, model output,
  and environment-derived settings. Parse into typed objects early and express
  relationships between settings there. Do not repeatedly validate trusted data.
- Prefer dataclasses for trusted internal state. Keep optional fields only when
  absence has meaning; use enums, tagged unions, or distinct types instead of
  combinations of unrelated optional fields.
- Prefer pure functions and immutable values where practical. Use frozen models
  when they clarify ownership or protect invariants; keep mutation localized.
- Prefer direct calls and small interfaces. Use protocols for interface-only
  polymorphism and shared bases when shared implementation earns them. Avoid
  factories, callback chains, and extension points for hypothetical consumers.
- Use lightweight dependency injection where collaborators naturally vary. Put
  genuinely shared types in a clearly named neutral module to avoid circular imports.
- Review public APIs for leaked internals and unclear contracts. Return data rather
  than presentation strings when callers need counts, flags, or structured results.

## Errors and concurrency

- Fail early on invalid configuration and broken invariants. Use specific exceptions
  when callers need the distinction; otherwise use built-in exceptions consistently.
- Catch errors where you can add context, translate, recover, or return a controlled
  failure. Do not mask invariant violations or let cleanup replace the original error.
- Retry only when safe to repeat. Prefer existing retry middleware where it fits;
  keep recovery logic explicit where it does not.
- Prefer structured concurrency. Use `asyncio.TaskGroup` when its sibling-cancellation
  behavior fits. Long-lived tasks need explicit ownership and shutdown behavior.
  Offload blocking I/O and CPU-heavy work from the event loop.
- Cleanup preserves behavior unless a change is intentional and called out. Preserve
  required ordering, cancellation, and measured performance benefits.

## Tests

- Test observable behavior through public APIs by default. Directly test pure helpers
  when they own a meaningful algorithm or boundary.
- Give each test one primary reason to fail: a concrete invariant, edge case, or rule.
  Avoid duplicate scenarios, trivial forwarding tests, private-state assertions,
  test-double bookkeeping, and expectations copied from the implementation.
- Organize by behavior with small, explicit fixtures. Prefer real lightweight
  collaborators. Patching environment variables, time, or temporary paths is fine
  when an abstraction would add complexity; patch internals only with a concrete reason.
- Use Docker when container behavior matters, not for pure logic or argument checks.
  Keep slow integration tests focused and mark them separately.
- Name tests for behavior, not migration history. Include meaningful tests with the
  behavior change; mechanical edits and docs do not need ceremonial tests. Broaden
  testing only when failures, changes, or unresolved risks justify it.

## Workflow

- Use `uv` and `pyproject.toml`. Run Ruff on changed Python code and the relevant tests
  before committing. Verify packaging changes with wheels and a clean installation.
- Do a simplification pass: question ownership and representation, not just formatting.
  Remove unnecessary state, optionality, defensive checks, and indirection while
  preserving required behavior. Stop when the scoped implementation is clear.
- Make one logical change per commit with its tests. Keep unrelated fixes and temporary
  validation scripts separate. Preserve additive review history unless asked to amend.
- Report what changed, verification, intentional behavior changes, and real uncertainty.
  Test counts alone do not establish correctness.
