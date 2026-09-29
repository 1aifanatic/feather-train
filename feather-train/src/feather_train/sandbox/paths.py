"""Absolute paths inside a coding sandbox"""

from __future__ import annotations

WORKSPACE_DIR = "/filesystem"
TASK_DIR = "/task"
TESTS_DIR = "/tests"
LOGS_DIR = "/logs"
VERIFIER_LOGS_DIR = f"{LOGS_DIR}/verifier"
SNAPSHOTS_DIR = "/snapshots"
# World ``tests/`` staged at provision/bake; grade-time restores into TESTS_DIR in the peer verifier.
BAKED_TESTS_DIR = "/opt/apex_grader_tests"
# JSON-state bearer: Compose injects the env into the backend; prepare materialises the
# value into this file inside the peer verifier for grade execs.
STATE_EXPORT_TOKEN_ENV = "APP_STATE_EXPORT_TOKEN"
STATE_EXPORT_TOKEN_PATH = "/run/feather_state_export_token"
