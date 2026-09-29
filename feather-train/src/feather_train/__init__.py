from feather_train.agent import run_e2b
from feather_train.sandbox.transport import E2BRuntime

# Verifiers scans these exports when loading training plugins; keep eval imports separate.
__all__ = [
    "E2BRuntime",
    "run_e2b",
    "FeatherHarness",
    "FeatherHarnessConfig",
    "FeatherTaskset",
]


def __getattr__(name):
    if name == "FeatherTaskset":
        from feather_train.tasks.taskset import FeatherTaskset

        return FeatherTaskset
    if name in {"FeatherHarness", "FeatherHarnessConfig"}:
        from . import harness

        return getattr(harness, name)
    if name in {"FeatherEnv", "load_environment"}:
        from . import environment

        return getattr(environment, name)
    raise AttributeError(name)
