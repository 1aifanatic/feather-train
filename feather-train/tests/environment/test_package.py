"""The replacement must work when the legacy packages are unavailable."""

import os
import subprocess
import sys


def test_environment_does_not_import_legacy_packages():
    script = """
import importlib.abc
import sys

class RejectLegacyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'harness', 'agents', 'feather_agent', 'tinker', 'fireworks'}:
            raise ModuleNotFoundError(f'Legacy package unavailable: {fullname}', name=fullname)

sys.meta_path.insert(0, RejectLegacyImports())
from feather_train import load_environment
from feather_train.clients.renderer import RendererClient

env = load_environment(hf_dataset='unused/test', hf_revision='a' * 40,
                       enable_braintrust=False)
assert env.harness.config.id == 'feather_train'
assert env.taskset.config.id == 'feather_train'
"""
    # Carry the test environment's dependency paths into a fresh interpreter.
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    result = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
