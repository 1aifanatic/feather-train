"""Production prose and descriptions; the execution loop is shared with training."""

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Literal, get_args

from .core.compaction import DEFAULT_COMPACTION_PROMPT, CompactionConfig
from .prompt import RESOURCE_ROOT, build_instructions
from .scripts.install_skills import WORKSPACE_SKILLS_PATH, render_skills_path
from .tools import Tool

Profile = Literal["train", "prod"]

PRODUCTION_PROMPT = (RESOURCE_ROOT / "production_prompt.md").read_text().strip()
PRODUCTION_COMPACTION_PROMPT = (
    (RESOURCE_ROOT / "production_compaction_prompt.md").read_text().strip()
)
PRODUCTION_TOOL_DESCRIPTIONS = json.loads(
    (RESOURCE_ROOT / "production_tool_descriptions.json").read_text()
)
VISION_ENABLED_FACT = (
    "Image inspection is enabled through read and view_image, including source images and "
    "rendered pages."
)
VISION_DISABLED_FACT = "Image inspection is disabled in this run; read handles text files only."
HOST_FACTS = (
    "This is a headless run with no live user-interaction, status, or approval channel. "
    "Shell commands run locally with the host's permissions. "
    "Check the environment before relying on a renderer or calculation engine."
)
TRAIN_CATALOG_HEADING = "\n\n# Sandbox capability instructions\n\n"
PRODUCTION_CATALOG_HEADING = "\n\n# Tool capability instructions\n\n"
IMAGE_READ_GUIDANCE = (
    "In this run, supported image paths return image content through the image reader "
    "rather than text; line ranges and text revision tracking do not apply to images. "
)
IMAGE_READ_PATH_HELP = (
    "Path to a text file or supported image inside the workspace. Relative paths resolve from "
    "the workspace root; absolute in-workspace or explicitly granted paths are also accepted."
)


def validate_profile(profile: Profile) -> None:
    if profile not in get_args(Profile):
        raise ValueError(f"Unknown Feather profile: {profile}")


def build_production_instructions(
    *,
    skills_root: Path,
    enable_image_tools: bool,
    task_instructions: str | None = None,
    host_facts: str = HOST_FACTS,
    skills_path: str = WORKSPACE_SKILLS_PATH,
    vision_fact: str | None = None,
) -> str:
    """Compose the production prompt for a host.

    `skills_path` is where the host installed the production skills, as written in the prompt.
    `vision_fact` replaces the image sentence derived from `enable_image_tools` for a host whose
    image support differs from both standard cases.
    """
    if vision_fact is None:
        vision_fact = VISION_ENABLED_FACT if enable_image_tools else VISION_DISABLED_FACT
    prompt = render_skills_path(PRODUCTION_PROMPT, skills_path)
    instructions = f"{prompt}\n\n## Host capabilities\n\n{vision_fact} {host_facts}"
    default_instructions = (RESOURCE_ROOT / "system_prompt.md").read_text().strip()
    if task_instructions and task_instructions.strip() != default_instructions:
        instructions += f"\n\n## Task-specific system instructions\n\n{task_instructions}"
    return build_instructions(
        instructions, skills_root=skills_root, skills_path=skills_path
    ).replace(TRAIN_CATALOG_HEADING, PRODUCTION_CATALOG_HEADING, 1)


def production_tools(tools: tuple[Tool, ...], *, enable_image_tools: bool) -> tuple[Tool, ...]:
    result = []
    for tool in tools:
        description = PRODUCTION_TOOL_DESCRIPTIONS.get(tool.spec.name)
        if description is None:
            result.append(tool)
            continue
        changes = {"description": description}
        if tool.spec.name == "read":
            if enable_image_tools:
                parameters = deepcopy(tool.spec.parameters)
                parameters["properties"]["file_path"]["description"] = IMAGE_READ_PATH_HELP
                changes["parameters"] = parameters
            else:
                changes["description"] = description.replace(IMAGE_READ_GUIDANCE, "")
        result.append(replace(tool, spec=tool.spec.model_copy(update=changes)))
    return tuple(result)


def production_compaction(config: CompactionConfig) -> CompactionConfig:
    """Use the production continuation prompt unless the caller supplied its own."""
    if config.prompt != DEFAULT_COMPACTION_PROMPT:
        return config
    return config.model_copy(update={"prompt": PRODUCTION_COMPACTION_PROMPT})
