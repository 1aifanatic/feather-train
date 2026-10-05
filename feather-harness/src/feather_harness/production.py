"""Production prose and descriptions; the execution loop is shared with training."""

import json
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
    "Image inspection is enabled through read, including source images and rendered pages."
)
VISION_DISABLED_FACT = "Image inspection is not available; read handles text files only."
HOST_FACTS = (
    "No live user-interaction, status, or approval channel is available. "
    "Shell commands run locally with the host's permissions. "
    "Check the environment before relying on a renderer or calculation engine."
)
TRAIN_CATALOG_HEADING = "\n\n# Sandbox capability instructions\n\n"
PRODUCTION_CATALOG_HEADING = "\n\n# Tool capability instructions\n\n"


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
        instructions += f"\n\n## Additional system instructions\n\n{task_instructions}"
    return build_instructions(
        instructions, skills_root=skills_root, skills_path=skills_path
    ).replace(TRAIN_CATALOG_HEADING, PRODUCTION_CATALOG_HEADING, 1)


def production_tools(tools: tuple[Tool, ...], *, image_reads: bool) -> tuple[Tool, ...]:
    """Swap in production descriptions; `read` has one for each image-read mode."""
    result = []
    for tool in tools:
        name = tool.spec.name
        if name == "read" and not image_reads:
            name = "read_without_images"
        description = PRODUCTION_TOOL_DESCRIPTIONS.get(name)
        if description is None:
            result.append(tool)
            continue
        update = {"description": description}
        result.append(replace(tool, spec=tool.spec.model_copy(update=update)))
    return tuple(result)


def production_compaction(config: CompactionConfig) -> CompactionConfig:
    """Use the production continuation prompt unless the caller supplied its own."""
    if config.prompt != DEFAULT_COMPACTION_PROMPT:
        return config
    return config.model_copy(update={"prompt": PRODUCTION_COMPACTION_PROMPT})
