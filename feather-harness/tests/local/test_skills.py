from __future__ import annotations

import pytest

from feather_harness.local import install_skills
from feather_harness.prompt import SKILLS_ROOT
from feather_harness.scripts.install_skills import install_task_skills


def test_install_skills_preserves_complete_tree_and_accepts_repeat_install(tmp_path):
    install_skills(tmp_path)
    install_skills(tmp_path)
    expected = {
        path.relative_to(SKILLS_ROOT): path.read_bytes()
        for path in SKILLS_ROOT.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    actual = {
        path.relative_to(tmp_path / ".agents"): path.read_bytes()
        for path in (tmp_path / ".agents").rglob("*")
        if path.is_file()
    }
    assert actual == expected
    assert expected


@pytest.mark.parametrize("conflict", ["modified", "symlink", "parent_file"])
def test_skill_conflict_is_detected_before_installation_changes_workspace(tmp_path, conflict):
    source = sorted(SKILLS_ROOT.rglob("SKILL.md"))[-1]
    target = tmp_path / ".agents" / source.relative_to(SKILLS_ROOT)
    if conflict == "parent_file":
        (tmp_path / ".agents").write_text("user file")
    else:
        target.parent.mkdir(parents=True)
        if conflict == "modified":
            target.write_text("local edits")
        else:
            outside = tmp_path / "user-file"
            outside.write_text("user file")
            target.symlink_to(outside)
    before = {
        str(path.relative_to(tmp_path)): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    with pytest.raises(FileExistsError):
        install_skills(tmp_path)
    after = {
        str(path.relative_to(tmp_path)): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_install_task_skills_copies_harbor_skills_beside_bundled(tmp_path):
    install_skills(tmp_path)
    extra = tmp_path / "harbor-skills" / "harbor-task"
    extra.mkdir(parents=True)
    (extra / "SKILL.md").write_text("---\nname: harbor-task\ndescription: Extra.\n---\n")
    install_task_skills(tmp_path, extra.parent)
    assert (tmp_path / ".agents" / "harbor-task" / "SKILL.md").read_text() == (
        "---\nname: harbor-task\ndescription: Extra.\n---\n"
    )


def test_install_task_skills_fails_on_bundled_name_collision(tmp_path):
    install_skills(tmp_path)
    bundled = next(path for path in (tmp_path / ".agents").iterdir() if path.is_dir())
    extra = tmp_path / "harbor-skills" / bundled.name
    extra.mkdir(parents=True)
    (extra / "SKILL.md").write_text("collision")
    with pytest.raises(FileExistsError, match="conflicts with installed content"):
        install_task_skills(tmp_path, extra.parent)
