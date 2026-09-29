"""Publish a complete transcript without exposing a partially written file."""

from pathlib import Path


def save_transcript(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
