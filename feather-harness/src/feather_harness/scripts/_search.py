"""Bounded ripgrep search; child processes never leave the sandbox worker."""

import base64
import json
import os
import re
import stat
import subprocess
import tempfile
import threading
from collections.abc import Callable, Iterator
from io import BufferedReader
from pathlib import Path
from typing import NoReturn

from _files import MAX_FILE_SIZE_BYTES, FileToolError, require_path

MAX_BYTES = 20_000
MAX_BRACE_DEPTH = 256


def search(payload, mode):
    args = payload["arguments"]
    root, display, metadata = require_path(
        payload["workspace"],
        payload["read_grants"],
        args["path"],
        write_grants=payload["write_grants"],
        parameter="path",
    )
    if metadata is None:
        kind = "directory" if mode == "glob" else "path"
        raise FileToolError(f"Invalid `path` parameter: {args['path']}. The {kind} does not exist.")
    is_file = stat.S_ISREG(metadata.st_mode)
    if mode == "glob" and not stat.S_ISDIR(metadata.st_mode):
        raise FileToolError(
            f"Invalid `path` parameter: {args['path']}. The path is not a directory."
        )
    if mode == "grep" and not (is_file or stat.S_ISDIR(metadata.st_mode)):
        raise FileToolError(
            f"Invalid `path` parameter: {args['path']}. The path is not a regular file or directory."
        )
    include = args.get("include") if not is_file else None
    for value in (args["pattern"], include):
        if value is not None and "\0" in value:
            reject_pattern("search patterns must not contain NUL bytes")
    result = BoundedSearch(str(root), payload["workspace"], args["limit"])
    if mode == "glob":
        result.collect_glob(args["pattern"])
    elif include is None:
        validate_regex(args["pattern"])
        result.collect_grep(result.grep_command(args["pattern"], [str(root)]))
    else:
        result.collect_grep_with_include(args["pattern"], include)
    return {"content": format_result(result, mode, args["pattern"], display, is_file, include)}


class BoundedSearch:
    def __init__(self, root: str, workspace: str, limit: int):
        self.root = root
        self.workspace = workspace
        self.limit = limit
        self.records: list[str] = []
        # Reserve space for the search response envelope, matching the pinned worker.
        self.record_bytes = 256
        self.observed = 0
        self.count_truncated = False
        self.byte_truncated = False

    def add_record(self, record: str) -> bool:
        self.observed += 1
        if self.observed > self.limit:
            self.count_truncated = True
            return False
        serialized = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        encoded_size = len(serialized.encode("utf-8"))
        separator_size = 1 if self.records else 0
        if self.record_bytes + separator_size + encoded_size > MAX_BYTES:
            self.byte_truncated = True
            return False
        self.records.append(record)
        self.record_bytes += separator_size + encoded_size
        return True

    def display_path(self, path: str) -> str:
        candidate = Path(path)
        if candidate.is_relative_to(self.workspace):
            return candidate.relative_to(self.workspace).as_posix()
        return candidate.as_posix()

    def collect_grep(self, command: list[str]) -> bool:
        process, errors = start_captured(command, stdout=subprocess.PIPE)
        try:
            while True:
                raw = process.stdout.readline(MAX_BYTES + 1)
                if not raw:
                    break
                # rg emits one JSON event per line. Stop before buffering an
                # unbounded matching line, even if no result has fit yet.
                if len(raw) > MAX_BYTES and not raw.endswith(b"\n"):
                    if raw.startswith(b'{"type":"match"'):
                        self.observed += 1
                    self.byte_truncated = True
                    return False
                event = json.loads(raw)
                if event.get("type") != "match":
                    continue
                data = event["data"]
                path = decode_field(data["path"])
                line = decode_field(data["lines"]).rstrip("\r\n")
                record = f"{self.display_path(path)}:{data.get('line_number')}:{line}"
                if not self.add_record(record):
                    return False
            check_exit(process, errors)
        finally:
            stop(process)
            errors.finish()
        return True

    def collect_glob(self, pattern: str) -> None:
        check_glob_complexity(pattern)
        validate_glob(pattern)
        matches = make_glob_matcher(pattern)
        command = self.list_files_command()
        process, errors = start_captured(command, stdout=subprocess.PIPE)
        try:
            for raw_path in read_null_records(process.stdout):
                path = raw_path.decode("utf-8", errors="replace")
                relative = os.path.relpath(path, self.root).replace(os.sep, "/")
                if not matches(relative):
                    continue
                if not self.add_record(self.display_path(path)):
                    return
            check_exit(process, errors)
        finally:
            stop(process)
            errors.finish()

    def list_files_command(self) -> list[str]:
        return [
            "rg",
            "--no-config",
            "--files",
            "--null",
            "--line-buffered",
            "--sort",
            "path",
            "--",
            self.root,
        ]

    def grep_command(self, pattern: str, paths: list[str]) -> list[str]:
        return [
            "rg",
            "--no-config",
            "--json",
            "--line-buffered",
            "--sort",
            "path",
            "--max-count",
            str(self.limit + 1),
            "--max-filesize",
            str(MAX_FILE_SIZE_BYTES),
            "-e",
            pattern,
            "--",
            *paths,
        ]

    def collect_grep_with_include(self, pattern: str, include: str) -> None:
        validate_regex(pattern)
        check_glob_complexity(include)
        validate_glob(include)
        matches = make_glob_matcher(include)
        enumeration = self.list_files_command()
        process, errors = start_captured(enumeration, stdout=subprocess.PIPE)
        try:
            # Search fixed-size batches in rg's path order, stopping as soon
            # as the shared result budget is exhausted.
            batch: list[str] = []
            for raw_path in read_null_records(process.stdout):
                path = raw_path.decode("utf-8", errors="replace")
                relative = os.path.relpath(path, self.root).replace(os.sep, "/")
                if not matches(relative):
                    continue
                batch.append(path)
                if len(batch) == 64:
                    if not self.collect_grep(self.grep_command(pattern, batch)):
                        return
                    batch = []
            check_exit(process, errors)
        finally:
            stop(process)
            errors.finish()
        if batch:
            self.collect_grep(self.grep_command(pattern, batch))


def reject_pattern(message: str) -> NoReturn:
    raise FileToolError(message.rstrip())


# Drain stderr concurrently: a full stderr pipe must not block reading stdout.
# Retain only a bounded prefix even when ripgrep reports many filesystem errors.
class ErrorCapture:
    def __init__(self, stream: BufferedReader):
        self.stream = stream
        self.data = bytearray()
        self.thread = threading.Thread(target=self._drain, daemon=True)
        self.thread.start()

    def _drain(self) -> None:
        while chunk := self.stream.read(8192):
            remaining = 65536 - len(self.data)
            if remaining > 0:
                self.data.extend(chunk[:remaining])

    def finish(self) -> bytes:
        self.thread.join()
        return bytes(self.data)


def start_captured(
    command: list[str],
    *,
    stdout: int,
) -> tuple[subprocess.Popen[bytes], ErrorCapture]:
    process = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=stdout)
    return process, ErrorCapture(process.stderr)


def stop(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def check_exit(process: subprocess.Popen[bytes], errors: ErrorCapture) -> None:
    code = process.wait()
    error = errors.finish()
    if code not in (0, 1):
        raise RuntimeError(error.decode("utf-8", errors="replace"))


def validate(command: list[str]) -> None:
    process, errors = start_captured(command, stdout=subprocess.DEVNULL)
    try:
        code = process.wait()
        error = errors.finish()
        if code not in (0, 1):
            if code == 2:
                reject_pattern(error.decode("utf-8", errors="replace"))
            raise RuntimeError(error.decode("utf-8", errors="replace"))
    finally:
        stop(process)
        errors.finish()


# ripgrep uses base64 for filenames or lines that are not valid UTF-8.
def decode_field(field: dict[str, str]) -> str:
    if "text" in field:
        return field["text"]
    return base64.b64decode(field["bytes"]).decode("utf-8", errors="replace")


# NUL separates filenames without confusing embedded spaces or newlines.
def read_null_records(stream: BufferedReader) -> Iterator[bytes]:
    pending = b""
    while True:
        chunk = stream.read1(8192)
        if not chunk:
            break
        pending += chunk
        parts = pending.split(b"\0")
        pending = parts.pop()
        for part in parts:
            yield part
        if len(pending) > 65536:
            raise RuntimeError("search path exceeded 65536 bytes")
    if pending:
        yield pending


def expand_braces(pattern: str) -> list[str]:
    pending = [pattern]
    expanded = []
    while pending:
        candidate = pending.pop()
        bounds = find_first_brace(candidate)
        if bounds is None:
            expanded.append(candidate)
            if len(expanded) > 256:
                raise ValueError("glob pattern is too complex")
            continue
        start, end = bounds
        alternatives = split_alternatives(candidate[start + 1 : end])
        if len(pending) + len(expanded) + len(alternatives) > 256:
            raise ValueError("glob pattern is too complex")
        prefix = candidate[:start]
        suffix = candidate[end + 1 :]
        # This is a stack: reverse the alternatives to retain their input order.
        pending.extend(prefix + alternative + suffix for alternative in reversed(alternatives))
    return expanded


def find_first_brace(pattern: str) -> tuple[int, int] | None:
    escaped = False
    start = None
    depth = 0
    for index, char in enumerate(pattern):
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
            if depth > MAX_BRACE_DEPTH:
                raise ValueError("glob pattern is too complex")
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                return start, index
    return None


def split_alternatives(body: str) -> list[str]:
    parts = []
    current = []
    depth = 0
    escaped = False
    for char in body:
        if escaped:
            current.extend(("\\", char))
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == "{":
            depth += 1
            current.append(char)
        elif char == "}":
            depth -= 1
            current.append(char)
        elif char == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    if escaped:
        current.append("\\")
    parts.append("".join(current))
    return parts


def translate_glob(pattern: str) -> str:
    # A bare filename pattern matches at any depth; paths start at the root.
    anchored = "/" in pattern
    pattern = pattern.removeprefix("/")
    prefix = "^" if anchored else r"(?:^|.*/)"
    output = [prefix]
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char == "*":
            if index + 1 < len(pattern) and pattern[index + 1] == "*":
                start = index
                while index < len(pattern) and pattern[index] == "*":
                    index += 1
                starts_component = start == 0 or pattern[start - 1] == "/"
                ends_component = index == len(pattern) or pattern[index] == "/"
                # Only a whole ** component crosses directory boundaries.
                component_globstar = starts_component and ends_component
                if component_globstar and index < len(pattern):
                    output.append(r"(?:.*/)?")
                    index += 1
                elif component_globstar:
                    output.append(".*")
                else:
                    output.append(r"[^/]*")
                continue
            output.append(r"[^/]*")
        elif char == "?":
            output.append(r"[^/]")
        elif char == "[":
            end = index + 1
            if end < len(pattern) and pattern[end] in ("!", "^"):
                end += 1
            # A leading ] is a member of the character class, not its end.
            if end < len(pattern) and pattern[end] == "]":
                end += 1
            while end < len(pattern) and pattern[end] != "]":
                end += 1
            content = pattern[index + 1 : end]
            if content.startswith("!"):
                content = "^" + content[1:]
            output.extend(("[", content.replace("\\", r"\\"), "]"))
            index = end
        elif char == "\\" and index + 1 < len(pattern):
            index += 1
            output.append(re.escape(pattern[index]))
        else:
            output.append(re.escape(char))
        index += 1
    output.append("$")
    return "".join(output)


# Filter the normal rg file listing ourselves: passing --glob directly can
# override ignore rules and include files the existing tool excludes.
def make_glob_matcher(pattern: str) -> Callable[[str], bool]:
    negated = pattern.startswith("!")
    if negated:
        pattern = pattern[1:]
    elif pattern.startswith(r"\!"):
        pattern = pattern[1:]
    try:
        expressions = [re.compile(translate_glob(item)) for item in expand_braces(pattern)]
    except (ValueError, re.error) as exc:
        reject_pattern(str(exc))

    def matches(path: str) -> bool:
        matched = any(expression.match(path) for expression in expressions)
        return not matched if negated else matched

    return matches


def check_glob_complexity(pattern: str) -> None:
    try:
        expand_braces(pattern)
    except (ValueError, re.error) as exc:
        reject_pattern(str(exc))


# Validate against an empty directory: ripgrep exits 1 when it finds nothing
# and 2 on a bad pattern, on every platform (Windows has no /dev/null).
def validate_glob(pattern: str) -> None:
    with tempfile.TemporaryDirectory() as empty:
        validate(["rg", "--no-config", "--files", "--glob", pattern, "--", empty])


def validate_regex(pattern: str) -> None:
    with tempfile.TemporaryDirectory() as empty:
        validate(["rg", "--no-config", "--json", "-e", pattern, "--", empty])


def format_result(
    result: BoundedSearch,
    mode: str,
    pattern: str,
    display: str,
    is_file: bool,
    include: str | None,
) -> str:
    if is_file:
        location = f"in {display}"
    elif display == ".":
        location = "in the workspace"
    else:
        location = f"under {display}"

    if mode == "glob":
        noun = "files"
        advice = "Narrow the pattern or path."
    else:
        noun = "matches"
        advice = "Narrow the pattern, include filter, or path."
    scope = f" in files matching {include!r}" if include else ""
    if not result.records:
        if result.byte_truncated:
            return (
                f"Showing 0 of at least {result.observed} {noun} within the "
                f"{MAX_BYTES:,}-byte search output for {pattern!r}{scope} "
                f"{location}. {advice}"
            )
        if mode == "glob":
            return f"No files matched {pattern!r} {location}."
        return f"No matches found for {pattern!r}{scope} {location}."
    output = "\n".join(result.records)
    if result.count_truncated:
        output += f"\nShowing {result.limit} of at least {result.limit + 1} {noun}. {advice}"
    elif result.byte_truncated:
        output += (
            f"\nShowing {len(result.records)} of at least {result.observed} {noun} "
            f"within the {MAX_BYTES:,}-byte search output. {advice}"
        )
    return output
