"""Read a text file. Input and output are JSON on stdin/stdout."""

import re

from _files import (
    DOCUMENT_EXTENSIONS,
    IMAGE_EXTENSIONS,
    FileToolError,
    load_text,
    main,
    require_file,
)
from view_image_tool import view_image


def read(payload):
    args = payload["arguments"]
    path, display, metadata = require_file(
        payload["workspace"],
        payload["read_grants"],
        args["file_path"],
        write_grants=payload["write_grants"],
    )
    extension = path.suffix.lower()
    if extension in IMAGE_EXTENSIONS:
        if args.get("enable_image_tools", False):
            return view_image(payload)
        return {"content": "Vision is not available in this environment."}
    if extension in DOCUMENT_EXTENSIONS:
        raise FileToolError(
            f"File validation failed for {display}: This tool reads text files only; the requested path has a binary document format."
        )
    text, encoding, revision = load_text(path, display, metadata)
    lines = re.split(r"\r\n|\r|\n", text)
    if lines[-1] == "":
        lines.pop()
    offset = args["offset"]
    if not lines and offset != 1:
        raise FileToolError(
            f"Invalid `offset` parameter: {offset}. The file is empty; use offset=1.",
        )
    if lines and offset > len(lines):
        raise FileToolError(
            f"Invalid `offset` parameter: {offset}. The file has {len(lines)} lines; choose an offset from 1 to {len(lines)}.",
        )
    end = min(len(lines), offset + args["limit"] - 1)
    selected = lines[offset - 1 : end] if lines else [""]
    numbered = "\n".join(
        f"{line_number:6}\t{line}" for line_number, line in enumerate(selected, offset)
    )
    content = f"Contents of {display}:\n{numbered}\n"
    if end < len(lines):
        content += f"Showing lines {offset}-{end} of {len(lines)}. Call read again with offset={end + 1} to continue."
    return {
        "content": content,
        "metadata": {
            "path": display,
            "revision": revision,
            "encoding": encoding,
        },
    }


if __name__ == "__main__":
    main(read)
