"""Read a path-confined image into a structured model-visible result."""

import base64

from _files import MAX_FILE_SIZE_BYTES, FileToolError, main, require_file


def view_image(payload):
    path, display, metadata = require_file(
        payload["workspace"],
        payload["read_grants"],
        payload["arguments"]["file_path"],
        write_grants=payload["write_grants"],
    )
    if metadata.st_size > MAX_FILE_SIZE_BYTES:
        raise FileToolError("Image exceeds the 10 MiB limit; resize it before inspection.")
    with path.open("rb") as stream:
        data = stream.read(MAX_FILE_SIZE_BYTES + 1)
    if len(data) > MAX_FILE_SIZE_BYTES:
        raise FileToolError("Image exceeds the 10 MiB limit; resize it before inspection.")
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    elif data.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    elif data.startswith((b"GIF87a", b"GIF89a")):
        mime = "image/gif"
    elif data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        mime = "image/webp"
    else:
        raise FileToolError("Expected a PNG, JPEG, GIF or WebP image; render documents to PNG first.")
    encoded = base64.b64encode(data).decode("ascii")
    return {"content": [
        {"type": "text", "text": f"Image from {display}:"},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
    ]}


if __name__ == "__main__":
    main(view_image)
