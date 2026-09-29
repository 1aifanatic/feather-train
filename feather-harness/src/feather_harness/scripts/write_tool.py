"""Create or overwrite a text file with revision checks."""

from pathlib import Path

from _files import (
    check_revision,
    load_text,
    main,
    require_current_read,
    require_file,
    validate_write_extension,
    write_text,
)


def write(payload):
    args = payload["arguments"]
    path, display, metadata = require_file(
        payload["workspace"],
        payload["read_grants"],
        args["file_path"],
        write_grants=payload["write_grants"],
        for_write=True,
        allow_missing=True,
    )
    validate_write_extension(path, display)
    if metadata is not None:
        expected = require_current_read(args["revisions"], display)
        _, encoding, actual = load_text(path, display, metadata)
        check_revision(display, expected, actual)
    else:
        encoding = "utf-8"
    # An exact file grant does not authorize creating its parent directories.
    create_parents = metadata is None and path not in [
        Path(grant) for grant in payload["write_grants"]
    ]
    revision = write_text(
        path,
        display,
        args["content"],
        encoding,
        root=payload["workspace"],
        write_grants=payload["write_grants"],
        create_parents=create_parents,
    )
    action = "Overwrote" if metadata is not None else "Created"
    return {
        "content": f"{action} {display}.",
        "metadata": {
            "path": display,
            "revision": revision,
            "encoding": encoding,
        },
    }


if __name__ == "__main__":
    main(write)
