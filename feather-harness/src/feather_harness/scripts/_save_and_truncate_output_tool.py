"""Spill and bound a host-formatted file-tool response where the tools execute."""

from _files import main, save_output_preview


def truncate(payload):
    args = payload["arguments"]
    head, notice, tail, saved_path = save_output_preview(
        args["content"], args["tool_name"], workspace=payload["workspace"]
    )
    return {
        "content": head + notice + tail,
        "metadata": {"head_chars": len(head), "tail_chars": len(tail), "saved_path": saved_path},
    }


if __name__ == "__main__":
    main(truncate)
