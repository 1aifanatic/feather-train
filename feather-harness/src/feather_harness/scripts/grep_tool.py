"""Run a bounded grep search with a JSON request and result."""

from _files import main
from _search import search


def grep(payload):
    return search(payload, "grep")


if __name__ == "__main__":
    main(grep)
