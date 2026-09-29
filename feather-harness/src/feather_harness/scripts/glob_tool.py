"""Run a bounded glob search with a JSON request and result."""

from _files import main
from _search import search


def glob(payload):
    return search(payload, "glob")


if __name__ == "__main__":
    main(glob)
