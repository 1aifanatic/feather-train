"""Write a copy of a bundled icon in another color, keeping its transparency."""

import argparse
from pathlib import Path

from PIL import Image

ICONS = Path(__file__).resolve().parent


def recolor(name: str, color: str, output: Path) -> None:
    source = Image.open(ICONS / f"{name}.png")
    alpha = source.convert("RGBA").getchannel("A")
    hex_color = color.lstrip("#")
    icon = Image.new("RGBA", alpha.size, tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4)))
    icon.putalpha(alpha)
    # The shared resolution is what places every icon at one scale.
    icon.save(output, dpi=source.info["dpi"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="Icon name as listed in index.tsv")
    parser.add_argument("color", help="Hex color such as FFFFFF")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    recolor(args.name, args.color, args.output)
