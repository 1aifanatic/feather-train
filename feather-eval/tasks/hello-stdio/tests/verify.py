from pathlib import Path

assert Path("/app/token.txt").read_text().strip() == "harbor-stdio-ok"
