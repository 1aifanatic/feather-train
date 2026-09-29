import json
from pathlib import Path

assert json.loads(Path('/workspace/summary.json').read_text()) == {
    'total_quantity': 15, 'item_count': 3,
}
assert Path('/workspace/inventory.csv').read_text() == (
    'item,quantity\npencils,7\nnotebooks,3\nfolders,5\n'
)
