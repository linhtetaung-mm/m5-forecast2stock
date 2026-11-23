"""Git clean filter to retain notebook source but omit local execution outputs."""

import json
import sys

notebook = json.load(sys.stdin)
notebook.get("metadata", {}).pop("widgets", None)
for cell in notebook.get("cells", []):
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
        cell.get("metadata", {}).pop("execution", None)
json.dump(notebook, sys.stdout, ensure_ascii=False, indent=1)
sys.stdout.write("\n")

