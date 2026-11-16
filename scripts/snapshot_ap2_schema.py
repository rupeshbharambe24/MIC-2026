"""Snapshot the REAL AP2 mandate field names into a test fixture.

Run with AP2_REPO pointing at a clone of github.com/google-agentic-commerce/AP2.
The snapshot lets `tests/test_schema_parity.py` detect drift between our Pydantic
mirror and upstream without installing the AP2 SDK (which pins pydantic).
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

FILES = (
    "code/sdk/python/ap2/models/mandate.py",
    "code/sdk/python/ap2/models/payment_request.py",
)
OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "ap2_schema_snapshot.json"


def class_fields(path: Path) -> dict[str, list[str]]:
    tree = ast.parse(path.read_text())
    out: dict[str, list[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            out[node.name] = [
                s.target.id
                for s in node.body
                if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)
            ]
    return out


def main() -> int:
    repo = os.getenv("AP2_REPO")
    if not repo:
        print("set AP2_REPO=/path/to/AP2 clone", file=sys.stderr)
        return 2
    root = Path(repo)
    sha = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    snapshot: dict[str, object] = {
        "source": "github.com/google-agentic-commerce/AP2",
        "commit": sha,
    }
    classes: dict[str, list[str]] = {}
    for rel in FILES:
        classes.update(class_fields(root / rel))
    snapshot["files"] = list(FILES)
    snapshot["classes"] = classes
    OUT.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    print(f"snapshot @ {sha[:10]}: {len(classes)} classes -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
