"""Structured JSON-lines logging: one line per attack attempt, per §4."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonlLogger:
    """Append-only JSON-lines sink. Creates parent directories on demand."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, default=str, sort_keys=True) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> JsonlLogger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
