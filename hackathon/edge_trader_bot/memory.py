"""Small JSONL memory for local audit and future cross-tick features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from edge_trader_bot.json_utils import json_safe


class JsonlMemory:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(json_safe(record), sort_keys=True) + "\n")
