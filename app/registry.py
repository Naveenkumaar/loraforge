"""Adapter registry — save, list, and hot-swap adapters onto one base model.

Adapters are tiny JSON files (only the low-rank ``A``/``B`` matrices — never the
base weight), so shipping a new behaviour is kilobytes, not gigabytes. The
registry is the control plane: one base model is loaded once, and any registered
adapter can be attached at serve time and swapped for another with no reload.
Optional SQLite metadata makes the catalogue survive a restart.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from app.lora import LoRALinear


class AdapterRegistry:
    def __init__(self, root: str | Path = "adapters", db_path: str | None = None) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(db_path) if db_path else None
        if self._db:
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS adapters "
                "(name TEXT PRIMARY KEY, path TEXT, r INTEGER, alpha REAL, created REAL)")
            self._db.commit()

    def save(self, name: str, state: dict, created: float = 0.0) -> Path:
        path = self.root / f"{name}.json"
        path.write_text(json.dumps(state))
        if self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO adapters VALUES (?,?,?,?,?)",
                (name, str(path), int(state["r"]), float(state["alpha"]), created or time.time()))
            self._db.commit()
        return path

    def load(self, name: str) -> dict:
        path = self.root / f"{name}.json"
        if not path.exists():
            raise KeyError(f"no adapter named {name!r}")
        return json.loads(path.read_text())

    def names(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.json"))

    # ---- hot-swap ------------------------------------------------------
    def attach(self, layer: LoRALinear, name: str) -> LoRALinear:
        """Load a registered adapter onto a live base layer (no reload)."""
        layer.load_adapter(self.load(name))
        return layer

    def swap(self, layer: LoRALinear, name: str) -> LoRALinear:
        """Detach whatever is attached, then attach ``name`` — the hot-swap."""
        layer.clear_adapter()
        return self.attach(layer, name)
