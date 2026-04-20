"""CheckpointManager: реестр весов с откатом при деградации метрик."""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import torch


class CheckpointManager:
    """Хранит реестр весов: checkpoints/registry.json."""

    def __init__(self, checkpoint_dir: str | Path = "checkpoints"):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.dir / "registry.json"
        self.registry: list[dict] = []
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                self.registry = json.load(f)

    def save(self, state_dict: dict, metrics: dict) -> Path:
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        path = self.dir / f"checkpoint_{ts}.pt"
        torch.save(state_dict, path)
        entry = {"path": str(path), "metrics": metrics, "timestamp": ts}
        self.registry.append(entry)
        self._write_registry()
        return path

    def rollback(self) -> None:
        if len(self.registry) < 2:
            return
        bad = self.registry.pop()
        Path(bad["path"]).unlink(missing_ok=True)
        self._write_registry()

    def best(self) -> Path | None:
        if not self.registry:
            return None
        return Path(self.registry[-1]["path"])

    def _write_registry(self) -> None:
        with open(self.registry_path, "w") as f:
            json.dump(self.registry, f, indent=2)
