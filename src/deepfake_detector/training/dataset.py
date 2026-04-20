"""VideoDataset: читает CSV/JSON-манифест и возвращает видео + метки."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    """Загружает видео по манифесту формата [{path, label, manipulation}]."""

    def __init__(self, manifest_path: str | Path, preprocessor=None):
        self.preprocessor = preprocessor
        with open(manifest_path) as f:
            self.records = json.load(f)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        path = Path(rec["path"])
        label = int(rec["label"])
        meta = {k: v for k, v in rec.items() if k not in ("path", "label")}

        if self.preprocessor is not None:
            video = self.preprocessor(path)
        else:
            video = torch.zeros(16, 3, 224, 224)

        return {"video": video, "label": label, "meta": meta}
