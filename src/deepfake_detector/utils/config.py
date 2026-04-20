"""Загрузка YAML-конфигурации + валидация через Pydantic."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class ModelCfg(BaseModel):
    embed_dim: int = 512
    n_layers: int = 6
    kernel: int = 3
    dropout: float = 0.1
    freeze_base: bool = False
    arc_s: float = 32.0
    arc_m: float = 0.5


class PreprocessingCfg(BaseModel):
    target_frames: int = 16
    max_side: int = 640
    face_min_conf: float = 0.92
    image_size: int = 224
    normalize_mean: list[float] = [0.485, 0.456, 0.406]
    normalize_std: list[float] = [0.229, 0.224, 0.225]


class TrainingCfg(BaseModel):
    batch: int = 4
    grad_accum: int = 16
    lr: float = 3e-4
    wd: float = 1e-4
    epochs: int = 60
    lam_arc: float = 1.0
    lam_tc: float = 0.2
    lam_short: float = 0.1
    mixed: bool = True
    num_workers: int = 2
    seed: int = 42


class InferenceCfg(BaseModel):
    device: str = "cpu"
    checkpoint_path: str = "checkpoints/best_model.pt"
    jit: bool = True
    dtype: str = "bfloat16"
    gradcam: bool = True
    threshold: float = 0.5


class ApiCfg(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
    max_upload_mb: int = 200
    celery_broker: str = "redis://localhost:6379/0"


class LoggingCfg(BaseModel):
    level: str = "INFO"
    tensorboard_dir: str = "runs"


class AdaptationCfg(BaseModel):
    adapt_lr: float = 1e-5
    adapt_batch: int = 32
    confidence_high: float = 0.9
    confidence_low: float = 0.1
    window: int = 1000
    kl_threshold: float = 0.35
    rollback_eer_tolerance: float = 0.02
    target_params: list[str] = ["head", "proj"]
    validation_set: str = "data/processed/adaptation_val/"


class AppConfig(BaseModel):
    model: ModelCfg = ModelCfg()
    preprocessing: PreprocessingCfg = PreprocessingCfg()
    training: TrainingCfg = TrainingCfg()
    inference: InferenceCfg = InferenceCfg()
    api: ApiCfg = ApiCfg()
    logging: LoggingCfg = LoggingCfg()
    adaptation: AdaptationCfg | None = None


def load_config(path: str | Path) -> AppConfig:
    """Загружает YAML и возвращает валидированный AppConfig."""
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    return AppConfig.model_validate(raw)
