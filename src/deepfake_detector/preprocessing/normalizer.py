"""Нормализация кадров для подачи в EfficientNet-B0.

Согласно §5.2 диплома: применяет ImageNet-нормализацию.
Вход:  (T, H, W, 3) uint8  (RGB, значения 0–255).
Выход: (T, 3, H, W) float32 (нормированные, в масштабе ImageNet).
"""
from __future__ import annotations

import numpy as np
import torch
from torchvision.transforms import v2

__all__ = ["FrameNormalizer"]


class FrameNormalizer:
    """Конвертирует последовательность кадров в нормированный тензор PyTorch.

    Применяет поканальную нормализацию:
        x_norm = (x / 255 - mean) / std

    со стандартными ImageNet-параметрами (μ = 0.485/0.456/0.406,
    σ = 0.229/0.224/0.225), если иное не задано в конфигурации.

    Args:
        mean: Список из 3 значений средних по каналам RGB.
        std:  Список из 3 значений стандартных отклонений по каналам RGB.
    """

    def __init__(
        self,
        mean: list[float] | tuple[float, ...] = (0.485, 0.456, 0.406),
        std: list[float] | tuple[float, ...] = (0.229, 0.224, 0.225),
    ) -> None:
        self._transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=list(mean), std=list(std)),
        ])

    def __call__(self, frames: np.ndarray) -> torch.Tensor:
        """Нормализует последовательность кадров.

        Args:
            frames: np.ndarray (T, H, W, 3) uint8 RGB.

        Returns:
            torch.Tensor (T, 3, H, W) float32.
        """
        tensors = [self._transform(frame) for frame in frames]
        return torch.stack(tensors, dim=0)
