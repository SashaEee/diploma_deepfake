"""Аугментации для обучения: RandAugment, CutMix, MixUp, JPEG-noise и т.д."""
from __future__ import annotations

import random
import io

import numpy as np
import torch
from PIL import Image


def jpeg_compress(img: np.ndarray, quality: int | None = None) -> np.ndarray:
    """Перекодирование JPEG с качеством в диапазоне [30, 95]."""
    q = quality if quality is not None else random.randint(30, 95)
    pil = Image.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return np.array(Image.open(buf))


def add_gaussian_noise(img: np.ndarray, sigma_max: float = 0.03) -> np.ndarray:
    """Шум Гаусса в нормированном пространстве [0, 1]."""
    sigma = random.uniform(0, sigma_max)
    noise = np.random.randn(*img.shape).astype(np.float32) * sigma
    return np.clip(img.astype(np.float32) / 255.0 + noise, 0.0, 1.0)


def cutmix(video_a: torch.Tensor, video_b: torch.Tensor, alpha: float = 1.0):
    """CutMix по пространственным осям одного кадра."""
    lam = np.random.beta(alpha, alpha)
    _, _, h, w = video_a.shape
    cut_h = int(h * (1 - lam) ** 0.5)
    cut_w = int(w * (1 - lam) ** 0.5)
    cx = random.randint(0, w)
    cy = random.randint(0, h)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, w)
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, h)
    mixed = video_a.clone()
    mixed[:, :, y1:y2, x1:x2] = video_b[:, :, y1:y2, x1:x2]
    lam_actual = 1 - (x2 - x1) * (y2 - y1) / (h * w)
    return mixed, lam_actual


def mixup(video_a: torch.Tensor, video_b: torch.Tensor, alpha: float = 0.2):
    """MixUp с коэффициентом α = 0.2."""
    lam = np.random.beta(alpha, alpha)
    return lam * video_a + (1 - lam) * video_b, lam
