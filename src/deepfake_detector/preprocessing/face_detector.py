"""Детекция и аффинное выравнивание лица через MTCNN.

Листинг 5.3 диплома (Зенковский А. М., ЮФУ).

Контракт:
    Вход:  RGB-кадр (H, W, 3) uint8.
    Выход: выравненное лицо (224, 224, 3) uint8 или None, если лицо не найдено.
"""
from __future__ import annotations

import numpy as np
import cv2
from facenet_pytorch import MTCNN

__all__ = ["FaceDetector"]

# Канонические координаты пяти ключевых точек в системе 224×224:
# левый глаз, правый глаз, нос, левый уголок рта, правый уголок рта.
_CANONICAL_LANDMARKS = np.float32([
    [70, 85], [154, 85], [112, 130], [80, 165], [144, 165],
])


class FaceDetector:
    """Детектор лиц на основе трёхкаскадной сети MTCNN.

    Для каждого кадра:
        1. MTCNN детектирует ограничивающий прямоугольник и пять точек.
        2. Если confidence ниже ``min_conf`` — возвращает None.
        3. Иначе вычисляет аффинное преобразование к каноническим координатам
           и выполняет ``cv2.warpAffine``.

    Args:
        min_conf: Минимальный порог confidence третьего каскада ONet (0–1).
        device:   Устройство PyTorch: ``"cpu"`` или ``"cuda"``.
    """

    def __init__(self, min_conf: float = 0.92, device: str = "cpu") -> None:
        self.mtcnn = MTCNN(
            image_size=224,
            keep_all=True,
            select_largest=True,
            post_process=False,
            device=device,
            thresholds=[0.6, 0.7, 0.7],
        )
        self.min_conf = min_conf
        self.canonical = _CANONICAL_LANDMARKS.copy()

    def __call__(self, frame: np.ndarray) -> np.ndarray | None:
        """Детектирует и выравнивает наиболее крупное лицо в кадре.

        Args:
            frame: RGB-изображение (H, W, 3) uint8.

        Returns:
            Выровненное лицо (224, 224, 3) uint8 или None.
        """
        boxes, probs, landmarks = self.mtcnn.detect(frame, landmarks=True)

        # boxes is None означает, что лиц не найдено вообще.
        if boxes is None:
            return None

        # probs может быть массивом; проверяем первое (наиболее крупное) лицо.
        if probs is None or float(probs[0]) < self.min_conf:
            return None

        # landmarks[0]: массив (5, 2) — точки первого (крупнейшего) лица.
        if landmarks is None:
            return None

        pts = np.float32(landmarks[0])
        M, _ = cv2.estimateAffinePartial2D(pts, self.canonical)
        if M is None:
            return None

        aligned = cv2.warpAffine(frame, M, (224, 224), flags=cv2.INTER_LINEAR)
        return aligned
