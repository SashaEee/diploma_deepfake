"""Применение FaceDetector к последовательности кадров видео.

Вспомогательная обёртка над FaceDetector согласно §4.1.3 ТЗ.
"""
from __future__ import annotations

import numpy as np

from deepfake_detector.preprocessing.face_detector import FaceDetector

__all__ = ["FaceAligner"]


class FaceAligner:
    """Выравнивает все T кадров видеопоследовательности.

    Для кадров, в которых лицо не найдено, подставляется копия последнего
    успешно выровненного кадра. Если ни один кадр не содержит лица,
    используются нулевые (чёрные) изображения.

    ``mask[i] = True``  — лицо в кадре i обнаружено и выровнено.
    ``mask[i] = False`` — кадр i заменён суррогатом.

    Args:
        detector: Инициализированный экземпляр :class:`FaceDetector`.
    """

    def __init__(self, detector: FaceDetector) -> None:
        self.detector = detector

    def __call__(self, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Выравнивает все кадры последовательности.

        Args:
            frames: (T, H, W, 3) uint8 — последовательность кадров.

        Returns:
            Кортеж ``(aligned_frames, mask)``:
                - aligned_frames: (T, 224, 224, 3) uint8.
                - mask: (T,) bool, True где лицо было найдено.
        """
        T = frames.shape[0]
        aligned = np.zeros((T, 224, 224, 3), dtype=np.uint8)
        mask = np.zeros(T, dtype=bool)
        last_valid: np.ndarray | None = None

        for i in range(T):
            result = self.detector(frames[i])
            if result is not None:
                aligned[i] = result
                mask[i] = True
                last_valid = result
            elif last_valid is not None:
                aligned[i] = last_valid
            # else: остаётся нулевым (чёрный кадр)

        return aligned, mask
