"""Потоковое декодирование видео с равномерным семплированием T кадров.

Листинг 5.2 диплома (Зенковский А. М., ЮФУ).

Контракт:
    Вход:  path — путь к видеофайлу форматов MP4, MOV, AVI, WebM.
    Выход: np.ndarray shape (T, H, W, 3), dtype=uint8, RGB-порядок.
    Ошибка: VideoProcessingError при повреждённом или нечитаемом контейнере.
"""
from __future__ import annotations

import av
import numpy as np
from pathlib import Path

__all__ = ["VideoLoader", "VideoProcessingError"]


class VideoProcessingError(Exception):
    """Выбрасывается при невозможности декодировать видеофайл."""


class VideoLoader:
    """Потоковое декодирование видео с равномерным семплированием T кадров.

    Алгоритм:
        1. Открыть контейнер через PyAV.
        2. Определить общее число кадров ``total``.
        3. Сформировать множество индексов для равномерного семплирования.
        4. Пройтись по пакетам, декодировать кадры с нужными индексами.
        5. Дополнить последовательность до T повторением последнего кадра.

    Args:
        target_frames: Желаемое число семплируемых кадров T (по умолчанию 16).
        max_side:       Максимальный размер наибольшей стороны при ресайзе (пкс).
    """

    def __init__(self, target_frames: int = 16, max_side: int = 640) -> None:
        if target_frames < 1:
            raise ValueError(f"target_frames must be >= 1, got {target_frames}")
        self.T = target_frames
        self.max_side = max_side

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def load(self, path: Path) -> np.ndarray:
        """Загружает и возвращает T кадров из видеофайла.

        Args:
            path: Путь к видеофайлу.

        Returns:
            np.ndarray shape ``(T, H, W, 3)``, dtype ``uint8``, RGB.

        Raises:
            VideoProcessingError: Если файл битый или не поддерживается PyAV.
        """
        try:
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                total = stream.frames or 0
                idx_set = self._select_indices(total)
                frames: list[np.ndarray] = []
                i = 0
                done = False
                for packet in container.demux(stream):
                    for frame in packet.decode():
                        # Если total неизвестен (idx_set пуст) — берём кадры подряд.
                        if not idx_set or i in idx_set:
                            img = frame.to_ndarray(format="rgb24")
                            frames.append(self._resize(img))
                        i += 1
                        if len(frames) == self.T:
                            done = True
                            break
                    if done:
                        break
        except av.FFmpegError as exc:
            raise VideoProcessingError(
                f"Cannot decode video '{path}': {exc}"
            ) from exc
        return self._pad(frames)

    # ------------------------------------------------------------------
    # Вспомогательные методы (доступны для unit-тестирования)
    # ------------------------------------------------------------------

    def _select_indices(self, total: int) -> set[int]:
        """Равномерно выбирает T индексов из диапазона [0, total).

        Если ``total == 0`` (неизвестная длина) или ``total <= T``,
        возвращает соответственно пустое множество или все индексы.
        """
        if total <= 0:
            return set()  # сигнал «собирать кадры подряд»
        if total <= self.T:
            return set(range(total))
        step = total / self.T
        return {int(step * k) for k in range(self.T)}

    def _resize(self, img: np.ndarray) -> np.ndarray:
        """Масштабирует кадр так, чтобы наибольшая сторона ≤ max_side."""
        h, w = img.shape[:2]
        k = self.max_side / max(h, w)
        if k >= 1.0:
            return img
        import cv2
        new_w, new_h = int(w * k), int(h * k)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _pad(self, frames: list[np.ndarray]) -> np.ndarray:
        """Дополняет список кадров до T, повторяя последний (или нули)."""
        while len(frames) < self.T:
            if frames:
                frames.append(frames[-1].copy())
            else:
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        return np.stack(frames[:self.T], axis=0)
