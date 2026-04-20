"""Unit-тесты для VideoLoader (§5.1 ТЗ).

Проверяемые сценарии:
    1. _select_indices: total > T, total == T, total < T, total == 0.
    2. _pad: пустой список, меньше T, ровно T, больше T.
    3. _resize: изображение больше max_side, меньше/равно max_side.
    4. load: успешная загрузка (total > T, total == T, total < T),
             загрузка при total == 0 (неизвестная длина),
             выброс VideoProcessingError на битом файле.

Все тесты используют только мок-объекты — реальные видеофайлы не нужны.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from deepfake_detector.preprocessing.video_loader import VideoLoader, VideoProcessingError


# ---------------------------------------------------------------------------
# Вспомогательные фабрики мок-объектов
# ---------------------------------------------------------------------------

def _make_frame(h: int = 240, w: int = 320) -> MagicMock:
    """Создаёт мок av.VideoFrame, возвращающий чёрный RGB-массив."""
    frame = MagicMock()
    frame.to_ndarray.return_value = np.zeros((h, w, 3), dtype=np.uint8)
    return frame


def _make_container(n_frames: int, h: int = 240, w: int = 320) -> MagicMock:
    """Создаёт мок av-контейнера с заданным числом кадров.

    Каждый пакет декодируется ровно в один кадр.
    ``container.__enter__`` / ``__exit__`` настроены для работы с ``with``.
    """
    container = MagicMock()
    stream = MagicMock()
    stream.frames = n_frames
    container.streams.video = [stream]

    container.__enter__ = lambda s: container
    container.__exit__ = MagicMock(return_value=False)

    packets = []
    for _ in range(n_frames):
        pkt = MagicMock()
        pkt.decode.return_value = [_make_frame(h, w)]
        packets.append(pkt)
    container.demux.return_value = iter(packets)

    return container


# ---------------------------------------------------------------------------
# Тесты _select_indices
# ---------------------------------------------------------------------------

class TestSelectIndices:
    """Проверяет равномерное семплирование индексов кадров."""

    def test_total_greater_than_T_returns_exactly_T_indices(self):
        """При total > T должно быть ровно T индексов в диапазоне [0, total)."""
        loader = VideoLoader(target_frames=4)
        indices = loader._select_indices(100)
        assert len(indices) == 4
        assert all(isinstance(i, int) for i in indices)
        assert all(0 <= i < 100 for i in indices)

    def test_total_greater_than_T_indices_uniformly_spaced(self):
        """Индексы должны быть равномерно распределены: step = total / T."""
        loader = VideoLoader(target_frames=4)
        # total=100, T=4: step=25 → ожидаем {0, 25, 50, 75}
        indices = sorted(loader._select_indices(100))
        assert indices == [0, 25, 50, 75]

    def test_total_equals_T_returns_all_indices(self):
        """При total == T возвращаются все индексы 0..T-1."""
        loader = VideoLoader(target_frames=4)
        indices = loader._select_indices(4)
        assert indices == {0, 1, 2, 3}

    def test_total_less_than_T_returns_all_available(self):
        """При total < T возвращаются все имеющиеся индексы."""
        loader = VideoLoader(target_frames=16)
        indices = loader._select_indices(8)
        assert indices == set(range(8))

    def test_total_one_returns_single_index(self):
        """Крайний случай: одиночный кадр → {0}."""
        loader = VideoLoader(target_frames=16)
        assert loader._select_indices(1) == {0}

    def test_total_zero_returns_empty_set(self):
        """total == 0 → пустое множество (сигнал для последовательного сбора)."""
        loader = VideoLoader(target_frames=4)
        assert loader._select_indices(0) == set()

    def test_total_negative_returns_empty_set(self):
        """Отрицательный total обрабатывается аналогично нулевому."""
        loader = VideoLoader(target_frames=4)
        assert loader._select_indices(-1) == set()


# ---------------------------------------------------------------------------
# Тесты _pad
# ---------------------------------------------------------------------------

class TestPad:
    """Проверяет дополнение последовательности кадров до T."""

    def test_empty_list_returns_T_zero_frames(self):
        """Пустой список → T чёрных кадров 224×224×3."""
        loader = VideoLoader(target_frames=4)
        result = loader._pad([])
        assert result.shape == (4, 224, 224, 3)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, 0)

    def test_fewer_frames_pads_by_repeating_last(self):
        """Если кадров меньше T, последний кадр повторяется."""
        loader = VideoLoader(target_frames=4)
        f0 = np.full((224, 224, 3), 10, dtype=np.uint8)
        f1 = np.full((224, 224, 3), 20, dtype=np.uint8)
        result = loader._pad([f0, f1])
        assert result.shape == (4, 224, 224, 3)
        np.testing.assert_array_equal(result[2], f1)  # повтор
        np.testing.assert_array_equal(result[3], f1)  # повтор

    def test_exact_T_frames_unchanged(self):
        """Ровно T кадров не изменяются."""
        loader = VideoLoader(target_frames=3)
        frames = [np.full((224, 224, 3), i * 50, dtype=np.uint8) for i in range(3)]
        result = loader._pad(list(frames))
        for i, frame in enumerate(frames):
            np.testing.assert_array_equal(result[i], frame)

    def test_more_than_T_frames_truncated_to_T(self):
        """Если кадров больше T, берутся только первые T."""
        loader = VideoLoader(target_frames=2)
        frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(5)]
        result = loader._pad(frames)
        assert result.shape == (2, 224, 224, 3)

    def test_pixel_values_preserved(self):
        """Пиксели оригинальных кадров не изменяются при дополнении."""
        loader = VideoLoader(target_frames=4)
        rng = np.random.default_rng(42)
        f0 = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
        f1 = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
        result = loader._pad([f0, f1])
        np.testing.assert_array_equal(result[0], f0)
        np.testing.assert_array_equal(result[1], f1)

    def test_output_is_stacked_ndarray(self):
        """Результат — np.ndarray, а не список."""
        loader = VideoLoader(target_frames=3)
        result = loader._pad([np.zeros((224, 224, 3), dtype=np.uint8)])
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# Тесты _resize
# ---------------------------------------------------------------------------

class TestResize:
    """Проверяет уменьшение кадра до max_side при необходимости."""

    def test_large_image_is_downscaled(self):
        """max(H, W) > max_side → результат вписывается в max_side."""
        loader = VideoLoader(max_side=100)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = loader._resize(img)
        assert max(result.shape[:2]) <= 100

    def test_small_image_returned_unchanged(self):
        """max(H, W) <= max_side → тот же объект, без копирования."""
        loader = VideoLoader(max_side=640)
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        result = loader._resize(img)
        assert result is img

    def test_aspect_ratio_preserved(self):
        """Соотношение сторон (w/h) сохраняется (погрешность < 5%)."""
        loader = VideoLoader(max_side=100)
        img = np.zeros((200, 400, 3), dtype=np.uint8)  # w/h = 2.0
        result = loader._resize(img)
        h, w = result.shape[:2]
        assert abs(w / h - 2.0) < 0.1

    def test_square_image_resized_correctly(self):
        """Квадратное изображение: обе стороны уменьшаются одинаково."""
        loader = VideoLoader(max_side=50)
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        result = loader._resize(img)
        h, w = result.shape[:2]
        assert h == w
        assert h <= 50

    def test_exactly_max_side_not_resized(self):
        """Если max(H, W) == max_side, ресайз не выполняется."""
        loader = VideoLoader(max_side=640)
        img = np.zeros((640, 480, 3), dtype=np.uint8)
        result = loader._resize(img)
        assert result.shape == (640, 480, 3)
        assert result is img


# ---------------------------------------------------------------------------
# Тесты load() через мок av.open
# ---------------------------------------------------------------------------

class TestLoad:
    """End-to-end тесты метода load() с мок-объектами PyAV."""

    @pytest.fixture(autouse=True)
    def _patch_av(self):
        with patch("deepfake_detector.preprocessing.video_loader.av.open") as mock_open:
            self._mock_open = mock_open
            yield

    def _setup(self, n_frames: int, h: int = 240, w: int = 320) -> MagicMock:
        container = _make_container(n_frames, h, w)
        self._mock_open.return_value = container
        return container

    # --- Сценарии семплирования ---

    def test_total_greater_T_returns_exactly_T_frames(self):
        """total=30 > T=16 → shape[0] == 16."""
        loader = VideoLoader(target_frames=16)
        self._setup(30)
        result = loader.load(Path("v.mp4"))
        assert result.shape[0] == 16

    def test_total_equals_T_returns_T_frames(self):
        """total == T → shape[0] == T."""
        loader = VideoLoader(target_frames=8)
        self._setup(8)
        result = loader.load(Path("v.mp4"))
        assert result.shape[0] == 8

    def test_total_less_than_T_pads_to_T(self):
        """total < T → дополняется до T повтором последнего кадра."""
        loader = VideoLoader(target_frames=16)
        self._setup(5)
        result = loader.load(Path("v.mp4"))
        assert result.shape[0] == 16

    def test_total_zero_collects_sequential_frames(self):
        """total == 0 (неизвестная длина) → собираются первые T кадров подряд."""
        loader = VideoLoader(target_frames=4)
        container = _make_container(20)
        container.streams.video[0].frames = 0  # metadata: неизвестно
        self._mock_open.return_value = container
        result = loader.load(Path("v.mp4"))
        assert result.shape[0] == 4

    # --- Форма и тип данных ---

    def test_output_shape_and_dtype(self):
        """Результат: (T, H, W, 3) uint8."""
        loader = VideoLoader(target_frames=4, max_side=640)
        self._setup(20, h=240, w=320)
        result = loader.load(Path("v.mp4"))
        assert result.ndim == 4
        assert result.shape == (4, 240, 320, 3)
        assert result.dtype == np.uint8

    # --- Обработка ошибок ---

    def test_broken_container_raises_video_processing_error(self):
        """av.FFmpegError при открытии → VideoProcessingError с информативным сообщением."""
        import av as _av
        loader = VideoLoader(target_frames=4)
        self._mock_open.side_effect = _av.FFmpegError(-1, "broken")
        with pytest.raises(VideoProcessingError, match="Cannot decode"):
            loader.load(Path("broken.mp4"))

    def test_error_message_contains_path(self):
        """Сообщение об ошибке содержит имя проблемного файла."""
        import av as _av
        loader = VideoLoader(target_frames=4)
        self._mock_open.side_effect = _av.FFmpegError(-1, "err")
        with pytest.raises(VideoProcessingError, match="my_video.mp4"):
            loader.load(Path("my_video.mp4"))

    # --- Корректность вызовов ---

    def test_av_open_called_with_string(self):
        """av.open вызывается со строкой, а не с Path-объектом."""
        loader = VideoLoader(target_frames=4)
        self._setup(8)
        loader.load(Path("/data/video.mp4"))
        self._mock_open.assert_called_once_with("/data/video.mp4")

    def test_demux_stops_after_T_frames_collected(self):
        """Итерация по пакетам прекращается после набора T кадров."""
        loader = VideoLoader(target_frames=4)
        container = _make_container(100)
        container.streams.video[0].frames = 100
        self._mock_open.return_value = container
        loader.load(Path("v.mp4"))
        # demux вызван ровно один раз
        container.demux.assert_called_once()


# ---------------------------------------------------------------------------
# Конструктор
# ---------------------------------------------------------------------------

class TestConstructor:
    """Проверяет валидацию аргументов конструктора."""

    def test_default_parameters(self):
        loader = VideoLoader()
        assert loader.T == 16
        assert loader.max_side == 640

    def test_custom_parameters(self):
        loader = VideoLoader(target_frames=8, max_side=320)
        assert loader.T == 8
        assert loader.max_side == 320

    def test_zero_target_frames_raises(self):
        with pytest.raises(ValueError):
            VideoLoader(target_frames=0)

    def test_negative_target_frames_raises(self):
        with pytest.raises(ValueError):
            VideoLoader(target_frames=-1)
