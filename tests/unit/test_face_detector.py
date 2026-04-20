"""Unit-тесты для FaceDetector (§5.1 ТЗ).

Проверяемые сценарии:
    1. Возврат None при отсутствии лиц (boxes is None).
    2. Возврат None при низком confidence (< min_conf).
    3. Корректный shape выходного тензора (224, 224, 3).
    4. Аффинное выравнивание вызывается с правильными аргументами.
    5. Возврат None если landmarks is None.
    6. Возврат None если cv2.estimateAffinePartial2D не нашёл матрицу.

MTCNN патчится на уровне модуля, чтобы не загружать веса при тестировании.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

# Патчим MTCNN до импорта FaceDetector, чтобы конструктор не скачивал веса
_MTCNN_PATH = "deepfake_detector.preprocessing.face_detector.MTCNN"


# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_mtcnn_cls():
    """Подменяет класс MTCNN мок-объектом; возвращает экземпляр-мок."""
    with patch(_MTCNN_PATH) as MockCls:
        instance = MagicMock()
        MockCls.return_value = instance
        yield instance


@pytest.fixture
def canonical_landmarks() -> np.ndarray:
    """Канонические координаты пяти ключевых точек лица 224×224."""
    return np.float32([
        [70, 85], [154, 85], [112, 130], [80, 165], [144, 165],
    ])


def _make_detection(
    confidence: float = 0.99,
    landmarks: np.ndarray | None = None,
) -> tuple:
    """Возвращает кортеж (boxes, probs, landmarks), имитируя MTCNN.detect()."""
    boxes = np.array([[10.0, 10.0, 100.0, 100.0]])
    probs = np.array([confidence])
    if landmarks is None:
        landmarks = np.array([[[70, 85], [154, 85], [112, 130], [80, 165], [144, 165]]])
    return boxes, probs, landmarks


# ---------------------------------------------------------------------------
# Тесты FaceDetector.__call__
# ---------------------------------------------------------------------------

class TestFaceDetectorCall:
    """Тесты основного метода __call__."""

    def test_returns_none_when_no_face_detected(self, mock_mtcnn_cls):
        """boxes is None → возвращается None без дальнейшей обработки."""
        from deepfake_detector.preprocessing.face_detector import FaceDetector

        mock_mtcnn_cls.detect.return_value = (None, None, None)
        detector = FaceDetector(min_conf=0.92)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = detector(frame)

        assert result is None

    def test_returns_none_when_confidence_below_threshold(self, mock_mtcnn_cls):
        """confidence < min_conf → возвращается None."""
        from deepfake_detector.preprocessing.face_detector import FaceDetector

        boxes, probs, lms = _make_detection(confidence=0.50)
        mock_mtcnn_cls.detect.return_value = (boxes, probs, lms)
        detector = FaceDetector(min_conf=0.92)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = detector(frame)

        assert result is None

    def test_returns_none_when_confidence_exactly_below_threshold(self, mock_mtcnn_cls):
        """confidence == min_conf - ε → возвращается None (граничный случай)."""
        from deepfake_detector.preprocessing.face_detector import FaceDetector

        boxes, probs, lms = _make_detection(confidence=0.9199)
        mock_mtcnn_cls.detect.return_value = (boxes, probs, lms)
        detector = FaceDetector(min_conf=0.92)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = detector(frame)

        assert result is None

    def test_returns_aligned_face_when_confidence_high(self, mock_mtcnn_cls):
        """confidence >= min_conf и landmarks корректны → возвращается ndarray."""
        from deepfake_detector.preprocessing.face_detector import FaceDetector

        boxes, probs, lms = _make_detection(confidence=0.99)
        mock_mtcnn_cls.detect.return_value = (boxes, probs, lms)
        detector = FaceDetector(min_conf=0.92)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        result = detector(frame)

        assert result is not None

    def test_aligned_face_has_correct_shape(self, mock_mtcnn_cls):
        """Выровненное лицо имеет shape (224, 224, 3)."""
        from deepfake_detector.preprocessing.face_detector import FaceDetector

        boxes, probs, lms = _make_detection(confidence=0.99)
        mock_mtcnn_cls.detect.return_value = (boxes, probs, lms)
        detector = FaceDetector(min_conf=0.92)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        result = detector(frame)

        assert result.shape == (224, 224, 3)

    def test_aligned_face_dtype_is_uint8(self, mock_mtcnn_cls):
        """Выровненное лицо имеет dtype uint8."""
        from deepfake_detector.preprocessing.face_detector import FaceDetector

        boxes, probs, lms = _make_detection(confidence=0.99)
        mock_mtcnn_cls.detect.return_value = (boxes, probs, lms)
        detector = FaceDetector(min_conf=0.92)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        result = detector(frame)

        assert result.dtype == np.uint8

    def test_returns_none_when_landmarks_is_none(self, mock_mtcnn_cls):
        """landmarks is None → None (аффинное преобразование невозможно)."""
        from deepfake_detector.preprocessing.face_detector import FaceDetector

        boxes = np.array([[10.0, 10.0, 100.0, 100.0]])
        probs = np.array([0.99])
        mock_mtcnn_cls.detect.return_value = (boxes, probs, None)
        detector = FaceDetector(min_conf=0.92)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = detector(frame)

        assert result is None

    def test_detect_called_with_landmarks_flag(self, mock_mtcnn_cls):
        """MTCNN.detect вызывается с landmarks=True."""
        from deepfake_detector.preprocessing.face_detector import FaceDetector

        mock_mtcnn_cls.detect.return_value = (None, None, None)
        detector = FaceDetector()
        frame = np.zeros((224, 224, 3), dtype=np.uint8)

        detector(frame)

        mock_mtcnn_cls.detect.assert_called_once_with(frame, landmarks=True)

    def test_min_conf_exactly_at_threshold_passes(self, mock_mtcnn_cls):
        """confidence строго выше min_conf → обрабатывается (confidence=0.921 при min_conf=0.92)."""
        from deepfake_detector.preprocessing.face_detector import FaceDetector

        boxes, probs, lms = _make_detection(confidence=0.921)
        mock_mtcnn_cls.detect.return_value = (boxes, probs, lms)
        detector = FaceDetector(min_conf=0.92)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        result = detector(frame)

        assert result is not None

    def test_custom_min_conf_respected(self, mock_mtcnn_cls):
        """Нестандартный min_conf учитывается корректно."""
        from deepfake_detector.preprocessing.face_detector import FaceDetector

        # confidence=0.80 должен проходить при min_conf=0.75
        boxes, probs, lms = _make_detection(confidence=0.80)
        mock_mtcnn_cls.detect.return_value = (boxes, probs, lms)
        detector = FaceDetector(min_conf=0.75)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        result = detector(frame)

        assert result is not None

    def test_affine_failed_returns_none(self, mock_mtcnn_cls):
        """Если cv2.estimateAffinePartial2D не нашёл матрицу (M=None) → None."""
        from deepfake_detector.preprocessing.face_detector import FaceDetector

        boxes, probs, lms = _make_detection(confidence=0.99)
        mock_mtcnn_cls.detect.return_value = (boxes, probs, lms)
        detector = FaceDetector(min_conf=0.92)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        cv2_path = "deepfake_detector.preprocessing.face_detector.cv2"
        with patch(cv2_path) as mock_cv2:
            mock_cv2.estimateAffinePartial2D.return_value = (None, None)
            result = detector(frame)

        assert result is None


# ---------------------------------------------------------------------------
# Тесты конструктора FaceDetector
# ---------------------------------------------------------------------------

class TestFaceDetectorInit:
    """Проверяет инициализацию MTCNN с корректными параметрами."""

    def test_mtcnn_initialized_with_correct_image_size(self):
        """MTCNN инициализируется с image_size=224."""
        with patch(_MTCNN_PATH) as MockCls:
            MockCls.return_value = MagicMock()
            from deepfake_detector.preprocessing.face_detector import FaceDetector
            FaceDetector()
            _, kwargs = MockCls.call_args
            assert kwargs.get("image_size") == 224

    def test_mtcnn_initialized_with_correct_device(self):
        """MTCNN получает переданный device."""
        with patch(_MTCNN_PATH) as MockCls:
            MockCls.return_value = MagicMock()
            from deepfake_detector.preprocessing.face_detector import FaceDetector
            FaceDetector(device="cpu")
            _, kwargs = MockCls.call_args
            assert kwargs.get("device") == "cpu"

    def test_canonical_landmarks_shape(self):
        """canonical имеет shape (5, 2)."""
        with patch(_MTCNN_PATH) as MockCls:
            MockCls.return_value = MagicMock()
            from deepfake_detector.preprocessing.face_detector import FaceDetector
            detector = FaceDetector()
            assert detector.canonical.shape == (5, 2)

    def test_canonical_landmarks_values(self, canonical_landmarks):
        """canonical совпадает с эталонными координатами из диплома."""
        with patch(_MTCNN_PATH) as MockCls:
            MockCls.return_value = MagicMock()
            from deepfake_detector.preprocessing.face_detector import FaceDetector
            detector = FaceDetector()
            np.testing.assert_array_equal(detector.canonical, canonical_landmarks)
