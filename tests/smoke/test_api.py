"""Smoke-тесты REST API (§5.3 ТЗ).

Проверяемые сценарии:
    GET /health:
        1. Возвращает 200 OK.
        2. Тело содержит {"status": "ok"}.
        3. uptime_s >= 0.
        4. model_version присутствует.

    POST /predict/image:
        5. JPEG-изображение → 200, поля probability/verdict/confidence.
        6. Пустой файл → 400 Bad Request.
        7. probability ∈ [0.0, 1.0].
        8. verdict ∈ {"real", "fake"}.

    POST /predict/video:
        9. MP4-файл → 200, поле job_id.
        10. estimated_ms присутствует.

    GET /predict/video/{job_id}:
        11. Несуществующий job_id → status == "pending".

Predictor патчится через dependency override, чтобы не требовать модели и GPU.
"""
from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image


# ---------------------------------------------------------------------------
# Мок Predictor
# ---------------------------------------------------------------------------

def _make_mock_predictor():
    """Предиктор, возвращающий фиксированный результат для всех методов."""
    predictor = MagicMock()
    predictor.predict_image.return_value = {
        "probability": 0.72,
        "verdict": "fake",
        "confidence": 0.85,
        "processing_ms": 123,
    }
    predictor.predict_video.return_value = {
        "probability": 0.30,
        "verdict": "real",
        "confidence": 0.90,
        "processing_ms": 456,
    }
    return predictor


def _make_jpeg_bytes(h: int = 64, w: int = 64) -> bytes:
    """Генерирует валидный JPEG в памяти."""
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """TestClient FastAPI с замоканным Predictor и Redis."""
    mock_predictor = _make_mock_predictor()

    # Патчим Predictor.from_config и глобальную переменную predictor
    with patch("deepfake_detector.api.main.Predictor") as MockPredictorCls, \
         patch("deepfake_detector.api.main.predictor", mock_predictor):
        MockPredictorCls.from_config.return_value = mock_predictor

        from deepfake_detector.api.main import app
        with TestClient(app, raise_server_exceptions=True) as c:
            # Подменяем predictor после startup
            import deepfake_detector.api.main as api_module
            api_module.predictor = mock_predictor
            yield c


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    """Проверяет эндпойнт здоровья сервиса."""

    def test_returns_200(self, client):
        """GET /health → 200 OK."""
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_status_is_ok(self, client):
        """Тело содержит {"status": "ok"}."""
        resp = client.get("/health")
        assert resp.json()["status"] == "ok"

    def test_uptime_nonnegative(self, client):
        """uptime_s >= 0."""
        resp = client.get("/health")
        assert resp.json()["uptime_s"] >= 0

    def test_model_version_present(self, client):
        """Поле model_version присутствует в ответе."""
        resp = client.get("/health")
        assert "model_version" in resp.json()


# ---------------------------------------------------------------------------
# POST /predict/image
# ---------------------------------------------------------------------------

class TestPredictImageEndpoint:
    """Проверяет синхронный эндпойнт предсказания по изображению."""

    def test_jpeg_returns_200(self, client):
        """Валидный JPEG → 200 OK."""
        resp = client.post(
            "/predict/image",
            files={"file": ("face.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 200

    def test_response_has_probability(self, client):
        """Ответ содержит поле probability."""
        resp = client.post(
            "/predict/image",
            files={"file": ("face.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        assert "probability" in resp.json()

    def test_probability_in_range(self, client):
        """probability ∈ [0.0, 1.0]."""
        resp = client.post(
            "/predict/image",
            files={"file": ("face.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        p = resp.json()["probability"]
        assert 0.0 <= p <= 1.0

    def test_verdict_is_real_or_fake(self, client):
        """verdict ∈ {'real', 'fake'}."""
        resp = client.post(
            "/predict/image",
            files={"file": ("face.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        assert resp.json()["verdict"] in ("real", "fake")

    def test_confidence_present(self, client):
        """Поле confidence присутствует."""
        resp = client.post(
            "/predict/image",
            files={"file": ("face.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        assert "confidence" in resp.json()

    def test_empty_file_returns_400(self, client):
        """Пустое тело (не-изображение) → 400 Bad Request."""
        resp = client.post(
            "/predict/image",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
        )
        assert resp.status_code == 400

    def test_corrupted_bytes_returns_400(self, client):
        """Случайные байты (не изображение) → 400."""
        resp = client.post(
            "/predict/image",
            files={"file": ("bad.jpg", b"\x00\xff\xde\xad", "image/jpeg")},
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# POST /predict/video
# ---------------------------------------------------------------------------

class TestPredictVideoEndpoint:
    """Проверяет асинхронный эндпойнт предсказания по видео."""

    @pytest.fixture(autouse=True)
    def _mock_celery(self):
        """Мокируем Celery task, чтобы не требовать Redis при тестировании."""
        with patch("deepfake_detector.api.tasks.process_video") as mock_task:
            mock_task.delay.return_value = MagicMock()
            yield mock_task

    def test_video_upload_returns_200(self, client):
        """POST /predict/video с любым файлом → 200 OK."""
        resp = client.post(
            "/predict/video",
            files={"file": ("video.mp4", b"fake_mp4_data", "video/mp4")},
        )
        assert resp.status_code == 200

    def test_response_has_job_id(self, client):
        """Ответ содержит поле job_id."""
        resp = client.post(
            "/predict/video",
            files={"file": ("video.mp4", b"fake_mp4_data", "video/mp4")},
        )
        assert "job_id" in resp.json()

    def test_job_id_is_string(self, client):
        """job_id — строка (UUID)."""
        resp = client.post(
            "/predict/video",
            files={"file": ("video.mp4", b"fake_mp4_data", "video/mp4")},
        )
        assert isinstance(resp.json()["job_id"], str)

    def test_estimated_ms_present(self, client):
        """Поле estimated_ms присутствует."""
        resp = client.post(
            "/predict/video",
            files={"file": ("video.mp4", b"fake_mp4_data", "video/mp4")},
        )
        assert "estimated_ms" in resp.json()


# ---------------------------------------------------------------------------
# GET /predict/video/{job_id}
# ---------------------------------------------------------------------------

class TestVideoStatusEndpoint:
    """Проверяет эндпойнт опроса статуса задачи."""

    @pytest.fixture(autouse=True)
    def _mock_redis(self):
        """Мокируем Redis.get() через точку импорта внутри функции (redis.Redis)."""
        with patch("redis.Redis") as MockRedis:
            instance = MagicMock()
            instance.get.return_value = None  # job не завершён → pending
            MockRedis.from_url.return_value = instance
            yield instance

    def test_unknown_job_returns_pending(self, client):
        """Несуществующий job_id → status == 'pending'."""
        resp = client.get("/predict/video/nonexistent-uuid")
        assert resp.status_code == 200
        assert resp.json()["status"] == "pending"

    def test_completed_job_returns_result(self, client, _mock_redis):
        """Завершённый job → status == 'done' с результатами."""
        _mock_redis.get.return_value = json.dumps({
            "job_id": "test-uuid",
            "status": "done",
            "probability": 0.91,
            "verdict": "fake",
            "processing_ms": 800,
        }).encode()
        resp = client.get("/predict/video/test-uuid")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "done"
        assert data["probability"] == pytest.approx(0.91)
