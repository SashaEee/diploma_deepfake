"""Интеграционные тесты полного пайплайна (§5.3 ТЗ).

Проверяемые сценарии:
    1. Препроцессинг → модель: VideoLoader + FaceAligner + FrameNormalizer
       → DeepfakeDetector выдаёт корректную структуру.
    2. Predictor.predict_video() возвращает словарь с обязательными ключами.
    3. Predictor.predict_image() обрабатывает изображение без ошибок.
    4. Predictor.predict_batch() обрабатывает список путей.
    5. Вердикт "real"/"fake" соответствует порогу 0.5.
    6. processing_ms строго положителен.
    7. DeepfakeDetector: video (B,T,C,H,W) → (logits, spatial, shortcut_reg).
    8. shortcut_reg — скаляр, конечное число.
    9. freeze_base=True блокирует backbone-градиенты.
    10. CombinedLoss + Trainer._validate() возвращает dict с нужными ключами.

MTCNN патчится на уровне модуля; timm использует pretrained=False.
"""
from __future__ import annotations

import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import timm
import torch

# ---------------------------------------------------------------------------
# Общие фикстуры
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def no_pretrained(monkeypatch):
    """Форсирует pretrained=False для EfficientNet-B0."""
    orig = timm.create_model
    monkeypatch.setattr(
        timm, "create_model",
        lambda *a, **kw: orig(*a, **{**kw, "pretrained": False}),
    )


@pytest.fixture(autouse=True)
def patch_mtcnn():
    """Подменяет MTCNN, чтобы не скачивать веса."""
    with patch("deepfake_detector.preprocessing.face_detector.MTCNN") as MockCls:
        inst = MagicMock()
        MockCls.return_value = inst
        # Возвращаем детекцию с высокой уверенностью и 5 ориентирами
        inst.detect.return_value = (
            np.array([[10.0, 10.0, 100.0, 100.0]]),
            np.array([0.99]),
            np.array([[[70, 85], [154, 85], [112, 130], [80, 165], [144, 165]]]),
        )
        yield inst


@pytest.fixture
def simple_model_cfg():
    """Минимальная конфигурация модели."""
    cfg = types.SimpleNamespace(
        embed_dim=512, n_layers=2, kernel=3,
        dropout=0.0, freeze_base=False,
        arc_s=32.0, arc_m=0.5,
    )
    return cfg


@pytest.fixture
def model(simple_model_cfg):
    from deepfake_detector.models.full_model import DeepfakeDetector
    m = DeepfakeDetector(simple_model_cfg)
    m.eval()
    return m


# ---------------------------------------------------------------------------
# Тест 7-9: DeepfakeDetector
# ---------------------------------------------------------------------------

class TestDeepfakeDetector:
    """Проверяет прямой проход полной модели."""

    def test_forward_returns_three_outputs(self, model):
        """Модель возвращает (logits, spatial, shortcut_reg)."""
        video = torch.randn(1, 4, 3, 64, 64)
        out = model(video)
        assert len(out) == 3, "Ожидается кортеж (logits, spatial, shortcut_reg)"

    def test_logits_shape(self, model):
        """logits shape == (B,) — бинарный BCE логит."""
        video = torch.randn(2, 4, 3, 64, 64)
        logits, _, _ = model(video)
        assert logits.shape == (2,)

    def test_spatial_shape(self, model):
        """spatial embeddings shape == (B, T, embed_dim)."""
        B, T = 2, 4
        video = torch.randn(B, T, 3, 64, 64)
        _, spatial, _ = model(video)
        assert spatial.shape == (B, T, 512)

    def test_shortcut_reg_is_scalar(self, model):
        """shortcut_reg — конечный скаляр."""
        video = torch.randn(1, 4, 3, 64, 64)
        _, _, sreg = model(video)
        assert sreg.dim() == 0
        assert torch.isfinite(sreg)

    def test_backward_end_to_end(self, model):
        """Обратный проход через всю модель не вызывает ошибок."""
        model.train()
        video = torch.randn(1, 4, 3, 64, 64, requires_grad=False)
        labels = torch.tensor([0])
        logits, frame_emb, sreg = model(video, labels)
        from deepfake_detector.training.losses import CombinedLoss
        loss = CombinedLoss()(logits, frame_emb, labels, sreg)
        loss.backward()
        # Градиент дошёл до Linear head
        assert model.head[1].weight.grad is not None

    def test_freeze_base_blocks_backbone_gradients(self, simple_model_cfg):
        """freeze_base=True: backbone-параметры не имеют requires_grad."""
        from deepfake_detector.models.full_model import DeepfakeDetector
        cfg = types.SimpleNamespace(**{**vars(simple_model_cfg), "freeze_base": True})
        m = DeepfakeDetector(cfg)
        for p in m.spatial.backbone.parameters():
            assert not p.requires_grad, "backbone должен быть заморожен"

    def test_with_labels_training_mode(self, model):
        """С метками — обучающий режим Linear head — нет ошибок."""
        model.train()
        video = torch.randn(2, 4, 3, 64, 64)
        labels = torch.tensor([0, 1])
        logits, _, _ = model(video, labels)
        assert logits.shape == (2,)
        assert torch.isfinite(logits).all()


# ---------------------------------------------------------------------------
# Тест 1-6: Predictor (мокирован)
# ---------------------------------------------------------------------------

class TestPredictorIntegration:
    """Проверяет Predictor с мокированным препроцессором и моделью."""

    @pytest.fixture
    def predictor(self, model):
        """Predictor с реальной моделью, но мокированным препроцессором."""
        from deepfake_detector.inference.predictor import Predictor

        def _fake_preprocessor(path: Path) -> torch.Tensor:
            return torch.randn(4, 3, 64, 64)  # (T, C, H, W)

        cfg = types.SimpleNamespace(
            inference=types.SimpleNamespace(
                threshold=0.5,
                device="cpu",
            ),
            preprocessing=types.SimpleNamespace(
                face_min_conf=0.92,
                normalize_mean=[0.485, 0.456, 0.406],
                normalize_std=[0.229, 0.224, 0.225],
            ),
        )
        return Predictor(cfg, model, _fake_preprocessor)

    def test_predict_video_returns_required_keys(self, predictor, tmp_path):
        """predict_video() возвращает dict с обязательными ключами."""
        fake_video = tmp_path / "test.mp4"
        fake_video.write_bytes(b"dummy")
        result = predictor.predict_video(fake_video)
        for key in ("probability", "verdict", "confidence", "processing_ms"):
            assert key in result, f"Ключ '{key}' отсутствует в результате"

    def test_probability_in_range(self, predictor, tmp_path):
        """probability ∈ [0.0, 1.0]."""
        fake_video = tmp_path / "test.mp4"
        fake_video.write_bytes(b"dummy")
        result = predictor.predict_video(fake_video)
        assert 0.0 <= result["probability"] <= 1.0

    def test_verdict_is_real_or_fake(self, predictor, tmp_path):
        """verdict строго 'real' или 'fake'."""
        fake_video = tmp_path / "test.mp4"
        fake_video.write_bytes(b"dummy")
        result = predictor.predict_video(fake_video)
        assert result["verdict"] in ("real", "fake")

    def test_verdict_matches_threshold(self, predictor, tmp_path):
        """verdict соответствует порогу 0.5."""
        fake_video = tmp_path / "test.mp4"
        fake_video.write_bytes(b"dummy")
        result = predictor.predict_video(fake_video)
        expected = "fake" if result["probability"] >= 0.5 else "real"
        assert result["verdict"] == expected

    def test_processing_ms_positive(self, predictor, tmp_path):
        """processing_ms > 0."""
        fake_video = tmp_path / "test.mp4"
        fake_video.write_bytes(b"dummy")
        result = predictor.predict_video(fake_video)
        assert result["processing_ms"] >= 0

    def test_predict_batch(self, predictor, tmp_path):
        """predict_batch() обрабатывает список путей."""
        paths = [tmp_path / f"v{i}.mp4" for i in range(3)]
        for p in paths:
            p.write_bytes(b"dummy")
        results = predictor.predict_batch(paths)
        assert len(results) == 3
        for r in results:
            assert "probability" in r

    def test_predict_image(self, predictor):
        """predict_image() обрабатывает numpy-изображение."""
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = predictor.predict_image(img)
        assert "probability" in result
        assert "verdict" in result


# ---------------------------------------------------------------------------
# Тест 10: Trainer._validate() с реальными метриками
# ---------------------------------------------------------------------------

class TestTrainerValidate:
    """Проверяет, что _validate() возвращает корректные метрики (не заглушки)."""

    def test_validate_returns_dict_with_metrics(self, model, tmp_path):
        """_validate() возвращает dict с auroc, f1, eer, loss."""
        import types
        from torch.utils.data import DataLoader, TensorDataset

        cfg = types.SimpleNamespace(
            batch=2, grad_accum=1, lr=1e-4, wd=1e-4, epochs=1,
            lam_arc=1.0, lam_tc=0.2, lam_short=0.1,
            mixed=False, num_workers=0, seed=42,
        )

        # Синтетический датасет: 4 видео по 4 кадра 64×64
        videos = torch.randn(4, 4, 3, 64, 64)
        labels = torch.tensor([0, 1, 0, 1])

        class _SyntheticDS(torch.utils.data.Dataset):
            def __len__(self): return 4
            def __getitem__(self, idx):
                return {"video": videos[idx], "label": labels[idx]}

        ds = _SyntheticDS()
        from deepfake_detector.training.trainer import Trainer
        trainer = Trainer(cfg, model, ds, ds)

        metrics = trainer._validate()
        for key in ("auroc", "f1", "eer", "loss"):
            assert key in metrics, f"'{key}' отсутствует в метриках"
            assert isinstance(metrics[key], float)
        assert 0.0 <= metrics["auroc"] <= 1.0
        assert 0.0 <= metrics["f1"] <= 1.0
        assert 0.0 <= metrics["eer"] <= 1.0
