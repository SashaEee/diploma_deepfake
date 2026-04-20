"""Unit-тесты для пространственного энкодера (§5.1 ТЗ).

Проверяемые сценарии:
    SEBlock:
        1. Форма выхода совпадает с формой входа.
        2. Нулевой вход → нулевой выход (x * sigmoid(w) = 0).
        3. Обратный проход вычисляет ненулевые градиенты.
        4. Работает с различными размерами каналов.

    FPN:
        5. Число выходных тензоров равно числу входных признаков.
        6. Все выходные карты имеют ровно out_ch каналов.
        7. Пространственные размеры выходов совпадают с входными.
        8. Один уровень признаков — нет аварийного завершения.
        9. Два уровня: top-down слияние корректно.

    SpatialEncoder:
        10. Выход shape == (B, embed_dim=512).
        11. Выход shape == (B, embed_dim=256) при кастомном параметре.
        12. Batch size == 1 обрабатывается корректно.
        13. Обратный проход — градиенты достигают proj-слоя.
        14. Eval-режим детерминирован.
        15. Три уровня FPN и SE-блоков (архитектурная инвариантность).
        16. proj-слой отображает 128×3 → embed_dim.

Примечание: timm.create_model патчится флагом pretrained=False,
чтобы не скачивать веса при тестировании.
"""
from __future__ import annotations

import timm
import torch
import pytest

from deepfake_detector.models.spatial_encoder import SEBlock, FPN, SpatialEncoder


# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def no_pretrained(monkeypatch):
    """Форсирует pretrained=False при создании backbone, исключая сетевые запросы."""
    orig = timm.create_model
    monkeypatch.setattr(
        timm,
        "create_model",
        lambda *args, **kwargs: orig(*args, **{**kwargs, "pretrained": False}),
    )


@pytest.fixture
def se_block() -> SEBlock:
    return SEBlock(channels=64, r=8)


@pytest.fixture
def fpn_3levels() -> FPN:
    """FPN с тремя входными уровнями как у EfficientNet-B0 out_indices=(2,3,4)."""
    return FPN(in_channels=[40, 80, 112], out_ch=128)


@pytest.fixture
def spatial_encoder() -> SpatialEncoder:
    return SpatialEncoder(embed_dim=512)


# ---------------------------------------------------------------------------
# Тесты SEBlock
# ---------------------------------------------------------------------------

class TestSEBlock:
    """Проверяет Squeeze-and-Excitation блок."""

    def test_output_shape_preserved(self, se_block):
        """Форма выхода совпадает с формой входа (B, C, H, W)."""
        x = torch.randn(4, 64, 16, 16)
        out = se_block(x)
        assert out.shape == x.shape

    def test_zero_input_gives_zero_output(self, se_block):
        """Нулевой вход → нулевой выход: x * sigmoid(...) = 0."""
        x = torch.zeros(2, 64, 8, 8)
        out = se_block(x)
        assert (out == 0).all()

    def test_positive_input_stays_nonnegative(self):
        """Sigmoid-веса ∈ [0,1]: положительный вход остаётся неотрицательным."""
        se = SEBlock(channels=16, r=4)
        x = torch.ones(2, 16, 8, 8)
        out = se(x)
        assert (out >= 0).all()

    def test_different_channel_sizes(self):
        """SEBlock работает с произвольным числом каналов (r делит ch без остатка)."""
        for ch in [16, 32, 64, 128]:
            se = SEBlock(channels=ch, r=4)
            x = torch.randn(1, ch, 4, 4)
            out = se(x)
            assert out.shape == x.shape, f"Shape mismatch at ch={ch}"

    def test_batch_size_one(self, se_block):
        """Batch size == 1 не вызывает ошибок."""
        x = torch.randn(1, 64, 8, 8)
        out = se_block(x)
        assert out.shape == (1, 64, 8, 8)

    def test_backward_propagates_gradients(self, se_block):
        """Обратный проход вычисляет ненулевые градиенты для входа."""
        x = torch.randn(2, 64, 8, 8, requires_grad=True)
        out = se_block(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Тесты FPN
# ---------------------------------------------------------------------------

class TestFPN:
    """Проверяет Feature Pyramid Network."""

    def test_output_length_matches_input(self, fpn_3levels):
        """Число выходных тензоров совпадает с числом входных признаков."""
        feats = [
            torch.randn(2, 40, 28, 28),
            torch.randn(2, 80, 14, 14),
            torch.randn(2, 112, 7, 7),
        ]
        outs = fpn_3levels(feats)
        assert len(outs) == 3

    def test_output_channels_correct(self, fpn_3levels):
        """Все выходные карты имеют ровно out_ch=128 каналов."""
        feats = [
            torch.randn(2, 40, 28, 28),
            torch.randn(2, 80, 14, 14),
            torch.randn(2, 112, 7, 7),
        ]
        outs = fpn_3levels(feats)
        for i, o in enumerate(outs):
            assert o.shape[1] == 128, f"Level {i}: expected 128 channels, got {o.shape[1]}"

    def test_spatial_sizes_match_inputs(self, fpn_3levels):
        """Пространственные размеры выходов совпадают с соответствующими входными."""
        sizes = [(28, 28), (14, 14), (7, 7)]
        feats = [
            torch.randn(2, 40, 28, 28),
            torch.randn(2, 80, 14, 14),
            torch.randn(2, 112, 7, 7),
        ]
        outs = fpn_3levels(feats)
        for i, (o, (h, w)) in enumerate(zip(outs, sizes)):
            assert o.shape[-2:] == (h, w), (
                f"Level {i}: expected {(h, w)}, got {o.shape[-2:]}"
            )

    def test_single_feature_level(self):
        """FPN с одним уровнем признаков не аварийно завершается."""
        fpn = FPN(in_channels=[64], out_ch=32)
        feats = [torch.randn(2, 64, 14, 14)]
        outs = fpn(feats)
        assert len(outs) == 1
        assert outs[0].shape == (2, 32, 14, 14)

    def test_two_feature_levels(self):
        """FPN с двумя уровнями: top-down слияние корректно."""
        fpn = FPN(in_channels=[32, 64], out_ch=16)
        feats = [
            torch.randn(1, 32, 8, 8),
            torch.randn(1, 64, 4, 4),
        ]
        outs = fpn(feats)
        assert len(outs) == 2
        for o in outs:
            assert o.shape[1] == 16

    def test_backward_propagates_gradients(self, fpn_3levels):
        """Обратный проход через FPN вычисляет градиенты для всех уровней."""
        feats = [
            torch.randn(2, 40, 28, 28, requires_grad=True),
            torch.randn(2, 80, 14, 14, requires_grad=True),
            torch.randn(2, 112, 7, 7, requires_grad=True),
        ]
        outs = fpn_3levels(feats)
        sum(o.sum() for o in outs).backward()
        for i, f in enumerate(feats):
            assert f.grad is not None, f"No grad for level {i}"


# ---------------------------------------------------------------------------
# Тесты SpatialEncoder
# ---------------------------------------------------------------------------

class TestSpatialEncoder:
    """Проверяет полный пространственный энкодер (EfficientNet-B0 + FPN + SE + proj)."""

    def test_output_shape_default_embed_dim(self, spatial_encoder):
        """Выход имеет shape (B, 512) при embed_dim=512."""
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = spatial_encoder(x)
        assert out.shape == (2, 512)

    def test_output_shape_custom_embed_dim(self):
        """Выход имеет shape (B, 256) при embed_dim=256."""
        enc = SpatialEncoder(embed_dim=256)
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = enc(x)
        assert out.shape == (2, 256)

    def test_batch_size_one(self, spatial_encoder):
        """Batch size == 1 обрабатывается без ошибок."""
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = spatial_encoder(x)
        assert out.shape == (1, 512)

    def test_backward_gradients_flow(self, spatial_encoder):
        """Обратный проход работает; градиент достигает proj-слоя."""
        x = torch.randn(2, 3, 64, 64)
        out = spatial_encoder(x)
        out.sum().backward()
        assert spatial_encoder.proj.weight.grad is not None
        assert spatial_encoder.proj.weight.grad.abs().sum() > 0

    def test_eval_mode_is_deterministic(self, spatial_encoder):
        """В eval-режиме одинаковый вход всегда даёт одинаковый выход."""
        spatial_encoder.eval()
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            out1 = spatial_encoder(x)
            out2 = spatial_encoder(x)
        assert torch.allclose(out1, out2)

    def test_fpn_and_se_have_three_levels(self, spatial_encoder):
        """Архитектура: три уровня FPN и три SE-блока."""
        assert len(spatial_encoder.se) == 3
        assert len(spatial_encoder.fpn.lateral) == 3
        assert len(spatial_encoder.fpn.smooth) == 3

    def test_proj_dimensions(self, spatial_encoder):
        """proj: Linear(128×3 → embed_dim) — правильные размерности."""
        assert spatial_encoder.proj.in_features == 128 * 3
        assert spatial_encoder.proj.out_features == 512
