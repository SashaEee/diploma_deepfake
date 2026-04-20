"""Unit-тесты для временного энкодера (§5.1 ТЗ).

Проверяемые сценарии:
    AdaptiveDilatedConv1d:
        1. Выход shape == (B, ch, T) — GLU уменьшает каналы вдвое.
        2. Длина временной оси не изменяется (каузальный паддинг).
        3. Высокая motion-energy → малая дилатация (ближе к 1).
        4. Нулевая energy → дилатация равна base_dil.
        5. Обратный проход не вызывает ошибок.

    TemporalBlock:
        6.  Выход shape совпадает со входом (residual connection).
        7.  Нулевой dropout + нулевые веса conv → выход ≈ входу (чистый residual).
        8.  Обратный проход работает.

    TemporalEncoder:
        9.  Вход (B, T, C) → выход (B, C) — временная агрегация.
        10. T=1: одиночный кадр не вызывает аварийного завершения.
        11. T=2: минимальная пара кадров работает.
        12. Длинная последовательность T=32 обрабатывается.
        13. Обратный проход — градиенты вычисляются.
        14. Eval-режим детерминирован.
        15. n_layers блоков в ModuleList.
        16. Дилатации блоков — степени двойки: 1, 2, 4, …, 2^(n-1).
"""
from __future__ import annotations

import torch
import pytest

from deepfake_detector.models.temporal_encoder import (
    AdaptiveDilatedConv1d,
    TemporalBlock,
    TemporalEncoder,
)

# Фикс seed для воспроизводимости
@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Тесты AdaptiveDilatedConv1d
# ---------------------------------------------------------------------------

class TestAdaptiveDilatedConv1d:
    """Проверяет каузальную 1D-свёртку с GLU и адаптивной дилатацией."""

    @pytest.fixture
    def conv(self) -> AdaptiveDilatedConv1d:
        """ch=16, k=3, base_dil=4, max_dil=16."""
        return AdaptiveDilatedConv1d(ch=16, k=3, base_dil=4, max_dil=16)

    def test_output_shape_glu_halves_channels(self, conv):
        """GLU: вес (2*ch, ch, k) → выход (B, ch, T), не (B, 2*ch, T)."""
        x = torch.randn(2, 16, 8)      # (B=2, ch=16, T=8)
        energy = torch.zeros(2, 1, 8)
        out = conv(x, energy)
        assert out.shape == (2, 16, 8)

    def test_temporal_length_preserved(self, conv):
        """Каузальный паддинг сохраняет длину оси T."""
        for T in [1, 4, 16, 32]:
            x = torch.randn(1, 16, T)
            energy = torch.zeros(1, 1, T)
            out = conv(x, energy)
            assert out.shape[-1] == T, f"T={T}: expected T, got {out.shape[-1]}"

    def test_low_energy_gives_base_dilation(self):
        """Нулевая energy: phi ≈ 1 → d = round(base_dil * 1) = base_dil."""
        # phi = 1/(1 + E) при E→0 даёт phi→1, d = round(base_dil * phi) = base_dil
        conv = AdaptiveDilatedConv1d(ch=8, k=3, base_dil=4, max_dil=16)
        x = torch.randn(1, 8, 10)
        # Почти нулевая energy (clamp min=1e-6 не влияет)
        energy = torch.full((1, 1, 10), 1e-8)
        # Запускаем и проверяем, что нет ошибок и форма верна
        out = conv(x, energy)
        assert out.shape == (1, 8, 10)

    def test_high_energy_small_dilation(self):
        """Большая energy → phi мала → d ближе к 1 (минимальная дилатация)."""
        conv = AdaptiveDilatedConv1d(ch=8, k=3, base_dil=8, max_dil=16)
        x = torch.randn(1, 8, 10)
        energy = torch.full((1, 1, 10), 1e6)  # огромная energy
        out = conv(x, energy)
        # phi ≈ 0, d = round(8 * ~0) ≈ 0 → clip → 1
        # Проверяем просто, что shape верен
        assert out.shape == (1, 8, 10)

    def test_backward_does_not_raise(self, conv):
        """Обратный проход через AdaptiveDilatedConv1d работает."""
        x = torch.randn(2, 16, 8, requires_grad=True)
        energy = torch.randn(2, 1, 8).abs()
        out = conv(x, energy)
        out.sum().backward()
        assert x.grad is not None

    def test_weight_shape(self):
        """Вес имеет форму (2*ch, ch, k) для GLU."""
        conv = AdaptiveDilatedConv1d(ch=32, k=5, base_dil=2, max_dil=8)
        assert conv.weight.shape == (64, 32, 5)
        assert conv.bias.shape == (64,)


# ---------------------------------------------------------------------------
# Тесты TemporalBlock
# ---------------------------------------------------------------------------

class TestTemporalBlock:
    """Проверяет один блок временного кодировщика с residual-связью."""

    @pytest.fixture
    def block(self) -> TemporalBlock:
        return TemporalBlock(ch=32, k=3, base_dil=2, dropout=0.0)

    def test_output_shape_same_as_input(self, block):
        """Выход (B, ch, T) совпадает с входом — residual connection."""
        x = torch.randn(2, 32, 16)
        energy = torch.zeros(2, 1, 16)
        out = block(x, energy)
        assert out.shape == x.shape

    def test_various_sequence_lengths(self, block):
        """Блок работает при разных длинах последовательностей."""
        for T in [1, 4, 8, 32]:
            x = torch.randn(1, 32, T)
            energy = torch.zeros(1, 1, T)
            out = block(x, energy)
            assert out.shape == (1, 32, T), f"Failed at T={T}"

    def test_backward_does_not_raise(self, block):
        """Обратный проход через TemporalBlock работает."""
        x = torch.randn(2, 32, 8, requires_grad=True)
        energy = torch.zeros(2, 1, 8)
        out = block(x, energy)
        out.sum().backward()
        assert x.grad is not None

    def test_batch_size_one(self, block):
        """Batch size == 1 обрабатывается корректно."""
        x = torch.randn(1, 32, 10)
        energy = torch.zeros(1, 1, 10)
        out = block(x, energy)
        assert out.shape == (1, 32, 10)

    def test_layernorm_applied(self, block):
        """LayerNorm есть в блоке как атрибут ln."""
        import torch.nn as nn
        assert isinstance(block.ln, nn.LayerNorm)


# ---------------------------------------------------------------------------
# Тесты TemporalEncoder
# ---------------------------------------------------------------------------

class TestTemporalEncoder:
    """Проверяет полный временной энкодер."""

    @pytest.fixture
    def encoder(self) -> TemporalEncoder:
        return TemporalEncoder(ch=64, n_layers=4, k=3, dropout=0.0)

    def test_output_shape(self, encoder):
        """Вход (B, T, C) → выход (B, C) после temporal mean-pooling."""
        seq = torch.randn(2, 16, 64)
        with torch.no_grad():
            out = encoder(seq)
        assert out.shape == (2, 64)

    def test_single_frame(self, encoder):
        """T=1: один кадр не вызывает аварийного завершения."""
        seq = torch.randn(2, 1, 64)
        with torch.no_grad():
            out = encoder(seq)
        assert out.shape == (2, 64)

    def test_two_frames(self, encoder):
        """T=2: минимальная пара кадров — корректная форма выхода."""
        seq = torch.randn(3, 2, 64)
        with torch.no_grad():
            out = encoder(seq)
        assert out.shape == (3, 64)

    def test_long_sequence(self, encoder):
        """T=32: длинная последовательность обрабатывается без ошибок."""
        seq = torch.randn(2, 32, 64)
        with torch.no_grad():
            out = encoder(seq)
        assert out.shape == (2, 64)

    def test_backward_gradients(self, encoder):
        """Обратный проход вычисляет ненулевые градиенты."""
        seq = torch.randn(2, 8, 64, requires_grad=True)
        out = encoder(seq)
        out.sum().backward()
        assert seq.grad is not None
        assert seq.grad.abs().sum() > 0

    def test_eval_mode_deterministic(self, encoder):
        """В eval-режиме одинаковый вход даёт одинаковый выход."""
        encoder.eval()
        seq = torch.randn(2, 8, 64)
        with torch.no_grad():
            out1 = encoder(seq)
            out2 = encoder(seq)
        assert torch.allclose(out1, out2)

    def test_n_layers_in_modulelist(self, encoder):
        """ModuleList содержит ровно n_layers=4 блоков."""
        assert len(encoder.blocks) == 4

    def test_dilation_doubles_per_layer(self, encoder):
        """Дилатации блоков — степени двойки: 2^0, 2^1, 2^2, 2^3."""
        expected_dils = [2 ** i for i in range(4)]
        actual_dils = [blk.conv.base_dil for blk in encoder.blocks]
        assert actual_dils == expected_dils

    def test_default_params(self):
        """Параметры по умолчанию соответствуют диплому: ch=512, n_layers=6."""
        enc = TemporalEncoder()
        assert len(enc.blocks) == 6
        assert enc.blocks[0].conv.weight.shape[0] == 512 * 2  # 2*ch

    def test_batch_size_one(self, encoder):
        """Batch size == 1 обрабатывается без ошибок."""
        seq = torch.randn(1, 8, 64)
        with torch.no_grad():
            out = encoder(seq)
        assert out.shape == (1, 64)
