"""Unit-тесты для функций потерь (§5.1 ТЗ).

Проверяемые сценарии:
    temporal_consistency_loss:
        1. На идентичных кадрах потеря близка к нулю.
        2. На случайных кадрах потеря строго положительна.
        3. Потеря дифференцируема (backward не вызывает ошибок).
        4. Возвращает скалярный тензор.

    CombinedLoss:
        5. Значение потерь положительно.
        6. Веса lambda_arc, lambda_tc, lambda_short влияют на результат.
        7. Корректно принимает произвольный shortcut_reg.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest

from deepfake_detector.training.losses import temporal_consistency_loss, CombinedLoss


# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def seed():
    """Фиксирует seed для воспроизводимости."""
    torch.manual_seed(42)


@pytest.fixture
def identical_embeddings() -> torch.Tensor:
    """(B=2, T=8, D=512) — все кадры одинаковы."""
    base = torch.randn(2, 1, 512)
    return base.expand(2, 8, 512).contiguous()


@pytest.fixture
def random_embeddings() -> torch.Tensor:
    """(B=2, T=8, D=512) — случайные некоррелированные эмбеддинги."""
    return torch.randn(2, 8, 512)


# ---------------------------------------------------------------------------
# Тесты temporal_consistency_loss
# ---------------------------------------------------------------------------

class TestTemporalConsistencyLoss:
    """Проверяет L_tc — регуляризатор временной согласованности."""

    def test_identical_frames_loss_is_zero(self, identical_embeddings):
        """L_tc на идентичных кадрах должна быть ≈ 0."""
        loss = temporal_consistency_loss(identical_embeddings)
        assert loss.item() < 1e-5, f"Expected ~0, got {loss.item()}"

    def test_random_frames_loss_is_positive(self, random_embeddings):
        """L_tc на случайных кадрах строго положительна."""
        loss = temporal_consistency_loss(random_embeddings)
        assert loss.item() > 0.0

    def test_returns_scalar_tensor(self, random_embeddings):
        """Функция возвращает скалярный тензор (dim == 0)."""
        loss = temporal_consistency_loss(random_embeddings)
        assert loss.dim() == 0

    def test_backward_does_not_raise(self, random_embeddings):
        """Обратный проход должен выполняться без ошибок."""
        emb = random_embeddings.requires_grad_(True)
        loss = temporal_consistency_loss(emb)
        loss.backward()  # не должно бросать исключений
        assert emb.grad is not None

    def test_loss_range_is_zero_to_two(self, random_embeddings):
        """L_tc = mean(1 - cosine_similarity) ∈ [0, 2]."""
        loss = temporal_consistency_loss(random_embeddings)
        assert 0.0 <= loss.item() <= 2.0

    def test_single_frame_sequence_raises_or_handles(self):
        """T=1: нет пар (T-1=0) → потеря должна быть 0 или nan (без краша)."""
        emb = torch.randn(2, 1, 512)
        # e[:, 1:] пуст → mean() на пустом тензоре даст nan;
        # проверяем, что функция не падает с исключением
        try:
            loss = temporal_consistency_loss(emb)
            assert isinstance(loss, torch.Tensor)
        except Exception as exc:
            pytest.fail(f"temporal_consistency_loss raised {exc} with T=1")

    def test_normalized_embeddings_identical_loss_zero(self):
        """L2-нормализованные одинаковые векторы → L_tc == 0."""
        v = torch.randn(512)
        v = F.normalize(v, dim=0)
        emb = v.unsqueeze(0).unsqueeze(0).expand(1, 4, 512).contiguous()
        loss = temporal_consistency_loss(emb)
        assert loss.item() < 1e-5

    def test_opposite_vectors_loss_is_two(self):
        """Противоположные векторы: cosine = -1 → (1 - (-1)) = 2."""
        v = torch.ones(512)
        emb = torch.stack([v, -v, v, -v], dim=0).unsqueeze(0)  # (1, 4, 512)
        loss = temporal_consistency_loss(emb)
        assert abs(loss.item() - 2.0) < 1e-4


# ---------------------------------------------------------------------------
# Тесты CombinedLoss
# ---------------------------------------------------------------------------

class TestCombinedLoss:
    """Проверяет комбинированную функцию потерь."""

    @pytest.fixture
    def dummy_batch(self):
        """Минимальный батч: logits (B=2, C=2), frame_emb (B=2, T=8, D=512)."""
        logits = torch.randn(2, 2)
        frame_emb = torch.randn(2, 8, 512)
        labels = torch.tensor([0, 1])
        shortcut_reg = torch.tensor(0.01)
        return logits, frame_emb, labels, shortcut_reg

    def test_loss_is_positive(self, dummy_batch):
        """Суммарная потеря строго положительна."""
        criterion = CombinedLoss()
        loss = criterion(*dummy_batch)
        assert loss.item() > 0.0

    def test_returns_scalar(self, dummy_batch):
        """CombinedLoss возвращает скаляр."""
        criterion = CombinedLoss()
        loss = criterion(*dummy_batch)
        assert loss.dim() == 0

    def test_backward_does_not_raise(self, dummy_batch):
        """Обратный проход через комбинированную потерю проходит без ошибок."""
        logits, frame_emb, labels, sreg = dummy_batch
        logits = logits.requires_grad_(True)
        frame_emb = frame_emb.requires_grad_(True)
        criterion = CombinedLoss()
        loss = criterion(logits, frame_emb, labels, sreg)
        loss.backward()
        assert logits.grad is not None

    def test_zero_tc_weight_disables_temporal_loss(self, dummy_batch):
        """При lam_tc=0 компонент L_tc не влияет на результат."""
        logits, frame_emb, labels, sreg = dummy_batch
        criterion_with = CombinedLoss(lam_arc=1.0, lam_tc=1.0, lam_short=0.0)
        criterion_without = CombinedLoss(lam_arc=1.0, lam_tc=0.0, lam_short=0.0)
        loss_with = criterion_with(logits, frame_emb, labels, sreg)
        loss_without = criterion_without(logits, frame_emb, labels, sreg)
        # С lam_tc=1 и lam_tc=0 значения должны различаться, если L_tc != 0
        tc_val = temporal_consistency_loss(frame_emb)
        if tc_val.item() > 1e-6:
            assert abs(loss_with.item() - loss_without.item()) > 1e-6

    def test_shortcut_reg_contributes(self, dummy_batch):
        """lam_short > 0 при sreg > 0 увеличивает суммарную потерю."""
        logits, frame_emb, labels, _ = dummy_batch
        sreg_zero = torch.tensor(0.0)
        sreg_nonzero = torch.tensor(1.0)
        criterion = CombinedLoss(lam_arc=1.0, lam_tc=0.0, lam_short=1.0)
        loss_zero = criterion(logits, frame_emb, labels, sreg_zero)
        loss_nonzero = criterion(logits, frame_emb, labels, sreg_nonzero)
        assert loss_nonzero.item() > loss_zero.item()

    def test_default_weights(self):
        """Веса по умолчанию соответствуют ТЗ: lam_arc=1.0, lam_tc=0.2, lam_short=0.1."""
        criterion = CombinedLoss()
        assert criterion.lam_arc == 1.0
        assert criterion.lam_tc == 0.2
        assert criterion.lam_short == 0.1
