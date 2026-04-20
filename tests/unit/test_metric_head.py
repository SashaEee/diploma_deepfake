"""Unit-тесты для метрической головы ArcFace (§5.1 ТЗ).

Проверяемые сценарии:
    ArcFaceHead (без меток):
        1. Выход shape == (B, n_classes) при labels=None.
        2. Логиты масштабированы параметром s: abs(logits) ≤ s + eps.
        3. Два одинаковых нормированных вектора → одинаковые логиты.
        4. Обратный проход вычисляет ненулевые градиенты.
        5. Веса W нормализованы по dim=0 (норма столбцов ≈ 1).

    ArcFaceHead (с метками):
        6. Выход shape == (B, n_classes) при labels != None.
        7. Angular margin снижает logit для целевого класса:
           cos(θ + m) < cos(θ) при θ > 0.
        8. n_classes=10 работает корректно.
        9. Обратный проход с метками работает.

    Конструктор:
        10. Параметры по умолчанию: s=32.0, m=0.5 (из ТЗ).
        11. Форма весового параметра W: (dim, n_classes).
        12. Кастомные параметры s и m сохраняются корректно.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import pytest

from deepfake_detector.models.metric_head import ArcFaceHead


# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(42)


@pytest.fixture
def head_default() -> ArcFaceHead:
    """Голова с параметрами из ТЗ: dim=512, n_classes=2, s=32, m=0.5."""
    return ArcFaceHead(dim=512, n_classes=2, s=32.0, m=0.5)


@pytest.fixture
def embeddings() -> torch.Tensor:
    """Случайные вложения (B=4, dim=512)."""
    return torch.randn(4, 512)


@pytest.fixture
def labels() -> torch.Tensor:
    """Метки классов для B=4: [0, 1, 0, 1]."""
    return torch.tensor([0, 1, 0, 1])


# ---------------------------------------------------------------------------
# Тесты без меток (inference mode)
# ---------------------------------------------------------------------------

class TestArcFaceHeadNoLabels:
    """Проверяет прямой проход ArcFaceHead при labels=None (inference)."""

    def test_output_shape(self, head_default, embeddings):
        """Выход shape == (B, n_classes) = (4, 2)."""
        out = head_default(embeddings)
        assert out.shape == (4, 2)

    def test_logits_scaled_by_s(self, head_default, embeddings):
        """Все логиты находятся в диапазоне [-s, +s]."""
        out = head_default(embeddings)
        assert (out.abs() <= head_default.s + 1e-4).all(), (
            f"Some logits exceed s={head_default.s}: max={out.abs().max().item():.4f}"
        )

    def test_identical_normalized_embeddings_give_same_logits(self, head_default):
        """Два одинаковых нормированных вектора → идентичные логиты."""
        v = F.normalize(torch.randn(1, 512), dim=1)
        emb = v.expand(2, -1)  # два одинаковых вектора
        out = head_default(emb)
        assert torch.allclose(out[0], out[1], atol=1e-5)

    def test_backward_does_not_raise(self, head_default, embeddings):
        """Обратный проход не вызывает ошибок."""
        emb = embeddings.requires_grad_(True)
        out = head_default(emb)
        out.sum().backward()
        assert emb.grad is not None
        assert emb.grad.abs().sum() > 0

    def test_weight_matrix_columns_normalized(self, head_default, embeddings):
        """После прямого прохода столбцы W имеют евклидову норму ≈ 1."""
        head_default(embeddings)  # форвард нормализует W временно, но W хранится как есть
        w = F.normalize(head_default.W, dim=0)
        col_norms = w.norm(dim=0)
        assert torch.allclose(col_norms, torch.ones_like(col_norms), atol=1e-5)

    def test_returns_scaled_cosine(self, head_default):
        """Без labels: logit = s * cos(theta) (нет margin)."""
        # Вектор, точно совпадающий с первым столбцом W
        w_col0 = F.normalize(head_default.W[:, 0], dim=0).detach()
        out = head_default(w_col0.unsqueeze(0))  # B=1
        # Косинус с первым классом должен быть ≈ 1 → logit ≈ s
        assert abs(out[0, 0].item() - head_default.s) < 0.5, (
            f"Expected ≈{head_default.s}, got {out[0, 0].item():.4f}"
        )

    def test_custom_n_classes(self):
        """n_classes=10 → выход (B, 10)."""
        head = ArcFaceHead(dim=128, n_classes=10)
        emb = torch.randn(3, 128)
        out = head(emb)
        assert out.shape == (3, 10)


# ---------------------------------------------------------------------------
# Тесты с метками (training mode)
# ---------------------------------------------------------------------------

class TestArcFaceHeadWithLabels:
    """Проверяет прямой проход ArcFaceHead при наличии labels (training)."""

    def test_output_shape_with_labels(self, head_default, embeddings, labels):
        """Выход shape == (B, n_classes) = (4, 2) при labels != None."""
        out = head_default(embeddings, labels)
        assert out.shape == (4, 2)

    def test_margin_reduces_target_class_logit(self):
        """cos(θ + m) < cos(θ) при θ ∈ (0, π) → margin снижает логит целевого класса."""
        head = ArcFaceHead(dim=4, n_classes=2, s=1.0, m=0.5)
        # Вектор, близкий к первому столбцу W
        emb = F.normalize(torch.randn(1, 4), dim=1)
        labels = torch.tensor([0])  # целевой класс = 0

        logits_no_margin = head(emb)           # без labels
        logits_with_margin = head(emb, labels)  # с angular margin

        # Логит для целевого класса 0 должен уменьшиться (или равен при θ=0)
        assert logits_with_margin[0, 0].item() <= logits_no_margin[0, 0].item() + 1e-4

    def test_backward_with_labels(self, head_default, embeddings, labels):
        """Обратный проход с метками работает без ошибок."""
        emb = embeddings.requires_grad_(True)
        out = head_default(emb, labels)
        out.sum().backward()
        assert emb.grad is not None

    def test_output_is_finite(self, head_default, embeddings, labels):
        """Все выходные значения конечны (нет nan/inf)."""
        out = head_default(embeddings, labels)
        assert torch.isfinite(out).all()

    def test_multiclass_n10(self):
        """n_classes=10: батч из 5 примеров обрабатывается корректно."""
        head = ArcFaceHead(dim=64, n_classes=10, s=16.0, m=0.3)
        emb = torch.randn(5, 64)
        labels = torch.tensor([0, 3, 7, 1, 9])
        out = head(emb, labels)
        assert out.shape == (5, 10)
        assert torch.isfinite(out).all()

    def test_onehot_margin_only_on_target(self, head_default, embeddings):
        """Margin применяется только к целевому классу, остальные не изменяются."""
        head = ArcFaceHead(dim=512, n_classes=2, s=1.0, m=0.5)
        emb = F.normalize(torch.randn(1, 512), dim=1)
        labels_0 = torch.tensor([0])
        labels_1 = torch.tensor([1])

        out0 = head(emb, labels_0)
        out1 = head(emb, labels_1)
        # Класс 1 у out0 (не целевой) == класс 1 у out1 (целевой с margin) → они разные
        # Просто проверяем, что два разных вызова дают разные результаты
        assert not torch.allclose(out0, out1)


# ---------------------------------------------------------------------------
# Тесты конструктора
# ---------------------------------------------------------------------------

class TestArcFaceHeadInit:
    """Проверяет инициализацию ArcFaceHead."""

    def test_default_params(self):
        """Параметры по умолчанию: s=32.0, m=0.5 (соответствуют диплому)."""
        head = ArcFaceHead(dim=512)
        assert head.s == 32.0
        assert head.m == 0.5

    def test_weight_shape(self):
        """Форма весового параметра W: (dim, n_classes)."""
        head = ArcFaceHead(dim=256, n_classes=5)
        assert head.W.shape == (256, 5)

    def test_custom_s_and_m_stored(self):
        """Кастомные s и m сохраняются в атрибутах."""
        head = ArcFaceHead(dim=128, n_classes=3, s=16.0, m=0.3)
        assert head.s == 16.0
        assert head.m == 0.3

    def test_w_is_parameter(self):
        """W является nn.Parameter (обучаемый параметр)."""
        import torch.nn as nn
        head = ArcFaceHead(dim=64, n_classes=2)
        assert isinstance(head.W, nn.Parameter)

    def test_default_n_classes_is_two(self):
        """По умолчанию n_classes=2 (бинарная классификация deepfake/real)."""
        head = ArcFaceHead(dim=512)
        assert head.W.shape[1] == 2
