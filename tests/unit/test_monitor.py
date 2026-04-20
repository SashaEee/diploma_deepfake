"""Unit-тесты для DomainMonitor (§5.1 ТЗ).

Проверяемые сценарии:
    1. _sym_kl на одинаковых гистограммах равно 0.
    2. _sym_kl на разных гистограммах строго положительно.
    3. _sym_kl симметрично: KL(p, q) == KL(q, p).
    4. До заполнения буфера update() возвращает False.
    5. При совпадающих гистограммах drift не обнаруживается.
    6. При значительном сдвиге распределения возвращается True.
    7. Буфер ограничен окном window (maxlen).
    8. При превышении порога kl_thr → True, ниже → False.
"""
from __future__ import annotations

import numpy as np
import pytest

from deepfake_detector.adaptation.monitor import DomainMonitor


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _uniform_hist(n_bins: int = 10) -> np.ndarray:
    """Равномерная гистограмма: все бины одинаковы, сумма == 1."""
    h = np.ones(n_bins, dtype=np.float64)
    return h / h.sum()


def _make_monitor(
    ref_hist: np.ndarray | None = None,
    window: int = 5,
    kl_thr: float = 0.35,
) -> DomainMonitor:
    if ref_hist is None:
        ref_hist = _uniform_hist()
    return DomainMonitor({"hist": ref_hist}, window=window, kl_thr=kl_thr)


# ---------------------------------------------------------------------------
# Тесты _sym_kl
# ---------------------------------------------------------------------------

class TestSymKL:
    """Проверяет симметризованную дивергенцию Кульбака–Лейблера."""

    def test_identical_distributions_kl_is_zero(self):
        """KL(p, p) = 0."""
        p = np.array([0.2, 0.5, 0.3])
        kl = DomainMonitor._sym_kl(p, p)
        assert kl < 1e-9, f"Expected 0, got {kl}"

    def test_uniform_vs_uniform_kl_is_zero(self):
        """Две одинаковые равномерные гистограммы → KL = 0."""
        p = _uniform_hist(10)
        kl = DomainMonitor._sym_kl(p, p)
        assert kl < 1e-9

    def test_different_distributions_kl_is_positive(self):
        """Разные распределения → KL > 0."""
        p = np.array([0.9, 0.1])
        q = np.array([0.1, 0.9])
        kl = DomainMonitor._sym_kl(p, q)
        assert kl > 0.0

    def test_symmetry(self):
        """KL(p, q) == KL(q, p) (симметризованный вариант)."""
        p = np.array([0.3, 0.4, 0.3])
        q = np.array([0.1, 0.8, 0.1])
        assert abs(DomainMonitor._sym_kl(p, q) - DomainMonitor._sym_kl(q, p)) < 1e-12

    def test_degenerate_distribution(self):
        """Вырожденное распределение (один бин = 1, остальные = 0) → конечное значение."""
        p = np.array([1.0, 0.0, 0.0])
        q = np.array([0.0, 0.5, 0.5])
        kl = DomainMonitor._sym_kl(p, q)
        assert np.isfinite(kl)
        assert kl > 0.0

    def test_kl_uses_eps_for_numerical_stability(self):
        """Нулевые бины не вызывают log(0) → нет inf или nan."""
        p = np.array([0.0, 0.0, 1.0])
        q = np.array([1.0, 0.0, 0.0])
        kl = DomainMonitor._sym_kl(p, q)
        assert np.isfinite(kl)


# ---------------------------------------------------------------------------
# Тесты DomainMonitor.update
# ---------------------------------------------------------------------------

class TestDomainMonitorUpdate:
    """Проверяет логику обнаружения дрейфа."""

    def test_returns_false_before_buffer_full(self):
        """До заполнения буфера update() всегда возвращает False."""
        monitor = _make_monitor(window=5)
        hist = _uniform_hist()
        for _ in range(4):
            result = monitor.update(hist)
            assert result is False

    def test_no_drift_on_identical_histograms(self):
        """Идентичные гистограммы → дрейф не обнаруживается (False)."""
        ref = _uniform_hist()
        monitor = _make_monitor(ref_hist=ref.copy(), window=3, kl_thr=0.35)
        for _ in range(3):
            monitor.update(ref.copy())
        result = monitor.update(ref.copy())
        assert result is False

    def test_drift_detected_on_shifted_distribution(self):
        """Значительно сдвинутое распределение → дрейф обнаруживается (True)."""
        ref = _uniform_hist(10)  # равномерное
        monitor = _make_monitor(ref_hist=ref.copy(), window=3, kl_thr=0.1)

        # Сдвинутое: всё в первый бин
        drifted = np.zeros(10, dtype=np.float64)
        drifted[0] = 1.0

        for _ in range(3):
            result = monitor.update(drifted)

        assert result is True

    def test_low_kl_no_drift(self):
        """KL ниже порога → False."""
        ref = np.array([0.5, 0.5])
        monitor = _make_monitor(ref_hist=ref.copy(), window=3, kl_thr=10.0)  # очень высокий порог
        slightly_different = np.array([0.49, 0.51])
        for _ in range(3):
            result = monitor.update(slightly_different)
        assert result is False

    def test_buffer_maxlen_respected(self):
        """Буфер не превышает window элементов."""
        monitor = _make_monitor(window=3)
        hist = _uniform_hist()
        for _ in range(10):
            monitor.update(hist)
        assert len(monitor.buf) == 3

    def test_update_returns_bool(self):
        """update() возвращает bool (Python-тип, не np.bool_)."""
        monitor = _make_monitor(window=2)
        hist = _uniform_hist()
        monitor.update(hist)
        result = monitor.update(hist)
        assert isinstance(result, bool)

    def test_drift_threshold_boundary(self):
        """Проверяет поведение точно на границе порога."""
        ref = np.array([0.5, 0.5])
        # Подбираем гистограмму с KL ровно чуть выше kl_thr
        slightly_different = np.array([0.15, 0.85])
        kl_val = DomainMonitor._sym_kl(slightly_different, ref)

        monitor_strict = _make_monitor(
            ref_hist=ref.copy(),
            window=3,
            kl_thr=kl_val - 0.001,  # порог чуть ниже реального KL → должно сработать
        )
        monitor_loose = _make_monitor(
            ref_hist=ref.copy(),
            window=3,
            kl_thr=kl_val + 0.001,  # порог чуть выше → не должно сработать
        )

        for _ in range(3):
            r_strict = monitor_strict.update(slightly_different)
            r_loose = monitor_loose.update(slightly_different)

        assert r_strict is True
        assert r_loose is False


# ---------------------------------------------------------------------------
# Тесты конструктора DomainMonitor
# ---------------------------------------------------------------------------

class TestDomainMonitorInit:
    """Проверяет корректность инициализации."""

    def test_window_sets_deque_maxlen(self):
        """Параметр window задаёт maxlen буфера."""
        monitor = DomainMonitor({"hist": _uniform_hist()}, window=42)
        assert monitor.buf.maxlen == 42

    def test_kl_threshold_stored(self):
        """Параметр kl_thr сохраняется в атрибут."""
        monitor = DomainMonitor({"hist": _uniform_hist()}, kl_thr=0.5)
        assert monitor.kl_thr == 0.5

    def test_ref_stats_stored(self):
        """Опорная статистика сохраняется в атрибут ref."""
        ref_hist = _uniform_hist(5)
        monitor = DomainMonitor({"hist": ref_hist})
        np.testing.assert_array_equal(monitor.ref["hist"], ref_hist)
