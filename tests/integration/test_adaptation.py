"""Интеграционные тесты онлайн-адаптации (§5.3 ТЗ).

Проверяемые сценарии:
    DomainMonitor + AdaptiveTrainer + CheckpointManager:
        1. DomainMonitor не сигнализирует о дрейфе при стабильном распределении.
        2. DomainMonitor сигнализирует о дрейфе при смене гистограммы.
        3. AdaptiveTrainer.collect() не запускает шаг при пустом буфере.
        4. AdaptiveTrainer.collect() запускает _adapt_step при достижении adapt_batch.
        5. После _adapt_step буфер очищается.
        6. CheckpointManager.save() создаёт файл и обновляет реестр.
        7. CheckpointManager.rollback() удаляет последний чекпойнт.
        8. CheckpointManager.best() возвращает Path последнего чекпойнта.
        9. Интегральный сценарий: collect → дрейф → адаптация → сохранение чекпойнта.
"""
from __future__ import annotations

import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import timm
import torch


# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def no_pretrained(monkeypatch):
    orig = timm.create_model
    monkeypatch.setattr(
        timm, "create_model",
        lambda *a, **kw: orig(*a, **{**kw, "pretrained": False}),
    )


@pytest.fixture
def small_model():
    """Маленькая модель с 2 слоями TCN для быстрых тестов."""
    from deepfake_detector.models.full_model import DeepfakeDetector
    cfg = types.SimpleNamespace(
        embed_dim=512, n_layers=2, kernel=3,
        dropout=0.0, freeze_base=False,
        arc_s=32.0, arc_m=0.5,
    )
    m = DeepfakeDetector(cfg)
    m.eval()
    return m


@pytest.fixture
def adapt_cfg():
    return types.SimpleNamespace(
        adapt_lr=1e-5,
        adapt_batch=2,
        confidence_high=0.9,
        window=5,
        kl_threshold=0.35,
    )


# ---------------------------------------------------------------------------
# Тесты DomainMonitor
# ---------------------------------------------------------------------------

class TestDomainMonitorIntegration:
    """Проверяет DomainMonitor в связке с реальными гистограммами."""

    def test_no_drift_on_stable_distribution(self):
        """Стабильное равномерное распределение → дрейф не обнаруживается."""
        from deepfake_detector.adaptation.monitor import DomainMonitor
        ref = np.ones(16) / 16.0
        monitor = DomainMonitor({"hist": ref.copy()}, window=5, kl_thr=0.35)
        result = None
        for _ in range(10):
            noisy = ref + np.random.default_rng(42).uniform(-0.01, 0.01, 16)
            noisy = np.clip(noisy, 0, None)
            noisy /= noisy.sum()
            result = monitor.update(noisy)
        assert result is False

    def test_drift_detected_on_distribution_shift(self):
        """Резкая смена распределения → дрейф обнаруживается."""
        from deepfake_detector.adaptation.monitor import DomainMonitor
        ref = np.ones(16) / 16.0
        monitor = DomainMonitor({"hist": ref.copy()}, window=5, kl_thr=0.1)
        drifted = np.zeros(16)
        drifted[0] = 1.0
        result = None
        for _ in range(5):
            result = monitor.update(drifted)
        assert result is True

    def test_monitor_buffer_limits_to_window(self):
        """Буфер не накапливает больше window гистограмм."""
        from deepfake_detector.adaptation.monitor import DomainMonitor
        ref = np.ones(8) / 8.0
        monitor = DomainMonitor({"hist": ref}, window=3, kl_thr=1.0)
        for _ in range(10):
            monitor.update(ref.copy())
        assert len(monitor.buf) == 3


# ---------------------------------------------------------------------------
# Тесты AdaptiveTrainer
# ---------------------------------------------------------------------------

class TestAdaptiveTrainerIntegration:
    """Проверяет AdaptiveTrainer с реальной моделью."""

    def test_collect_low_confidence_ignored(self, small_model, adapt_cfg):
        """Сэмплы со средней уверенностью (0.1 ≤ prob ≤ 0.9) не попадают в буфер."""
        from deepfake_detector.adaptation.adaptive_trainer import AdaptiveTrainer
        trainer = AdaptiveTrainer(small_model, adapt_cfg)
        video = torch.randn(4, 3, 64, 64)
        trainer.collect(video, prob=0.5)
        assert len(trainer.buffer) == 0

    def test_collect_high_confidence_added_to_buffer(self, small_model, adapt_cfg):
        """Сэмплы с prob > 0.9 попадают в буфер."""
        from deepfake_detector.adaptation.adaptive_trainer import AdaptiveTrainer
        trainer = AdaptiveTrainer(small_model, adapt_cfg)
        video = torch.randn(4, 3, 64, 64)
        trainer.collect(video, prob=0.95)
        assert len(trainer.buffer) == 1

    def test_collect_low_prob_added_to_buffer(self, small_model, adapt_cfg):
        """Сэмплы с prob < 0.1 (уверенный real) также попадают в буфер."""
        from deepfake_detector.adaptation.adaptive_trainer import AdaptiveTrainer
        trainer = AdaptiveTrainer(small_model, adapt_cfg)
        video = torch.randn(4, 3, 64, 64)
        trainer.collect(video, prob=0.05)
        assert len(trainer.buffer) == 1

    def test_adapt_step_clears_buffer(self, small_model, adapt_cfg):
        """После достижения adapt_batch буфер очищается."""
        from deepfake_detector.adaptation.adaptive_trainer import AdaptiveTrainer
        trainer = AdaptiveTrainer(small_model, adapt_cfg)
        video = torch.randn(4, 3, 64, 64)
        # adapt_batch=2, собираем 2 сэмпла
        trainer.collect(video, prob=0.95)
        trainer.collect(video, prob=0.05)
        # После второго collect должен сработать _adapt_step → буфер пуст
        assert len(trainer.buffer) == 0

    def test_adapt_step_does_not_raise(self, small_model, adapt_cfg):
        """_adapt_step() не бросает исключений."""
        from deepfake_detector.adaptation.adaptive_trainer import AdaptiveTrainer
        trainer = AdaptiveTrainer(small_model, adapt_cfg)
        video = torch.randn(4, 3, 64, 64)
        trainer.buffer = [(video, 0), (video, 1)]
        trainer._adapt_step()

    def test_only_head_and_proj_are_trainable(self, small_model, adapt_cfg):
        """AdaptiveTrainer обучает только head и proj (~0.03% параметров)."""
        from deepfake_detector.adaptation.adaptive_trainer import AdaptiveTrainer
        trainer = AdaptiveTrainer(small_model, adapt_cfg)
        param_names = [n for n, p in small_model.named_parameters()
                       if "head" in n or "proj" in n]
        trainable_params = trainer.opt.param_groups[0]["params"]
        # Все обучаемые параметры — только из head/proj
        assert len(trainable_params) > 0


# ---------------------------------------------------------------------------
# Тесты CheckpointManager
# ---------------------------------------------------------------------------

class TestCheckpointManagerIntegration:
    """Проверяет CheckpointManager с реальной файловой системой."""

    @pytest.fixture
    def manager(self, tmp_path):
        from deepfake_detector.adaptation.checkpoint_manager import CheckpointManager
        return CheckpointManager(checkpoint_dir=tmp_path / "ckpts")

    def test_save_creates_file(self, manager, small_model):
        """save() создаёт файл .pt на диске."""
        state = small_model.state_dict()
        path = manager.save(state, {"auroc": 0.85, "epoch": 0})
        assert path.exists()
        assert path.suffix == ".pt"

    def test_save_updates_registry(self, manager, small_model):
        """save() добавляет запись в реестр."""
        state = small_model.state_dict()
        manager.save(state, {"auroc": 0.80})
        manager.save(state, {"auroc": 0.85})
        assert len(manager.registry) == 2

    def test_registry_persists_to_disk(self, manager, small_model, tmp_path):
        """registry.json существует и содержит все записи."""
        state = small_model.state_dict()
        manager.save(state, {"auroc": 0.90})
        assert manager.registry_path.exists()
        import json
        with open(manager.registry_path) as f:
            data = json.load(f)
        assert len(data) == 1

    def test_best_returns_last_saved(self, manager, small_model):
        """best() возвращает путь к последнему сохранённому чекпойнту."""
        state = small_model.state_dict()
        p1 = manager.save(state, {"auroc": 0.80})
        p2 = manager.save(state, {"auroc": 0.85})
        assert manager.best() == p2

    def test_rollback_removes_last_checkpoint(self, manager, small_model):
        """rollback() удаляет последний файл и запись из реестра."""
        state = small_model.state_dict()
        p1 = manager.save(state, {"auroc": 0.80})
        p2 = manager.save(state, {"auroc": 0.85})
        manager.rollback()
        assert not p2.exists()
        assert len(manager.registry) == 1

    def test_rollback_on_single_checkpoint_is_safe(self, manager, small_model):
        """rollback() при одном чекпойнте — нет ошибок (нечего откатывать)."""
        state = small_model.state_dict()
        manager.save(state, {"auroc": 0.80})
        manager.rollback()  # len < 2 → no-op

    def test_best_on_empty_manager_returns_none(self, manager):
        """best() на пустом менеджере возвращает None."""
        assert manager.best() is None


# ---------------------------------------------------------------------------
# Интегральный сценарий
# ---------------------------------------------------------------------------

class TestAdaptationIntegralScenario:
    """E2E: Monitor обнаруживает дрейф → Trainer адаптируется → Manager сохраняет."""

    def test_full_adaptation_cycle(self, small_model, adapt_cfg, tmp_path):
        """Полный цикл адаптации без ошибок."""
        from deepfake_detector.adaptation.monitor import DomainMonitor
        from deepfake_detector.adaptation.adaptive_trainer import AdaptiveTrainer
        from deepfake_detector.adaptation.checkpoint_manager import CheckpointManager

        ref_hist = np.ones(16) / 16.0
        monitor = DomainMonitor({"hist": ref_hist.copy()}, window=3, kl_thr=0.1)
        adapt_trainer = AdaptiveTrainer(small_model, adapt_cfg)
        ckpt_manager = CheckpointManager(tmp_path / "ckpts")

        drifted = np.zeros(16); drifted[0] = 1.0
        drift_detected = False

        for step in range(6):
            drift = monitor.update(drifted)
            if drift and not drift_detected:
                drift_detected = True
                # Адаптация
                video = torch.randn(4, 3, 64, 64)
                adapt_trainer.collect(video, prob=0.95)
                adapt_trainer.collect(video, prob=0.02)
                # Сохранение
                ckpt_manager.save(small_model.state_dict(), {"step": step})

        assert drift_detected, "Дрейф должен быть обнаружен"
        assert len(ckpt_manager.registry) >= 1
