from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, f1_score, roc_curve

from deepfake_detector.training.losses import CombinedLoss

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, cfg, model, train_ds, val_ds):
        self.cfg = cfg
        self.model = model
        self.loader = DataLoader(
            train_ds, batch_size=cfg.batch, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=False,
        )
        self.val = DataLoader(val_ds, batch_size=cfg.batch)
        self.opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
        self.sched = CosineAnnealingLR(self.opt, T_max=cfg.epochs)
        self.loss_fn = CombinedLoss(cfg.lam_arc, cfg.lam_tc, cfg.lam_short)
        self.scaler = torch.amp.GradScaler(enabled=cfg.mixed)

    def fit(self):
        best_auroc = 0.0
        for epoch in range(self.cfg.epochs):
            train_loss = self._train_epoch(epoch)
            metrics = self._validate()
            self.sched.step()
            logger.info(
                "Epoch %03d | train_loss=%.4f | auroc=%.4f | f1=%.4f | eer=%.4f",
                epoch, train_loss, metrics["auroc"], metrics["f1"], metrics["eer"],
            )
            if metrics["auroc"] > best_auroc:
                best_auroc = metrics["auroc"]
                self._save_checkpoint(epoch, metrics)

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        accum = self.cfg.grad_accum
        total_loss = 0.0
        n_steps = 0
        for step, batch in enumerate(self.loader):
            device = next(self.model.parameters()).device
            video = batch["video"].to(device)
            labels = batch["label"].to(device)
            with torch.amp.autocast("cpu", enabled=self.cfg.mixed, dtype=torch.bfloat16):
                logits, frame_emb, sreg = self.model(video, labels)
                loss = self.loss_fn(logits, frame_emb, labels, sreg) / accum
            self.scaler.scale(loss).backward()
            total_loss += loss.item() * accum
            n_steps += 1
            if (step + 1) % accum == 0:
                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad(set_to_none=True)
        return total_loss / max(n_steps, 1)

    def _validate(self) -> dict:
        self.model.eval()
        all_probs: list[float] = []
        all_labels: list[int] = []
        total_loss = 0.0
        n_batches = 0
        device = next(self.model.parameters()).device

        with torch.no_grad():
            for batch in self.val:
                video = batch["video"].to(device)
                labels = batch["label"].to(device)
                logits, frame_emb, sreg = self.model(video)
                loss = self.loss_fn(logits, frame_emb, labels, sreg)
                total_loss += loss.item()
                n_batches += 1
                if logits.dim() == 1:
                    probs = torch.sigmoid(logits)
                else:
                    probs = torch.softmax(logits, dim=1)[:, 1]
                all_probs.extend(probs.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        avg_loss = total_loss / max(n_batches, 1)
        if len(set(all_labels)) < 2:
            # Валидация невозможна без обоих классов
            return {"auroc": 0.0, "f1": 0.0, "eer": 0.0, "loss": avg_loss}

        probs_arr = np.array(all_probs, dtype=np.float64)
        labels_arr = np.array(all_labels, dtype=np.int32)

        auroc = float(roc_auc_score(labels_arr, probs_arr))
        preds = (probs_arr >= 0.5).astype(np.int32)
        f1 = float(f1_score(labels_arr, preds, zero_division=0))
        eer = self._compute_eer(labels_arr, probs_arr)

        return {"auroc": auroc, "f1": f1, "eer": eer, "loss": avg_loss}

    @staticmethod
    def _compute_eer(labels: np.ndarray, probs: np.ndarray) -> float:
        """Equal Error Rate: точка, где FAR == FRR."""
        fpr, tpr, _ = roc_curve(labels, probs)
        fnr = 1.0 - tpr
        idx = int(np.argmin(np.abs(fnr - fpr)))
        return float((fpr[idx] + fnr[idx]) / 2.0)

    def _save_checkpoint(self, epoch: int, metrics: dict) -> None:
        ckpt_dir = Path("checkpoints")
        ckpt_dir.mkdir(exist_ok=True)
        path = ckpt_dir / f"best_epoch{epoch:03d}_auroc{metrics['auroc']:.4f}.pt"
        torch.save(
            {
                "epoch": epoch,
                "metrics": metrics,
                "state_dict": self.model.state_dict(),
                "optimizer": self.opt.state_dict(),
            },
            path,
        )
        logger.info("Checkpoint saved → %s", path)
