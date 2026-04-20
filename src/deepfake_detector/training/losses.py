import torch
import torch.nn as nn
import torch.nn.functional as F


def temporal_consistency_loss(frame_emb: torch.Tensor) -> torch.Tensor:
    """L_tc: штраф за рассогласование соседних покадровых эмбеддингов."""
    e = F.normalize(frame_emb, dim=-1)
    sim = (e[:, 1:] * e[:, :-1]).sum(dim=-1)
    return (1.0 - sim).mean()


class CombinedLoss(nn.Module):
    def __init__(self, lam_arc: float = 1.0, lam_tc: float = 0.2, lam_short: float = 0.1):
        super().__init__()
        self.lam_arc, self.lam_tc, self.lam_short = lam_arc, lam_tc, lam_short
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, frame_emb, labels, shortcut_reg):
        # Поддержка обеих архитектур:
        # - logits (B,)   → Linear head (BCE) — текущая обученная модель
        # - logits (B, C) → ArcFace head (CE)  — ТЗ / обратная совместимость
        if logits.dim() == 1:
            loss_arc = F.binary_cross_entropy_with_logits(logits, labels.float())
        else:
            loss_arc = self.ce(logits, labels)
        loss_tc = temporal_consistency_loss(frame_emb)
        return self.lam_arc * loss_arc + self.lam_tc * loss_tc + self.lam_short * shortcut_reg
