import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):
    """Метрическая голова ArcFace: проекция на гиперсферу + угловой margin."""

    def __init__(self, dim: int, n_classes: int = 2, s: float = 32.0, m: float = 0.5):
        super().__init__()
        self.s, self.m = s, m
        self.W = nn.Parameter(torch.empty(dim, n_classes))
        nn.init.xavier_uniform_(self.W)

    def forward(self, emb: torch.Tensor, labels: torch.Tensor | None = None):
        emb = F.normalize(emb, dim=1)
        w = F.normalize(self.W, dim=0)
        logits = emb @ w
        if labels is None:
            return logits * self.s
        cos = logits.clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cos)
        onehot = F.one_hot(labels, num_classes=logits.size(1)).float()
        margin = onehot * self.m
        return self.s * torch.cos(theta + margin)
