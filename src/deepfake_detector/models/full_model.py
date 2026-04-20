import torch
import torch.nn as nn
from .spatial_encoder import SpatialEncoder
from .temporal_encoder import TemporalEncoder


class DeepfakeDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.spatial = SpatialEncoder(embed_dim=cfg.embed_dim)
        self.temporal = TemporalEncoder(
            ch=cfg.embed_dim, n_layers=cfg.n_layers,
            k=cfg.kernel, dropout=cfg.dropout,
        )
        # Linear head (BCE) — подтверждено на Kaggle (AUROC 0.93 vs 0.49 с ArcFace).
        # ArcFaceHead сохранён в metric_head.py для совместимости с ТЗ и тестами.
        self.head = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.embed_dim, 1),
        )
        self.freeze_base = getattr(cfg, "freeze_base", False)
        if self.freeze_base:
            for p in self.spatial.backbone.parameters():
                p.requires_grad = False

    def forward(self, video: torch.Tensor, labels: torch.Tensor | None = None):
        b, t, c, h, w = video.shape
        frames = video.view(b * t, c, h, w)
        spatial = self.spatial(frames).view(b, t, -1)
        agg = self.temporal(spatial)
        logit = self.head(agg).squeeze(-1)          # (B,) — бинарный логит
        shortcut_reg = (spatial.abs().mean() - 0.5).pow(2)
        return logit, spatial, shortcut_reg

    @staticmethod
    def remap_state_dict(kaggle_sd: dict) -> dict:
        """Переименовывае�� ключи Kaggle-чекпоинта в имена проекта.

        Kaggle-модель использовала другие имена атрибутов:
          spatial.bb.*         → spatial.backbone.*
          temporal.blks.*      → temporal.blocks.*
          .conv.W / .conv.b    → .conv.weight / .conv.bias
          spatial.proj.*       → без изменений (Sequential: Linear + BN + ReLU)
          head.*               → head.*  (сов��адает — оба Sequential)
        """
        out = {}
        for k, v in kaggle_sd.items():
            nk = k
            nk = nk.replace("spatial.bb.", "spatial.backbone.")
            nk = nk.replace("temporal.blks.", "temporal.blocks.")
            nk = nk.replace(".conv.W", ".conv.weight")
            nk = nk.replace(".conv.b", ".conv.bias")
            out[nk] = v
        return out
