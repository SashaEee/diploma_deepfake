import torch
import torch.nn as nn
import timm


class SEBlock(nn.Module):
    def __init__(self, channels: int, r: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // r), nn.ReLU(inplace=True),
            nn.Linear(channels // r, channels), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        s = self.pool(x).view(b, c)
        w = self.fc(s).view(b, c, 1, 1)
        return x * w


class FPN(nn.Module):
    def __init__(self, in_channels: list[int], out_ch: int = 128):
        super().__init__()
        self.lateral = nn.ModuleList(
            [nn.Conv2d(c, out_ch, 1) for c in in_channels]
        )
        self.smooth = nn.ModuleList(
            [nn.Conv2d(out_ch, out_ch, 3, padding=1) for _ in in_channels]
        )

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        lats = [l(f) for l, f in zip(self.lateral, feats)]
        outs = [lats[-1]]
        for l in reversed(range(len(lats) - 1)):
            up = nn.functional.interpolate(
                outs[0], size=lats[l].shape[-2:], mode="nearest",
            )
            outs.insert(0, lats[l] + up)
        return [s(o) for s, o in zip(self.smooth, outs)]


class SpatialEncoder(nn.Module):
    """Пространственный модуль: EfficientNet-B0 + FPN + SE + Linear projection."""

    def __init__(self, embed_dim: int = 512):
        super().__init__()
        backbone = timm.create_model(
            "efficientnet_b0", pretrained=True,
            features_only=True, out_indices=(2, 3, 4),
        )
        self.backbone = backbone
        self.fpn = FPN(backbone.feature_info.channels(), out_ch=128)
        self.se = nn.ModuleList([SEBlock(128) for _ in range(3)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(128 * 3, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        pyr = self.fpn(feats)
        attn = [se(p) for se, p in zip(self.se, pyr)]
        pooled = [self.pool(a).flatten(1) for a in attn]
        return self.proj(torch.cat(pooled, dim=1))
