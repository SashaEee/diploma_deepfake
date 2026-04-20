import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveDilatedConv1d(nn.Module):
    """Каузальная 1D-свёртка с динамически выбираемой дилатацией."""

    def __init__(self, ch: int, k: int = 3, base_dil: int = 1, max_dil: int = 16):
        super().__init__()
        self.k = k
        self.base_dil = base_dil
        self.max_dil = max_dil
        self.weight = nn.Parameter(torch.empty(2 * ch, ch, k))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.bias = nn.Parameter(torch.zeros(2 * ch))

    def forward(self, x: torch.Tensor, energy: torch.Tensor) -> torch.Tensor:
        phi = 1.0 / (1.0 + energy.mean().clamp(min=1e-6))
        d = int(max(1, min(self.max_dil, round(self.base_dil * phi.item()))))
        pad = (self.k - 1) * d
        y = F.conv1d(F.pad(x, (pad, 0)), self.weight, self.bias, dilation=d)
        a, b = y.chunk(2, dim=1)
        return a * torch.sigmoid(b)


class TemporalBlock(nn.Module):
    def __init__(self, ch: int, k: int, base_dil: int, dropout: float):
        super().__init__()
        self.conv = AdaptiveDilatedConv1d(ch, k, base_dil)
        self.ln = nn.LayerNorm(ch)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, energy: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.conv(x, energy)
        y = self.ln(y.transpose(1, 2)).transpose(1, 2)
        return self.dropout(y) + residual


class TemporalEncoder(nn.Module):
    def __init__(self, ch: int = 512, n_layers: int = 6, k: int = 3, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            TemporalBlock(ch, k, 2 ** i, dropout) for i in range(n_layers)
        ])

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: (B, T, C) -> (B, C, T)
        x = seq.transpose(1, 2)
        energy = (x[..., 1:] - x[..., :-1]).pow(2).mean(dim=1, keepdim=True)
        energy = F.pad(energy, (1, 0))
        for blk in self.blocks:
            x = blk(x, energy)
        return x.mean(dim=-1)
