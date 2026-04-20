import torch
import torch.nn.functional as F
from torch.optim import AdamW


class AdaptiveTrainer:
    """Онлайн-адаптация по псевдо-меткам с высокой уверенностью; только head + proj."""

    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        trainable = [p for n, p in model.named_parameters()
                     if "head" in n or "proj" in n]
        self.opt = AdamW(trainable, lr=cfg.adapt_lr)
        self.buffer = []

    def collect(self, video: torch.Tensor, prob: float):
        if prob > 0.9 or prob < 0.1:
            pseudo = 1 if prob > 0.5 else 0
            self.buffer.append((video, pseudo))
        if len(self.buffer) >= self.cfg.adapt_batch:
            self._adapt_step()

    def _adapt_step(self):
        self.model.train()
        for video, label in self.buffer:
            label_t = torch.tensor([label])
            logits, _, _ = self.model(video.unsqueeze(0), label_t)
            if logits.dim() == 1:
                loss = F.binary_cross_entropy_with_logits(logits, label_t.float())
            else:
                loss = F.cross_entropy(logits, label_t)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
        self.buffer.clear()
        self.model.eval()
