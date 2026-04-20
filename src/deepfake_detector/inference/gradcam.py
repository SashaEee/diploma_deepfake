import torch
import torch.nn.functional as F
import numpy as np


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.acts = None
        self.grads = None
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, _m, _inp, out):
        self.acts = out.detach()

    def _bwd(self, _m, _gin, gout):
        self.grads = gout[0].detach()

    def __call__(self, x: torch.Tensor, target: int = 1) -> np.ndarray:
        self.model.zero_grad()
        logits = self.model(x)
        logits[:, target].sum().backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = (w * self.acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
        return cam[0, 0].cpu().numpy()
