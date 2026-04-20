import numpy as np
from collections import deque


class DomainMonitor:
    """Мониторинг доменного дрейфа по симметризованному KL-расхождению гистограмм."""

    def __init__(self, ref_stats: dict, window: int = 1000, kl_thr: float = 0.35):
        self.ref = ref_stats
        self.window = window
        self.kl_thr = kl_thr
        self.buf = deque(maxlen=window)

    def update(self, activation_hist: np.ndarray) -> bool:
        self.buf.append(activation_hist)
        if len(self.buf) < self.window:
            return False
        hist = np.mean(np.stack(self.buf, axis=0), axis=0)
        kl = self._sym_kl(hist, self.ref["hist"])
        return bool(kl > self.kl_thr)

    @staticmethod
    def _sym_kl(p, q, eps: float = 1e-9):
        p = p + eps
        q = q + eps
        return 0.5 * (np.sum(p * np.log(p / q)) + np.sum(q * np.log(q / p)))
