"""Predictor: объединяет препроцессинг и модель для инференса."""
from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from deepfake_detector.utils.config import load_config


class Predictor:
    def __init__(self, cfg, model, preprocessor):
        self.cfg = cfg
        self.model = model
        self.preprocessor = preprocessor
        self.model.eval()

        # Reusable face detector and normalizer (avoid re-creating MTCNN each call)
        from deepfake_detector.preprocessing.face_detector import FaceDetector
        from deepfake_detector.preprocessing.normalizer import FrameNormalizer

        self._face_detector = FaceDetector(
            min_conf=cfg.preprocessing.face_min_conf,
            device=cfg.inference.device,
        )
        self._normalizer = FrameNormalizer(
            mean=cfg.preprocessing.normalize_mean,
            std=cfg.preprocessing.normalize_std,
        )
        self._target_frames = cfg.preprocessing.target_frames  # 16

        # Frame buffer for camera mode (stores aligned 224x224 uint8 faces)
        self._frame_buffer: deque[np.ndarray] = deque(maxlen=self._target_frames)

    @classmethod
    def from_config(cls, cfg_path: str) -> "Predictor":
        cfg = load_config(cfg_path)
        from deepfake_detector.models.full_model import DeepfakeDetector
        from deepfake_detector.preprocessing.video_loader import VideoLoader
        from deepfake_detector.preprocessing.face_detector import FaceDetector
        from deepfake_detector.preprocessing.aligner import FaceAligner
        from deepfake_detector.preprocessing.normalizer import FrameNormalizer

        model = DeepfakeDetector(cfg.model)

        # Загружаем чекпоинт если указан в конфиге
        ckpt_path = getattr(cfg.inference, "checkpoint_path", None)
        if ckpt_path:
            ckpt_path = Path(ckpt_path)
            if ckpt_path.exists():
                raw = torch.load(ckpt_path, map_location="cpu")
                sd = raw.get("state_dict", raw)
                sd = DeepfakeDetector.remap_state_dict(sd)
                missing, unexpected = model.load_state_dict(sd, strict=False)
                if missing:
                    print(f"[Predictor] missing keys: {missing[:5]}{'...' if len(missing)>5 else ''}")
                if unexpected:
                    print(f"[Predictor] unexpected keys: {unexpected[:5]}")
                print(f"[Predictor] loaded checkpoint: {ckpt_path}")
            else:
                print(f"[Predictor] checkpoint not found: {ckpt_path}")

        video_loader = VideoLoader(
            target_frames=cfg.preprocessing.target_frames,
            max_side=cfg.preprocessing.max_side,
        )
        face_detector = FaceDetector(
            min_conf=cfg.preprocessing.face_min_conf,
            device=cfg.inference.device,
        )
        aligner = FaceAligner(face_detector)
        normalizer = FrameNormalizer(
            mean=cfg.preprocessing.normalize_mean,
            std=cfg.preprocessing.normalize_std,
        )

        def preprocessor(path: Path) -> torch.Tensor:
            frames = video_loader.load(path)
            aligned, _ = aligner(frames)
            return normalizer(aligned)

        return cls(cfg, model, preprocessor)

    # ── Video (file) ───────────��──────────────────────────────────────────

    def predict_video(self, path: Path | str) -> dict:
        """Возвращает {probability, verdict, confidence, face_detected, processing_ms}."""
        t0 = time.time()
        path = Path(path)
        video = self.preprocessor(path)
        result = self._run_model(video.unsqueeze(0))
        result["face_detected"] = True
        result["processing_ms"] = int((time.time() - t0) * 1000)
        return result

    # ── Single image (upload) ─────────────────────────────────────────────

    def predict_image(self, img: np.ndarray) -> dict:
        """Анализ одного изображения (загрузка файла).

        Детектит лицо MTCNN → affine alignment → если нет лица face_detected=False.
        Реплицирует на T фреймов для temporal encoder.
        """
        t0 = time.time()

        aligned = self._detect_and_align(img)
        if aligned is None:
            return self._no_face_result(t0)

        # Replicate single aligned face to T frames
        T = self._target_frames
        frames = np.stack([aligned] * T)  # (T, 224, 224, 3)
        video = self._normalizer(frames).unsqueeze(0)  # (1, T, 3, 224, 224)

        result = self._run_model(video)
        result["face_detected"] = True
        result["processing_ms"] = int((time.time() - t0) * 1000)
        return result

    # ── Camera frame (real-time with temporal buffer) ─────────────────────

    def predict_camera_frame(self, img: np.ndarray) -> dict:
        """��нализ кадра с камеры с накоплением временного буфера.

        Каждый вызов:
          1. Детектим лицо → align → если есть, добавляем в буфер (deque, max T).
          2. Если буфер пуст → face_detected=False.
          3. Собираем T фреймов из буфера → модель получает реальную
             временную последовательность → temporal encoder работает.
        """
        t0 = time.time()

        aligned = self._detect_and_align(img)
        face_detected = aligned is not None

        if face_detected:
            self._frame_buffer.append(aligned)

        buf_len = len(self._frame_buffer)

        if buf_len == 0:
            result = self._no_face_result(t0)
            result["buffer_frames"] = 0
            return result

        # Build T-frame sequence from buffer
        T = self._target_frames
        buf = list(self._frame_buffer)

        if buf_len < T:
            # Pad by repeating the earliest frame at the front
            pad = [buf[0]] * (T - buf_len)
            frames_list = pad + buf
        else:
            frames_list = buf[-T:]

        video = self._normalizer(np.stack(frames_list)).unsqueeze(0)

        result = self._run_model(video)
        result["face_detected"] = face_detected
        result["buffer_frames"] = buf_len
        result["processing_ms"] = int((time.time() - t0) * 1000)
        return result

    def reset_buffer(self):
        """Сброс буфера кадров (вызывать при остановке камеры)."""
        self._frame_buffer.clear()

    # ── Batch ───────���──────────────────────��──────────────────────────────

    def predict_batch(self, paths: list[Path]) -> list[dict]:
        return [self.predict_video(p) for p in paths]

    # ── Internal ──────────────────────────────────────────────────────────

    def _detect_and_align(self, img: np.ndarray) -> np.ndarray | None:
        """Детектит и выравнивает лицо через MTCNN + affine alignment."""
        h, w = img.shape[:2]
        max_side = getattr(self.cfg.preprocessing, "max_side", 640)
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), cv2.INTER_AREA)
        return self._face_detector(img)

    def _no_face_result(self, t0: float) -> dict:
        return {
            "probability": 0.5,
            "verdict": "no_face",
            "confidence": 0.0,
            "face_detected": False,
            "processing_ms": int((time.time() - t0) * 1000),
        }

    def _run_model(self, video: torch.Tensor) -> dict:
        with torch.no_grad():
            logit, _, _ = self.model(video)   # logit: (B,)
            prob_fake = float(torch.sigmoid(logit[0]))
        threshold = self.cfg.inference.threshold
        verdict = "fake" if prob_fake >= threshold else "real"
        # Уверенность — энтропия Бернулли, нормированная в [0, 1]
        p = np.clip(prob_fake, 1e-9, 1 - 1e-9)
        entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
        confidence = float(1.0 - entropy / np.log(2))
        return {"probability": prob_fake, "verdict": verdict, "confidence": confidence}
