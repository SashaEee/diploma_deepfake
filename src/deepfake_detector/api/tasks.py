"""Celery-задачи для асинхронной обработки видео."""
import json
import os
from pathlib import Path

from celery import Celery

broker = os.environ.get("CELERY_BROKER", "redis://localhost:6379/0")
celery_app = Celery("deepfake_detector", broker=broker, backend=broker)


@celery_app.task(name="deepfake_detector.api.tasks.process_video")
def process_video(path: str, job_id: str) -> None:
    from redis import Redis
    from deepfake_detector.inference.predictor import Predictor

    r = Redis.from_url(broker)
    r.setex(f"job:{job_id}", 3600, json.dumps({"job_id": job_id, "status": "running"}))

    try:
        predictor = Predictor.from_config("configs/base.yaml")
        result = predictor.predict_video(Path(path))
        payload = {
            "job_id": job_id,
            "status": "done",
            "probability": result["probability"],
            "verdict": result["verdict"],
        }
    except Exception as exc:
        payload = {"job_id": job_id, "status": "error", "error": str(exc)}
    finally:
        Path(path).unlink(missing_ok=True)

    r.setex(f"job:{job_id}", 3600, json.dumps(payload))
