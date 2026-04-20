import time
import numpy as np
import cv2

from fastapi import FastAPI, UploadFile, File, HTTPException

from deepfake_detector.api.schemas import ImageResult, VideoJobCreated, VideoResult, HealthStatus
from deepfake_detector.inference.predictor import Predictor

app = FastAPI(title="Deepfake Detector API", version="1.0.0")

_start_time = time.time()
predictor: Predictor | None = None


@app.on_event("startup")
async def startup():
    global predictor
    predictor = Predictor.from_config("configs/base.yaml")


@app.post("/predict/image", response_model=ImageResult)
async def predict_image(file: UploadFile = File(...)):
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "empty file")
    arr = np.frombuffer(raw, dtype=np.uint8)
    try:
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        img = None
    if img is None:
        raise HTTPException(400, "invalid image")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = predictor.predict_image(img)
    return ImageResult(**result)


@app.post("/predict/video", response_model=VideoJobCreated)
async def predict_video(file: UploadFile = File(...)):
    import uuid
    from deepfake_detector.api.tasks import process_video
    import tempfile, pathlib

    job_id = str(uuid.uuid4())
    raw = await file.read()
    tmp = pathlib.Path(tempfile.mktemp(suffix=pathlib.Path(file.filename).suffix))
    tmp.write_bytes(raw)
    process_video.delay(str(tmp), job_id)
    return VideoJobCreated(job_id=job_id, estimated_ms=5000)


@app.get("/predict/video/{job_id}", response_model=VideoResult)
async def video_status(job_id: str):
    from deepfake_detector.api.tasks import celery_app
    import json
    from redis import Redis

    r = Redis.from_url("redis://localhost:6379/0")
    raw = r.get(f"job:{job_id}")
    if raw is None:
        return VideoResult(job_id=job_id, status="pending")
    data = json.loads(raw)
    return VideoResult(**data)


@app.get("/health", response_model=HealthStatus)
async def health():
    return HealthStatus(
        status="ok",
        model_version="1.0.0",
        uptime_s=int(time.time() - _start_time),
    )


@app.get("/metrics")
async def metrics():
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
