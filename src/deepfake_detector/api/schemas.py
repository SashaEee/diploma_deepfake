from pydantic import BaseModel, Field


class ImageResult(BaseModel):
    probability: float = Field(..., ge=0.0, le=1.0)
    verdict: str  # "real" | "fake"
    confidence: float
    processing_ms: int


class VideoJobCreated(BaseModel):
    job_id: str
    estimated_ms: int


class VideoResult(BaseModel):
    job_id: str
    status: str  # "pending" | "running" | "done" | "error"
    probability: float | None = None
    verdict: str | None = None
    per_frame: list[float] | None = None
    gradcam_url: str | None = None
    error: str | None = None


class HealthStatus(BaseModel):
    status: str
    model_version: str
    uptime_s: int
