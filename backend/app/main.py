import logging
import threading
import uuid
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import pipeline
from .config import ROOT, settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("convoy")
app = FastAPI(title="AI Vehicle Convoy Analysis", version="0.1.0")
JOBS: dict = {}
ALLOWED = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class StartReq(BaseModel):
    upload_id: str
    model: str = settings.models[0]
    mode: str = Field("detection+segmentation", pattern=r"^(detection|detection\+segmentation)$")
    conf: float = Field(settings.conf, ge=0.05, le=0.95)
    iou: float = Field(settings.iou, ge=0.1, le=0.95)
    stride: int = Field(1, ge=1, le=30)
    link_dist_frac: float = Field(0.2, gt=0, le=1)  # max link distance as fraction of frame width
    min_members: int = Field(3, ge=2, le=20)
    min_persist_frames: int = Field(15, ge=1)


def _find(uid: str) -> Path:
    if not uid.isalnum():
        raise HTTPException(400, "Invalid id")
    for p in settings.upload_dir.glob(f"{uid}.*"):
        return p
    raise HTTPException(404, "Upload not found")


@app.get("/api/v1/system")
def system():
    import torch
    cuda = torch.cuda.is_available()
    return {"device": torch.cuda.get_device_name(0) if cuda else "CPU", "cuda": cuda, "models": settings.models}


@app.post("/api/v1/upload")
async def upload(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED:
        raise HTTPException(415, f"Unsupported type. Allowed: {sorted(ALLOWED)}")
    uid = uuid.uuid4().hex
    dest, size, too_big = settings.upload_dir / f"{uid}{ext}", 0, False
    with dest.open("wb") as f:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > settings.max_upload_mb << 20:
                too_big = True
                break
            f.write(chunk)
    if too_big:
        dest.unlink(missing_ok=True)
        raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB")
    cap = cv2.VideoCapture(str(dest))
    if not cap.isOpened():
        dest.unlink(missing_ok=True)
        raise HTTPException(422, "File is not a readable video")
    meta = {"upload_id": uid, "width": int(cap.get(3)), "height": int(cap.get(4)),
            "fps": cap.get(cv2.CAP_PROP_FPS) or 25.0, "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}
    cap.release()
    log.info("upload %s %s bytes", uid, size)
    return meta


@app.get("/api/v1/video/{uid}")
def video(uid: str):
    return FileResponse(_find(uid))


def _work(aid, req, src):
    job = JOBS[aid]
    try:
        pipeline.run(job, req, src, settings.result_dir / f"{aid}.json")
        job["status"] = "cancelled" if job["cancel"] else "completed"
    except Exception as e:  # surfaced to UI, never silent
        log.exception("analysis %s failed", aid)
        job.update(status="failed", error=str(e))


@app.post("/api/v1/analysis/start")
def start(req: StartReq):
    if req.model not in settings.models:
        raise HTTPException(422, f"Model not allowed. Choose from {settings.models}")
    src, aid = _find(req.upload_id), uuid.uuid4().hex[:12]
    JOBS[aid] = {"id": aid, "status": "queued", "progress": 0.0, "fps": 0.0, "error": None, "cancel": False}
    threading.Thread(target=_work, args=(aid, req, src), daemon=True).start()
    return JOBS[aid]


def _job(aid):
    if aid not in JOBS:
        raise HTTPException(404, "Unknown analysis")
    return JOBS[aid]


@app.get("/api/v1/analysis/{aid}")
def status(aid: str):
    return _job(aid)


@app.post("/api/v1/analysis/{aid}/stop")
def stop(aid: str):
    _job(aid)["cancel"] = True
    return {"ok": True}


@app.get("/api/v1/analysis/{aid}/result")
def result(aid: str):
    if _job(aid)["status"] not in ("completed", "cancelled"):
        raise HTTPException(409, "Result not ready")
    return FileResponse(settings.result_dir / f"{aid}.json", media_type="application/json")


app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="ui")
