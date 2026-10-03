import json
import time
from dataclasses import asdict

import cv2

from .analysis.convoy import ConvoyAnalyzer
from .config import settings
from .detection.yolo import YoloDetector


def run(job, req, src, out):
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise ValueError("Cannot open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    job["status"] = "loading_model"
    det = YoloDetector(req.model, settings.device, req.conf, req.iou, req.mode != "detection")
    det.load()
    ana = ConvoyAnalyzer(req.link_dist_frac * w, min_members=req.min_members,
                         min_persist=req.min_persist_frames, fps=fps)
    job["status"] = "processing"
    frames, fid, t0 = [], 0, time.time()
    while not job["cancel"]:
        ok, frame = cap.read()
        if not ok:
            break
        if fid % req.stride == 0:
            dets = det.predict(frame)
            frames.append({"frame_id": fid, "t": round(fid / fps, 3),
                           "detections": [asdict(d) for d in dets], "groups": ana.update(fid, dets)})
        fid += 1
        job["progress"], job["fps"] = min(fid / total, 1.0), fid / (time.time() - t0)
    cap.release()
    meta = {"width": w, "height": h, "fps": fps, "total_frames": total, "stride": req.stride,
            "model": req.model, "mode": req.mode, "device": det.device, "conf": req.conf, "iou": req.iou,
            "link_dist_px": req.link_dist_frac * w, "min_members": req.min_members,
            "min_persist_frames": req.min_persist_frames, "direction_note": "image-plane, not geographic"}
    out.write_text(json.dumps({"meta": meta, "frames": frames, "groups": ana.summary()}))
