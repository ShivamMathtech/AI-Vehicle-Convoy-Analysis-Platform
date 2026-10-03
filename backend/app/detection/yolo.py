import cv2
import numpy as np
from ultralytics import YOLO

from ..config import settings
from ..interfaces import BaseDetector, Det


def resolve_device(pref: str) -> str:
    import torch
    if pref == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda:0"
    if pref == "cuda":
        raise RuntimeError("CUDA requested but not available")
    return "cpu"


class YoloDetector(BaseDetector):
    """Detection + optional instance masks + persistent IDs (ByteTrack) via ultralytics."""

    def __init__(self, model, device="auto", conf=0.4, iou=0.5, segment=True, tracker="bytetrack.yaml"):
        self.model_path, self.conf, self.iou = model, conf, iou
        self.segment, self.tracker = segment, tracker
        self.device = resolve_device(device)

    def load(self):
        self.model = YOLO(self.model_path)
        self.names = self.model.names

    def predict(self, frame):
        r = self.model.track(frame, persist=True, tracker=self.tracker, conf=self.conf, iou=self.iou,
                             classes=settings.classes, device=self.device, verbose=False)[0]
        if r.boxes is None or r.boxes.id is None:
            return []
        ids, cls = r.boxes.id.int().tolist(), r.boxes.cls.int().tolist()
        confs, xyxy = r.boxes.conf.tolist(), r.boxes.xyxy.tolist()
        polys = r.masks.xy if (self.segment and r.masks is not None) else [None] * len(ids)
        out = []
        for tid, c, cf, b, p in zip(ids, cls, confs, xyxy, polys):
            has = p is not None and len(p) > 2
            out.append(Det(tid, c, self.names[c], round(cf, 3), [round(v, 1) for v in b],
                           ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2), (b[2] - b[0]) * (b[3] - b[1]),
                           np.round(p, 1).tolist() if has else None,
                           float(cv2.contourArea(p.astype(np.float32))) if has else 0.0))
        return out
