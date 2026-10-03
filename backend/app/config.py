import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _p(k, d):
    return Path(os.getenv(k, ROOT / d))


class Settings:
    upload_dir = _p("UPLOAD_DIR", "storage/uploads")
    result_dir = _p("RESULT_DIR", "storage/results")
    models = os.getenv("MODELS", "yolov8n-seg.pt,yolov8s-seg.pt,yolov8n.pt").split(",")
    device = os.getenv("DEVICE", "auto")  # auto | cpu | cuda
    conf = float(os.getenv("CONFIDENCE_THRESHOLD", 0.40))
    iou = float(os.getenv("IOU_THRESHOLD", 0.50))
    max_upload_mb = int(os.getenv("MAX_UPLOAD_SIZE", 500))
    # COCO ids: 2 car, 3 motorcycle, 5 bus, 7 truck
    classes = [int(c) for c in os.getenv("VEHICLE_CLASSES", "2,3,5,7").split(",")]


settings = Settings()
for _d in (settings.upload_dir, settings.result_dir):
    _d.mkdir(parents=True, exist_ok=True)
