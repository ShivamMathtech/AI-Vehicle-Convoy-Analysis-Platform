# AI Vehicle Convoy Analysis Platform (core slice: phases 1-6)
For authorised analysis of supplied video only. Outputs are image-plane measurements, not targeting or engagement guidance.

## Run
    pip install -r requirements.txt
    uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
    open http://127.0.0.1:8000     (API docs: /docs)
    pytest

First run downloads YOLO weights (internet needed once). Defaults detect COCO car/motorcycle/bus/truck;
set VEHICLE_CLASSES / MODELS in .env for a custom-trained model.

## Layout
- backend/app/interfaces.py   BaseDetector / BaseAnalyzer
- backend/app/detection/yolo.py   detect + segment + ByteTrack IDs (ultralytics)
- backend/app/analysis/       image-plane motion, graph-based convoy-like grouping w/ persistence
- backend/app/pipeline.py     video -> per-frame results JSON
- frontend/index.html         dashboard (served by FastAPI)

## Not built yet
Next.js UI, PostgreSQL/Redis queue, auth/rate limiting, SAM, datasets/annotation, training, evaluation metrics, reports, Docker.
Group confidence = mean detection confidence x direction consistency (heuristic, not calibrated).
