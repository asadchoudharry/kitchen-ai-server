from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI(title="KitchenAI Vision (YOLOv8)")

# Auto-download weights on first run
model = YOLO("yolov8n.pt")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        return {"error": "Only jpg/png/webp allowed"}

    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")

    results = model.predict(img, verbose=False)

    detections = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            detections.append({
                "label": model.names.get(cls, str(cls)),
                "confidence": round(conf, 3)
            })

    return {"detections": detections, "count": len(detections)}
