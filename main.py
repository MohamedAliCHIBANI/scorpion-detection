#main.py
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import json
from ultralytics import YOLO
import os
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import OpenCV avec fallback
try:
    import cv2
    CV2_AVAILABLE = True
    logger.info("OpenCV disponible")
except ImportError:
    logger.warning("OpenCV non disponible")
    CV2_AVAILABLE = False

app = FastAPI()

# Autoriser les origines
origins = [
    "https://MohamedAliCHIBANI.github.io",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://your-app-name.azurewebsites.net"  # Ajoutez votre URL Azure
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Charger le modèle
try:
    model_path = os.path.join(os.path.dirname(__file__), "best.pt")
    model = YOLO(model_path)
    logger.info("Modèle YOLO chargé avec succès")
except Exception as e:
    logger.error(f"Erreur lors du chargement du modèle: {e}")
    model = None

@app.get("/")
def home():
    return {
        "status": "ok", 
        "message": "Server running", 
        "cv2_available": CV2_AVAILABLE,
        "model_loaded": model is not None
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    if not CV2_AVAILABLE or model is None:
        await websocket.accept()
        await websocket.send_text(json.dumps({
            "error": "OpenCV or model not available",
            "cv2_available": CV2_AVAILABLE,
            "model_loaded": model is not None
        }))
        await websocket.close()
        return

    logger.info("Connexion WebSocket entrante...")
    await websocket.accept()
    logger.info("WebSocket acceptée!")
    
    try:
        while True:
            data = await websocket.receive_bytes()
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                await websocket.send_text(json.dumps({"error": "Failed to decode image"}))
                continue

            results = model(frame)
            detections = []

            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        detections.append({
                            "bbox": [x1, y1, x2, y2],
                            "confidence": conf,
                            "class_id": cls_id,
                            "class_name": model.names[cls_id]
                        })

            await websocket.send_text(json.dumps({"detections": detections}))

    except Exception as e:
        logger.error(f"Erreur WebSocket: {e}")
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)