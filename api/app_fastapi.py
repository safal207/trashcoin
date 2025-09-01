from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
from ml.inference.predictor import Predictor
import os

# --- Model and Artifact Configuration ---
# This is the direct download link for the MobileNetV2 model from the ONNX Model Zoo.
MODEL_URL = "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx"
LABELS_PATH = "ml/artifacts/labels.json"
VERSION_PATH = "ml/artifacts/VERSION"

# Create the artifacts directory if it doesn't exist
os.makedirs("ml/artifacts", exist_ok=True)


app = FastAPI(title="TrashCoin ML API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# This will now download the model on first startup if it's not cached
predictor = Predictor(
    model_url=MODEL_URL,
    labels_path=LABELS_PATH,
    version_path=VERSION_PATH
)

class PredictResponse(BaseModel):
    class_: str
    confidence: float
    model_version: str
    inference_ms: int

@app.post("/api/classify", response_model=PredictResponse)
async def classify(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Unsupported file type, please upload an image.")

    image_bytes = await file.read()

    try:
        t0 = time.time()
        label, conf = predictor.predict(image_bytes)
        dt = int((time.time() - t0) * 1000)

        return PredictResponse(
            class_=label,
            confidence=conf,
            model_version=predictor.version,
            inference_ms=dt
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
