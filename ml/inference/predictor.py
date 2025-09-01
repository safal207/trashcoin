import json
import io
import onnxruntime as ort
import numpy as np
from PIL import Image
from .preproc import preprocess
from .postproc import softmax
import os
import urllib.request

class Predictor:
    def __init__(self, model_url: str, labels_path: str, version_path: str):
        # Define a local cache path for the model
        self.model_path = "/tmp/model.onnx"

        # Download the model if it doesn't exist in the cache
        if not os.path.exists(self.model_path):
            print(f"Downloading model from {model_url} to {self.model_path}...")
            urllib.request.urlretrieve(model_url, self.model_path)
            print("Model downloaded successfully.")

        self.sess = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])

        # The labels and version can still be loaded from the repo
        with open(labels_path, "r", encoding="utf-8") as f:
            self.labels = json.load(f)
        with open(version_path, "r", encoding="utf-8") as f:
            self.version = f.read().strip()

        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

    def predict(self, image_bytes: bytes):
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        x = preprocess(img)
        outputs = self.sess.run([self.output_name], {self.input_name: x})[0]
        probs = softmax(outputs[0])
        idx = int(np.argmax(probs))
        label = self.labels[idx]
        confidence = float(probs[idx])
        return label, confidence
