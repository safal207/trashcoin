# Machine Learning Model and API

This document describes the machine learning components of the TrashCoin project, including the model, the inference API, and instructions for training a new model.

## Model

The current model is a pre-trained **MobileNetV2** from the ONNX Model Zoo, converted to the ONNX format. This serves as a baseline for the project.

-   **Model File:** The application downloads `model.onnx` on its first run. The URL is configured in `api/app_fastapi.py`. The model is cached locally in `/tmp/model.onnx`.
-   **Version:** `taco-mobilenet-v1` (as specified in `ml/artifacts/VERSION`). This name is a placeholder for the future custom-trained model.
-   **Classes:** The model is trained on the ImageNet dataset, which contains 1000 classes. The full list of classes is in `ml/artifacts/labels.json`. This is a general-purpose model and is **not** yet fine-tuned for trash classification.

## Inference API

The ML inference is served via a FastAPI application, which provides a clean, modern API.

-   **Source Code:** `api/app_fastapi.py`
-   **Endpoint:** `POST /api/classify`

### How to Run the API

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the server:**
    ```bash
    uvicorn api.app_fastapi:app --host 0.0.0.0 --port 8001
    ```
    On the first run, the server will download the `model.onnx` file (approx. 14MB) and cache it.

### API Usage

Send a `POST` request with a multipart form containing the image file.

**Example with `curl`:**
```bash
curl -X POST -F "file=@/path/to/your/image.jpg" http://localhost:8001/api/classify
```

**Success Response (`200 OK`):**
```json
{
  "class_": "water bottle",
  "confidence": 0.9749,
  "model_version": "taco-mobilenet-v1",
  "inference_ms": 326
}
```

## Training a New Model (Track B)

The project is structured to support training a custom model on the TACO dataset. The necessary skeleton files are in the `/ml/training` directory.

### Steps to Train

1.  **Prepare the dataset:** Download the [TACO dataset](http://taco-dataset.org/) and prepare it according to the structure and classes defined in `config/classes.yaml`.
2.  **Install training dependencies:**
    ```bash
    pip install -r api/requirements-ml.txt
    ```
3.  **Implement and run the training script:**
    The `ml/training/train.py` script needs to be fully implemented based on the user's specification. Once implemented, you would run:
    ```bash
    python ml/training/train.py
    ```
4.  **Implement and run the export script:**
    After training, the `ml/training/export_onnx.py` script (which also needs to be implemented) will convert the best PyTorch model (`.pth`) to the ONNX format.
    ```bash
    python ml/training/export_onnx.py
    ```
5.  **Update Artifacts:**
    -   Host the new `model.onnx` at an accessible URL and update the `MODEL_URL` in `api/app_fastapi.py`.
    -   Update `ml/artifacts/labels.json` with your custom class list (which should match `config/classes.yaml`).
    -   Update `ml/artifacts/VERSION` with a new, descriptive model version string.
