import json
import os
from dataclasses import dataclass

import numpy as np
import requests
from tensorflow.keras.preprocessing import image

# Default path; override with MODEL_PATH env var
_DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "trashnet_model.keras")
_DEFAULT_CLASSES_PATH = os.path.join(os.path.dirname(__file__), "model", "classes.json")

# Fallback class order (matches TrashNet alphabetical labels)
_FALLBACK_CLASSES = ["Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash"]


@dataclass(frozen=True)
class ExternalClassifierConfig:
    api_url: str | None
    api_token: str | None
    timeout: int = 20


def get_external_classifier_config() -> ExternalClassifierConfig:
    return ExternalClassifierConfig(
        api_url=os.environ.get("EXTERNAL_CLASSIFIER_API_URL"),
        api_token=os.environ.get("EXTERNAL_CLASSIFIER_API_TOKEN"),
        timeout=int(os.environ.get("EXTERNAL_CLASSIFIER_TIMEOUT", "20")),
    )


def classify_with_external_api(
    image_path: str,
    config: ExternalClassifierConfig,
) -> str:
    """Classify image using an external HTTP API.

    Expected API response formats:
    - {"classification": "Plastic"}
    - {"label": "Plastic"}
    """
    if not config.api_url:
        raise RuntimeError(
            "Classification model is unavailable and EXTERNAL_CLASSIFIER_API_URL is not configured."
        )

    headers = {}
    if config.api_token:
        headers["Authorization"] = f"Bearer {config.api_token}"

    with open(image_path, "rb") as image_file:
        response = requests.post(
            config.api_url,
            files={"file": image_file},
            headers=headers,
            timeout=config.timeout,
        )

    if response.status_code >= 400:
        raise RuntimeError(
            f"External classifier API error: {response.status_code}"
        )

    payload = response.json()
    classification = payload.get("classification") or payload.get("label")
    if not classification:
        raise RuntimeError("External classifier API returned invalid payload.")

    return classification


def _load_classes(classes_path: str) -> list[str]:
    try:
        with open(classes_path) as f:
            return [c.capitalize() for c in json.load(f)]
    except (FileNotFoundError, json.JSONDecodeError):
        return _FALLBACK_CLASSES


def load_trashnet_model():
    """Load trained model from disk.

    Returns None if model file is not found (use `python train.py` to generate it).
    """
    model_path = os.environ.get("MODEL_PATH", _DEFAULT_MODEL_PATH)

    if not os.path.exists(model_path):
        print(
            f"Model file not found at '{model_path}'. "
            "Run `python train.py` to train and save the model. "
            "Falling back to external API if configured."
        )
        return None

    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from '{model_path}'.")
        return model
    except Exception as exc:
        print(f"Failed to load model from '{model_path}': {exc}")
        return None


# Load model once at startup
TRASHNET_MODEL = load_trashnet_model()

# Load class names alongside model
_CLASSES_PATH = os.environ.get(
    "CLASSES_PATH",
    os.path.join(os.path.dirname(os.environ.get("MODEL_PATH", _DEFAULT_MODEL_PATH)), "classes.json"),
)
TRASH_CLASSES = _load_classes(_CLASSES_PATH)


def classify_with_local_model(image_path: str, model) -> str:
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array, verbose=0)
    return TRASH_CLASSES[np.argmax(prediction)]


def classify_trash(image_path: str, model=None) -> str:
    """Classify image using local model, or external API fallback."""
    model = TRASHNET_MODEL if model is None else model

    if model is not None:
        return classify_with_local_model(image_path, model)

    external_config = get_external_classifier_config()
    return classify_with_external_api(image_path, external_config)
