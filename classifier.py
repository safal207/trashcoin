import os
from dataclasses import dataclass

import numpy as np
import requests
from tensorflow.keras.preprocessing import image


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


# Загрузка модели (заглушка)
def load_trashnet_model():
    """Load local model if available.

    Current repository does not include trained weights.
    """
    print("Model loading is stubbed out as the model file is unavailable.")
    return None


# Загружаем модель один раз при старте
TRASHNET_MODEL = load_trashnet_model()


def classify_with_local_model(image_path: str, model) -> str:
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    trash_classes = [
        "Glass",
        "Paper",
        "Cardboard",
        "Plastic",
        "Metal",
        "Trash",
    ]
    return trash_classes[np.argmax(prediction)]


def classify_trash(image_path: str, model=None) -> str:
    """Classify image using local model, or external API fallback."""
    model = TRASHNET_MODEL if model is None else model

    if model is not None:
        return classify_with_local_model(image_path, model)

    external_config = get_external_classifier_config()
    return classify_with_external_api(image_path, external_config)
