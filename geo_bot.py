import os

import numpy as np
import requests
import tensorflow as tf
from telegram import Update
from telegram.ext import (
    Application,
    CallbackContext,
    CommandHandler,
    MessageHandler,
    filters,
)
from tensorflow.keras.preprocessing import image

# Путь для сохранения загруженных фото
PHOTO_SAVE_PATH = "./photos"

# Optional external inference API settings
EXTERNAL_CLASSIFIER_API_URL = os.environ.get("EXTERNAL_CLASSIFIER_API_URL")
EXTERNAL_CLASSIFIER_API_TOKEN = os.environ.get("EXTERNAL_CLASSIFIER_API_TOKEN")
EXTERNAL_CLASSIFIER_TIMEOUT = int(os.environ.get("EXTERNAL_CLASSIFIER_TIMEOUT", "20"))

# Убедимся, что директория для фото существует
if not os.path.exists(PHOTO_SAVE_PATH):
    os.makedirs(PHOTO_SAVE_PATH)


def classify_with_external_api(image_path: str) -> str:
    """Classify image using external HTTP API.

    Expected API response formats:
    - {"classification": "Plastic"}
    - {"label": "Plastic"}
    """
    if not EXTERNAL_CLASSIFIER_API_URL:
        raise RuntimeError(
            "Classification model is unavailable and EXTERNAL_CLASSIFIER_API_URL is not configured."
        )

    headers = {}
    if EXTERNAL_CLASSIFIER_API_TOKEN:
        headers["Authorization"] = f"Bearer {EXTERNAL_CLASSIFIER_API_TOKEN}"

    with open(image_path, "rb") as image_file:
        response = requests.post(
            EXTERNAL_CLASSIFIER_API_URL,
            files={"file": image_file},
            headers=headers,
            timeout=EXTERNAL_CLASSIFIER_TIMEOUT,
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
    """
    Stub function for loading the model.
    Returns None because the model is currently unavailable.
    """
    print("Model loading is stubbed out as the model file is unavailable.")
    return None


# Загружаем модель один раз при старте
TRASHNET_MODEL = load_trashnet_model()


# Функция для классификации мусора на фотографии
def classify_trash(image_path, model):
    """Classifies trash on an image with a loaded model.

    If local model is unavailable, tries EXTERNAL_CLASSIFIER_API_URL.
    """
    if model is None:
        return classify_with_external_api(image_path)

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
    trash_class = trash_classes[np.argmax(prediction)]

    return trash_class


# Обработчик команды /start
async def start(update: Update, context: CallbackContext) -> None:
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Привет! Отправь мне свою геолокацию.",
    )


# Обработчик геолокации
async def location(update: Update, context: CallbackContext) -> None:
    user_location = update.message.location
    latitude = user_location.latitude
    longitude = user_location.longitude
    message_text = (
        f"Твоя геолокация: Широта - {latitude}, Долгота - {longitude}"
    )
    await context.bot.send_message(
        chat_id=update.effective_chat.id, text=message_text
    )
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Теперь отправь мне фото мусора.",
    )


# Обработчик фотографий
async def photo(update: Update, context: CallbackContext) -> None:
    # Получаем file_id самого большого фото
    file_id = update.message.photo[-1].file_id
    new_file = await context.bot.get_file(file_id)

    # Скачиваем и сохраняем файл локально
    photo_path = os.path.join(PHOTO_SAVE_PATH, f"{file_id}.jpg")
    await new_file.download_to_drive(photo_path)

    # Классификация мусора на фото (используем уже загруженную модель)
    trash_class = classify_trash(photo_path, TRASHNET_MODEL)

    # Отправка сообщения с результатом
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=f"Обнаруженный тип мусора: {trash_class}",
    )


def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        print("Error: TELEGRAM_BOT_TOKEN environment variable not set.")
        return

    application = Application.builder().token(token).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.LOCATION, location))
    application.add_handler(MessageHandler(filters.PHOTO, photo))

    application.run_polling()


if __name__ == "__main__":
    main()
