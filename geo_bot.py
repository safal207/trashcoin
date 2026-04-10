import logging
import os

from telegram import Update
from telegram.ext import (
    Application,
    CallbackContext,
    CommandHandler,
    MessageHandler,
    filters,
)

from classifier import TRASHNET_MODEL, classify_trash
from logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

PHOTO_SAVE_PATH = "./photos"

if not os.path.exists(PHOTO_SAVE_PATH):
    os.makedirs(PHOTO_SAVE_PATH)


async def start(update: Update, context: CallbackContext) -> None:
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Привет! Отправь мне свою геолокацию.",
    )


async def location(update: Update, context: CallbackContext) -> None:
    user_location = update.message.location
    latitude = user_location.latitude
    longitude = user_location.longitude
    logger.info(
        "Location received from chat_id=%s: lat=%.6f lon=%.6f",
        update.effective_chat.id, latitude, longitude,
    )
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=f"Твоя геолокация: Широта - {latitude}, Долгота - {longitude}",
    )
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Теперь отправь мне фото мусора.",
    )


async def photo(update: Update, context: CallbackContext) -> None:
    file_id = update.message.photo[-1].file_id
    new_file = await context.bot.get_file(file_id)

    photo_path = os.path.join(PHOTO_SAVE_PATH, f"{file_id}.jpg")
    await new_file.download_to_drive(photo_path)
    logger.info("Photo saved: %s", photo_path)

    try:
        trash_class = classify_trash(photo_path, TRASHNET_MODEL)
        logger.info("Photo '%s' classified as '%s'", file_id, trash_class)
    except Exception:
        logger.exception("Failed to classify photo '%s'", file_id)
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Не удалось классифицировать фото. Попробуй ещё раз.",
        )
        return

    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=f"Обнаруженный тип мусора: {trash_class}",
    )


def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN environment variable not set.")
        return

    application = Application.builder().token(token).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.LOCATION, location))
    application.add_handler(MessageHandler(filters.PHOTO, photo))

    logger.info("Bot started.")
    application.run_polling()


if __name__ == "__main__":
    main()
