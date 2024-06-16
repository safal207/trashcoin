import os
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Путь для сохранения загруженных фото
PHOTO_SAVE_PATH = './photos'

# Убедимся, что директория для фото существует
if not os.path.exists(PHOTO_SAVE_PATH):
    os.makedirs(PHOTO_SAVE_PATH)

# Загрузка модели  (предварительно обученной модели)
def load_trashnet_model():
    model_url = 'https://github.com/garythung/trashnet/raw/masteret-main/model/garbage_deploy.h5'
    model_path = tf.keras.utils.get_file('garbage_deploy.h5', model_url, cache_subdir='models')
    return tf.keras.models.load_model(model_path)

# Функция для классификации мусора на фотографии
def classify_trash(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.

    prediction = model.predict(img_array)
    trash_classes = ['Glass', 'Paper', 'Cardboard', 'Plastic', 'Metal', 'Trash']
    trash_class = trash_classes[np.argmax(prediction)]

    return trash_class

# Обработчик команды /start
async def start(update: Update, context: CallbackContext) -> None:
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Привет! Отправь мне свою геолокацию.")

# Обработчик геолокации
async def location(update: Update, context: CallbackContext) -> None:
    user_location = update.message.location
    latitude = user_location.latitude
    longitude = user_location.longitude
    await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Твоя геолокация: Широта - {latitude}, Долгота - {longitude}")
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Теперь отправь мне фото мусора.")

# Обработчик фотографий
async def photo(update: Update, context: CallbackContext) -> None:
    file_id = update.message.photo[-1].file_id  # Получаем file_id самого большого фото
    new_file = await context.bot.get_file(file_id)

    # Скачиваем и сохраняем файл локально
    photo_path = os.path.join(PHOTO_SAVE_PATH, f"{file_id}.jpg")
    await new_file.download(photo_path)

    # Загрузка модели TrashNet
    trashnet_model = load_trashnet_model()

    # Классификация мусора на фото
    trash_class = classify_trash(photo_path, trashnet_model)

    # Отправка сообщения с результатом
    await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Обнаруженный тип мусора: {trash_class}")

def main():
    application = Application.builder().token('7339488730:AAHEVnTQ3hpqTblbBZ0hOo6L2BhgnXiDNs8').build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.LOCATION, location))
    application.add_handler(MessageHandler(filters.PHOTO, photo))

    application.run_polling()

if __name__ == '__main__':
    main()
