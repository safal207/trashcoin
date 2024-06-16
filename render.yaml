# trashcoin
services:
  - type: web
    name: telegram-bot
    env: python
    buildCommand: "pip install -r requirements.txt"
    startCommand: "python bot.py"
    plan: free
    envVars:
      - key: YOUR_TELEGRAM_BOT_TOKEN
        value: "ваш_токен_бота"
