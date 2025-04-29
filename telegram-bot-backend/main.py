import os
import asyncio
from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from aiohttp import web
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("BOT_TOKEN")
WEBHOOK_PATH = os.getenv("WEBHOOK_PATH", "/webhook")
PORT = int(os.getenv("PORT", 8080))
USE_POLLING = os.getenv("USE_POLLING", "false").lower() == "true"

if not TOKEN:
    raise ValueError("BOT_TOKEN is not set")

bot = Bot(TOKEN)
dp = Dispatcher()


@dp.message()
async def echo(msg: Message):
    if msg.voice:
        file = await bot.get_file(msg.voice.file_id)
        file_path = file.file_path
        if not file_path:
            await msg.answer("Ошибка: file_path не найден.")
            return
        dest = f"telegram-bot-backend/voice_files/{msg.voice.file_id}.ogg"
        await bot.download_file(file_path, dest)
        await msg.answer("Voice saved.")
        return
    await msg.answer(msg.text or "(no text)")


async def on_startup(app):
    if not USE_POLLING:
        webhook_url = os.getenv("WEBHOOK_URL")
        if webhook_url:
            await bot.set_webhook(f"{webhook_url}{WEBHOOK_PATH}")
        info = await bot.get_webhook_info()
        if info.url:
            print(f"Webhook установлен: {info.url}")
        else:
            print("Вебхук не установлен")


async def on_shutdown(app):
    if not USE_POLLING:
        await bot.delete_webhook()
    await bot.session.close()


async def start_polling():
    await bot.delete_webhook()
    await dp.start_polling(bot, allowed_updates=["message", "edited_message", "callback_query"], skip_updates=True)


async def main_polling():
    print("Запуск бота в режиме polling")
    await start_polling()


def main():
    if USE_POLLING:
        asyncio.run(main_polling())
    else:
        print("Запуск бота в режиме webhook")
        app = web.Application()
        SimpleRequestHandler(dispatcher=dp, bot=bot).register(app, path=WEBHOOK_PATH)
        setup_application(app, dp, bot=bot)
        app.on_startup.append(on_startup)
        app.on_shutdown.append(on_shutdown)
        web.run_app(app, port=PORT)


if __name__ == "__main__":
    main()
