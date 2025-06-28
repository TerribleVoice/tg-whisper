import asyncio
import logging
import os

import dramatiq
from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from aiohttp import web
from dotenv import load_dotenv
from dramatiq import Message as DramatiqMessage
from dramatiq.brokers.rabbitmq import RabbitmqBroker

load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

TOKEN = os.getenv("BOT_TOKEN")
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")

env_prefix = "TG_BOT_BACKEND_"
WEBHOOK_PATH = os.getenv(f"{env_prefix}WEBHOOK_PATH", "/webhook")
PORT = int(os.getenv(f"{env_prefix}PORT", 8080))
USE_POLLING = os.getenv(f"{env_prefix}USE_POLLING", "true") == "true"
TASK_QUEUE_NAME = os.getenv("TASK_QUEUE_NAME")

if not TOKEN or not TASK_QUEUE_NAME or not RABBITMQ_URL:
    raise ValueError(
        f"{TOKEN=}, {TASK_QUEUE_NAME=} и {RABBITMQ_URL=}, какая-то переменная окружения не установлена."
    )

broker = RabbitmqBroker(url=RABBITMQ_URL)
logging.info(
    f"Telegram-bot-backend: Используется RabbitmqBroker для Dramatiq: {RABBITMQ_URL}"
)
dramatiq.set_broker(broker)

bot = Bot(TOKEN)
dp = Dispatcher()


@dp.message()
async def produce_to_processing(msg: Message):
    if msg.voice:
        file = await bot.get_file(msg.voice.file_id)
        file_path = file.file_path
        if not file_path:
            await msg.answer("Ошибка: file_path не найден.")
            return

        file_url = f"https://api.telegram.org/file/bot{TOKEN}/{file_path}"
        chat_id = msg.chat.id
        message_date_iso = msg.date.isoformat()

        logging.info(
            f"Отправка задачи в очередь '{TASK_QUEUE_NAME}' для актора '{TASK_QUEUE_NAME}' для файла {file_path} из чата {chat_id}"
        )

        message_payload = {
            "file_url": file_url,
            "chat_id": chat_id,
            "message_date": message_date_iso,
        }

        task_message = DramatiqMessage(
            queue_name=TASK_QUEUE_NAME,  # type: ignore
            actor_name=TASK_QUEUE_NAME,  # type: ignore
            args=(),
            kwargs=message_payload,
            options={},
        )
        broker.enqueue(task_message)

        await msg.answer("Ваше голосовое сообщение принято в обработку.")
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
    await dp.start_polling(
        bot,
        allowed_updates=["message", "edited_message", "callback_query"],
        skip_updates=True,
    )


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
