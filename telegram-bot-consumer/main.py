import logging
import os

import dramatiq
from aiogram import Bot
from dotenv import load_dotenv
from dramatiq.brokers.rabbitmq import RabbitmqBroker
from dramatiq.middleware import AsyncIO

load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

TOKEN = os.getenv("BOT_TOKEN")
RESULTS_QUEUE_NAME = os.getenv("RESULTS_QUEUE_NAME")
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
if not TOKEN or not RESULTS_QUEUE_NAME or not RABBITMQ_URL:
    raise ValueError(f"{TOKEN=}, {RESULTS_QUEUE_NAME=} и {RABBITMQ_URL=}, какая-то переменная окружения не установлена.")


broker = RabbitmqBroker(url=RABBITMQ_URL)
logging.info(f"Telegram-bot-consumer: Используется RabbitmqBroker для Dramatiq: {RABBITMQ_URL}")

broker.add_middleware(AsyncIO())
dramatiq.set_broker(broker)

bot = Bot(TOKEN)


@dramatiq.actor(queue_name=RESULTS_QUEUE_NAME, actor_name=RESULTS_QUEUE_NAME)
async def handle_transcription_result(original_chat_id: int, transcript: str, error: str | None = None):
    actor_name_for_logs = handle_transcription_result.actor_name
    logging.info(f"[{actor_name_for_logs}] Получен результат для чата {original_chat_id}:")
    if error:
        logging.error(f"Ошибка транскрипции: {error}")
        await bot.send_message(original_chat_id, f"Произошла ошибка при транскрипции: {error}")
    else:
        logging.info(f"Транскрипция: {transcript}")
        await bot.send_message(original_chat_id, transcript)


if __name__ == "__main__":
    logging.error("Для запуска воркеров telegram-bot-consumer используйте команду:")
    logging.error("dramatiq telegram-bot-consumer.main")
    logging.error("Убедитесь, что RabbitMQ сервер запущен и доступен, и переменные окружения настроены.")
