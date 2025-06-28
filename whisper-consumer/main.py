import logging
import tempfile
from pathlib import Path

import dramatiq
import httpx
import librosa
import transcription
from batch_processor import BatchProcessor
from batch_task import BatchTask
from config import (
    RABBITMQ_URL,
    TASK_QUEUE_NAME,
    WHISPER_CONFIG_JSON_PATH,
)
from dramatiq.brokers.rabbitmq import RabbitmqBroker
from dramatiq.middleware import AsyncIO
from whisper_model import WhisperXModel
from whisper_model.config import WhisperXConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

whisper_config = WhisperXConfig.from_json(WHISPER_CONFIG_JSON_PATH)  # type: ignore

broker = RabbitmqBroker(url=RABBITMQ_URL)
logging.info(f"whisper-consumer: Используется RabbitmqBroker для Dramatiq: {RABBITMQ_URL}")
broker.add_middleware(AsyncIO())
dramatiq.set_broker(broker)

whisper_model_instance = WhisperXModel(config=whisper_config)
batch_processor = BatchProcessor(whisper_model_instance, broker)


async def download_audio_file(file_url: str) -> Path:
    async with httpx.AsyncClient() as client:
        response = await client.get(file_url, timeout=30.0)
        response.raise_for_status()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".oga") as tmp_file:
        tmp_file.write(response.content)
        return Path(tmp_file.name)


def get_audio_duration(file_path: Path) -> float:
    try:
        duration = librosa.get_duration(path=file_path)
        return duration
    except Exception as e:
        logging.warning(f"Не удалось определить длительность файла {file_path}: {e}")
        return 60.0  # Fallback на 1 минуту


@dramatiq.actor(queue_name=TASK_QUEUE_NAME, actor_name=TASK_QUEUE_NAME)
async def transcribe_audio_task(file_url: str, chat_id: int, message_date: str):
    try:
        file_path = await download_audio_file(file_url)
        logging.info(f"Файл загружен: {file_path}")

        duration = get_audio_duration(file_path)
        logging.info(f"Длительность файла: {duration:.1f}s")

        task = BatchTask(file_path=file_path, chat_id=chat_id, message_date=message_date, audio_duration=duration)
        await batch_processor.add_task(task)

    except Exception as e:
        logging.exception(f"Ошибка загрузки файла {file_url}: {e}")
        await transcription.send_result(broker, chat_id, None, f"Ошибка загрузки файла: {str(e)}")


if __name__ == "__main__":
    logging.error("Для запуска воркеров whisper-consumer используйте команду:")
    logging.error("dramatiq whisper-consumer.main")
    logging.error("Убедитесь, что RabbitMQ сервер запущен и доступен, и переменные окружения настроены.")
