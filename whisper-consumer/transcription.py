import logging
import tempfile
from pathlib import Path
from typing import Optional

import httpx
from config import RESULTS_QUEUE_NAME, TASK_QUEUE_NAME
from dramatiq import Message
from whisper_model import WhisperXModel


async def transcribe_single_audio(file_url: str, chat_id: int, message_date: str, whisper_model: WhisperXModel, broker):
    transcript = None
    error_message = None
    audio_file_path_temp: Path | None = None

    try:
        logging.info(f"[{TASK_QUEUE_NAME}] Загрузка файла из {file_url}...")
        async with httpx.AsyncClient() as client:
            response = await client.get(file_url, timeout=30.0)
            response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".oga") as tmp_audio_file:
            tmp_audio_file.write(response.content)
            audio_file_path_temp = Path(tmp_audio_file.name)

        logging.info(f"[{TASK_QUEUE_NAME}] Файл сохранен во временный файл: {audio_file_path_temp}")

        logging.info(f"[{TASK_QUEUE_NAME}] Начало транскрипции файла {audio_file_path_temp}...")
        transcription_result = whisper_model.transcribe(audio_path=audio_file_path_temp)

        transcript = transcription_result.text
        logging.info(f"[{TASK_QUEUE_NAME}] Транскрипция завершена. Результат: {transcript[:100]}...")

    except httpx.HTTPStatusError as e:
        logging.error(f"[{TASK_QUEUE_NAME}] Ошибка загрузки файла: {e}")
        error_message = f"Ошибка загрузки файла: {e.response.status_code}"
    except Exception as e:
        logging.exception(f"[{TASK_QUEUE_NAME}] Ошибка во время транскрипции: {e}")
        error_message = f"Внутренняя ошибка сервера при транскрипции: {str(e)}"
    finally:
        if audio_file_path_temp:
            audio_file_path_temp.unlink(missing_ok=True)
            logging.info(f"[{TASK_QUEUE_NAME}] Временный файл {audio_file_path_temp} удален.")

    await send_result(broker, chat_id, transcript, error_message)


async def send_result(broker, chat_id: int, transcript: Optional[str], error: Optional[str]):
    message_data = {
        "original_chat_id": chat_id,
        "transcript": transcript,
        "error": error,
    }
    result_message = Message(
        queue_name=RESULTS_QUEUE_NAME,  # type: ignore
        actor_name=RESULTS_QUEUE_NAME,  # type: ignore
        args=(),
        kwargs=message_data,
        options={},
    )
    broker.enqueue(result_message)
    logging.info(f"[{TASK_QUEUE_NAME}] Результат отправлен в очередь '{RESULTS_QUEUE_NAME}'.")
