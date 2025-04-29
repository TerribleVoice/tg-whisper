import os
import tempfile
import logging
from typing import Optional

from .models import WhisperModel

logger = logging.getLogger("whisper-server")


class TranscriptionService:
    """Сервис для транскрипции аудио."""

    def __init__(self, model_name: str = "base"):
        """
        Инициализация сервиса транскрипции.

        Args:
            model_name: Название модели Whisper
        """
        logger.debug(f"Инициализация сервиса транскрипции с моделью: {model_name}")
        self.model = WhisperModel(model_name)

    async def transcribe_audio(self, audio_data: bytes, language: Optional[str] = None) -> str:
        """
        Транскрибировать аудио из байтов.

        Args:
            audio_data: Байты аудиофайла
            language: Язык аудио (если None, будет определен автоматически)

        Returns:
            Текст транскрипции
        """
        logger.debug(f"Начало транскрипции аудио размером {len(audio_data)} байт")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_path = temp_file.name
            logger.debug(f"Аудио сохранено во временный файл: {temp_path}")

        try:
            logger.debug("Запуск процесса транскрипции")
            text = self.model.transcribe(temp_path, language)
            logger.debug(f"Транскрипция завершена, получен текст длиной {len(text)} символов")
            return text
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                logger.debug(f"Временный файл удален: {temp_path}")
