from typing import Optional
import logging
import os
from faster_whisper import WhisperModel as FasterWhisperModel
import torch

logger = logging.getLogger("whisper-server")


class WhisperModel:
    """Класс для работы с моделью Whisper для транскрипции аудио."""

    def __init__(self, model_name: str = "base"):
        # Проверка доступности CUDA
        device = "cuda"
        compute_type = "float16"  # half precision для CUDA

        if not torch.cuda.is_available():
            logger.warning("CUDA is not available, using CPU")
            device = "cpu"
            compute_type = "int8"  # int8 для CPU для улучшения производительности

        self.device = device

        # Путь к кэшу моделей
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")

        logger.info(f"Загрузка faster-whisper модели '{model_name}' на устройство {device}")
        logger.info(f"Тип вычислений: {compute_type}, кэш: {cache_dir}")

        self.model = FasterWhisperModel(model_name, device=device, compute_type=compute_type, download_root=cache_dir)
        self.model_name = model_name

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        """
        Транскрибирование аудиофайла.

        Args:
            audio_path: Путь к аудиофайлу
            language: Язык аудио (если None, будет определен автоматически)

        Returns:
            Текст транскрипции
        """
        # Настройки транскрипции
        beam_size = 5
        best_of = 5

        transcription_options = {}

        if language:
            transcription_options["language"] = language
            logger.info(f"Транскрибирование аудио с указанным языком: {language}")
        else:
            logger.info("Транскрибирование аудио с автоопределением языка")

        # В faster-whisper транскрипция возвращает сегменты
        segments, info = self.model.transcribe(audio_path, beam_size=beam_size, best_of=best_of, **transcription_options)

        logger.info(f"Определенный язык: {info.language} (вероятность: {info.language_probability:.2f})")

        # Объединяем текст всех сегментов
        result_text = " ".join(segment.text for segment in segments)

        if not result_text:
            logger.error("Ошибка транскрипции: результат не содержит текст")
            raise ValueError("Failed to transcribe audio")

        return result_text
