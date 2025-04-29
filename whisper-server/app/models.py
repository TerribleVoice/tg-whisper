from typing import Optional
import logging
import os
from faster_whisper import WhisperModel as FasterWhisperModel
import torch

from .config import get_whisper_config

logger = logging.getLogger("whisper-server")


class WhisperModel:
    """Класс для работы с моделью Whisper для транскрипции аудио."""

    def __init__(self):
        config = get_whisper_config()
        self.model_name = config.get("model_name", "base")

        device = "cuda"
        if not torch.cuda.is_available():
            logger.warning("CUDA is not available, using CPU")
            device = "cpu"

        self.device = device

        compute_type_config: dict[str, str] = config.get("compute_type", {})
        compute_type = compute_type_config.get(device, "float16" if device == "cuda" else "int8")

        download_root = config.get("download_root")
        if download_root is None:
            download_root = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")

        logger.info(f"Загрузка faster-whisper модели '{self.model_name}' на устройство {device}")
        logger.info(f"Тип вычислений: {compute_type}, кэш: {download_root}")

        self.model = FasterWhisperModel(self.model_name, device=device, compute_type=compute_type, download_root=download_root)

        self.beam_size = config.get("beam_size", 5)
        self.best_of = config.get("best_of", 5)

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        """
        Транскрибирование аудиофайла.

        Args:
            audio_path: Путь к аудиофайлу
            language: Язык аудио (если None, будет определен автоматически)

        Returns:
            Текст транскрипции
        """
        transcription_options = {}

        if language:
            transcription_options["language"] = language
            logger.info(f"Транскрибирование аудио с указанным языком: {language}")
        else:
            logger.info("Транскрибирование аудио с автоопределением языка")

        segments, info = self.model.transcribe(audio_path, beam_size=self.beam_size, best_of=self.best_of, **transcription_options)

        logger.info(f"Определенный язык: {info.language} (вероятность: {info.language_probability:.2f})")

        result_text = " ".join(segment.text for segment in segments)

        if not result_text:
            logger.error("Ошибка транскрипции: результат не содержит текст")
            raise ValueError("Failed to transcribe audio")

        return result_text
