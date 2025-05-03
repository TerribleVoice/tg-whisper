import gc
import logging
from pathlib import Path
from typing import Any, Dict

import torch
import whisperx

from ..config import WhisperConfig
from ..utils.audio_processor import AudioProcessor
from ..utils.suppress_std import SuppressStd

logger = logging.getLogger("whisper-benchmark")


class WhisperXModel:
    def __init__(self):
        self.models = {}

    def set_model(self, config: WhisperConfig):
        model_key = f"{config.model_name}_{config.compute_type or 'default'}"
        device = config.device
        compute_type = config.compute_type

        model = whisperx.load_model(whisper_arch=config.model_name, device=device, compute_type=compute_type)
        logger.info(f"Используется модель WhisperX {config.model_name} на устройстве {device} с типом вычислений {compute_type}")
        self.models[model_key] = model

    def transcribe(self, audio_path: Path, config: WhisperConfig) -> Dict[str, Any]:
        model_key = f"{config.model_name}_{config.compute_type or 'default'}"
        model = self.models[model_key]

        temp_audio_path = None
        processed_audio_path = audio_path

        transcription_options = {}

        if config.batch_size:
            transcription_options["batch_size"] = config.batch_size

        try:
            audio = whisperx.load_audio(str(processed_audio_path))
            with SuppressStd(logger):
                result = model.transcribe(audio, **transcription_options)

            language = result["language"]
            result_text = " ".join(segment["text"] for segment in result["segments"])

            return {
                "text": result_text,
                "language": language,
                "audio_speed": config.audio_speed,
            }

        finally:
            AudioProcessor.cleanup_temp_file(temp_audio_path, audio_path)

    def preload_models(self, configs: list[WhisperConfig]):
        for config in configs:
            self.set_model(config)

    def unload_all_models(self):
        self.models.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
