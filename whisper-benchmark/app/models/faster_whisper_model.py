import logging
from pathlib import Path
from typing import Any, Dict

from faster_whisper import WhisperModel

from ..config import WhisperConfig
from ..utils.audio_processor import AudioProcessor

logger = logging.getLogger("whisper-benchmark")


class FasterWhisperModel:
    def __init__(self):
        self.models = {}

    def set_model(self, config: WhisperConfig):
        model_key = f"{config.model_name}_{config.compute_type or 'default'}"
        device = config.device
        compute_type = config.compute_type

        model = WhisperModel(model_size_or_path=config.model_name, device=device, compute_type=compute_type)
        logger.info(f"Используется модель {config.model_name} на устройстве {device} с типом вычислений {compute_type}")
        self.models[model_key] = model

    def transcribe(self, audio_path: Path, config: WhisperConfig) -> Dict[str, Any]:
        if isinstance(audio_path, str):
            audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Аудиофайл не найден: {audio_path}")

        model_key = f"{config.model_name}_{config.compute_type or 'default'}"
        model = self.models[model_key]

        temp_audio_path = None
        processed_audio_path = audio_path

        transcription_options = {}

        if config.beam_size:
            transcription_options["beam_size"] = config.beam_size
        if config.best_of:
            transcription_options["best_of"] = config.best_of
        if config.batch_size:
            transcription_options["batch_size"] = config.batch_size

        try:
            segments, info = model.transcribe(str(processed_audio_path), **transcription_options)
        finally:
            AudioProcessor.cleanup_temp_file(temp_audio_path, audio_path)

        result_text = " ".join(segment.text for segment in segments)

        result = {
            "text": result_text,
            "language": info.language,
            "language_probability": info.language_probability,
            "audio_speed": config.audio_speed,
        }

        return result

    def preload_models(self, configs: list[WhisperConfig]):
        for config in configs:
            self.set_model(config)

    def unload_all_models(self):
        self.models.clear()
        # gc.collect()
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
