import json
from pathlib import Path
import random
from typing import Dict
from app.config import LocalDatasetConfig

import logging

logger = logging.getLogger("whisper-benchmark")


class LocalDatasetLoader:
    def __init__(self, local_dataset_config: LocalDatasetConfig):
        self.samples_dir = local_dataset_config.path / "samples"
        self.references_path = local_dataset_config.path / "references.json"
        self.limit = local_dataset_config.limit
        self.shuffle = local_dataset_config.shuffle

    def load_dataset(self) -> Dict[Path, str]:
        if not self.samples_dir.exists():
            raise FileNotFoundError(
                f"Директория с примерами аудио не найдена: {self.samples_dir}"
            )

        audio_extensions = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".opus"}
        audio_files = [
            f
            for f in self.samples_dir.glob("*")
            if f.is_file() and f.suffix.lower() in audio_extensions
        ]
        if self.limit:
            audio_files = audio_files[: self.limit]
            if self.shuffle:
                random.shuffle(audio_files)

        if not audio_files:
            raise ValueError(f"Не найдено аудиофайлов в директории: {self.samples_dir}")
        if not self.references_path.exists():
            raise FileNotFoundError(
                f"Файл с эталонными транскрипциями не найден: {self.references_path}"
            )

        with open(self.references_path, "r", encoding="utf-8") as f:
            references = json.load(f)

        audio_paths_with_transcriptions = {}
        for audio_file in audio_files:
            if audio_file.name not in references:
                raise ValueError(
                    f"Нет эталонной транскрипции для файла: {audio_file.name}"
                )

            audio_paths_with_transcriptions[audio_file] = references[audio_file.name]
        logger.info(
            f"Найдено {len(audio_paths_with_transcriptions)} аудиофайлов с расшифровками"
        )
        return audio_paths_with_transcriptions
