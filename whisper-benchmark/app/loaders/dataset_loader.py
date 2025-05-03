import logging
from pathlib import Path
from typing import Dict

from datasets import Dataset, load_dataset


logger = logging.getLogger("whisper-benchmark")


class DatasetLoader:
    def __init__(self, dataset_path: str, dataset_name: str, dataset_split: str, cache_dir: str, dataset_limit: int = 5000):
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.cache_dir = cache_dir
        self.dataset_limit = dataset_limit

    def load_dataset(self) -> Dict[Path, str]:
        logger.info(f"Загрузка датасета {self.dataset_path}/{self.dataset_name}, разделение: {self.dataset_split}")

        dataset = load_dataset(
            self.dataset_path, self.dataset_name, split=self.dataset_split, cache_dir=self.cache_dir, trust_remote_code=True
        )

        if not isinstance(dataset, Dataset):
            raise ValueError(f"{type(dataset)} не поддерживается")

        if self.dataset_limit:
            dataset = dataset.select(range(min(self.dataset_limit, len(dataset))))

        logger.info(f"Загружен датасет: {type(dataset)}")
        logger.info(f"Количество примеров: {len(dataset)}")

        def extract_audio_and_sentence(batch):
            paths = [a["path"] for a in batch["audio"]]
            return {"audio_paths": paths, "sentences": batch["sentence"]}

        mapped = dataset.map(extract_audio_and_sentence, batched=True, remove_columns=dataset.column_names)
        filtered = [(Path(p), s) for p, s in zip(mapped["audio_paths"], mapped["sentences"]) if s and any(c.isalpha() for c in s)]
        audio_paths_with_transcriptions = dict(filtered)

        logger.info(f"Загружено {len(audio_paths_with_transcriptions)} примеров из датасета")

        return audio_paths_with_transcriptions
