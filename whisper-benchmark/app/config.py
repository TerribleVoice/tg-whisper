from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field
from whisper_model import WhisperXConfig

DEFAULT_CONFIG_PATH = Path("config.json")


class BenchmarkWhisperConfig(WhisperXConfig):
    config_name: str = Field(
        ..., description="Имя конфигурации для формирования названий файлов"
    )
    audio_batch_size: int = Field(1, description="Количество файлов в одном батче")


class DatasetConfig(BaseModel):
    use_dataset: bool = Field(
        False, description="Использовать датасет вместо локальных файлов"
    )
    dataset_path: str = Field(
        "fsicoli/common_voice_17_0", description="Путь к датасету в HuggingFace"
    )
    dataset_name: str = Field("ru", description="Имя датасета")
    dataset_split: str = Field(
        "train", description="Разделение датасета (train, test, validation)"
    )
    dataset_cache_dir: str = Field(
        "C:/Datasets/Common_Voice_Rus",
        description="Директория для кеширования датасета",
    )
    dataset_limit: int = Field(
        5000, description="Максимальное количество записей из датасета"
    )


class LocalDatasetConfig(BaseModel):
    path: Path = Field(..., description="Путь к директории с локальным датасетом")
    limit: int | None = Field(
        None, description="Максимальное количество записей из датасета"
    )
    shuffle: bool = Field(False, description="Перемешать записи")


class BenchmarkConfig(BaseModel):
    whisper_configs: List[BenchmarkWhisperConfig] = Field(
        ..., description="Список конфигураций для тестирования"
    )
    dataset: Optional[DatasetConfig] = Field(
        None, description="Настройки для работы с датасетом"
    )
    local_dataset: Optional[LocalDatasetConfig] = Field(
        None, description="Настройки для работы с локальным датасетом"
    )
    results_path: Path = Field(
        ..., description="Путь к директории для сохранения результатов"
    )
    repeat_count: int = Field(1, description="Количество повторов тестирования")
