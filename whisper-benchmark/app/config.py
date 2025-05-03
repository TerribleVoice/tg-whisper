import json
from pathlib import Path
from typing import List, Optional, Union

from pydantic import BaseModel, Field


class WhisperConfig(BaseModel):
    model_name: str = Field(..., description="Название модели Whisper (tiny, base, small, medium, large)", min_length=4)
    compute_type: str = Field("float16", description="Тип вычислений (float16)", min_length=4)
    device: str = Field(..., description="Устройство для выполнения вычислений (cpu, cuda)", min_length=3)
    library: str = Field("faster_whisper", description="Library to use: faster_whisper or whisperx")
    beam_size: Optional[int] = Field(None, description="Размер луча для декодирования")
    best_of: Optional[int] = Field(None, description="Количество образцов для выбора лучшего")
    batch_size: Optional[int] = Field(None, description="Размер batch для инференса")
    name: Optional[str] = Field(None, description="Имя конфигурации для формирования названий файлов")
    audio_speed: Optional[float] = Field(None, description="Коэффициент ускорения аудио (например, 1.5 для ускорения в 1.5 раза)")
    flash_attention: bool = Field(False, description="Использовать Flash Attention")


class DatasetConfig(BaseModel):
    use_dataset: bool = Field(False, description="Использовать датасет вместо локальных файлов")
    dataset_path: str = Field("fsicoli/common_voice_17_0", description="Путь к датасету в HuggingFace")
    dataset_name: str = Field("ru", description="Имя датасета")
    dataset_split: str = Field("train", description="Разделение датасета (train, test, validation)")
    dataset_cache_dir: str = Field("C:/Datasets/Common_Voice_Rus", description="Директория для кеширования датасета")
    dataset_limit: int = Field(5000, description="Максимальное количество записей из датасета")


class BenchmarkConfig(BaseModel):
    whisper_configs: List[WhisperConfig] = Field(..., description="Список конфигураций для тестирования")
    source_path: Path = Field(..., description="Путь к директории со структурами 'samples', 'results' и 'references.json'")
    repeat_count: int = Field(1, description="Количество повторений для каждой конфигурации")
    dataset: Optional[DatasetConfig] = Field(None, description="Настройки для работы с датасетом")

    @property
    def samples_dir(self) -> Path:
        return self.source_path / "samples"

    @property
    def references_path(self) -> Path:
        return self.source_path / "references.json"

    @property
    def output_dir(self) -> Path:
        return self.source_path / "results"


class Config:
    DEFAULT_CONFIG_PATH = Path("config.json")

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        if config_path is None:
            config_path = self.DEFAULT_CONFIG_PATH
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        if "source_path" in config_data:
            config_data["source_path"] = Path(config_data["source_path"])

        self.config = BenchmarkConfig(**config_data)

    @property
    def whisper_configs(self) -> List[WhisperConfig]:
        return self.config.whisper_configs

    @property
    def samples_dir(self) -> Path:
        return self.config.samples_dir

    @property
    def references_path(self) -> Path:
        return self.config.references_path

    @property
    def output_dir(self) -> Path:
        return self.config.output_dir

    @property
    def repeat_count(self) -> int:
        return self.config.repeat_count

    @property
    def dataset_config(self) -> Optional[DatasetConfig]:
        return self.config.dataset
