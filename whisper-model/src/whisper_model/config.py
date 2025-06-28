import json
from pydantic import BaseModel, Field


class AsrOptions(BaseModel):
    beam_size: int | None = Field(None, ge=1)
    best_of: int | None = Field(None, ge=1)


class TranscribeOptions(BaseModel):
    batch_size: int | None = Field(None, ge=1)


class WhisperConfig(BaseModel):
    whisper_arch: str = Field(..., min_length=2)
    compute_type: str = Field("int8_float16", min_length=4)
    device: str = Field("cuda", min_length=3)
    asr_options: AsrOptions = Field(...)
    transcribe_options: TranscribeOptions = Field(...)
    language: str | None = Field(None, min_length=2)


class AlignConfig(BaseModel):
    language_code: str = Field("ru", min_length=2)
    device: str = Field("cuda", min_length=3)
    model_name: str | None = Field(None, min_length=5)


class PeakConfig(BaseModel):
    min_duration: float = Field(1)
    alpha: float = Field(0.18)


class SegmentationConfig(BaseModel):
    device: str = Field("cuda", min_length=3)
    model: str = Field("pyannote/segmentation", min_length=5)
    batch_size: int = Field(32)
    step: float = Field(0.75)
    peak_config: PeakConfig = Field(...)


class WhisperXConfig(BaseModel):
    whisper_config: WhisperConfig = Field(...)
    align_config: AlignConfig = Field(...)
    segmentation_config: SegmentationConfig = Field(...)
    
    @staticmethod
    def from_json(json_path: str) -> "WhisperXConfig":
        with open(json_path, "r") as f:
            config_dict = json.load(f)
        return WhisperXConfig.model_validate(config_dict)
