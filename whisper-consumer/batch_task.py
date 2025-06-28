from dataclasses import dataclass
from pathlib import Path


@dataclass
class BatchTask:
    file_path: Path
    chat_id: int
    message_date: str
    audio_duration: float
