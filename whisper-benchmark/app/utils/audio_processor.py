import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger("whisper-benchmark")


class AudioProcessor:
    @staticmethod
    def speed_up_audio(audio_path: Path, speed_factor: float) -> Path:
        if speed_factor <= 0:
            raise ValueError(f"Некорректный коэффициент ускорения: {speed_factor}, должен быть > 0")

        if speed_factor == 1.0:
            return audio_path

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_file.close()
        temp_path = Path(temp_file.name)

        try:
            filter_complex = f"atempo={speed_factor}"
            cmd = ["ffmpeg", "-y", "-i", str(audio_path), "-filter:a", filter_complex, "-c:a", "pcm_s16le", "-f", "wav", str(temp_path)]

            logger.debug(f"Запуск команды для ускорения аудио: {' '.join(cmd)}")

            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)

            logger.debug(f"Аудио успешно ускорено в {speed_factor} раз")
            return temp_path

        except subprocess.CalledProcessError as e:
            logger.error(f"Ошибка при ускорении аудио: {e.stderr}")
            return audio_path
        except Exception as e:
            logger.error(f"Неожиданная ошибка при ускорении аудио: {str(e)}")
            return audio_path

    @staticmethod
    def cleanup_temp_file(temp_path: Optional[Path], original_path: Path) -> None:
        if temp_path is not None and temp_path != original_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Не удалось удалить временный файл {temp_path}: {str(e)}")
