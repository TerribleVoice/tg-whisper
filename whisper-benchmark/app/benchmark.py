import logging
import os
import time
from pathlib import Path
from typing import Any, Dict

import audioread
import numpy as np
import soundfile as sf

from .config import Config
from .loaders import DatasetLoader, LocalDatasetLoader
from .models import FasterWhisperModel, WhisperXModel
from .utils import AudioProcessor, ResultsAnalyzer, TranscriptionMetrics
from .utils.gpu_monitor import GPUMonitor

logger = logging.getLogger("whisper-benchmark")


class Benchmark:
    def __init__(self, config: Config):
        self.config = config
        self.transcriber = FasterWhisperModel()
        self.analyzer = ResultsAnalyzer(config.output_dir)
        self.gpu_monitor = GPUMonitor()
        os.makedirs(config.output_dir, exist_ok=True)

    def _prepare_models(self):
        self.transcriber.preload_models(self.config.whisper_configs)
        self.transcriber.unload_all_models()

    def _get_audio_files_with_transcriptions(self) -> Dict[Path, str]:
        if self.config.dataset_config is None and self.config.samples_dir is None:
            raise ValueError("samples_dir и dataset_config не установлены")
        if self.config.dataset_config:
            dataset_config = self.config.dataset_config
            loader = DatasetLoader(
                dataset_path=dataset_config.dataset_path,
                dataset_name=dataset_config.dataset_name,
                dataset_split=dataset_config.dataset_split,
                cache_dir=dataset_config.dataset_cache_dir,
                dataset_limit=dataset_config.dataset_limit,
            )
            return loader.load_dataset()
        else:
            if self.config.samples_dir is None or self.config.references_path is None:
                raise ValueError("samples_dir и references_path не установлены")
            loader = LocalDatasetLoader(self.config.samples_dir, self.config.references_path)
            return loader.load_dataset()

    @staticmethod
    def _get_audio_duration(audio_path: Path) -> float:
        try:
            with sf.SoundFile(str(audio_path)) as f:
                return len(f) / f.samplerate
        except Exception:
            try:
                with audioread.audio_open(str(audio_path)) as f:
                    return f.duration
            except Exception as e:
                raise RuntimeError(f"Не удалось определить длительность файла {audio_path}: {e}")

    def run(self) -> Dict[str, Dict[str, Any]]:
        audio_paths_with_transcriptions = self._get_audio_files_with_transcriptions()
        results = {}
        self._prepare_models()

        for i, whisper_config in enumerate(self.config.whisper_configs):
            config_name = whisper_config.name or f"config_{i + 1}_{whisper_config.model_name}"
            logger.info(f"Тестирование конфигурации {i + 1}/{len(self.config.whisper_configs)}: {config_name}")

            config_results = {"config": whisper_config.model_dump(), "files": {}}
            if whisper_config.library == "whisperx":
                self.transcriber = WhisperXModel()
            else:
                self.transcriber = FasterWhisperModel()

            self.transcriber.set_model(whisper_config)
            config_results["files"] = self._process_files(whisper_config, audio_paths_with_transcriptions)

            self._save_results({config_name: config_results}, config_name)
            if i < len(self.config.whisper_configs) - 1:
                self.transcriber.unload_all_models()

            results[config_name] = config_results

        self.analyzer.analyze_results(results)

        return results

    def _process_files(self, whisper_config, audio_paths_with_transcriptions: Dict[Path, str]) -> Dict[str, Any]:
        results = {}
        for audio_path, reference in audio_paths_with_transcriptions.items():
            file_name = audio_path.name
            times = []
            hypotheses = []
            gpu_stats_list = []

            for _ in range(self.config.repeat_count):
                t0 = time.time()
                self.gpu_monitor.start()
                hypothesis = self._perform(whisper_config, audio_path)
                gpu_stats = self.gpu_monitor.stop()
                elapsed = time.time() - t0
                times.append(elapsed)
                hypotheses.append(hypothesis)
                gpu_stats_list.append(gpu_stats)

            all_metrics = [TranscriptionMetrics.calculate_all(reference, h) for h in hypotheses]
            avg_metrics = {key: float(np.mean([m[key] for m in all_metrics])) for key in all_metrics[0].keys()}
            avg_time = float(np.mean(times))
            duration = self._get_audio_duration(audio_path)
            if duration <= 0:
                raise ValueError(f"Длительность аудиофайла {audio_path} не может быть меньше или равна 0")
            processing_speed = round(duration / avg_time, 2)

            avg_gpu_metrics = {}
            if gpu_stats_list:
                for key in gpu_stats_list[0].keys():
                    avg_gpu_metrics[key] = round(float(np.mean([stats[key] for stats in gpu_stats_list])), 2)
            else:
                avg_gpu_metrics = {"max_memory_used_mb": 0, "avg_utilization": 0}

            results[file_name] = {
                "reference": reference,
                "hypotheses": hypotheses,
                "duration": duration,
                "processing_speed": processing_speed,
                "metrics": avg_metrics,
                "processing_time": round(sum(times), 3),
                "gpu_metrics": avg_gpu_metrics,
            }
        return results

    def _perform(self, whisper_config, audio_file) -> str | None:
        audio_path = audio_file
        if whisper_config.audio_speed is not None and whisper_config.audio_speed != 1.0:
            logger.debug(f"Ускорение аудио в {whisper_config.audio_speed} раз: {audio_file}")
            audio_path = AudioProcessor.speed_up_audio(audio_file, whisper_config.audio_speed)

        try:
            result = self.transcriber.transcribe(audio_path, whisper_config)
            return result["text"]
        except Exception as e:
            logger.error(f"Ошибка при транскрипции {audio_file}: {str(e)}")
            raise e

    def _save_results(self, results: Dict[str, Any], config_name: str) -> None:
        cfg = list(results.values())[0]["config"]
        model = cfg["model_name"]
        compute = cfg.get("compute_type") or "default"
        self.analyzer.save_results(results, model, compute, config_name)
