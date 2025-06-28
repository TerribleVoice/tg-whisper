import gc
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import audioread
import soundfile as sf
import torch
from whisper_model.config import WhisperXConfig
from whisper_model.whisperx_model import (
    TranscriptionMetrics,
    TranscriptionResult,
    WhisperXModel,
)

from .config import BenchmarkConfig, BenchmarkWhisperConfig
from .loaders import DatasetLoader, LocalDatasetLoader
from .utils import ResultsAnalyzer
from .utils.gpu_monitor import GPUMonitor
from .utils.metrics import word_error_rate

logger = logging.getLogger("whisper-benchmark")


class Benchmark:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.analyzer = ResultsAnalyzer(config.results_path)
        self.gpu_monitor = GPUMonitor()

    def _preload_models(self):
        for whisper_config in self.config.whisper_configs:
            base_whisper_config = whisper_config.model_dump(exclude={"config_name"})
            self.transcriber = WhisperXModel(WhisperXConfig(**base_whisper_config))
            gc.collect()
            torch.cuda.empty_cache()
            self.transcriber = None

    def _get_audio_files_with_transcriptions(self) -> Dict[Path, str]:
        if self.config.dataset:
            loader = DatasetLoader(self.config.dataset)
        elif self.config.local_dataset:
            loader = LocalDatasetLoader(self.config.local_dataset)
        else:
            raise ValueError("dataset и local_dataset не установлены")

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
                raise RuntimeError(
                    f"Не удалось определить длительность файла {audio_path}: {e}"
                )

    def run(self) -> Dict[str, Dict[str, Any]]:
        audio_paths_with_transcriptions = self._get_audio_files_with_transcriptions()
        results = {}
        self._preload_models()

        for i, whisper_config in enumerate(self.config.whisper_configs):
            base_whisper_config = whisper_config.model_dump(exclude={"config_name"})
            self.transcriber = WhisperXModel(WhisperXConfig(**base_whisper_config))
            config_name = whisper_config.config_name
            logger.info(
                f"Тестирование конфигурации {i + 1}/{len(self.config.whisper_configs)}: {config_name}"
            )

            config_results = {"config": whisper_config.model_dump(), "files": {}}

            config_results["files"] = self._process_files(
                audio_paths_with_transcriptions, whisper_config
            )

            self._save_result({config_name: config_results}, config_name)
            self.transcriber = None
            gc.collect()
            torch.cuda.empty_cache()

            results[config_name] = config_results

        self.analyzer.analyze_results(results)

        return results

    def _get_batch_iter(
        self, data: Dict[Path, str], batch_size: int
    ) -> Iterable[Dict[Path, str]]:
        it = iter(data.items())
        while True:
            batch = dict(list(it)[:batch_size])
            if not batch:
                return
            yield batch

    def _process_files(
        self,
        audio_paths_with_transcriptions: Dict[Path, str],
        whisper_config: BenchmarkWhisperConfig,
    ) -> Dict[str, Any]:
        results = {}
        if self.transcriber is None:
            raise ValueError("Модель не инициализирована")

        if whisper_config.audio_batch_size > 1:
            # --- ПАКЕТНАЯ ОБРАБОТКА ---
            for references_per_file in self._get_batch_iter(
                audio_paths_with_transcriptions, whisper_config.audio_batch_size
            ):
                if not references_per_file:
                    continue

                accumulated_metrics_per_file: Dict[str, List[TranscriptionMetrics]] = (
                    defaultdict(lambda: [])
                )
                last_run_batch_results: Dict[str, TranscriptionResult] | None = None

                self.gpu_monitor.start()
                for _ in range(self.config.repeat_count):
                    last_run_batch_results = self.transcriber.transcribe_batch(
                        list(references_per_file.keys())
                    )
                    for file_path, res in last_run_batch_results.items():
                        accumulated_metrics_per_file[file_path].append(res.metrics)

                gpu_stats_batch = self.gpu_monitor.stop()

                if not last_run_batch_results:
                    raise ValueError("Не удалось получить результаты транскрипции")

                for file_path, reference in references_per_file.items():
                    file_name = file_path.name
                    duration = self._get_audio_duration(file_path)
                    if duration <= 0:
                        raise ValueError(
                            f"Длительность аудиофайла {file_path} не может быть меньше или равна 0"
                        )

                    file_metrics = accumulated_metrics_per_file[file_name]
                    result = self._build_result(
                        reference,
                        last_run_batch_results[file_name].text,
                        duration,
                        file_metrics,
                        gpu_stats_batch,
                    )

                    results[file_name] = result

        else:
            # --- ОБРАБОТКА ПО ОДНОМУ ФАЙЛУ ---
            for file_path, reference in audio_paths_with_transcriptions.items():
                file_name = file_path.name

                accumulated_metrics: List[TranscriptionMetrics] = []
                transcribe_result: TranscriptionResult | None = None
                self.gpu_monitor.start()
                for _ in range(self.config.repeat_count):
                    transcribe_result = self.transcriber.transcribe(file_path)
                    accumulated_metrics.append(transcribe_result.metrics)
                gpu_stats_file = self.gpu_monitor.stop()

                duration = self._get_audio_duration(file_path)
                if duration <= 0:
                    raise ValueError(
                        f"Длительность аудиофайла {file_path} не может быть меньше или равна 0"
                    )

                if not transcribe_result:
                    raise ValueError("Не удалось получить результаты транскрипции")

                result = self._build_result(
                    reference,
                    transcribe_result.text,
                    duration,
                    accumulated_metrics,
                    gpu_stats_file,
                )
                results[file_name] = result
        return results

    def _save_result(self, results: Dict[str, Any], config_name: str) -> None:
        cfg_data = list(results.values())[0]["config"]
        whisper_cfg = cfg_data.get("whisper_config", cfg_data)
        model = whisper_cfg.get("whisper_arch", "unknown_model")
        compute = whisper_cfg.get("compute_type") or "default"
        self.analyzer.save_results(results, model, compute, config_name)

    def _build_result(
        self,
        reference: str,
        hypothesis: str,
        audio_duration: float,
        metrics: List[TranscriptionMetrics],
        gpu_stats: Dict[str, Any],
    ):
        wer = word_error_rate(reference, hypothesis)

        # for each item in metrics, get the value of the key "transcribe_time"
        avg_transcribe = sum(metric.transcribe_time for metric in metrics) / len(
            metrics
        )
        avg_align = sum(metric.align_time for metric in metrics) / len(metrics)
        avg_segmentation = sum(metric.segmentation_time for metric in metrics) / len(
            metrics
        )

        avg_metrics = {
            "transcribe_time": avg_transcribe,
            "align_time": avg_align,
            "segmentation_time": avg_segmentation,
            "total_processing_time": avg_transcribe + avg_align + avg_segmentation,
        }

        for metric_name, metric_value in avg_metrics.copy().items():
            if metric_value > 0:
                speed_metric_name = f"{metric_name.replace('_time', '')}_speed"
                avg_metrics[speed_metric_name] = round(audio_duration / metric_value, 2)

        return {
            "reference": reference,
            "hypothesis": hypothesis,
            "metrics": {
                "duration": audio_duration,
                "wer": wer,
                **avg_metrics,
                **gpu_stats,
            },
        }
