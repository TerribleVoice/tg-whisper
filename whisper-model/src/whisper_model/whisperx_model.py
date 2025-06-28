import logging
import tempfile
import time
from dataclasses import dataclass
from os import getenv
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import librosa
import numpy as np
import torch
import whisperx
from pyannote.audio import Inference
from pyannote.audio.utils.signal import Peak
from pyannote.core import Segment, SlidingWindowFeature, Timeline
from scipy.io import wavfile as scipy_wavfile
from whisperx.alignment import SingleAlignedSegment, SingleWordSegment

from .config import WhisperXConfig
from .suppress_std import SuppressStd
from .text_formatter import TextFormatter

logger = logging.getLogger("whisper-model")

HF_TOKEN = getenv("HF_TOKEN")


@dataclass
class TranscriptionMetrics:
    transcribe_time: float
    align_time: float
    segmentation_time: float

    def __init__(self, metrics: Dict[str, float]):
        self.transcribe_time = metrics.get("transcribe_time", 0)
        self.align_time = metrics.get("align_time", 0)
        self.segmentation_time = metrics.get("segmentation_time", 0)


@dataclass
class TranscriptionResult:
    text: str
    metrics: TranscriptionMetrics


class WhisperXModel:
    def __init__(self, config: WhisperXConfig):
        self.whisper_config = config.whisper_config
        self.align_config = config.align_config
        self.segmentation_config = config.segmentation_config

        with SuppressStd(logger):
            whisper_model = whisperx.load_model(
                **self.whisper_config.model_dump(exclude={"transcribe_options"})
            )
            logger.info(
                f"WhisperX {self.whisper_config.whisper_arch} loaded with config: {self.whisper_config.model_dump()}"
            )
            align_model, metadata = whisperx.load_align_model(
                **self.align_config.model_dump()
            )
            logger.info(
                f"WhisperX Align model {self.align_config.model_name or 'default'} loaded with config: {self.align_config.model_dump()}"
            )

            segmentation_model = Inference(
                **self.segmentation_config.model_dump(
                    exclude={"device", "peak_config"}
                ),
                device=torch.device(self.segmentation_config.device),
                pre_aggregation_hook=lambda p: np.max(
                    np.abs(np.diff(p, n=1, axis=1)), axis=2, keepdims=True
                ),
            )
            logger.info(
                f"Segmentation model {self.segmentation_config.model} loaded with config: {self.segmentation_config.model_dump()}"
            )

        self.whisper_model = whisper_model
        self.align_model = align_model
        self.align_metadata = metadata
        self.segmentation_model = segmentation_model

    def _measured_call(self, func: Callable):
        start = time.monotonic()
        result = func()
        return result, time.monotonic() - start

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        transcription_options = self.whisper_config.transcribe_options.model_dump()
        metrics: Dict[str, float] = {}

        with SuppressStd(logger):
            audio = whisperx.load_audio(str(audio_path))
            transcribe_result, metrics["transcribe_time"] = self._measured_call(
                lambda: self.whisper_model.transcribe(audio, **transcription_options)
            )

            aligned_result, metrics["align_time"] = self._measured_call(
                lambda: whisperx.align(
                    transcribe_result["segments"],
                    self.align_model,
                    self.align_metadata,
                    audio,
                    self.align_config.device,
                )
            )

            segmentation_result, metrics["segmentation_time"] = self._measured_call(
                lambda: self._perfom_segmentation(audio_path)
            )
            assigned_result = self._assign_words_to_segments(
                aligned_result["word_segments"], segmentation_result
            )
            result_text = TextFormatter.format_segments(assigned_result)

        return TranscriptionResult(
            text=result_text, metrics=TranscriptionMetrics(metrics)
        )

    def _perfom_segmentation(self, audio_path: Path) -> Timeline:
        segmentation_prob = self.segmentation_model(audio_path)
        if not isinstance(segmentation_prob, SlidingWindowFeature):
            raise ValueError(
                f"Segmentation инференс вернул не SlidingWindowFeature, а {type(segmentation_prob)}"
            )

        segmentation_prob.labels = ["SPEAKER_CHANGE"]
        peak = Peak(**self.segmentation_config.peak_config.model_dump())
        segments_timeline = peak(segmentation_prob)

        return segments_timeline

    def _assign_words_to_segments(
        self, word_segments: List[SingleWordSegment], segments_timeline: Timeline
    ) -> List[List[SingleWordSegment]]:
        segments = []
        current_segment = []
        current_segment_index = 0

        for i, word in enumerate(word_segments):
            if word.get("end") is None:
                current_segment.append(word)
                continue

            if (
                word["end"] > segments_timeline[current_segment_index].end
                or i == len(word_segments) - 1
            ):
                segments.append(current_segment)
                current_segment = []
                current_segment_index += 1

            current_segment.append(word)

        return segments

    def transcribe_batch(
        self, audio_paths: List[Path], silence_duration_s: float = 2.0
    ) -> Dict[str, TranscriptionResult]:
        metrics: Dict[str, float] = {}
        concat_audio_path, concat_audio_data, original_files_info = (
            self._create_concat_audio(audio_paths, silence_duration_s)
        )

        try:
            transcription_options = self.whisper_config.transcribe_options.model_dump()

            with SuppressStd(logger):
                concat_transcribe_result, metrics["transcribe_time"] = (
                    self._measured_call(
                        lambda: self.whisper_model.transcribe(
                            concat_audio_data, **transcription_options
                        )
                    )
                )

                concat_aligned_result, metrics["align_time"] = self._measured_call(
                    lambda: whisperx.align(
                        concat_transcribe_result["segments"],
                        self.align_model,
                        self.align_metadata,
                        concat_audio_data,
                        self.align_config.device,
                    )
                )

                concat_segmentation_timeline, metrics["segmentation_time"] = (
                    self._measured_call(
                        lambda: self._perfom_segmentation(Path(concat_audio_path))
                    )
                )

            decomposed_words = self._decompose_words(
                original_files_info,
                concat_aligned_result["segments"],
                silence_duration_s,
            )
            decomposed_segments = self._decompose_segments(
                original_files_info, concat_segmentation_timeline
            )
            processed_files = {
                file_name: {"word_segments": words, "timeline": timeline}
                for (file_name, words), (_, timeline) in zip(
                    decomposed_words.items(),
                    decomposed_segments.items(),
                )
            }

            segments_by_file = {
                file_name: self._assign_words_to_segments(
                    file_info["word_segments"], file_info["timeline"]
                )
                for file_name, file_info in processed_files.items()
            }

            metrics_by_file = self._calculate_metrics_by_file(
                original_files_info, metrics
            )

        finally:
            Path(concat_audio_path).unlink(missing_ok=True)

        return {
            file_name: TranscriptionResult(
                text=TextFormatter.format_segments(segments),
                metrics=metrics_by_file[file_name],
            )
            for file_name, segments in segments_by_file.items()
        }

    def _create_concat_audio(
        self, audio_paths: List[Path], silence_duration_s: float = 1.0
    ) -> Tuple[Path, np.ndarray, List[Dict[str, Any]]]:
        sample_rate = 16000
        silence_array = np.zeros(
            int(silence_duration_s * sample_rate), dtype=np.float32
        )
        original_files_info = []
        all_audio_data = []
        current_time_s = 0.0

        for i, audio_path in enumerate(audio_paths):
            try:
                audio_data = whisperx.load_audio(str(audio_path))
            except Exception as e:
                logger.error(f"Failed to load audio file {audio_path}: {e}")
                raise

            file_duration_s = librosa.get_duration(path=audio_path)
            original_files_info.append(
                {
                    "path": audio_path,
                    "duration_s": file_duration_s,
                    "start_s_in_concat": current_time_s,
                }
            )
            all_audio_data.append(audio_data)
            current_time_s += file_duration_s

            if i < len(audio_paths) - 1:
                all_audio_data.append(silence_array)
                current_time_s += silence_duration_s

        concatenated_audio_data = np.concatenate(all_audio_data)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            concated_audio_path = tmpfile.name

        if concatenated_audio_data.dtype != np.float32:
            concatenated_audio_data = concatenated_audio_data.astype(np.float32)

        scipy_wavfile.write(concated_audio_path, sample_rate, concatenated_audio_data)

        return Path(concated_audio_path), concatenated_audio_data, original_files_info

    def _adjust_time(self, audio_info, abs_start, abs_end):
        file_start_s = audio_info["start_s_in_concat"]

        relative_start = max(0, abs_start - file_start_s)
        relative_end = min(audio_info["duration_s"], abs_end - file_start_s)

        return relative_start, relative_end

    def _decompose_words(
        self,
        file_infos: List[Dict[str, Any]],
        word_segments: List[SingleAlignedSegment],
        silence_duration_s: float,
    ) -> Dict[str, List[SingleWordSegment]]:
        words_by_file: Dict[str, List[SingleWordSegment]] = {
            info["path"].name: [] for info in file_infos
        }
        current_file_index = 0
        for segment in word_segments:
            abs_start = segment.get("start")
            abs_end = segment.get("end")

            while current_file_index < len(file_infos):
                current_file_info = file_infos[current_file_index]
                file_start_s = current_file_info["start_s_in_concat"]
                file_end_s = file_start_s + current_file_info["duration_s"]

                if current_file_index < len(file_infos) - 1:
                    file_end_s += silence_duration_s * 0.5

                if current_file_index > 0:
                    file_start_s -= silence_duration_s * 0.5

                if abs_end >= file_start_s and abs_start <= file_end_s:
                    break

                current_file_index += 1

            current_file_info = file_infos[current_file_index]
            file_name = current_file_info["path"].name
            for word in segment["words"]:
                adjusted_word = word.copy()
                if "start" in word and "end" in word:
                    adj_start, adj_end = self._adjust_time(
                        current_file_info, word["start"], word["end"]
                    )
                    adjusted_word["start"] = adj_start
                    adjusted_word["end"] = adj_end

                words_by_file[file_name].append(adjusted_word)

        return words_by_file

    def _decompose_segments(
        self, file_infos: List[Dict[str, Any]], segments_timeline: Timeline
    ) -> Dict[str, Timeline]:
        segments_by_file: Dict[str, List[Segment]] = {
            info["path"].name: [] for info in file_infos
        }

        for segment_event in segments_timeline.segments_list_:
            abs_start_event, abs_end_event = segment_event.start, segment_event.end

            for file_info in file_infos:
                if abs_start_event >= abs_end_event:  # Сегмент полностью обработан
                    break

                file_name = file_info["path"].name
                file_start_s = file_info["start_s_in_concat"]
                file_end_s = file_start_s + file_info["duration_s"]

                # Вычисляем часть сегмента, которая пересекается с текущим файлом
                overlap_start_abs = max(abs_start_event, file_start_s)
                overlap_end_abs = min(abs_end_event, file_end_s)

                if (
                    overlap_start_abs < overlap_end_abs
                ):  # Если есть реальное пересечение
                    # Корректируем времена относительно текущего файла
                    adj_start, adj_end = self._adjust_time(
                        file_info, overlap_start_abs, overlap_end_abs
                    )

                    if adj_end - adj_start > 0.3:
                        segments_by_file[file_name].append(Segment(adj_start, adj_end))

                # Для следующего файла часть сегмента начинается с конца этого пересечения
                abs_start_event = overlap_end_abs

        file_timelines = {}
        for file_name, segments in segments_by_file.items():
            file_timelines[file_name] = Timeline(segments=segments)

        return file_timelines

    def _calculate_metrics_by_file(
        self,
        original_files_info: List[Dict[str, Any]],
        total_metrics: Dict[str, float],
    ) -> Dict[str, TranscriptionMetrics]:
        metrics_by_file: Dict[str, TranscriptionMetrics] = {}
        total_audio_duration_no_silence = sum(
            info["duration_s"] for info in original_files_info
        )

        for info in original_files_info:
            file_duration_no_silence = info["duration_s"]
            proportion = (
                file_duration_no_silence / total_audio_duration_no_silence
                if total_audio_duration_no_silence > 0
                else 0
            )

            file_specific_metrics = {
                "transcribe_time": total_metrics.get("transcribe_time", 0) * proportion,
                "align_time": total_metrics.get("align_time", 0) * proportion,
                "segmentation_time": total_metrics.get("segmentation_time", 0)
                * proportion,
            }
            metrics_by_file[info["path"].name] = TranscriptionMetrics(
                file_specific_metrics
            )

        return metrics_by_file
