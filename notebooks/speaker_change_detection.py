# %%

import gc
import os
import time
from pathlib import Path

import librosa
import numpy as np
import torch
import whisperx
from pyannote.audio import Inference
from pyannote.audio.utils.signal import Peak
from pyannote.core import Segment, notebook
from rich import print

samples_dir = Path("C:/Users/Миша/Desktop/whisper bench/long+short/samples")
AUDIO_FILE = samples_dir / "денис_даша_рассказ_1.wav"


def get_audio_duration(wav_file):
    return librosa.get_duration(path=wav_file)


def to_scd(probability):
    return np.max(np.abs(np.diff(probability, n=1, axis=1)), axis=2, keepdims=True)


def timed_call(func):
    t0 = time.monotonic()
    result = func()
    t1 = time.monotonic()
    return result, t1 - t0


def get_scd_result(wav_file, inference_model, peak_alpha=0.05):
    scd_prob = inference_model(wav_file)
    scd_prob.labels = ["SPEAKER_CHANGE"]
    peak = Peak(alpha=peak_alpha)
    peak_result = peak(scd_prob)
    return peak_result


def load_audio(file_name):
    global AUDIO_FILE
    AUDIO_FILE = samples_dir / file_name
    duration = get_audio_duration(AUDIO_FILE)
    notebook.crop = Segment(0, duration)


def create_inference_model(model_name, **kwargs):
    return Inference(model_name, **kwargs, pre_aggregation_hook=to_scd)


def get_dialogue(aligned, scd_result):
    dialogue = []
    for segment in scd_result:
        start, end = segment.start, segment.end
        segment_words = []
        for word_data in aligned["word_segments"]:
            word_start = word_data.get("start", 0)
            word_end = word_data.get("end", word_data.get("start", 0) + 0.1)
            if word_start >= start and word_end <= end:
                segment_words.append(word_data.get("word", ""))
        if segment_words:
            dialogue.append("- " + " ".join(segment_words))
    return "\n".join(dialogue)


# %%
# Транскрибация и выравнивание

for file_name in os.listdir(samples_dir)[:10]:
    print("-" * 20)
    print(file_name)
    load_audio(file_name)

    # Загрузка модели WhisperX
    model_whisper = whisperx.load_model(
        "large-v3-turbo", device="cuda", compute_type="int8_float16"
    )
    align_model, align_metadata = whisperx.load_align_model(
        "ru", "cuda", "jonatasgrosman/wav2vec2-xls-r-1b-russian"
    )

    # Транскрипция и выравнивание
    audio_path = str(AUDIO_FILE)
    transcription, transcription_time = timed_call(
        lambda: model_whisper.transcribe(audio_path)
    )
    aligned, aligned_time = timed_call(
        lambda: whisperx.align(
            transcription["segments"],
            model=align_model,
            align_model_metadata=align_metadata,
            device="cuda",
            audio=audio_path,
        )
    )

    print(
        f"{transcription_time=:0.2f}s, speed={get_audio_duration(audio_path) / transcription_time:.2f}x"
    )
    print(
        f"{aligned_time=:0.2f}s, speed={get_audio_duration(audio_path) / aligned_time:.2f}x"
    )

    # warmup
    for _ in range(10):
        segmentation_model = create_inference_model(
            "pyannote/segmentation",
            device=torch.device("cuda"),
            batch_size=32,
            step=0.75,
        )
        scd_result, scd_time = timed_call(
            lambda: get_scd_result(AUDIO_FILE, segmentation_model, peak_alpha=0.2)
        )

    for peak_alpha, batch_size, step in [
        (0.15, 32, 0.75),
        (0.17, 32, 0.75),
        (0.18, 32, 0.75),  # норм
        (0.2, 32, 0.75),  # норм
    ]:
        print(f"{batch_size=}, {step=}, {peak_alpha=}")
        times = []
        segmentation_model = create_inference_model(
            "pyannote/segmentation",
            device=torch.device("cuda"),
            batch_size=batch_size,
            step=step,
        )

        for _ in range(10):
            scd_result, scd_time = timed_call(
                lambda: get_scd_result(
                    AUDIO_FILE, segmentation_model, peak_alpha=peak_alpha
                )
            )
            times.append(scd_time)

        avg_time = sum(times) / len(times)
        # Формирование итогового текста
        final_text, final_text_time = timed_call(
            lambda: get_dialogue(aligned, scd_result)
        )
        print(final_text)

        print(
            f"{avg_time=:0.2f}s, speed={get_audio_duration(audio_path) / avg_time:.2f}x"
        )

        # Очистка памяти
        segmentation_model = None
        torch.cuda.empty_cache()
        gc.collect()


# %%
