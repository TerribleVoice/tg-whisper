# # %%

# import os
# import time
# from pathlib import Path

# import librosa
# import nltk
# import numpy as np
# import whisperx
# from montreal_forced_aligner.alignment import PretrainedAligner
# from pyannote.audio import Inference
# from pyannote.audio.utils.signal import Peak
# from pyannote.core import Segment, notebook
# from rich import print

# from montreal_forced_aligner.alignment.pretrained import PretrainedAligner
# from montreal_forced_aligner.models import AcousticModel
# from montreal_forced_aligner.db import Dictionary
# from montreal_forced_aligner.utils import get_mfa_version

# samples_dir = Path("C:/Users/Миша/Desktop/whisper bench/my_samples/samples")
# AUDIO_FILE = samples_dir / "денис_даша_рассказ_1.wav"


# def get_audio_duration(wav_file):
#     return librosa.get_duration(path=wav_file)


# def to_scd(probability):
#     return np.max(np.abs(np.diff(probability, n=1, axis=1)), axis=2, keepdims=True)


# def timed_call(func):
#     t0 = time.monotonic()
#     result = func()
#     t1 = time.monotonic()
#     return result, t1 - t0


# def get_scd_result(wav_file, inference_model, peak_alpha=0.05):
#     scd_prob = inference_model(wav_file)
#     scd_prob.labels = ["SPEAKER_CHANGE"]
#     peak = Peak(alpha=peak_alpha)
#     peak_result = peak(scd_prob)
#     return peak_result


# def load_audio(file_name):
#     global AUDIO_FILE
#     AUDIO_FILE = samples_dir / file_name
#     duration = get_audio_duration(AUDIO_FILE)
#     notebook.crop = Segment(0, duration)


# def create_inference_model(model_name, **kwargs):
#     return Inference(model_name, **kwargs, pre_aggregation_hook=to_scd)


# def get_dialogue(aligned, scd_result):
#     dialogue = []
#     for segment in scd_result:
#         start, end = segment.start, segment.end
#         segment_words = []
#         for word_data in aligned["word_segments"]:
#             word_start = word_data.get("start", 0)
#             word_end = word_data.get("end", word_data.get("start", 0) + 0.1)
#             if word_start >= start and word_end <= end:
#                 segment_words.append(word_data.get("word", ""))
#         if segment_words:
#             dialogue.append("- " + " ".join(segment_words))
#     return "\n".join(dialogue)


# # %%
# # Транскрибция

# load_audio("денис_даша_рассказ_1.wav")

# # Загрузка модели WhisperX
# model_whisper = whisperx.load_model("large-v3-turbo", device="cuda", compute_type="int8_float16")

# # Транскрипция и выравнивание
# audio_path = str(AUDIO_FILE)
# transcription_segments, transcription_time = timed_call(lambda: model_whisper.transcribe(audio_path))
# transcription_text = str.join(" ", [segment["text"] for segment in transcription_segments["segments"]])


# # %%
# # Выравнивание с помощью MFA

# # Инициализация MFA
# corpus_directory = Path("C:/Users/Миша/Desktop/whisper bench/quick/samples/mfa")

# print(f"Используется версия MFA: {get_mfa_version()}")
# print(f"Директория корпуса: {corpus_directory}")

# model_name = "russian_mfa"
# acoustic_model_path = AcousticModel.get_pretrained_path(model_name)
# # dictionary_path = Dictionary.get_pretrained_path(model_name)


# mfa = PretrainedAligner(
#     corpus_directory=corpus_directory,
#     acoustic_model_path=acoustic_model_path,
#     # dictionary_path=dictionary_path,
# )
# mfa.setup_acoustic_model()
# mfa.setup()

# # Выравнивание с помощью MFAgr
# mfa_alignment, mfa_time = timed_call(lambda: mfa.align())

# print(f"Время выравнивания MFA: {mfa_time:.2f} сек")
# print(f"Результат MFA: {mfa_alignment}")

# # %%
