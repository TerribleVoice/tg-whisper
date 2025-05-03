import re
from typing import Dict
from jiwer import wer, mer, wil, cer

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^\w\s]", "", text)
    return text

class TranscriptionMetrics:
    @staticmethod
    def word_error_rate(reference: str, hypothesis: str, preprocess: bool = True) -> float:
        if preprocess:
            reference = preprocess_text(reference)
            hypothesis = preprocess_text(hypothesis)
        return wer(reference, hypothesis)

    @staticmethod
    def character_error_rate(reference: str, hypothesis: str, preprocess: bool = True) -> float:
        if preprocess:
            reference = preprocess_text(reference)
            hypothesis = preprocess_text(hypothesis)
        return cer(reference, hypothesis) # type: ignore

    @staticmethod
    def match_error_rate(reference: str, hypothesis: str, preprocess: bool = True) -> float:
        if preprocess:
            reference = preprocess_text(reference)
            hypothesis = preprocess_text(hypothesis)
        return mer(reference, hypothesis)

    @staticmethod
    def word_information_lost(reference: str, hypothesis: str, preprocess: bool = True) -> float:
        if preprocess:
            reference = preprocess_text(reference)
            hypothesis = preprocess_text(hypothesis)
        return wil(reference, hypothesis)

    @classmethod
    def calculate_all(cls, reference: str, hypothesis: str, preprocess: bool = True) -> Dict[str, float]:
        if preprocess:
            reference = preprocess_text(reference)
            hypothesis = preprocess_text(hypothesis)
        metrics = {
            "wer": cls.word_error_rate(reference, hypothesis, preprocess=False),
            "cer": cls.character_error_rate(reference, hypothesis, preprocess=False),
            "mer": cls.match_error_rate(reference, hypothesis, preprocess=False),
            "wil": cls.word_information_lost(reference, hypothesis, preprocess=False),
        }
        return metrics
